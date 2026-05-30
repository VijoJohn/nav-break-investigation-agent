"""Offline evaluation harness for the NAV break investigation agent.

Runs the agent over a labelled golden set under one or more configs and scores:
  - classification accuracy (agent category == expected category),
  - groundedness (agent cited the rule the taxonomy expects for that category),
  - latency (per-case wall-clock seconds),
  - LLM-as-judge quality (1-5, heuristic fallback offline).

Writes per-config detail CSVs and a combined summary so two configs (e.g. two
Azure deployments, or LLM vs offline fallback) can be compared side by side.

Usage:
    python -m eval.run_eval                      # default configs
    python -m eval.run_eval --golden eval/golden.jsonl
    python -m eval.run_eval --configs fallback gpt-4o-mini gpt-4o

Each config name is interpreted as:
    "fallback"            -> force the offline keyword fallback (use_llm=False)
    "<deployment-name>"   -> route the LLM to that Azure deployment
"""

import argparse
import csv
import json
import os
import time

from domain import expected_rule_for
from llm.azure_openai_client import is_configured
from orchestration.agent_workflow import investigate
from eval.judge import judge

HERE = os.path.dirname(__file__)
DEFAULT_GOLDEN = os.path.join(HERE, "golden.jsonl")
RESULTS_DIR = os.path.join(HERE, "results")


def load_golden(path):
    cases = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def config_to_agent_kwargs(config_name):
    if config_name == "fallback":
        return {"use_llm": False}
    return {"use_llm": True, "deployment": config_name}


def run_config(config_name, cases):
    """Run the agent over all cases for one config; return (rows, summary)."""
    rows = []
    cfg = config_to_agent_kwargs(config_name)

    correct = 0
    grounded = 0
    judge_total = 0
    latencies = []
    judge_mode_seen = set()

    for case in cases:
        start = time.perf_counter()
        result = investigate(case["observation"], config=cfg)
        elapsed = time.perf_counter() - start
        latencies.append(elapsed)

        is_correct = result["category"] == case["expected_category"]
        expected_rule = case.get("expected_rule") or expected_rule_for(case["expected_category"])
        is_grounded = expected_rule in (result.get("cited_rules") or [])
        score, reason, jmode = judge(case, result)
        judge_mode_seen.add(jmode)

        correct += int(is_correct)
        grounded += int(is_grounded)
        judge_total += score

        rows.append(
            {
                "id": case["id"],
                "expected_category": case["expected_category"],
                "agent_category": result["category"],
                "correct": int(is_correct),
                "expected_rule": expected_rule,
                "cited_rules": "|".join(result.get("cited_rules") or []),
                "grounded": int(is_grounded),
                "judge_score": score,
                "judge_reason": reason,
                "latency_s": round(elapsed, 3),
                "agent_backend": result.get("backend"),
                "retrieval_backend": result.get("retrieval_backend"),
            }
        )

    n = len(cases)
    summary = {
        "config": config_name,
        "n": n,
        "accuracy": round(correct / n, 3) if n else 0.0,
        "groundedness": round(grounded / n, 3) if n else 0.0,
        "avg_judge_score": round(judge_total / n, 3) if n else 0.0,
        "avg_latency_s": round(sum(latencies) / n, 3) if n else 0.0,
        "p95_latency_s": round(sorted(latencies)[max(0, int(0.95 * n) - 1)], 3) if n else 0.0,
        "judge_mode": "+".join(sorted(judge_mode_seen)),
    }
    return rows, summary


def write_detail_csv(config_name, rows):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    safe = config_name.replace(":", "_").replace("/", "_")
    path = os.path.join(RESULTS_DIR, f"{safe}.csv")
    fields = list(rows[0].keys()) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_summary_csv(summaries):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "summary.csv")
    fields = ["config", "n", "accuracy", "groundedness", "avg_judge_score",
              "avg_latency_s", "p95_latency_s", "judge_mode"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)
    return path


def print_summary(summaries):
    print("\n=== EVAL SUMMARY ===")
    header = f"{'config':<16}{'n':>4}{'acc':>8}{'ground':>8}{'judge':>8}{'avg_s':>8}{'p95_s':>8}  judge_mode"
    print(header)
    print("-" * len(header))
    for s in summaries:
        print(
            f"{s['config']:<16}{s['n']:>4}{s['accuracy']:>8}{s['groundedness']:>8}"
            f"{s['avg_judge_score']:>8}{s['avg_latency_s']:>8}{s['p95_latency_s']:>8}  {s['judge_mode']}"
        )


def main():
    parser = argparse.ArgumentParser(description="NAV agent evaluation harness")
    parser.add_argument("--golden", default=DEFAULT_GOLDEN, help="path to golden.jsonl")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=None,
        help="configs to run; 'fallback' or an Azure deployment name",
    )
    args = parser.parse_args()

    cases = load_golden(args.golden)
    print(f"Loaded {len(cases)} golden cases from {args.golden}")

    if args.configs:
        configs = args.configs
    elif is_configured():
        # Compare the offline baseline against the configured LLM deployment.
        configs = ["fallback", os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")]
    else:
        configs = ["fallback"]
        print("Azure OpenAI not configured -> running 'fallback' config only.")

    summaries = []
    for config_name in configs:
        print(f"\nRunning config: {config_name}")
        rows, summary = run_config(config_name, cases)
        detail_path = write_detail_csv(config_name, rows)
        print(f"  wrote {detail_path}")
        summaries.append(summary)

    summary_path = write_summary_csv(summaries)
    print(f"\nWrote summary: {summary_path}")
    print_summary(summaries)


if __name__ == "__main__":
    main()
