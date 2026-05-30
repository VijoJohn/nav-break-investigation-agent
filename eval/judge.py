"""LLM-as-judge scoring for investigation summaries, with a heuristic fallback.

When an Azure OpenAI deployment is configured, the judge scores each
investigation summary 1-5 for usefulness and faithfulness to the cited rule.
When no deployment is available, a transparent heuristic produces a comparable
1-5 score so the harness still yields a quality signal offline.
"""

import json

from llm.azure_openai_client import is_configured, LLMUnavailable
from llmops.llm_router import route_llm

JUDGE_SYSTEM = (
    "You are a meticulous fund accounting reviewer scoring NAV break "
    "investigation summaries. Be strict and concise."
)

JUDGE_TEMPLATE = """Score the following NAV break investigation on a 1-5 scale.

Original observation:
{observation}

Expected break category: {expected_category}

Agent output:
  category: {category}
  cited_rules: {cited_rules}
  root_cause: {root_cause}
  recommended_control: {recommended_control}
  summary: {summary}

Scoring guide:
  5 = correct category, cites the right rule, root cause and control are sound and specific.
  3 = mostly right but vague or missing the rule citation.
  1 = wrong category or ungrounded / generic.

Respond ONLY with JSON: {{"score": <1-5 integer>, "reason": "<one sentence>"}}.
"""


def _heuristic_score(case, result):
    """Deterministic offline quality proxy in the same 1-5 range."""
    score = 1
    reason_parts = []
    if result.get("category") == case["expected_category"]:
        score += 2
        reason_parts.append("category correct")
    else:
        reason_parts.append("category wrong")
    if case["expected_rule"] in (result.get("cited_rules") or []):
        score += 1
        reason_parts.append("rule cited")
    else:
        reason_parts.append("rule not cited")
    if (result.get("recommended_control") or "").strip():
        score += 1
        reason_parts.append("control present")
    return min(score, 5), "; ".join(reason_parts) + " (heuristic)"


def judge(case, result):
    """Return (score:int 1-5, reason:str, mode:str)."""
    if is_configured():
        prompt = JUDGE_TEMPLATE.format(
            observation=case["observation"],
            expected_category=case["expected_category"],
            category=result.get("category"),
            cited_rules=result.get("cited_rules"),
            root_cause=result.get("root_cause"),
            recommended_control=result.get("recommended_control"),
            summary=result.get("summary"),
        )
        try:
            raw = route_llm(
                prompt, temperature=0.0, json_mode=True, system=JUDGE_SYSTEM
            )
            data = json.loads(raw)
            score = int(data.get("score", 0))
            return max(1, min(score, 5)), data.get("reason", ""), "llm"
        except (LLMUnavailable, json.JSONDecodeError, ValueError, TypeError):
            pass
    score, reason = _heuristic_score(case, result)
    return score, reason, "heuristic"
