# Agentic AI for NAV Break Investigation

A working prototype showing how an **agentic AI workflow can assist NAV break
investigations in fund accounting operations**. It pairs deterministic fund
accounting control checks with a wired LangGraph agent that classifies each
break, cites the validation rule that governs it, and drafts a root cause,
investigation steps, and a recommended control.

The agent runs end to end **with or without an Azure OpenAI key**: with a key it
uses the LLM in JSON mode; without one it falls back to a transparent keyword
classifier so the pipeline (and the evaluation harness) still run offline.

---

## Business Context

Daily NAV validation is one of the most critical control processes in fund
accounting. Unexpected NAV movements arise from several operational drivers:

- Security price movements (pricing / stale vendor feeds)
- Income accrual differences (coupons, dividends)
- Expense accrual discrepancies (fee rates, NAV base)
- Distribution discrepancies (declared vs available income)
- Position quantity mismatches (custody vs accounting records)

When a NAV break occurs, operations teams must identify the driver before the
final NAV is confirmed. This prototype explores how an agentic workflow can
assist that triage.

---

## Architecture

```
data CSVs ──► break detection ──►  LangGraph agent  ──► structured result
(control layer)                  retrieve → investigate → reflect
```

### 1. Control layer — `controls/break_detection.py`
Deterministic checks over the NAV mechanics datasets (price tolerance, accrual
and distribution variances) emit structured break observations. Each carries
the category the rule implies plus a human-readable description.

### 2. Break taxonomy — `domain.py`
A single source of truth for the five break categories and the validation rule
each maps to (`price_variance → R2`, `income_accrual_mismatch → R3`, …). The
detection layer, the agent, and the eval harness all share this vocabulary.

### 3. Chunked RAG knowledge layer — `rag/`
`knowledge_loader.py` splits the validation rules and investigation playbook
into rule-tagged chunks (`[R1]`..`[R6]`). `vector_store.py` embeds each chunk
with Sentence Transformers (`all-MiniLM-L6-v2`) and indexes one point per chunk
in an in-memory Qdrant collection, so retrieval returns the specific rule and
playbook step relevant to a break — not a whole file as a single vector. If the
embedding model or Qdrant is unavailable, it degrades gracefully to transparent
lexical scoring over the same chunks.

### 4. Agent workflow — `orchestration/agent_workflow.py`
A real LangGraph `StateGraph` with three wired nodes:

- **retrieve** — pull the top-k relevant knowledge chunks for the observation.
- **investigate** — classify the break, cite the governing rule(s), and draft
  root cause / steps / control. Azure OpenAI in JSON mode when configured,
  otherwise the offline keyword fallback.
- **reflect** — validate the category and enforce a groundedness guard: if the
  taxonomy expects a rule for the chosen category and that rule was actually
  retrieved, ensure it is cited.

Public entry point:

```python
from orchestration.agent_workflow import investigate

result = investigate("Price variance for XYZ Equity: 16% move exceeds tolerance.")
# -> {category, cited_rules, root_cause, investigation_steps,
#     recommended_control, summary, backend, retrieval_backend}
```

### 5. Agentic memory — `memory/nav_break_memory.json`
Historical resolved breaks (root cause, resolution, recommended control) passed
to the LLM as pattern-matching context.

### 6. LLMOps layer — `llm/`, `llmops/`
`azure_openai_client.py` wraps Azure OpenAI (supports both `AZURE_OPENAI_API_KEY`
and `AZURE_OPENAI_KEY`, per-call deployment override, JSON mode). `llm_router.py`
is the single routing/logging choke point and lets the eval harness send the
same prompt to different deployments for model comparison.

---

## Evaluation harness — `eval/`

The claims above are backed by a runnable offline evaluation harness rather than
asserted.

- `eval/golden.jsonl` — 25 labelled NAV breaks (5 per category, varied phrasing),
  each annotated with the expected category and expected rule.
- `eval/run_eval.py` — runs the agent over the golden set under one or more
  configs and scores **classification accuracy**, **groundedness** (did the agent
  cite the expected rule?), **latency** (avg / p95), and an **LLM-as-judge**
  quality score (1–5, with a deterministic heuristic fallback offline). Writes a
  per-config detail CSV and a combined `summary.csv`.
- `eval/judge.py` — the LLM-as-judge with heuristic fallback.

```bash
# Offline (no key): scores the keyword-fallback baseline.
python -m eval.run_eval

# With an Azure key: compares the offline baseline against your deployment.
python -m eval.run_eval --configs fallback gpt-4o-mini gpt-4o
```

Example offline run (keyword fallback baseline, no LLM):

```
config             n     acc  ground   judge   avg_s   p95_s  judge_mode
fallback          25    0.84    0.84    4.52   ~0.02*  0.025  heuristic
```

The 0.84 baseline is honest: the naive keyword classifier confuses a few cases
(e.g. "distribution exceeds available *income*" pulls the income keyword). It is
the baseline the LLM config is expected to beat. (*First call includes one-time
embedding-model load; steady-state per-case latency is ~0.02s.)

---

## Repository structure

```
nav-break-investigation-agent/
  app.py                         # demo: detect breaks, investigate each
  domain.py                      # break taxonomy + category→rule mapping
  controls/
    break_detection.py           # deterministic NAV control checks
  data/                          # sample NAV mechanics datasets (CSV)
  knowledge/
    nav_validation_rules.txt     # rule-tagged [R1]..[R6]
    investigation_playbooks.txt  # rule-tagged drivers + steps
  rag/
    knowledge_loader.py          # parse knowledge into rule-tagged chunks
    vector_store.py              # chunked Qdrant index + lexical fallback
    embedding_model.py           # sentence-transformers (lazy-loaded)
  orchestration/
    agent_workflow.py            # LangGraph StateGraph + investigate()
  llm/
    azure_openai_client.py       # Azure OpenAI wrapper
  llmops/
    llm_router.py                # routing + logging
    evaluation.py                # lightweight response sanity checks
  memory/
    nav_break_memory.json        # historical resolved breaks
  eval/
    golden.jsonl                 # labelled evaluation set
    run_eval.py                  # accuracy / groundedness / latency / judge
    judge.py                     # LLM-as-judge + heuristic fallback
  requirements.txt
  .env.example
```

---

## Technology stack

Python · LangGraph · Azure OpenAI · Qdrant (in-memory) · Sentence Transformers
(`all-MiniLM-L6-v2`) · Pandas

---

## Running the prototype

```bash
pip install -r requirements.txt

# Run the investigation demo (works offline; uses Azure OpenAI if configured)
python app.py

# Run the evaluation harness
python -m eval.run_eval
```

### Environment configuration

Copy `.env.example` to `.env` and add your Azure OpenAI credentials to use the
LLM path:

```
AZURE_OPENAI_KEY=...
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-4o
```

Without these, the agent and the eval harness run on the offline fallback.

---

## Scope and honest limitations

- Sample CSVs and a small hand-written knowledge base — illustrative, not a
  fund's real data.
- The offline fallback is a deliberately simple keyword classifier; it exists so
  the pipeline runs without a key and as a baseline for evaluation.
- This is a working prototype, not a deployed system. Containerisation (Docker)
  is a planned next step.

## Possible extensions

- Larger, reviewer-labelled golden set and category-level error analysis.
- Real pricing-vendor and custody-feed integration.
- Confidence thresholds and human-in-the-loop escalation.
- Docker packaging for reproducible runs.
