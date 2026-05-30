"""NAV break investigation agent — a real, wired LangGraph workflow.

Pipeline (StateGraph over a dict state):

    retrieve  ->  investigate  ->  reflect  ->  END

  retrieve   : pull rule-tagged knowledge chunks relevant to the observation
               (chunked RAG; vector backend with lexical fallback).
  investigate: classify the break into a category, cite the governing rule(s),
               and draft root cause / steps / control. Uses Azure OpenAI in
               JSON mode when configured, otherwise a transparent keyword
               fallback so the pipeline runs fully offline.
  reflect    : self-check — validate the category, and ensure the rule the
               taxonomy expects for that category is cited when it was actually
               retrieved (groundedness guard).

Public entry point: investigate(observation, config=None) -> dict
Output schema:
    {
        "category": str,                 # one of domain.CATEGORIES or "unknown"
        "cited_rules": [str, ...],       # e.g. ["R2"]
        "root_cause": str,
        "investigation_steps": [str, ...],
        "recommended_control": str,
        "summary": str,
        "backend": str,                  # "llm:<deployment>" or "fallback"
        "retrieval_backend": str,        # "vector" or "lexical"
    }
"""

import json
import os
from typing import List, TypedDict

from langgraph.graph import StateGraph, END

from domain import (
    CATEGORIES,
    CATEGORY_KEYWORDS,
    CATEGORY_LABEL,
    expected_rule_for,
    is_valid_category,
)
from rag.vector_store import get_index
from llm.azure_openai_client import LLMUnavailable
from llmops.llm_router import route_llm

MEMORY_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "memory", "nav_break_memory.json")


class InvestigationState(TypedDict, total=False):
    """Mutable state threaded through the LangGraph workflow.

    total=False so each node can return only the keys it updates; LangGraph
    merges them into the running state (last-value-wins per key).
    """

    observation: str
    deployment: str
    use_llm: bool
    k: int
    retrieved: List[dict]
    retrieval_backend: str
    category: str
    cited_rules: List[str]
    root_cause: str
    investigation_steps: List[str]
    recommended_control: str
    summary: str
    backend: str


def _load_memory():
    try:
        with open(MEMORY_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _format_context(chunks):
    return "\n".join(f"- {c['rule_id']}: {c['text']}" for c in chunks)


# ----------------------------------------------------------------------
# Offline fallback classifier
# ----------------------------------------------------------------------
def fallback_investigate(observation, retrieved):
    """Keyword-based classification used when no LLM deployment is configured."""
    text = observation.lower()
    best_cat, best_score = "unknown", 0
    for cat, keywords in CATEGORY_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text)
        if score > best_score:
            best_cat, best_score = cat, score

    rule = expected_rule_for(best_cat) if best_cat != "unknown" else ""
    cited = [rule] if rule else []

    # Pull the playbook driver text for the chosen rule, if retrieved.
    driver = next(
        (c["text"] for c in retrieved if c.get("rule_id") == rule and c.get("source") == "investigation_playbook"),
        "",
    )
    steps = [
        "Confirm the observation against source data.",
        "Recalculate the expected value from contractual terms.",
        "Compare recorded vs expected and document the variance.",
    ]
    label = CATEGORY_LABEL.get(best_cat, "Unclassified break")
    return {
        "category": best_cat,
        "cited_rules": cited,
        "root_cause": f"Likely {label.lower()} based on the reported variance.",
        "investigation_steps": steps,
        "recommended_control": "Add an automated tolerance / reconciliation check for this break type.",
        "summary": f"{label}. {driver[:200]}".strip(),
        "backend": "fallback",
    }


# ----------------------------------------------------------------------
# LLM investigation
# ----------------------------------------------------------------------
_PROMPT_TEMPLATE = """You are assisting a fund accounting team investigating a NAV break.

Detected break observation:
{observation}

Relevant validation rules and playbook (cite by rule id, e.g. R2):
{context}

Historical resolved breaks (for pattern matching):
{memory}

Classify the break into exactly one category from this list:
{categories}

Respond ONLY with a JSON object using these keys:
  "category": one of the categories above,
  "cited_rules": list of rule ids you relied on (e.g. ["R2"]),
  "root_cause": one sentence,
  "investigation_steps": list of 2-4 short steps,
  "recommended_control": one sentence,
  "summary": 1-2 sentence plain-language summary.
"""


def llm_investigate(observation, retrieved, deployment=None):
    prompt = _PROMPT_TEMPLATE.format(
        observation=observation,
        context=_format_context(retrieved),
        memory=json.dumps(_load_memory()),
        categories=", ".join(CATEGORIES),
    )
    raw = route_llm(prompt, deployment=deployment, temperature=0.1, json_mode=True)
    data = json.loads(raw)
    # Normalise shape.
    result = {
        "category": data.get("category", "unknown"),
        "cited_rules": data.get("cited_rules", []) or [],
        "root_cause": data.get("root_cause", ""),
        "investigation_steps": data.get("investigation_steps", []) or [],
        "recommended_control": data.get("recommended_control", ""),
        "summary": data.get("summary", ""),
        "backend": f"llm:{deployment or os.getenv('AZURE_OPENAI_DEPLOYMENT', 'gpt-4o')}",
    }
    if isinstance(result["cited_rules"], str):
        result["cited_rules"] = [result["cited_rules"]]
    return result


# ----------------------------------------------------------------------
# Graph nodes
# ----------------------------------------------------------------------
def retrieve_node(state):
    index = get_index()
    chunks = index.retrieve(state["observation"], k=state.get("k", 3))
    return {"retrieved": chunks, "retrieval_backend": index.backend}


def investigation_node(state):
    observation = state["observation"]
    retrieved = state.get("retrieved", [])
    deployment = state.get("deployment")
    use_llm = state.get("use_llm", True)

    if use_llm:
        try:
            return llm_investigate(observation, retrieved, deployment=deployment)
        except (LLMUnavailable, json.JSONDecodeError, KeyError):
            pass  # fall through to offline path
    return fallback_investigate(observation, retrieved)


def reflection_node(state):
    """Validate the result and enforce a groundedness guard."""
    category = state.get("category", "unknown")
    cited = list(state.get("cited_rules", []))
    retrieved_rule_ids = {c.get("rule_id") for c in state.get("retrieved", [])}

    if not is_valid_category(category):
        category = "unknown"

    # Groundedness guard: if the taxonomy expects a rule for this category and
    # that rule was actually retrieved, make sure it is cited.
    expected = expected_rule_for(category)
    if expected and expected in retrieved_rule_ids and expected not in cited:
        cited.append(expected)

    return {"category": category, "cited_rules": cited}


# ----------------------------------------------------------------------
# Graph construction
# ----------------------------------------------------------------------
def build_workflow():
    graph = StateGraph(InvestigationState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("investigate", investigation_node)
    graph.add_node("reflect", reflection_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "investigate")
    graph.add_edge("investigate", "reflect")
    graph.add_edge("reflect", END)

    return graph.compile()


# Backwards-compatible alias for older imports.
def investigation_workflow():
    return build_workflow()


_WORKFLOW = None


def _get_workflow():
    global _WORKFLOW
    if _WORKFLOW is None:
        _WORKFLOW = build_workflow()
    return _WORKFLOW


def investigate(observation, config=None):
    """Run the full investigation pipeline for a single break observation.

    Args:
        observation: free-text description of the detected break.
        config: optional dict; recognised keys:
            "deployment" -> Azure deployment name to route to,
            "use_llm"    -> bool, force the offline fallback when False,
            "k"          -> number of knowledge chunks to retrieve.

    Returns the structured result dict (see module docstring).
    """
    config = config or {}
    initial = {
        "observation": observation,
        "deployment": config.get("deployment"),
        "use_llm": config.get("use_llm", True),
        "k": config.get("k", 3),
    }
    final = _get_workflow().invoke(initial)

    return {
        "category": final.get("category", "unknown"),
        "cited_rules": final.get("cited_rules", []),
        "root_cause": final.get("root_cause", ""),
        "investigation_steps": final.get("investigation_steps", []),
        "recommended_control": final.get("recommended_control", ""),
        "summary": final.get("summary", ""),
        "backend": final.get("backend", "fallback"),
        "retrieval_backend": final.get("retrieval_backend", "lexical"),
    }
