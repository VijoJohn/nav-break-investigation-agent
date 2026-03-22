import pandas as pd
import json

from llmops.llm_router import route_llm
from llmops.evaluation import evaluate_investigation

from rag.embedding_model import embed_text
from rag.vector_store import store_vector, search_vectors

from orchestration.agent_workflow import investigation_workflow


print("\nAGENTIC AI – NAV BREAK INVESTIGATION PROTOTYPE\n")


# -----------------------------
# Load NAV mechanics datasets
# -----------------------------

positions = pd.read_csv("data/positions.csv")
prices = pd.read_csv("data/prices.csv")
income = pd.read_csv("data/income_accruals.csv")
expenses = pd.read_csv("data/expense_accruals.csv")
distributions = pd.read_csv("data/distributions.csv")


# -----------------------------
# Break Detection (Control Layer)
# -----------------------------

breaks = []

for _, row in prices.iterrows():

    change = abs(row["price_current"] - row["price_previous"]) / row["price_previous"]

    if change > 0.10:
        breaks.append(f"Price variance detected for {row['security']}")


for _, row in income.iterrows():

    if row["recorded_income"] > row["expected_income"]:
        breaks.append(f"Income accrual mismatch for {row['security']}")


for _, row in expenses.iterrows():

    if row["recorded_expense"] > row["expected_expense"]:
        breaks.append(f"Expense accrual variance for {row['expense_type']}")


for _, row in distributions.iterrows():

    if row["recorded_distribution"] > row["expected_distribution"]:
        breaks.append(f"Distribution variance detected for {row['fund']}")


# -----------------------------
# Load Knowledge (RAG Layer)
# -----------------------------

rules = open("knowledge/nav_validation_rules.txt").read()
playbook = open("knowledge/investigation_playbooks.txt").read()


# Convert to embeddings
rules_vector = embed_text(rules)
playbook_vector = embed_text(playbook)


# Store vectors
store_vector("rules", rules_vector)
store_vector("playbook", playbook_vector)


# Retrieve relevant context
retrieved_context = search_vectors(embed_text("NAV break investigation"))


# -----------------------------
# Load Agentic Memory
# -----------------------------

with open("memory/nav_break_memory.json") as f:
    memory = json.load(f)


# -----------------------------
# Investigation Prompt
# -----------------------------

prompt = f"""
You are assisting a fund accounting team investigating NAV breaks.

Detected breaks:
{breaks}

Validation rules:
{retrieved_context}

Historical break patterns:
{memory}

Provide a short investigation summary explaining:

1. Possible root causes
2. Suggested investigation steps
3. Recommended controls
"""


# -----------------------------
# Agent Workflow
# -----------------------------

workflow = investigation_workflow()


# -----------------------------
# LLMOps → Enterprise LLM
# -----------------------------

# -----------------------------
# LLMOps → Enterprise LLM
# -----------------------------

try:
    analysis = route_llm(prompt)

except Exception as e:

    print("Azure OpenAI not configured. Using fallback investigation logic.")

    analysis = f"""
NAV Investigation Summary

Detected breaks:
{breaks}

Suggested investigation steps:

- Verify security price movements
- Review income accrual calculations
- Validate expense accrual entries
- Confirm distribution adjustments

Recommended control:
Implement tolerance checks for pricing and accrual validation.
"""


# -----------------------------
# Reflection Agent
# -----------------------------

reflection_prompt = f"""
Review the NAV break investigation summary and refine the reasoning.

Summary:
{analysis}
"""

try:
    final_analysis = route_llm(reflection_prompt)

except Exception:
    final_analysis = analysis


# -----------------------------
# LLMOps Evaluation
# -----------------------------

evaluate_investigation(final_analysis)


# -----------------------------
# Output
# -----------------------------

print("\nNAV BREAK INVESTIGATION SUMMARY\n")
print(final_analysis)
