import pandas as pd
import json

from llmops.llm_router import route_llm
from llmops.evaluation import evaluate_investigation


print("\nAGENTIC AI – NAV BREAK INVESTIGATION PROTOTYPE\n")


# -----------------------------------------
# Load NAV datasets (NAV mechanics)
# -----------------------------------------

positions = pd.read_csv("data/positions.csv")
prices = pd.read_csv("data/prices.csv")
income = pd.read_csv("data/income_accruals.csv")
expenses = pd.read_csv("data/expense_accruals.csv")
distributions = pd.read_csv("data/distributions.csv")


# -----------------------------------------
# Break Detection (Control Agents)
# -----------------------------------------

breaks = []

# Price variance check
for _, row in prices.iterrows():

    change = abs(row["price_current"] - row["price_previous"]) / row["price_previous"]

    if change > 0.10:
        breaks.append(f"Price variance detected for {row['security']}")


# Income accrual validation
for _, row in income.iterrows():

    if row["recorded_income"] > row["expected_income"]:
        breaks.append(f"Income accrual mismatch for {row['security']}")


# Expense accrual validation
for _, row in expenses.iterrows():

    if row["recorded_expense"] > row["expected_expense"]:
        breaks.append(f"Expense accrual variance for {row['expense_type']}")


# Distribution validation
for _, row in distributions.iterrows():

    if row["recorded_distribution"] > row["expected_distribution"]:
        breaks.append(f"Distribution variance detected for {row['fund']}")


# -----------------------------------------
# Retrieve RAG Knowledge
# -----------------------------------------

rules = open("knowledge/nav_validation_rules.txt").read()
playbook = open("knowledge/investigation_playbooks.txt").read()


# -----------------------------------------
# Retrieve Agentic Memory
# -----------------------------------------

with open("memory/nav_break_memory.json") as f:
    memory = json.load(f)


# -----------------------------------------
# Investigation Prompt
# -----------------------------------------

prompt = f"""
You are assisting a fund accounting team investigating NAV breaks.

Detected breaks:
{breaks}

NAV validation rules:
{rules}

Investigation playbook:
{playbook}

Historical break patterns:
{memory}

Provide a short investigation summary explaining:

1. Possible root causes
2. Suggested investigation steps
3. Recommended controls
"""


# -----------------------------------------
# LLMOps Router → Enterprise LLM
# -----------------------------------------

analysis = route_llm(prompt)


# -----------------------------------------
# Reflection Agent
# -----------------------------------------

reflection_prompt = f"""
Review the following NAV break investigation summary.

Ensure the explanation is clear, accurate, and aligned with fund accounting controls.

Investigation summary:
{analysis}
"""

final_analysis = route_llm(reflection_prompt)


# -----------------------------------------
# LLMOps Evaluation
# -----------------------------------------

evaluate_investigation(final_analysis)


# -----------------------------------------
# Final Output
# -----------------------------------------

print("\nNAV BREAK INVESTIGATION SUMMARY\n")

print(final_analysis)
