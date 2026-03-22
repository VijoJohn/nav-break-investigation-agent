import pandas as pd
import json


print("\nAGENTIC AI – NAV BREAK INVESTIGATION PROTOTYPE\n")


# -----------------------------
# Load NAV datasets
# -----------------------------

positions = pd.read_csv("data/positions.csv")
prices = pd.read_csv("data/prices.csv")
income = pd.read_csv("data/income_accruals.csv")
expenses = pd.read_csv("data/expense_accruals.csv")
distributions = pd.read_csv("data/distributions.csv")


# -----------------------------
# Break Detection Logic
# -----------------------------

breaks = []

for _, row in prices.iterrows():
    change = abs(row["price_current"] - row["price_previous"]) / row["price_previous"]

    if change > 0.1:
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
# Retrieve RAG Knowledge
# -----------------------------

rules = open("knowledge/nav_validation_rules.txt").read()
playbook = open("knowledge/investigation_playbooks.txt").read()


# -----------------------------
# Retrieve Agentic Memory
# -----------------------------

with open("memory/nav_break_memory.json") as f:
    memory = json.load(f)


# -----------------------------
# Investigation Output
# -----------------------------

print("NAV BREAK INVESTIGATION SUMMARY\n")

if not breaks:
    print("No NAV breaks detected.")
else:
    for b in breaks:
        print("-", b)

print("\nRelevant Validation Rules:\n")
print(rules[:300], "...")

print("\nInvestigation Guidance:\n")
print(playbook[:300], "...")

print("\nHistorical Investigation Memory:\n")

for item in memory:
    print(
        f"- Previous break: {item['break_type']} | Root cause: {item['root_cause']}"
    )


# -----------------------------
# Reflection Agent
# -----------------------------

print("\nReflection Agent Review\n")

print(
    "The investigation suggests reviewing pricing feeds, accrual calculations, and distribution entries to confirm the root cause of the detected NAV discrepancies."
)
