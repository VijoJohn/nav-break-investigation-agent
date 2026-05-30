"""NAV break detection (control layer).

Applies deterministic validation checks to the NAV mechanics datasets and emits
structured break observations. Each observation carries the category the
detection rule implies plus a human-readable description that the agent then
investigates. This is the layer the original app.py short-circuited with an
`if not breaks: exit()` guard placed *before* the checks ran.
"""

import os

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

PRICE_TOLERANCE = 0.10  # 10% day-over-day move triggers a price variance break


def _read(data_dir, name):
    return pd.read_csv(os.path.join(data_dir, name))


def detect_breaks(data_dir=DATA_DIR):
    """Return a list of break observations.

    Each item: {"category": str, "subject": str, "observation": str}.
    """
    breaks = []

    prices = _read(data_dir, "prices.csv")
    for _, row in prices.iterrows():
        prev = row["price_previous"]
        if prev == 0:
            continue
        change = abs(row["price_current"] - prev) / prev
        if change > PRICE_TOLERANCE:
            breaks.append(
                {
                    "category": "price_variance",
                    "subject": row["security"],
                    "observation": (
                        f"Price variance for {row['security']}: previous {prev}, "
                        f"current {row['price_current']} ({change:.0%} move) exceeds "
                        f"the {PRICE_TOLERANCE:.0%} tolerance."
                    ),
                }
            )

    income = _read(data_dir, "income_accruals.csv")
    for _, row in income.iterrows():
        if row["recorded_income"] > row["expected_income"]:
            breaks.append(
                {
                    "category": "income_accrual_mismatch",
                    "subject": row["security"],
                    "observation": (
                        f"Income accrual mismatch for {row['security']}: recorded income "
                        f"{row['recorded_income']} exceeds expected {row['expected_income']}."
                    ),
                }
            )

    expenses = _read(data_dir, "expense_accruals.csv")
    for _, row in expenses.iterrows():
        if row["recorded_expense"] > row["expected_expense"]:
            breaks.append(
                {
                    "category": "expense_accrual_variance",
                    "subject": row["expense_type"],
                    "observation": (
                        f"Expense accrual variance for {row['expense_type']}: recorded "
                        f"expense {row['recorded_expense']} exceeds expected {row['expected_expense']}."
                    ),
                }
            )

    distributions = _read(data_dir, "distributions.csv")
    for _, row in distributions.iterrows():
        if row["recorded_distribution"] > row["expected_distribution"]:
            breaks.append(
                {
                    "category": "distribution_accrual_discrepancy",
                    "subject": row["fund"],
                    "observation": (
                        f"Distribution accrual discrepancy for {row['fund']}: recorded "
                        f"distribution {row['recorded_distribution']} exceeds expected "
                        f"{row['expected_distribution']}."
                    ),
                }
            )

    return breaks


if __name__ == "__main__":
    for b in detect_breaks():
        print(f"[{b['category']}] {b['observation']}")
