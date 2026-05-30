"""Domain definitions shared across the NAV break investigation agent.

Centralises the NAV break taxonomy and the mapping from each break category
to the validation rule that governs it. Keeping this in one place lets the
detection layer, the agent, and the evaluation harness agree on a single
vocabulary (important for measuring classification accuracy and groundedness).
"""

# Canonical NAV break categories. These are the labels the agent must choose
# from and the labels the golden evaluation set is annotated with.
CATEGORIES = [
    "price_variance",
    "income_accrual_mismatch",
    "expense_accrual_variance",
    "distribution_accrual_discrepancy",
    "position_quantity_mismatch",
]

# Each category is governed by a specific tagged validation rule (see
# knowledge/nav_validation_rules.txt). Groundedness scoring checks that the
# agent cites the rule expected for the category it assigns.
CATEGORY_RULE = {
    "price_variance": "R2",
    "income_accrual_mismatch": "R3",
    "expense_accrual_variance": "R4",
    "distribution_accrual_discrepancy": "R5",
    "position_quantity_mismatch": "R6",
}

# Human-readable labels for reporting.
CATEGORY_LABEL = {
    "price_variance": "Price variance",
    "income_accrual_mismatch": "Income accrual mismatch",
    "expense_accrual_variance": "Expense accrual variance",
    "distribution_accrual_discrepancy": "Distribution accrual discrepancy",
    "position_quantity_mismatch": "Position quantity mismatch",
}

# Lightweight keyword signals used by the offline fallback classifier when no
# Azure OpenAI deployment is configured. Deliberately simple and transparent.
CATEGORY_KEYWORDS = {
    "price_variance": ["price", "pricing", "vendor", "stale", "market", "valuation"],
    "income_accrual_mismatch": ["income", "coupon", "dividend", "interest"],
    "expense_accrual_variance": ["expense", "fee", "management fee", "custody", "fee rate"],
    "distribution_accrual_discrepancy": ["distribution", "declaration", "payout"],
    "position_quantity_mismatch": ["position", "quantity", "custody", "shares", "reconcile"],
}


def is_valid_category(category: str) -> bool:
    return category in CATEGORIES


def expected_rule_for(category: str) -> str:
    """Return the validation rule id expected for a given category."""
    return CATEGORY_RULE.get(category, "")
