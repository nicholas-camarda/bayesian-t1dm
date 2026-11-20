"""Number formatting shared by every report: one rule for how a p-value is written."""

from __future__ import annotations


def format_p(p: float) -> str:
    """A p-value as it is written everywhere: ``p < 0.001`` below a thousandth, otherwise ``p = 0.012`` to three decimals."""
    if p < 0.001:
        return "p < 0.001"
    return f"p = {p:.3f}"
