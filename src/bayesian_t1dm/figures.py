"""One color, label and marker per group, defined once and used by every figure (JCO palette).

A group looks and is named the same in every figure. Marks that are not groups (reference bands, interval bars) use the
neutral colors. Figure text is ASCII so every glyph renders at the exported size.
"""

from __future__ import annotations

from dataclasses import dataclass

from .pump_profiles import UNKNOWN_REASONS

JCO = ("#0073C2", "#EFC000", "#868686", "#CD534C", "#7AA6DC", "#003C67", "#8F7700", "#3B3B3B", "#A73030", "#4A6990")
NEUTRAL_LINE = "#3B3B3B"
NEUTRAL_BAND = "#D9D9D9"
NEUTRAL_BAR = "#BDBDBD"


@dataclass(frozen=True)
class GroupStyle:
    label: str
    color: str
    marker: str
    linestyle: str


GROUP_STYLE = {
    "more insulin": GroupStyle("Change gives more insulin", JCO[0], "o", "-"),
    "less insulin": GroupStyle("Change gives less insulin", JCO[3], "s", "--"),
    "mixed": GroupStyle("Change in both directions", JCO[1], "D", "-."),
}


# The predictors of the forecast harness: one color, label, marker and line type in every figure (the reference baseline is grey).
PREDICTOR_STYLE = {
    "last 28 days": GroupStyle("Last 28 days", JCO[2], "o", "-"),
    "last 14 days": GroupStyle("Last 14 days", JCO[1], "s", "-"),
    "recency-weighted": GroupStyle("Recency-weighted", JCO[4], "^", "-"),
    "ridge": GroupStyle("Ridge", JCO[5], "D", "-"),
    "hour-day model": GroupStyle("Hour-day model", JCO[0], "o", "-"),
    "hour-day model (planned settings)": GroupStyle("Hour-day model, planned settings", JCO[3], "o", "--"),
    "hour-day model (settings pinned)": GroupStyle("Hour-day model, settings pinned", JCO[7], "v", "--"),
    "hour-day model (trailing-mean counts)": GroupStyle("Hour-day model, trailing-mean counts", JCO[6], "P", "--"),
    "last 365 days": GroupStyle("Last 365 days", JCO[8], "X", "-"),
    "all history": GroupStyle("All history", JCO[9], "*", "-"),
}


# The pump behind each stretch of the record, in order: the same label and color in every figure.
PUMP_STYLE = {
    "first pump": GroupStyle("First pump", JCO[0], "o", "-"),
    "second pump": GroupStyle("Second pump", JCO[3], "s", "--"),
}
# Why a setting is unknown (ProfileHistory.unknown_reasons): one color per reason, in the replay's branch order.
# The palette entries that are not a pump's color (JCO[0], JCO[3]) are used for the reasons, in order; hours with unknown settings take one of them.
_NOT_PUMP = [color for index, color in enumerate(JCO) if index not in (0, 3)]
REASON_COLORS = dict(zip(UNKNOWN_REASONS, _NOT_PUMP))
UNKNOWN_SETTINGS_COLOR = _NOT_PUMP[0]


def assert_ascii(*texts: str) -> None:
    """Raise when a figure label holds a character outside ASCII (it may not render in every font)."""
    for text in texts:
        if not str(text).isascii():
            raise ValueError(f"Figure text must be ASCII: {text!r}")

