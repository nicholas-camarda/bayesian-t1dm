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


# The two sets of changes the replay-as-predictor test is run on: one color, label, marker and line type in every figure and on the front page.
CHANGE_SET_STYLE = {
    "isolated changes": GroupStyle("Isolated changes", JCO[0], "o", "-"),
    "all changes": GroupStyle("All changes", JCO[3], "s", "--"),
}


@dataclass(frozen=True)
class StatusStyle:
    """A diagnostic status or a verdict as it is drawn: the text, the glyph and the colors. Color is never the only signal.

    ``color`` is the JCO color of the outline; ``ink`` is the darker shade of it used for the text, chosen so the text keeps a contrast of at
    least 4.5 to 1 on the chip's own tinted background (a test checks this). ``outline`` marks the states that must not look like a pass (a
    check that was not run, a result that is not available): a dashed outline and no fill.
    """

    label: str
    glyph: str
    color: str
    outline: bool = False
    ink: str = ""

    @property
    def text(self) -> str:
        return self.ink or self.color


# Darker shades of JCO[0] and JCO[6] for text on a tinted chip.
INK_BLUE = "#00579A"
INK_GOLD = "#6E5A00"

# Diagnostic statuses (the four a workbook may hold): "not assessed" is dashed and hollow so it never looks like "no concern".
STATUS_STYLE = {
    "no concern": StatusStyle("No concern", "+", JCO[0], ink=INK_BLUE),
    "concern": StatusStyle("Concern", "!", JCO[8]),
    "not assessed": StatusStyle("Not assessed", "?", JCO[6], outline=True, ink=INK_GOLD),
    "not applicable": StatusStyle("Not applicable", "-", JCO[7]),
}

# The win-test verdict, whether the record validates the settings effects, and the state of a result that is not there.
VERDICT_STYLE = {
    "win": StatusStyle("Win", "+", JCO[0], ink=INK_BLUE),
    "tie": StatusStyle("Tie", "=", JCO[6], ink=INK_GOLD),
    "loss": StatusStyle("Loss", "-", JCO[8]),
    "missing": StatusStyle("Not available", "?", JCO[7], outline=True),
    "shown": StatusStyle("Shown", "+", JCO[0], ink=INK_BLUE),
    "not shown": StatusStyle("Not shown", "-", JCO[8]),
}

# Outcomes of the outcomes model, named the same everywhere they are listed.
OUTCOME_LABEL = {"tir": "Time in range (70-180 mg/dL)", "below70": "Time below 70 mg/dL", "below54": "Time below 54 mg/dL"}
OUTCOME_SHORT = {"tir": "time in range", "below70": "time below 70", "below54": "time below 54"}


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


# The personal readout's context marks on a day's glucose trace: one color each, the same on every day.
READOUT_STYLE = {
    "glucose": GroupStyle("Glucose", NEUTRAL_LINE, "", "-"),
    "range": GroupStyle("Target range 70-180 mg/dL", "#20854E", "", "-"),
    "sleep": GroupStyle("Sleep", JCO[4], "", "-"),
    "workout": GroupStyle("Workout", JCO[1], "", "-"),
    "carbs": GroupStyle("Carbs entered", JCO[3], "^", ""),
    "site": GroupStyle("Site change", JCO[2], "|", "--"),
    "estimate": GroupStyle("Estimate with 90% interval", JCO[0], "o", "-"),
}

def assert_ascii(*texts: str) -> None:
    """Raise when a figure label holds a character outside ASCII (it may not render in every font)."""
    for text in texts:
        if not str(text).isascii():
            raise ValueError(f"Figure text must be ASCII: {text!r}")

