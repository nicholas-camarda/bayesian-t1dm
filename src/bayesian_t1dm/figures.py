"""One color, label and marker per group, defined once and used by every figure (JCO palette).

A group looks and is named the same in every figure. Marks that are not groups (reference bands, interval bars) use the
neutral colors. Figure text is ASCII so every glyph renders at the exported size.
"""

from __future__ import annotations

from dataclasses import dataclass

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
    "placebo": GroupStyle("Placebo dates", JCO[2], "x", ":"),
}


def assert_ascii(*texts: str) -> None:
    """Raise when a figure label holds a character outside ASCII (it may not render in every font)."""
    for text in texts:
        if not str(text).isascii():
            raise ValueError(f"Figure text must be ASCII: {text!r}")


def format_p(value: float) -> str:
    """A p-value as the manuscripts write it: 'p < 0.001' below 0.001, otherwise 'p = 0.012'."""
    return "p < 0.001" if value < 0.001 else f"p = {value:.3f}"
