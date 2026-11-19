"""The diagnostics workbook: the checks behind a result, readable without the code or the patient data.

Sheet order is fixed: ``Specification_Coding`` first, then ``Denominators_Flow``, then the method-specific sheets, then
``Warnings_Limitations`` and ``Artifacts_Provenance`` last. A sheet that has a ``status`` column may hold only the four
statuses below, so a check that was not run can never look as if it passed. A workbook that has a ``Record_Issues`` sheet
holds record identifiers: it stays in the Project Vault and out of Git.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Font
from openpyxl.utils import get_column_letter

STATUSES = ("no concern", "concern", "not assessed", "not applicable")
FIRST_SHEETS = ("Specification_Coding", "Denominators_Flow")
LAST_SHEETS = ("Warnings_Limitations", "Artifacts_Provenance")
MAX_COLUMN_WIDTH = 70


def write_workbook(path: str | Path, sheets: dict[str, pd.DataFrame]) -> Path:
    """Write ``sheets`` in the fixed order: the required first and last sheets, the method sheets between them in the order given."""
    missing = [name for name in (*FIRST_SHEETS, *LAST_SHEETS) if name not in sheets]
    if missing:
        raise ValueError(f"The diagnostics workbook lacks required sheets: {missing}")
    for name, frame in sheets.items():
        if len(name) > 31:
            raise ValueError(f"Sheet name {name!r} is longer than Excel's 31 characters")
        if "status" in frame.columns:
            bad = sorted(set(frame["status"].astype(str)) - set(STATUSES))
            if bad:
                raise ValueError(f"Sheet {name} has statuses outside {STATUSES}: {bad}")
    middle = [name for name in sheets if name not in (*FIRST_SHEETS, *LAST_SHEETS)]
    order = [*FIRST_SHEETS, *middle, *LAST_SHEETS]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name in order:
            sheets[name].to_excel(writer, sheet_name=name, index=False)
        for name in order:
            _format(writer.sheets[name], sheets[name])
    return path


def _format(sheet, frame: pd.DataFrame) -> None:
    sheet.freeze_panes = "A2"
    for cell in sheet[1]:
        cell.font = Font(bold=True)
        cell.alignment = Alignment(wrap_text=True, vertical="top")
    for index, column in enumerate(frame.columns, start=1):
        longest = max([len(str(column))] + [len(str(value)) for value in frame[column].head(200)])
        sheet.column_dimensions[get_column_letter(index)].width = min(MAX_COLUMN_WIDTH, max(10, longest + 2))
        for cell in sheet[get_column_letter(index)][1:]:
            cell.alignment = Alignment(wrap_text=True, vertical="top")


def read_workbook(path: str | Path) -> dict[str, pd.DataFrame]:
    """Read every sheet back, in file order, to reconcile it with the result tables."""
    workbook = load_workbook(path, read_only=True)
    names = list(workbook.sheetnames)
    workbook.close()
    return pd.read_excel(path, sheet_name=names)
