from __future__ import annotations

import numpy as np
import pandas as pd

from bayesian_t1dm.insulin import expand_bolus_to_grid, insulin_action_curve


def test_insulin_curve_starts_at_full_dose_and_decays():
    curve = insulin_action_curve(6.0, duration_minutes=300, step_minutes=5)

    assert np.isclose(curve.loc[curve["minutes"] == 0, "iob_units"].iloc[0], 6.0)
    assert curve["ia_units"].ge(0).all()
    assert curve["iob_units"].iloc[-1] < 0.25
    assert curve["ia_units"].idxmax() > 0


def test_expand_bolus_to_grid_accumulates_exposure():
    grid = pd.DataFrame({"timestamp": pd.date_range("2023-06-01 08:00", periods=4, freq="5min")})
    bolus = pd.DataFrame({"timestamp": [pd.Timestamp("2023-06-01 08:00")], "bolus_units": [4.0]})

    out = expand_bolus_to_grid(bolus, grid)

    assert {"bolus_units", "insulin_activity_units", "iob_units"}.issubset(out.columns)
    assert out.loc[0, "iob_units"] > out.loc[1, "iob_units"]
    assert out.loc[0, "insulin_activity_units"] == 0
