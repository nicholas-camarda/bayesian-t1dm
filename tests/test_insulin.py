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


def test_insulin_curve_conserves_the_dose():
    for duration, peak in [(300, 75), (360, 75), (240, 55)]:
        curve = insulin_action_curve(1.0, duration_minutes=duration, step_minutes=1, peak_minutes=peak)

        # All insulin is on board at delivery and none is left at the end of action.
        assert curve["iob_units"].iloc[0] == 1.0
        assert abs(curve["iob_units"].iloc[-1]) < 1e-9
        # IOB only ever goes down, and activity integrates to the dose.
        assert (np.diff(curve["iob_units"]) <= 1e-12).all()
        integrate = getattr(np, "trapezoid", None) or np.trapz  # numpy < 2 only has trapz
        assert np.isclose(integrate(curve["ia_units"], curve["minutes"]), 1.0, atol=1e-3)
        assert int(curve.loc[curve["ia_units"].idxmax(), "minutes"]) == peak
