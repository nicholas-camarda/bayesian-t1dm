from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from bayesian_t1dm.controller import programmed_units_by_slot


def test_programmed_units_split_at_edits_and_mask_unknown_delivery():
    start = pd.Timestamp("2025-03-04 09:05")
    edit = start + pd.Timedelta(minutes=2)
    later = start + pd.Timedelta(minutes=5)
    basal = pd.DataFrame({"pump_serial": ["1", "1"], "start_timestamp": [start, later],
                          "end_timestamp": [later, later + pd.Timedelta(minutes=5)]})
    history = SimpleNamespace(events=pd.DataFrame({"pump_serial": ["1"], "timestamp": [edit], "start_minutes": [0]}),
                              values_at=lambda pumps, times, field: np.select([times < edit, times < later], [1, 2], default=np.nan))
    analysis = SimpleNamespace(merged=SimpleNamespace(tandem_data=SimpleNamespace(basal=basal)), history=history)
    units = programmed_units_by_slot(analysis)
    assert units[start] == pytest.approx((2 + 3 * 2) / 60)
    assert np.isnan(units[later])


def test_proposal_evidence_excludes_snapshot_only_settings():
    from bayesian_t1dm import therapy_research as meals

    settings = SimpleNamespace(icr_for_segment=lambda s: 7, correction_factor_for_segment=lambda s: 60)
    windows = pd.DataFrame({"timestamp": [pd.Timestamp("2025-03-04 08:30")], "segment": ["morning"],
                            "meal_id": ["m1"], "stated_carbs": [30], "premeal_glucose": [110]})
    support = []
    evidence = meals._effective_icr_evidence(windows, {"m1": object()}, settings=settings, support=support)
    assert evidence.loc[evidence.analysis.eq("primary"), "n_meals"].tolist() == [0]
    assert support[0]["historical_basal_supported"]
    assert not support[0]["settings_supported"]
    assert not support[0]["in_estimate"]
