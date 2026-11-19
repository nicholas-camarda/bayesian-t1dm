from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from bayesian_t1dm.icr_comparison import programmed_units_by_slot


def test_programmed_units_split_at_edits_and_mask_unknown_delivery():
    start = pd.Timestamp("2025-03-04 09:05")
    edit = start + pd.Timedelta(minutes=2)
    later = start + pd.Timedelta(minutes=5)
    basal = pd.DataFrame({"pump_serial": ["1", "1"], "start_timestamp": [start, later],
                          "end_timestamp": [later, later + pd.Timedelta(minutes=5)]})
    history = SimpleNamespace(events=pd.DataFrame({"pump_serial": ["1"], "timestamp": [edit], "start_minutes": [0]}),
                              basal_rates=lambda pumps, times: np.select([times < edit, times < later], [1, 2], default=np.nan))
    analysis = SimpleNamespace(merged=SimpleNamespace(tandem_data=SimpleNamespace(basal=basal)), history=history)
    units = programmed_units_by_slot(analysis)
    assert units[start] == pytest.approx((2 + 3 * 2) / 60)
    assert np.isnan(units[later])
