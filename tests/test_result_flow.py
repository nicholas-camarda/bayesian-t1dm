import pandas as pd
import pytest

from bayesian_t1dm.result_flow import sleep_flow


def test_known_sleep_begins_at_wake_and_expires_after_36_hours():
    end = pd.Timestamp("2025-05-01 08:00")
    sleep = pd.DataFrame({"sleep_end": [end], "total_sleep_hours": [7], "in_bed_hours": [0]})
    grid = pd.DataFrame({"timestamp": [end - pd.Timedelta(seconds=1), end, end + pd.Timedelta(hours=37)],
                         "prior_night_sleep_known": [0, 1, 0], "prior_night_total_sleep_hours": [0, 7, 0]})
    counts = sleep_flow(sleep, grid)
    assert counts["both flags"] == counts["known sleep grid rows"] == 1
    grid.loc[0, "prior_night_sleep_known"] = 1
    with pytest.raises(ValueError, match="flag disagree"):
        sleep_flow(sleep, grid)
