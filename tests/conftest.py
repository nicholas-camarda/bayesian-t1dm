from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest


_test_home = Path(tempfile.mkdtemp(prefix="bayesian_t1dm_pytest_"))
_test_cache = _test_home / "cache"
_test_mpl = _test_home / "matplotlib"
_test_runtime = _test_home / "runtime"
_test_cloud = _test_home / "cloud"
for path in (_test_cache, _test_mpl, _test_runtime, _test_cloud):
    path.mkdir(parents=True, exist_ok=True)

os.environ["HOME"] = str(_test_home)
os.environ["USERPROFILE"] = str(_test_home)
os.environ["XDG_CACHE_HOME"] = str(_test_cache)
os.environ["MPLCONFIGDIR"] = str(_test_mpl)
os.environ["BAYESIAN_T1DM_RUNTIME_ROOT"] = str(_test_runtime)
os.environ["BAYESIAN_T1DM_CLOUD_ROOT"] = str(_test_cloud)
os.environ["PYTENSOR_FLAGS"] = f"compiledir={_test_home / 'pytensor'}"


@pytest.fixture(scope="session")
def tandem_fixture_dir() -> Path:
    return Path(__file__).parent / "fixtures" / "tandem"


@pytest.fixture(scope="session")
def tandem_realistic_fixture_dir() -> Path:
    return Path(__file__).parent / "fixtures" / "tandem_realistic_14d"
