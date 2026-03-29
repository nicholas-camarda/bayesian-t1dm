from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest


_test_home = Path(tempfile.gettempdir()) / "bayesian_t1dm_test_home"
_test_cache = _test_home / "Library" / "Caches"
_test_mpl = _test_home / "matplotlib"
_test_home.mkdir(parents=True, exist_ok=True)
_test_cache.mkdir(parents=True, exist_ok=True)
_test_mpl.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("HOME", str(_test_home))
os.environ.setdefault("USERPROFILE", str(_test_home))
os.environ.setdefault("XDG_CACHE_HOME", str(_test_cache))
os.environ.setdefault("MPLCONFIGDIR", str(_test_mpl))
os.environ.setdefault("PYTENSOR_FLAGS", f"compiledir={tempfile.gettempdir()}/bayesian_t1dm_pytensor")


@pytest.fixture(scope="session")
def tandem_fixture_dir() -> Path:
    return Path(__file__).parent / "fixtures" / "tandem"
