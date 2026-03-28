from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest


os.environ.setdefault("PYTENSOR_FLAGS", f"compiledir={tempfile.gettempdir()}/bayesian_t1dm_pytensor")


@pytest.fixture(scope="session")
def tandem_fixture_dir() -> Path:
    return Path(__file__).parent / "fixtures" / "tandem"
