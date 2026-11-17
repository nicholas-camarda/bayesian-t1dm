from __future__ import annotations

import os
import re
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
# Keep any backend flags from the environment (for example mode=NUMBA) and only
# redirect the compile cache into the test home.
os.environ["PYTENSOR_FLAGS"] = ",".join(
    flag
    for flag in (os.environ.get("PYTENSOR_FLAGS", ""), f"compiledir={_test_home / 'pytensor'}")
    if flag
)


@pytest.fixture(scope="session")
def tandem_fixture_dir() -> Path:
    return Path(__file__).parent / "fixtures" / "tandem"


@pytest.fixture(scope="session")
def tandem_realistic_fixture_dir() -> Path:
    return Path(__file__).parent / "fixtures" / "tandem_realistic_14d"


_INTERNAL_CODE = re.compile(r"\b[a-z0-9]+(?:_[a-z0-9]+)+\b")


@pytest.fixture
def assert_answer_first_page():
    """Review pages answer the question first and never show raw internal codes."""

    def check(text: str) -> None:
        assert "<script" not in text, "pages must be self-contained without scripts"
        assert "plotly" not in text.lower()
        body = re.sub(r"<style>.*?</style>", "", text, flags=re.S)
        assert body.index("<h1>") < body.index("class='card verdict'") < body.index("<details")
        body = re.sub(r"<code>.*?</code>", " ", body, flags=re.S)  # formats the reader is asked to type
        visible = re.sub(r"<[^>]+>", " ", body)
        visible = re.sub(r"\b[\w.-]+\.(?:md|csv|html)\b", " ", visible)  # file names are allowed
        leaked = sorted(set(_INTERNAL_CODE.findall(visible)))
        assert not leaked, f"internal codes shown to the reader: {leaked}"

    return check
