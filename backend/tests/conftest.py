import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ROOT_STR = str(ROOT)
if ROOT_STR not in sys.path:
    sys.path.insert(0, ROOT_STR)


@pytest.fixture(autouse=True)
def _default_access_key_env(monkeypatch):
    monkeypatch.setenv("ACCESS_KEY", "123")
