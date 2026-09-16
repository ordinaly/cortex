from __future__ import annotations
import json
import pytest

try:
    from cortex import Cortex, NATIVE_AVAILABLE, native_version
except Exception:
    NATIVE_AVAILABLE = False

pytestmark = pytest.mark.skipif(not NATIVE_AVAILABLE, reason="native Rust extension not built")


def test_smoke_api():
    m = Cortex(dim=3, budget=4)
    r = m.step([0.0, 0.0, 0.0], 1.0)
    assert r.stored == 1
    assert 0 <= r.prediction <= 1
    s = json.loads(m.snapshot_json())
    assert s["t"] == 1
    assert native_version()
