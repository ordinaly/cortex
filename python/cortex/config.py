from __future__ import annotations
import json
from pathlib import Path


def load_config(path: str | Path) -> str:
    """Return canonical JSON config accepted by the native constructor."""
    data = json.loads(Path(path).read_text())
    return json.dumps(data, sort_keys=True, separators=(",", ":"))
