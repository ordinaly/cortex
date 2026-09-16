"""Python research interface for the native Cortex runtime."""
from __future__ import annotations

try:
    from ._cortex_native import Cortex, CortexRead, native_version
    NATIVE_AVAILABLE = True
except ImportError as exc:  # source-tree use before maturin build
    NATIVE_AVAILABLE = False
    _NATIVE_IMPORT_ERROR = exc

    class Cortex:  # pragma: no cover - explanatory failure path
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "The Rust Cortex extension is not built. Run `maturin develop --release` "
                "or install a wheel produced by CI. The frozen Python specification lives "
                "under reference/v0_9_5/."
            ) from _NATIVE_IMPORT_ERROR

    CortexRead = None

    def native_version() -> str:
        return "unavailable"

__all__ = ["Cortex", "CortexRead", "NATIVE_AVAILABLE", "native_version"]
