"""Python research interface for the native Cortex runtime."""
from __future__ import annotations

try:
    from ._cortex_native import (
        Articulation,
        ArticulationRead,
        Cortex,
        CortexRead,
        FuzzyMemory,
        FuzzyMemoryRead,
        native_version,
    )
    NATIVE_AVAILABLE = True
except ImportError as exc:  # source-tree use before maturin build
    NATIVE_AVAILABLE = False
    _NATIVE_IMPORT_ERROR = exc

    class _MissingNative:
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "The Rust Cortex extension is not built. Run `maturin develop --release` "
                "or install a wheel produced by CI. The frozen Python specification lives "
                "under reference/v0_9_5/."
            ) from _NATIVE_IMPORT_ERROR

    Cortex = _MissingNative
    Articulation = _MissingNative
    FuzzyMemory = _MissingNative
    CortexRead = None
    ArticulationRead = None
    FuzzyMemoryRead = None

    def native_version() -> str:
        return "unavailable"

__all__ = [
    "Articulation",
    "ArticulationRead",
    "Cortex",
    "CortexRead",
    "FuzzyMemory",
    "FuzzyMemoryRead",
    "NATIVE_AVAILABLE",
    "native_version",
]
