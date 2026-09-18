"""Python research interface for the native Cortex runtime."""
from __future__ import annotations

__version__ = "1.0.0rc2"
__spec_version__ = "spec-v0.9.5"

try:
    from ._cortex_native import (
        Articulation,
        ArticulationRead,
        Cortex,
        CortexRead,
        CortexRuntime,
        CortexRuntimeRead,
        CortexRuntimeStageTimings,
        FuzzyMemory,
        FuzzyMemoryRead,
        GraphEvidence,
        GraphEvidenceRead,
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
    CortexRuntime = _MissingNative
    Articulation = _MissingNative
    FuzzyMemory = _MissingNative
    GraphEvidence = _MissingNative
    CortexRead = None
    CortexRuntimeRead = None
    CortexRuntimeStageTimings = None
    ArticulationRead = None
    FuzzyMemoryRead = None
    GraphEvidenceRead = None

    def native_version() -> str:
        return "unavailable"

from .hybrid import FrozenEncoder, HybridCortexRuntime, NeuralObservation, cosine_mse_embedding

__all__ = [
    "Articulation",
    "ArticulationRead",
    "Cortex",
    "CortexRead",
    "CortexRuntime",
    "CortexRuntimeRead",
    "CortexRuntimeStageTimings",
    "FrozenEncoder",
    "FuzzyMemory",
    "FuzzyMemoryRead",
    "GraphEvidence",
    "GraphEvidenceRead",
    "HybridCortexRuntime",
    "NATIVE_AVAILABLE",
    "NeuralObservation",
    "__spec_version__",
    "__version__",
    "cosine_mse_embedding",
    "native_version",
]
