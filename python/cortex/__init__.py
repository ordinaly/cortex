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

from .autonomy import ObservationFrame, TrainingReport, train_dataset
from .lpu import LPUResult, LPUStep, LogicalProcessingUnit
from .hypergraph_geometry import (
    BasinAdjacency,
    HypergraphGeometry,
    HypergraphLocalGeometry,
    Hyperpath,
    RelationalDirection,
    RelationalHyperedge,
)
from .topology import (
    CausalCoordinate,
    EntitySupport,
    FeatureCoordinate,
    LocalArticulation,
    RelationCoordinate,
    TopologyView,
)
from .knowledge import (
    KnowledgeFeedReport,
    KnowledgeGraphView,
    KnowledgeLocalArticulation,
    KnowledgeQualifier,
    KnowledgeStatement,
    KnowledgeStatementIncidence,
    KnowledgeSymbolEncoder,
    PropertyValueCoordinate,
    feed_knowledge,
    make_knowledge_runtime,
)

__all__ = [
    "Articulation",
    "ArticulationRead",
    "Cortex",
    "CortexRead",
    "CortexRuntime",
    "CortexRuntimeRead",
    "CortexRuntimeStageTimings",
    "FuzzyMemory",
    "FuzzyMemoryRead",
    "GraphEvidence",
    "GraphEvidenceRead",
    "LPUResult",
    "LPUStep",
    "LogicalProcessingUnit",
    "KnowledgeFeedReport",
    "RelationalHyperedge",
    "RelationalDirection",
    "Hyperpath",
    "HypergraphLocalGeometry",
    "HypergraphGeometry",
    "BasinAdjacency",
    "PropertyValueCoordinate",
    "KnowledgeLocalArticulation",
    "TopologyView",
    "RelationCoordinate",
    "LocalArticulation",
    "FeatureCoordinate",
    "EntitySupport",
    "CausalCoordinate",
    "KnowledgeGraphView",
    "KnowledgeQualifier",
    "KnowledgeStatement",
    "KnowledgeStatementIncidence",
    "KnowledgeSymbolEncoder",
    "NATIVE_AVAILABLE",
    "ObservationFrame",
    "TrainingReport",
    "__spec_version__",
    "__version__",
    "feed_knowledge",
    "make_knowledge_runtime",
    "native_version",
    "train_dataset",
]
