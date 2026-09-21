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

from .articulation_controller import (
    ArticulationController,
    ArticulationDecision,
    ArticulationProfile,
    ArticulationState,
    ControlledExecution,
    ControllerPolicy,
    ExecutionBackend,
    RuleAssessment,
    profile_rules,
)
from .controller_calibration import (
    BACKEND_FEATURE_NAMES,
    EXECUTE_FEATURE_NAMES,
    UNRESOLVED_FEATURE_NAMES,
    CalibrationBundle,
    CalibratedArticulationController,
    CalibratedDecision,
    CalibratedExecution,
    CalibratedRuleAssessment,
    LinearProbabilitySurface,
    SurfaceObservation,
    backend_features,
    epistemic_execute_features,
    epistemic_unresolved_features,
    fit_logistic_surface,
    select_precision_threshold,
)
from .autonomy import ObservationFrame, TrainingReport, train_dataset
from .lpu import LPUResult, LPUStep, LogicalProcessingUnit
from .hypergraph_lpu import (
    EnumeratingRuleExecutor,
    ExactRelationWitness,
    LPURuleExecutor,
    RuleExecutionRead,
    raw_lpu_reachability,
)
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
    "raw_lpu_reachability",
    "RuleExecutionRead",
    "LPURuleExecutor",
    "ExactRelationWitness",
    "EnumeratingRuleExecutor",
    "BasinAdjacency",
    "PropertyValueCoordinate",
    "KnowledgeLocalArticulation",
    "TopologyView",
    "RelationCoordinate",
    "LocalArticulation",
    "FeatureCoordinate",
    "EntitySupport",
    "CausalCoordinate",
    "select_precision_threshold",
    "fit_logistic_surface",
    "epistemic_unresolved_features",
    "epistemic_execute_features",
    "backend_features",
    "SurfaceObservation",
    "LinearProbabilitySurface",
    "CalibratedRuleAssessment",
    "CalibratedExecution",
    "CalibratedDecision",
    "CalibratedArticulationController",
    "CalibrationBundle",
    "UNRESOLVED_FEATURE_NAMES",
    "EXECUTE_FEATURE_NAMES",
    "BACKEND_FEATURE_NAMES",
    "profile_rules",
    "RuleAssessment",
    "ExecutionBackend",
    "ControllerPolicy",
    "ControlledExecution",
    "ArticulationState",
    "ArticulationProfile",
    "ArticulationDecision",
    "ArticulationController",
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
