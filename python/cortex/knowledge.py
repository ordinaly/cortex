"""Neutral structured-knowledge ingestion for autonomous Cortex.

World semantics are data. The only fixed structure here is a tiny statement
grammar that preserves subject/property/object roles over Cortex's generic,
undirected relation evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Iterable, Mapping

from .autonomy import ObservationFrame
from .topology import LocalArticulation, TopologyView


ROLE_SUBJECT = "@syntax:subject"
ROLE_PROPERTY = "@syntax:property"
ROLE_OBJECT = "@syntax:object"
ROLE_QUALIFIER = "@syntax:qualifier"


def _validate_world_symbol(name: str, value: str) -> None:
    if not value:
        raise ValueError(f"{name} must be non-empty")
    if value.startswith("@"):
        raise ValueError(
            f"{name} cannot use reserved Cortex knowledge syntax namespace '@'"
        )


@dataclass(frozen=True, slots=True)
class KnowledgeQualifier:
    property: str
    value: str

    def __post_init__(self) -> None:
        _validate_world_symbol("qualifier property", self.property)
        _validate_world_symbol("qualifier value", self.value)


@dataclass(frozen=True, slots=True)
class KnowledgeStatement:
    statement_id: str
    subject: str
    property: str
    value: str
    qualifiers: tuple[KnowledgeQualifier, ...] = ()
    rank: str = "normal"
    references: tuple[str, ...] = ()
    provenance: str | None = None

    def __post_init__(self) -> None:
        if not self.statement_id:
            raise ValueError("statement_id must be non-empty")
        for name, value in (
            ("subject", self.subject),
            ("property", self.property),
            ("value", self.value),
        ):
            _validate_world_symbol(name, value)


@dataclass(frozen=True, slots=True)
class EncodedKnowledgeFrame:
    frame: ObservationFrame
    symbols: tuple[str, ...]
    positive_pairs: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class PropertyValueCoordinate:
    property: str
    value: str
    structural_support: float


@dataclass(frozen=True, slots=True)
class KnowledgeLocalArticulation:
    topology: LocalArticulation
    properties: tuple[PropertyValueCoordinate, ...]


@dataclass(frozen=True, slots=True)
class KnowledgeFeedReport:
    statements: int
    frames: int
    symbols: int
    cortex_entities: int
    binding_conflicts: int
    symbol_bindings: Mapping[str, int]
    snapshot_json: str


class KnowledgeSymbolEncoder:
    """Map opaque symbols to deterministic, well-separated observation vectors.

    The vectors carry identity only. They intentionally do not encode lexical
    or ontological similarity.
    """

    def __init__(self, feature_dim: int = 64, amplitude: float = 1.5):
        if feature_dim < 16:
            raise ValueError("feature_dim must be >= 16 for stable hashed identity")
        if amplitude <= 0.0:
            raise ValueError("amplitude must be positive")
        self.feature_dim = int(feature_dim)
        self.amplitude = float(amplitude)

    def vector(self, symbol: str) -> tuple[float, ...]:
        if not symbol:
            raise ValueError("symbol must be non-empty")
        raw = hashlib.shake_256(symbol.encode("utf-8")).digest(self.feature_dim)
        a = self.amplitude
        return tuple(a if byte & 1 else -a for byte in raw)

    @staticmethod
    def _statement_symbol(statement_id: str) -> str:
        return f"@statement:{statement_id}"

    @staticmethod
    def _slot_symbol(statement_id: str, role: str) -> str:
        return f"@slot:{statement_id}:{role}"

    @staticmethod
    def _qualifier_symbol(statement_id: str, index: int) -> str:
        return f"@qualifier:{statement_id}:{index}"

    def encode(self, statement: KnowledgeStatement) -> EncodedKnowledgeFrame:
        sid = statement.statement_id
        statement_symbol = self._statement_symbol(sid)
        subject_slot = self._slot_symbol(sid, "subject")
        property_slot = self._slot_symbol(sid, "property")
        object_slot = self._slot_symbol(sid, "object")

        symbols: list[str] = [
            statement_symbol,
            subject_slot,
            property_slot,
            object_slot,
            ROLE_SUBJECT,
            ROLE_PROPERTY,
            ROLE_OBJECT,
            statement.subject,
            statement.property,
            statement.value,
        ]
        edges: list[tuple[str, str]] = [
            (statement_symbol, subject_slot),
            (statement_symbol, property_slot),
            (statement_symbol, object_slot),
            (subject_slot, ROLE_SUBJECT),
            (property_slot, ROLE_PROPERTY),
            (object_slot, ROLE_OBJECT),
            (subject_slot, statement.subject),
            (property_slot, statement.property),
            (object_slot, statement.value),
        ]

        for index, qualifier in enumerate(statement.qualifiers):
            qnode = self._qualifier_symbol(sid, index)
            qproperty_slot = f"@slot:{sid}:qualifier:{index}:property"
            qvalue_slot = f"@slot:{sid}:qualifier:{index}:object"
            symbols.extend(
                [
                    qnode,
                    qproperty_slot,
                    qvalue_slot,
                    ROLE_QUALIFIER,
                    qualifier.property,
                    qualifier.value,
                ]
            )
            edges.extend(
                [
                    (statement_symbol, qnode),
                    (qnode, ROLE_QUALIFIER),
                    (qnode, qproperty_slot),
                    (qnode, qvalue_slot),
                    (qproperty_slot, ROLE_PROPERTY),
                    (qvalue_slot, ROLE_OBJECT),
                    (qproperty_slot, qualifier.property),
                    (qvalue_slot, qualifier.value),
                ]
            )

        symbols = list(dict.fromkeys(symbols))
        index_by_symbol = {symbol: i for i, symbol in enumerate(symbols)}
        relation_observations = tuple(
            (
                index_by_symbol[left],
                index_by_symbol[right],
                1,
            )
            for left, right in edges
        )
        provenance = statement.provenance or f"knowledge:{sid}"
        return EncodedKnowledgeFrame(
            frame=ObservationFrame(
                detections=tuple(self.vector(symbol) for symbol in symbols),
                relation_observations=relation_observations,
                provenance=provenance,
            ),
            symbols=tuple(symbols),
            positive_pairs=tuple(edges),
        )


def knowledge_runtime_config_json(
    runtime_type,
    *,
    feature_dim: int = 64,
    max_entities: int = 4096,
    budget: int = 64,
) -> str:
    """Build a runtime config appropriate for already-distilled claims."""

    base = runtime_type(
        feature_dim=feature_dim,
        max_entities=max_entities,
        budget=budget,
    )
    cfg = json.loads(base.config_json())
    cfg["articulation"]["fixed_group"] = [0]
    cfg["graph"]["relation_min_exposure"] = 1
    cfg["graph"]["relation_active_threshold"] = 0.65
    cfg["graph"]["relation_absent_threshold"] = 0.35
    return json.dumps(cfg, sort_keys=True)


def make_knowledge_runtime(
    runtime_type,
    *,
    feature_dim: int = 64,
    max_entities: int = 4096,
    budget: int = 64,
):
    config_json = knowledge_runtime_config_json(
        runtime_type,
        feature_dim=feature_dim,
        max_entities=max_entities,
        budget=budget,
    )
    return runtime_type(config_json=config_json)


def feed_knowledge(
    runtime,
    statements: Iterable[KnowledgeStatement],
    *,
    encoder: KnowledgeSymbolEncoder | None = None,
) -> KnowledgeFeedReport:
    encoder = encoder or KnowledgeSymbolEncoder()
    bindings: dict[str, int] = {}
    conflicts = 0
    frame_count = 0
    statement_count = 0

    for statement in statements:
        encoded = encoder.encode(statement)
        read = runtime.step(
            [list(vector) for vector in encoded.frame.detections],
            list(encoded.frame.relation_observations),
            None,
            [],
            None,
        )
        if len(read.bindings) != len(encoded.symbols):
            raise RuntimeError("Cortex binding count does not match encoded symbols")
        for symbol, binding in zip(encoded.symbols, read.bindings):
            previous = bindings.get(symbol)
            if previous is not None and previous != int(binding):
                conflicts += 1
            bindings[symbol] = int(binding)
        frame_count += 1
        statement_count += 1

    snapshot_json = runtime.snapshot_json()
    snapshot = json.loads(snapshot_json)
    return KnowledgeFeedReport(
        statements=statement_count,
        frames=frame_count,
        symbols=len(bindings),
        cortex_entities=int(snapshot["articulation"]["active_entities"]),
        binding_conflicts=conflicts,
        symbol_bindings=dict(bindings),
        snapshot_json=snapshot_json,
    )


class KnowledgeGraphView:
    """Read statement motifs back from a Cortex runtime snapshot."""

    def __init__(
        self,
        snapshot_json: str,
        symbol_bindings: Mapping[str, int],
        config_json: str | None = None,
    ):
        snapshot = json.loads(snapshot_json)
        self.snapshot_json = snapshot_json
        self.config_json = config_json
        self.bindings = {str(k): int(v) for k, v in symbol_bindings.items()}
        self.reverse = {value: key for key, value in self.bindings.items()}
        self.adjacency: dict[int, set[int]] = {}
        self.edge_posterior: dict[tuple[int, int], float] = {}
        for entry in snapshot["graph"]["relations"]:
            if int(entry["evidence"]["state"]) != 1:
                continue
            a = int(entry["a"])
            b = int(entry["b"])
            evidence = entry["evidence"]
            alpha = float(evidence["a"])
            beta = float(evidence["b"])
            key = (min(a, b), max(a, b))
            self.edge_posterior[key] = alpha / max(1.0e-12, alpha + beta)
            self.adjacency.setdefault(a, set()).add(b)
            self.adjacency.setdefault(b, set()).add(a)

    def _id(self, symbol: str) -> int | None:
        return self.bindings.get(symbol)

    def _neighbors(self, node: int) -> set[int]:
        return self.adjacency.get(node, set())

    def objects(self, subject: str, property: str) -> tuple[str, ...]:
        subject_id = self._id(subject)
        property_id = self._id(property)
        role_subject = self._id(ROLE_SUBJECT)
        role_property = self._id(ROLE_PROPERTY)
        role_object = self._id(ROLE_OBJECT)
        if None in (
            subject_id,
            property_id,
            role_subject,
            role_property,
            role_object,
        ):
            return ()

        subject_slots = self._neighbors(subject_id) & self._neighbors(role_subject)
        property_slots = self._neighbors(property_id) & self._neighbors(role_property)
        out: set[str] = set()

        for subject_slot in subject_slots:
            statements = self._neighbors(subject_slot) - {subject_id, role_subject}
            for statement in statements:
                if not (self._neighbors(statement) & property_slots):
                    continue
                object_slots = {
                    node
                    for node in self._neighbors(statement)
                    if role_object in self._neighbors(node)
                }
                for object_slot in object_slots:
                    candidates = self._neighbors(object_slot) - {
                        statement,
                        role_object,
                    }
                    for candidate in candidates:
                        symbol = self.reverse.get(candidate)
                        if symbol is not None and not symbol.startswith("@"):
                            out.add(symbol)
        return tuple(sorted(out))

    def _edge_support(self, left: int, right: int) -> float:
        return self.edge_posterior.get(
            (min(left, right), max(left, right)),
            0.0,
        )

    def property_coordinates(
        self,
        subject: str,
    ) -> tuple[PropertyValueCoordinate, ...]:
        subject_id = self._id(subject)
        role_subject = self._id(ROLE_SUBJECT)
        role_property = self._id(ROLE_PROPERTY)
        role_object = self._id(ROLE_OBJECT)
        if None in (subject_id, role_subject, role_property, role_object):
            return ()

        rows: dict[tuple[str, str], float] = {}
        subject_slots = self._neighbors(subject_id) & self._neighbors(role_subject)
        for subject_slot in subject_slots:
            for statement in self._neighbors(subject_slot) - {
                subject_id,
                role_subject,
            }:
                property_slots = {
                    node
                    for node in self._neighbors(statement)
                    if role_property in self._neighbors(node)
                }
                object_slots = {
                    node
                    for node in self._neighbors(statement)
                    if role_object in self._neighbors(node)
                }
                for property_slot in property_slots:
                    property_candidates = [
                        node
                        for node in self._neighbors(property_slot)
                        - {statement, role_property}
                        if (
                            self.reverse.get(node) is not None
                            and not self.reverse[node].startswith("@")
                        )
                    ]
                    for object_slot in object_slots:
                        object_candidates = [
                            node
                            for node in self._neighbors(object_slot)
                            - {statement, role_object}
                            if (
                                self.reverse.get(node) is not None
                                and not self.reverse[node].startswith("@")
                            )
                        ]
                        for property_id in property_candidates:
                            for object_id in object_candidates:
                                property_name = self.reverse[property_id]
                                value_name = self.reverse[object_id]
                                edges = (
                                    (subject_id, subject_slot),
                                    (subject_slot, role_subject),
                                    (subject_slot, statement),
                                    (statement, property_slot),
                                    (property_slot, role_property),
                                    (property_slot, property_id),
                                    (statement, object_slot),
                                    (object_slot, role_object),
                                    (object_slot, object_id),
                                )
                                support = min(
                                    self._edge_support(left, right)
                                    for left, right in edges
                                )
                                key = (property_name, value_name)
                                rows[key] = max(rows.get(key, 0.0), support)

        return tuple(
            PropertyValueCoordinate(
                property=property_name,
                value=value,
                structural_support=support,
            )
            for (property_name, value), support in sorted(rows.items())
        )

    def local_articulation(
        self,
        symbol: str,
        *,
        top_k: int = 8,
        min_support: float = 1.0e-6,
    ) -> KnowledgeLocalArticulation:
        if self.config_json is None:
            raise ValueError(
                "config_json is required for local topology articulation"
            )
        entity_id = self._id(symbol)
        if entity_id is None:
            raise KeyError(symbol)
        topology = TopologyView(
            self.snapshot_json,
            self.config_json,
            labels=self.reverse,
        ).at_entity(
            entity_id,
            top_k=top_k,
            min_support=min_support,
        )
        return KnowledgeLocalArticulation(
            topology=topology,
            properties=self.property_coordinates(symbol),
        )


def statement_to_json(statement: KnowledgeStatement) -> dict:
    return {
        "statement_id": statement.statement_id,
        "subject": statement.subject,
        "property": statement.property,
        "value": statement.value,
        "qualifiers": [
            {"property": qualifier.property, "value": qualifier.value}
            for qualifier in statement.qualifiers
        ],
        "rank": statement.rank,
        "references": list(statement.references),
        "provenance": statement.provenance,
    }


def statement_from_json(value: Mapping) -> KnowledgeStatement:
    return KnowledgeStatement(
        statement_id=str(value["statement_id"]),
        subject=str(value["subject"]),
        property=str(value["property"]),
        value=str(value["value"]),
        qualifiers=tuple(
            KnowledgeQualifier(
                property=str(item["property"]),
                value=str(item["value"]),
            )
            for item in value.get("qualifiers", ())
        ),
        rank=str(value.get("rank", "normal")),
        references=tuple(str(item) for item in value.get("references", ())),
        provenance=(
            str(value["provenance"])
            if value.get("provenance") is not None
            else None
        ),
    )
