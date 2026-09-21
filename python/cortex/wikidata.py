"""Wikibase JSON -> neutral Cortex knowledge statements."""
from __future__ import annotations

import bz2
from collections.abc import Iterable, Iterator, Mapping
import gzip
import json
from pathlib import Path
from typing import Any, TextIO

from .knowledge import KnowledgeQualifier, KnowledgeStatement


def _canonical_data_value(snak: Mapping[str, Any]) -> str:
    property_name = str(snak.get("property", "unknown"))
    snaktype = str(snak.get("snaktype", "value"))
    if snaktype != "value":
        return f"wd:{snaktype}:{property_name}"

    datavalue = snak.get("datavalue")
    if not isinstance(datavalue, Mapping):
        return f"wd:missing:{property_name}"
    value = datavalue.get("value")
    value_type = str(datavalue.get("type", "unknown"))

    if value_type == "wikibase-entityid" and isinstance(value, Mapping):
        entity_id = value.get("id")
        if entity_id:
            return f"wd:{entity_id}"

    if value_type == "time" and isinstance(value, Mapping):
        return (
            "wdlit:time:"
            f"{value.get('time', '')}"
            f"|precision={value.get('precision', '')}"
            f"|calendar={value.get('calendarmodel', '')}"
        )

    if value_type == "quantity" and isinstance(value, Mapping):
        return (
            "wdlit:quantity:"
            f"{value.get('amount', '')}"
            f"|unit={value.get('unit', '')}"
            f"|lower={value.get('lowerBound', '')}"
            f"|upper={value.get('upperBound', '')}"
        )

    if value_type == "globecoordinate" and isinstance(value, Mapping):
        return (
            "wdlit:coordinate:"
            f"{value.get('latitude', '')},{value.get('longitude', '')}"
            f"|precision={value.get('precision', '')}"
            f"|globe={value.get('globe', '')}"
        )

    if value_type == "monolingualtext" and isinstance(value, Mapping):
        return (
            "wdlit:text:"
            f"{value.get('language', '')}:{value.get('text', '')}"
        )

    payload = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return f"wdlit:{value_type}:{payload}"


def statements_from_entity(
    entity: Mapping[str, Any],
    *,
    include_deprecated: bool = False,
    provenance: str = "wikidata",
) -> Iterator[KnowledgeStatement]:
    entity_id = entity.get("id")
    if not entity_id:
        return
    subject = f"wd:{entity_id}"
    claims = entity.get("claims", {})
    if not isinstance(claims, Mapping):
        return

    for property_name, property_claims in claims.items():
        if not isinstance(property_claims, Iterable):
            continue
        for claim_index, claim in enumerate(property_claims):
            if not isinstance(claim, Mapping):
                continue
            rank = str(claim.get("rank", "normal"))
            if rank == "deprecated" and not include_deprecated:
                continue
            mainsnak = claim.get("mainsnak")
            if not isinstance(mainsnak, Mapping):
                continue

            qualifiers: list[KnowledgeQualifier] = []
            raw_qualifiers = claim.get("qualifiers", {})
            if isinstance(raw_qualifiers, Mapping):
                for qualifier_property, snaks in raw_qualifiers.items():
                    if not isinstance(snaks, Iterable):
                        continue
                    for snak in snaks:
                        if isinstance(snak, Mapping):
                            qualifiers.append(
                                KnowledgeQualifier(
                                    property=f"wd:{qualifier_property}",
                                    value=_canonical_data_value(snak),
                                )
                            )

            references: list[str] = []
            for reference in claim.get("references", ()) or ():
                if isinstance(reference, Mapping) and reference.get("hash"):
                    references.append(str(reference["hash"]))

            claim_id = str(
                claim.get("id")
                or f"{entity_id}-{property_name}-{claim_index}"
            )
            yield KnowledgeStatement(
                statement_id=claim_id,
                subject=subject,
                property=f"wd:{property_name}",
                value=_canonical_data_value(mainsnak),
                qualifiers=tuple(qualifiers),
                rank=rank,
                references=tuple(references),
                provenance=provenance,
            )


def _open_text(path: Path) -> TextIO:
    if path.suffix == ".bz2":
        return bz2.open(path, mode="rt", encoding="utf-8")
    if path.suffix == ".gz":
        return gzip.open(path, mode="rt", encoding="utf-8")
    return path.open(mode="rt", encoding="utf-8")


def iter_dump_entities(
    path: str | Path,
    *,
    wanted_ids: set[str] | None = None,
) -> Iterator[dict[str, Any]]:
    """Stream Wikidata's JSON-array dump without loading it into memory."""

    path = Path(path)
    with _open_text(path) as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line in {"[", "]"}:
                continue
            if line.endswith(","):
                line = line[:-1]
            entity = json.loads(line)
            if not isinstance(entity, dict):
                continue
            entity_id = entity.get("id")
            if wanted_ids is not None and entity_id not in wanted_ids:
                continue
            yield entity


def iter_dump_statements(
    path: str | Path,
    *,
    wanted_ids: set[str] | None = None,
    include_deprecated: bool = False,
) -> Iterator[KnowledgeStatement]:
    for entity in iter_dump_entities(path, wanted_ids=wanted_ids):
        yield from statements_from_entity(
            entity,
            include_deprecated=include_deprecated,
            provenance=f"wikidata-dump:{Path(path).name}",
        )
