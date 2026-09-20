"""Persistence and gateway wrapper for Cortex Hybrid Core."""
from __future__ import annotations

from hashlib import sha256
from json import dumps, loads
from os import replace as atomic_replace
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Mapping, Sequence

from .hybrid_core import (
    BootstrapRecord,
    HYBRID_CORE_VERSION,
    HybridCoreConfig,
    HybridCoreSession,
)
from .hybrid_exchange import (
    HybridOrchestrationPolicy,
)
from .language_gateway import ChatResult
from .neural_runtime import (
    NeuralRuntime,
    OllamaNeuralRuntime,
)


HYBRID_CORE_SESSION_SCHEMA = (
    "cortex-hybrid-core-session-v1"
)


class HybridCoreFileSessionStore:
    def __init__(
        self,
        root: str | Path,
    ) -> None:
        self.root = Path(
            root
        ).expanduser()
        self.root.mkdir(
            parents=True,
            exist_ok=True,
        )

    @staticmethod
    def _safe_name(
        session_id: str,
    ) -> str:
        digest = sha256(
            session_id.encode(
                "utf-8"
            )
        ).hexdigest()[:16]
        readable = "".join(
            char
            if (
                char.isalnum()
                or char
                in {
                    "-",
                    "_",
                }
            )
            else "_"
            for char in session_id
        ).strip(
            "_"
        )[:48]
        return (
            f"{readable or 'session'}"
            f"-hybrid-v1-{digest}.json"
        )

    def path_for(
        self,
        session_id: str,
    ) -> Path:
        return (
            self.root
            / self._safe_name(
                session_id
            )
        )

    def exists(
        self,
        session_id: str,
    ) -> bool:
        return self.path_for(
            session_id
        ).is_file()

    def save(
        self,
        session: HybridCoreSession,
    ) -> Path:
        target = self.path_for(
            session.session_id
        )
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(
                self.root
            ),
            prefix=".cortex-hybrid-",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(
                dumps(
                    _session_packet(
                        session
                    ),
                    sort_keys=True,
                    indent=2,
                )
            )
            handle.write(
                "\n"
            )
            temporary = Path(
                handle.name
            )
        atomic_replace(
            temporary,
            target,
        )
        return target

    def load_packet(
        self,
        session_id: str,
    ) -> Mapping[str, Any]:
        path = self.path_for(
            session_id
        )
        if not path.is_file():
            raise FileNotFoundError(
                "no persisted Hybrid Core session: "
                + session_id
            )
        payload = loads(
            path.read_text(
                encoding="utf-8"
            )
        )
        if (
            not isinstance(
                payload,
                dict,
            )
            or payload.get(
                "schema"
            )
            != HYBRID_CORE_SESSION_SCHEMA
        ):
            raise ValueError(
                "unsupported Hybrid Core session schema"
            )
        if payload.get(
            "session_id"
        ) != session_id:
            raise ValueError(
                "Hybrid Core session ID mismatch"
            )
        return payload

    def delete(
        self,
        session_id: str,
    ) -> bool:
        path = self.path_for(
            session_id
        )
        if not path.exists():
            return False
        path.unlink()
        return True

    def list_sessions(
        self,
    ) -> list[str]:
        sessions: list[
            str
        ] = []
        for path in sorted(
            self.root.glob(
                "*-hybrid-v1-*.json"
            )
        ):
            try:
                payload = loads(
                    path.read_text(
                        encoding="utf-8"
                    )
                )
            except Exception:
                continue
            if (
                isinstance(
                    payload,
                    dict,
                )
                and payload.get(
                    "schema"
                )
                == HYBRID_CORE_SESSION_SCHEMA
                and isinstance(
                    payload.get(
                        "session_id"
                    ),
                    str,
                )
            ):
                sessions.append(
                    payload[
                        "session_id"
                    ]
                )
        return sessions


def _session_packet(
    session: HybridCoreSession,
) -> dict[str, Any]:
    packet = (
        session.persistence_packet()
    )
    packet[
        "schema"
    ] = HYBRID_CORE_SESSION_SCHEMA
    packet[
        "gateway_version"
    ] = HYBRID_CORE_VERSION
    packet[
        "bootstrap_records"
    ] = [
        record.packet()
        for record
        in session.bootstrap_records.values()
    ]
    packet[
        "relation_provenance"
    ] = dict(
        session.relation_provenance
    )
    return packet


def _restore_session_packet(
    session: HybridCoreSession,
    payload: Mapping[str, Any],
) -> None:
    session.restore_packet(
        payload
    )
    raw_records = payload.get(
        "bootstrap_records",
        [],
    )
    if not isinstance(
        raw_records,
        list,
    ):
        raise ValueError(
            "Hybrid Core bootstrap records must be an array"
        )
    raw_provenance = payload.get(
        "relation_provenance",
        {},
    )
    if not isinstance(
        raw_provenance,
        dict,
    ):
        raise ValueError(
            "Hybrid Core relation provenance must be an object"
        )
    session.relation_provenance = {
        str(key): str(value)
        for key, value
        in raw_provenance.items()
    }

    session.bootstrap_records.clear()
    for item in raw_records:
        if not isinstance(
            item,
            dict,
        ):
            continue
        record = BootstrapRecord(
            subject=str(
                item.get(
                    "subject",
                    "",
                )
            ),
            predicate=str(
                item.get(
                    "predicate",
                    "",
                )
            ),
            object=str(
                item.get(
                    "object",
                    "",
                )
            ),
            asserted=bool(
                item.get(
                    "asserted",
                    True,
                )
            ),
            confidence=float(
                item.get(
                    "confidence",
                    0.0,
                )
            ),
            topic=str(
                item.get(
                    "topic",
                    "",
                )
            ),
            model=str(
                item.get(
                    "model",
                    "",
                )
            ),
            provenance=str(
                item.get(
                    "provenance",
                    "neural_bootstrap",
                )
            ),
        )
        session.bootstrap_records[
            record.key
        ] = record


class HybridCoreGateway:
    def __init__(
        self,
        config: HybridCoreConfig,
        *,
        runtime: NeuralRuntime | None = None,
        orchestration_policy: (
            HybridOrchestrationPolicy
            | None
        ) = None,
    ) -> None:
        self.config = config
        self.runtime = (
            runtime
            or OllamaNeuralRuntime(
                config.ollama_url
            )
        )
        self.orchestration_policy = (
            orchestration_policy
        )
        self.sessions: dict[
            str,
            HybridCoreSession,
        ] = {}
        self.store = (
            HybridCoreFileSessionStore(
                config.persistence_dir
            )
            if config.persistence_dir
            else None
        )

    def _new_session(
        self,
        session_id: str,
    ) -> HybridCoreSession:
        return HybridCoreSession(
            session_id,
            self.config,
            self.runtime,
            orchestration_policy=(
                self.orchestration_policy
            ),
        )

    def session(
        self,
        session_id: str,
    ) -> HybridCoreSession:
        normalized = (
            session_id.strip()
        )
        if not normalized:
            raise ValueError(
                "session_id must not be empty"
            )
        existing = (
            self.sessions.get(
                normalized
            )
        )
        if existing is not None:
            return existing

        session = (
            self._new_session(
                normalized
            )
        )
        if (
            self.store is not None
            and self.store.exists(
                normalized
            )
        ):
            _restore_session_packet(
                session,
                self.store.load_packet(
                    normalized
                ),
            )
        self.sessions[
            normalized
        ] = session
        return session

    def chat(
        self,
        session_id: str,
        user_text: str,
        *,
        diagnostics: bool | None = None,
    ) -> ChatResult:
        session = self.session(
            session_id
        )
        result = session.chat(
            user_text,
            diagnostics=diagnostics,
        )
        if self.store is not None:
            self.store.save(
                session
            )
        return result

    def bootstrap(
        self,
        session_id: str,
        topics: Sequence[str],
    ) -> tuple[
        BootstrapRecord,
        ...,
    ]:
        session = self.session(
            session_id
        )
        imported = (
            session.bootstrap_topics(
                topics
            )
        )
        if self.store is not None:
            self.store.save(
                session
            )
        return imported

    def set_diagnostics(
        self,
        session_id: str,
        enabled: bool,
    ) -> None:
        session = self.session(
            session_id
        )
        session.set_diagnostics(
            enabled
        )
        if self.store is not None:
            self.store.save(
                session
            )

    def save(
        self,
        session_id: str,
    ) -> str:
        if self.store is None:
            raise RuntimeError(
                "gateway persistence is disabled"
            )
        return str(
            self.store.save(
                self.session(
                    session_id
                )
            )
        )

    def load(
        self,
        session_id: str,
    ) -> HybridCoreSession:
        if self.store is None:
            raise RuntimeError(
                "gateway persistence is disabled"
            )
        session = (
            self._new_session(
                session_id
            )
        )
        _restore_session_packet(
            session,
            self.store.load_packet(
                session_id
            ),
        )
        self.sessions[
            session_id
        ] = session
        return session

    def reset(
        self,
        session_id: str,
        *,
        delete_persisted: bool = True,
    ) -> None:
        self.sessions.pop(
            session_id,
            None,
        )
        if (
            delete_persisted
            and self.store is not None
        ):
            self.store.delete(
                session_id
            )

    def list_persisted_sessions(
        self,
    ) -> list[str]:
        if self.store is None:
            return []
        return (
            self.store.list_sessions()
        )
