# Cortex Language Gateway v0.1

Status: **experimental integration prototype**.

This gateway connects Cortex to Ollama without forking Ollama.

The design keeps model inference and structural cognition separate:

```text
OpenAI-compatible client
        |
        v
Cortex Language Gateway
        |
        +--> Ollama structured semantic extraction
        |
        +--> Ollama entity embeddings
        |        |
        |        v
        |    CortexRuntime
        |        |
        |        +--> persistent entities / relations
        |        +--> memory / revision / novelty signals
        |        +--> optional diagnostics
        |
        +--> Cortex context packet
                 |
                 v
            Ollama answer
                 |
                 v
              client
```

The gateway does not alter the frozen executable specification. It is an
experimental orchestration layer around the existing native runtime.

## Why this is a gateway rather than an Ollama fork

Ollama already provides:

- chat inference;
- structured JSON-schema output;
- embedding generation;
- model loading and scheduling;
- an OpenAI-compatible API surface.

Cortex does not currently need to intervene inside token generation. Keeping
the two processes separate avoids inheriting Ollama's inference-engine
maintenance while allowing us to experiment with persistent cognition.

A fork remains possible later if Cortex needs hooks inside generation itself.

## Request pipeline

For each new user message:

1. Ollama receives a constrained semantic-extraction request.
2. Every extracted entity/relation must include a verbatim evidence snippet
   anchored in the current user message; unanchored items are rejected.
3. Entity descriptions are embedded through Ollama's embedding endpoint.
4. The embedding vectors become Cortex detections.
5. Extracted relations become Cortex relation observations for the current
   frame.
6. Cortex updates persistent entity identity, relation evidence, memory, and
   continual state.
7. The gateway builds a compact Cortex context packet.
8. Ollama generates the natural-language answer using the packet plus recent
   conversation history.

The LLM is therefore the language perception/articulation layer while Cortex
owns persistent structural state.

## Diagnostics switch

Diagnostics are disabled by default unless configured otherwise.

They can be enabled persistently for one session:

```http
POST /cortex/sessions/research/diagnostics
Content-Type: application/json

{"enabled": true}
```

or for one chat request:

```http
POST /v1/chat/completions
X-Cortex-Session: research
X-Cortex-Diagnostics: true
Content-Type: application/json
```

Diagnostics are derived from the Cortex runtime read only.

Current diagnostic signals include:

- unresolved continual state;
- unresolved historical-memory retrieval;
- memory budget pressure;
- discovery;
- revision;
- reactivation;
- relation/causal evidence changes;
- diffuse prototype membership;
- weak nuisance-subgroup evidence margin.

The structured diagnosis is deterministic. The natural-language summary is
also generated deterministically in v0.1.

Most importantly:

> Diagnostic text is never re-ingested as an external-world observation.

This prevents a self-description such as "I am uncertain" from becoming new
evidence that the external world itself is uncertain.

## HTTP surface

The prototype exposes:

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`
- `GET /cortex/sessions`
- `GET /cortex/sessions/{session_id}`
- `POST /cortex/sessions/{session_id}/save`
- `POST /cortex/sessions/{session_id}/load`
- `POST /cortex/sessions/{session_id}/diagnostics`
- `DELETE /cortex/sessions/{session_id}`

The public OpenAI model name defaults to `cortex`.

The actual Ollama model is configured independently.

### Session selection

Use:

```http
X-Cortex-Session: research
```

or the request extension:

```json
{
  "cortex_session": "research"
}
```

If no session is supplied, the gateway uses `default`.

## Installation

Build/install Cortex as usual, then install the optional gateway dependencies:

```shell
pip install -e '.[gateway]'
```

Run Ollama locally and ensure the requested models are installed.

Example defaults:

```shell
ollama pull qwen3:8b
ollama pull embeddinggemma
```

Start the gateway:

```shell
cortex-gateway \
  --chat-model qwen3:8b \
  --embedding-model embeddinggemma \
  --state-dir ~/.cortex/gateway \
  --port 11435
```

The gateway then listens on:

```text
http://127.0.0.1:11435
```

## OpenAI client example

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:11435/v1",
    api_key="cortex",
)

response = client.chat.completions.create(
    model="cortex",
    messages=[
        {
            "role": "user",
            "content": "Alice works with Bob on Cortex.",
        }
    ],
    extra_headers={
        "X-Cortex-Session": "research",
        "X-Cortex-Diagnostics": "true",
    },
)

print(response.choices[0].message.content)
```

## Current context packet

The language model receives an internal JSON packet containing a bounded view
of:

- persistent Cortex entity IDs;
- observed names/kinds/descriptions;
- user-message evidence snippets and extraction provenance;
- persistent labeled relations from the semantic gateway;
- current prediction;
- unresolved/revision/reactivation/discovery signals;
- memory pressure;
- optional structured diagnostics.

The packet is intentionally bounded. v0.1 does not dump the complete Cortex
snapshot into the LLM context.

## Semantic relation handling

The native Cortex graph currently represents relation evidence between bound
entities without preserving arbitrary natural-language relation labels.

For v0.1:

- the native graph receives pair-level asserted/not-asserted evidence;
- the gateway keeps the extracted natural-language relation label in a
  separate semantic ledger keyed by persistent Cortex entity IDs.

A later language-facing semantic layer should make typed relation semantics a
first-class bridge rather than a gateway-side ledger.

## Persistence

Gateway sessions are persisted by default under:

```text
~/.cortex/gateway
```

The location can be changed with:

```shell
cortex-gateway --state-dir /path/to/cortex-state
```

or:

```shell
export CORTEX_STATE_DIR=/path/to/cortex-state
```

Persistence can be disabled explicitly with:

```shell
cortex-gateway --no-persistence
```

### Event-sourced reconstruction

The native runtime currently exposes snapshot serialization but not a native
snapshot-restore constructor. Gateway v0.1 therefore uses a versioned
event-sourced session format.

Each persisted frame stores:

- the validated semantic entities and relations;
- their source-evidence snippets;
- the **exact embedding vectors** originally submitted to Cortex.

On restore, the gateway creates a fresh native runtime and replays those saved
vectors in their original order.

It does **not**:

- call the semantic extractor again;
- call the embedding model again;
- reinterpret historical text.

This matters because changing or upgrading the Ollama embedding model cannot
silently rewrite old Cortex memory.

The session file also stores:

- recent conversational history;
- the diagnostics switch;
- the chat-model identifier used when the session was written;
- the embedding-model identifier;
- Cortex software/specification identifiers;
- a native runtime snapshot witness when the native binding exposes one.

The **chat model may change** across restarts: it is an articulation component,
not part of Cortex's stored metric space.

The **embedding model must match** before a persisted session is loaded. New
observations must remain in the same embedding coordinate system as the vectors
already stored in that Cortex session.

When a native snapshot witness is present, the gateway also compares the
replayed native state against that saved witness and rejects a divergent replay.

### Atomic files and schema version

Session files use the schema identifier:

```text
cortex-language-session-v1
```

Writes go through a temporary file followed by an atomic filesystem replace.

File names contain a sanitized session label plus a hash of the complete
session ID, avoiding direct use of arbitrary user-provided paths.

### Automatic and explicit lifecycle

A successful chat turn automatically saves its session.

The first access to a session ID automatically restores it when a matching
persisted session exists.

Explicit operations are also available:

```http
GET  /cortex/sessions
POST /cortex/sessions/research/save
POST /cortex/sessions/research/load
```

Deleting a session through:

```http
DELETE /cortex/sessions/research
```

removes both the active in-memory session and its persisted file by default.

The persistence design is intentionally backend-neutral. A future SQLite,
object-store, or graph-database implementation can replace the file store
without changing the chat/session API.

### Local data note

Gateway v0.1 session files are ordinary local JSON files. They contain recent
conversation history, extracted semantic evidence, entity/relation metadata,
and embedding vectors. They are **not encrypted by Cortex**. Use an
appropriately protected state directory if the conversation is sensitive.

## Deliberate v0.1 limitations

The prototype currently does **not** provide:

- streaming chat completions;
- tool-call orchestration;
- native O(1) snapshot restore (v0.1 rebuilds the runtime by deterministic event replay);
- full typed natural-language relation semantics inside the native graph;
- source-confidence weighting;
- semantic retrieval over large Cortex histories;
- promotion of recent research-only reasoning modules into the native runtime;
- automatic use of proof dependency / novelty-aware replanning in language
  conversations;
- compound multimodal observations.

The extraction model is explicitly instructed not to infer unstated facts.
Every extracted entity and relation must also carry a verbatim evidence snippet
that the gateway verifies against the current user message before updating
Cortex. This provides a first provenance barrier against extractor
hallucinations.

It does **not** yet provide source-confidence weighting, contradiction-aware
trust policies, or cryptographic/source-system provenance. Those remain
required before treating conversational memory as high-trust long-term
knowledge.

## Diagnostics invariance requirement

The core safety/engineering invariant for this milestone is:

```text
same user observation + diagnostics off
and
same user observation + diagnostics on
        =>
same Cortex observation/update
```

The diagnostics switch may change only:

- the introspection packet;
- the answer context available to the language model.

It must not create an additional Cortex step or external-world observation.

The unit suite explicitly checks this invariant.

## Next integration work

Recommended follow-ups:

1. source-confidence and trust-policy envelopes;
2. semantic retrieval for context selection;
3. typed language relations;
4. native snapshot restoration to replace event replay when promoted safely;
5. tool calling;
6. streaming;
7. metacognitive self-diagnosis over historical trends;
8. language-level benchmark comparing Ollama alone versus Ollama + Cortex;
9. integration of passed research primitives such as explicit proof dependency
   and novelty-aware replanning behind a stable orchestration API.
