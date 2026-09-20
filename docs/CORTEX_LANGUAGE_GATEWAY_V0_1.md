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
2. The extraction contains only entities and explicit relations from the
   current user message.
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
- `GET /cortex/sessions/{session_id}`
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

Session state is currently in memory for the lifetime of the gateway process.

The native runtime already supports snapshot serialization, but full restore and
gateway-session persistence are not yet exposed.

The next persistence step should add:

```text
save(session)
load(session)
list sessions
```

without requiring a graph database initially.

## Deliberate v0.1 limitations

The prototype currently does **not** provide:

- streaming chat completions;
- tool-call orchestration;
- persistent session restore across gateway restarts;
- full typed natural-language relation semantics inside the native graph;
- source-confidence weighting;
- semantic retrieval over large Cortex histories;
- promotion of recent research-only reasoning modules into the native runtime;
- automatic use of proof dependency / novelty-aware replanning in language
  conversations;
- compound multimodal observations.

The extraction model is explicitly instructed not to infer unstated facts, but
v0.1 still relies on LLM extraction quality. Provenance/confidence hardening is
therefore a required follow-up before treating conversational memory as
high-trust long-term knowledge.

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

1. provenance + source-confidence envelopes;
2. persistent save/restore;
3. semantic retrieval for context selection;
4. typed language relations;
5. tool calling;
6. streaming;
7. metacognitive self-diagnosis over historical trends;
8. language-level benchmark comparing Ollama alone versus Ollama + Cortex;
9. integration of passed research primitives such as explicit proof dependency
   and novelty-aware replanning behind a stable orchestration API.
