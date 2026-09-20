# Cortex Language Gateway v0.3 — Hybrid Cognition

Status: **experimental implementation**.

Gateway v0.3 changes the responsibility split between the neural language
model and Cortex.

The neural model is again allowed to do what pretrained language models are
good at:

- natural-language understanding;
- conversational continuity;
- broad pretrained factual knowledge;
- ordinary answer generation.

Cortex is used as a separate structural/metacognitive layer for:

- explicit session memory;
- structural relation reasoning;
- corroboration;
- contradiction detection;
- uncertainty;
- revision;
- novelty/diagnostic state;
- later planning/proof/reasoning integrations.

The core design is:

```text
                         user
                          |
                          v
                +-------------------+
                | neural chat model |
                | language + broad  |
                | pretrained facts  |
                +---------+---------+
                          |
                    candidate answer
                          |
             +------------+-------------+
             |                          |
             v                          v
        PromptIR                  CandidateAnswerIR
             |                          |
             +------------+-------------+
                          |
                          v
                      +--------+
                      | Cortex |
                      +---+----+
                          |
                    DiagnosisIR
                          |
                          v
        neural answer + Cortex diagnosis
```

## Epistemic separation

The important distinction is no longer "LLM facts are forbidden".

Instead:

```text
neural pretrained claim != observed fact
```

Candidate claims extracted from the neural answer are tagged as:

```text
provenance = neural_prior
```

They are **not written to Cortex memory**.

Only explicit user observations accepted through PromptIR may update the
structural session state in v0.3.

This lets Cortex say:

- "the neural model knows this, but Cortex has no matching session evidence";
- "Cortex independently corroborates this";
- "this conflicts with Cortex's stored evidence";
- "Cortex could not structurally assess this turn".

without pretending that pretrained model recall is direct observation.

## Neural answer comes first

A valid neural answer must not depend on Cortex successfully parsing the turn.

The pipeline is therefore intentionally ordered:

1. generate the ordinary neural answer;
2. parse the user prompt for Cortex;
3. update/query Cortex using explicit user observations only;
4. parse factual claims from the neural candidate answer;
5. compare those claims with Cortex state;
6. append a Cortex-owned diagnosis.

If steps 2-5 fail, step 1 still returns.

This is a major difference from strict v0.2, where malformed PromptIR could
prevent the response.

## Example: pretrained knowledge

User:

```text
What is the capital of Mongolia?
```

Neural answer:

```text
The capital of Mongolia is Ulaanbaatar.
```

If Cortex has no matching session evidence, v0.3 reports approximately:

```text
Cortex diagnosis: The neural model supplied the factual answer from pretrained
knowledge. Cortex found no matching stored session evidence, so it treated
those claims as neural priors and did not write them into memory.
```

The answer is not blocked just because Cortex had never learned Ulaanbaatar.

## Example: corroboration

After the explicit user observation:

```text
Alice works with Bob.
```

a later question:

```text
Does Alice work with Bob?
```

can produce:

```text
Alice works with Bob.

Cortex diagnosis: The neural model supplied the answer and Cortex independently
found compatible support in its structural session state.
```

## Example: conflict

If Cortex previously stored:

```text
Alice does not work with Bob.
```

while the neural answer later says:

```text
Alice works with Bob.
```

the diagnosis reports a conflict.

Cortex does not silently replace its stored relation and does not force the
neural model's claim into memory.

## Conversation separation

The user-visible answer contains the Cortex diagnosis.

The neural chat model's own conversation history does **not**.

Only the neural answer itself is written to the model-facing chat history:

```text
user -> neural answer
user -> neural answer
...
```

The Cortex diagnosis is maintained separately.

This prevents the neural model from gradually treating Cortex's own
metacognitive reports as external-world facts.

## Deterministic Cortex representation

Like strict v0.2, hybrid v0.3 does not use a neural embedding model for Cortex
state.

Prompt entities are encoded with:

```text
sha256-symbolic-v1
```

This keeps the internal Cortex geometry deterministic and separate from the
knowledge model.

The neural model is allowed to contribute semantic/factual proposals, but it
does not directly control Cortex's metric state.

## Running

Create the structured parser model if needed:

```shell
ollama create cortex-parser -f examples/ollama/Modelfile.parser
```

Run the recommended chatbot mode:

```shell
cortex-gateway \
  --hybrid-cognition \
  --chat-model qwen3:8b \
  --parser-model cortex-parser \
  --state-dir ~/.cortex/gateway \
  --trace-stages
```

No embedding model or strict renderer model is required.

### Alpaca transport

For Alpaca, use the native **Ollama (External)** instance type:

```text
URL: http://127.0.0.1:11435
Model: cortex:latest
```

The gateway exposes:

- `GET /api/tags`;
- `POST /api/show`;
- `GET /api/ps`;
- `POST /api/chat`.

Streaming chat uses Ollama-compatible newline-delimited JSON rather than
OpenAI SSE. Alpaca's structured title-generation request is detected and
answered as a utility operation without entering Cortex state.

### Alpaca diagnosis UI experiment

On the `feature/alpaca-cortex-metadata` branch, the Ollama transport maps
the hybrid result onto three distinct UI channels:

```text
Ollama message.content  -> neural answer
Ollama message.thinking -> Cortex diagnosis
Ollama timing fields    -> performance metadata
```

The Cortex diagnosis is a structured procedural assessment, not hidden
chain-of-thought. It can contain the diagnosis status, reasoning mode,
corroboration/conflict counts, memory-change state, runtime flags, and
analysis warnings.

Alpaca renders `message.thinking` in its existing reasoning/thought panel.
Its metadata table is limited by Alpaca to standard Ollama timing/token
fields, so Cortex-specific semantic diagnostics are intentionally not encoded
into fake performance metrics.

The feature remains isolated from `main` until validated interactively.

The OpenAI-compatible `/v1/chat/completions` surface remains available, but
the native Ollama path is the recommended Alpaca integration because it matches
Alpaca's first-class Ollama client directly.

The terminal reports stages as:

```text
[cortex-hybrid:default] knowledge:start
[cortex-hybrid:default] knowledge:done (...)
[cortex-hybrid:default] cortex-prompt:start
[cortex-hybrid:default] cortex-prompt:done (...)
[cortex-hybrid:default] candidate:start
[cortex-hybrid:default] candidate:done (...)
```

## Modes

The gateway now has three explicit experimental configurations.

### v0.1 legacy

Original neural extraction + embedding + Cortex context generation.

### v0.2 strict neural boundary

```text
text -> parser -> Cortex -> AnswerIR -> renderer -> text
```

Use this for knowledge-isolation experiments and benchmarks.

### v0.3 hybrid cognition

```text
neural knowledge answer + Cortex structural diagnosis
```

This is the recommended mode for interactive chatbot testing.

## Persistence

Hybrid sessions use a separate schema:

```text
cortex-language-session-v3
```

with separate `-v3-` session files.

This avoids silently interpreting strict v0.2 state as hybrid conversational
state.

Only Cortex structural frames and neural-only conversation history are
persisted.

## Current scope

v0.3 currently diagnoses candidate binary relation claims against the native
Cortex relation/session state.

It does not yet route every research reasoning module through the chatbot.

In particular, later integrations should expose:

- proof dependency;
- recursive typed composition;
- planning/replanning;
- novelty-aware replanning;
- belief revision;
- structural analogy;
- richer typed/causal relations.

The important architectural contract should remain:

```text
LLM supplies language and broad knowledge.
Cortex supplies explicit structural cognition about that knowledge.
```
