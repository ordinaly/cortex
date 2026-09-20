# Cortex Hybrid Core v0.1

Status: **experimental / draft**.

Hybrid Core v0.1 rebuilds the neural/Cortex boundary around a provider-independent
runtime and a smaller number of neural inference passes.

## Architecture

```text
                         Cortex Hybrid Core
                                |
                  +-------------+-------------+
                  |                           |
                  v                           v
             Cortex state               NeuralRuntime
                                              |
                                  +-----------+-----------+
                                  |                       |
                                  v                       v
                           Ollama backend           future backends
                                                   (in-process,
                                                    remote, etc.)
```

Cortex does not depend on Qwen or Ollama directly. It depends on the
`NeuralRuntime` contract.

## NeuralRuntime capabilities

Backends advertise capabilities explicitly:

- generation;
- structured generation;
- streaming;
- embeddings;
- logits;
- hidden states;
- KV-cache reuse;
- adapter activation;
- adapter training.

The initial `OllamaNeuralRuntime` supports generation, structured generation,
and embeddings. It deliberately reports deeper capabilities as unavailable.

This is important: Cortex can evolve toward tighter NN integration without
hard-coding assumptions into the cognitive architecture.

## Fused neural turn

Gateway v0.3 used three neural passes for a normal turn:

```text
1. user -> PromptIR
2. user -> neural answer
3. neural answer -> CandidateAnswerIR
```

Hybrid Core v0.1 fuses steps 2 and 3:

```text
1. user -> PromptIR
2. user + Cortex context -> {answer, candidate sidecar}
```

The hidden sidecar contains only claims explicitly present in the natural
answer. Cortex validates that sidecar against its structural state.

Expected call counts:

```text
normal turn                 2
one Cortex conflict         3
```

A conflict revision also returns an answer plus sidecar, so there is no
additional post-hoc parse pass.

## Provenance

Hybrid Core keeps distinct epistemic roles:

```text
user observation
neural prior
Cortex derivation
neural bootstrap
```

A neural prior is not automatically written into Cortex.

A neural bootstrap claim is also not treated as verified truth. It is a
provisional seed that Cortex may later corroborate, revise, or reject.

Hybrid Core also prevents **circular corroboration**. If a neural answer agrees
with a relation that originally came only from neural bootstrap, the diagnosis
is `bootstrap_aligned`, not `supported`. If it disagrees, the diagnosis is
`bootstrap_divergence`, not a hard Cortex conflict.

Only independent evidence such as an explicit user observation can promote the
stored relation's provenance so that later agreement counts as corroboration.

## Neural bootstrap memory

An existing pretrained neural model can seed small parts of Cortex memory.

Example conceptual flow:

```text
Qwen / other NN
      |
      | structured bootstrap compiler
      v
provisional claims
      |
      | confidence + provenance gate
      v
Cortex memory
```

Each imported record contains:

- subject;
- predicate;
- object;
- assertion polarity;
- heuristic neural confidence;
- source topic;
- source model;
- `provenance=neural_bootstrap`;
- `status=provisional`.

The default minimum heuristic confidence is 0.75.

The model's confidence is **not calibrated truth**. It is only an import filter.

### Why bootstrap is topic-scoped

Hybrid Core intentionally does not attempt to dump an entire language model
into Cortex.

The NN remains the broad latent knowledge field. Cortex articulates only useful
parts into explicit memory.

This gives a lazy compilation model:

```text
latent neural knowledge
       |
       | relevance / bootstrap request
       v
provisional Cortex structure
       |
       | independent evidence
       v
supported / revised structure
```

## Bootstrap API

When running Hybrid Core, a session can be seeded through:

```text
POST /cortex/sessions/{session_id}/bootstrap
```

Body:

```json
{
  "topics": [
    "basic European geography",
    "TCP/IP fundamentals"
  ]
}
```

The response lists the provisional records actually imported.

Bootstrap is deliberately explicit. Normal chat does not silently copy neural
claims into Cortex memory.

## Alpaca unified backend

Hybrid Core is still exposed to Alpaca as one Ollama-compatible model:

```text
Instance type: Ollama (External)
URL: http://127.0.0.1:11435
Model: cortex:latest
```

Run:

```bash
cortex-gateway \
  --hybrid-runtime \
  --chat-model qwen3:8b \
  --parser-model cortex-parser \
  --state-dir ~/.cortex/gateway \
  --trace-stages
```

The transport mapping is:

```text
message.content  -> final answer
message.thinking -> Cortex diagnosis + Hybrid Runtime report
Ollama metadata  -> measured stage timings
```

The Alpaca transport emits one complete Thought event and one complete answer
event. This is a compatibility workaround for Alpaca's GTK idle-queue behavior,
not part of Cortex cognition.

## Runtime observability

Every Hybrid Core response reports:

- backend name;
- backend capabilities;
- neural call count;
- per-call stage;
- per-call latency;
- prompt/completion token counts when the backend supplies them;
- revision-cycle count.

Example:

```text
Hybrid runtime
Backend: ollama
Neural calls: 2
Revision cycles: 0
Stages: prompt_parse -> answer_with_sidecar
```

This makes performance regressions visible instead of hiding them behind total
request latency.

## Current limitation

Ollama remains an external inference server in v0.1. Hybrid Core does **not**
yet share:

- transformer hidden states;
- KV cache;
- logits;
- adapter tensors;
- optimizer state.

The architecture now has explicit capability slots for those features.

The next backend experiment should be an in-process model runtime built on an
existing inference library rather than a new transformer implementation.

## Next experiments

1. Compare Gateway v0.3 and Hybrid Core v0.1 on the same local Qwen model.
2. Measure normal-turn and conflict-turn latency.
3. Measure bootstrap cold-start benefits on a controlled knowledge domain.
4. Test neural/Cortex divergence with synthetic worlds.
5. Add a second `NeuralRuntime` backend.
6. Only then investigate hidden-state/KV-cache access and trainable adapters.

Base-model weight mutation is outside Hybrid Core v0.1.
