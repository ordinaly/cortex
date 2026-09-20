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

## Orchestration interface

Hybrid Core now separates three internal contracts:

```text
HybridCoreSession
      |
      +--> OrchestrationPolicy
      |       chooses the per-turn plan
      |
      +--> HybridExchange
      |       typed shared cognitive state
      |
      +--> NeuralSession
              model-local execution/session state
                  |
                  v
              NeuralRuntime
                  |
                  v
              backend/provider
```

### HybridExchange

`HybridExchange` is the cognitive exchange object for one turn. It contains:

- the user signal;
- the explicit orchestration plan;
- normalized PromptIR when parsing is enabled;
- the Cortex contribution;
- the neural proposal;
- Cortex diagnosis;
- revision count;
- an ordered, inspectable event trace.

The exchange is backend-independent.

A text-only backend such as Ollama currently receives a serialized projection
through `TextHybridExchangeAdapter`. The JSON/system-prompt representation is
therefore an adapter implementation, not the cognitive architecture.

A future in-process backend can consume the same exchange through a different
adapter such as soft conditioning, adapter gates, cache state, or activation
interfaces.

In addition, every `NeuralRequest` can carry an out-of-band `control` payload.
Hybrid Core places the live `HybridExchange` object there for answer and
revision stages. The default Ollama session ignores this control channel and
uses the text adapter; a specialized in-process `NeuralSession` can consume
the typed exchange directly without parsing its own prompt serialization.

### NeuralSession

`NeuralSession` sits between Hybrid Core and `NeuralRuntime`.

It owns:

- the active exchange ID;
- per-exchange neural call trace;
- stage labels;
- cache policy requests;
- future adapter selection;
- backend capability validation.

The current implementation delegates execution to `NeuralRuntime`. A future
backend can keep KV caches, adapters, or other model-local state behind the
same session boundary.

`HybridCoreGateway` accepts an injectable neural-session factory, so a backend
can create a specialized session per Cortex conversation while retaining the
same Hybrid Core orchestration and persistence boundary.

### OrchestrationPolicy

Each turn has an explicit `OrchestrationPlan` controlling:

- whether semantic parsing runs;
- whether the turn is observed by Cortex;
- whether Cortex context is injected before neural generation;
- whether a claim sidecar is required;
- revision budget;
- Cortex context limits;
- bootstrap-context limits.

The default `full` policy preserves the validated Hybrid Core behavior.

Controlled ablation policies are also available:

```text
full
neural-direct
no-revision
no-cortex-context
```

Examples:

```bash
# Full two-way hybrid route
cortex-gateway --hybrid-runtime --hybrid-policy full ...

# One neural generation, no PromptIR/Cortex pre-pass
cortex-gateway --hybrid-runtime --hybrid-policy neural-direct ...

# Cortex observes and contributes context, but cannot request a revision
cortex-gateway --hybrid-runtime --hybrid-policy no-revision ...

# Cortex observes and assesses after generation, but its state is withheld
# from pre-generation neural conditioning
cortex-gateway --hybrid-runtime --hybrid-policy no-cortex-context ...
```

These modes are primarily experimental controls. They let the same Hybrid Core
measure the marginal effect of Cortex parsing, pre-generation guidance, and
revision without implementing separate pipelines.

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

## Hybrid Symbiosis v0.1 characterization

A deterministic architecture campaign was added to separate hybrid-control
behavior from stochastic language-model behavior.

Protocol:

```text
hybrid-symbiosis-v0.1
11 scenario families
10 synthetic worlds per family
110 total cases
```

The campaign covers:

- neural-prior passthrough with empty Cortex memory;
- independent Cortex corroboration;
- Cortex context influencing the neural candidate before generation completes;
- explicit polarity conflict and bounded neural revision;
- a stubborn neural conflict that remains exposed after the revision budget;
- neural-bootstrap alignment without circular corroboration;
- neural-bootstrap divergence without forced revision;
- promotion of bootstrap memory after independent user evidence;
- parser failure with neural-answer survival;
- alternative-object divergence;
- cold Cortex versus neural-bootstrapped Cortex.

Observed deterministic rates:

| Metric | Result |
| --- | ---: |
| Neural-prior passthrough | 1.000 |
| Independent support | 1.000 |
| Pre-generation Cortex guidance | 1.000 |
| Conflict revision trigger | 1.000 |
| Conflict revision success | 1.000 |
| Stubborn-conflict boundedness | 1.000 |
| Stubborn-conflict preservation | 1.000 |
| Bootstrap alignment | 1.000 |
| Bootstrap circular-confirmation prevention | 1.000 |
| Bootstrap divergence preserved without forced revision | 1.000 |
| Bootstrap promotion after independent evidence | 1.000 |
| Parser-failure answer survival | 1.000 |
| Bootstrap cold-start resolution | 1.000 |
| Normal two-neural-call path | 1.000 |
| Conflict three-neural-call path | 1.000 |
| Non-empty final answer | 1.000 |
| Alternative-object conflict detection | **0.000** |

The last result is a useful limitation rather than a hidden failure.

Current Cortex relation memory can identify contradiction when the same
subject/predicate/object tuple changes polarity. It deliberately does not infer
that:

```text
S --p--> A
S --p--> B
```

is necessarily contradictory, because many relations legitimately admit
multiple objects and real-world regularities can change.

The next target is therefore **not** a hard predicate ontology such as
`functional(p)=true`. Cortex should learn soft, contextual, revisable
expectations about relation structure from experience: typical cardinality,
persistence, symmetry, temporal scope, and other regularities.

A second object under a historically one-object relation should initially
produce structural surprise/tension. If coherent exceptions persist, Cortex
must be able to revise the learned relation schema rather than reject reality.

The campaign therefore gates the established v0.1 contracts but records
alternative-object divergence as a characterization metric and a target for
future relation-schema plasticity experiments.

The CI workflow is
`.github/workflows/hybrid-symbiosis-v0.1.yml`.

## Next experiments

1. Compare the `neural-direct`, `no-cortex-context`, `no-revision`, and
   `full` orchestration policies on the same live Qwen scenarios.
2. Compare Gateway v0.3 and Hybrid Core v0.1 on the same local Qwen model.
3. Measure normal-turn and conflict-turn latency.
4. Measure bootstrap cold-start benefits on a controlled knowledge domain.
5. Test relation-schema plasticity without hard predicate cardinalities.
6. Add a second, preferably in-process, `NeuralRuntime` backend.
7. Use `NeuralSession` to evaluate KV-cache reuse and adapter activation.
8. Only then investigate trainable adapters or deeper activation interfaces.

Base-model weight mutation is outside Hybrid Core v0.1.
