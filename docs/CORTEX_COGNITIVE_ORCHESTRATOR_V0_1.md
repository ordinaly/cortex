# Cortex Cognitive Orchestrator v0.1

Status: **experimental / draft feature branch**.

The Cognitive Orchestrator turns the neural model and Cortex into a
bidirectional system instead of placing Cortex only after the LLM.

## Core idea

```text
User turn
   |
   v
Cortex structural pre-pass
   |
   +--> session evidence / structural result
   |
   v
Adaptive Neural Interface
   |
   +--> base model or pre-registered adapter/model variant
   |
   v
Neural answer
   |
   v
Cortex candidate check
   |
   +--> corroborated
   +--> neural prior
   +--> conflict
   |
   +-- conflict --> one bounded neural revision
   |
   v
Final answer + Cortex diagnosis + learning proposal
```

The neural model remains responsible for:

- natural language;
- broad pretrained factual knowledge;
- fuzzy semantic interpretation;
- candidate hypotheses;
- final articulation.

Cortex is responsible for:

- explicit session memory;
- structural relations;
- evidence provenance;
- consistency / conflict checks;
- uncertainty;
- revision state;
- learning-control signals.

## Bidirectional cognition

The important change from Gateway v0.3 is that Cortex can influence the neural
answer **before** generation.

The neural model receives a bounded object:

```text
CORTEX_SESSION_CONTEXT
```

containing only provenance-labelled Cortex session information relevant to the
current turn.

The model is instructed to distinguish this session evidence from universal
pretrained knowledge.

After generation, Cortex checks the neural candidate. If an explicit conflict
is found, the orchestrator allows at most one revision cycle by default.

The revision prompt contains:

- the original user request;
- the candidate answer;
- Cortex's conflict status;
- Cortex evidence supporting the conflict.

If the disagreement remains unresolved, it is preserved.

## Adaptive Neural Interface v0.1

The first Adaptive Neural Interface is intentionally conservative.

It supports four learning decisions:

```text
none
context_only
adapter_candidate
blocked
```

### none

No explicit new evidence requires adaptation.

### context_only

Cortex stores explicit user evidence and makes it available as session context,
but does not propose neural adaptation.

### adapter_candidate

Repeated explicit evidence crossed the configured confirmation threshold.

This is **only a proposal** in v0.1.

It does not:

- train an adapter;
- update a LoRA matrix;
- alter an Ollama model;
- alter base-model parameters.

The proposal requires explicit approval/future implementation.

### blocked

Adaptation is blocked when Cortex observes conditions such as:

- neural/Cortex conflict;
- incomplete structural analysis;
- unresolved Cortex state;
- memory-budget pressure.

## Weight-update safety boundary

Cognitive Orchestrator v0.1 has:

```text
allow_weight_updates = false
```

and there is no training API call in the implementation.

Therefore:

```text
base-model weights mutated = false
```

for every v0.1 turn.

This is deliberate.

The purpose of v0.1 is to validate whether Cortex can provide a useful
**plasticity-control signal** before introducing actual online neural training.

## Reversible adapter selection

The orchestrator may select a pre-existing Ollama model/adapter variant for a
route.

Example:

```shell
--adapter-model mixed_structural=cortex-reasoning-adapter:latest
```

This does not create or train that adapter. It only chooses an already
available model variant.

The activation is reversible on the next turn.

## Evidence confirmation

Explicit PromptIR observations are stored in a persistent evidence ledger.

A canonical relation such as:

```text
Alice | works_with | Bob | asserted
```

increments when the same explicit evidence is supplied again.

The default adapter-candidate threshold is:

```text
3 confirmations
```

This count is not a truth score and is not sufficient evidence for automatic
training. It is only a plasticity trigger candidate.

Future versions should distinguish source diversity, trust, tool observations,
and independent corroboration.

## Routing

v0.1 exposes simple deterministic routes:

- `neural_knowledge`;
- `memory_update`;
- `mixed_query`;
- `mixed_structural`.

A configured adapter may be attached to any route.

This routing policy is intentionally simple and inspectable. A learned router
can be evaluated later.

## Persistence

Cognitive Orchestrator sessions use:

```text
cortex-language-session-v4
```

and persist:

- Cortex frames/state;
- neural conversation history;
- explicit-evidence confirmation counts.

They do not persist any modified neural weights because v0.1 does not create
any.

## Alpaca unified backend

Alpaca still sees one external Ollama model:

```text
cortex:latest
```

Recommended setup:

```text
Instance type: Ollama (External)
URL: http://127.0.0.1:11435
Model: cortex:latest
```

Run the backend:

```shell
cortex-gateway \
  --cognitive-orchestrator \
  --chat-model qwen3:8b \
  --parser-model cortex-parser \
  --state-dir ~/.cortex/gateway \
  --trace-stages
```

Optional static adapter route:

```shell
cortex-gateway \
  --cognitive-orchestrator \
  --chat-model qwen3:8b \
  --parser-model cortex-parser \
  --adapter-model mixed_structural=cortex-reasoning-adapter:latest \
  --state-dir ~/.cortex/gateway \
  --trace-stages
```

The Alpaca UI mapping on the parent experimental branch is:

```text
message.content  -> final neural answer
message.thinking -> Cortex diagnosis + orchestration + learning decision
timing metadata  -> measured pipeline duration
```

## Thought-panel information

The Cortex Thought panel can expose:

- diagnosis status;
- reasoning mode;
- route;
- selected neural model;
- adapter active/inactive;
- revision-cycle count;
- Cortex runtime flags;
- learning action;
- evidence-confirmation count;
- whether approval is required;
- base-weight mutation status.

This is a procedural/structured report, not hidden chain-of-thought.

## Validation goals

Before introducing real LoRA training, v0.1 should demonstrate:

1. Cortex evidence changes the neural model's response when appropriate.
2. Neural pretrained knowledge remains available.
3. Cortex can detect and preserve conflict.
4. Revision improves consistency without hiding uncertainty.
5. Adapter selection is reversible.
6. Evidence gating avoids immediate learning from one observation.
7. The unified Alpaca backend remains usable.
8. The orchestration overhead is measurable.
9. No base-model parameter mutation occurs.

## Next stage: Adaptive Neural Interface v0.2

Only after v0.1 validation should we add an actual trainable adapter path.

A conservative v0.2 experiment would use a small LoRA/PEFT adapter and require:

- Cortex evidence threshold;
- provenance checks;
- no unresolved conflict;
- explicit adaptation approval;
- a replay/retention set;
- regression tests before promotion;
- adapter snapshots and rollback;
- base model frozen.

The intended progression is:

```text
Cortex memory
    ->
context conditioning
    ->
static adapter selection
    ->
temporary trainable adapter
    ->
validated adapter consolidation
    ->
(optional, much later) base-weight consolidation
```
