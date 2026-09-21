# Hybrid Cortex Stage I — frozen neural perception + structural inference

> **Historical research report.** The active Stage-I adapter and hybrid benchmark were removed from the main package path by the Autonomous Core + LPU v0.1 pivot. The original implementation remains recoverable from Git history.

Status: historical / retained for research provenance.

This document specifies the first hybrid neural/Cortex boundary without changing
the frozen Cortex executable specification or the native reasoning semantics.

## Research hypothesis

The Stage-I hypothesis is deliberately narrow:

A frozen neural encoder can provide perceptual separability while Cortex turns
that separability into persistent, bounded, revisable structure over time.

This is not yet a claim that Cortex performs general reasoning over arbitrary
neural representations. The first target is persistent entity/regime inference
from chronological perceptual embeddings.

## Architecture

    raw observation
          |
          v
    frozen neural encoder
          |
          | latent embedding z in R^d
          v
    metric adapter
          |
          | x = sqrt(d/2) * z / ||z||_2
          v
    CortexRuntime
          |
          +--> entity binding / persistent identity
          +--> relation and causal evidence
          +--> bounded memory
          +--> recurrence / adaptation / novelty

The encoder is frozen. There is no Cortex-to-network feedback and no gradient
path through Cortex in Stage I.

## Observation packet

The Python research boundary uses NeuralObservation:

- embedding: required neural latent vector;
- confidence: optional scalar perceptual confidence;
- uncertainty: optional per-dimension uncertainty metadata;
- bbox: optional detector box;
- timestamp: optional stream time;
- source: optional provenance identifier.

Only embedding enters the native Cortex core in Stage I. Other fields are
validated and retained as interface semantics for diagnostics and later
experiments. They must not silently change structural decisions yet.

Ground-truth labels or track IDs are explicitly forbidden from the packet.

## Metric contract

Generic latent coordinates have no meaningful cyclic ordering. Feeding them
directly into the current articulation nuisance group would make coordinate
shifts semantically arbitrary.

Stage I therefore defaults to an explicit fixed identity nuisance group:

    fixed_group = [0]

A fixed group is not inferred from evidence: it is applied from the first
matching decision, uses the normal spawn gate immediately, and disables
subgroup-evidence accumulation. This is distinct from merely supplying one
learnable group candidate, which would otherwise remain in bootstrap mode
because no evidence margin can exist between a single candidate and itself.

An encoder may opt into the native learned nuisance mechanism only when its
coordinate action has an intentional domain meaning.

Raw neural embeddings are L2-normalized and scaled:

    x = sqrt(d / 2) z / ||z||_2

For two embeddings a and b, Cortex's unweighted MSE then satisfies:

    MSE(x_a, x_b) = 1 - cosine(a, b)

This gives the existing Cortex distance thresholds a defined interpretation
instead of depending on arbitrary embedding magnitude.

## Stage-I invariants

1. Frozen perception — encoder parameters do not change during a run.
2. No label leakage — evaluation identity is never part of the input packet.
3. Chronology preserved — observations are consumed in stream order.
4. Native semantics preserved — after metric adaptation, the normal
   CortexRuntime.step path is used unchanged.
5. Hybrid/direct parity — passing adapted vectors directly to the same native
   runtime must reproduce hybrid bindings and state.
6. Explicit uncertainty limitation — confidence/uncertainty metadata is not
   used for structural weighting until a separately specified later stage.
7. No arbitrary nuisance action — generic embeddings use identity-only
   coordinate action.

## Implemented API

Example:

    from cortex import HybridCortexRuntime, NeuralObservation

    hybrid = HybridCortexRuntime(embedding_dim=512)

    read = hybrid.step([
        NeuralObservation(
            embedding=z,
            confidence=0.94,
            bbox=(x, y, w, h),
            timestamp=t,
        )
    ])

A FrozenEncoder protocol and encode_and_step convenience path are also
provided. The library intentionally does not depend on PyTorch; neural
frameworks remain optional research dependencies.

## First reference neural adapter

The historical `benchmark_hybrid_caviar.py` campaign used an ImageNet-pretrained ResNet-18 as
a frozen generic visual encoder. Its classifier head is replaced by identity,
yielding the 512-dimensional penultimate representation. The official weight
preprocessing is used. Manual CAVIAR boxes remain an oracle detector so the
experiment isolates representation + structural inference from detector errors.

The benchmark records exact Torch/Torchvision versions and model-weight
identity in every result row.

This is intentionally a generic encoder, not a person-reidentification model.
Its purpose is to test whether a broadly learned visual representation gives
Cortex enough separability to improve substantially over the hand-designed hue
descriptor.

## Acceptance criteria for Stage I

Stage I is useful if, with Cortex unchanged:

- neural embeddings materially reduce identity impurity, switch rate, and/or
  fragmentation relative to camera-pixels-v1;
- structural state remains bounded and explicit;
- failures remain attributable to representation vs structural inference;
- later baseline tests show that Cortex adds value beyond using the same
  embedding with a simpler online matcher.

The final point is essential. Better neural features alone are not evidence for
Cortex's added value.

## Stage II boundary

Stage I itself remains one-way and frozen.

The first bidirectional research prototype is now specified separately as
[Hybrid semantic grounding v1](HYBRID_SEMANTIC_GROUNDING_V1_PROTOCOL.md).
That Stage-II-A experiment is deliberately **inference-time only**: neural
perception proposes label distributions, Cortex combines them with persistent
structural evidence, and Cortex may return `reinforce`, `reinspect`,
`broaden`, or `hold` feedback.

The semantic-grounding protocol preserves the invariant that concept identity
does not depend on its current label. It does not change encoder weights or the
stable native runtime.

Trainable Cortex-to-network feedback remains deferred to a later Stage II-B.
That later step must separately test stability, representation collapse, and
whether gains truly come from Cortex feedback rather than simply from extra
training.
