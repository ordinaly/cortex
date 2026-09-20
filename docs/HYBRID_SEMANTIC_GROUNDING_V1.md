# Hybrid semantic grounding v1 — official result

Status: **official campaign passed**.

Protocol: [HYBRID_SEMANTIC_GROUNDING_V1_PROTOCOL.md](HYBRID_SEMANTIC_GROUNDING_V1_PROTOCOL.md)

Aggregate result:
[`benchmarks/results/hybrid_semantic_grounding_v1_summary.json`](../benchmarks/results/hybrid_semantic_grounding_v1_summary.json)

## Question

Can Cortex use persistent relational structure to adjudicate neural label proposals,
including demoting a high-confidence perceptual label when the global semantic
mapping contradicts it, without inventing certainty in structurally ambiguous
cases?

## Frozen official campaign

The protocol was frozen after development preflight. Official seeds
`300..319` were then evaluated without changing:

- the grounding implementation;
- neural/structural evidence weights;
- acceptance thresholds;
- corruption fixtures;
- semantic families;
- official gates.

The campaign contains **560 rows**.

## Main results

### Clean neural perception

- resolved accuracy: **1.000**;
- coverage: **1.000**;
- false `reinspect` rate: **0.000**.

Cortex did not degrade already-correct perceptual labels.

### One adversarial swapped pair

The neural layer assigns two concepts high-confidence mutually swapped labels.

- neural top-label accuracy: **0.600**;
- Cortex resolved accuracy at 8 structural facts: **0.9933**;
- Cortex coverage: **0.9933**;
- correction rate on neural mistakes: **0.9667**;
- `reinspect` precision: **0.9667**;
- false `reinspect` rate on neural-correct concepts: **0.000**.

### Two adversarial swapped pairs

Four of five neural top labels are wrong while still forming a valid one-to-one
assignment.

- neural top-label accuracy: **0.200**;
- Cortex resolved accuracy at 12 structural facts: **0.960**;
- Cortex coverage: **1.000**;
- effective correct fraction: **0.960**;
- correction rate on neural mistakes: **0.950**;
- false `reinspect` rate on the remaining neural-correct concept: **0.000**.

Because the corrupted neural assignments remain a valid bijection, this
correction cannot be attributed merely to enforcing one-to-one label usage.
The structural relation evidence changes the preferred global correspondence.

### Exact structural ambiguity

The symmetry control contains two concepts that are genuinely interchangeable
under both the structural graph and neural evidence.

- twin unresolved fraction: **1.000**;
- mean twin label support: **0.500**;
- non-twin resolved accuracy: **1.000**.

Cortex therefore did not use its "final word" to force an arbitrary label when
the joint evidence remained symmetric.

## Interpretation

The result demonstrates a useful form of inference-time mutual reinforcement:

1. perception proposes a distribution over semantic labels;
2. Cortex evaluates those proposals against persistent relational structure;
3. agreement produces `reinforce`;
4. a justified global contradiction can produce `reinspect`, including an
   explicit promoted and demoted label;
5. unresolved structural ambiguity remains unresolved.

The strongest design invariant remains:

$$
\boxed{
\text{concept identity}\neq\text{current semantic label}
}
$$

Cortex can revise the name without replacing the concept.

## What this result does not establish

This experiment still supplies the concept identities and the candidate label
vocabulary. It therefore does not establish:

- latent concept discovery;
- open-vocabulary naming;
- synonym or taxonomy induction;
- general natural-language grounding;
- trainable neural feedback;
- calibrated real-world semantic confidence.

The intended integration sequence remains:

$$
\text{latent concept induction}
\rightarrow
\text{semantic grounding}
\rightarrow
\text{noisy real-world calibration}.
$$

## Next development

The next scientific task is to connect this validated grounding mechanism to
concepts that Cortex discovers itself rather than concepts supplied by the
fixture.

Only after that integration should Stage II-B allow Cortex feedback to modify
the neural representation itself.
