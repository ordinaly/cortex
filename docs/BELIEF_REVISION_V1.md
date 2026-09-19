# Cortex Reasoning Map v1 — Belief Revision Under Contradiction v1

Status: **passed official benchmark / research-only**.

Protocol: `belief-revision-v1`.

Official run: GitHub Actions run `35435047833`, job `105876169421`.

## Question

Can Cortex revise a strongly supported structural belief under sustained
contradiction without becoming either permanently stubborn or memoryless?

The target causal belief sits inside the only path

$$
A\rightarrow B\rightarrow C.
$$

The revised edge is

$$
A\rightarrow B.
$$

The benchmark therefore measures both the local evidence state and the
downstream reasoning consequence $A\leadsto C$.

## Evidence semantics

The experiment uses the same cumulative intervention/control evidence
thresholds as the stable Cortex graph contract.

No decay, reset, or sliding window is added.

This matters because it makes revision inertia an emergent consequence of
accumulated evidence rather than a separately tuned hysteresis parameter.

Three initial confidence conditions were tested:

- prior-1: one causal evidence batch;
- prior-2: two causal evidence batches;
- prior-3: three causal evidence batches.

Then sustained null-effect evidence contradicts the causal belief.

## Required state trajectory

The benchmark preregistered the qualitative sequence

$$
\text{active}
\rightarrow
\text{unresolved}
\rightarrow
\text{retracted}
\rightarrow
\text{unresolved}
\rightarrow
\text{recovered}.
$$

One contradictory batch should not erase the belief.

Sustained contradiction must eventually retract it.

When the causal regime returns, the belief must recover.

## Official campaign

Untouched official seeds:

$$
200,\ldots,219.
$$

Three prior strengths per seed produced

$$
20\times3=60
$$

official cases.

Development seeds below 100 were never reused for the official claim.

## Result

Every frozen gate passed.

| metric | result |
|---|---:|
| initial active rate | **1.000** |
| one-batch retention rate | **1.000** |
| retraction success | **1.000** |
| transition-order validity | **1.000** |
| ambiguous-unresolved rate | **1.000** |
| ambiguous false-retraction rate | **0.000** |
| recovery success | **1.000** |
| downstream state alignment | **1.000** |
| contradiction-margin monotonicity | **1.000** |
| recovery-margin monotonicity | **1.000** |
| unrelated-structure preservation | **1.000** |
| sticky-baseline retraction success | **0.000** |
| latest-batch premature-retraction rate | **1.000** |

## Revision latency scales with prior confidence

The most informative result is the latency structure.

| initial causal support | mean retraction latency | mean recovery latency |
|---:|---:|---:|
| 1 batch | **7** contradiction batches | **3** causal batches |
| 2 batches | **14** | **5** |
| 3 batches | **21** | **7** |

For every one of the 20 official seeds:

$$
L_r(1)<L_r(2)<L_r(3)
$$

for retraction, and

$$
L_{rec}(1)<L_{rec}(2)<L_{rec}(3)
$$

for recovery.

The maximum observed retraction latency was 21 batches.

The maximum recovery latency was 7 batches.

This is a useful form of evidence-proportional inertia:

> stronger accumulated support requires more contradictory evidence to erase,
> but it remains revisable.

## Ambiguous contradiction

A separate condition supplied weaker counterevidence for 20 batches.

Every case ended **unresolved**, and none became confidently null.

That behavior matters because it distinguishes uncertainty from retraction:

$$
\text{weakened causal belief}
\not\Rightarrow
\text{confident absence}.
$$

## Downstream reasoning followed the belief state

The only route from $A$ to $C$ passes through the revised edge.

Across all official cases:

- initial active belief → downstream **affected**;
- target unresolved → downstream **unresolved**;
- target retracted → downstream **unaffected**;
- target recovered → downstream **affected**.

So revision was not confined to an internal score. It changed reasoning
consequences in the expected epistemic direction.

## Locality

Only the target edge received contradictory/recovery updates.

A deterministic fingerprint over every other causal evidence cell remained
unchanged in every official case.

Measured unrelated-structure preservation:

$$
1.000.
$$

This supports a narrow locality property: evidence contradicting one structural
belief did not rewrite unrelated causal state.

## Baseline discrimination

The sticky baseline never retracts:

$$
\text{retraction success}=0.
$$

The latest-batch baseline uses only the most recent evidence and therefore
retracts after the first contradictory batch:

$$
\text{premature-retraction rate}=1.
$$

Cortex lies between these extremes:

- one contradiction is insufficient;
- sustained contradiction eventually wins;
- required evidence scales with prior support.

## Interpretation

Within the declared cumulative-evidence fixtures, Cortex demonstrated:

1. resistance to one-off contradiction;
2. eventual evidence-driven retraction;
3. explicit unresolved intermediate state;
4. evidence-proportional revision inertia;
5. recovery after regime reversal;
6. downstream epistemic propagation;
7. exact preservation of unrelated structural evidence.

This behavior comes from the existing cumulative evidence semantics rather than
a new bespoke belief-revision controller.

## Important limitation

The evidence is deterministic.

v1 therefore tests the **revision architecture**, not robustness under noisy,
adversarial, or conflicting real-world evidence.

The next generation should introduce:

- stochastic evidence;
- intermittent contradiction;
- simultaneous belief changes;
- changing evidence reliability;
- memory pressure;
- calibration metrics.

## Claim boundary

A defensible claim is:

> Within controlled cumulative-evidence fixtures, Cortex can revise a strongly
> supported causal belief through an unresolved state, retract it under
> sustained contradiction, recover it after regime reversal, propagate the
> epistemic change to downstream reasoning, and preserve unrelated structure.

Do not generalize this result to arbitrary belief revision or real-world causal
epistemology.

## Reproducibility

Implementation:

- `research/belief_revision.py`
- `research/belief_revision_campaign.py`
- `research/belief_revision_gate.py`
- `tests/test_belief_revision.py`

Frozen protocol:

- [`BELIEF_REVISION_V1_PROTOCOL.md`](BELIEF_REVISION_V1_PROTOCOL.md)

Aggregate result:

- [`benchmarks/results/belief_revision_v1_summary.json`](../benchmarks/results/belief_revision_v1_summary.json)

## Next Reasoning Map axis

The next planned campaign is **structural analogy / isomorphism**.

It should test whether Cortex can recognize the same relational organization
across worlds whose entity labels and relation labels are both independently
replaced, then reuse knowledge only when the correspondence is structurally
valid.
