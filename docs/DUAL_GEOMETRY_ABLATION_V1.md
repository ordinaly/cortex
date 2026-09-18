# Dual Geometry Ablation Campaign v1

Status: **finite synthetic characterization experiment** for Cortex v1.3-R.

## Scientific question

Does the v1.3-R tensor bridge genuinely combine semantic structure with typed
interaction structure, or can either side solve the prospective rare-event
problem on its own?

The campaign is designed so that:

- the evaluated source/target pair has zero direct training history;
- source-only and target-only interaction marginals are deliberately
  uninformative about the rare relation type;
- raw interaction frequency is dominated by common relations;
- semantic family labels are not given to the predictor as discrete labels;
- the tensor receives only semantic feature vectors and interaction-rate
  targets from observed pairs.

The intended discriminating result is:

\[
\text{semantic-only}
\approx
\text{interaction-only}
<
\text{full dual geometry}.
\]

## World construction

There are four hidden semantic families on the source side and four on the
target side.

Each entity receives a noisy semantic vector concentrated around its family
basis vector. The hidden family integer is used only by the synthetic generator
to determine ground truth.

Relations are:

1. \`near\` — frequent;
2. \`contact\` — medium frequency;
3. four rare typed events \(r_0,\dots,r_3\).

For a source family \(a\) and target family \(b\), the rare event type is

\[
\boxed{
r(a,b)=(a+b)\bmod 4.
}
\]

This Latin-square construction matters because every source family and every
target family participates equally often in every rare relation type when
marginalized over its partner. The rare relation is therefore a **pairwise
conjunction**, not a source-only or target-only property.

## Held-out contract

For every ordered semantic-family pair \((a,b)\), one concrete entity pair is
removed entirely from tensor fitting.

The future test event for that pair is the corresponding rare relation
\(r(a,b)\).

No direct rate, count or label for the held-out pair enters model fitting.

## Raw interaction rates

Observed training pairs have approximately:

\[
P(\text{near})\approx0.65,
\]

\[
P(\text{contact})\approx0.20,
\]

\[
P(r(a,b))\approx0.08,
\]

with other rare relations near background.

Small seeded perturbations are added to prevent exact lookup by numerical
symmetry.

Consequently an unfiltered predictor is expected to prefer the common
\`near\` event even when the prospective evaluation asks for the rare event.

## Interaction resolution

The global relation frequency vector is fixed symmetrically for the four rare
types:

\[
[10000,3000,150,150,150,150].
\]

The v1.3-R surprisal resolution therefore gives all rare event types the same
global rarity prior. Fine interaction resolution can suppress common relations,
but it cannot tell which rare event is correct without pairwise structure.

## Ablations

The campaign compares:

### 1. Semantic-only

Uses only semantic proximity of the entity pair. It has no learned mapping from
semantic conjunctions to relation types, so all rare event types receive equal
score.

### 2. Interaction-marginal-only

Builds source and target marginal typed-interaction profiles from training
pairs, then averages them for the held-out pair.

Because of the Latin-square design, those marginals are approximately
uninformative about the correct rare relation.

### 3. Tensor without interaction resolution

Uses the semantic tensor bridge but no rarity filter.

It can learn which rare relation belongs to a semantic pair, but the common
\`near\` event should still dominate raw intensity.

### 4. Tensor + fine interaction resolution

Uses the same bridge with \(\eta\) centered on the rare-relation surprisal.

This is the full v1.3-R prediction path.

### 5. Shuffled-semantic control

Refits the same tensor after permuting semantic vectors across entities while
leaving interaction observations untouched.

If performance remains high, the bridge is not actually using semantic
structure as intended.

## Metrics

For each held-out pair, the model ranks all six relation/event types.

We record:

- top-1 accuracy;
- mean reciprocal rank;
- rare-event margin over the strongest wrong rare relation;
- common-vs-rare resolution reversal.

Results are aggregated across deterministic seeds.

## Pass/failure interpretation

A strong result requires:

1. full tensor + fine resolution substantially above chance;
2. interaction-only near chance for rare-type prediction;
3. semantic-only at chance;
4. tensor without rarity resolution preferring common events;
5. shuffled-semantic control substantially worse than the full system;
6. no direct held-out-pair leakage.

Failure would mean at least one claimed component is unnecessary under this
fixture, or the tensor does not transfer pairwise semantic structure.

## Non-claims

Even a successful result would establish only that the designed dual geometry
can solve this finite compositional forecasting problem.

It would not establish:

- general real-world event prediction;
- causal reasoning;
- optimality of the tensor form;
- superiority to modern temporal knowledge-graph models.

The next external test would need neural/Cortex entities and naturally
occurring typed event streams.


## Results

### Base ablation — 50 seeds

All 16 evaluated pairs per seed were completely absent from tensor fitting.

| Method | Hit@1 | MRR | Mean rare-event margin |
|---|---:|---:|---:|
| Semantic-only | 0.2500 | 0.5208 | 0.0000 |
| Interaction-marginal-only | 0.2538 | 0.5236 | -0.0006 |
| Tensor, no interaction resolution | 0.0000 | 0.3333 | +0.0733 |
| **Full dual geometry** | **1.0000** | **1.0000** | **+0.0733** |
| Shuffled-semantic control | 0.2213 | 0.4975 | -0.0065 |

The tensor without interaction resolution already learns which **rare** relation
belongs to the semantic pair, as shown by its positive rare-event margin.
However, the globally common \`near\` relation still wins the unfiltered
six-way event ranking, giving 0% Hit@1 for the prospective rare event.

At coarse interaction resolution, the full model predicts the common \`near\`
event with 100% accuracy. At fine rarity resolution, it predicts the held-out
rare typed event with 100% accuracy.

No evaluated held-out pair appeared in tensor fitting.

This isolates the role of the two components:

\[
\text{tensor bridge}
\Rightarrow
\text{which rare event fits this semantic conjunction},
\]

while

\[
\text{interaction resolution}
\Rightarrow
\text{which frequency/specificity regime is being queried}.
\]

### Hardness sweep

Because the base task is intentionally clean, a second campaign varied semantic
feature noise and the fraction of observed training pairs. Each cell averages
20 deterministic seeds.

| Semantic noise | Training pairs kept | Fine rare Hit@1 |
|---:|---:|---:|
| 0.025 | 100% | 1.000 |
| 0.025 | 50% | 1.000 |
| 0.025 | 25% | 0.991 |
| 0.025 | 12.5% | 0.919 |
| 0.10 | 100% | 1.000 |
| 0.10 | 50% | 1.000 |
| 0.10 | 25% | 0.994 |
| 0.10 | 12.5% | 0.903 |
| 0.20 | 100% | 1.000 |
| 0.20 | 50% | 1.000 |
| 0.20 | 25% | 0.969 |
| 0.20 | 12.5% | 0.884 |
| 0.35 | 100% | 0.878 |
| 0.35 | 50% | 0.856 |
| 0.35 | 25% | 0.781 |
| 0.35 | 12.5% | 0.659 |

Chance rare-type Hit@1 is 0.25.

The degradation is therefore orderly rather than binary. Reducing pair coverage
hurts, but the stronger boundary in this fixture is semantic corruption: with
high-quality semantic coordinates, the bridge generalizes surprisingly well
from sparse pair coverage; once semantic families overlap substantially,
transfer quality falls even when all training pairs are retained.

This is consistent with the intended architecture:

> the tensor cannot transfer structure that the semantic geometry no longer
> represents reliably.

### What this finite experiment supports

Within the tested compositional world:

1. semantic information alone does not identify the future typed event;
2. interaction marginals alone do not identify it;
3. a semantic-pair tensor learns the rare relation structure;
4. interaction rarity resolution is necessary to surface that rare event over
   common background interactions;
5. destroying semantic alignment destroys transfer;
6. performance degrades as semantic geometry becomes noisy or pair evidence
   becomes sparse.

The result is therefore evidence that the two components carry complementary
information in this fixture.

It is not evidence that this dense tensor is the optimal bridge, or that the
same advantage will survive neural perception and natural real-world event
streams.
