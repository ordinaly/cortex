# Cortex v1.5-R — Robust Articulation and Latent Event Discovery

Status: **research phase / not a production release**.

v1.4-R showed that learned semantic forecasting succeeds when identity is stable
and fails when perceptual noise corrupts persistent entity assignments.

v1.5-R therefore attacks two upstream privileges:

1. observations no longer have to commit immediately to permanent identities;
2. interaction/event labels are no longer supplied to the learner.

The two tracks are tested independently before they are combined.

## Track A — Robust articulation

### Hypothesis

A bounded agent should distinguish:

\[
\text{persistent identity}
\quad\neq\quad
\text{single uncertain observation}.
\]

The reference mechanism introduces **provisional identity hypotheses**.

An observation can therefore produce

\[
h_t \in
\{
\text{committed entity},
\text{provisional hypothesis}
\}.
\]

A provisional hypothesis accumulates multiple compatible observations before it
may create a persistent entity.

If its aggregate prototype later matches an existing persistent entity, it is
retrospectively reconciled:

\[
p_k \rightarrow e_j.
\]

Evidence attached to the provisional token can then be reassigned to \(e_j\)
without having contaminated another permanent identity in the meantime.

### Reference mechanism

The research prototype uses:

- cosine distance over normalized embeddings;
- a strict direct-acceptance threshold;
- a looser provisional tracklet threshold;
- coordinate-wise median prototypes;
- minimum support before commitment;
- retrospective reconciliation into existing entities;
- one-to-one Hungarian matching within a frame.

The thresholds are frozen before the campaign.

### Metrics

The robust prototype is compared against current Hybrid Cortex on the exact same
observations using:

- identity purity;
- fragmentation;
- resolved-evidence coverage;
- persistent entity count;
- reconciliation count.

Coverage is intentionally explicit. A method that obtains perfect purity by
refusing to resolve nearly all evidence is not considered successful.

## Track B — Latent event discovery

### Hypothesis

Cortex should eventually infer recurring event types from observed dynamics
rather than receiving symbolic labels such as "contact" or "unlock."

The reference event signature contains only measurable changes:

- relative distance change;
- relative speed;
- source/target velocity changes;
- source/target state changes;
- contact;
- alignment;
- transfer balance;
- persistence.

The hidden generator contains six event families, but their integer labels are
never supplied to the discoverer.

### Reference mechanism

The first auditable reference is an online DP-means-like clusterer:

\[
c_t
=
\arg\min_c
\|x_t-\mu_c\|.
\]

If the nearest prototype is farther than a frozen novelty threshold, a new event
cluster is created.

This is deliberately simpler than a neural event encoder. The goal is to test
whether the event structure is observable at all before adding model capacity.

### Metrics

Because discovered cluster labels are arbitrary, scoring is permutation
invariant:

- cluster purity;
- event fragmentation;
- pairwise F1;
- number of discovered clusters.

Cluster frequencies induce a learned event surprisal:

\[
s_c=-\log P(c).
\]

This lets the v1.3-R interaction-resolution idea survive even though the event
types themselves are latent.

## Failure interpretation

The two prototypes can fail independently.

Robust articulation fails if delayed commitment merely trades fragmentation for
unresolved evidence or false merges.

Latent event discovery fails if trajectory/consequence signatures do not
separate into stable recurring structures without labels.

Neither finite success would prove that these are the final mechanisms. They are
reference implementations used to identify whether the proposed information
flow is viable.

## Next combined gate

If both tracks pass independently, the next campaign feeds discovered event
clusters through identities produced by:

1. current Hybrid Cortex;
2. robust provisional articulation;
3. oracle identity.

The same behavioral-semantic learner and tensor bridge are then asked to predict
a held-out future **latent event cluster**.

That combined experiment tests the complete error chain:

\[
\text{perception}
\to
\text{identity}
\to
\text{latent event discovery}
\to
\text{semantic geometry}
\to
\text{future prediction}.
\]

## Complexity warning

The robust reference stores a bounded recent sample window per identity and uses
dense Hungarian matching in the small synthetic fixture. The latent event
reference performs linear prototype search.

Neither implementation is approved for the production hot path.

A native implementation should be considered only after the research campaign
shows a measurable downstream benefit.


## Campaign results

### Robust articulation

Across eight seeds, the provisional/reconciliation layer preserved near-perfect
purity in regimes where direct hard binding degraded sharply.

At perception noise \(0.25\) with no sparse corruption:

\[
\text{hard Cortex purity}=0.242,
\qquad
\text{robust purity}=0.965,
\]

with resolved-evidence coverage \(0.940\).

At noise \(0.25\) with 10% sparse corruption:

\[
\text{hard purity}=0.244,
\qquad
\text{robust purity}=0.949,
\]

with coverage \(0.867\).

The price is explicit unresolved evidence and, in moderate-noise regimes,
residual fragmentation. The gain is that uncertain observations do not
immediately contaminate committed entity state.

### Latent event discovery

The original label-free DP-means-like learner behaved well at low noise but
over-segmented severely at high event noise.

| Event noise | Baseline clusters | Baseline pair-F1 | Robust clusters | Robust pair-F1 | Robust coverage |
|---:|---:|---:|---:|---:|---:|
| 0.08 | 6.0 | 1.000 | 6.0 | 0.997 | 1.000 |
| 0.18 | 8.33 | 0.951 | 6.0 | 0.995 | 0.985 |
| 0.30 | 163.83 | 0.121 | 8.92 | 0.863 | 0.967 |

At noise \(0.30\), baseline event purity remained deceptively high at \(0.986\)
despite catastrophic fragmentation. Provisional event concepts reduce that
structural explosion while preserving most evidence.

This repeats the central v1.5-R lesson at the event level:

\[
\boxed{
\text{uncertain evidence should not automatically create permanent ontology}.
}
\]

### Combined world-model prediction

Latent event structure was discovered perfectly in the combined fixture
(pair-F1 \(=1\)), allowing identity quality to be isolated.

At perceptual noise \(0.12\):

\[
\text{hard-ID future Hit@1}=0.275,
\]

\[
\text{robust-ID Hit@1}=0.550,
\]

\[
\text{oracle-ID Hit@1}=0.825.
\]

At perceptual noise \(0.25\):

\[
\text{hard-ID Hit@1}=0.163,
\]

\[
\text{robust-ID Hit@1}=0.629,
\]

\[
\text{oracle-ID Hit@1}=0.825.
\]

Thus robust articulation restores a substantial fraction of the downstream
forecasting capacity lost to incorrect evidence ownership. It does not reach
the oracle ceiling, which motivates replacing forced identity commitment with a
fully soft predictive representation in the next research phase.

## v1.6-R motivation

v1.5-R still assumes that provisional uncertainty should eventually resolve
into discrete committed identities and event concepts.

The next research question is more fundamental:

> Does Cortex need discrete persistent ontology internally at all?

v1.6-R will test a fuzzy predictive field in which evidence can remain
distributed across possibilities and hard entity/concept identity is only an
optional downstream projection.
