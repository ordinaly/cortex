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
