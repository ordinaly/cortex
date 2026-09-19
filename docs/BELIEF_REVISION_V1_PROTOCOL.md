# Cortex Reasoning Map v1 — Belief Revision Under Contradiction

Status: **frozen protocol / official campaign passed**.

Protocol ID: \`belief-revision-v1\`.

This is the second experiment in **Cortex Reasoning Map v1** after the passing
structural counterfactual benchmark.

## Research question

Can Cortex revise a strongly supported structural belief when new evidence
contradicts it, without either:

- clinging to the old belief indefinitely; or
- overreacting to one contradictory batch?

The desired trajectory is:

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

The benchmark also asks whether unrelated structure remains exactly unchanged.

## Scientific boundary

v1 tests revision of a single intervention-sensitive causal belief under
controlled deterministic evidence.

It does not yet test:

- stochastic noisy evidence;
- hidden confounders;
- simultaneous revision of many interacting beliefs;
- natural-language contradiction;
- long-horizon memory interference.

Those belong in later Reasoning Map campaigns.

## Native semantic alignment

The revised belief is a Cortex-style causal evidence cell using the same
thresholds as the stable graph contract and
\`counterfactual-reasoning-v1\`:

- minimum intervention count: 8;
- minimum control count: 8;
- causal when $\Delta\ge0.24$;
- resolved-null when $|\Delta|\le0.10$;
- unresolved otherwise.

The evidence accumulator is cumulative. No decay, sliding window, or
post-hoc reset is added in v1.

This makes revision inertia measurable rather than hidden.

## World structure

Every world has 8 anonymously relabeled nodes.

The distinguished causal chain is:

$$
A\rightarrow B\rightarrow C.
$$

The revised belief is the edge:

$$
A\rightarrow B.
$$

The downstream edge $B\rightarrow C$ remains stable throughout the campaign.

Two additional stable causal edges are included as unrelated structure:

$$
D\rightarrow E,
\qquad
F\rightarrow G.
$$

All other ordered pairs are initialized as resolved null.

This means the query

$$
A\leadsto C
$$

tracks the reasoning consequence of revising only $A\rightarrow B$.

## Initial confidence levels

The benchmark uses three prior-strength conditions.

One **causal support batch** contains:

$$
n_{do}^{+}=18,\quad n_{do}^{-}=2,
$$

$$
n_{ctrl}^{+}=2,\quad n_{ctrl}^{-}=18.
$$

The target belief receives:

- **prior-1:** 1 causal batch;
- **prior-2:** 2 causal batches;
- **prior-3:** 3 causal batches.

The target must be active after initialization in every condition.

Stronger priors should require more contradictory evidence to retract.

## Strong contradiction phase

One strong contradictory batch is a null-effect batch:

$$
n_{do}^{+}=2,\quad n_{do}^{-}=18,
$$

$$
n_{ctrl}^{+}=2,\quad n_{ctrl}^{-}=18.
$$

Batches are added sequentially to the same cumulative target cell.

For each prior strength, record:

- first batch where the target leaves active state;
- first batch where it becomes resolved null;
- the full effect-margin trajectory;
- the downstream query state $A\leadsto C$.

The phase stops after retraction or 40 contradiction batches.

## Single-batch stability test

A separate fresh copy receives exactly **one** strong contradictory batch.

Cortex should not retract the previously active belief immediately.

This tests resistance to one-off contradiction.

A memoryless latest-batch baseline will retract immediately and is therefore
used as an overreaction control.

## Ambiguous contradiction phase

A separate fresh copy receives 20 weaker contradictory batches:

$$
n_{do}^{+}=4,\quad n_{do}^{-}=16,
$$

$$
n_{ctrl}^{+}=2,\quad n_{ctrl}^{-}=18.
$$

This evidence weakens the original causal effect but does not itself establish
a clean null regime.

After 20 batches the target should be **unresolved**, not confidently null.

This tests whether Cortex can preserve ambiguity rather than force a binary
decision.

## Recovery phase

After the strong contradiction phase reaches resolved null, causal support
batches are reintroduced one at a time.

Record:

- first batch where the belief leaves null;
- first batch where it becomes causal again;
- effect-margin trajectory;
- downstream query $A\leadsto C$.

The phase stops after recovery or 20 causal batches.

## Downstream epistemic consequence

The target belief controls the only possible route from $A$ to $C$.

Therefore the expected query progression is:

### Initial active belief

$$
A\leadsto C
=
\text{affected}.
$$

### Target unresolved

$$
A\leadsto C
=
\text{unresolved}.
$$

### Target retracted / null

$$
A\leadsto C
=
\text{unaffected}.
$$

### Target recovered

$$
A\leadsto C
=
\text{affected}.
$$

This links local belief revision to reasoning consequences rather than scoring
the evidence cell in isolation.

## Locality requirement

Only the target cell $A\rightarrow B$ may change during contradiction and
recovery.

A deterministic fingerprint of all other causal evidence cells is captured
before the revision sequence and compared after every phase.

The unrelated structure must remain bitwise identical.

## Prior-strength ordering

Cumulative evidence should create rational inertia.

The frozen qualitative hypotheses are:

$$
L_{retract}(prior\text{-}1)
<
L_{retract}(prior\text{-}2)
<
L_{retract}(prior\text{-}3),
$$

and

$$
L_{recover}(prior\text{-}1)
<
L_{recover}(prior\text{-}2)
<
L_{recover}(prior\text{-}3).
$$

The benchmark does not hard-code the exact expected batch counts into the
official gate; it tests the ordering plus maximum latency bounds.

## Margin monotonicity

During strong contradiction, the causal effect margin $\Delta$ should be
monotonically non-increasing.

During recovery, it should be monotonically non-decreasing.

This verifies that confidence evolution is consistent with the accumulated
evidence rather than state-threshold artifacts alone.

## Baselines

### Sticky baseline

Once a belief becomes active, it never revises.

Expected behavior:

- no premature retraction;
- complete failure to adapt.

### Latest-batch baseline

Uses only the most recent evidence batch.

Expected behavior:

- immediate retraction after one contradictory batch;
- immediate recovery after one causal batch.

This baseline adapts quickly but discards accumulated confidence.

Cortex should occupy the middle ground: inertia proportional to prior evidence,
but eventual revision.

## Official seeds

Development/preflight uses seeds below 100.

The untouched official campaign uses:

$$
200,\ldots,219.
$$

There are:

$$
20\text{ seeds}\times 3\text{ prior strengths}=60
$$

official revision cases.

Seed variation permutes opaque node labels and stable-context placement but
does not change the frozen evidence counts.

## Metrics

Per case:

- initial active state;
- single-batch retention;
- first unresolved contradiction batch;
- first null/retraction batch;
- retraction success;
- recovery first-unresolved batch;
- recovery active batch;
- recovery success;
- strong-contradiction margin monotonicity;
- recovery-margin monotonicity;
- ambiguous-contradiction final state;
- downstream query state at each revision stage;
- unrelated-state fingerprint preservation;
- sticky-baseline retraction success;
- latest-batch premature-retraction indicator.

Campaign-level metrics:

- initial active rate;
- single-batch retention rate;
- retraction success rate;
- maximum and mean retraction latency;
- recovery success rate;
- maximum and mean recovery latency;
- transition-order validity;
- prior-strength latency-order validity;
- ambiguous-unresolved rate;
- ambiguous false-retraction rate;
- downstream state-alignment rate;
- monotone-margin rate;
- unrelated-structure preservation rate.

## Frozen gates

The official campaign passes only if all of the following hold.

### Establishment and stability

1. initial active rate = **1.0**;
2. single-batch retention rate = **1.0**.

### Revision

3. retraction success rate = **1.0**;
4. every case enters unresolved before resolved null;
5. maximum retraction latency <= **22 batches**;
6. retraction latency is strictly ordered by prior strength on every seed.

### Ambiguity

7. after 20 ambiguous contradiction batches, unresolved rate = **1.0**;
8. ambiguous false-retraction rate = **0.0**.

### Recovery

9. recovery success rate = **1.0**;
10. every case enters unresolved before returning active;
11. maximum recovery latency <= **8 batches**;
12. recovery latency is strictly ordered by prior strength on every seed.

### Reasoning consequence

13. downstream state-alignment rate = **1.0** across initial, unresolved,
    retracted, and recovered checkpoints.

### Confidence trajectory

14. strong-contradiction margin monotonicity = **1.0**;
15. recovery-margin monotonicity = **1.0**.

### Locality

16. unrelated-structure preservation rate = **1.0**.

### Baseline discrimination

17. sticky baseline retraction success = **0.0**;
18. latest-batch premature-retraction rate = **1.0**.

No threshold may be weakened after official campaign results are observed.

## Negative-result policy

If preflight reveals a mismatch between implementation and this protocol, the
protocol version must advance before any official campaign.

If the official campaign fails, that result remains part of the research
record. Any revised official claim must use fresh untouched seeds.

## Planned files

- \`research/belief_revision.py\`
- \`research/belief_revision_campaign.py\`
- \`research/belief_revision_gate.py\`
- \`tests/test_belief_revision.py\`

The official campaign must not run until preflight is green.

## Intended claim boundary

A passing result would support:

> Within the declared cumulative-evidence fixtures, Cortex can revise a strongly
> supported causal belief through an unresolved state, retract it under
> sustained contradiction, recover it when the regime reverses, propagate the
> epistemic change to downstream reasoning, and preserve unrelated structure.

It would not establish robust belief revision under stochastic or adversarial
real-world evidence.


## Official outcome

The untouched official campaign on seeds 200–219 passed every frozen gate
across 60 cases.

See [BELIEF_REVISION_V1.md](BELIEF_REVISION_V1.md) and
[\`benchmarks/results/belief_revision_v1_summary.json\`](../benchmarks/results/belief_revision_v1_summary.json).

The frozen thresholds above were not weakened after evaluation.
