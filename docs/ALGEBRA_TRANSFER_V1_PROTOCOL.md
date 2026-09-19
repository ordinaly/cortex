# Algebra transfer v1 — frozen protocol

Status: **prepared / official campaign not yet run**.

Protocol ID: `algebra-transfer-v1.6`.

This experiment follows the passing
[`algebra-form-induction-v1.7`](ALGEBRA_FORM_INDUCTION_V1.md) milestone.

## Research question

Does a law form discovered in one collection of anonymous finite algebras become
a reusable abstraction that improves reasoning in a new independently relabeled
algebra with less target evidence than rediscovery from scratch?

The experiment separates:

$$
\text{law discovery}
$$

from

$$
\text{law transfer + target re-grounding}.
$$

A successful transfer must improve target sample efficiency without importing
source element identities, source applicability scopes, or source table cells.

## Transfer object

The only transferable object is a canonical equation form together with
source-world evidence metadata:

- canonical equation syntax;
- fraction of source worlds that selected it;
- mean source structural membership;
- mean source predictive membership;
- expression complexity.

The transfer object must contain **no source element labels** and **no source
applicability basin**.

Target applicability is always reconstructed from target observations.

## Source-world construction

Four independent source families are used.

| transfer library | source orders |
|---|---|
| cyclic | C5 and C7 |
| dihedral | D3 and D5 |
| min semilattice | orders 5 and 6 |
| left-zero semigroup | orders 5 and 6 |

For every source order:

- 4 deterministic seeds;
- 40% of the table held out;
- independently randomized element labels;
- source labels namespaced so they cannot overlap target labels;
- source law discovery uses the frozen v1.7 form inducer unchanged.

Each family therefore contributes 8 source worlds.

A form enters a transfer library only when:

$$
r_{\mathrm{source}} \ge 0.75,
$$

and its mean source structural membership is at least 0.98.

Source predictive membership is recorded for diagnostics but is **not** a
transfer requirement. v1.7 deliberately separated structural discovery from
predictive applicability; requiring source predictive activation would
incorrectly discard a structurally stable form merely because a sparse source
world offered too few opportunities to use it predictively.

Libraries are complexity-ranked and capped at 24 forms.

## Target worlds

Compatible target orders are deliberately different from every source order.

| transfer library | compatible target |
|---|---|
| cyclic | C8 |
| dihedral | D4 |
| min semilattice | order 7 |
| left-zero semigroup | order 7 |

Target labels are independently randomized and use a target-only namespace.

The official campaign will evaluate target observation fractions:

$$
0.20,\quad 0.30,\quad 0.40.
$$

The remainder of each target table is held out for evaluation.

Each compatible target condition uses 10 deterministic seeds. The v1.6
official campaign uses the fresh target seed range **100–109**. Seeds 0–9 were
exposed by the failed v1.5 official campaign and are retained only as
development evidence.

## Target re-grounding

A transferred form receives no automatic right to fire.

For each target table Cortex recomputes:

1. target structural consistency;
2. target cross-fitted predictive evidence;
3. a new target applicability basin;
4. joint law-set consistency.

The source prior lowers only the **target evidence mass needed to consider the
same form again**. It does not lower the truth/consistency threshold.

Target evidence minima are frozen as:

| form arity | target structural witnesses |
|---|---:|
| 1 variable | 2 |
| 2 variables | 2 |
| 3 variables | 4 |

Target structural membership for each imported form is

$$
\mu_{\mathrm{struct}}
=
\min(
\mu_{\mathrm{source}},
\mu_{\mathrm{target\ structural}}
).
$$

Predictive activation is evaluated at the **joint law-set level** rather than
requiring every form to predict a held-aside target cell in isolation. This is
necessary because algebraic laws can become predictive only after composition
with other supported laws.

The bounded candidate bundle is **leave-one-cell-out cross-fitted** against
observed target cells. Each validation cell is predicted from all other
observed target cells, preserving strict out-of-sample evaluation while
retaining maximal training evidence in sparse targets.

Target law-set search is conservative and forward:

1. accept the best individually target-validated transferred form when one
   exists;
2. otherwise search all transferred **pairs** for a safe compositional seed;
3. add another form only if the enlarged bundle produces strictly more correct
   leave-one-out target predictions with zero errors and zero conflicts.

Any accepted seed or bundle must produce at least two cross-fitted predictions
and membership at least 0.98. If no safe seed exists, Cortex remains
unresolved.

The activation threshold remains 0.98.

### Multiplicity-aware predictive evidence

The failed v1.5 official campaign showed that a zero-error streak of only two
or three predictions is not sufficient after adaptively searching many
transferred forms.

For a random operation table of order $n$, a fixed deterministic prediction is
correct with probability $1/n$. If $H(M)$ candidate bundles are inspected from
a transfer library of size $M$, Cortex uses the conservative search bound

$
H(M)
=
M
+
\binom{M}{2}
+
M^2.
$

The minimum number $k$ of zero-error target cross-fit predictions is the
smallest integer satisfying

$
H(M)n^{-k}
\le
0.01.
$

Thus the family-wise chance probability under the uniform-random null is at
most 1% by a Bonferroni bound.

This criterion is computed from library size and target order. It is not tuned
per algebra family or seed.

No target threshold may be weakened after official results are observed.

## Applicability

Source applicability sets are always discarded.

Transfer introduces a **cross-world universality prior**. A canonical form that
is structurally stable across several independent source worlds and also passes
the target structural gate may be *hypothesized* to quantify over all target
elements.

This does not make the law automatically true on the target. Universal
applicability is operational only if the target-only leave-one-out predictive
validator accepts the law or law bundle with zero errors and zero closure
conflicts.

Thus transfer carries:

$$
\text{confidence in a universal form}
$$

but never:

$$
\text{source entity-specific applicability}.
$$

This is deliberately stronger than rebuilding the v1.7 empirical basin from
sparse target witnesses and is the mechanism by which cross-world abstraction
can improve sample efficiency.

## Baselines

Every compatible target is evaluated against:

1. **transfer Cortex** — searches only the transferred form library and
   re-grounds each form locally;
2. **scratch Cortex** — frozen v1.7 law-form induction over the full 375-form
   grammar using only target evidence;
3. **direct memory** — exact lookup of observed target cells only.

The scratch baseline receives exactly the same target cells as transfer Cortex.

## Incompatible-transfer controls

Two negative-transfer controls are frozen.

### Random magmas

Every one of the four source libraries is transferred to independently
generated random order-6 magmas at 30% target observation.

The required safety behavior is abstention rather than forced structure.

### Subtraction control

The cyclic source library is transferred to subtraction mod 7 at 30% target
observation.

The canonical three-variable rebracketing form must be rejected by target
evidence even if it has strong source support.

Other genuinely compatible forms are not prohibited.

## Metrics

For every target condition record:

- held-out accuracy;
- coverage;
- resolved accuracy;
- number of wrong resolved predictions;
- transferred library size;
- transferred laws structurally supported on target;
- transferred laws selected for target closure;
- scratch supported/selected law counts;
- coverage gain over scratch;
- source/target label overlap;
- source and target algebra orders;
- closure conflicts.

Campaign aggregates also report:

- coverage gain by target evidence fraction;
- mean candidate-count reduction versus the 375-form scratch grammar;
- compatible-target transfer AUC over the three evidence fractions;
- negative-transfer resolution rate.

## Frozen gates

The official campaign passes only if all of the following hold.

1. **No identity leakage**
   - maximum source/target label overlap = 0;
   - no compatible target order appears among its source orders.

2. **Conservative correctness**
   - zero wrong resolved transfer predictions across all compatible conditions;
   - zero transfer closure conflicts across all compatible conditions.

3. **Low-evidence transfer gain**
   - at 20% target observation, mean transfer coverage exceeds scratch coverage
     by at least **0.10** across the four compatible families;
   - mean transfer coverage at 20% is at least **0.20**.

4. **Mid-evidence transfer gain**
   - at 30% target observation, mean transfer coverage exceeds scratch by at
     least **0.05**.

5. **Compact reusable abstraction**
   - every transfer library contains at most 24 forms;
   - transfer therefore evaluates at most 24 imported forms versus the
     375-form scratch grammar.

6. **Random-magma abstention**
   - zero wrong resolved predictions;
   - zero resolved predictions across all random-magma negative-transfer
     cases.

7. **Subtraction rejection**
   - zero wrong resolved predictions;
   - the imported canonical rebracketing form is selected on at most 5% of
     subtraction target seeds.

8. **Direct-memory control**
   - direct memory has zero held-out coverage.

Failed gates remain part of the record. They must not be silently weakened
after the official campaign.

## Scientific interpretation

A passing result would support the narrow claim:

> A law form discovered from several anonymous source algebras can be retained
> as an abstract reusable object, re-grounded in a new independently relabeled
> algebra, and improve target sample efficiency without forcing the same law
> onto incompatible worlds.

It would **not** establish universal abstraction, theorem proving, or general
transfer learning.

## Planned implementation

- `research/algebra_transfer.py`
- `research/algebra_transfer_campaign.py`
- `research/algebra_transfer_gate.py`
- `tests/test_algebra_transfer.py`

### Preflight revision record

The initial `algebra-transfer-v1` preflight required source predictive
membership as well as structural membership. This excluded structurally stable
forms, including the canonical rebracketing form in the small cyclic preflight,
because source predictive opportunities were sparse. No official campaign had
been run.

`algebra-transfer-v1.1` corrected that mismatch: **form confidence transfers;
source applicability does not**. A second preflight then showed that requiring
every imported form to make a cross-fitted prediction *individually* was still
too strict at sparse target evidence. Useful laws can be predictive only after
they compose with other supported laws.

`algebra-transfer-v1.2` therefore keeps per-form structural re-grounding but
moves predictive acceptance to a conservative **joint target law-set
cross-fit**. The accepted bundle must make target predictions with zero errors
and zero conflicts. No held-out evaluation cells are used during selection.

A third preflight showed that four-fold target validation removed too much
evidence from an already sparse target table: structurally supported cyclic
forms were present, but the joint validator produced exactly zero predictions.
No official campaign had been run.

`algebra-transfer-v1.3` therefore uses leave-one-cell-out target validation.
Every validation remains strictly out-of-sample, but each fold preserves all
other target observations.

A fourth preflight showed that leave-one-out evidence alone was insufficient
because backward pruning could discard a useful simple transferred law before
it demonstrated predictive value. No official campaign had been run.

`algebra-transfer-v1.4` therefore replaces backward pruning with conservative
forward target validation: safe singles first, safe compositional pairs second,
then only strictly prediction-improving additions.

A fifth preflight showed that safe forward search still abstained because
v1.7-style empirical target scopes required local witness coverage for each
element role. That is appropriate for one-world discovery but prevents a
cross-world universal form from acting as a transferable abstraction.

`algebra-transfer-v1.5` therefore transfers a **universality hypothesis**, not
a source applicability set. The hypothesis is still gated by target structural
evidence and must survive target-only leave-one-out prediction with zero errors
and zero conflicts.

### v1.5 official negative result

The v1.5 official campaign was run on target seeds 0–9 and is preserved as a
failed result.

Compatible transfer was strong:

- at 20% target evidence: transfer coverage **0.5141** versus scratch
  **0.1686**, a gain of **+0.3455**;
- at 30%: **0.7900** versus **0.2247**, a gain of **+0.5653**;
- at 40%: **0.8453** versus **0.3341**, a gain of **+0.5112**;
- zero wrong resolved predictions and zero closure conflicts on all 120
  compatible cases.

However, random-magma controls failed:

- 7 of 40 random-control cases produced at least one resolved prediction;
- 38 random held-out cells were resolved;
- 22 of those resolved predictions were wrong.

The failed gates were
`random_controls_zero_wrong` and `random_controls_abstain`.

This exposed a multiple-hypothesis effect: adaptively searching up to 24
transferred forms makes a short zero-error streak possible by chance.

`algebra-transfer-v1.6` adds the multiplicity-aware evidence bound above and
moves all official target/control evaluation to the untouched seed range
100–109. The v1.5 seeds remain development data and are not reused for the
v1.6 official claim.

The official v1.6 campaign must not be run until its implementation and
unit-level preflight are green.
