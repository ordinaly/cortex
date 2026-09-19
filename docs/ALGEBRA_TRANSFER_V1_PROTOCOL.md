# Algebra transfer v1 — frozen protocol

Status: **prepared / official campaign not yet run**.

Protocol ID: `algebra-transfer-v1`.

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

its mean source structural membership is at least 0.98, and its mean source
predictive membership is at least 0.98.

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

Each compatible target condition uses 10 deterministic seeds.

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

At least one cross-fitted target prediction is required.

The fuzzy combination is conservative:

$$
\mu_{\mathrm{struct}}
=
\min(
\mu_{\mathrm{source}},
\mu_{\mathrm{target\ structural}}
),
$$

$$
\mu_{\mathrm{transfer}}
=
\min(
\mu_{\mathrm{struct}},
\mu_{\mathrm{target\ predictive}}
).
$$

The activation threshold remains 0.98.

No target threshold may be weakened after official results are observed.

## Applicability

The v1.7 applicability rule is retained:

- one- and two-variable laws use target-derived marginal variable support;
- three-variable laws additionally require target-derived pairwise support.

Source applicability sets are discarded during transfer.

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

The official campaign must not be run until the implementation and unit-level
preflight are reviewed against this frozen protocol.
