# Cortex algebraic form induction v1.7

Status: **passed research benchmark / not part of the stable native runtime**.

Protocol: `algebra-form-induction-v1.7`.

## Research question

Can Cortex discover useful algebraic **equation forms** from a bounded grammar
without being given named laws such as associativity, commutativity, or
idempotence?

The inducer receives only:

- anonymous elements;
- a partially observed binary-operation table;
- variables `x`, `y`, `z`;
- one anonymous binary operator;
- a grammar bounded to at most two operator nodes per expression.

The grammar generates 375 canonical equation forms after quotienting by
variable renaming and equation-side exchange.

The inducer implementation contains no named target-law vocabulary.

## Fuzzy law model

For a generated form

$$
E_1(x,y,z)=E_2(x,y,z),
$$

Cortex separates three ideas that proved experimentally distinct.

### Structural membership

Structural membership measures how consistently the equation is supported by
complete observed witnesses.

A form can therefore be strongly articulated even when it cannot safely make a
new prediction for every substitution.

### Predictive activation

A structurally supported form is promoted into the bounded closure set only
when cross-fitted predictions on observed cells remain sufficiently reliable.

This prevents a form from being used merely because it fits observations that
were also used to construct it.

### Applicability basin

v1.7 additionally records where a law is supported.

For one- and two-variable laws, Cortex preserves marginal support for each
variable role. For three-variable laws, it also preserves pairwise support.

This distinction was necessary because sparse evidence in earlier versions
allowed a locally valid form to be extrapolated to unsupported combinations.

The result is deliberately fuzzy:

$$
	ext{law confidence}
\neq
	ext{confidence that the law applies to this substitution}.
$$

## Campaign

The official hosted campaign used:

- 20 deterministic seeds per family;
- six algebra families;
- 120 cases total;
- 40% table holdout;
- independently anonymized element labels;
- 375 generated equation forms.

Families:

- cyclic group C7;
- noncommutative dihedral group D4;
- min semilattice;
- left-zero semigroup;
- subtraction mod 7;
- random order-6 magmas.

## Target forms used only for evaluation

The campaign checks whether the generated grammar independently produces and
supports canonical forms corresponding to:

$$
(x*y)*z=x*(y*z),
$$

$$
x*y=y*x,
$$

$$
x*x=x,
$$

and

$$
x*y=x.
$$

These target names and meanings are not available to the inducer.

## Results

| family | held-out coverage | resolved accuracy | notable form result |
|---|---:|---:|---|
| cyclic C7 | **0.9450** | **1.0000** | rebracketing 1.00, swap 1.00 |
| dihedral D4 | **0.8673** | **1.0000** | rebracketing 1.00, swap 0.00 |
| min semilattice | **0.5893** | **1.0000** | repeat 1.00, rebracketing 1.00, swap 1.00 |
| left-zero semigroup | **0.9786** | **1.0000** | left projection 1.00 |
| subtraction mod 7 | **0.7000** | **1.0000** | rebracketing 0.00 |
| random magmas | **0.0000** | unresolved | no target forms activated |

Across all 120 cases:

- no Cortex-resolved held-out prediction was wrong;
- no closure conflict was recorded;
- every random magma case remained completely unresolved;
- direct table memory had zero held-out coverage.

The fit-only enumerator is a useful negative baseline. It can achieve higher
coverage in some structured families, but without the same conservative
predictive-selection discipline. The progression from earlier protocol
versions was driven specifically by false deductions that this benchmark was
designed to expose.

## Negative-result history

The final protocol was not obtained by weakening gates.

- **v1 preflight:** assumed a non-associative subtraction operation should have
  low recoverable coverage. This was false; other short exact identities can
  still determine the table.
- **v1.1:** per-partition witness minima suppressed true low-arity forms.
- **v1.2:** fixing evidence mass admitted sparse-fit forms that did not
  generalize predictively.
- **v1.3:** a single held-out predictive split was too sparse.
- **v1.4:** four-fold cross-fitting plus joint law-set validation still
  over-generalized locally supported dihedral identities.
- **v1.5:** explicit applicability basins fixed that overreach but temporarily
  conflated discovering a form with applying it.
- **v1.6:** structural membership and predictive activation were separated, but
  one random-magma seed still allowed a three-variable coincidence to recombine
  unsupported marginals.
- **v1.7:** three-variable laws additionally require pairwise applicability
  support. The final 120-case campaign passed every frozen gate.

These failures are part of the scientific record.

## What the result supports

Within the declared finite grammar and fixtures, Cortex can:

1. generate algebraic equation forms without named laws;
2. assign fuzzy structural support to those forms;
3. distinguish structural discovery from predictive applicability;
4. use supported forms to derive held-out operation-table entries;
5. abstain when generated structure is insufficiently justified.

A fair concise claim is:

> Cortex demonstrated bounded algebraic form discovery with fuzzy evidence,
> scoped applicability, conservative closure, and zero false resolved
> predictions in the declared 120-case campaign.

## What it does not support

This is not unrestricted theorem discovery.

The grammar, operator count, variables, expression-depth bound, and evidence
machinery are supplied in advance.

The campaign does not establish:

- general mathematical reasoning;
- open-ended axiom discovery;
- superiority over theorem provers or neural methods;
- scalability to large grammars;
- native runtime support.

## Next experiment: cross-algebra transfer

The next discriminator is whether a discovered form becomes a reusable
abstraction.

The intended test is:

$$
W_1
\xrightarrow{\text{form induction}}
L
$$

followed by transfer to a new independently relabeled world

$$
L + W_2^{\text{sparse}}
\rightarrow
\text{prediction}.
$$

The transferred law must improve sample efficiency on compatible structures
while being rejected or deactivated on incompatible structures.

This separates rediscovery from genuine structural reuse.

## Reproducibility

Implementation:

- `research/algebra_form_induction.py`
- `research/algebra_form_induction_campaign.py`
- `research/algebra_form_induction_gate.py`
- `tests/test_algebra_form_induction.py`

Aggregate result:

- [`benchmarks/results/algebra_form_induction_v1_7_summary.json`](../benchmarks/results/algebra_form_induction_v1_7_summary.json)

Hosted passing run:

- workflow run `35429621226`
- algebra-form-induction job `105861746685`
