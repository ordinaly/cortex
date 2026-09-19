# Cortex algebra induction v1

Status: **research logical-reasoning characterization / not part of the stable native runtime**.

Protocol: `algebra-induction-v1`.

## Question

Can Cortex move beyond applying a known composition table and derive missing
composition rules from global algebraic consistency?

This experiment deliberately separates that question from language, entity
semantics, and numeric labels. Cortex sees only a **partial binary-operation
table over anonymous symbols**. Forty percent of the table is removed before
law scoring.

The element labels are randomly permuted for every seed, so symbols such as
`E0` or `E4` do not retain semantic meaning across runs.

## Candidate-law induction

The first experiment uses a deliberately small preregistered law library:

1. associativity;
2. commutativity;
3. existence of a unique two-sided identity.

This is **candidate-law induction**, not unrestricted theorem discovery.
Cortex is not told which algebra family generated the observations, but the
space of candidate laws is supplied by the experiment.

A law is activated only when enough directly observed witnesses support it and
its empirical consistency exceeds the frozen confidence threshold.

### Associativity evidence

For observed products, Cortex looks for complete witnesses to

$$
(a\star b)\star c
=
a\star(b\star c).
$$

If enough complete witnesses agree, associativity becomes an admissible
constraint.

### Commutativity evidence

For pairs where both directions are observed, Cortex tests

$$
a\star b
=
b\star a.
$$

### Identity evidence

For each candidate element $e$, Cortex checks directly observed instances of

$$
e\star a=a,
\qquad
a\star e=a.
$$

A two-sided identity is activated only if exactly one candidate has sufficient
consistent evidence.

## Deriving an unseen product

Once associativity is accepted, Cortex can derive a product that was never
observed directly.

If

$$
a\star b=u,
\qquad
b\star c=v,
$$

and one side of

$$
u\star c
=
a\star v
$$

is known while the other is missing, associativity supplies the missing
product.

The closure process repeats until no new entries can be derived.

Commutativity and identity, when supported, provide additional exact closure
rules.

Any conflict between a proposed derivation and an existing table entry is
counted explicitly. No conflicting closure occurred in the official campaign.

## Hidden algebra families

The campaign uses three structurally different operations.

| family | order | associative | commutative | two-sided identity |
|---|---:|---:|---:|---:|
| cyclic group C7 | 7 | yes | yes | yes |
| dihedral group D4 | 8 | yes | no | yes |
| subtraction mod 7 | 7 | no | no | no |

The subtraction operation

$$
a\star b = a-b \pmod 7
$$

is a structured non-associative control. It is included to detect a reasoner
that imposes elegant group-like laws even when the data contradict them.

## Campaign

- 20 deterministic seeds per family;
- 60 runs total;
- 40% of every operation table held out before law scoring;
- element labels independently permuted;
- held-out products never available to direct table memory.

Baselines:

- **direct table memory** — returns only observed products;
- **majority-product baseline** — always predicts the most frequent observed
  result symbol.

## Results

### Held-out product recovery

| family | Cortex accuracy | Cortex coverage | resolved accuracy | direct-memory coverage | majority accuracy |
|---|---:|---:|---:|---:|---:|
| cyclic C7 | **0.985** | **0.985** | **1.000** | 0.000 | 0.060 |
| dihedral D4 | **1.000** | **1.000** | **1.000** | 0.000 | 0.0615 |
| subtraction mod 7 | 0.000 | **0.000** | 0.000 | 0.000 | 0.0625 |

The distinction between **accuracy** and **resolved accuracy** matters.

For cyclic C7, Cortex left a small fraction of held-out products unresolved on
one sparse seed. Every product it did derive was correct:

$$
P(\text{correct}\mid\text{resolved})=1.000.
$$

For the noncommutative dihedral group, Cortex derived every held-out product
correctly.

For the non-associative control, Cortex derived none of the missing products,
which is the desired conservative behavior because no candidate law passed the
evidence gate.

### Law detection

| family | exact law-set detection | associativity active | commutativity active | identity active |
|---|---:|---:|---:|---:|
| cyclic C7 | **1.00** | 1.00 | 1.00 | 1.00 |
| dihedral D4 | **0.95** | 1.00 | 0.00 | 0.95 |
| subtraction mod 7 | **1.00** | 0.00 | 0.00 | 0.00 |

The 95% exact law-set rate on D4 is retained as measured. In one seed, direct
evidence for the identity element was insufficient to pass the frozen gate.
Cortex therefore declined to activate the identity law.

Importantly, associativity was still identified on every D4 seed, and
associative closure alone was sufficient to recover every held-out product.

No threshold was changed after observing this result.

## What this establishes

Within these finite controlled algebras, Cortex can now demonstrate a stronger
capability than the earlier kinship experiment:

$$
\text{partial observations}
\rightarrow
\text{law evidence}
\rightarrow
\text{accepted structural laws}
\rightarrow
\text{previously unseen products}.
$$

The answer is not recovered by exact table lookup: direct memory has zero
held-out coverage.

The non-associative control also shows that the current mechanism does not
blindly apply associativity whenever a table is incomplete.

## What this does not establish

This experiment does **not** show open-ended mathematical discovery.

The candidate law vocabulary is supplied in advance. Cortex chooses which laws
are supported and applies their consequences.

It has not yet invented an equation form such as associativity from an
unrestricted hypothesis language.

The next stronger experiment is therefore:

> search over a bounded grammar of candidate equations and ask Cortex to
> discover which algebraic identities compress and predict the observations,
> without being told in advance that associativity, commutativity, or identity
> are the laws of interest.

That would move from **law selection** toward genuine **law-form discovery**.

## Reproducibility

Implementation:

- `research/algebra_induction.py`
- `research/algebra_induction_campaign.py`
- `research/algebra_induction_gate.py`
- `tests/test_algebra_induction.py`

Committed aggregate result:

- [`benchmarks/results/algebra_induction_v1_summary.json`](../benchmarks/results/algebra_induction_v1_summary.json)
