# Autonomous pattern discovery v0.1

## Contract

`research/pattern_discovery.py` builds on the typed composition layer. It takes
`Episode` records containing source-traceable assumed facts in a single context
and separately observed, withheld outcomes. It enumerates connected subsets of
two to four atoms (three by default), canonicalizes their typed relational shape,
and abstracts shapes recurring in at least two distinct episodes. Repeated copies
inside one episode do not increase support. Roles, equality, distinctness, types,
symbol versions and polarity retain the composition layer's semantics.

No candidate motif is supplied. Primitive identities, role types, episode
boundaries, explicit outcome queries and the choice of admissible assumed facts
are still supplied. Use `from_hyperedge` from `compositional_language.py` for native
relational memory edges: soft support remains evidence, not a truth probability.
An integration test covers native edge conversion, mining, transfer and retraction.

## Proposal, selection and test

The miner pairs connected premise subsets with positive observed outcomes whose
entities all occur in those premises. Recurrence produces empirical rule
hypotheses and named motif definitions. Negative outcomes never propose a positive
rule but do test it. The miner excludes target symbols from their own premises.
Selection conservatively excludes any candidate whose premises use a target
symbol of another candidate, keeping one predictive stratum. General recursive
or multi-step rule discovery is outside this version.

Definitions are available as reusable symbolic vocabulary. Predictive hypotheses
currently use primitive conjunctions directly; this version does not claim that
macro substitution improves computation or physical storage.

Selection uses separate validation episodes. Correct prediction costs zero,
explicit error costs two, and abstention costs one. The baseline abstains on all
withheld queries. A candidate must have at least two correct predictions and no
explicit errors. Greedy selection retains it only when its marginal reduction in
portfolio validation loss exceeds its complexity charge. Complexity counts atom
and argument occurrences, multiplied by 0.1 by default. This is a declared
heuristic regularizer, not a description length in bits. Redundant hypotheses
receive no marginal reward. Ties are deterministic.

A frozen selection exposes `test` for fresh episodes. Discovery/validation/test
ID overlap is rejected, as are outcomes already present in premises. These checks
cannot detect a caller relabeling the same underlying evidence with new IDs.
Independent provenance and sampling are a caller obligation. Validation drives
selection and is not an unbiased test estimate. Repeated test-driven tuning would
also require another untouched test set. There is no confidence guarantee.

## Bounds and failure semantics

Discovery counts every examined subset, including disconnected subsets, and
every candidate outcome pairing. The default limit is 10,000 examinations;
exceeding it raises without returning partial discovery. Exact canonicalization
is bounded to four atoms. The existing per-model grounding and proof budgets
apply during evaluation and fail explicitly. This is finite batch discovery over
memory snapshots, not continuous mining of an unbounded stream.

## Reproducible result

Run `python research/pattern_discovery_demo.py`. The checked-in output is
`research/data/pattern_discovery_v0_1.json`. This deliberately small synthetic
fixture supplies opaque relational identities, renamed entities and disconnected
noise, but no candidate pattern. Four discovery episodes yield one motif and one
rule after 20 examinations. Four separate validation episodes retain the rule;
its syntax charge is 0.9 cost units.

| Frozen test condition | Correct | Wrong | Abstained | Loss |
| --- | ---: | ---: | ---: | ---: |
| Eight unseen entity triples | 8 | 0 | 0 | 0 |
| No-rule ablation | 0 | 0 | 8 | 8 |
| Reversed premise roles | 0 | 0 | 8 | 8 |
| Changed outcome law | 0 | 8 | 0 | 16 |

Explicit negative validation rejects the proposed rule. Increasing the syntax
cost to one per unit also rejects it. The changed-world result is an intentional
failure: a previously selected hypothesis can become false. No automatic drift
response or causal discovery is claimed. These fixtures establish mechanism
behavior, not broad empirical generalization or superiority to another learner.

Next work should evaluate diverse noisy memory streams, independent query
sampling, stronger baselines and revision after regime changes before promotion
into the native controller. Mining and retention of deeper macro compositions
should be evaluated separately against their actual computation/storage cost.
