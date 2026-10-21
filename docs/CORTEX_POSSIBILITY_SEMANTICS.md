# Semantics for Cortex possibility geometry

Research note, 21 October 2026. This is a focused design review, not a novelty
claim or a reconstruction of all theories of meaning. The implementations below
are our finite engineering choices; the cited works do not validate Cortex.

## 1. Meaning as an information-state update

Frank Veltman's *Defaults in Update Semantics* (1996) develops meaning through
updates to information states and studies defeasible reasoning. It distinguishes
accepted information from conclusions that can be withdrawn.

Source: [author manuscript](https://staff.fnwi.uva.nl/f.j.m.m.veltman/papers/FVeltman-dius.pdf),
introduction and section 2;
[journal record](https://doi.org/10.1007/BF00248150).

Cortex adaptation: an observation has an operational effect on the represented
possibilities. Survival need not identify the explanation. Preserve uncertainty
rather than promoting the most supported candidate into an established fact.
The filter implements probabilistic conditioning and temporal advance, not
Veltman's linguistic logic or default-rule calculus. Repeated independent noisy
measurements can change beliefs; logical idempotence must not be imposed on them.
A singleton posterior establishes certainty only within the supplied model.

## 2. Meaning grounded in possible experiments

Littman, Sutton and Singh's *Predictive Representations of State* (NIPS 2001)
represents dynamical state using predictions about multi-step action-conditioned
observations. This gives a concrete route from abstract state descriptions to
observable tests.

Source: [conference paper](https://proceedings.neurips.cc/paper/2001/file/1e4d36177d71bbb3558e43af9577d70e-Paper.pdf),
introduction and predictive-state definitions.

Cortex adaptation: a question must declare its experiment and accepted traces.
`Probe` and `probe_profile` implement this grounding. Distinguishability is always
relative to the supplied probe family and horizon. A two-step test can separate
candidates with identical one-step distributions. This reference remains a latent
finite model with derived predictive profiles; it is not a learned minimal PSR.
Candidate labels and assumption strings alone carry no executable semantics.

## 3. Behavioural equivalence before exact compression

Blute, Desharnais, Edalat and Panangaden's *Bisimulation for Labelled Markov
Processes* (LICS 1997) studies probabilistic process equivalence and motivates
behaviour-preserving state abstraction.

Source: [author-hosted paper](https://www.cs.mcgill.ca/~prakash/Pubs/lics97.pdf),
introduction and bisimulation definitions.

Cortex adaptation: before claiming an exact recursive quotient, compare every
member's probability of each successor cell and observation under every action.
`is_exact_partition` checks this finite joint-kernel closure condition directly.
No theorem about general state spaces is imported. Probe equality alone does not
satisfy the check, and the implementation does not yet construct the quotient.
Approximate similarity is useful for proposing merges but is not a certificate.

## What these concepts contribute now

| Concept | Implemented consequence | Important boundary |
|---|---|---|
| Information-state update | Ambiguity-preserving assimilation and explicit failure | No language parser or default calculus |
| Operational/predictive meaning | Coherent finite trace probes | Questions and models are supplied |
| Behavioural equivalence | Exact partition admissibility check | No automatic or approximate compression |

## Next research step

Give a learned alternative a semantic contract: its context of applicability,
shared-memory referents, experiments that could distinguish it, and predictions
recorded before testing. Compare a new conditional distinction with an unchanged
model on fresh observations. Discovery data must not double as independent
validation evidence.

Relational role meaning and compositional semantics remain important: two objects
can play analogous roles without sharing all observable properties. The current
predictive distance is not a universal semantic distance. A future typed profile
should distinguish identity, relational role, action consequences and valuation,
with separate evidence and access conditions. We should test transfer across new
objects and role substitutions before claiming that predictive similarity captures
understanding. No automatic rule should treat unknown or inapplicable questions
as negative answers.
