# Cortex: Geometry of Possible Futures — Formal Ground v0.1

Date: 21 October 2026  
Project author: Arnold von Bauer-Gauss  
Status: proposed research specification. Not an implemented runtime contract, a quantum theory, or an established route to AGI.

## 1. Foundational statement

Cortex maintains a bounded, revisable geometry of possible future situations. Experience updates the support of those possibilities, resolves observable distinctions, and advances the remaining futures. Persistent inadequacy can trigger revision of the represented possibilities themselves.

“Collapse” means observation-conditioned restriction and reweighting, followed by temporal advance. It does not mean that one observation identifies a unique world, that the system causes an external event, or that a technical simplicial collapse occurs.

The inexhaustibility commitment is operational: no finite representation is certified complete merely because it fits the observations so far. Unknown alternatives remain a standing possibility. This is a design commitment, not a proof of an ontological claim.

## 2. Scope and primitive objects

The exact core uses finite sets. This makes its update semantics executable and avoids unspecified measure-theoretic assumptions.

- A: available actions, including passive observation where appropriate.
- O: observable outcomes. An outcome can include an explicitly modelled missing-observation symbol.
- h_t = (a_0,o_1,...,a_{t-1},o_t): interaction history through time t.
- V_t: finite set of candidate situated hypotheses.
- b_t in Delta(V_t): normalized support over candidates.
- Q_t: finite nonempty family of accessible yes/no probes with a declared finite horizon H.
- W_t: bounded evidence witnesses and provenance.

A candidate v contains a compressed current situation together with assumptions about how it can evolve. It induces an action-conditioned probability law over future observation sequences. Thus vertices represent generators of possible futures, not necessarily single deterministic trajectories.

Candidates must be mutually exclusive alternatives at the latent bookkeeping level. Overlapping verbal explanations are represented as features of joint alternatives: “resilient material” and “soft surface” may both hold. They must not be normalized as though they were mutually exclusive causes.

Each probe q specifies a fixed feasible action policy and a yes/no event on its resulting observation trace. Candidate predictions must come from a coherent generative law; independently invented probe probabilities need not be jointly consistent. The same probe protocol is used when comparing candidates.

## 3. Predictive distinguishability

For a fixed representation version, define

$$
d_t(v,w)=\max_{q\in Q_t}|P(q=1\mid v)-P(q=1\mid w)|.
$$

This is a bounded pseudometric: it is nonnegative, symmetric, zero on the diagonal, and satisfies the triangle inequality. Distinct candidates may have distance zero because the accessible probes do not separate them.

The operational identification is

$$
v\sim_t w \iff d_t(v,w)=0.
$$

This says only “indistinguishable under Q_t.” It does not establish identical hidden states, identical causal structure, or equivalence at arbitrary future horizons. Adding probes can separate previously identified candidates. With fixed candidate laws, enlarging Q_t cannot decrease d_t.

## 4. Topology and support are different objects

At resolution epsilon >= 0, define the abstract simplicial complex

$$
K_t^\epsilon=\{\sigma\subseteq V_t:\ d_t(v,w)\leq\epsilon
\text{ for every }v,w\in\sigma\}.
$$

It contains every face of each simplex. Increasing epsilon gives an inclusion filtration. A simplex means mutual predictive similarity at this resolution. It does not mean that its vertices occur simultaneously, and it is not evidence of a higher-order causal interaction.

The representational state is

$$
\mathcal C_t=(V_t,b_t,J_t,Q_t,\{K_t^\epsilon\}_{\epsilon\geq0},W_t),
$$

where J_t is the evolution/observation kernel defined below.

The vector b_t supplies probability mass; the complexes supply neighbourhood structure. They must not be conflated. Reweighting b_t alone need not change any edge.

An optional display or computation threshold tau gives an active induced subcomplex on {v:b_t(v)>tau}. Threshold pruning is an approximation, not evidence that excluded candidates are impossible. Its excluded probability mass must be reported. With tau=0, exact elimination removes only candidates of zero posterior mass.

## 5. Evolution and observation-driven collapse

During one assimilation step, freeze the model version and its finite candidate space V. Define a normalized joint kernel

$$
J_t(v',o\mid v,a)\geq0,\qquad
\sum_{v',o}J_t(v',o\mid v,a)=1.
$$

It jointly specifies the successor candidate and the next observation. It may factor into a transition law and an observation law, but that factorization is not required.

Before observing o, the predicted successor mass is

$$
\bar b_{t+1}(v')=\sum_{v,o'}b_t(v)J_t(v',o'\mid v,a_t).
$$

After observing o_{t+1}, compute

$$
Z_t=\sum_{v,v'}b_t(v)J_t(v',o_{t+1}\mid v,a_t).
$$

For Z_t>0, the collapse operator is

$$
b_{t+1}^{+}(v')=
\frac{\sum_v b_t(v)J_t(v',o_{t+1}\mid v,a_t)}{Z_t}.
$$

This is an exact finite conditioning rule relative to the assumed kernel. Its normalized output does not certify that the kernel is correct.

### Dynamical probabilities on the topology

Define the action-conditioned transition probabilities

$$
T_t^a(v,v')=\sum_o J_t(v',o\mid v,a).
$$

Each row sums to one. Probability is transported by these directed transitions:

$$
\bar b_{t+1}(v')=\sum_v b_t(v)T_t^{a_t}(v,v').
$$

The directed transition graph and the predictive-similarity complex are distinct structures on the same candidate set. Similarity does not imply reachability, and transitions need not follow similarity edges. Blind diffusion over similarity edges would introduce an unjustified dynamical assumption.

In the useful special case

$$
J_t(v',o\mid v,a)=T_t^a(v,v')L_t(o\mid v',a),
$$

the update separates into transport followed by observational reweighting:

$$
b_{t+1}^{+}(v')=
\frac{L_t(o_{t+1}\mid v',a_t)\bar b_{t+1}(v')}
{\sum_u L_t(o_{t+1}\mid u,a_t)\bar b_{t+1}(u)}.
$$

The general joint kernel must be retained when the observation also depends on the predecessor in a way not encoded by v'.

Probability mass resides on vertices, with the mass of a vertex region R given by the sum over v in R. Simplex masses are not independently normalized: simplices overlap and would double-count alternatives. If continuous positions inside simplices are introduced later, they require a separately defined measure and transition kernel.

There are three coupled timescales: temporal transport of mass; evidence-driven conditioning of mass; and learning-driven revision of kernels, probes, and topology. The first two are specified exactly for a fixed model. The third remains a learning problem to be tested prospectively.

If Z_t=0, return model_failure and preserve the observation for revision. Never divide by zero or arbitrarily select a survivor. A positive but persistently small evidence likelihood can also motivate investigation; it does not by itself establish which model change is appropriate.

An observed event conditions the current episode. It does not erase the general possibility of a different outcome in another episode. An event not observed by Cortex cannot directly condition Cortex's beliefs.

## 6. Advancing the future

For a complete candidate future trajectory, conditioning retains the trajectories compatible with the observed prefix and shifts attention to their remaining suffixes. In the compact kernel representation, v' is the advanced situated hypothesis, from which future probes are evaluated again.

After assimilation, recalculate the predictive profiles for the next horizon and rebuild affected neighbourhoods. Topology may change because future distributions, available probes, or candidates changed. There is no assumed monotone shrinking topology across time, no canonical inclusion between successive complexes, and no guarantee that uncertainty decreases on every step.

Representation revision follows assimilation as a separately logged operation. It may generate candidates, revise transition assumptions, introduce probes, or refine a compressed state. It must not retrospectively replace the predictions that were scored before the observation.

## 7. The minimal collapse example

Consider three equally weighted, mutually exclusive toy situations:

| Candidate | Next outcome | Later diagnostic observation |
|---|---|---|
| A: fragile cup, hard floor | broken | irrelevant here |
| B: resilient cup, hard floor | intact | hard floor |
| C: fragile cup, soft floor | intact | soft floor |

These deterministic rules are a test fixture, not a physical account of all cups.

Observation “intact” changes support from (1/3,1/3,1/3) to (0,1/2,1/2). It resolves the outcome but leaves two explanations.

With only the intact/broken probe, B and C have distance zero. Adding the floor-inspection probe makes their distance one in this fixture. Observing “hard floor” then gives (0,1,0).

This demonstrates three distinct operations: evidence reweighting, enrichment of accessible distinctions, and subsequent disambiguation. With noisy observations, the weights generally remain soft instead of becoming exactly zero or one.

## 8. Adaptive compression and structural novelty

A compression c:V->U pushes forward mass:

$$
b_U(u)=\sum_{v:c(v)=u}b(v).
$$

Small pairwise predictive distance is a candidate criterion for merging, not a transitive equivalence at positive tolerance. Connected components of a threshold graph must not automatically become merged cells: chains can connect endpoints with large disagreement.

For a cell C of positive mass, use the posterior-weighted mixture of its member future laws. Its current probe probabilities are then well-defined. If the cell diameter is at most epsilon, each member's answer probability differs from the mixture by at most epsilon for every current probe. This follows because the mixture lies between the smallest and largest member answer probabilities.

That bound is local to the declared probes and current mixture. It does not certify exact recursive dynamics after compression. Exact recursive aggregation requires additional closure, for example equality of each member's joint probabilities of every successor cell and observation under every action. Otherwise, retain hidden mixture information or explicitly measure approximation error.

Physical compression must include all retained witnesses, metadata, and transition data in its budget. A discarded history cannot be reconstructed by “unfolding.” Reopening uses retained evidence or newly acquired observations.

Novelty handling separates: an unexpected event under an adequate probabilistic model; a missing current-state distinction; an inadequate transition model; and insufficient accessible information. No finite procedure is assumed always able to tell these apart.

An unknown-alternatives flag is not a calibrated probability model. A numerical “other” hypothesis can participate in conditioning only if it has a specified predictive law.

## 9. Evidence-seeking action

The controller can prefer actions that separate live alternatives, subject to task value, cost, and constraints. One optional score is expected information gain about a declared target variable, computed under the current predictive model, plus expected task utility minus acquisition cost.

The target must be specified: current state, model identity, or successor state are not interchangeable. Exploring every unpredictable variable is not useful. The controller must also be permitted to act with unresolved uncertainty.

No quantum amplitudes, interference rule, or physical collapse is assumed. “Superposed” describes concurrent represented alternatives. A genuinely quantum probability model would require a separate specification and comparative evidence.

## 10. Cortex implementation boundary

Implement this first as an opt-in research layer over shared relational memory. A candidate stores references to shared evidence and local differing assumptions, rather than a full copy of the world model. Bounded candidate generation and pruning are explicit approximations.

Required interfaces:

1. predict(action): record candidate-specific next-observation laws before evidence arrives.
2. assimilate(action, observation): execute the normalized update or return model_failure.
3. probe_profile(candidate, probes): expose coherent future answer probabilities.
4. geometry(resolution): derive the neighbourhood complex.
5. revise(witnesses, budget): propose and prospectively validate representation changes.
6. compress(budget): record actual retained cost, lost mass, and approximation obligations.

The simplicial complex is derived from predictive profiles in v0.1. Its algorithmic benefit remains a testable question; a geometric visualization alone does not improve reasoning.

This proposal does not change the frozen Cortex specification or assert that existing research modules already implement these interfaces.

## 11. Acceptance gates

- The deterministic example preserves B/C ambiguity after survival and resolves it only after inspection.
- Noisy outcomes produce normalized soft posteriors; an unlikely outcome is not automatically treated as model failure.
- A zero-likelihood observation returns explicit failure and triggers a revision opportunity.
- Missing observations use their declared observation mechanism; hidden simulator events never leak into the update.
- Exact compression cases agree with an uncompressed finite oracle; approximate cases report predictive error and retained cost.
- A similarity chain cannot silently justify a low-diameter merge.
- Different joint future traces remain distinguishable even when their one-step marginals agree.
- Prediction logs are immutable; evidence used to generate a candidate is not reused as an independent prospective test.
- Recurrence can reactivate retained alternatives; pruning never claims lossless recovery.
- Compare against a single-best-hypothesis system and an equally budgeted non-topological mixture. The latter determines whether geometry earns its cost.

Use new evaluation streams after freezing the design. Record calibration, predictive score, decision quality, ambiguity preservation, adaptation delay, retained bytes, and computation cost. No universal numerical pass thresholds are claimed before the test domain is fixed.

## 12. Mathematical and scientific limits

Normalization for positive Z, the pseudometric properties, face closure, and the stated cell-mixture bound follow directly from the definitions. They do not prove learning convergence, calibration under misspecification, tractable candidate discovery, global stability, or AGI.

Conditioning is not generally a contraction in distinguishability: a rare observation can amplify differences. Likewise, a particular observation can increase entropy. Neither monotonic confidence nor monotonic geometric contraction is an invariant of this specification.

The foundational update is:

$$
\text{possible futures}
\xrightarrow{\text{action and observed evidence}}
\text{conditioned possibilities}
\xrightarrow{\text{temporal advance and justified revision}}
\text{new possible futures}.
$$

Future possibility remains plural until evidence warrants a narrower commitment.
