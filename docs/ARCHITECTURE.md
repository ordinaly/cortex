# Cortex architecture

> Canonical architecture guide for the current Cortex repository.
>
> **Software/runtime:** `v1.0.0-rc.2`  
> **Frozen executable specification:** `spec-v0.9.5`  
> **Research frontier:** `v1.13-R+`

Cortex is an experimental architecture for **resource-bounded continual structural reasoning**.

The shortest useful description is:

> Cortex turns a chronological stream of structured observations into a bounded, revisable model of **what persists, what relates, what recurs, what changes, what should split, what can merge again, and what remains unresolved**.

It is not a foundation model and it is not a raw perception model. It can consume vectors produced by a neural encoder, but the stable native runtime is designed to operate **after perception**, where the main problem is no longer "what pixels are present?" but rather:

- Is this the same entity as before?
- Which transformations should count as irrelevant nuisance variation?
- Which relations are becoming reliable?
- Which changes represent recurrence, adaptation, or novelty?
- Is one existing concept hiding multiple predictively different regimes?
- Can two previously separated concepts be safely merged again?
- What should the system do when its finite representation budget is exhausted?

This document explains how the current implementation answers those questions.

---

## 1. The mental model

A useful way to think about Cortex is as four interacting forms of memory:

1. **Entity memory** — persistent identities and nuisance-invariant prototypes.
2. **Relational memory** — sparse evidence about relations and directional causal effects.
3. **Historical memory** — bounded fuzzy prototypes that summarize recurring articulation states.
4. **Continual structural memory** — predictive regimes that may adapt, recur, split, merge, or be declared novel.

A frame enters at the top and becomes progressively more structural.

```mermaid
flowchart TD
    A["Structured observations<br/>feature vectors + relation evidence + outcomes"]
    B["Articulation<br/>Who is who?<br/>Which transform aligns them?<br/>Which features are reliable?"]
    C["Sparse relational graph<br/>Which pairs are related?<br/>Which directed effects are supported?"]
    D["12-D public articulation state<br/>Compact structural summary"]
    E["Fuzzy historical memory<br/>What past states resemble this one?"]
    F["Continual structural reasoner<br/>Maintain / adapt / reactivate / split / merge / novel / unresolved"]

    A --> B
    B --> C
    C --> D
    D --> E
    D --> F

    classDef stable fill:#eef,stroke:#445,stroke-width:1px;
    class A,B,C,D,E,F stable;
```

The most important detail in this diagram is the fork after the 12-dimensional articulation vector:

> **Historical memory and continual reasoning consume the same public articulation state independently.**

The reconstruction produced by fuzzy historical memory is deliberately **not** fed back into the continual core in the current frozen semantics. This prevents approximate memory compression from silently rewriting active structural state.

---

## 2. Stable runtime versus research frontier

Cortex currently has three version coordinates because they serve different purposes.

| Layer | Meaning | Current state |
|---|---|---|
| Software release | The packaged Rust/Python implementation | `v1.0.0-rc.2` |
| Executable specification | The frozen behavioral oracle | `spec-v0.9.5` |
| Research frontier | Experimental mechanisms not yet promoted into the runtime contract | `v1.13-R+` |

This separation is architectural, not administrative.

The stable native runtime implements the current published behavioral contract. New research mechanisms such as fuzzy predictive fields, local adaptive resolution, support-sparse prediction, and kinship relation algebra are evaluated separately until they are mature enough to justify a new executable specification.

```mermaid
flowchart LR
    R["Research prototypes<br/>v1.1-R ... v1.13-R+"]
    G["Validation gates<br/>differential tests<br/>frozen campaigns<br/>negative results retained"]
    S["New executable spec<br/>future spec-vX"]
    N["Native Rust implementation"]
    P["Published runtime"]

    R --> G
    G -->|"accepted semantics"| S
    S --> N
    N --> P

    G -.->|"failed / exploratory"| R
```

The current `spec-v0.9.5` is immutable. Research does not silently mutate it.

---

## 3. One frame through Cortex

The native top-level object is `NativeCortexRuntime` in `cortex-runtime`.

A call to `step(...)` receives:

- a list of detection/observation vectors;
- optional binary relation observations between detections;
- an optional intervention source;
- optional binary target outcomes for causal evidence;
- an optional scalar outcome for the continual predictive core.

The transition is coarse-grained and stateful.

```mermaid
sequenceDiagram
    participant U as Caller
    participant A as cortex-articulation
    participant G as cortex-graph
    participant R as cortex-runtime
    participant M as cortex-memory
    participant C as cortex-core

    U->>A: detections
    A-->>R: persistent entity bindings + shifts + subgroup state

    R->>G: entity IDs + relation observations + intervention/outcomes
    G-->>R: sparse relation/causal updates

    R->>R: build 12-D articulation vector

    par historical memory
        R->>M: articulation vector
        M-->>R: fuzzy memberships + reconstruction + unresolved/compressed
    and continual reasoning
        R->>C: articulation vector + optional scalar outcome
        C-->>R: prediction + structural action/state
    end

    R-->>U: RuntimeRead
```

In pseudocode, the production path is intentionally simple:

```text
articulation = articulation.observe(detections)
bindings     = convert_to_graph_node_ids(articulation.bindings)

graph = graph.observe(
    bindings,
    relation_observations,
    intervention_source,
    target_outcomes,
)

z = build_articulation_vector(articulation_state, graph_summary)

memory_read    = memory.step(z)
continual_read = continual.step(z, optional_outcome)

return all reads
```

The system therefore has one explicit public structural bottleneck:

$$
z_t \in \mathbb{R}^{12}.
$$

Everything below articulation and graph evidence is conditioned on this compact state.

---

# Part I — Articulation

## 4. What "articulation" means in Cortex

Articulation is the process of turning raw feature vectors into persistent, structured entities.

Suppose a frame contains detections

$$
x_1,\ldots,x_k \in \mathbb{R}^d.
$$

Cortex must decide whether each detection corresponds to:

- an existing persistent entity;
- an existing entity under an allowed nuisance transformation;
- or a genuinely new entity.

At the same time it learns which feature dimensions are trustworthy and which nuisance transformations are consistently supported by evidence.

The stable implementation uses **cyclic coordinate shifts** as its native nuisance group. This is deliberately a simple, auditable transformation family. For generic neural embeddings the hybrid adapter normally disables this mechanism and uses the identity transformation only.

---

## 5. Entity prototypes and reliability

Every persistent entity stores running evidence including:

- accumulated aligned feature values;
- observation count;
- most recent raw observation;
- per-feature anomaly evidence;
- a prototype vector;
- a per-feature reliability vector;
- a per-feature discrete noise state.

For entity $e$, the prototype is the running aligned mean:

$$
\mu_e(f)
=
\frac{1}{n_e}
\sum_{t \in e}
x_t^{\text{aligned}}(f).
$$

Feature reliability is derived from anomaly evidence. A feature repeatedly behaving like unexplained noise is downweighted during matching.

The matching distance is a weighted mean squared error:

$$
d_w(x,y)
=
\frac{
\sum_f w_f(x_f-y_f)^2
}{
\sum_f w_f
}.
$$

This matters because identity should not depend equally on dimensions that the system has learned are unstable.

---

## 6. Nuisance transformations

For an existing entity $e$, Cortex evaluates the observation against transformed versions of the prototype:

$$
g\cdot\mu_e,
\qquad
g\in G.
$$

For each candidate transformation,

$$
c_g
=
d_w(x,g\cdot\mu_e).
$$

The best transform determines the hard alignment used for the state update, but Cortex also computes a soft transform-evidence distribution:

$$
q(g)
=
\frac{
\exp(-\beta c_g)
}{
\sum_h \exp(-\beta c_h)
}.
$$

So the architecture distinguishes:

- **the selected alignment** used operationally;
- **the distribution of evidence over alignments** used for nuisance-structure learning.

---

## 7. Global simultaneous binding

A subtle but important constraint is that Cortex does not greedily bind detections one at a time.

It builds a cost matrix for the complete frame and solves a rectangular Hungarian assignment problem.

```mermaid
flowchart LR
    D["Frame detections<br/>x1 ... xk"]
    E["Existing entities<br/>e1 ... en"]
    T["Evaluate allowed transforms<br/>weighted MSE"]
    C["Cost matrix<br/>existing entities + spawn columns"]
    H["Hungarian assignment"]
    B["Injective bindings<br/>one entity per detection within frame"]
    S["Spawn new entity<br/>when matching cost exceeds gate"]

    D --> T
    E --> T
    T --> C
    C --> H
    H --> B
    H --> S
```

Spawn columns are added to the assignment matrix. Therefore "create a new identity" participates in the same optimization as "reuse an existing identity."

The resulting binding must be **injective within the frame**: two simultaneous detections cannot be assigned to the same persistent entity.

This avoids a common tracking failure where locally plausible greedy matches become globally inconsistent.

### 7.1 Entity confidence and the attractor interpretation

It is useful to interpret a persistent entity prototype as an **attractor-like
center** in articulation space. The current native implementation does not,
however, expose a single scalar such as "entity confidence = 0.87."

For an existing entity $e$, the articulation layer already computes a
transformation-minimized matching cost of the form

$$
d_e(x)
=
\min_{g\in G}
d_w\!\left(x,g\cdot\mu_e\right).
$$

The selected entity is determined by the **global assignment problem**, with
new-entity spawn columns competing against existing entities. Therefore the
effective basin of an entity is defined jointly by:

- its prototype $\mu_e$;
- its per-feature reliability weights;
- the currently admissible nuisance transformations;
- competing entity prototypes;
- the current spawn gate;
- simultaneous injectivity constraints from the rest of the frame.

The implementation already retains several uncertainty-related signals:

| signal | current meaning | publicly exposed? |
|---|---|---|
| entity matching cost | geometric fit to an entity under its best transform | not in `ArticulationRead` |
| transform evidence $q(g)$ | uncertainty over nuisance alignment for a chosen entity | internal `BindingResult` |
| per-feature reliability | stability of each feature dimension for an entity | snapshot |
| subgroup evidence margin | confidence that one nuisance subgroup is better supported than the runner-up | yes |
| observation count / history | temporal support accumulated by an entity | internal state |

So Cortex currently has **evidence for confidence**, but no calibrated
entity-identity confidence scalar.

A natural future quantity is an **attractor-separation margin**. If $e^*$ is
the chosen entity and $d_{\mathrm{alt}}$ is the best competing explanation
(other entity or spawn), then

$$
\Delta_e
=
d_{\mathrm{alt}}
-
d_{e^*}.
$$

Large positive $\Delta_e$ means the selected attractor basin is well separated;
a value near zero means the observation lies close to a decision boundary.

Because binding is solved jointly with the Hungarian assignment, the most
rigorous confidence would use **global assignment regret**:

$$
\Delta_{\mathrm{global}}
=
C_{\mathrm{best\ alternative\ assignment}}
-
C_{\mathrm{optimal\ assignment}}.
$$

That quantity respects simultaneous competition between detections, whereas a
simple row-wise margin is only an approximation.

Neither margin should automatically be called a probability. To obtain a
probability such as $P(\text{identity correct}\mid\text{evidence})$, Cortex
would need calibration against held-out identity outcomes.

---

## 8. Learning the active nuisance subgroup

Cortex does not automatically prefer maximal invariance.

It accumulates transformation evidence and compares candidate transformation groups. A candidate is rewarded when its member transformations consistently explain observations, with evidence normalized by group size.

The active subgroup is used only when evidence has become sufficiently decisive. Otherwise Cortex remains conservative and retains the larger candidate set.

The runtime exposes both:

- the selected subgroup;
- a **subgroup evidence margin** representing confidence in that choice.

This is architecturally important:

> invariance itself is learned under evidence rather than treated as a free simplification.

---

## 9. Residual and noise state

After alignment, Cortex computes the residual between the aligned observation and the entity prototype.

Per feature, repeated large residuals accumulate evidence that the dimension behaves like noise. Repeated small residuals accumulate evidence that it reflects stable structure.

The native discrete states are conceptually:

| State | Interpretation |
|---|---|
| structure-supported | residuals are persistently small |
| unresolved | insufficient or ambiguous evidence |
| noise-supported | residuals are persistently anomalous |

The public articulation vector does not expose every feature state separately. Instead it exposes the **fractions** of feature coordinates currently in these categories.

---

# Part II — Sparse relational structure

## 10. Why the graph is sparse

If Cortex tracks $N$ entities, a naïve relation system allocates all

$$
O(N^2)
$$

possible pairs.

The native graph does not.

A relation or causal cell is created only after the stream supplies evidence for that pair.

```mermaid
flowchart TD
    A["Potential entity pair"]
    B{"Has the stream supplied evidence?"}
    C["No graph cell allocated"]
    D["Allocate sparse evidence cell"]
    E["Update sufficient statistics"]
    F["unresolved / active / absent or null"]

    A --> B
    B -->|"no"| C
    B -->|"yes"| D
    D --> E
    E --> F
```

Worst-case storage can still become quadratic if evidence genuinely touches every pair. The important guarantee is narrower:

> Cortex does not pay quadratic storage merely because the pairs are theoretically possible.

---

## 11. Relation evidence

Undirected relation cells maintain Beta-style sufficient statistics.

Conceptually, after binary observations $y \in \{0,1\}$,

$$
a \leftarrow a+y,
\qquad
b \leftarrow b+(1-y).
$$

The posterior mean is

$$
\hat p
=
\frac{a}{a+b}.
$$

After a minimum number of exposures, configured thresholds classify the pair as:

- **active**;
- **absent**;
- or **unresolved**.

Until enough evidence exists, the graph explicitly remains unresolved.

---

## 12. Intervention-sensitive causal evidence

Directional causal cells distinguish observations collected:

- when a source entity was the declared intervention source;
- from matched control observations.

Each directed pair stores separate Beta-style sufficient statistics for intervention and control.

If

$$
p_{\text{do}}
=
\frac{a_{\text{do}}}{a_{\text{do}}+b_{\text{do}}},
\qquad
p_{\text{ctrl}}
=
\frac{a_{\text{ctrl}}}{a_{\text{ctrl}}+b_{\text{ctrl}}},
$$

then evidence is based on

$$
\Delta
=
p_{\text{do}}-p_{\text{ctrl}}.
$$

With enough intervention and control samples:

- sufficiently positive $\Delta$ supports a causal edge;
- sufficiently small $|\Delta|$ supports a null edge;
- intermediate evidence remains unresolved.

This is not a full causal-discovery framework. It is an explicit online evidence contract for intervention-sensitive directional effects.

---

## 13. Local dependency closure

The graph crate also provides a directed dependency graph.

When a node becomes dirty, Cortex can compute the transitive closure of only its dependents:

$$
\operatorname{closure}(S)
=
S
\cup
\{v : \exists u\in S,\; u\leadsto v\}.
$$

This supports a broader architectural principle:

> local change should reopen only the structure that depends on it unless a global witness requires more.

---

# Part III — The 12-dimensional public articulation state

## 14. Why compress the structural state

Articulation and graph state can be large:

- many entity prototypes;
- many feature reliabilities;
- many residual/noise states;
- many sparse relation and causal cells.

The continual reasoner does not consume this complete internal state directly.

Instead, `cortex-runtime` constructs a fixed 12-dimensional public summary:

$$
z_t \in \mathbb{R}^{12}.
$$

This creates a stable contract between "perceptual/relational articulation" and "continual structural reasoning."

The current coordinates are:

| Index | Meaning |
|---:|---|
| 0 | active entity fraction relative to configured entity capacity |
| 1 | normalized nuisance-subgroup evidence margin |
| 2–5 | one-hot indicators for up to four configured/default nuisance groups |
| 6 | fraction of residual features supporting stable structure |
| 7 | fraction of residual features unresolved |
| 8 | fraction of residual features supporting noise |
| 9 | fraction of possible entity pairs with active relations |
| 10 | fraction of possible entity pairs whose relation state is resolved |
| 11 | fraction of possible directed entity pairs with active causal evidence |

The last three denominators are based on the complete active entity population, not merely allocated sparse cells. An unallocated pair therefore has the same unresolved meaning it had in the frozen dense reference.

This is a good example of an implementation optimization that preserves semantics:

$$
\text{sparse storage}

\neq
\text{different meaning}.
$$

---

# Part IV — Fuzzy historical memory

## 15. The role of historical memory

The `cortex-memory` crate implements **Fuzzy Accordion Memory**.

Its purpose is not to choose the active regime. Instead it answers a different question:

> Which previously stored structural states resemble the current one, and can history be compressed without exceeding a declared distortion budget?

For prototypes $c_i$, Cortex computes RMS distances

$$
d_i
=
\operatorname{RMS}(z,c_i)
$$

and fuzzy membership scores

$$
m_i
=
\frac{
\exp(-d_i/\tau)
}{
\sum_j \exp(-d_j/\tau)
}.
$$

An $\alpha$-cut determines which memories are considered actively relevant.

The reconstruction is the membership-weighted prototype mixture:

$$
\hat z
=
\frac{
\sum_i m_i c_i
}{
\sum_i m_i
}.
$$

---

## 16. Memory update policy

Historical memory follows a bounded decision process.

```mermaid
flowchart TD
    A["New articulation vector z"]
    B{"Memory empty?"}
    C["Store first prototype"]
    D{"Nearest prototype within fit tolerance?"}
    E["Update prototype in place"]
    F{"Capacity available?"}
    G["Append new prototype"]
    H{"Certified safe recompression possible?"}
    I["Merge redundant prototypes<br/>then append new one"]
    J{"Existing fuzzy reconstruction<br/>within distortion budget?"}
    K["Compress implicitly<br/>no new structural state"]
    L["UNRESOLVED<br/>do not fabricate capacity"]

    A --> B
    B -->|"yes"| C
    B -->|"no"| D
    D -->|"yes"| E
    D -->|"no"| F
    F -->|"yes"| G
    F -->|"no"| H
    H -->|"yes"| I
    H -->|"no"| J
    J -->|"yes"| K
    J -->|"no"| L
```

The term **certified recompression** means that two stored prototypes may be merged only when:

1. they are sufficiently redundant;
2. their weighted merged prototype stays within the configured distortion budget.

The system therefore does not silently delete information simply because memory is full.

---

## 17. Why memory reconstruction does not drive the continual core

This boundary is intentional.

A fuzzy memory reconstruction is an approximation:

$$
z_t
\mapsto
\hat z_t.
$$

If the continual reasoner consumed $\hat z_t$, then a storage optimization could alter active structural decisions.

The current runtime instead sends the original public articulation vector $z_t$ to both consumers:

```text
                 ┌──> historical memory
articulation z ──┤
                 └──> continual reasoner
```

Thus:

> approximate historical storage cannot silently rewrite the present.

---

# Part V — Continual structural reasoning

## 18. Regime prototypes

The `cortex-core` crate maintains bounded predictive regimes.

Each regime prototype stores, among other fields:

- a stable unique ID;
- a centroid in the 12-D articulation space;
- soft count and utility;
- predictive successes/failures;
- creation and last-use times;
- recent predictive statistics;
- lineage information for splits;
- refinement depth.

Given the current state $z_t$, Cortex computes distances to all prototypes and soft memberships similar to fuzzy memory:

$$
m_i
\propto
\exp(-d_i/\tau).
$$

These memberships contribute to prediction and to decisions about whether the current state belongs to:

- the current regime;
- another known regime;
- or no known regime.

---

## 19. The structural action vocabulary

At a high level, the continual core implements the following decision vocabulary:

```mermaid
stateDiagram-v2
    [*] --> Maintain

    Maintain --> Adapt: nearby drift
    Maintain --> Reactivate: another known regime gains sufficient evidence
    Maintain --> Novel: persistent evidence supports no known regime
    Maintain --> SplitCandidate: local predictive residual shows structured multi-directional variation

    Adapt --> Maintain
    Reactivate --> Maintain
    Novel --> Maintain

    SplitCandidate --> Split: prospective validation shows predictive gain
    SplitCandidate --> Maintain: evidence insufficient / candidate rejected
    Split --> MergeCandidate: descendants later reconverge
    MergeCandidate --> Maintain: merge evidence insufficient
    MergeCandidate --> Merge: sustained predictive + geometric compatibility
    Merge --> Maintain

    Novel --> Unresolved: budget exhausted and safe recompression impossible
    SplitCandidate --> Unresolved: split requires unavailable capacity
    Unresolved --> Maintain: later evidence/capacity permits resolution
```

This diagram is conceptual: the implementation exposes booleans and counters rather than a single explicit enum for every arrow. But the behavioral contract follows this logic.

---

## 20. Maintain and adapt

If the current observation remains sufficiently close to the active regime, Cortex stays in place.

Small local drift updates the existing regime rather than creating a new one.

This is the default bias:

$$
\text{adapt existing structure}
<
\text{create new structure}
$$

unless evidence justifies the latter.

---

## 21. Recurrence and reactivation

If the current regime no longer fits but another stored regime is sufficiently plausible, Cortex accumulates recurrence evidence.

Only when that evidence passes a threshold does the active regime switch.

Thus recurrence is not represented as:

> "This looks different from now, therefore it is new."

Instead:

> "This looks inconsistent with now, but consistent with a previously known state."

A successful switch is recorded as **reactivation**, not discovery.

---

## 22. Novelty and explicit budget pressure

If the observation is far from both:

- the current regime;
- and every known regime,

Cortex accumulates novelty evidence.

When novelty evidence crosses threshold, Cortex attempts to store a new regime.

If capacity exists, or if a safe recompression can free capacity, the regime is created.

Otherwise:

$$
\boxed{
\text{novel evidence}
+
\text{no safe capacity}
\Rightarrow
\text{unresolved + budget pressure}
}
$$

not "pretend the closest existing regime is correct."

This is one of the central Cortex invariants.

---

# Part VI — Conditional predictive refinement

## 23. Why Cortex has a tensor side path

A regime centroid can be a good coarse model while still hiding a structured predictive residual.

For example, two observations may occupy nearly the same structural region but have systematically different outcomes along some local direction.

The continual core therefore maintains a small conditional tensor/residual system per regime.

Its purpose is diagnostic and corrective:

1. detect coherent predictive residual structure;
2. provide a low-amplitude predictive correction;
3. monitor whether residual geometry suggests that the regime itself may need to split.

The tensor machinery is conditional: it can sleep, wake, refresh, and run in bursts. This prevents expensive refinement work from becoming permanently active when it is not useful.

---

## 24. Curvature and rank evidence

For an active regime with centroid $\mu$, define local displacement

$$
\delta_t = z_t-\mu.
$$

With predictive residual

$$
e_t = y_t-\hat y_t,
$$

the core maintains a curvature-like matrix using terms of the form

$$
e_t\,\delta_t\delta_t^\top.
$$

The implementation periodically examines the eigenspectrum of this matrix.

If significant structure persists in more than one direction, the regime becomes a candidate for refinement.

This is not enough to split.

It is only a **proposal mechanism**.

That distinction is fundamental:

> suspicious geometry may propose a structural distinction; only future predictive evidence may promote it.

---

## 25. Prospective split validation

When a split candidate is launched, Cortex creates temporary **shadow directions** rather than immediately modifying active state.

For candidate direction $u$, observations are provisionally separated by the sign of

$$
u^\top(z-\mu).
$$

Cortex then compares:

- the predictive loss of the unsplit parent;
- the predictive loss of the provisional two-branch model.

The candidate moves through two phases:

1. **selection** — compare candidate directions;
2. **validation** — evaluate the best candidate prospectively on later evidence.

A split is promoted only if:

- both branches receive enough evidence;
- mean predictive gain exceeds threshold;
- gain exceeds an explicit complexity penalty;
- the two resulting centroids are not effectively redundant;
- budget permits the new representation.

```mermaid
flowchart LR
    A["Residual curvature<br/>rank witness"]
    B["Candidate directions"]
    C["Shadow branch models<br/>active state unchanged"]
    D["Prospective future evidence"]
    E{"Predictive gain survives<br/>complexity penalty?"}
    F["Reject candidate"]
    G["Promote split"]
    H["Two lineage-linked regimes"]

    A --> B --> C --> D --> E
    E -->|"no"| F
    E -->|"yes"| G --> H
```

This is one of the strongest architectural protections against overfitting.

---

## 26. Merge and reversibility

A structural distinction is not permanent merely because it was once useful.

Split descendants retain lineage information. Cortex periodically asks whether they have reconverged sufficiently in:

- geometry;
- predictive behavior;
- recent evidence.

Only after persistent merge evidence are descendants recombined.

Therefore the structure is **reversible**:

$$
\text{coarse}
\rightarrow
\text{split}
\rightarrow
\text{coarse again}.
$$

This is architecturally different from systems where model complexity only grows.

---

# Part VII — Boundedness and uncertainty

## 26.1 Does `unresolved` have a confidence score?

Not yet as one universal scalar.

Different Cortex layers already expose the evidence from which an
**unresolved strength** could be constructed:

- **Historical memory** returns fuzzy `memberships`, `nearest_dist`, and a
  boolean `unresolved`. Memory becomes unresolved only when capacity is full,
  safe recompression fails, and reconstruction distortion exceeds the declared
  budget.
- **Continual reasoning** returns regime memberships, nearest-regime distance,
  `unresolved`, and `budget_pressure`. Its snapshot also retains
  `pending_score`, `pending_count`, the adaptive hazard, and the current
  pending structural hypothesis.
- **Relations and causal edges** retain Beta-style sufficient statistics and
  discrete unresolved states internally, although the public graph read does
  not currently expose a calibrated posterior confidence.
- **Entity articulation** has no `unresolved` identity state in the stable
  runtime. It binds to an existing entity, spawns a new one, or raises an error
  if a required spawn exceeds `max_entities`.

For fuzzy historical memory, a natural non-probabilistic severity measure would
be the amount by which reconstruction error exceeds the allowed distortion:

$$
u_{\mathrm{mem}}
=
\max\!\left(
0,
\frac{d_{\mathrm{RMS}}(z,\hat z)-\varepsilon_D}
     {\varepsilon_D}
\right),
$$

where $\varepsilon_D$ is the configured distortion budget.

For a pending continual transition, an evidence margin could similarly compare
the accumulated score with the current decision threshold:

$$
u_{\mathrm{struct}}
=
s_{\mathrm{pending}}
-
\theta_t.
$$

These are useful **evidence margins**, but they are not yet part of the frozen
public runtime contract and should not be described as calibrated
probabilities.

A future confidence interface should therefore preserve the distinction between:

1. confidence that a particular entity/regime is the best explanation;
2. confidence in a nuisance transformation;
3. evidence that the current state is genuinely unresolved;
4. confidence that a new structural distinction is needed.

Collapsing all four into one number too early would discard useful structure.

## 27. Bounded state is a design constraint

Several parts of Cortex have explicit budgets:

- maximum active entities;
- historical memory budget;
- continual regime budget;
- refinement depth;
- conditional tensor activity/refresh policy;
- sparse graph allocation only after evidence.

Fixed configuration therefore induces explicit finite bounds on important state components.

But Cortex does not treat "bounded" as "always able to explain everything."

When the system cannot represent new evidence without violating its declared resource or distortion contract, it exposes the failure.

```mermaid
flowchart LR
    E["New evidence"]
    K{"Known structure sufficient?"}
    A["Adapt / reactivate"]
    N{"New structure justified?"}
    B{"Budget / safe recompression available?"}
    S["Create structure"]
    U["UNRESOLVED<br/>budget pressure"]

    E --> K
    K -->|"yes"| A
    K -->|"no"| N
    N -->|"no"| A
    N -->|"yes"| B
    B -->|"yes"| S
    B -->|"no"| U
```

The governing principle is:

$$
\boxed{
\text{resource exhaustion must reduce certainty, not fabricate certainty.}
}
$$

---

# Part VIII — Snapshots, observability and provenance

## 28. Inspectable state

Each major layer exposes a serializable snapshot.

The runtime snapshot contains:

- articulation state;
- sparse graph state;
- fuzzy memory state;
- continual reasoner state;
- the current 12-D articulation vector.

This matters for two reasons.

First, Cortex is intended to be inspectable: structural changes should be visible rather than buried in an opaque hidden state.

Second, the native implementation is validated through deterministic differential replay against the frozen Python specification.

---

## 29. Differential implementation contract

The native Rust implementation is not considered correct because it "looks similar" or because benchmark averages match.

For shared semantics, deterministic fixtures compare:

- numerical predictions;
- entity bindings;
- relation/causal state transitions;
- memory behavior;
- regime decisions;
- split/merge behavior;
- final snapshots.

Floating-point outputs use explicit tolerances. Discrete structural decisions are expected to agree.

```mermaid
flowchart TD
    I["Identical chronological input"]
    P["Frozen Python spec-v0.9.5"]
    R["Native Rust runtime"]
    SP["Python snapshots / reads"]
    SR["Rust snapshots / reads"]
    D{"Differential contract"}
    PASS["Semantics accepted"]
    FAIL["Migration/optimization rejected"]

    I --> P
    I --> R
    P --> SP
    R --> SR
    SP --> D
    SR --> D
    D -->|"within numerical tolerance + same structural decisions"| PASS
    D -->|"otherwise"| FAIL
```

This makes optimization safer: performance work can be rejected if it silently changes reasoning semantics.

---

# Part IX — Hybrid neural/Cortex architecture

## 30. Cortex does not require hand-engineered input features

Stage-I Hybrid Cortex places a frozen neural encoder in front of the unchanged native runtime.

```mermaid
flowchart LR
    X["Raw observation<br/>image / signal / other modality"]
    N["Frozen neural encoder"]
    Z["Embedding z"]
    A["Metric adapter"]
    C["Native Cortex runtime"]
    O["Persistent structure"]

    X --> N --> Z --> A --> C --> O
```

For a $d$-dimensional embedding $z$, the adapter uses

$$
x
=
\sqrt{\frac d2}
\frac{z}{\|z\|_2}.
$$

For two adapted embeddings $x_a,x_b$, ordinary unweighted MSE becomes exactly cosine distance between the original embeddings:

$$
\operatorname{MSE}(x_a,x_b)
=
1-\cos(a,b).
$$

This gives Cortex's existing metric thresholds a defined interpretation instead of allowing arbitrary neural embedding magnitudes to control identity decisions.

Generic embeddings use a fixed identity nuisance group because cyclic shifts of latent coordinates generally have no semantic meaning.

---

## 31. Why the encoder is frozen in Stage I

The one-way boundary is deliberate:

$$
\text{encoder}
\rightarrow
\text{Cortex}.
$$

If both systems learned simultaneously, it would be difficult to determine whether a performance gain came from:

- better neural features;
- Cortex structural reasoning;
- or feedback between them.

Stage I isolates the question:

> Given a fixed representation, does Cortex add useful persistent structural reasoning?

Only after that is established should a Stage II experiment consider

$$
\text{encoder}
\leftrightarrow
\text{Cortex}.
$$

---

# Part X — Research architecture beyond the frozen runtime

## 32. Predictive semantic fields

The current research frontier asks whether Cortex should preserve a distinction merely because perception can see it.

The research answer is increasingly:

> preserve distinctions only when they matter to prediction.

Instead of treating every perceptual anchor as a hard ontological entity, the fuzzy predictive-field research maintains responsibilities over anchors and derives semantic distance from their learned future distributions.

At candidate resolution $\rho$,

$$
K_\rho(i,j)
=
\exp
\left[
-\frac{d_P(i,j)^2}{2\rho^2}
\right].
$$

This supports a continuum:

$$
\text{fine distinction}
\longleftrightarrow
\text{predictive equivalence}.
$$

---

## 33. Predictive rate-distortion

Research v1.7-R formalized adaptive resolution as:

$$
\rho_t^*
=
\arg\min_\rho C_t(\rho)
\quad
\text{subject to}
\quad
D_t(\rho)
\le
D_t^{\min}+\varepsilon.
$$

In plain language:

> choose the simplest representation whose predictive error remains close enough to the best available representation.

This is stronger than minimizing an arbitrary weighted sum of accuracy and complexity because the acceptable predictive distortion is explicit.

---

## 34. Hysteretic local resolution

A globally selected resolution can oscillate and can waste detail where only one part of the field needs refinement.

The research line therefore introduced:

- **hysteresis** — sharpen quickly, coarsen only after sustained evidence;
- **local resolution** — different anchors may use different predictive resolution.

```mermaid
flowchart LR
    A["Predictive field"]
    B1["Region A<br/>coarse"]
    B2["Region B<br/>coarse"]
    B3["Region C<br/>fine"]
    B4["Region D<br/>coarse"]

    A --> B1
    A --> B2
    A --> B3
    A --> B4
```

The interpretation is important:

> representation complexity is allocated where predictive evidence requires distinction.

It is not uniformly increased across the whole model.

---

## 35. Relational reasoning research

Cortex now has two complementary relation-reasoning research tracks.

### Prospective multi-hop relational inference

A learned graph diffusion operator can assign high potential to previously unseen pairs connected through indirect evidence.

The heat kernel

$$
K_\rho
=
e^{-\rho L}
$$

implicitly contains contributions from paths of many lengths.

Controlled experiments demonstrated unseen-pair prediction at first supporting depths 3, 5, and 7.

### Typed relational composition and law induction

The kinship-composition experiment adds a finite learned relation algebra.

Conceptually,

$$
s_{t+1}
=
C(s_t,r_t),
$$

where $C$ is learned from solved relation paths.

After training only on paths of length at most 5, the research prototype generalized recursively to unseen entities and path lengths 6–10.

The follow-up `algebra-induction-v1` experiment goes one step further. Cortex receives a partial anonymous binary-operation table and first decides which candidate structural laws are supported by evidence. The current candidate vocabulary contains associativity, commutativity and a unique two-sided identity.

For example, once associativity is evidence-gated, a missing product can be derived from

$$
(a\star b)\star c
=
a\star(b\star c).
$$

With 40% of each operation table hidden, this research path recovered held-out products at 1.000 resolved accuracy on both cyclic and noncommutative dihedral groups, while declining to derive products on a non-associative control.

The follow-up \`algebra-form-induction-v1.7\` removes the named-law vocabulary. A bounded grammar generates canonical equation forms directly, and fuzzy evidence is split into:

- **structural membership** — whether the form itself is supported;
- **predictive activation** — whether it has survived cross-fitted predictive checks;
- **applicability scope** — which substitutions are justified by witnessed support.

For one- and two-variable laws the scope uses marginal variable support. Three-variable laws additionally preserve pairwise support so independently witnessed marginals cannot be recombined into unsupported tuples.

On the hosted 120-case campaign, every structured-family resolved prediction was correct and every random-magma case remained unresolved. This is still bounded form discovery: the grammar and search depth remain externally supplied.

### Cross-world law transfer

`algebra-transfer-v1.6` tests whether a discovered canonical equation form can survive replacement of the underlying algebra.

Only the form and aggregate source evidence transfer. Source entity labels, source table cells, and source applicability basins do not.

On the final untouched campaign, transferred forms improved mean held-out coverage over scratch discovery by **+0.320 at 20% target evidence**, **+0.486 at 30%**, and **+0.537 at 40%**, while preserving zero wrong resolved compatible predictions and abstaining on random-magmas.

The failed v1.5 negative-transfer campaign is retained because it exposed a multiple-hypothesis problem that motivated v1.6's multiplicity-aware evidence requirement.

### Structural counterfactual reasoning

`counterfactual-reasoning-v1` starts the broader **Cortex Reasoning Map v1** program.

The research prototype mirrors the native intervention/control causal-state thresholds, then constructs:

- a **confirmed graph** from resolved causal edges;
- a **possible graph** from causal plus unresolved edges.

A query returns:

$$
\text{affected}
$$

when the target is reachable in the confirmed graph,

$$
\text{unresolved}
$$

when it is reachable only in the possible graph, and

$$
\text{unaffected}
$$

otherwise.

A structural counterfactual temporarily evaluates

$$
G' = G\setminus E_{remove}
$$

without mutating the learned base graph.

Across 140 untouched worlds, all frozen v1 gates passed: multi-hop composition, negative reachability, redundant-path preservation, bridge cuts, missing-bridge abstention, proof-path validity, and exact state restoration all measured **1.000**.

The direct-effect evidence in this first campaign is deterministic, so the result establishes the reasoning operation rather than noisy causal discovery.

See [REASONING_MAP_V1.md](REASONING_MAP_V1.md).

### Belief revision under contradiction

`belief-revision-v1` characterizes what the existing cumulative causal sufficient statistics imply when evidence reverses over time.

No decay, sliding window, reset, or dedicated belief-revision controller is added. As contradictory null-effect evidence accumulates, the target causal margin moves through the existing thresholds:

$$
\text{causal}
\rightarrow
\text{unresolved}
\rightarrow
\text{null}.
$$

When causal evidence returns, the same cumulative state reverses through

$$
\text{null}
\rightarrow
\text{unresolved}
\rightarrow
\text{causal}.
$$

Across 60 untouched official cases, initial support strengths 1, 2, and 3 produced retraction latencies of **7, 14, and 21** contradiction batches and recovery latencies of **3, 5, and 7** causal batches respectively. Every case preserved unrelated causal evidence exactly, and the downstream query state followed the local belief as affected → unresolved → unaffected → affected.

This is a research characterization of the cumulative evidence semantics, not a claim that the stable runtime contains a general-purpose epistemic revision controller. The evidence in v1 is deterministic.

The next Reasoning Map axis is structural analogy / isomorphism.

See [BELIEF_REVISION_V1.md](BELIEF_REVISION_V1.md) and [REASONING_MAP_V1.md](REASONING_MAP_V1.md).

These capabilities are currently **research modules**, not part of the stable native runtime.

---

# Part XI — Measured performance architecture

## 36. Why Cortex is implemented as one coarse native transition

The first migration experiments showed that moving only isolated numerical kernels to Rust did not produce the expected full-stack speedup.

The large improvement appeared when the complete stateful frame transition moved behind one Rust boundary.

On the declared 1,400-frame benchmark:

| Execution mode | Mean step | Peak RSS |
|---|---:|---:|
| Frozen Python | 10,315.3 µs | 87.43 MiB |
| Hybrid | 10,141.9 µs | 86.93 MiB |
| Fully native | **78.66 µs** | **21.99 MiB** |

On that workload this corresponds to roughly:

$$
131\times
$$

lower mean step time and about

$$
4\times
$$

lower peak process memory.

These are workload-specific measurements, not universal complexity claims.

---

## 37. Where native runtime time currently goes

The top-level `stage-attribution-v1` campaign found articulation to dominate the measured native runtime.

For the baseline profiling case, attributed work was approximately:

```mermaid
pie showData
    title Baseline native stage attribution
    "Articulation" : 89.6
    "Sparse graph" : 5.2
    "Public summary" : 4.0
    "Continual core" : 0.6
    "Binding conversion + fuzzy memory" : 0.6
```

This chart is workload-specific.

A second profiler then decomposed articulation itself and found cyclic transform-distance evaluation to dominate that stage. This is why current native optimization work targets exact transform-distance kernels rather than, for example, replacing the Hungarian assignment algorithm.

Measured bottlenecks determine optimization priority.

---

# Part XII — Complexity map

## 38. Approximate scaling surfaces

The exact cost depends strongly on configuration and realized state, but the main scaling surfaces are understandable.

Let:

- $k$ = detections visible in the current frame;
- $N$ = active persistent entities;
- $d$ = articulation feature dimension;
- $\lvert G\rvert$ = active nuisance-transform count;
- $R$ = represented sparse relation cells;
- $C$ = represented causal cells;
- $B_m$ = fuzzy-memory prototype count;
- $B_r$ = continual regime count.

A rough architecture-level map is:

| Component | Dominant scaling idea |
|---|---|
| Entity/transform scoring | approximately $O(kN\lvert G\rvert d)$ |
| Hungarian assignment | polynomial in the frame assignment matrix; measured smaller than transform scoring in current fixtures |
| Current-frame relation updates | approximately quadratic in visible entities, not total entity capacity |
| Stored graph memory | $O(R+C)$, allocated from observed evidence |
| Public 12-D summary | constant-dimensional output; graph counts maintained incrementally |
| Fuzzy memory matching | approximately $O(12B_m)$ |
| Continual regime matching | approximately $O(12B_r)$ |
| Exact curvature rank checks | dense in the fixed continual dimension and conditional/sampled rather than every frame |

These expressions are architectural guides, not formal asymptotic theorems for every code path.

---

# Part XIII — Design invariants

## 39. The invariants that define Cortex

The architecture is easier to understand if its implementation details are summarized as a small set of invariants.

### 39.1 Evidence before structure

A structural distinction is not created merely because one observation is surprising.

Persistent evidence is required.

### 39.2 Recurrence before rediscovery

If a known structure explains the observation, reactivate it rather than creating a duplicate.

### 39.3 Adapt before proliferating

Nearby drift should modify an existing representation unless predictive evidence justifies another.

### 39.4 Prediction before split

Geometric complexity may propose a split; future predictive gain must validate it.

### 39.5 Reversibility

A distinction that stops mattering should be allowed to merge again.

### 39.6 Boundedness is explicit

Entity, memory and regime capacities are configuration contracts rather than accidental machine limits.

### 39.7 Uncertainty is a valid output

When evidence or capacity is insufficient, `unresolved` is preferable to unsupported certainty.

### 39.8 Approximate memory does not control active truth

Historical compression cannot silently rewrite the present structural state.

### 39.9 Research does not silently redefine production semantics

New mechanisms must survive their own validation and specification process before entering the stable runtime.

---

# Part XIV — Repository map

## 40. Where the architecture lives

```text
crates/
  cortex-articulation/
      Persistent entity identity
      Nuisance-transform evidence
      Simultaneous assignment
      Reliability and residual/noise state

  cortex-graph/
      Sparse relation evidence
      Intervention-sensitive causal evidence
      Dependency closure

  cortex-memory/
      Bounded fuzzy historical memory
      Certified recompression
      Explicit unresolved state

  cortex-core/
      Continual regime prototypes
      Recurrence / adaptation / novelty
      Conditional tensor refinement
      Curvature/rank diagnostics
      Prospective split validation
      Merge/reconciliation

  cortex-runtime/
      Coarse-grained composition
      12-D public articulation state
      Runtime snapshots
      Optional profiling path

  cortex-python/
      PyO3 boundary for research and Python callers

python/cortex/
  hybrid.py
      Frozen-neural-encoder adapter

research/
      Experimental predictive geometry,
      adaptive resolution,
      relational inference,
      logical composition, and campaigns

reference/v0_9_5/
      Immutable Python executable specification

benchmarks/
      Performance and scientific campaign runners

benchmarks/results/
      Committed aggregate benchmark summaries

tests/
      Unit, differential, migration, hybrid, and research tests
```

---

# Part XV — What Cortex is not

## 41. Non-claims

The current architecture should not be described as:

- a general-purpose replacement for neural networks;
- a universal theorem prover;
- a proven optimal continual-learning algorithm;
- a complete causal discovery system;
- an AGI architecture;
- proof of the Sigma–Lambda theory;
- a system that can always resolve novelty under fixed resources.

What is implemented and supported by evidence is narrower:

- bounded state;
- persistent entity articulation;
- learned nuisance handling;
- sparse relation and causal evidence;
- fuzzy bounded historical memory;
- recurrence, adaptation, novelty, split and merge behavior;
- explicit unresolved/budget-pressure states;
- differential Rust/Python behavioral validation;
- specific research demonstrations of predictive relational inference, adaptive resolution, and recursive relation composition.

---

# Part XVI — Architecture in one page

## 42. Complete conceptual map

```mermaid
flowchart TB
    subgraph Input["Observation layer"]
        RAW["Raw modality<br/>(optional)"]
        NN["Frozen neural encoder<br/>(Hybrid Stage I)"]
        DET["Structured detection vectors"]
        REL["Relation observations"]
        INT["Intervention + outcomes"]
        Y["Optional scalar predictive outcome"]

        RAW --> NN --> DET
    end

    subgraph Art["Articulation"]
        MATCH["Entity × transform scoring"]
        ASSIGN["Global injective assignment"]
        PROTO["Persistent prototypes"]
        RELIAB["Feature reliability"]
        NUIS["Learned nuisance subgroup"]
        NOISE["Residual/noise state"]

        DET --> MATCH
        PROTO --> MATCH
        RELIAB --> MATCH
        NUIS --> MATCH
        MATCH --> ASSIGN
        ASSIGN --> PROTO
        ASSIGN --> RELIAB
        ASSIGN --> NUIS
        ASSIGN --> NOISE
    end

    subgraph Graph["Sparse relational state"]
        RG["Relation evidence"]
        CG["Intervention-sensitive causal evidence"]
        DEP["Dependency closure"]

        ASSIGN --> RG
        ASSIGN --> CG
        REL --> RG
        INT --> CG
        RG --> DEP
        CG --> DEP
    end

    subgraph Summary["Public structural interface"]
        Z["12-D articulation vector"]
        PROTO --> Z
        NUIS --> Z
        NOISE --> Z
        RG --> Z
        CG --> Z
    end

    subgraph Memory["Historical memory"]
        FM["Fuzzy Accordion Memory"]
        RC["Certified recompression"]
        MU["Memory unresolved"]
        Z --> FM
        FM --> RC
        RC --> MU
    end

    subgraph Core["Continual structural reasoner"]
        REG["Regime prototypes"]
        REC["Recurrence / reactivation"]
        NOV["Novelty"]
        TENSOR["Conditional residual tensor"]
        CURV["Curvature / rank witness"]
        SPLIT["Prospective split"]
        MERGE["Merge / reconciliation"]
        UNR["Unresolved / budget pressure"]

        Z --> REG
        Y --> REG
        REG --> REC
        REG --> NOV
        REG --> TENSOR
        TENSOR --> CURV
        CURV --> SPLIT
        SPLIT --> REG
        REG --> MERGE
        MERGE --> REG
        NOV --> UNR
        SPLIT --> UNR
    end

    subgraph Research["Research-only frontier"]
        PF["Predictive semantic fields"]
        RD["Rate-distortion resolution"]
        LR["Hysteretic local resolution"]
        RI["Multi-hop relation inference"]
        KA["Typed relation algebra"]

        PF --> RD --> LR
        RI --> KA
    end

    Z -. "research prototypes" .-> PF
    RG -. "research prototypes" .-> RI
```

---

## 43. The architectural principle

All of these mechanisms point to one common design rule:

$$
\boxed{
\text{Use the smallest revisable structure that available evidence can justify.}
}
$$

Cortex therefore treats representation as something that may:

- appear;
- become more precise;
- recur;
- split;
- merge;
- be compressed;
- or remain unresolved.

The system is not designed around the assumption that the correct internal ontology is known in advance.

It is designed around the idea that **structure should be earned by evidence and remain reversible when the evidence changes**.

---

## 44. Further reading

For current project state and roadmap:

- [PROJECT_STATUS.md](PROJECT_STATUS.md)
- [PROJECT_MILESTONES.md](PROJECT_MILESTONES.md)
- [REASONING_MAP_V1.md](REASONING_MAP_V1.md)

For the frozen behavioral contract:

- [V1_CONTRACT.md](V1_CONTRACT.md)

For native migration and implementation status:

- [NATIVE_MIGRATION.md](NATIVE_MIGRATION.md)

For runtime profiling:

- [FULL_RUNTIME_BENCHMARK.md](FULL_RUNTIME_BENCHMARK.md)
- [NATIVE_SCALING_SWEEP_V1.md](NATIVE_SCALING_SWEEP_V1.md)
- [NATIVE_STAGE_ATTRIBUTION.md](NATIVE_STAGE_ATTRIBUTION.md)
- [ARTICULATION_ATTRIBUTION.md](ARTICULATION_ATTRIBUTION.md)

For the research lineage:

- [RESEARCH_HISTORY.md](RESEARCH_HISTORY.md)

For neural/Cortex integration:

- [HYBRID_STAGE_I.md](HYBRID_STAGE_I.md)

For software/specification/protocol versioning:

- [VERSIONING.md](VERSIONING.md)
