# Cortex v1.1-R — Multiresolution Semantic Geometry

Status: **research model / not a software release**.

This document proposes the next Cortex research model after Hybrid Stage I. It
does not replace the current \`v1.0.0-rc.2\` software version or the frozen
\`spec-v0.9.5\` executable specification.

## Core idea

A neural perception layer can propose symbols and perceptual evidence, but
Cortex should not inherit the neural network's latent geometry as its final
notion of meaning.

Cortex instead learns a semantic geometry from:

1. the informational role of each symbol;
2. the relations accumulated between symbols;
3. a **resolution parameter** controlling how far relational context is allowed
   to influence semantic proximity.

The resulting object is not one global distance. For each symbol pair
\(i,j\), Cortex obtains a **resolution spectrum**

\[
\mathcal D_{ij} = \{d_\rho(i,j): \rho \ge 0\}.
\]

At fine resolution, symbols are compared by their own functional/predictive
roles. At coarser resolution, relational neighborhoods increasingly influence
their meaning.

A hammer and a nail may therefore be far apart at \(\rho=0\), because their
functions differ, while becoming progressively closer for \(\rho>0\), because
they repeatedly participate in the same relational structure.

## Why semantic proximity is not identity

This distinction is mandatory:

\[
\text{semantic proximity at coarse resolution}
\neq
\text{identity}.
\]

A hammer and a nail can become semantically close without becoming the same
entity or the same functional concept.

Therefore:

- entity binding remains a fine-scale/perceptual-temporal operation;
- structural merge decisions require fine-scale predictive equivalence;
- coarse semantic geometry is used for association, analogy, context and
  higher-order reasoning;
- no merge may be justified solely because \(d_\rho\) is small for large
  \(\rho\).

## 1. Fine-scale informational representation

For each stable Cortex symbol \(i\), let

\[
q_i^{(k)}
\]

be a learned probability distribution describing one informational role
channel, for example:

- future structural outcomes;
- affordances/effects;
- intervention outcomes;
- transition regimes;
- other explicitly defined predictive channels.

For each channel we use the square-root probability embedding

\[
\phi_i^{(k)} = \sqrt{q_i^{(k)}}
\]

elementwise, and concatenate channels with non-negative weights:

\[
\phi_0(i)
=
\bigoplus_k \sqrt{\omega_k}\,\phi_i^{(k)}.
\]

Then the fine-scale distance

\[
d_0(i,j)
=
\frac{1}{\sqrt 2}
\left\|
\phi_0(i)-\phi_0(j)
\right\|_2
\]

is a weighted Hellinger-style informational distance.

The neural embedding is **not** part of this semantic distance by default. It
is evidence used to articulate and maintain symbols. If Cortex has
insufficient structural evidence to estimate \(\phi_0(i)\), semantic distance
is unresolved rather than silently replaced by visual similarity.

## 2. Learned semantic relation graph

Let Cortex maintain a non-negative symmetric semantic-affinity matrix

\[
A \in \mathbb R_{\ge0}^{N\times N}.
\]

An entry \(A_{ij}\) represents supported semantic connection evidence between
symbols \(i\) and \(j\).

A general typed construction is

\[
A
=
\sum_{\ell\in\mathcal R}
\theta_\ell A^{(\ell)},
\qquad
\theta_\ell\ge0,
\]

where relation channels may include co-use, co-occurrence, spatial relation,
shared event membership, or other explicitly defined symmetric affinities.

Evidence confidence should modulate edge weights. Directed causal evidence is
not automatically symmetrized into semantic affinity; it remains a directed
object unless a separately specified causal-association channel is justified.

Define the weighted graph Laplacian

\[
L = D-A,
\qquad
D_{ii}=\sum_j A_{ij}.
\]

## 3. Resolution operator

Resolution is represented by a diffusion scale \(\rho\ge0\).

Define the heat operator

\[
H_\rho = e^{-\rho L}.
\]

The contextual semantic representation is

\[
\Phi_\rho = H_\rho\Phi_0,
\]

where row \(i\) of \(\Phi_0\) is \(\phi_0(i)\).

The semantic distance at resolution \(\rho\) is

\[
\boxed{
d_\rho(i,j)
=
\frac{1}{\sqrt2}
\left\|
\Phi_\rho(i)-\Phi_\rho(j)
\right\|_2
}
\]

At \(\rho=0\),

\[
H_0=I
\quad\Longrightarrow\quad
d_\rho=d_0.
\]

For every fixed \(\rho\), \(d_\rho\) is a **pseudometric** because it is
ordinary Euclidean distance between rows of the contextual representation
\(\Phi_\rho\): it is non-negative, symmetric, and obeys the triangle
inequality. Identity of indiscernibles may intentionally fail at coarser
resolution when two distinct symbols become contextually indistinguishable.
That failure is semantic coarsening, not entity merging.

For larger \(\rho\), each symbol is increasingly represented through the
semantic structure around it.

For implementation, user-facing discrete resolutions may be

\[
\rho_r = r\Delta,\qquad r=0,1,2,\ldots
\]

while the mathematical model remains continuous.

## 4. Exact hammer–nail result

Consider only two symbols, hammer \(h\) and nail \(n\), connected by edge
weight \(w>0\):

\[
A=
\begin{pmatrix}
0&w\\
w&0
\end{pmatrix}.
\]

Then

\[
L=
\begin{pmatrix}
w&-w\\
-w&w
\end{pmatrix}
\]

and the difference vector is an eigenvector of \(L\) with eigenvalue \(2w\).

Therefore

\[
\Phi_\rho(h)-\Phi_\rho(n)
=
e^{-2w\rho}
\left[
\Phi_0(h)-\Phi_0(n)
\right]
\]

and hence

\[
\boxed{
d_\rho(h,n)
=
e^{-2w\rho}d_0(h,n).
}
\]

So two functionally distinct but strongly related symbols become
exponentially closer as resolution broadens.

This is not metaphorical: it is an exact consequence of the proposed
resolution operator.

## 5. Why path depth appears naturally

The matrix exponential expands as

\[
e^{-\rho L}
=
I-\rho L
+\frac{\rho^2L^2}{2!}
-\frac{\rho^3L^3}{3!}
+\cdots.
\]

Consequently:

- first-order relational structure enters at order \(\rho\);
- two-step relational structure enters through \(L^2\);
- deeper paths progressively enter at higher orders.

Resolution therefore controls how much relational depth contributes to
meaning without requiring a hard path-length cutoff.

This also allows a practical discrete interpretation:

- resolution 0: intrinsic informational role;
- resolution 1: direct relational context dominates first;
- resolution 2: relations-of-relations become significant;
- higher resolution: broader semantic communities and domains become visible.

## 6. Connection depth

For a chosen semantic tolerance \(\varepsilon\), define

\[
r_\varepsilon(i,j)
=
\inf
\{r\in\mathbb N:
d_{r\Delta}(i,j)\le\varepsilon
\}.
\]

This is the **connection depth** of two symbols.

Interpretation:

- small \(r_\varepsilon\): deeply/directly connected concepts;
- larger \(r_\varepsilon\): connection appears only through broader context;
- \(+\infty\): the tested resolution range never makes them semantically close.

Cortex can therefore ask not only

> How similar are these symbols?

but

> At what resolution do these symbols become part of the same semantic
> structure?

## 7. Semantic spectrum

For practical reasoning, Cortex should retain

\[
\mathbf d_{ij}
=
[
d_0(i,j),
d_{\Delta}(i,j),
d_{2\Delta}(i,j),
\ldots,
d_{R\Delta}(i,j)
].
\]

Two symbol pairs can have the same distance at one resolution while having very
different spectra.

Examples:

- **functional analogues**: small \(d_0\), little additional contraction;
- **complementary objects**: large \(d_0\), rapid contraction at low resolution;
- **domain relatives**: large \(d_0\), slow contraction over several scales;
- **unrelated symbols**: large distance across the whole tested spectrum.

The spectrum therefore contains more information than any single scalarized
semantic distance.

## 8. Neural/Cortex architecture

The hybrid architecture becomes:

    raw sensory stream
            |
            v
    neural perception
            |
            | perceptual embeddings
            v
    Cortex articulation
            |
            | persistent symbols
            v
    informational role estimates Phi_0
            |
            +------ semantic relation evidence ------+
            |                                        |
            v                                        v
      fine geometry d_0                       relation graph A
            |                                        |
            +----------------+-----------------------+
                             |
                             v
                   resolution operator H_rho
                             |
                             v
                 multiresolution geometry d_rho
                             |
                             v
             reasoning / analogy / context / memory

The neural network answers what the current observation looks like.

Cortex learns what distinctions remain informative, which symbols participate
in common structures, and at what resolution those distinctions matter.

## 9. Operations affected by resolution

### Identity

Use perceptual-temporal evidence and fine structural evidence. Coarse semantic
proximity must not collapse persistent identities.

### Split

A split remains justified when fine-scale predictive/causal evidence requires a
distinction.

### Merge

A merge requires fine-scale equivalence. Small coarse-scale distance is
insufficient.

### Association

Use \(d_\rho\) at a task-selected non-zero resolution.

### Analogy

Compare semantic spectra and relation-conditioned neighborhoods rather than
only base embeddings.

### Retrieval

A query can request nearby symbols at a specific resolution.

### Novelty

A symbol may be perceptually novel while semantically close at coarse
resolution, or perceptually familiar while structurally novel. These become
different measurable events.

## 10. Confidence and unresolved semantics

Every informational channel and graph edge carries evidence support.

Cortex should expose semantic distance as unresolved when support is
insufficient. A future implementation should propagate uncertainty through
\(\Phi_0\), \(A\), and \(H_\rho\) rather than converting low evidence into a
false precise metric.

## 11. Complexity warning

A dense matrix exponential costs roughly cubic time and quadratic memory in the
number of symbols. Using it directly in the online Cortex hot path would worsen
complexity and is **not** proposed as the production implementation.

The dense exponential is appropriate only for small research fixtures.

Potential scalable approximations include:

- sparse Krylov action \(e^{-\rho L}X\);
- Chebyshev polynomial approximation;
- bounded iterative diffusion;
- local neighborhood truncation with explicit error bounds.

Any production method must be benchmarked against the exact small-graph
reference before adoption.

## 12. Research claims and non-claims

The proposed construction gives a precise mechanism by which relational
resolution changes semantic distance. It does not yet establish that this is
the unique or optimal semantic geometry for Cortex.

Related mathematical ideas exist in graph diffusion, multiscale community
detection, information geometry and behavioral/bisimulation metrics. Novelty
claims require a separate literature audit.

The immediate research target is narrower:

> Does multiresolution informational geometry expose useful relationships that
> cannot be recovered from neural embedding distance or one-scale Cortex
> structure alone?

## 13. First falsifiable experiments

1. **Hammer/nail toy world** — verify exact two-node contraction.
2. **Resolution-order reversal** — construct a case where hammer is closer to
   screwdriver at resolution 0 but closer to nail at a coarser resolution.
3. **Disconnected control** — verify unrelated components do not acquire the
   same spectrum as directly related concepts.
4. **Pseudometric check** — verify triangle inequality at every tested
   resolution while permitting coarse contextual collapse.
5. **No-merge invariant** — coarse semantic proximity must not trigger entity
   identity merging.
6. **Synthetic compositional world** — objects, tools, actions and outcomes;
   test whether relationally meaningful groups emerge at different scales.
7. **Neural stream** — apply the geometry to persistent entities learned from
   frozen neural embeddings and compare against cosine-only retrieval.

A successful result would be a stable, interpretable semantic spectrum whose
resolution transitions track known relational structure without destroying
fine-scale identity distinctions.
