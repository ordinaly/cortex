# Geometric Temporal Transport v1.1

## Goal

Promote `K = temporal transport` from a symbolic state-ID transition table to a transport kernel over Cortex articulations.

The central requirement is that identity may persist through **rearticulation**. A higher-order simplex can appear, disappear, or change without forcing Cortex to declare a new entity when the lower-dimensional geometry remains continuous.

## Multiscale transport

For two `SimplicialIntegrationRead` objects, Cortex compares all shared probe marginals through a protected transport scale.

The v1.1 reference policy uses dimensions 0 and 1:

- singleton probe marginals: weight `0.60`;
- pairwise marginals: weight `0.40`.

The transport distance is the weighted mean of total-variation distances over those shared supports, plus a bounded penalty when probe support is born or lost.

Higher-order joint distance and topology change are still reported as diagnostics, but they do not define identity in v1.1.

The successor kernel is

`K(y | x) proportional to exp(-beta * d_K(x,y))`.

Candidate labels are used only for reporting/tie-breaking; transport is determined by articulation geometry.

## Adversarial benchmark

Four probes are used. Probe 0 is a persistent identity marker. Probes 1–3 carry a parity articulation whose singleton and pairwise marginals are identical under EVEN, ODD, and independent UNIFORM regimes.

Two identities are constructed:

- source A: marker `p=0.30`, EVEN parity;
- source B: marker `p=0.70`, ODD parity.

Their successors are deliberately rearticulated:

- target A: marker `p=0.30`, UNIFORM higher-order structure (coface disappears);
- target B: marker `p=0.70`, EVEN parity.

At the population level raw full-joint similarity is adversarial:

- source A is closer in full joint TV to target B than target A;
- source B is closer in full joint TV to target A than target B.

So a nearest-neighbor matcher over the complete instantaneous representation swaps both identities.

By contrast, singleton/pairwise parity geometry is unchanged, and the marker remains stable. The multiscale transport kernel should therefore preserve A→A and B→B.

A second campaign removes one non-marker probe from both candidate targets. The correct identity should survive using shared geometry with overlap `3/4`.

## Acceptance

On development and held-out sampled campaigns:

- geometric identity accuracy = 100%;
- naive full-joint accuracy <= 10%;
- mean correct successor probability >= 0.65;
- every held-out seed retains a positive correct-vs-cross distance margin above 0.05;
- probe-drop geometric identity accuracy = 100%;
- observed probe overlap in the drop campaign = 0.75.

## Scope

This milestone does not yet learn object correspondences under arbitrary permutations, births/splits/merges, or transport maps between nonmatching probe semantics. It establishes the more basic claim that temporal continuity should be defined on protected articulation geometry rather than equality or full-joint resemblance.
