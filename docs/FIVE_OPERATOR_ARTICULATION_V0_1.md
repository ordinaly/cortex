# Five-Operator Articulation v0.1

Status: research-only overlay. It does not change `spec-v0.9.5` or the native Rust runtime.

This experiment implements one shared evolving articulation substrate for five typed operations:

- `Delta` — local distinguishability;
- `K` — temporal transport;
- `C` — intervention-sensitive causal influence;
- `V` — orientation / valuation;
- `I` — local-to-global integration.

It is a functional architecture experiment. The presence of these mechanisms is **not** treated as evidence that Cortex is conscious or has subjective experience.

## Shared substrate

The implementation lives in `python/cortex/five_operator_loop.py`.

A persistent population of `DistinctionNeuron` objects supplies typed local distinction operators. The same neuron set is used by `DynamicDistinguishabilityComplex`, whose active simplicial topology changes with the current relevance field.

The coordinator is:

```text
previous articulation
      |
      v
K: temporal transport / surprise
      |
observation --> Delta: local distinctions
      |                |
      |                v
      +----------> relevance field <---------- C: causal influence
                         ^
                         |
                    V: orientation
                         |
                         v
              active simplicial complex
                         |
                         v
              I: local/global integration
                         |
                         v
                 current articulation
```

No operator owns a separate representation. They all act on the same active neuron complex and articulation frame.

## Delta: distinguishability

`DistinctionBank` contains typed `DistinctionNeuron` instances. Each neuron returns a normalized response in `[0,1]` for a pair of structured observations.

The bank exposes:

- per-neuron distinction;
- per-channel weighted distinction;
- a population aggregate.

The module deliberately does not require a single neuron to separate every state.

## K: temporal transport

`TemporalTransport` stores an empirical kernel `K(y | x)` over articulated state identifiers. Identity is therefore represented as stable transport through change rather than equality of observations.

A state may be strongly distinguishable from its predecessor under `Delta` while still being the dominant temporal successor under `K`.

## C: causal influence

`CausalModel` keeps intervention and control evidence separate. Influence is only non-zero after both are present:

```text
C(source -> target) = |mean(do outcomes) - mean(control outcomes)|
```

This is intentionally more conservative than treating temporal succession or correlation as causal evidence.

## V: orientation / valuation

`OrientationField` is vector-valued. Neurons may declare relevant orientation axes, and their contextual salience contributes to activation. The implementation does not collapse orientation into a universal scalar reward, although a signed utility readout is available for inspection.

## Dynamic neuronal topology

`DynamicDistinguishabilityComplex` maintains a fixed potential simplicial complex and a context-dependent active subcomplex.

A simplex receives a raw score from its vertex relevance plus optional higher-order synergy. The final score is face-consistent:

```text
score(simplex) = min(raw_score(face) for every non-empty face)
```

This guarantees that a higher-order interaction cannot activate without all of its faces.

Activation uses hysteresis:

```text
score >= activation_threshold     -> active
score <= deactivation_threshold   -> inactive
otherwise                         -> retain previous state
```

The implementation includes a defensive assertion that the result remains a simplicial complex.

## I: integration / composition

`Integrator` combines `LocalView` objects whose supports are active simplices. It computes a confidence-weighted least-squares global articulation and separately reports:

- integration residual;
- conflicting coordinates;
- number of local views actually used.

Contradictions are therefore exposed instead of being silently erased by averaging.

## Coordinated loop

`FiveOperatorArticulationLoop.step()` executes the shared loop:

1. compare the previous and current observations with `Delta`;
2. compute temporal surprise from `K`, then record observed transport;
3. ingest intervention/control evidence into `C`;
4. compute orientation salience with `V`;
5. form neuron relevance from novelty, temporal surprise, causal influence, valuation, and a base term;
6. update the active distinguishability complex with hysteresis;
7. integrate local views through `I` on the active complex only;
8. retain the resulting frame as the next temporal predecessor.

## Invariants covered by tests

`tests/test_five_operator_loop.py` checks:

- typed channel aggregation under `Delta`;
- temporal transport independent of static similarity;
- causal influence requiring both intervention and control evidence;
- face closure and hysteresis in the active complex;
- contradiction reporting by the integrator;
- end-to-end coupling of all operators through one topology.

## Deliberate limitations in v0.1

This version is an executable reference, not a learned production mechanism.

- Distinction neurons are supplied by the caller; they are not yet learned.
- Potential simplices are supplied by the caller; topology selection is dynamic, topology discovery is not yet learned.
- Synergy is an input signal rather than an inferred higher-order statistic.
- `TemporalTransport` is an empirical discrete-state kernel, not yet a learned transport map between continuous articulation geometries.
- `CausalModel` uses normalized scalar outcomes and does not yet perform causal identification from passive data.
- Orientation axes and weights are supplied externally.
- Integration currently operates on scalar named coordinates; richer typed fibers/sheaves remain future work.
- The research loop is not wired into the frozen native runtime.

These limitations are intentional. Promotion should require held-out evidence that each mechanism earns its representational and computational cost.
