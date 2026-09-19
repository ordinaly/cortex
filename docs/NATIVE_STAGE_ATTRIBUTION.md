# Native stage attribution — Cortex v1.0.0-rc.2

This document records the first top-level stage attribution campaign for the composed native Cortex runtime and the accompanying regime-budget saturation campaign.

## Provenance

- Cortex release: `1.0.0-rc.2`
- frozen executable specification: `spec-v0.9.5`
- stage protocol: `stage-attribution-v1`
- saturation protocol: `regime-saturation-v1`
- benchmarked commit: `079f7816af7cf87495dd710c7f53d9af3167277d`
- repetitions: 3 per case

The normal production `CortexRuntime.step()` remains uninstrumented. The campaign used a parity-gated profiled transition.

## Stage-attribution result

Across the six declared stress cases, articulation dominates attributed native time:

| case | internal mean | articulation share | next-largest measured cost |
|---|---:|---:|---|
| baseline | 126.92 µs | 89.6% | graph 5.2% |
| 96 nominal entities | 281.00 µs | 92.2% | public summary 4.6% |
| 16 visible entities | 442.11 µs | 85.8% | graph 11.6% |
| 64-D features | 384.51 µs | 95.1% | public summary 2.7% |
| sparse topology | 229.36 µs | 86.4% | public summary 10.1% |
| high tensor duty | 131.63 µs | 82.9% | continual core 7.8% |

Binding conversion and fuzzy-memory stages remain below 1% in every case.

The high-tensor-duty fixture raises continual-core cost but does not displace articulation as the dominant stage. This is evidence against optimizing the tensor path before articulation.

## Follow-up attribution

The follow-up protocol `articulation-attribution-v1` has now been completed.

It shows that cyclic transform-distance work dominates articulation itself. In the measured cases, matching plus subgroup-evidence transform scans consume approximately **85–96%** of attributed articulation work.

That follow-up supersedes the original recommendation to merely “decompose articulation”; decomposition is no longer pending.

See [ARTICULATION_ATTRIBUTION.md](ARTICULATION_ATTRIBUTION.md) for the current native optimization target.

## Regime-saturation result

The dedicated saturation fixture deliberately fills the configured regime budget and then presents additional novel regimes.

| budget | attempted regimes | stored regimes | first pressure step |
|---:|---:|---:|---:|
| 2 | 6 | 2 | 13 |
| 4 | 8 | 4 | 25 |
| 8 | 12 | 8 | 49 |
| 16 | 20 | 16 | 97 |

For every tested budget:

- stored regime count reaches the configured bound exactly;
- it never exceeds that bound;
- the first post-capacity novel regime immediately exposes `budget_pressure` and `unresolved`;
- additional novelty does not cause hidden structural growth.

This is an engineering validation of the implemented bounded-state contract under the declared fixture, not a universal proof.

## Reproducible result files

- `benchmarks/results/stage_attribution_v1_medians.csv`
- `benchmarks/results/regime_saturation_v1_medians.csv`
- `benchmarks/benchmark_stage_attribution.py`
- `benchmarks/benchmark_regime_saturation.py`

## Current engineering decision

The measured native optimization sequence is now:

1. full runtime migration;
2. remove accidental global graph snapshot work;
3. top-level stage attribution;
4. fine-grained articulation attribution;
5. optimize the exact transform-distance kernel without changing semantics;
6. rerun differential and attribution campaigns before considering broader SIMD, parallelism, or alternative assignment machinery.
