# Native articulation attribution — `articulation-attribution-v1`

This document records the fine-grained profiling campaign run after `stage-attribution-v1` identified native articulation as the dominant Cortex runtime stage.

## Provenance

- Cortex release: `1.0.0-rc.2`
- Frozen executable specification: `spec-v0.9.5`
- Benchmark protocol: `articulation-attribution-v1`
- Campaign source commit: `fd2ec85b6b5b72c0a805a01513520dc834ecbbc4`
- Repetitions: 3 fresh processes per case
- Measured frames: 900 per repetition after 300 warm-up frames
- Runner: GitHub Actions `ubuntu-latest`, Python 3.13, Rust 1.98.1

The profiler is opt-in. Production `Articulation.step()` remains uninstrumented. The profiled transition mirrors the production transition and is parity-gated against it: the final Python 3.11 and 3.13 differential jobs, Rust tests, Clippy, and wheel build all passed on the campaign commit.

The committed aggregate table is [`../benchmarks/results/articulation_attribution_v1_medians.csv`](../benchmarks/results/articulation_attribution_v1_medians.csv).

## Sub-stages

The profiler attributes articulation time to input validation, entity/transform match scoring plus cost-matrix construction, Hungarian assignment, binding finalization, aligned residual/statistics updates, subgroup-evidence accumulation, touched-entity deduplication, prototype/reliability refresh, noise-state refresh, and subgroup selection.

It also reports deterministic work counters such as entity-pair scores, transform evaluations, assignment dimensions, spawns, and subgroup-evidence transform evaluations. These counters are important because they distinguish a genuinely expensive primitive from a stage that is merely called more often.

## Main result

The dominant computation is now localized much more narrowly than “articulation”. In the baseline case (32 nominal entities, 31 active, 8 visible detections, 16-D features), median internal profiled articulation time was **116.16 µs**. The attributed mean-stage composition was:

| sub-stage | mean time | share of attributed time |
|---|---:|---:|
| entity/transform match scoring | **90.56 µs** | **81.9%** |
| subgroup-evidence transform scan | **11.64 µs** | **10.4%** |
| aligned entity/statistics update | 2.45 µs | 2.2% |
| Hungarian assignment | 2.01 µs | 1.8% |
| subgroup selection | 1.81 µs | 1.6% |
| prototype/reliability refresh | 0.69 µs | 0.6% |
| binding finalization | 0.61 µs | 0.6% |
| noise-state refresh | 0.57 µs | 0.5% |
| deduplication | 0.19 µs | 0.2% |
| validation | 0.11 µs | 0.1% |

The two transform-distance stages therefore account for approximately **92.4%** of attributed baseline articulation time.

## Entity and visibility scaling

At fixed 16-D feature size, match-scoring cost follows the number of entity/transform comparisons closely.

| case | active entities | visible | match transform evals/frame | match scoring | ns / transform | transform-kernel share* |
|---|---:|---:|---:|---:|---:|---:|
| nominal entities 8 | 8 | 8 | 256 | 21.90 µs | 84.5 ns | 85.5% |
| baseline / nominal 32 | 31 | 8 | 992 | 90.56 µs | 91.2 ns | 92.4% |
| nominal entities 64 | 57 | 8 | 1,763 | 158.49 µs | 90.5 ns | 93.7% |
| nominal entities 96 | 78 | 8 | 2,496 | 208.97 µs | 84.0 ns | 89.5% |
| visible 16 | 61 | 16 | 3,904 | 346.02 µs | 88.6 ns | 95.0% |

\* Match scoring plus subgroup-evidence transform scans.

Across the principal 16-D entity/visibility cases, the per-transform match cost remains roughly **84–91 ns**. This is strong evidence that the major scaling effect is the count of transform-distance evaluations rather than Hungarian assignment or downstream state maintenance.

Hungarian assignment remains small in this workload. Even at 16 visible detections it costs about **5.57 µs**, only about **1.4%** of attributed articulation time. Replacing the assignment algorithm before addressing transform scoring would therefore target the wrong bottleneck.

## Feature-dimension scaling

Feature dimension exposes a second effect:

| feature dimension | match scoring | subgroup evidence | subgroup-evidence share | combined transform share |
|---:|---:|---:|---:|---:|
| 8 | 51.45 µs | 5.00 µs | 8.1% | 89.0% |
| 16 | 89.79 µs | 11.78 µs | 10.5% | 92.5% |
| 32 | 143.67 µs | 32.85 µs | 17.6% | 94.8% |
| 64 | 232.90 µs | **112.61 µs** | **31.4%** | **96.2%** |

After subgroup inference stabilizes in these fixtures, the matching subgroup contains four shifts, so ordinary entity matching performs approximately

`visible × active_entities × 4`

transform evaluations, with each distance evaluation scanning the feature vector.

Subgroup evidence is structurally different: qualifying detections are compared over **all feature-dimension cyclic shifts**. Each shifted distance itself scans the feature vector. Its dominant work therefore grows like `qualifying_detections × d × d` under the present implementation. This explains why subgroup evidence becomes a substantial fraction of the 64-D case even though it is secondary at 16-D.

## Source-level interpretation

The measurements line up with a concrete implementation hotspot.

`entity_cost_and_q` currently materializes a shifted prototype vector for each candidate transform before calling `weighted_mse`. It also computes softmax transformation evidence for every detection/entity pair and stores it in `qcache`; the composed `observe` path ultimately uses the selected binding and shift but does not consume `BindingResult.evidence` afterward.

The subgroup-evidence loop similarly materializes a shifted previous vector for every possible cyclic shift and constructs a unit-weight vector inside each distance evaluation.

These are not semantic problems; they are implementation costs. They suggest a narrow optimization experiment before considering SIMD, parallelism, or a different assignment algorithm:

1. implement allocation-free cyclic weighted-MSE kernels that index shifted coordinates directly rather than constructing shifted vectors;
2. reuse or eliminate the temporary unit-weight vector in subgroup evidence;
3. split the internal hot binding path from the public evidence-producing `bind` API so `observe` does not compute/store transformation probability vectors it subsequently discards, while preserving the public `bind` semantics;
4. rerun `articulation-attribution-v1` and the frozen differential suite as an A/B gate.

Only after those changes should SIMD or parallel entity scoring be evaluated. The current data says the first question is how much redundant allocation/evidence construction can be removed from the transform kernel while preserving exact behavior.

## Secondary observations

Subgroup selection is normally small, but reaches about **16.82 µs / 7.2%** in the nominal-96-entity case. That is worth revisiting after the dominant transform kernel is reduced; optimizing it first would have limited end-to-end effect today.

Topology does not introduce a new articulation bottleneck. The sparse-topology case is still dominated by match scoring (88.3%) plus subgroup evidence (5.9%), consistent with the prior conclusion that historical graph topology is no longer the primary per-frame cost.

Profiler timing overhead is visible but bounded: median unattributed internal time ranges from roughly 2–23 µs across the campaign and grows with frame work. For optimization decisions, stage shares and work-normalized comparisons are therefore more informative than treating the profiled total as a production-latency measurement.

## Conclusion

The performance problem has now been localized from “native runtime” to “articulation”, and from “articulation” to a specific family of **cyclic transform-distance evaluations**. In the measured cases these kernels consume roughly **85–96%** of attributed articulation work, with match scoring dominant under entity/visibility growth and the all-shifts subgroup-evidence scan becoming increasingly important as feature dimension grows.

The next optimization pass should therefore be allocation/evidence-elision inside those exact kernels, followed by the same differential and benchmark gates.
