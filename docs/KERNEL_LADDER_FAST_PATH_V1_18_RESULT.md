# Cortex v1.18-R — Exact Kernel-Ladder Fast Path Result

Status: **numerically valid candidate rejected by frozen performance gates**.

Protocol: **kernel-ladder-fast-path-v1.18-R**.

Official GitHub Actions run: **35503800382**.

## Candidate

For the canonical resolution ladder 0.03 / 0.06 / 0.12 / 0.25, the candidate
computes the 0.12 kernel once, derives 0.06 and 0.03 by repeated squaring, and
computes 0.25 independently.

Development exposed a scale crossover, so the candidate is enabled only from
192 anchors upward. Smaller banks use the unchanged direct kernel path.

## Official numerical result

Across untouched official seeds:

- maximum kernel error: **1.89e-15**;
- maximum candidate-prediction error: **1.11e-16**;
- distance/feature/probability/outcome-count error: **0**;
- underflow-pattern mismatches: **0**;
- dirty-mask disagreements: **0**;
- structural disagreements: **0**.

The algebraic transformation is therefore behaviorally safe on the declared
fixtures.

## Frozen official performance result

| anchors | kernel speedup | refresh speedup | fused speedup |
|---:|---:|---:|---:|
| 72 | 0.963× | 0.998× | 1.001× |
| 144 | 0.972× | 0.991× | 1.017× |
| 288 | **1.115×** | **1.017×** | **0.998×** |

Frozen 288-anchor gates required:

- kernel speedup >= **1.40×**;
- refresh speedup >= **1.10×**;
- fused speedup >= **1.02×**.

All three failed.

## Interpretation

The development sample had suggested a much larger large-scale gain. The
untouched official campaign did not reproduce it.

This is evidence that the fine-grained v1.17 attribution is strongly affected
by instrumentation overhead: kernel regeneration is expensive enough to be
visible inside the instrumented cache, but reducing its direct cost does not
translate into a meaningful full-step improvement on the coarse paired timing.

The ladder is therefore **not promoted into the default cache path**.

The research-only candidate and failed official result are retained for
reproducibility.

## Engineering consequence

Do not continue selecting cache micro-optimizations solely from fine-grained
within-refresh timing shares.

The best next engineering experiment is a fresh **low-overhead fused-stage
crossover profile** on the current v1.17/v1.16 optimized path. It should
compare cache refresh, candidate readout, local readout, state scanning, and
field update using coarse timing boundaries and enough seeds to stabilize
ordering.

Only if cache refresh is still clearly dominant should another cache-write
optimization be attempted. If candidate readout is larger, engineering should
move there instead.

## Claim boundary

This result rejects the preregistered performance claim for the canonical
Python/NumPy kernel ladder. It does not imply the same result for Rust or
different vectorized math libraries.
