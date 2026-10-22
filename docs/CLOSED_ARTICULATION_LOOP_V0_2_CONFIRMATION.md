# Closed Articulation Loop v0.2 — Confirmation Amendment

Status: **frozen development confirmation panel**.

## Why this amendment exists

The first factorized v0.2 tuning campaign behaved correctly as a falsification
exercise: the tuning worlds favored a more aggressive evidence configuration
and a 128-basin memory, but the separate 12-world-per-stage holdout showed that
those choices did not generalize cleanly.

On that first holdout, the selected full candidate:

- improved OWR-B transition behavior substantially;
- reduced total research memory from 256 to 128 basins;
- but reduced OWR-A trap coverage relative to the v0.1 baseline;
- and reduced OWR-D query coverage relative to the v0.1 baseline.

Therefore the first-pass full candidate is **not promoted**.

The strongest generalizing signal was the temporal subsystem:

```text
fast_decay         0.50 -> 0.65
pressure_decay     0.90 -> 0.85
suspicion_ratio    0.05 -> 0.03
confirmation_steps 5    -> candidate {3,5}

recurrence_min_history 5 -> candidate 3
```

A third independent development material is now reserved for a small frozen
ablation panel before any default changes.

## Frozen confirmation panel

The panel contains exactly:

1. v0.1 baseline;
2. temporal core only, confirmation_steps = 3;
3. recurrence only;
4. temporal core + recurrence, confirmation_steps = 3;
5. temporal core + recurrence, confirmation_steps = 5;
6. the aggressive full v0.2 first-pass candidate as a control.

No other candidate is added after results are observed.

The confirmation material is:

```text
closed-loop-v0.2-confirmation-<stage>
```

with 20 development worlds per stage.

OWR-E and official OWR seeds remain untouched.

## Promotion gates

A non-baseline candidate may be promoted only if, relative to the baseline on
the same confirmation worlds:

### OWR-A

- copied-majority trap selective risk remains 0;
- trap coverage regresses by no more than 0.05 absolute;
- overall fresh-target selective risk remains <= 0.01.

### OWR-B

- transition wrong-answer rate is at most 75% of baseline;
- transition correct-answer rate is at least baseline;
- each settled-regime selective risk remains <= 0.03.

### OWR-C

- local relation coverage = 1;
- local resolved accuracy = 1;
- every world emits at least one multi-subject background candidate.

### OWR-D

- no research-layer hard failure;
- selective risk = 0;
- query coverage regresses by no more than 0.03 absolute.

## Selection

Among candidates satisfying every gate, selection is lexicographic:

1. lowest OWR-B transition wrong-answer rate;
2. highest OWR-B transition correct-answer rate;
3. highest OWR-A trap coverage;
4. highest OWR-D query coverage.

If no candidate passes all gates, v0.1 remains the default.

This amendment is development-only. A selected candidate is still not an
official OWR result.
