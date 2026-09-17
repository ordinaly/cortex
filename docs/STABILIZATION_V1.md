# Cortex stabilization-v1

## Question

`stabilization-v1` asks a narrower and more demanding question than the earlier
characterization campaign:

> Does repeated chronological experience cause Cortex to settle onto persistent
> structure, while still reactivating old concepts and responding locally to
> genuine novelty?

The campaign deliberately separates two different notions of stability:

1. **articulation stability** — whether repeated observations of the same latent
   entity keep the same entity identity under nuisance transforms and noise;
2. **continual-regime stability** — whether the continual learner stops creating
   regimes once a recurring stream is familiar, while retaining reactivation and
   novelty sensitivity.

This is a controlled synthetic prerequisite for a future real camera/sensor
stream. It is not evidence that Cortex forms human-like concepts from raw real
world perception.

## Provenance

Successful research run: GitHub Actions run `35207624460`.

Scientific source commit:

```text
9121f587d5c0859240f51b325d4a5c554e9d0b2d
```

Protocol: `stabilization-v1`  
Cortex: `1.0.0-rc.2`  
Python package: `1.0.0rc2`  
Specification: `spec-v0.9.5`  
Python: `3.13.15`

The campaign contains 15 conditions x 3 deterministic repetitions = 45 result
rows. `PYTHONHASHSEED`, OpenMP, OpenBLAS, and MKL thread counts are fixed in CI.
The benchmark records the exact Git SHA and configuration hash in every row.

An earlier harness run failed because the benchmark attempted to read
`active_entities` from `CortexRuntimeRead`, where that field is not exposed.
No scientific conclusion was taken from that failed run. The harness was fixed
to read the authoritative runtime snapshot and the full campaign was rerun.
The successful run and the normal Cortex CI both pass on the corrected source.

## 1. Familiar-world entity stabilization

Eight latent entities were observed for 12,000 frames, four at a time, under the
four-element cyclic nuisance group. Entity-count increments were sampled every
2,000 frames.

### Clean stream

All three repetitions produced exactly:

```text
new entities per 2,000-frame window = [8, 0, 0, 0, 0, 0]
```

The final learned entity count was exactly 8 in every run. Purity, dominant
identity share, and nuisance-subgroup recovery were all 1.0; fragmentation was
1.0 and identity switch rate was 0.

This is the desired stabilization signature for this fixture: the model forms
the required identities early and then continues updating without creating new
entity structure.

### Persistent 2% sparse corruption

With the same Gaussian noise (`sigma = 0.05`) plus 2% sparse feature corruption,
Cortex did **not** stabilize its entity inventory.

Across the three runs, final active entity counts were:

```text
23, 29, 24
```

for only 8 true entities. Median fragmentation was 3.125 identities per true
entity over the full stream and 2.625 over the final 4,000 frames. Median late
new-entity creation was 4 additional entities in the second half of the run.
Despite this, median purity remained effectively 1.0 and the nuisance subgroup
was recovered correctly in every measured frame.

The important distinction is therefore:

```text
semantic assignment is mostly correct
but structural inventory does not converge under persistent sparse corruption
```

This independently reproduces the false-split bias identified by
`characterization-v1` and shows that it persists over a substantially longer
stream rather than washing out with more experience.

## 2. Return after long absence

After 5,000 frames of familiarization, one target entity disappeared while the
remaining seven entities continued to be observed. Absence lengths were 100,
1,000, 5,000, and 20,000 frames. The target was then forced visible for 300
frames.

For every gap length and all three repetitions:

- the very first return was bound to the pre-gap identity;
- the same pre-gap identity was used for 100% of the 300 return observations;
- no entity was created during the absence;
- no entity was created on return;
- the total entity inventory remained exactly 8.

This establishes persistent identity reuse for the tested fixture, including a
20,000-frame absence.

### Important limitation

The articulation layer currently retains entity records indefinitely and has no
entity deletion/decay path. Therefore the 20,000-frame result should **not** be
interpreted as a measured forgetting horizon. It verifies that matching still
selects the old representation after a long intervening stream, but persistence
of the stored representation itself is architectural in the current release.
A future memory-limited entity lifecycle would make the time horizon itself a
nontrivial measurement.

## 3. Localized novelty and restabilization

After 6,000 frames containing eight familiar entities, a ninth latent entity was
introduced for 500 frames and the stream then continued for another 4,000 frames
with all nine entities.

All three runs changed the structural inventory from exactly 8 to exactly 9 and
remained at 9 thereafter. No old familiar identity changed during the novelty
period, and no further entities were created in the 4,000-frame settling phase.

Novel-identity creation delays were:

```text
0, 0, 6 observations
```

Two runs used a single identity for the novel entity from introduction onward.
One run briefly used two identities during the introduction, but the post phase
still had fragmentation 1.0 and dominant share 1.0 for the novel entity.

The tested behavior is therefore close to the desired local response:

```text
stable 8-entity world -> one localized novelty -> stable 9-entity world
```

rather than a global restructuring of already learned identities.

## 4. Transient corruption and structural debris

The previous campaign showed that persistent sparse corruption causes
fragmentation. This campaign asks a sharper question: if corruption is only a
short transient burst, does clean experience afterwards repair the structure?

Eight identities were first stabilized for 5,000 clean frames. One target entity
was then corrupted for a burst of 1, 5, 20, or 100 consecutive observations,
followed by 1,000 clean target observations.

### 10% sparse corruption, scale 1.25

| Burst length | Median new identities during burst | Median permanent excess identities | Median fraction using original ID during burst | Clean recovery |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0 | 0 | 1.00 | 1.00 |
| 5 | 1 | 1 | 0.80 | 1.00 |
| 20 | 1 | 1 | 0.90 | 1.00 |
| 100 | 4 | 4 | 0.94 | 1.00 |

### 20% sparse corruption, scale 1.25

| Burst length | Median new identities during burst | Median permanent excess identities | Median fraction using original ID during burst | Clean recovery |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0 | 0 | 1.00 | 1.00 |
| 20 | 3 | 3 | 0.85 | 1.00 |

In every run, the **first clean observation after the corruption burst returned
to the original identity**, and all 1,000 clean recovery observations used that
original identity. However, every false identity spawned during corruption
remained in the entity inventory: the number of permanent excess identities was
exactly the number created during the burst in these experiments.

This gives a very precise diagnosis:

```text
behavioral recovery is immediate
but structural recovery is absent
```

The current articulation implementation can return to the correct identity, but
it cannot retrospectively remove or reconcile false entities because entity
merge/reconciliation does not yet exist at this layer.

That is stronger evidence for the earlier design conclusion:

> uncertainty should not automatically become permanent ontology.

## 5. Continual-regime stabilization over 63,000 steps

The continual core was tested separately so entity fragmentation could not hide
its own behavior. Five well-separated 12-D latent regimes were used.

Each run contained:

1. 40,000 steps cycling four familiar regimes;
2. 10,000 steps with regime 0 absent while regimes 1-3 continued;
3. 2,000 steps with all four original regimes again;
4. 1,000 steps containing a genuinely novel fifth regime;
5. 10,000 steps cycling all five regimes.

The complete experiment was repeated three times at noise `sigma = 0.03` and
three times at `sigma = 0.12`.

### Result

All six runs produced the same structural behavior.

During the first 5,000-step window Cortex discovered four regimes. Every
subsequent 5,000-step window before the novelty event contained **zero new
discoveries**, while reactivation and revision continued.

The representative discovery trace is:

```text
window end       5k  10k  15k  20k  25k  30k  35k  40k  45k  50k  55k  60k
new discoveries   4    0    0    0    0    0    0    0    0    0    1    0
stored regimes    4    4    4    4    4    4    4    4    4    4    5    5
```

The first observation of the long-absent regime was a **reactivation**, not a
discovery, in all six runs. The 2,000-step return phase contained zero
discoveries and 250 reactivation events per run.

The genuinely novel fifth regime was detected on its first observation in all
six runs and caused exactly one discovery. The subsequent 10,000-step settling
phase contained zero further discoveries.

Final values in every run were:

```text
stored regimes       = 5
cumulative discoveries = 5
unresolved            = 0
budget pressure       = 0
split promotions      = 0
merge promotions      = 0
```

Cumulative reactivation count was 7,746.

### What this does establish

For this well-separated stationary/recurring regime family, Cortex exhibits the
behavioral stabilization signature we wanted:

```text
familiarization -> zero structural discovery churn
                 + continuing reactivation/revision
long absence    -> reactivation without rediscovery
true novelty    -> one localized discovery
post novelty    -> zero discovery churn again
```

This is a concrete finite demonstration of **semantic stability with statistical
plasticity** in the continual core.

### What this does not establish

The fixture is intentionally easy to identify: regimes are well separated and
the noise model is controlled. The run does not prove convergence for arbitrary
streams, does not prove that the five learned regimes correspond to useful
real-world concepts, and does not test raw perception. The deterministic event
counts also reflect the block schedule used by this benchmark; they should not
be interpreted as universal rates.

## Main scientific interpretation

`stabilization-v1` sharpens the architecture's strengths and its current missing
piece.

At the continual-regime level, Cortex already demonstrates the desired pattern:
structural growth stops once recurring regimes are familiar, old regimes are
reactivated after long absence, and new structure is introduced locally when a
new regime appears.

At the articulation/entity level, the same claim is true in clean streams but
breaks under sparse corruption. More experience does not automatically repair
false entity creation. Cortex rapidly resumes using the correct identity once
clean evidence returns, yet erroneous identities remain permanently stored.

The central empirical distinction is therefore:

```text
Cortex can restabilize behavior after transient corruption
without yet restabilizing ontology.
```

This identifies a concrete architectural research target: **provisional entity
hypotheses and/or evidence-driven entity reconciliation**. Simply increasing
`max_entities` would hide the symptom, consume more compute, and leave the
scientific problem untouched.

## Next discriminating experiments

The next implementation experiment should compare, under the same frozen burst
protocol:

1. current immediate spawn semantics;
2. provisional identities requiring persistence before promotion;
3. retrospective evidence-driven entity merge/reconciliation;
4. a robust observation distance or outlier gate that distinguishes transient
   sparse corruption from persistent identity change.

Any added mechanism should be accepted only if it lowers permanent false
fragmentation without increasing false merges or suppressing genuine novelty.

After that controlled ablation, the same stabilization measurements should be
ported to a **real chronological stream**: first a controlled camera/workbench
recording with hidden ground-truth object IDs, then an untouched external
sequence. The key real-world plots should remain the same: entity/concept count,
new-structure rate, reactivation rate, fragmentation, unresolved rate, and
recovery after absence/novelty/corruption.
