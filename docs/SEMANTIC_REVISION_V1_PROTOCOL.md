# Semantic revision v1 — temporal label revision

Status: **frozen protocol / development preflight pending**.

Protocol ID: `semantic-revision-v1`.

## Question

Can Cortex maintain a persistent semantic label with rational inertia, resist a
single contradictory perceptual shock, withdraw the label under sustained
contradiction, adopt a better-supported replacement, and later recover the
original mapping when the evidence regime reverses?

This extends the passed `hybrid-semantic-grounding-v1` result from static
adjudication to chronological semantic belief revision.

## Invariant

Concept identity is independent of semantic state:

$$
Z_t = Z_{t+1}
$$

does not require

$$
\ell_t = \ell_{t+1}.
$$

A revision changes Cortex's accepted label, not the underlying concept.

## Temporal evidence

At each epoch Cortex may receive:

- a neural candidate-label distribution;
- typed structural observations.

Evidence is retained for a finite horizon of **12 epochs**. Within that horizon,
reliability decays exponentially:

$$
w_N(a)=0.72^a,
\qquad
w_R(a)=0.82^a,
$$

where $a$ is evidence age.

The finite horizon guarantees bounded revision memory. The unequal rates encode
the frozen v1 hypothesis that persistent structural evidence should decay more
slowly than instantaneous neural perception.

## Grounding posterior

Every epoch rebuilds the already validated finite semantic-grounding model from
the currently retained, age-weighted evidence. The underlying grounding
thresholds remain:

- label support >= **0.70**;
- best-vs-second margin >= **0.15**.

The semantic posterior and the accepted Cortex belief are deliberately distinct.

## Revision hysteresis

The accepted state follows:

$$
\text{stable}
\rightarrow
\text{challenged}
\rightarrow
\text{unresolved}
\rightarrow
\text{stable(new label)}.
$$

A stable incumbent is challenged whenever the current grounding posterior no
longer resolves to that incumbent.

### Withdrawal

A label is not withdrawn after one conflict. v1 requires **2 consecutive
conflicting epochs**.

After the second conflict:

- the incumbent becomes unaccepted;
- phase becomes `unresolved`;
- the old label is retained only as provenance.

### Replacement

While unresolved, a resolved challenger must persist for **2 consecutive
epochs** before becoming the new accepted label.

This prevents direct stable-to-stable semantic flips.

## Revision events

The controller records immutable semantic events:

- `accepted`;
- `challenged`;
- `challenge_cleared`;
- `withdrawn`;
- `revised`;
- `recovered`.

Each event records epoch, concept, old/new/candidate labels, posterior
confidence, margin, and reason.

## Worlds

The campaign reuses the three asymmetric typed semantic families from
`hybrid-semantic-grounding-v1`.

Every world contains:

- 5 opaque concepts;
- 5 disjoint opaque labels;
- 2 relation types;
- one base concept-label mapping.

A shifted semantic regime is constructed by swapping exactly one pair of label
assignments while keeping all other labels fixed.

The corresponding structural observations are regenerated from the same
external semantic graph under the shifted mapping. Thus only two concept labels
should revise.

## Chronological protocol

Each case performs:

1. **establishment:** 3 epochs of base neural + structural evidence;
2. **shock:** 1 epoch of contradictory neural evidence only;
3. **restabilization:** 1 base epoch;
4. **sustained shift:** up to 8 epochs of shifted neural + structural evidence;
5. **recovery:** up to 8 epochs of base neural + structural evidence.

## Required behavior

The single neural shock must not change any accepted label.

During sustained shift, both swapped concepts must pass through:

$$
\text{challenged}
\rightarrow
\text{withdrawn/unresolved}
\rightarrow
\text{revised}.
$$

The three unaffected concepts must never be withdrawn or revised.

When the base regime returns, the complete original mapping must recover.

## Development and official seeds

Development seeds: `0..19`.

Official untouched seeds: `400..419`.

There are 3 structural families, producing 60 development and 60 official
cases.

Official seeds may not be evaluated until development gates pass and this
protocol is frozen.

## Frozen gates

The campaign passes only if:

1. initial accepted mapping correctness = **1.0**;
2. single-shock preservation rate = **1.0**;
3. shifted-mapping revision success = **1.0**;
4. base-mapping recovery success = **1.0**;
5. valid challenged → withdrawn → revised path rate = **1.0**;
6. unaffected-concept preservation rate = **1.0**;
7. maximum shift latency <= **8 epochs**;
8. maximum recovery latency <= **8 epochs**;
9. neural history boundedness rate = **1.0**;
10. structural history boundedness rate = **1.0**.

No threshold may be weakened after the official campaign is observed.

## Claim boundary

A passing result would support that, within the declared finite semantic
fixtures, Cortex can maintain a temporally revisable semantic belief with
bounded fading evidence, resist one-off perceptual contradiction, pass through
explicit uncertainty before replacement, revise only affected labels under a
persistent regime change, and recover when the regime reverses.

It would not establish optimal Bayesian forgetting, real-world concept drift,
open-vocabulary revision, or trainable neural feedback.
