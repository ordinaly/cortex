# General Knowledge Feed v0.1

Status: **experimental autonomous-learning input path**.

## Goal

Give Cortex persistent general knowledge without making a neural language model
part of the reasoning loop and without turning an external knowledge graph's
properties into hard Cortex predicates.

The first source adapter targets **Wikibase JSON**, with Wikidata as the initial
dataset. Wikidata's full JSON representation is preferred over its truthy RDF
projection because the full statements preserve multiple values, qualifiers,
rank, and references.

## Knowledge Packet

The neutral packet is:

```text
statement id
subject symbol
property symbol
value symbol
qualifiers[]
rank
references[]
provenance
```

Only the statement structure is interpreted by the adapter. Symbols such as
`wd:P36` or `capital` remain opaque identities.

## Statement grammar

Cortex's current native relation evidence is generic and undirected. Direction
and role are therefore represented by a small reified graph motif:

```text
                       @syntax:subject
                              |
subject --- subject-slot --- statement
                              |
property -- property-slot ----+
          |                   |
 @syntax:property             |
                              |
object ---- object-slot ------+
          |
   @syntax:object
```

The diagram is schematic: all three slots connect to the statement node.

These role markers are **syntax, not ontology**. They are analogous to field
positions in a record. Cortex is never told that a particular world property
must be functional, transitive, immutable, or otherwise constrained.

Qualifiers are represented as nested qualifier motifs attached to the statement.

## Identity encoding

The current adapter maps opaque symbols to deterministic 64-dimensional signed
hash vectors. The vector contains no lexical embedding and no pretrained
semantic similarity. Equal symbols yield identical observations; distinct
symbols are deliberately far apart under Cortex's MSE matching metric.

For structured claims, nuisance transforms are disabled with the identity group.

## Distilled-claim evidence mode

The native sensory graph normally waits for repeated observations before
resolving a relation. A knowledge packet is already a distilled discrete claim;
replaying it eight times would falsely pretend there were eight independent
observations.

The knowledge runtime therefore uses a separate graph configuration:

- relation minimum exposure: 1;
- active threshold: 0.65;
- absent threshold: 0.35;
- fixed nuisance group: identity only.

This changes how structural punctuation is admitted, not the meaning of the
world property itself.

## What is preserved outside the Cortex graph in v0.1

Rank, reference hashes, and provenance are retained in the Knowledge Packet but
are not yet used by Cortex's confidence model.

This is intentional. Source trust should become a separate epistemic mechanism
rather than being smuggled into relation strength.

## Wikidata adapter

`cortex.wikidata` parses Wikibase JSON claims into Knowledge Statements. It
preserves:

- multiple values for one property;
- qualifiers;
- normal/preferred rank;
- reference hashes;
- entity-valued claims;
- time, quantity, coordinate, monolingual-text and generic literal values;
- explicit `novalue` / `somevalue` states.

Deprecated statements are skipped by default but can be retained for revision
experiments.

`scripts/prepare_wikidata_knowledge.py` streams plain, gzip, or bzip2 Wikidata
JSON-array dumps and writes normalized JSONL packets without loading the dump
into memory.

## Pilot validation

The repository smoke campaign feeds 22 general-knowledge statements spanning
geography, astronomy, chemistry and a deliberately multi-valued language fact.

Hard gates:

- zero symbol-binding conflicts;
- exact retrieval of held-in-memory facts;
- empty negative-control query;
- both values of a multi-valued property survive;
- query results are invariant to statement presentation order;
- one Cortex entity per unique packet symbol.

This is a **storage/retrieval test**, not yet a learning or inference claim.

## Pilot result

The first CI campaign passed every frozen gate:

- 22 statements presented once each;
- 129 unique packet symbols;
- 129 Cortex entities;
- 198 active structural relations;
- symbol-binding conflicts: **0**;
- exact retrieval: **12/12 (1.000)**;
- negative-control query: **empty**;
- multi-valued relation preservation: **passed**;
- shuffled-presentation query equivalence: **passed**.

The two feed orders completed in approximately 12.7 ms and 11.6 ms on the
hosted runner. These timings are descriptive only and are not performance
claims.

The result establishes that the reified grammar can survive Cortex articulation
and be read back from Cortex's own relation snapshot. It does not establish
held-out inference, source-trust reasoning, or large-scale knowledge ingestion.

The next experiment should hold statements out and ask whether Cortex can infer
new relational structure from what it was shown.


## Local informational articulation

The knowledge feed now exposes the same stored Cortex state through a sparse
local articulation rather than only exact graph lookup.

For a query point (x), entity support is derived from the native articulation
metric:

```math
d_i(x) = min_g weighted_mse(x, g p_i; w_i)
```

and normalized locally as:

```math
s_i(x) = exp(-beta d_i(x)) / sum_j exp(-beta d_j(x)).
```

These values are called **supports**, not calibrated probabilities.

A `LocalArticulation` contains:

- the strongest local entity supports;
- per-feature prototype value, reliability, and noise state;
- adjacent active/unresolved relation evidence with Beta posterior mean;
- adjacent causal evidence;
- normalized entity-support entropy;
- residual support outside the returned sparse top-k set.

For a knowledge symbol, `KnowledgeGraphView.local_articulation(...)` adds
decoded `(property, value, structural_support)` coordinates obtained from the
same reified statement motifs already stored in Cortex. No second semantic fact
store is created.

This makes the intended readout explicit:

```text
topological point
  -> fuzzy entity support
  -> local feature reliability
  -> local relation/causal evidence
  -> optional decoded knowledge coordinates
```

The readout is strictly non-mutating.
