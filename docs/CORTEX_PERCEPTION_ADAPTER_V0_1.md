# Perception adapter v0.1

## A concrete first task

Observe a short video containing two visually distinct objects. Preserve each
identity as objects move or briefly disappear, attach readable names, and record
directed spatial relation proposals. Measure identity switches, label ambiguity,
missed detections and incorrect relations separately. This is narrower than
unrestricted video understanding.

## Implemented boundary

`python/cortex/perception.py` defines a model-neutral `PerceptionProvider` protocol.
Its `perceive(image, frame_id=..., timestamp=..., source=...)` method returns a
`PerceptionFrame` containing:

- Frame identity, timestamp, source, model version and feature-space version.
- Detections with frame-local IDs, finite feature vectors and candidate names.
- Directed, typed relation proposals with endpoints and provider scores.

`PerceptionAdapter` passes feature vectors to the runtime and binds returned
persistent entity IDs to name evidence and relation endpoints. Names never enter
the matching features. Two objects with the same label remain separate entities.
A model must supply instance-discriminative features; category names alone cannot
track two similar objects. The native runtime must be configured for the declared
feature dimension and identity-only nuisance transforms unless the encoder's
transformations have a justified interpretation. Feature-space changes require an
explicit migration, not automatic reuse of existing prototypes.

The display policy takes the maximum provider score for each candidate text,
returns an opaque ID when competing texts have close scores, and always appends
the persistent ID to a displayed label. It retains provenance and alternatives.
Scores are not calibrated probabilities. Repeated correlated frames do not
multiply confidence. Alias, proper-name and category kinds are recorded, but
alias equivalence and name preference are not learned automatically. Conflicting
aliases may therefore yield a conservative opaque display. High-scoring old names
can remain influential; temporal label revision is future work.

Directed typed relations remain proposals. They are not silently passed through
the native generic undirected binary relation channel. Explicit
`accepted_statements(frame_id, proposal_ids)` exports selected proposals as
`KnowledgeStatement` objects using persistent entity tokens, preserving roles and
source/model provenance. Calling this method is an acceptance decision, not a
proof that a model's prediction is true. Outputs are not automatically inserted
into either the native graph or the research rule learner. Relation scores remain
in the proposal log; the statement format has no probability field.

Empty frames are recorded without advancing the native runtime, which has no
empty-frame/elapsed-time update API here. An unresolved runtime read does not
attach labels or bind relation proposals. Invalid inputs fail before runtime
mutation. Runtime exceptions or invalid bindings poison the adapter: callers must
restore/recreate the runtime rather than assume rollback. A frame limit bounds
history length; per-frame object/evidence size must also be bounded by the provider.

## Local video interface

`python/cortex/video_perception.py` implements bounded local decoding through
ffmpeg, then calls the supplied provider and adapter. NumPy and ffmpeg are optional
requirements for this path. There is no neural dependency or automatic download.
Sample timestamps are resampled offsets from zero, not original presentation
timestamps. Fixed-size resizing can change aspect ratio. Precise temporal or
geometric applications should replace this decoder with an appropriate one.

Example integration, after supplying a provider and a correctly configured runtime:

```python
from cortex.perception import PerceptionAdapter
from cortex.video_perception import ingest_video

adapter = PerceptionAdapter(
    runtime, feature_space="my-encoder:weights-v1", feature_dim=256
)
reads = ingest_video(
    "clip.mp4", provider, adapter,
    sample_fps=2, max_frames=60, width=320, height=240,
)
```

A neural provider should run an object detector, extract an appearance embedding
for each crop, and supply label candidates separately. Explicit relation proposals
may come from measured geometry or another model, with their origin recorded.
Do not put detector track IDs or labels into embeddings merely to guarantee stable
identity; that would conceal the task we need to evaluate.

## Completed test and remaining neural experiment

Run `PYTHONPATH=python python research/perception_video_demo.py`.
The demo writes and decodes an actual generated lossless clip using ffmpeg. A
classical color-segmentation provider extracts RGB features and names. The frozen
Python reference Cortex articulation core binds observations with identity-only
transformations. Detection order alternates; objects cross horizontally, one
briefly disappears, and one frame is empty.

Observed: 12 decoded frames, one empty frame, two persistent entities, zero
identity switches and both directions of the left-of relation. Unit tests also
cover equal names on distinct entities, conflicting names, malformed inputs,
feature-space changes, empty frames, unresolved reads and failed runtime outputs.

This is a controlled video plumbing test, not a neural or natural-video result.
No neural vision packages, cached weights, native extension or Rust compiler were
available in the execution environment. No pretrained detector was downloaded or
run. A supplied short video and an installed detector/appearance encoder are needed
for the next experiment. Evaluate identity and relation accuracy against manual
annotations before claiming real-video performance. Do not promote this fixture
into an autonomous world-understanding claim.
