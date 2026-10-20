# Higher-Order Irreducibility v0.9 — Results

## Outcome

**PASS**

The benchmark separates a genuinely three-way distinction from an otherwise identical complete pairwise boundary.

## Development seeds 0–11

### Positive noisy parity control

- complete K3 boundary: **12/12**
- all three probes retained: **12/12**
- filled 2-simplex: **12/12**
- mean joint distance for parity contexts: **0.7979**
- mean boundary distance for the same contexts: **0.0768**
- mean irreducibility score: **0.7210**
- minimum score: **0.6680**

### Negative reducible control

- complete K3 boundary: **12/12**
- all three probes retained: **12/12**
- false filled 2-simplex: **0/12**
- mean joint distance: **0.0853**
- mean boundary distance: **0.0732**
- mean irreducibility score: **0.0120**
- maximum score: **0.0234**

## Holdout seeds 12–23

### Positive noisy parity control

- complete K3 boundary: **12/12**
- all three probes retained: **12/12**
- filled 2-simplex: **12/12**
- mean joint distance for parity contexts: **0.8040**
- mean boundary distance: **0.0723**
- mean irreducibility score: **0.7318**
- minimum score: **0.6875**
- maximum score: **0.7773**

### Negative reducible control

- complete K3 boundary: **12/12**
- all three probes retained: **12/12**
- false filled 2-simplex: **0/12**
- mean joint distance: **0.0915**
- mean boundary distance: **0.0762**
- mean irreducibility score: **0.0140**
- maximum score: **0.0391**

The reference activation threshold is `0.25`, leaving a large empirical margin between positive and negative cases.

## Clique-completion ablation

A naive rule that fills every complete three-edge graph would activate a triangle in **100% of negative holdout runs**, because the K3 boundary is deliberately identical in both controls.

Thus the higher-order residual is doing necessary work: `three edges != filled triangle`.

## Fixed-boundary transition

The same three vertices and same three pairwise edges were preserved while synchronized evidence changed:

- initial synergistic regime: **3 edges, 1 triangle**
- reducible regime: **3 edges, 0 triangles**, residual **0.0039**
- restored synergistic regime: **3 edges, 1 triangle**, residual **0.7422**

This demonstrates a topology change confined to dimension 2: the 1-skeleton remains unchanged while the coface appears and disappears with genuinely collective evidence.

## Interpretation

v0.9 establishes a conservative operational sufficient condition for higher-order composition. It does not prove that the residual is a unique or canonical measure of synergy, nor that all meaningful higher-order structure must satisfy pairwise invisibility. The result supports the narrower architectural claim that Cortex can distinguish a triangle boundary from a filled 2-simplex using learner-visible synchronized evidence.
