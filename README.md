# doom-classifier

Browser-native research harness for ultra-low-latency structured-state → typed-decision policies.

The working hypothesis is that a useful real-time controller can combine a cheap semantic prior, tiny temporal memory, a separable online experience residual, calibrated uncertainty, and a policy-blind runtime state machine.

DOOM is the first real target environment, not the architecture.

## Experiments

### Experiment 001 — synthetic decision lab

- deterministic browser arena;
- structural hash baseline for control measurements;
- opt-in MobileBERT-MNLI and DistilBERT-MNLI zero-shot semantic scorers through Transformers.js;
- temporal-delta memory;
- linear online Q residual that can be trained, frozen, or bypassed;
- entropy, margin and novelty signals;
- trace export plus policy p50/p95/p99 latency instrumentation.

Current deterministic benchmark: semantic-only mean reward/step -0.03711; frozen learned residual on a different seed -0.01105. See docs/results.md.

### Experiment 002 — real DOOM primitive policy

doom.html boots a pinned Chocolate Doom 3.1.1 + Freedoom 0.13.0 WebAssembly runtime and reads native structured telemetry. Unlike tactical-macro demonstrations, the policy may choose only player-level primitives: forward, back, turn, strafe, fire, use, or wait.

The real-engine page deliberately does not provide MOVE_TO_ENEMY, RETREAT, FACE_ENEMY, pathfinding, or hard-coded firing thresholds. See docs/doom-integration.md.

## Run

Serve the repository over HTTP because browser ES modules do not reliably run from file URLs:

    python -m http.server 8000

Open:

- http://localhost:8000/ for the synthetic lab;
- http://localhost:8000/doom.html for the real-engine experiment.

The real DOOM page downloads about 31 MB for its pinned engine/Freedoom runtime on first boot. Learned NLI backbones are separate opt-in downloads.

## Tests

    node tests/smoke.mjs
    node tests/doom-adapter.mjs
    node tests/benchmark.mjs

GitHub Actions runs syntax checks, the smoke test, the DOOM adapter test and the adaptation benchmark on every push.

## Research invariants

1. The state machine may understand program lifecycle, never strategy.
2. Environment adapters expose observations, mechanics and reward, never a behavioral policy.
3. The normal fast path remains one bounded decision pass.
4. Online adaptation is separable, tiny and disableable.
5. Teacher/escalation runs must remain distinguishable from teacher-off evaluation.
6. DOOM-specific signals must not leak into the core policy API.
7. Every meaningful component should be ablatable.

## Near-term sequence

1. Browser-test the real DOOM page across desktop and iOS.
2. Replace the fixed scalar entity projection with a bounded set/entity encoder.
3. Run MobileBERT/DistilBERT semantic-only vs semantic+residual comparisons in DOOM.
4. Build our own minimal telemetry-enabled Chocolate Doom artifact from pinned source.
5. Add uncertainty-triggered teacher collection and teacher-off evaluation.
6. Add natural-map episodes and cross-environment/schema-transfer benchmarks.

GitHub Pages deployment is already configured, but this newly created repository still requires the one-time Pages setting to be enabled for GitHub Actions before a public site URL exists.