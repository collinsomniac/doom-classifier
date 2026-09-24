# doom-classifier

Browser-native research harness for **ultra-low-latency structured-state → typed-decision neural policies**.

Live lab: https://collinsomniac.github.io/doom-classifier/

DOOM is the first real harness, not the architecture. The project is testing whether a very small neural controller can consume native structured state, score a request-time set of typed actions with useful uncertainty, learn consequences online, and borrow semantic knowledge from a larger model without paying that model's latency on every control tick.

## Current architecture

The real-time controller now combines:

- **schema-conditioned numeric encoding** — static field/action metadata is compiled once;
- **optional MiniLM semantic compilation** — a ~23 MB int8 sentence model embeds the objective, fields, collections, action fields and action descriptions once, then can be disposed;
- **variable-record neural set encoding** — globals plus arbitrary collections such as actors, geometry, candidates or events;
- **temporal channel** — recent normalized state deltas are encoded separately from instantaneous state;
- **action-conditioned attention** — each candidate action asks a different question of the record set;
- **typed numeric action parameters** — candidate values such as coordinates, strength, price, threshold, velocity or other numbers are part of the action representation without changing network shape;
- **three-head bootstrap value ensemble** — shared encoder/attention, independent scalar heads for an epistemic disagreement signal;
- **online reward learning** — tiny bootstrapped TD updates adapt the controller to actual consequences;
- **adaptive semantic teacher** — MobileBERT-MNLI / DistilBERT-MNLI can supervise only on bootstrap, periodic refresh, novelty, ambiguity or ensemble disagreement;
- **teacher distillation** — expensive semantic calls update the tiny controller instead of becoming a permanent inference dependency;
- **policy-blind orchestration** — runtime state machines handle lifecycle/timing but contain no gameplay strategy.

The fast network currently has **6,803 trainable parameters**. Schema/action cardinality does not change that parameter count.

## Experiments

### Experiment 001 — synthetic decision lab

The synthetic arena is the controlled ablation surface. It supports:

- semantic-only, neural-only and hybrid policies;
- frozen vs online adaptation;
- temporal memory on/off;
- action/schema reordering and renaming tests;
- action-cardinality scaling;
- trace export and latency distributions.

An early linear-residual baseline improved from -0.03711 mean reward/step to -0.01105 when frozen and evaluated on a different seed. That result is retained as a historical floor, not the current architecture.

### Experiment 002 — real DOOM primitive policy

`doom.html` boots a pinned Chocolate Doom 3.1.1 + Freedoom 0.13.0 WebAssembly runtime and reads native structured telemetry.

The controller sees:

- global player/resource state;
- variable world-entity records;
- variable map-geometry records;
- recent temporal deltas.

It may choose only player-level primitives:

- forward / back;
- turn left / right;
- strafe left / right;
- fire;
- use;
- wait.

There is deliberately no `MOVE_TO_ENEMY`, `RETREAT`, `FACE_ENEMY`, pathfinding policy, or distance-triggered firing rule. Those would move competence out of the learned policy and into deterministic software.

The live page also exposes chosen-action record attention so we can inspect which native records most influenced the currently selected action.

## Fast-path measurements

Current GitHub Actions CPU benchmark, 768 structured records and 12 actions:

- p50: ~5.49 ms
- p95: ~7.42 ms
- parameters: 6,803

Action-cardinality benchmark on the same tiny fixed-size network:

- 8 actions: ~1.10 ms p50
- 128 actions: ~2.88 ms p50
- 512 actions: ~9.27 ms p50
- 1,024 actions: ~17.82 ms p50

These are engineering microbenchmarks, not a claim of equivalent accuracy to Jev or Laya. The important result is that native structured state can be scored without serializing the changing environment back through a large language encoder every tick.

## Semantic compile path

The default fast schema representation includes a cheap lexical feature-hash channel.

The optional **MiniLM schema compiler** adds sentence-level semantic vectors for static metadata. It batch-embeds schema/action language, compresses those vectors into the controller's fixed schema space, reconfigures the policy, clears incompatible learned weights, and disposes the compiler model.

This gives the fast controller a way to begin with useful relationships between paraphrased concepts while preserving a tiny per-tick network.

## Confidence / escalation

The lab intentionally separates several notions that are often incorrectly called "confidence":

- **entropy** — how diffuse the selected policy distribution is;
- **margin** — top-1 vs top-2 action separation;
- **novelty** — online deviation from previously observed scalar state;
- **epistemic disagreement** — Jensen-Shannon-style disagreement among bootstrap value heads.

Adaptive teacher scheduling can use all four signals. Teacher-assisted traces remain distinguishable from teacher-off evaluation.

## Run locally

Serve the repository over HTTP:

    python -m http.server 8000

Open:

- http://localhost:8000/ — synthetic lab
- http://localhost:8000/doom.html — real-engine lab

The DOOM page downloads the pinned engine/Freedoom runtime on first boot. Learned semantic components are opt-in downloads.

## Tests

Important CI checks include:

    node tests/smoke.mjs
    node tests/doom-adapter.mjs
    node tests/neural-set.mjs
    node tests/neural-temporal.mjs
    node tests/epistemic-ensemble.mjs
    node tests/action-attention.mjs
    node tests/schema-transfer.mjs
    node tests/semantic-vectors.mjs
    node tests/typed-action-params.mjs
    node tests/action-cardinality.mjs
    node tests/adaptive-teacher.mjs
    node tests/nli-state-summary.mjs
    node tests/benchmark.mjs

A separate Playwright workflow boots real DOOM in Chromium, activates MobileBERT, verifies a neural-fast control tick, and exercises the compile-time semantic path.

## Research invariants

1. The state machine may understand program lifecycle, never strategy.
2. Environment adapters expose facts, mechanics and reward, never a behavioral policy.
3. Static semantics should be compiled; changing numeric state should dominate per-tick compute.
4. Request-time action sets and numeric action parameters must not require changing model size.
5. Online adaptation must remain tiny, separable and disableable.
6. Teacher calls must never become an invisible requirement for evaluation.
7. Confidence must distinguish policy ambiguity from epistemic uncertainty.
8. DOOM-specific signals must not leak into the core policy API.
9. Meaningful components must remain ablatable.
10. Transfer tests should prefer semantic equivalence over identical field names.

## Next research steps

The highest-value next work is:

1. train/evaluate the neural controller across many real DOOM episodes rather than smoke-test steps;
2. add proper calibration metrics (ECE/Brier/log score) and calibrate ensemble/teacher probabilities;
3. build our own minimal telemetry-enabled engine artifact from pinned source instead of borrowing a neighboring runtime binary;
4. add identity-aware temporal tracking for variable entities, not only global temporal deltas;
5. create non-DOOM structured environments to measure true zero-shot schema/action transfer;
6. distill teacher-generated supervision offline into a portable tiny checkpoint;
7. compare accuracy, calibration, throughput and teacher-query rate against Laya/Jev-style typed-decision baselines.
