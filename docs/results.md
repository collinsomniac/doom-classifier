# Experiment results

## Status

The project now has two distinct evaluation surfaces:

- **real DOOM** — primary product/behavior benchmark;
- **synthetic arena** — legacy controlled ablation benchmark.

The current system does not yet establish that it beats Jev or Laya on decision quality. It does establish a small, browser-native structured-state controller with real engine actuation, semantic supervision, online consequence learning, teacher-off evaluation, and single-digit-millisecond CPU inference in the current stress benchmark.

## Historical synthetic baseline — 2026-09-23

The first scaffold used an intentionally weak hash semantic scorer plus a linear residual learner.

| phase | steps | residual training | exploration | mean reward / step | mean episode return |
|---|---:|---|---|---:|---:|
| semantic-only baseline | 6,000 | off | off | -0.03711 | -3.4787 |
| online adaptation | 12,000 | on | on | -0.03022 | -2.4824 |
| frozen learned residual, different seed | 6,000 | off | off | -0.01105 | -1.4714 |

This justified retaining a separable consequence-learning path. It is not the current architecture.

## Real browser integration

Playwright verifies:

1. real Chocolate Doom + Freedoom boot;
2. native structured telemetry;
3. real engine FIRE actuation;
4. weapon/category semantics;
5. compiled MiniLM schema semantics;
6. semantic teacher bootstrap/distillation;
7. tiny neural fast-path action;
8. attention inspection;
9. teacher-off control after preparation.

The page may also mount a user-owned Doom IWAD locally in browser memory.

## Current fast model

Model: **SchemaSemanticValueSetNet**

Current trainable parameters: **7,316**

Components:

- global encoder;
- temporal-delta encoder;
- shared variable-record encoder;
- mean/max set summary;
- state-conditioned action attention;
- typed action parameters;
- semantic-prior head;
- separate consequence-value hidden branch;
- three bootstrap value heads.

A regression test confirms reward-only learning does not alter the semantic-prior logits:

- semantic drift after reward TD: **0**
- value branch movement in fixture: **0.0333**
- combined-score movement: **0.0167**
- semantic teacher subsequently moves semantic branch: **0.1736**

## Fast-path benchmark

Latest GitHub Actions CPU stress sample:

- records: 768
- parameters: 7,316
- p50: **7.45 ms**
- p95: **9.82 ms**

CI runners are noisy; earlier split-head samples were somewhat faster. These measurements are engineering timing samples, not hardware-normalized model benchmarks.

### Candidate-cardinality benchmark

| candidates | p50 | p95 | parameters |
|---:|---:|---:|---:|
| 8 | 1.18 ms | 2.47 ms | 7,316 |
| 32 | 1.45 ms | 2.66 ms | 7,316 |
| 128 | 2.95 ms | 3.41 ms | 7,316 |
| 256 | 5.32 ms | 5.97 ms | 7,316 |
| 512 | 9.37 ms | 12.95 ms | 7,316 |
| 1,024 | 17.61 ms | 20.00 ms | 7,316 |

Candidate count changes compute but not model size.

## Real-DOOM fine-tune benchmark

The one-off Chromium benchmark performs:

1. Prepare Recommended;
2. frozen teacher-off evaluation for 24 decisions;
3. 64 real environment training decisions;
4. frozen teacher-off evaluation for another 24 decisions.

### Split semantic/value architecture + independent NLI teacher

Pre-training frozen evaluation:

- reward: **+2.526**
- hostile damage: **65**
- kills: **1**
- player damage received: **0**
- fire-capable actions: **24 / 24**
- action diversity: **1**
- dominant action: **back + fire**
- mean top probability: **0.106**
- mean normalized entropy: **0.987**
- p95 neural decision time: **5.89 ms**

64-decision training phase:

- neural updates including replay: **189**
- teacher calls: **8**
- replay size: **64**
- training return: **+5.246**
- target syncs after run: **8**

Post-training frozen evaluation:

- reward: **+2.526**
- hostile damage: **65**
- kills: **1**
- player damage received: **0**
- fire-capable actions: **24 / 24**
- action diversity: **1**
- dominant action: **back + fire**
- mean top probability: **0.127**
- mean normalized entropy: **0.976**
- p95 neural decision time: **6.50 ms**

Interpretation:

- this is materially better than the earlier conflated-head controller, which could collapse into backing/strafe behavior with zero firing;
- semantic/value separation preserves combat behavior through reward fine-tuning;
- exact action separation remains weak;
- the next target is not “make it shoot” but “make related movement+fire choices state-sensitive and less sticky.”

This benchmark predates the newest entity/projectile naming and spatial-novelty additions; a fresh rerun is the next comparison.

## Teacher diagnostics

### Token-budget failure discovered

An early MobileBERT probe produced byte-identical action distributions for:

- an enemy directly ahead;
- an empty quiet room.

Cause: verbose scalar descriptions exhausted the NLI token budget before world-entity collections.

Fix:

- compact scalar serialization;
- preserve categorical/binary collection facts;
- place non-empty collection summaries inside the premise budget.

After the fix, the teacher became state-sensitive.

### Current MobileBERT limitation

Even after evidence-conditioned wording and independent NLI entailment scoring, MobileBERT still strongly associates FIRE with the standing “neutralize threats” objective in deliberately empty-room probes.

One measured independent-entailment probe:

| probe | aggregate P(fire) |
|---|---:|
| hostile directly ahead | ~0.818 |
| low health / under fire | ~0.835 |
| quiet room / no entities | ~0.871 |

This is a useful negative result: MobileBERT-MNLI is a semantic prior, but not yet a reliable state-action affordance judge.

No “if no enemy -> forbid FIRE” rule has been introduced to hide this weakness.

DistilBERT and DeBERTa-v3-xsmall browser probes are being evaluated as heavier teacher alternatives.

## Typed primitive actions

The DOOM action set uses literal engine inputs, including movement+fire combinations.

Each compound candidate also has structured primitive fields. The UI therefore reports marginal probabilities such as:

- P(fire)
- P(strafe)
- P(turn)
- P(forward)
- P(back)
- P(use)

This prevents a broad “shoot while moving” intent from looking artificially indecisive merely because probability is split among several compound candidates.

## Entity semantics

The adapter now preserves more factual engine meaning:

- equipped weapon names;
- named Chocolate Doom mobj categories where known;
- hostile actor;
- collectible item;
- projectile / attack effect;
- other object.

These categories affect both schema-compiled neural representations and teacher summaries.

They are factual telemetry, not strategy.

## Progress signal

The borrowed bridge does not expose explicit level completion progress.

The current interim reward includes a small first-visit bonus for new coarse player-position cells.

The observation also exposes:

- distinct visited cells;
- current-cell revisit count;
- exploration novelty.

This supplies a route-agnostic progress signal without a pathfinder or scripted navigation policy.

## Temporal and attention invariants

Tests establish that:

- memory-enabled recent state changes alter neural scores;
- memory-disabled mode zeros the temporal channel;
- record order does not alter set behavior;
- the same action can attend to different records when global state changes;
- action attention weights remain normalized.

Example deterministic attention shift after changing only global state: ~0.025 maximum record-weight change.

## Target network

The value learner uses a delayed target network.

Tests establish:

- online/target value estimates diverge between syncs;
- target parameters move on the configured interval;
- sync difference is zero immediately after target copy.

Teacher refreshes synchronize semantic/shared target parameters only; they do not prematurely copy the delayed target value branch.

## Deployment

Live:

https://collinsomniac.github.io/doom-classifier/

The current main revision has successful full validation/deployment; the immediately preceding functional JavaScript revisions also pass the real Chromium smoke suite.

## What remains unproven

The strongest missing evidence is still:

1. multi-seed / multi-episode real DOOM performance;
2. actual level completion and navigation progress;
3. reliable semantic teacher affordance calibration;
4. teacher-query rate vs competence over longer training;
5. Brier / log-score / ECE calibration;
6. cross-map generalization;
7. non-DOOM transfer;
8. controlled comparison against Laya/Jev on equivalent typed-decision tasks.

## Next experiments

- rerun real-DOOM fine-tune benchmark with named entity/projectile semantics + spatial novelty;
- compare MobileBERT, DistilBERT and DeBERTa teacher state sensitivity;
- add previous-action / identity-aware record recurrence;
- build a project-owned telemetry engine artifact with level-completion, item/secret, projectile and damage-attribution events;
- train/distill a decision-specialized teacher rather than relying on generic MNLI;
- create a portable tiny checkpoint and measure cold-start vs prepared performance.
