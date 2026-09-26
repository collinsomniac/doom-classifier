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

Current trainable parameters: **7,971**

Components:

- global encoder;
- temporal-delta encoder;
- shared variable-record encoder;
- mean/max set summary;
- state-conditioned action attention;
- typed action parameters;
- semantic-prior head plus a tiny semantic calibration residual;
- separate low-rank reward-specific value residual;
- three bootstrap value heads;
- delayed target network and n-step replay.

A regression test confirms reward-only learning does not alter the semantic-prior logits:

- semantic drift after reward TD: **0**
- value branch movement in fixture: **0.0333**
- combined-score movement: **0.0167**
- semantic teacher subsequently moves semantic branch: **0.1736**

## Fast-path benchmark

Latest GitHub Actions CPU stress sample:

- records: 768
- parameters: 7,971
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

## Progress and causal events

The project-owned runtime now exposes monotonic native counters for player-attributed hostile damage, player-attributed kills, successful pickups, level completions, and secret exits. The browser adapter differences those counters per transition before reward learning.

Spatial first-visit novelty remains a small task-agnostic exploration signal. The observation also exposes distinct visited cells, current-cell revisit count, and exploration novelty. No pathfinder or scripted navigation policy is used.

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

## Owned-runtime causal reward benchmark

The project-owned Chocolate Doom WASM runtime is now the default demo runtime. A Chromium benchmark using the same 7,316-parameter policy, MobileBERT semantic preparation, four-step consequence learning, adaptive teacher supervision, replay, and KL-constrained fusion produced the following on the starting encounter.

### Frozen before training

- return: **+2.526**
- player-attributed hostile damage: **65**
- player-attributed kills: **1**
- player damage received: **0**
- causal combat attribution available: **24 / 24 decisions**
- exact-action diversity: **1**
- dominant action: **back + fire**
- teacher calls during frozen evaluation: **0**

### 64-decision online training burst

- player-attributed hostile damage: **90**
- player-attributed kills: **3**
- successful pickups: **2**
- causal combat attribution available: **64 / 64 decisions**
- reward/value updates including replay: **189**
- adaptive teacher refreshes: **8**
- reward replay size: **64**
- action diversity: **15 / 15**
- switches: **57**
- maximum identical-action streak: **2**

### Frozen after training

- return: **+2.526**
- player-attributed hostile damage: **65**
- player-attributed kills: **1**
- player damage received: **0**
- teacher calls: **0**
- dominant action remains **back + fire**
- mean value-fusion beta: **16.59**
- mean prior KL: **0.0787**
- mean KL-budget utilization: **99.97%**
- p95 neural decision time: **5.80 ms**

The important result is not an improved score yet. It is that the consequence branch is now trained and evaluated against **engine-attributed causal combat events**, while the frozen fast path still runs without teacher calls. The remaining behavioral bottleneck is action separation: the value head shifts strongly during training, but KL-constrained fusion still preserves a dominant semantic-prior action in the frozen encounter.

## Repeated-rollout causal benchmark

A 256-decision fine-tune with a **64-decision rollout horizon** resets the environment three times during training so the learner sees repeated informative starting encounters instead of spending most of the longer run wandering after the initial combat.

On the 7,971-parameter reward-residual model:

- attributed hostile damage during training: **180**
- player-attributed kills: **4**
- causal attribution ticks: **256 / 256**
- value/replay updates: **765**
- action diversity: **15 / 15**
- rollout restarts: **3**
- teacher refreshes: **32**

This doubled attributed damage relative to the earlier 256-step continuous run (90) and increased kills from 3 to 4, confirming that bounded rollouts improve training-data density.

However, frozen performance still regressed:

- before training: **+2.236**, 55 attributed damage, 1 kill, dominant **forward + fire**
- after training: **+0.776**, 40 attributed damage, 0 kills, dominant **fire**
- value/prior top-action agreement: **0**
- mean post-training value top-two gap: only **~0.00045**
- mean fused ensemble disagreement: **~0.048**
- KL budget utilization: essentially **100%**

The important negative result is that more causal experience alone does not solve policy improvement. The critic can consume the full KL authority while its bootstrap members still disagree. This motivated a second, state-dependent trust constraint that bounds value authority by critic epistemic disagreement in addition to prior KL.

## Dual-trust policy fusion benchmark

The repeated-rollout experiment showed that a KL budget alone could grant full policy authority while the critic's bootstrap members still disagreed. The current fusion therefore constrains learned value by **two state-dependent trust boundaries**:

1. KL divergence from the semantic prior;
2. bootstrap-ensemble epistemic disagreement.

On the same 256-decision / 64-step-rollout setup:

### Frozen before training

- return: **+2.626**
- player-attributed hostile damage: **70**
- player-attributed kills: **1**
- dominant action: **fire**

### Training

- player-attributed hostile damage: **170**
- player-attributed kills: **2**
- causal attribution: **256 / 256 decisions**
- value/replay updates: **765**
- rollout restarts: **3**
- action diversity: **15 / 15**

### Frozen after training

- return: **+2.141**
- player-attributed hostile damage: **45**
- player-attributed kills: **1**
- dominant action: **forward + fire** (22 / 24)
- mean prior KL: **0.0330**
- mean KL-budget utilization: **41.3%**
- mean epistemic disagreement: **~0.0250**, at the configured critic trust cap
- mean value top-two gap: **~0.00077**
- p95 decision time: **~7.58 ms**

A budget sweep from 0.04 through 0.32 did **not** increase value authority: the epistemic limit bound first at beta ~222 and prior KL ~0.026. This is the intended behavior. Raising the semantic KL allowance cannot force an uncertain critic to dominate.

This is a stability improvement rather than a final performance win: frozen return still trails its own pre-training baseline, but the earlier post-training collapse to zero kills was avoided.

## Typed-action projection ablation

The current critic can project its per-candidate Q estimates onto the schema's typed action fields and blend that structured projection back into the critic scores before fusion. This is deliberately generic: the projection uses `schema.actionFields`, not DOOM-specific button names.

A fixed-policy Chromium sweep varied the maximum typed-value blend **after one 256-decision / 64-step-rollout training run**, so no retraining confounded the comparison.

| typed blend | frozen return | attributed damage | kills | dominant action |
|---:|---:|---:|---:|---|
| 0.00 | +2.034 | 40 | 1 | fire |
| 0.15 | +2.034 | 40 | 1 | fire |
| 0.30 | +2.034 | 40 | 1 | fire |
| 0.45 | +2.034 | 40 | 1 | fire |
| 0.65 | +2.034 | 40 | 1 | fire |

The mean critic top-two gap remained about **0.0040** throughout. In this particular trained state, post-hoc typed smoothing therefore did **not** change behavior. It remains useful for:

- exposing reusable primitive/field preferences;
- regularizing noisy exact-action values;
- transferring structure to schemas with combinatorial action menus;
- diagnosing whether compound-action disagreement is really field-level disagreement.

The same trained policy was also swept over semantic KL authority:

| KL budget | fused action |
|---:|---|
| 0.04 | fire |
| 0.08 | fire |
| 0.12 | fire |
| 0.16 | strafe right + fire |
| 0.24 | strafe right + fire |
| 0.32 | strafe right + fire |

The raw critic preferred **strafe right + fire**. At the first flip (~0.16 KL), fused ensemble disagreement was only about **0.0096**, below the current 0.025 epistemic ceiling. This isolates the remaining issue: the critic can earn enough confidence to disagree with the semantic prior, but the default semantic trust region may still be the active limiter in some states.

The next trust experiment should therefore be **state-dependent**, not a global KL increase. Candidate signals now exposed by the policy include bootstrap-head top-action agreement, top-vs-runner-up critic gap, and bootstrap margin signal-to-noise ratio.

## Confidence-gated semantic authority

A later 256-decision / four-rollout run kept the same 0.08 base semantic KL budget but allowed per-state expansion only when the bootstrap critic agreed on its ranking and the margin was stable relative to the critic's own value spread.

Compared with the immediately preceding fixed-budget run:

| frozen post-training | fixed 0.08 KL | confidence-gated KL |
|---|---:|---:|
| return | +0.676 | **+2.026** |
| attributed damage | 35 | **40** |
| player kills | 0 | **1** |
| teacher calls | 0 | **0** |

The confidence gate did not create broad action diversity—the dominant frozen action remained `fire`—but it recovered combat behavior without globally increasing semantic authority. The bootstrap/epistemic guard remains independent of the KL expansion.

The associated deterministic invariant gives a fully agreeing critic up to a 2x KL ceiling (0.08 -> 0.16), while a deliberately split bootstrap ensemble receives zero extra authority.

## Null-state NLI label-bias calibration

Generic MNLI teachers have strong action-label priors even when game evidence is withheld. A null-state calibration subtracts each action's logit on a state-withheld premise before centering the result.

For MobileBERT:

| metric | raw | null calibrated |
|---|---:|---:|
| aggregate FIRE, enemy ahead | 0.768 | 0.484 |
| aggregate FIRE, quiet room | 0.730 | 0.399 |
| enemy-minus-quiet FIRE separation | 0.038 | **0.085** |

The correction materially reduces MobileBERT's generic FIRE bias and improves state discrimination. DistilBERT did not benefit reliably, so the live path enables this calibration only for the default MobileBERT teacher rather than treating it as a universal NLI correction.

## What remains unproven

The strongest missing evidence is still:

1. multi-seed / multi-episode real DOOM performance;
2. actual level completion and navigation progress;
3. reliable semantic teacher affordance calibration;
4. teacher-query rate vs competence over longer training;
5. Brier / log-score / ECE calibration;
6. cross-map generalization;
7. broader non-DOOM transfer beyond the current workload-routing adapter;
8. controlled comparison against Laya/Jev on equivalent typed-decision tasks.

## Next experiments

- compare dual KL + epistemic trust against the repeated-rollout baseline;
- test schema-driven factored/typed action-value projection before integrating it into live fusion;
- expand real-DOOM evaluation across seeds/maps and level completion;
- train/distill a decision-specialized teacher rather than relying on generic MNLI;
- publish a validated portable tiny starter checkpoint and measure cold-start vs prepared performance;
- extend the existing non-DOOM workload-routing transfer test to additional structured tasks.
