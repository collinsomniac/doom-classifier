# Experiment results

## Historical synthetic baseline — 2026-09-23

The first scaffold used an intentionally weak hash semantic scorer plus a linear residual Q learner.

| phase | steps | residual training | exploration | mean reward / step | mean completed-episode return |
|---|---:|---|---|---:|---:|
| semantic-only baseline | 6,000 | off | off | -0.03711 | -3.4787 |
| online adaptation | 12,000 | on | on | -0.03022 | -2.4824 |
| frozen learned residual, different seed | 6,000 | off | off | -0.01105 | -1.4714 |

This result justified keeping a separable consequence-learning path. It is no longer the main model.

## Real DOOM browser integration

Playwright CI verifies the browser chain:

1. static `doom.html` loads;
2. pinned Chocolate Doom + Freedoom downloads and instantiates;
3. native structured telemetry is available;
4. the neural controller scores primitive actions;
5. a selected primitive advances the real engine;
6. MobileBERT-MNLI loads through Transformers.js;
7. the teacher is distilled before control;
8. a subsequent control tick stays on the neural fast path.

The test is intentionally an integration proof, not a gameplay score.

## Neural set controller

Current fast model features:

- shared global encoder;
- learned temporal projection;
- shared record encoder;
- mean/max set summary;
- action-conditioned record attention;
- typed numeric action representations;
- three bootstrap value heads.

Current parameter count in CI: **6,803**.

### Large-state microbenchmark

Synthetic input:

- 768 variable records;
- 12 typed actions.

Latest GitHub Actions CPU sample:

- p50: **5.49 ms**
- p95: **7.42 ms**

Earlier precompiled-hash optimization reduced the same class of workload from roughly 20 ms to ~5 ms by moving schema text processing out of the tick loop.

### Action-cardinality microbenchmark

Latest CI sample:

| actions | p50 | p95 | parameters |
|---:|---:|---:|---:|
| 8 | 1.10 ms | 2.05 ms | 6,803 |
| 32 | 1.32 ms | 1.87 ms | 6,803 |
| 128 | 2.88 ms | 3.63 ms | 6,803 |
| 256 | 5.14 ms | 6.31 ms | 6,803 |
| 512 | 9.27 ms | 10.69 ms | 6,803 |
| 1,024 | 17.82 ms | 18.77 ms | 6,803 |

This demonstrates that request-time option count changes compute but not parameter count.

It does **not** demonstrate equivalent accuracy or calibration to Laya/Jev.

## Temporal regression

A dedicated regression test now proves:

- memory-enabled state produces non-zero recent deltas;
- memory-disabled state zeros the temporal channel;
- two identical instantaneous observations with different recent histories receive different neural values;
- TD learning receives matching current and next temporal context.

Example CI score difference for the history ablation: ~8e-4 in the tested random initialization.

## Epistemic ensemble

The three scalar heads begin with different value estimates.

Example CI sample:

- initial normalized ensemble disagreement: ~0.0113;
- after repeated shared teacher distillation: ~0.01127.

The small decline is qualitatively expected but not yet a calibration result. We still need to measure whether disagreement predicts actual action error / regret.

## Schema transfer

Tests currently cover:

- action reorderings;
- renamed/rescaled action parameter fields with stable semantic descriptions;
- variable action cardinality;
- record permutation invariance;
- semantic-vector influence independent of surface ids.

The optional MiniLM schema compiler is intended to turn description paraphrase transfer into an empirical test rather than relying only on lexical overlap.

## Attention inspection

The controller can report the highest-weight records for the chosen action. The attention query is now conditioned on action, global state and recent temporal state.

Tests verify:

- attention weights are finite and normalized;
- record permutation does not change record-specific weights;
- teacher distillation changes action preferences while attention remains differentiable.

Attention display is a debugging aid. It should not be interpreted as complete causal explanation.

## Deployment

GitHub Pages deployment is active:

https://collinsomniac.github.io/doom-classifier/

Both validation and deployment jobs are green on current successful revisions.

## What remains unproven

The project does **not** yet establish that the small policy beats Jev or Laya on decision quality.

The next meaningful evidence requires:

1. multi-seed real DOOM episode returns;
2. survival, damage, kill, progress and resource-efficiency metrics;
3. teacher-query rate over learning;
4. Brier/log-score/ECE calibration;
5. frozen teacher-off evaluation;
6. multi-environment transfer;
7. equivalent typed-decision benchmark datasets against external baselines.


## State-conditioned attention update

The action-only attention query was replaced with an action + global-state + temporal-state query. In the deterministic fixture, changing only global state while keeping the action and record set fixed shifted record attention weights by about 0.025. The change increased the fast model from 6,419 to 6,803 parameters and modestly increased the large-state CPU benchmark, but removes a meaningful expressivity shortcut.
