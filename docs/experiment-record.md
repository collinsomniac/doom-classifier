# DOOM semantic-control experiment: record and takeaways

Status: **frozen diagnostic baseline.** The branch `feat/grounded-decision-loop-v4`
(PR #22) is *not* merge-ready. This document is the durable record of what was
built, what was measured, and what should not be repeated.

The successor project starts from the conclusions here, not from this code.

---

## 1. What this experiment was for

Use DOOM as a small, exactly reproducible environment to study one question:

> Can a compact model turn factual environment state into a **semantically
> correct** action decision, rather than merely into actions that happen to
> earn reward?

Success was defined as *visible relational understanding*: hostile left ->
orient left, mirrored state -> reversed preference, aligned hostile -> firing
becomes appropriate, no visible hostile -> firing becomes inappropriate. Kills
and action diversity were explicitly **not** accepted as evidence.

## 2. What was built

| Component | Purpose | Verdict |
| --- | --- | --- |
| Chocolate Doom WASM runtime with a telemetry bridge | factual state, line of sight, entities | keep |
| Exact tic stepping (`PromptFPS_StepTics`) | deterministic interventions | keep |
| Savegame snapshot/restore | counterfactual branching | keep, but slow |
| Counterfactual probe labels | measured-consequence supervision | keep the idea, fix the horizon |
| Factorized control head (movement/view/trigger/interaction) | typed decision composition | keep, retrained |
| Neural set residual (7.9k params, entity tokens) | learned value | **contributes ~nothing** |
| Semantic prior (MiniLM schema embeddings) | language grounding | outcompeted by a 300-param head |
| Typed value projection + KL-budgeted fusion | prior/value arbitration | removed (artifact + dilution) |
| Benchmark with mirrored probe + 300 s gameplay | behavioral diagnostics | keep, extend |

Measured on an iPhone 17 Pro Max in WebKit (live E1M1):

| Operation | Time |
| --- | --- |
| 4 engine tics | 0.01 ms (~400k tics/s) |
| JSON observation (65 KB) | 0.9 ms |
| save / restore snapshot | 3.2 ms / 1.7 ms |
| 15-branch 24-tic probe | 59 ms |
| one decision | 1.9 ms |
| one training step (7,971 params) | 2.5 ms |

The environment and the model are both cheap. **The cost is JSON telemetry,
savegame snapshots, and single-threaded execution.**

## 3. The failure that started this pass

Typed target fit ~0.9998 while mirrored directional preference ~0.0086 and
correct-turn rate ~0.024. Four independent defects explained it.

### 3.1 The label horizon was not what the code claimed
`probeTics: 24` was accepted and reported but **never used**; labels came from a
single 4-tic step (`plannerDepth: 1`). DOOM damage lands one interval later:
in the CI trace, `P(damage | fire at t-1) = 0.18` versus `P(damage | fire at t) = 0.05`,
and `P(damage | no fire at t-1) = 0.00`. The trigger axis therefore received
essentially no causal signal (0 of 83 informative labels had a fire top action;
its weights stayed near zero).

### 3.2 The factor head was fitted axis-by-axis
Each axis was trained on its own marginal of the target distribution. Composing
independent marginals then made the **all-neutral composite (`wait`)** the mode
whenever the label mass was split across composites sharing one factor
(`turn_left` vs `strafe_left`). Reproduced in isolation in
`tools/wait-bias-repro.mjs`. The fix is proper **joint cross-entropy over valid
composites**; `tests/factor-grounding.mjs` fails on the old objective and passes
on the new one.

### 3.3 The typed value projection fabricated preferences
A ridge projection of labels onto typed field columns broke exact return ties
toward the action with the most distinctive column pattern: `use` became the
modal target in **every** regime (`tools/projection-artifact.mjs`). Blend set to 0.

### 3.4 Firing at nothing was exactly tied with holding fire
Without an ammunition term, the environment reported no difference between
firing at a wall and holding fire, so no label could teach appropriateness.
Added the engine's factual ammo consumption.

### 3.5 A metric that was never a learning metric
`typedTargetFit` measured the *label's* fit onto the additive typed basis, not
the model's fit. It reads ~1.0 while the deployed policy is ungrounded. It
should never have been reported as training quality.

## 4. Results

All runs: 768 causal steps, 24-tic probes, 300 s simulated benchmark.
Replay within a run is exact (same canonical start snapshot, restored RNG).

| Commit | Turn margin | Fire contrast | Correct / wrong turn | Fire, no enemy | State->action dependence | 300 s kills |
| --- | --- | --- | --- | --- | --- | --- |
| 6f9ad8d (before) | 0.036 | -0.012 | 0.02 / - | 0.11 | 0.02 | 9 |
| 37e9fff joint head + delayed credit | **0.173** | 0.337 | **0.66 / 0.19** | 0.88 | 0.54 | 4 |
| 0f13fcb + no projection + ammo cost | 0.104 | **0.90** | 0.44 / 0.01 | **0.00** | **0.66** | 6 |
| 5dde363 continuation set (reverted) | 0.165 | 0.18 | 0.07 / 0.02 | 0.00 | 0.15 | **0** (25 deaths) |
| afa28fc revert = 0f13fcb code | 0.028 | 0.68 | 0.02 / 0.01 | 0.00 | 0.30 | 10 |

**Recovered:** firing became genuinely state-coupled (always when aligned,
never at nothing; fire log-odds ~12.7 at one point). **Not recovered:**
orientation. In the best run the policy mostly walks forward (2,011 of 2,625
decisions) because a turn's payoff falls outside the probe window.

### The most important row is the last two
`afa28fc` is the **same source** as `0f13fcb` (`git diff` differs only by a
stray tracked file) and produced turn margin 0.028 versus 0.104. **Single runs
cannot rank changes.** Nothing in this project should be compared without
several seeds.

## 5. Negative results worth keeping

- **Best-of-`{wait, fire}` continuation labels regressed to 0 kills / 25 deaths.**
  Valuing a branch by whatever a later primitive could rescue teaches the rescue,
  not the first action. Reverted.
- **A 4-tic probe cannot see firing.** Any horizon shorter than the causal delay
  makes the action invisible, no matter how well the rest is trained.
- **Optimistic-return targets inflate rare actions**; they interacted badly with
  coverage exploration.
- **Auxiliary gates (`criticAgreementFloor`, `criticSnrFloor`, `criticKlFloor`)
  were tuned so they never visibly bound.** Unmeasured thresholds are decoration.
- **The obsolete CI gate `informative > causalSteps * 0.55`** encoded the old
  assumption that every probe should be informative. Rejecting tie-valued probes
  is correct. It was replaced with per-regime coverage plus grounding gates.

## 6. Conclusion: why this architecture is the wrong container

A decomposition trace (`tools/fusion-trace.mjs`) on the trained checkpoint, on
the four mirrored probe states:

```
final score spread 0.72   factor head 0.72   semantic prior 0.055   value 0.000   beta 0.000
```

The deployed decision is a **300-parameter linear head** over ~24 hand-picked
relational fields. The 7.9k-parameter entity network contributes nothing; the
semantic prior contributes nothing; the value branch never received policy
authority. The "learned semantics" were the hand-designed features plus a
linear map.

Consequences:

1. The model cannot handle choices it was not compiled for; the 15 actions are
   a fixed vocabulary, so the Jev/Laya capability (typed objects in, confidence
   out) is not represented at all.
2. Relations are supplied as features rather than learned from entities, so the
   work is done by the person writing the schema.
3. Stacked arbitration mechanisms (prior, value, KL budget, critic gates,
   projection, calibration) made the system hard to attribute. Each had a
   plausible justification and none demonstrably helped.
4. Credit assignment is the actual bottleneck, not capacity.

## 7. Environment adapter contract (worth carrying forward)

The environment layer earned its place. Interfaces to preserve:

- `observe()` -> typed, schema-described observation.
- `stepTics(actionId, n)` -> exact, unthrottled simulation; returns reward and a
  factual outcome record.
- `saveSnapshot()` / `restoreSnapshot(token)` -> exact state, including RNG.
- schema declares fields, collections, action fields, enums, and ranges;
  the model side must consume that schema generically.

Fix in the next version: a **reward-only fast path** for probes (avoid 65 KB of
JSON per branch), in-memory snapshots, and parallel environment workers.

## 8. Reproducing this record

```bash
node tests/factor-grounding.mjs          # unit grounding gate (fails on the pre-fix objective)
node tools/wait-bias-repro.mjs           # marginal-composition attractor
node tools/projection-artifact.mjs       # typed projection fabricates "use"
node tools/factor-probe.mjs              # factor head alone on mirrored states
```

The neural-set model segfaults under iSH's Node build. Run browser-side traces
through WebKit (see the successor project's tooling) or in CI.
