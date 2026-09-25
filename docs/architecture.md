# Working architecture

## Research target

Primary benchmark:

`structured environment state -> typed decision distribution`

Perception is intentionally outside the current core question. If a program already has useful state, how little learned compute is required to turn that state into fast, semantically grounded, adaptable actions?

DOOM is the first harness because it supplies dynamics, consequences, variable objects, geometry, resources and a recognizable control surface. The policy API is designed to remain useful outside games.

## Three compute timescales

### 1. Compile time

Static language should not be re-tokenized every control tick.

The optional compile path:

1. reads objective and schema metadata;
2. feature-hashes labels/descriptions as a zero-download fallback;
3. optionally embeds objective, fields, enum values, collections and actions with MiniLM;
4. projects those embeddings into the fixed policy feature space;
5. initializes/reconfigures the tiny neural controller;
6. disposes MiniLM.

### 2. Teacher time

A larger semantic model may be called:

- at preparation/bootstrap;
- periodically;
- on novelty;
- on high entropy / low margin;
- on epistemic disagreement.

Teacher output is distilled locally. The expensive model is not required on frozen neural ticks.

### 3. Control time

Every fast tick:

1. receive native scalar state + variable record collections;
2. encode recent scalar deltas;
3. encode records with a shared set network;
4. form an action/state/temporal-conditioned attention query;
5. attend to records separately for each candidate action;
6. combine semantic-prior and consequence-value scores;
7. decode a typed action;
8. execute literal primitive control inputs.

No language-tokenization pass is required on ordinary neural-only ticks.

## Fast model: SchemaSemanticValueSetNet

Current trainable parameter count: **7,316**.

Model size is independent of record count and candidate-action count.

### Compiled schema semantics

Each field / action may contain:

- id;
- human label / description;
- numeric range or scale;
- categorical enum meanings;
- optional MiniLM semantic vector.

Categorical engine codes therefore do not have to remain opaque. The DOOM adapter uses the same generic enum mechanism for weapon names and entity categories.

### Global state

Declared scalar fields modulate their compiled field representations.

The objective itself contributes to the compiled global representation.

### Temporal state

A separate learned channel encodes recent normalized scalar deltas. Temporal information is therefore distinguishable from instantaneous values rather than silently concatenated.

Current limitation: variable records are not yet identity-tracked through time.

### Variable collections

For collection `C = {r_1 ... r_n}`, every record uses the same small record MLP.

The global set context includes:

- mean record latent;
- max record latent;
- record count;
- collection count.

Record ordering is intentionally irrelevant.

### State-conditioned action attention

For candidate action `a`, global state `s`, recent temporal state `ds`, and record latent `z_i`:

`q = Query(action_embedding(a), global(s), temporal(ds))`

`alpha_i = softmax(q dot z_i / sqrt(d))`

`attended(a) = sum_i alpha_i * z_i`

This permits the same action to inspect different records in different states.

The live attention inspector is a debugging aid, not a complete causal attribution claim.

## Typed actions

Actions are request-time objects rather than fixed output-neuron identities.

A candidate may contain:

- label / description;
- optional semantic vector;
- numeric/categorical `params` governed by `schema.actionFields`.

The DOOM harness uses this to represent compound literal controls such as `strafe_left + fire` as:

`{ strafe_left: 1, fire: 1, ... }`

Thus related candidates share primitive structure even though the final decoder still evaluates coherent compound actions.

This generalizes to coordinates, actuator strengths, bids, thresholds, tool arguments, route candidates, etc.

## Semantic prior and consequence value

This is the most important current separation.

For each action the shared encoder produces features `h(s,a)`.

Two training paths sit on those features:

### Semantic prior

`S(s,a)`

- trained by semantic-teacher distillation;
- intended to encode action applicability / semantic plausibility;
- teacher preparation can iterate locally until a bounded KL fit target is reached.

### Consequence value

`V(s,a)`

- trained from real transition reward;
- has a small hidden layer and three bootstrap scalar heads;
- uses replay;
- bootstraps from a delayed target value network.

Final neural score:

`score(s,a) = S(s,a) + value_weight * mean(V_heads(s,a))`

Current `value_weight` is 0.5.

Reward-only TD updates do **not** backpropagate through the semantic/shared branch. A regression test asserts semantic-logit drift is exactly zero during reward-only updates.

Teacher distillation may update the semantic/shared branch, but it synchronizes only the target network's semantic representation. Target value weights remain delayed until their normal TD sync interval.

## Online value learning

The default learner uses a short four-step return. For a prefix beginning at action `a_t`:

`R_t^(n) = r_t + gamma r_(t+1) + ... + gamma^(n-1) r_(t+n-1)`

`target = R_t^(n) + gamma^n * max_a V_target(s_(t+n), a)`

Shorter prefixes are flushed at terminal states and at the end of a bounded browser training burst. Only the consequence-value branch receives this gradient.

The browser learner adds:

- replay capacity: 96 **aggregated** transitions;
- replay batch: 2 extra transitions per live update;
- target sync interval: 24 value updates;
- bootstrap subset of value heads;
- policy-proportional exploration from the model's own calibrated distribution with a small uniform floor;
- bounded semantic teacher replay;
- KL-bounded fusion between semantic prior and learned consequence value.

## Semantic teacher

Current practical teacher: MobileBERT-MNLI in Transformers.js.

Experimental teacher presets:

- DistilBERT-MNLI;
- DeBERTa-v3-xsmall NLI.

The adapter:

- creates a compact state premise;
- preserves named enum/category values;
- summarizes variable collections with category prevalence, binary flags, selected numeric statistics and representative records;
- bounds premise tokens;
- scores actions independently with NLI entailment-vs-contradiction odds.

The independent scoring matters: a standard single-label zero-shot classifier forces candidate likelihoods to sum to one even when all choices are weak.

Current limitation: generic NLI models still over-associate mechanically relevant words (notably FIRE) with the standing objective. They are semantic priors, not yet reliable decision-specialized affordance models.

## Decode calibration

Teacher supervision also fits a small decode-temperature parameter by minimizing cross-entropy against teacher probabilities over a bounded temperature grid.

Teacher-only mode uses temperature 1 so this calibration cannot make the teacher baseline artificially sharper.

## Epistemic uncertainty

Three bootstrapped value heads share the encoder.

For each candidate they produce slightly different consequence estimates. Jensen-Shannon-style disagreement among the induced policies is exposed as an epistemic signal.

The UI distinguishes:

- entropy;
- top-two margin;
- scalar-state novelty;
- value-head epistemic disagreement.

## Replay and target network

Experience replay reduces dependence on the latest transition.

The target network is training-only state; it does not add inference work.

Two target sync paths are deliberately separate:

- semantic teacher refresh -> semantic/shared target parameters only;
- scheduled TD target sync -> semantic + value parameters.

This prevents frequent teacher refreshes from accidentally collapsing the delayed value target.

## DOOM adapter

The adapter exposes facts and mechanics, not tactical policy.

### Scalar state

Includes:

- health / armor;
- ammunition;
- equipped weapon category;
- recent received / dealt damage;
- under-fire flag;
- player position / velocity / heading;
- kills;
- visited-cell count / cell revisit count / spatial novelty.

### Entity records

Include:

- absolute and relative position;
- velocity;
- dimensions;
- health;
- distance / relative angle;
- line-of-sight;
- hostile / collectible / targets-player flags;
- stable record id;
- factual coarse category;
- named Chocolate Doom mobj categories when known.

Projectile/attack-effect categories are facts from the engine type table. The policy is not told that a projectile implies any particular response.

### Geometry records

Expose map-line geometry, blocking flag, special, tag and line flags.

### Primitive actuation

The policy outputs actual button masks. FIRE is Chocolate Doom's real fire input.

No tactical macros are supplied.

## Reward integrity

Environment telemetry and learning reward are deliberately different contracts.

The current borrowed Chocolate Doom bridge exposes `killcount`, incoming `damagecount`, and entity health, but it does **not** expose attacker-attributed combat events. The adapter can observe hostile HP decreases and intermission kill-count changes, but in vanilla single-player Chocolate Doom both can reflect monster infighting or other non-player causes.

Accordingly:

- `recent_hostile_hp_loss` is factual state telemetry;
- kill-count changes and hostile HP loss contribute **zero** reward until a project-owned bridge exposes attacker attribution.

This prevents a coincidental action from receiving positive value merely because hostile HP happened to decrease during its control interval.

## Progress reward

The current borrowed bridge does not expose direct exit/completion progress.

As an interim policy-blind progress signal, the adapter tracks coarse player-position cells per episode and gives a small reward for first visits.

This says only “new space is informative”; it does not specify a route, destination or action.

## Runtime state machine

Allowed responsibilities:

- engine/model readiness;
- running / paused / resetting / tuning / error;
- action timing;
- episode boundaries;
- teacher lifecycle;
- training vs evaluation;
- logs / metrics / export;
- compute budgets.

Disallowed responsibility:

- behavioral strategy.

**The FSM understands program state, never gameplay policy.**

## Current complexity

Latest CI CPU stress sample:

- 7,316 trainable parameters;
- 768 variable records;
- p50 ~7.45 ms;
- p95 ~9.82 ms.

Action-cardinality scaling remains parameter-count invariant.

## Architectural limitations / next work

### Previous-action + record recurrence

The policy sees global deltas but does not yet explicitly encode which previous control produced them, nor track individual entity identities through time.

### Decision-specialized teacher

NLI provides semantic knowledge but is not trained for calibrated state-action affordance decisions. A small generic decision model or decision-specific fine-tune is likely the next major semantic improvement.

### Owned engine artifact

The project now builds and publishes its own pinned Chocolate Doom browser runtime. Native hooks expose monotonic player-attributed damage, player-attributed kills, successful pickups, level completions, and secret exits. The browser adapter differences those counters per transition before reward learning. The build artifact carries source provenance, licenses, the exact telemetry patch, and the instrumentation script.

### Calibration

Add held-out Brier score, log score, ECE/reliability, and test whether epistemic disagreement actually predicts regret/error.

### Multi-environment transfer

DOOM cannot prove generality. Add unrelated structured environments with renamed/paraphrased schemas and different action parameter types.
