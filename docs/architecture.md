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

Current trainable parameter count: **7,971**.

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

### Temporal / causal state

A separate learned channel encodes recent normalized scalar deltas. The same fixed-width channel now also receives a decayed **action → observed consequence trace** containing the previous typed action, reward, terminal flag, and numeric/boolean outcome facts.

This is deliberately generic. It lets identical instantaneous states score differently after different recent actions—for example, repeated motion with no useful consequence versus an action followed by positive progress—without a game-specific anti-loop rule and without increasing parameter count.

This is an intermediate recurrence mechanism, not the final learned belief state. Variable records are still not identity-tracked through time and the trace itself is not yet a trainable recurrent latent.

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

The DOOM harness now exposes four orthogonal semantic axes:

- movement: hold / forward / backward / strafe left / strafe right;
- view: keep heading / turn left / turn right;
- trigger: hold fire / fire;
- interaction: no use / use.

The literal button facts remain attached to every action for actuation and diagnostics. A compound packet such as `strafe_left + fire` therefore shares the same `trigger=fire` representation as every other firing packet while retaining its distinct movement choice.

This is closer to a Jev-style decomposed decision interface than the earlier eight independent button booleans, although the current fast core still evaluates the finite joint candidates before typed projection.

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

### State-dependent semantic authority

The base semantic trust region remains a KL budget of 0.08. The critic may expand that budget toward 0.16 only when three scale-free signals agree:

- bootstrap heads substantially agree on the same top typed action;
- the top-vs-runner margin is stable across bootstrap heads (margin SNR);
- the gap is meaningful relative to the critic's own action-value spread.

Those gates form a critic-ranking confidence. It is multiplied by experience trust (value-update count) to produce per-state critic authority. Epistemic disagreement remains a separate hard ceiling, so confidence-gated KL expansion cannot bypass the ensemble uncertainty guard.


### Probability calibration layer

The final typed action distribution has a separate optional post-fusion temperature calibrator. This is intentionally downstream of semantic/value fusion: fitting it can change confidence sharpness but cannot change the ranking of actions.

It is fitted only from labelled held-out decisions using a proper scoring objective. The core calibration module reports:

- multiclass negative log-likelihood;
- multiclass Brier score;
- expected calibration error (ECE) and reliability bins;
- maximum calibration error;
- top-1 accuracy and mean stated confidence.

The calibrator is checkpointed with the policy. If no valid labelled calibration set has been supplied for the current task, it remains an identity transform and the browser reports **unverified** rather than calling normalized softmax scores calibrated probabilities.

This is distinct from two other temperature/bias operations:

- **null-state label-bias correction** removes state-independent NLI action-label preference from the semantic teacher;
- **student temperature fitting** adjusts the distilled semantic head to better match a teacher distribution.

Neither of those establishes empirical probability calibration.

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

## Why the current policy can look non-player-like

The repeated frozen starter gate exposed a structural failure rather than a simple reward-scale problem. On the latest pre-change run, the semantic baseline produced useful combat in only one of three frozen rollouts; the other two collapsed to `turn_right`. After staged reward training, repeated rollouts collapsed mostly to `forward` with zero combat.

The common pattern is **motor-mode stickiness**. The current critic primarily learns values for complete joint actions. Typed structure is projected back into those scores after the fact, so an easy-to-reinforce exact action can become a stable attractor even when the underlying semantic dimensions should be changing independently.

The next core should invert that relationship.

## Proposed next core: recurrent typed-decision model

The target fast architecture is:

`structured records → semantic set encoder → recurrent belief state h_t → parallel typed option heads → constrained composition → literal actuator packet`

### 1. Semantic set encoder

Keep the schema-compiled scalar and record encoders, but move from a single mean/max summary toward a small learned slot bank. Slots should preserve several simultaneously relevant concepts—threat, resource, traversable space, interaction candidate—without baking those names into the core. The slots are learned from schema semantics and consequences rather than hard-coded game categories.

### 2. Learned recurrent belief state

Replace the hand-built scalar delta memory with a tiny recurrent state, e.g. a GRU-style state in the 32–96 dimensional range:

`h_t = GRU(h_(t-1), encoded_state_t, previous_typed_action, observed_consequence_t)`

The important addition is the action/outcome pair. The model must be able to represent not only “what is true now?” but “what did I just try, and what changed because of it?”

The causal trace implemented in the current branch is the low-risk precursor to this learned state.

### 3. Parallel typed option heads

Treat each declared control axis/question as a native output head, not merely as metadata attached to a joint action classifier. For DOOM:

- movement;
- view;
- trigger;
- interaction.

Each option is still represented semantically at request time, so the architecture remains useful for different schemas. All heads share the same belief state and run in one forward pass.

### 4. Small interaction/composition layer

Independent axes are not perfectly independent. Add a deliberately low-rank interaction term between chosen axis options, then compose only actuator-valid packets. This captures synergies such as “move + fire” without requiring the network to relearn every combination as an unrelated class.

### 5. Multi-horizon consequence prediction

A single scalar Q target is too easy to game with locally repetitive behavior. Predict several consequence summaries/horizons from the same latent:

- immediate reward/event distribution;
- short-horizon return;
- longer-horizon return;
- successor latent / state-change prediction.

This gives the hidden state pressure to encode what actions *do*, not merely which button recently correlated with reward.

### 6. Same-state causal supervision

Use the owned engine snapshot/fork mechanism as a training oracle on selected states. Counterfactual rollouts should supervise the typed option heads with proper distributions and train the consequence model against measured successor differences. These probes are expensive training-time evidence and disappear from the fast frozen path.

### 7. Calibrated autonomy

Keep ensemble disagreement, but calibrate confidence against **decision regret**: when the model reports high confidence, the chosen composed action should rarely be materially worse than the measured alternatives. This is more useful than softmax sharpness alone.

### 8. Compute placement

The intended deployment split is:

- environment and deterministic simulation: WASM;
- tiny recurrent control core: whichever of JS/WASM-SIMD/WebGPU benchmarks fastest for its size;
- schema compiler / larger Transformers.js teacher: WebGPU where available;
- browser visualization: GPU-rendered but observational only.

The fast core should not be moved to WebGPU merely for architectural symmetry; at very small matrix sizes dispatch/copy overhead can dominate.

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

## Consequence learning stability

The reward-specific branch uses several training-only stabilizers while leaving the frozen inference path unchanged:

- four-step n-step returns;
- bounded replay;
- a delayed target network;
- **Double-DQN bootstrap targets**: the online critic selects the next action and the delayed target critic evaluates that selected action, reducing maximization bias from taking the target network's own noisy maximum;
- a three-head bootstrap ensemble whose disagreement limits policy authority;
- KL-constrained fusion against the semantic prior, with confidence-gated state-dependent expansion.

The Double-DQN change adds no inference cost. It only changes how reward targets are constructed during browser fine-tuning.

