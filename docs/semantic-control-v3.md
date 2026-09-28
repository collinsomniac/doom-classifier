# Semantic Control v3 — target architecture

## Goal

Build a machine-native decision model for structured environments:

```
typed environment state
  -> compact persistent belief
  -> predicted consequences
  -> objective-conditioned utility
  -> factorized typed controls + calibrated confidence
```

DOOM remains the primary behavioral harness, but no component below may depend on a DOOM tactic.

The target is not a smaller language model. It is a small decision model whose interface is closer to code: typed state in, parallel distributions over a known control schema out.

## Why v2 stalls

The current v2 controller established useful invariants:

- real browser-native Chocolate Doom WASM;
- structured scalar/entity/geometry state rather than pixels;
- semantic prior separated from reward value;
- reward learning cannot overwrite semantic logits;
- exact deterministic state forks;
- typed primitive action parameters;
- bootstrap uncertainty and KL-bounded fusion;
- single-digit-millisecond-class CPU inference at the previous action cardinality.

But the repeated frozen runs exposed four structural limits.

### 1. Joint action enumeration is the wrong abstraction

A player command is naturally factored:

- translation: neutral / forward / back / strafe-left / strafe-right;
- view: neutral / left / right;
- trigger: off / on;
- interaction: off / on.

Enumerating packets is useful for compatibility with the engine and exact same-state probing, but it should not define the learned output space.

v3 should predict those axes in parallel and let code compose the final literal controller packet.

### 2. A generic NLI teacher is semantics, not behavior

MobileBERT can cheaply initialize meanings such as left/right/fire/use and relate schema language to state language. It is not trained to be an expert FPS controller.

The teacher therefore belongs at compile/bootstrap time and as an occasional semantic regularizer. Behavioral grounding must come from consequences, demonstrations, or a decision-specialized teacher.

### 3. Scalar Q throws away reusable knowledge

A scalar value says that an action was useful for one current reward function. It does not preserve *why*.

A better reusable target is a vector of predicted future factual outcomes. The environment declares outcome features; the model predicts their discounted future occupancy/effect. Objective-specific utility can then be changed without relearning the world's dynamics.

This is the project analogue of successor features.

### 4. Scalar delta memory is not a belief state

Current temporal memory is a smoothed vector of scalar differences. It cannot represent persistent latent facts such as:

- an actor briefly leaving line of sight;
- an action's delayed consequence;
- a doorway or threat encountered several decisions ago;
- whether a repeated control is making progress.

v3 needs a trainable recurrent belief that remains cheap enough for online learning.

## Proposed hot path

### A. Structured relational encoder

Inputs remain schema-described facts.

1. Encode scalar fields with compiled field semantics.
2. Encode each collection record with a shared record network.
3. Preserve collection identity.
4. Use collection-balanced aggregation so record count alone cannot dominate evidence.
5. Retain stable engine record IDs only as identity keys, never as semantic categories.

DOOM may additionally expose policy-blind relational summaries such as nearest hostile distance/bearing. Those are observations, not actions or strategy.

### B. Recurrent belief state

Maintain a small recurrent state `b_t` from:

- current global representation;
- compact entity/geometry summaries;
- previous factorized action;
- factual outcome/event vector from the previous transition.

A Recurrent Trace Unit (RTU)-style cell is the leading candidate because it is explicitly designed for efficient online recurrent reinforcement learning with exact/efficient real-time recurrent learning.

Initial target:

- 32–64 recurrent dimensions;
- fixed parameter count independent of entity/action cardinality;
- online update every control decision;
- no requirement for backpropagation through a long stored sequence.

The recurrence is responsible for *state estimation*, not strategy.

### C. Successor / consequence representation

Replace the single learned scalar `V(s,a)` as the primary learned object with:

```
psi(s, a) = expected discounted future outcome-feature vector
```

The environment schema declares factual outcome fields. A DOOM adapter can expose, for example:

- player-attributed damage dealt;
- damage received;
- player kills;
- successful pickups;
- spatial novelty/progress;
- level completion;
- survival/death;
- resource deltas.

A different environment declares different fields without changing the core model.

Then:

```
utility(s,a | objective) = psi(s,a) dot w_objective
```

where `w_objective` may come from:

- explicit environment reward weights;
- a compiled semantic objective mapping;
- learned preference/reward fitting;
- a mixture of these.

This separates reusable environment knowledge from one reward definition.

### D. Multi-horizon prediction

Exact forks already show that 6/12 DOOM tics can be too short while 24/35 tics separate useful actions.

Instead of forcing one horizon, v3 predicts several:

- immediate;
- short;
- medium;
- longer local consequence.

The controller can learn that an action has low immediate reward but useful medium-horizon consequences.

### E. Factorized decision heads

The native learned output is not sixty packet logits.

For each declared control axis:

```
P(translation option | belief)
P(view option | belief)
P(trigger option | belief)
P(interaction option | belief)
```

Each axis includes an explicit neutral option.

A small low-rank compatibility residual may model interactions such as movement + fire without converting the model back into a giant flat categorical classifier.

Literal controller packets are composed in code.

This gives:

- no action-cardinality prior caused by enumeration;
- parallel outputs;
- clearer calibration;
- fewer action-conditioned attention queries;
- natural reuse when a new combination is introduced.

## Training hierarchy

### 1. Semantic compile

Use MiniLM / Transformers.js, preferably WebGPU when available, to compile static objective/schema/action meanings.

This work stays off the control loop.

### 2. Semantic bootstrap

Use a semantic teacher only to initialize typed meanings and broad applicability.

For factorized controls, ask axis-level questions including explicit neutral options.

Do not treat generic NLI as the behavioral oracle.

### 3. Exact simulator supervision

When an adapter supports snapshots:

1. snapshot one state;
2. branch every legal typed control from the identical state;
3. measure factual outcome vectors over multiple horizons;
4. restore state;
5. train the consequence/successor model from those measured targets.

This is high-information supervision: one state can compare every candidate under matched conditions.

### 4. Online interaction learning

Use actual transitions for long-horizon credit and states not covered by local forks.

The recurrent belief and successor model should receive the learning signal. Semantic schema parameters remain protected unless explicitly undergoing semantic distillation.

### 5. Demonstration grounding

Optional demonstrations are a major source of player-like priors without game-specific rules.

A demonstration sample is simply:

```
(structured state, previous belief, typed control axes, observed outcome)
```

Sources may include human browser play or engine-native demo playback when available.

Behavior cloning should initialize the control policy; consequence learning remains responsible for correcting it.

### 6. Frozen evaluation + promotion

Never publish the latest checkpoint merely because training reward increased.

For each candidate checkpoint:

- teacher off;
- no exploration;
- multiple fresh rollouts;
- exact-fork regret on held-out states;
- outcome quality;
- action-axis diversity/stickiness;
- calibration metrics.

Keep the best validated checkpoint, including the pre-training baseline if learning made behavior worse.

## Confidence

Confidence should attach to factual claims the model makes, not only the final packet.

Track separately:

- axis ambiguity: entropy/margin of each factorized control head;
- successor uncertainty: ensemble variance for predicted consequences;
- semantic uncertainty: teacher/student disagreement;
- state novelty;
- held-out probability calibration;
- exact-fork regret.

A controller can then distinguish:

> I know what will happen but several actions are similarly good

from:

> I do not know what this action will cause.

Those require different responses.

## WebGPU / WASM split

Use hardware by workload, not branding.

### WebGPU

Best current targets:

- Transformers.js MiniLM schema compilation;
- Transformers.js semantic teacher;
- future batched embedding / larger learned encoders;
- visualization compute when useful.

### CPU / WASM / JS

The tiny online controller is small enough that GPU dispatch can dominate arithmetic.

Keep the recurrent/factorized fast path on CPU or WASM-SIMD until profiling shows a real WebGPU win.

Chocolate Doom remains WASM.

## Expected v3 decision graph

```
                         compile-time semantics (WebGPU optional)
                                       |
structured state -> relational encoder -> recurrent belief ------------------+
       |                    ^                 |                               |
       |                    | previous action/outcome                        |
       |                                      v                               |
       +--------------------------> successor consequence ensemble             |
                                              |                               |
objective semantics --------------------------+--> utility / uncertainty      |
                                                                              |
belief ----------------------------------------------------> factorized axes --+
                                                                              |
                                                   code composes literal packet
                                                                              |
                                                            -> environment/WASM
```

## Research basis

Useful foundations for this direction:

- TypeSafe System One / Jev: machine-native typed probabilistic decisions and decomposed workflows rather than autoregressive strings.
- Barreto et al., *Successor Features for Transfer in Reinforcement Learning* (NeurIPS 2017): decouple environment dynamics from reward for transfer.
- Barreto et al., *Transfer in Deep RL Using Successor Features and Generalised Policy Improvement* (2019): deep successor-feature transfer.
- Elelimy et al., *Real-Time Recurrent Learning using Trace Units in Reinforcement Learning* (NeurIPS 2024): lightweight recurrent state trained efficiently online.
- Relational/entity-centric RL work: structured object representations can improve sample efficiency and generalization when the environment naturally exposes entities.

## Implementation sequence

1. **v2.1 (current PR):** clean Play/Inspect UI, complete typed controller lattice, axis-factorized semantic teacher, relational factual summaries, private critic attention, measured-fork critic training, checkpoint migration.
2. **v2.2:** use measured-fork trainer for starter publication and benchmark promotion; add human/demo trajectory capture.
3. **v3a:** replace packet decoder with native factorized axis heads + compatibility residual.
4. **v3b:** add RTU-style recurrent belief with previous action/outcome input.
5. **v3c:** replace scalar critic target with multi-horizon successor/outcome feature prediction.
6. **v3d:** calibrate axis probabilities + consequence uncertainty against held-out exact forks and evaluate cross-map/cross-task transfer.
