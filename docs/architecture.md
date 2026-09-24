# Working architecture

## Research target

The primary benchmark is:

[
	ext{structured environment state} ightarrow 	ext{typed decision distribution}
]

Perception can be added later, but the current question is deliberately narrower: if useful machine state already exists, how little learned compute is required to turn it into fast, calibrated-enough, adaptable actions?

DOOM is the first harness because it supplies rich dynamics, consequences, entities, geometry and a recognizable control surface. The core policy API must remain usable outside games.

## Design principle: language at compile time, numbers at control time

Large typed-decision models such as Laya accept text/JSON and recover semantics inside a language encoder on each request. That is flexible, but repeated state fields, action descriptions and schema prose are mostly static.

doom-classifier separates two paths.

### Compile path

Potentially expensive and infrequent:

1. read the environment objective and schema metadata;
2. compile lexical metadata;
3. optionally embed objective / field / collection / action language with MiniLM;
4. compile typed numeric action-field definitions;
5. initialize or reconfigure the small neural controller;
6. optionally call a larger semantic teacher and distill its distribution.

### Control path

Bounded and repeated:

1. receive native numeric globals and variable record collections;
2. compute recent temporal deltas;
3. encode globals, temporal state and records;
4. perform action-conditioned record attention;
5. score every request-time candidate action;
6. aggregate the bootstrap value ensemble;
7. softmax the resulting action preferences;
8. optionally schedule a teacher asynchronously;
9. execute the selected primitive action.

No textual state serialization is required on ordinary neural-only ticks.

## Neural controller

Current controller: `SchemaHashAttentionSetNet`.

The parameter count is independent of the number of records and action candidates.

### Static schema representation

Each field/action receives a fixed-size representation from:

- lexical feature hashing over labels/descriptions/units;
- optional MiniLM sentence embeddings projected into the same fixed space;
- normalized numeric action parameters where defined.

The lexical path guarantees a zero-download fallback. The semantic-vector path improves paraphrase relationships without placing MiniLM on the real-time loop.

### Global state

Current scalar fields are normalized from declared ranges or scales and modulate their compiled field representations.

The objective is also part of the compiled global representation. This is important: the neural controller is not merely learning an environment-specific Q-table; its state representation is conditioned on what the environment says success means.

### Temporal state

Recent normalized scalar deltas are maintained by a leaky temporal memory.

Temporal features are encoded through a separate learned projection using metadata equivalent to "recent change in <field>". They are not silently concatenated with instantaneous values.

This currently covers global fields. Identity-aware temporal tracking for variable records is a future extension.

### Variable collections

An observation can contain arbitrary named collections:

[
C_k = {r_1, r_2, ldots, r_n}
]

Every record is encoded with the same small MLP. Record ordering is intentionally irrelevant.

A global set summary uses mean and max pooling plus bounded record-count information.

### Action-conditioned attention

Mean/max pooling is not enough when different actions should inspect different parts of the world.

For action (a), the action embedding produces a query (q_a). Every record latent (z_i) receives:

[
alpha_i(a)=mathrm{softmax}left(rac{q_a^	op z_i}{sqrt d}ight)
]

and the action receives:

[
c_a=sum_i alpha_i(a)z_i
]

The final score head therefore sees:

- global state;
- temporal state;
- permutation-invariant set summary;
- action-specific attended record context;
- the typed action embedding.

The live UI exposes top attention records for the chosen action as a diagnostic, not as a claim of causal interpretability.

## Request-time typed actions

Actions are not represented by fixed output neurons alone.

A candidate can include:

- id / label / description;
- optional static semantic vector;
- optional numeric `params` / `values` governed by `schema.actionFields`.

This permits candidate spaces such as:

- discrete game controls;
- click coordinates;
- actuator values;
- ranked objects;
- thresholds;
- prices / bids;
- route candidates;
- tool calls with numeric arguments.

Changing candidate order or cardinality does not change parameter count.

## Value ensemble and uncertainty

A single softmax peak is not equivalent to knowledge.

The controller now uses three independently initialized scalar value heads over a shared encoder. Reward updates bootstrap a subset of heads; teacher distillation supervises all heads.

The policy reports four distinct signals:

1. **entropy** — distribution diffuseness;
2. **margin** — top-1 / top-2 separation;
3. **novelty** — scalar-state deviation from online history;
4. **epistemic disagreement** — normalized Jensen-Shannon disagreement among ensemble policy distributions.

The ensemble is intentionally cheap: only the scalar output heads are independent in the current version. This is a first epistemic approximation, not a fully independent deep ensemble.

## Semantic teacher

The normal fast path can run without a large semantic model.

Three inference modes are supported:

- **hybrid** — semantic model runs every decision and is combined with neural value;
- **adaptive** — neural policy controls immediately; semantic teacher is scheduled asynchronously when warranted;
- **neural** — no semantic calls.

Adaptive teacher triggers can include:

- initial/bootstrap supervision;
- fixed refresh interval;
- high entropy;
- small top-two margin;
- state novelty;
- ensemble disagreement.

A teacher result is distilled into the small network. Generation/version guards prevent stale teacher results from being applied after model/schema reconfiguration.

## Online consequence learning

Reward adaptation uses tiny TD-style updates.

For chosen action (a_t):

[
delta_t=r_t+gamma max_a Q(s_{t+1},a)-Q(s_t,a_t)
]

The shared representation and a bootstrapped subset of ensemble heads are updated.

The reward signal defines consequences, not strategy. The environment is allowed to say that death is bad or progress is rewarded; it is not allowed to say "if health is low, retreat."

## Orchestration boundary

The state machine may understand:

- readiness;
- running / paused / reset / error;
- action timing;
- episode boundaries;
- logging;
- learning enabled/disabled;
- teacher enabled/disabled;
- compute budgets.

It must not encode behavioral rules.

**The FSM understands program state, never strategy.**

## DOOM adapter

The current adapter exposes:

- global player/resource state;
- a variable entity collection;
- a variable geometry collection;
- reward and terminal state;
- primitive input actuation.

It does not expose tactical macros.

The current browser engine is a pinned Chocolate Doom/Freedoom research runtime borrowed for bootstrap convenience. A project-owned minimal telemetry build remains a milestone.

## Transfer invariants

The test suite now checks or is designed to check:

- record permutation invariance;
- field/action reordering;
- renamed identifiers with stable descriptions;
- range/unit rescaling;
- request-time action cardinality;
- typed numeric action parameters;
- temporal channel influence;
- action-conditioned attention;
- compiled semantic-vector influence;
- teacher-off neural inference.

Future tests should add:

- semantic description paraphrases with actual MiniLM compilation;
- collection renaming;
- missing fields;
- distractor records;
- unseen objectives;
- cross-domain environments.

## Current complexity

At the present configuration the fast model has 6,803 trainable parameters.

A GitHub Actions CPU microbenchmark with 768 records and 12 actions is approximately 5.5 ms p50 / 7.4 ms p95. Candidate cardinality scales without increasing parameter count.

Those numbers describe compute behavior, not policy quality.

## Next architectural milestones

### Calibration

Add held-out Brier score, log score, ECE/reliability diagrams and temperature/isotonic calibration. Ensemble disagreement should be validated against actual error rather than assumed useful.

### Record identity / recurrence

Global temporal deltas are now real, but collections are still encoded independently each tick. Add generic record identity association and recurrent/set-memory variants.

### Offline distillation checkpoint

Collect state/action/teacher/reward traces, train the small controller offline, and ship a portable checkpoint rather than beginning every environment from random weights.

### Multi-environment transfer suite

DOOM alone cannot establish generality. Add structured environments with unrelated semantics and action types.

### Owned engine artifact

Build the minimal telemetry ABI from pinned open source so the project no longer relies on a neighboring prebuilt research runtime.
