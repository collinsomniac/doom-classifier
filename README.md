# doom-classifier

Browser-native research harness for **ultra-low-latency structured-state → typed-decision neural policies**.

**Live real-DOOM lab:** https://collinsomniac.github.io/doom-classifier/

DOOM is the first real harness, not the architecture. The project asks whether a very small neural controller can consume native structured state, score request-time typed actions, preserve semantic priors, learn action consequences online, and use a larger semantic model only when that extra compute is worth paying for.

## How to use the live lab

The main page now opens the real Chocolate Doom experiment. The old synthetic arena is retained at `synthetic.html` for controlled ablations.

Recommended workflow:

1. **Boot engine.** Freedoom is included. If you own Doom, you can select a local `DOOM.WAD` / `DOOM2.WAD`; it stays in browser memory and is not uploaded.
2. **Prepare Recommended.** MiniLM compiles static schema/action meanings once, then MobileBERT supplies a bounded semantic teacher distribution that is distilled into the tiny policy.
3. **Choose a run profile.**
   - **Adaptive assisted:** neural fast path; teacher only on uncertainty/novelty.
   - **Teacher-only zero-shot:** semantic baseline; slowest.
   - **Online learning:** adaptive teacher + reward learning + exploration.
   - **Frozen neural evaluation:** no teacher, no learning, no random exploration.
4. **Optional browser fine-tune.** Run 64/128/256 real transitions, then freeze.
5. **Evaluate frozen.** This is the cleanest small-model measurement: teacher calls are disabled and weights do not change.

A manual primitive tester is also available after engine boot so FIRE/movement can be verified independently of the model.

## Current fast controller

Current model: **`SchemaSemanticValueSetNet`**, **7,316 trainable parameters**.

The controller contains:

- compiled lexical + optional MiniLM schema semantics;
- scalar/global-state encoder;
- learned temporal-delta channel;
- shared variable-record encoder;
- permutation-invariant mean/max set summary;
- action + state + temporal-conditioned record attention;
- request-time typed action embeddings;
- a **semantic-prior head** trained only by teacher distillation;
- a separate **consequence-value branch** trained only by reward TD updates;
- three bootstrapped value heads for epistemic disagreement;
- bounded replay buffer + delayed target value network.

The semantic/value separation is deliberate. Reward learning cannot overwrite the semantic-prior logits; tests assert semantic drift is exactly zero under reward-only TD updates.

## DOOM control surface

The policy receives native structured engine state, including:

- health, armor, ammunition, equipped weapon name;
- recent incoming damage and **unattributed hostile HP loss** as telemetry;
- position, velocity, heading, kills;
- coarse exploration novelty / visited cells;
- variable world-entity records;
- variable map-geometry records;
- factual entity categories such as hostile actor, collectible, and projectile/attack effect;
- named Chocolate Doom entity types where available.

The action set contains literal player inputs, including simultaneous button combinations:

- forward / back;
- turn left / right;
- strafe left / right;
- fire;
- use/interact;
- wait;
- forward+fire, back+fire, strafe+fire, turn+fire variants.

Every compound action is also represented as typed primitive fields such as `fire=1`, `strafe_left=1`. The UI therefore reports both exact compound-action probabilities and primitive marginals such as **P(fire)** and **P(strafe)**.

There is deliberately no `MOVE_TO_ENEMY`, `RETREAT`, `FACE_ENEMY`, pathfinder policy, or “enemy visible → fire” rule.

## Semantic path

Static language is moved off the tick loop:

- **MiniLM schema compiler** embeds objective, fields, enum/category meanings, collections and actions once, projects them into the fixed-size policy representation, then can be disposed.
- **MobileBERT-MNLI** is the current practical browser teacher.
- **DistilBERT-MNLI** and **DeBERTa-v3-xsmall NLI** are available as heavier experimental teachers.
- teacher state serialization is bounded and preserves categorical flags / named enum values.
- NLI actions are scored independently (entailment vs contradiction log-odds) rather than being forced into a single-label 100% distribution.

The current NLI teacher is useful but not yet a reliable affordance oracle. MobileBERT strongly associates FIRE with the objective even in deliberately empty probe states. That limitation is measured openly rather than hidden behind a hand-coded action mask.

## Training

The browser learner uses:

- consequence learning on the value branch with **4-step discounted returns**;
- experience replay + delayed target value network;
- bounded **semantic teacher replay** so new teacher states do not immediately erase prior semantic fits;
- semantic-only target synchronization after teacher refreshes;
- teacher-guided decode-temperature calibration;
- **policy-proportional training exploration** with a small uniform floor rather than epsilon-greedy argmax;
- KL-bounded fusion so learned value may move the policy away from the semantic prior only inside an explicit trust budget.

Current DOOM reward uses only signals the borrowed bridge can attribute safely:

- player-attributed kill-count increases;
- player health loss/recovery and death;
- a small step cost;
- a small first-visit spatial novelty bonus.

The bridge also exposes hostile entity HP, so the adapter reports hostile HP loss as structured telemetry. That HP loss is **not rewarded** because the bridge does not identify who caused it; monster infighting or other world events could otherwise teach false action values. The novelty bonus provides a progress signal without telling the policy which door, corridor or direction is correct.

## Current measurements

Latest GitHub Actions CPU sample, 768 structured records:

- parameters: **7,316**
- p50: **~7.45 ms**
- p95: **~9.82 ms**

Action-cardinality microbenchmark:

| candidates | p50 | p95 |
|---:|---:|---:|
| 8 | 1.18 ms | 2.47 ms |
| 32 | 1.45 ms | 2.66 ms |
| 128 | 2.95 ms | 3.41 ms |
| 256 | 5.32 ms | 5.97 ms |
| 512 | 9.37 ms | 12.95 ms |
| 1,024 | 17.61 ms | 20.00 ms |

These are compute microbenchmarks, not claims of Jev/Laya-equivalent decision quality.

### Current real-DOOM frozen benchmark

On the split semantic/value architecture, a 24-decision teacher-off evaluation from the starting encounter produced:

- **65 hostile damage**
- **1 kill**
- **0 player damage received**
- **+2.526 return**
- **24/24 fire-capable actions**
- dominant exact action: `back+fire`

A 64-transition browser fine-tune preserved the kill/damage behavior instead of collapsing away from firing, but exact-action diversity remains poor. Improving state-conditioned action separation is therefore a current research target.

## Confidence / observability

The UI keeps several concepts separate:

- **entropy** — distribution diffuseness;
- **margin** — top-1 vs top-2 separation;
- **novelty** — online state novelty;
- **epistemic disagreement** — bootstrap value-head disagreement;
- **semantic prior score** — teacher-distilled action applicability;
- **learned value score** — reward-trained consequence residual;
- **combined score** — value used for neural decoding.

Chosen-action record attention is also shown as a diagnostic, not a complete causal explanation.

## State machine boundary

The runtime state machine may manage:

- boot/readiness;
- running / paused / reset / tuning / error;
- action timing;
- episode boundaries;
- logging;
- teacher availability;
- training/evaluation mode;
- compute budgets.

It must not encode strategy.

**The state machine understands program state, never gameplay policy.**

## Run locally

    python -m http.server 8000

Then open:

- `http://localhost:8000/` — real DOOM lab
- `http://localhost:8000/synthetic.html` — legacy synthetic ablation lab

## Research invariants

1. Environment adapters may expose facts, mechanics and reward, but not behavioral rules.
2. Static semantics should be compiled; changing numeric state should dominate per-tick compute.
3. Request-time action count / parameters must not change model size.
4. Teacher usage must remain visible, optional and disableable.
5. Frozen evaluation must make zero teacher calls and zero weight updates.
6. Reward learning must not erase semantic priors.
7. Confidence must distinguish action ambiguity from epistemic uncertainty.
8. DOOM-specific facts may live in the DOOM adapter; DOOM strategy must not leak into the core policy.
9. Important mechanisms remain ablatable.
10. Transfer tests should prefer semantic equivalence over identical identifiers.

## Highest-value next experiments

- multi-seed, multi-episode real-DOOM training/evaluation;
- better decision-specialized semantic teacher or generic decision fine-tune;
- previous-action and identity-aware record memory;
- calibration metrics: Brier, log score, ECE / reliability;
- owned telemetry-enabled Chocolate Doom build with richer native events;
- non-DOOM structured environments for true transfer tests;
- portable offline-distilled tiny checkpoints;
- controlled comparison against Laya / Jev-style typed-decision baselines.
