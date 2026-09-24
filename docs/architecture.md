# Working architecture

## Research target

This project is not attempting to solve perception first. The primary benchmark is:

\[
\text{structured environment state} \rightarrow \text{typed semantic decision}
\]

The reference point is the behavior demonstrated by Jev-like typed decision systems and open implementations such as Laya, while keeping the design sufficiently generic that DOOM is only one environment adapter.

## Core decomposition

The working hypothesis is that a useful low-latency controller can be decomposed into five independently measurable pieces:

1. **semantic prior** — understands field/action meaning;
2. **temporal state** — retains cheap history without resending a textual transcript;
3. **experience residual** — learns which semantically plausible actions actually work;
4. **calibrated uncertainty** — detects ambiguity / novelty and optionally gates escalation;
5. **policy-blind orchestration** — manages lifecycle and timing without encoding strategy.

The intended score is conceptually:

\[
L(a_t)=S_{\text{semantic}}(s_t,a_t)+\lambda Q_{\text{experience}}(h_t,a_t)
\]

with

\[
\pi(a_t)=\mathrm{softmax}(L(a_t)/T)
\]

The current implementation contains an intentionally weak structural semantic adapter and a tiny linear residual Q learner. This is a measurement scaffold, not the final model.

## Semantic compilation

Environment schemas and action descriptions are largely static. A future pretrained adapter should compile them once:

\[
e_i=E(\text{field metadata}_i),\quad u_j=E(\text{action metadata}_j)
\]

Runtime values can then be fused with cached representations instead of repeatedly tokenizing the same prose each decision tick.

The adapter contract therefore separates:

- **compile(schema, actions)** — slow/static path;
- **score(observation)** — bounded fast path.

A Laya-derived ONNX scorer, NLI cross-encoder, embedding scorer, or purpose-trained model can implement the same boundary.

## State machine boundary

The orchestration state machine may know model/environment readiness, running/paused/reset/error states, action scheduling, episode boundaries, logging, replay, whether learning or escalation is enabled, and compute budgets.

It must not know that low health implies retreat, an enemy implies firing, lack of ammunition implies weapon switching, or any other strategic state → action rule.

**The FSM understands program state, never strategy.**

## Online learning

Runtime adaptation should touch as few parameters as possible. The current residual is a linear Q approximator over normalized observation features and temporal deltas. It can be switched off without changing the semantic policy.

Later variants can replace it with a small MLP residual, LoRA/adapters on a frozen encoder, learned recurrent state, distributional value head, or consequence/transition prediction head.

Normal inference must not wait for training. Browser versions should train between episodes or in a separate worker unless profiling proves synchronous micro-updates are harmless.

## Uncertainty and escalation

The current lab reports normalized action entropy, top-1/top-2 probability margin, and online feature novelty.

A future escalation gate should combine them rather than equating max-softmax probability with knowledge.

Teacher-assisted runs must always be distinguished from teacher-off evaluation. The long-term metric is not only performance but declining teacher-query rate after adaptation.

## Anti-overfitting tests

A useful controller should survive field reordering, key renaming with descriptions preserved, description paraphrases, unit/scale changes, dropped fields, irrelevant distractor fields, action reordering, changed action labels, changed option cardinality, and unseen structured environments.

Schema dropout and temporal dropout should become training-time augmentations.

## DOOM integration boundary

A DOOM adapter should expose generic operations similar to:

\`\`\`text
reset() -> observation
observe() -> structured observation
step(typedAction) -> { observation, reward, done, info }
getSchema() -> field metadata
getActions() -> typed action schema
\`\`\`

Engine telemetry belongs in the adapter. Strategic interpretation does not.

A later player-parity benchmark can replace direct telemetry with perception-derived state without changing the policy interface.

## Milestones

- **M0 — runnable lab:** synthetic environment, policy API, memory, online residual, uncertainty and latency metrics.
- **M1 — real semantic adapter:** browser model adapter using ONNX Runtime Web / Transformers.js or a converted dedicated scorer.
- **M2 — DOOM WASM:** in-browser engine with structured telemetry and typed actuation.
- **M3 — teacher collection:** uncertainty-triggered escalation and distillation dataset generation.
- **M4 — consequence learning:** short-horizon transition/value prediction and optional bounded planning.
- **M5 — transfer suite:** multiple structured environments plus schema perturbation benchmarks.
