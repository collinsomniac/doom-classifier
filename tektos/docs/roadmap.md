# tektos roadmap

Stages are ordered by what they de-risk, not by ambition. Each stage names the
evidence that would justify moving on, and the evidence that would stop it.

## Stage 0 — Correct gradients (current)

Fix `docs/known-issues.md` item 1, either by re-deriving the interaction term's
backward pass or by replacing the hand-written backward with a small
reverse-mode autodiff over the same forward, using `tests/gradient-check.mjs` as
the oracle.

**Proceed when:** every parameter group matches finite differences within the
Float32 noise floor, and `tests/grounding.mjs` passes.

**Stop if:** the forward itself is unstable (loss oscillation that persists with
correct gradients and a conservative learning rate) — then revisit the
architecture, not the optimizer.

## Stage 1 — Grounded typed choice, one environment

Train on DOOM through an adapter, with the same protocol as the predecessor
benchmark: mirrored grounding probes, correct-vs-wrong turn rate, fire contrast,
state-action dependence, and multiple seeds.

**Proceed when:** the mirrored turn preference is materially positive and the
model handles a candidate set it was not trained on (an added, renamed or
reordered option).

**Stop if:** grounding only appears after DOOM-specific feature engineering —
that would mean the schema is doing the work rather than the model.

## Stage 2 — Teacher text

Add the third input: an LLM teacher supplying a distribution over candidates
plus a rationale used only for an auxiliary objective. Then measure student
performance with the teacher present and removed, at equal label budgets.

**Proceed when:** teacher-trained models reach the same grounding with fewer
labels, and no runtime path depends on teacher text.

**Stop if:** the student only works when the teacher is in the loop.

## Stage 3 — Consequence learning and credit

Exercise the multi-horizon consequence head on the exact-state interventions the
predecessor environment already provides, and measure whether it improves
sample efficiency or grounding at equal data.

**Proceed when:** consequence learning shortens the horizon needed for
orientation to receive credit (the specific failure of the predecessor).

## Stage 4 — Scale and the phone

Only after the above: measure what parameter count buys with matched data, and
measure adapter training in the browser on the target device (memory, latency,
thermal behaviour). Treat the 500M-class backbone as a hypothesis to test, not a
plan to execute.

## Cross-cutting rules

- No single-run claims. Report variance across seeds.
- No threshold in the code that has not been shown to bind.
- Every benchmark must be able to fail, and a failure must be preserved as a
  finding rather than tuned away.
- Environment-specific code lives in adapters; the core stays environment-blind.
