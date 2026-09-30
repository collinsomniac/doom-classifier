# tektos

A typed-choice decision model: **information + choices in, calibrated typed
choices out**, optionally trained with an LLM teacher, at classifier latency.

The design thesis is in [`../NORTH-STAR.md`](../NORTH-STAR.md). The previous
experiment that motivated this project, including what was rejected and why, is
in [`../docs/experiment-record.md`](../docs/experiment-record.md).

## Status

Early. The core is implemented and its forward path is tested; the trainer's
backward pass is **partially verified** and is the current blocker. Read
[`docs/known-issues.md`](docs/known-issues.md) before trusting any training
result. Nothing here should be described as working end to end yet.

## What exists

| Path | What it is |
| --- | --- |
| `src/core/schema.js` | typed schema: observation fields, collections, option template, validation |
| `src/core/tokens.js` | schema-driven tokenization; text through a supplied embedder, numbers kept numeric |
| `src/core/model.js` | shared encoder, candidate-conditioned scoring, typed argument construction, optional consequence head |
| `src/core/tensor.js` | hand-written numeric utilities (no framework dependency) |
| `src/train-forward.js` | cached forward used by both training and evaluation, so they cannot drift |
| `src/loss.js` | supervised loss without gradients |
| `src/train.js` | supervised trainer (experimental) |
| `src/adapters/embedders.js` | hash and stem embedders for testing without a model |
| `tests/` | gradient check, grounding gate, core invariants |
| `tools/` | diagnostics written while isolating the gradient defect |

## Run

```bash
node tests/core-invariants.mjs     # passing: schema, tokenization, forward, checkpoint, invariances
node tests/gradient-check.mjs      # asserts verified gradients, reports the rest
node tools/diagnose-learning.mjs   # optimizer stability
node tests/grounding.mjs           # acceptance gate; currently fails (see known issues)
```

Node on iSH (the iOS sandbox) crashes in the heavier numeric loops. Run the
suites through WebKit, or in CI. `tools/stage-browser.sh` copies the tree into a
browser-servable location with `node:assert` shimmed.

## Design commitments

1. **Choices are inputs, not a vocabulary.** Reordering, renaming, adding or
   removing candidates is an input change, not a retraining event.
2. **Runtime never needs the teacher.** Teacher text is training evidence; if
   the production path can read it, training has leaked the answer.
3. **Outputs are constructed.** Enum values, booleans, bounded numbers, and
   references into supplied collections, each with confidence and provenance.
4. **Validity is structural, not tactical.** A schema may forbid impossible
   values; it must never encode "fire only when aligned".
5. **One runtime scoring path.** Auxiliary heads train; they do not silently
   arbitrate at inference.
6. **No unmeasured thresholds.** Every gate in the code must demonstrably bind,
   and every benchmark must be able to fail.
