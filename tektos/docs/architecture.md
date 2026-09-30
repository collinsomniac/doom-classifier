# tektos architecture

The thesis is [`../../NORTH-STAR.md`](../../NORTH-STAR.md). This file is the
implementation view: what the pieces are, why they are shaped that way, and
where the current defect sits.

## Data flow

```
information ─┐
schema ──────┼─► tokenize ──► shared encoder ──┬─► choice scoring        (runtime, one pass)
choices ─────┘                                  ├─► argument construction (runtime)
teacher ──(training only)───────────────────────┴─► consequence prediction (auxiliary)
```

## Tokenization (`src/core/tokens.js`)

The token stream is the model's entire input surface. Two rules:

1. **Numbers stay numeric.** A bearing is carried as a normalized scalar in a
   typed slot, not as text. Text embedding of numbers would discard the
   precision the decision needs.
2. **Text goes through a supplied embedder** (`{dim, embed(text)}`). The core has
   no model dependency, so a hash embedder serves tests and a frozen sentence
   encoder serves production, and the invariants are testable without one.

Token types: `bos`, `objective`, `field`, `collection`, `entity`,
`choice_field`, `choice`, `teacher`, `sep`. The type is added as a learned tag
embedding, which is the modern continuation of the control-code idea.

## Schema (`src/core/schema.js`)

Declares observation fields, observation collections, and the **option
template** — the fields a concrete choice may set. Everything downstream is
generated from it; nothing is compiled against a particular environment.

`validateChoice` is deliberately *structural*: types, enum membership, bounds,
and references into supplied collections. It must never encode tactical policy
("fire only when aligned") — that is the behaviour the model has to learn.

## Model (`src/core/model.js`)

A small transformer over the token stream, then:

**Choice scoring.** Each choice token is scored by

```
score(i) = (Wq · state_i) · (Wk · context) / sqrt(d)   // interaction with the state
         + probe · state_i                              // how option-like is this token
         + bias
```

The interaction term exists because a state-independent probe can only express
"this looks like an option"; relating an option to the *situation* requires the
option and the context to meet. The first version lacked this term and the model
could not discriminate options that differed only in their relation
(`tests/grounding.mjs` run of 2026-09-30).

Scoring is **candidate-conditioned**: the score comes from reading the
candidate's own token, so reordering, renaming, adding or removing candidates is
an input change, not a retraining event. This is the Jev/Laya capability, and it
is what `tests/core-invariants.mjs` pins down.

**Argument construction.** One head per option field, reading the pooled
context:

| Field kind | Mechanism | Output |
| --- | --- | --- |
| enum | softmax over declared values plus an `unset` slot | value or explicit null |
| bool | two-class softmax | boolean |
| number | bounded scalar (requires min and max in the schema) | number |
| ref | attention over entity tokens of the named collection | object id + provenance |

**Consequence prediction.** Optional multi-horizon regression from the context.
It trains the representation; it does not arbitrate at inference. This is the
part of the design that faces the credit-assignment problem the DOOM
experiment died on, and it is not yet exercised.

## Training (`src/train.js`, `src/loss.js`, `src/train-forward.js`)

The cached forward used by training lives in `train-forward.js` and is shared
with the loss-only evaluator, so evaluation cannot silently measure a different
objective than training. `tests/gradient-check.mjs` asserts that the two agree.

Targets are a distribution over candidates (so a teacher can supply soft
labels), optional field values, and optional future measurements. Advantage
weighting is a per-example scalar — the classification-style RL surface.

**The backward pass is not fully verified.** See
[`known-issues.md`](known-issues.md). Head gradients are exact; attention
projection gradients in the full configuration are not, and the interaction term
destabilises training until that is fixed. `Trainer` is experimental.

## Deliberate omissions

Compared with the predecessor experiment these were removed and must not come
back without evidence:

- a separate semantic prior fused by a KL budget;
- a typed projection of the label onto action columns (it fabricated
  preferences from ties);
- ensemble critic gates with thresholds that never visibly bound;
- a linear control head over hand-picked relational features.
