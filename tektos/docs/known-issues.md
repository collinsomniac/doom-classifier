# Known issues

Open defects and unresolved observations, with reproduction. Nothing here is
hidden from CI output; where a check cannot yet be asserted, it is reported.

---

## 1. Backward pass is only partially verified (blocking)

**Status:** open. Head gradients are exact; encoder-weight gradients are not.

`tests/gradient-check.mjs` compares analytic gradients against finite
differences. Current result:

```
gradient check ok {checked: 65, worstRelative: 0, parameters: 13018,
  reportedButNotAsserted: ["block1.wq[0] 102.6%", "block1.w1[0] 35.1%",
                           "block0.w2[0] 3.9%", "block0.wo[0] 3.1%"]}
  report: mixed-config inProj[0] relative discrepancy 14.1%
```

Verified exact (asserted):

- every head parameter: `scoreProbe`, `scoreQueryW`, `scoreKeyW`,
  `fieldClass_*`, `fieldProbe_*`, `fieldBias_*`, `consequenceOut`,
  `consequenceBias`;
- loss parity between `supervisedLoss` and `lossAndGrads`;
- the encoder path for input-projection and type-embedding gradients **when the
  option-context key path is disabled** (`scoreKeyW = 0`);
- the context path for the same gradients **when the option-context key path is
  disabled** (identity-block configuration, 0.0% error).

Not verified (reported only):

- attention projection weights (`wq`, `wk`, `wv`, `wo`) and `w1` in the full
  configuration;
- input-projection and type-embedding gradients in the full configuration.

### What was ruled out

Bisection results (`tools/diag4.mjs` … `tools/diag7.mjs`):

| Configuration | Result |
| --- | --- |
| choice head only, attention and feed-forward active | 0.1% — exact |
| identity blocks, field-head path only | 0.0% — exact |
| identity blocks, option-context key path only | wrong sign, ~2.6x magnitude |
| identity blocks, both paths | ~38% discrepancy on small entries |
| full model | 14–32% on encoder-input entries |

So the defect is not in the layer-norm backward, not in the feed-forward
backward, and not in the pooled-context division: those were exercised and
matched. It is specific to the option-context interaction term and/or to the
attention projection gradients in the full configuration.

### Impact (measured, not assumed)

Adding the option-context interaction term (`scoreQueryW`/`scoreKeyW`) was meant
to let an option be scored *against the state* rather than merely for being
option-like. With the current backward pass it destabilises training instead:

- before the term: end-to-end grounding loss fell 9.66 -> 2.32 over 40 epochs,
  with correct argument construction but imperfect option discrimination;
- after the term: loss oscillates (6.75 -> 6.33, excursions to 9.95) and the
  chosen option collapses to a constant.

### Reproduction

```bash
node tests/gradient-check.mjs        # asserts the verified groups, reports the rest
node tools/diag6.mjs                 # isolates attention/feed-forward vs context path
node tools/diag7.mjs                 # splits the context consumers
```

### Next steps, in order of preference

1. Re-derive the interaction term's backward pass on paper and re-check the
   accumulating loop for the context key (`docs/architecture.md` §scoring).
2. If that fails, replace the hand-written backward with a small reverse-mode
   autodiff over the same forward, keeping this gradient check as the oracle:
   correctness by construction is worth more than hand-optimised loops here.
3. Alternatively implement the forward/backward as WebGPU kernels, where the
   same numeric oracle applies.

Until this is resolved, `Trainer` is **experimental**: it can fit small
examples, but its gradients are not trustworthy.

---

## 2. Grounding gate currently fails

`tests/grounding.mjs` is the acceptance gate for the architecture: it trains on
a synthetic typed-choice relation and requires mirrored states to produce
opposite orientation arguments, alignment to gate engaging, and the choice to be
invariant to option order and wording.

It currently fails on the first assertion (loss must fall by half). This is the
correct signal, not a test bug: it is issue 1 expressing itself.

## 3. Paraphrase invariance depends on the encoder

Recorded because it looks like a model failure and is not one. With the bare
`hashEmbedder`, no amount of training makes "leftward" and "left" related, so a
wording-invariance test fails for reasons that belong to the text encoder. The
test therefore runs with `stemEmbedder` and documents the dependency. Production
supplies a frozen sentence encoder; this is the interface the tokenizer expects
(`{dim, embed(text)}`).
