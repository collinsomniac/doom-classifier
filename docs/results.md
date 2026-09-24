# Experiment results

## Synthetic baseline — 2026-09-23

CI benchmark command: node tests/benchmark.mjs

The current test uses the intentionally weak hash structural semantic scorer. Its purpose is to isolate whether the tiny residual learner can learn consequences at all before a learned semantic backbone is involved.

| phase | steps | residual training | exploration | mean reward / step | mean completed-episode return |
|---|---:|---|---|---:|---:|
| semantic-only baseline | 6,000 | off | off | -0.03711 | -3.4787 |
| online adaptation | 12,000 | on | on | -0.03022 | -2.4824 |
| frozen learned residual, different seed | 6,000 | off | off | -0.01105 | -1.4714 |

Interpretation: the linear residual is worth retaining as the minimum-cost learner. The frozen evaluation is substantially better than the semantic-only baseline, so the improvement is not explained solely by exploration during training.

This is not evidence that the policy is good in an absolute sense: all reported returns remain negative, the environment is simple, and only one deterministic benchmark configuration has been measured. It is evidence that the separable online-learning path is functioning and can be meaningfully ablated.

Next results should add multiple seeds, confidence intervals, action distributions and learned NLI backbones.


## Real DOOM browser integration — 2026-09-23

GitHub Actions browser smoke test: tests/doom-browser.spec.mjs

Verified in headless Chromium:

1. doom.html loads as a static site;
2. the pinned Chocolate Doom 3.1.1 + Freedoom 0.13.0 runtime downloads and instantiates;
3. the native bridge returns the 14-field factual observation used by the primitive policy;
4. the hash baseline executes one primitive action and advances the real engine;
5. the page switches to MobileBERT-MNLI through Transformers.js;
6. learned zero-shot scoring completes successfully;
7. MobileBERT selects a primitive action, the engine executes it, and a new trace/state is recorded.

This is an integration result, not yet a gameplay-performance result. The next useful measurement is a controlled multi-episode comparison of semantic-only, semantic + frozen residual, and online adaptation on the same real-engine scenario.
