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
