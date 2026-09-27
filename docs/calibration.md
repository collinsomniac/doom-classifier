# Decision probability calibration

A normalized action distribution is not automatically a calibrated probability distribution.

For a calibrated multiclass decision model, predictions around 0.7 should be correct at approximately the corresponding empirical frequency under the evaluation definition. Top-1 accuracy alone cannot establish this property.

## Three different operations in this project

### 1. NLI label-bias correction

The default MobileBERT semantic teacher is evaluated once on a state-withheld premise. Its state-independent action-label logits are subtracted from state-conditioned logits. This reduces generic lexical/action priors such as an unconditional FIRE preference.

This is contextual bias correction, not probability calibration.

### 2. Student temperature fit

During teacher distillation the small semantic head adjusts its decode temperature to better match the teacher distribution. This improves teacher/student distribution fidelity.

This is distillation fit, not empirical correctness calibration.

### 3. Held-out decision calibration

The final fused typed-action logits can be passed through an optional temperature calibrator trained on labelled held-out decisions. The calibrator is post-fusion and therefore cannot change action ranking.

The current implementation measures NLL, Brier score, ECE, maximum calibration error, reliability bins, accuracy and mean confidence. A fitted calibrator is stored in policy checkpoints.

If a task has no defensible decision labels, the calibration layer stays inactive and the UI reports `unverified`.

## DOOM

DOOM gives strong causal reward telemetry but does not yet provide a unique ground-truth label for the best one-step action among every candidate. Reward and Q-learning can improve the policy, but treating those signals as direct categorical correctness labels would conflate value estimation with probability calibration.

Future stronger DOOM calibration experiments could use reproducible state snapshots plus counterfactual rollouts of every candidate action, or a separately specified labelled decision suite. Until then the DOOM probabilities remain normalized model beliefs rather than empirically verified correctness frequencies.

## Why this matters for Jev comparisons

TypeSafe describes Jev as returning calibrated probabilities/confidence, not only a top-ranked typed answer. Any serious comparison therefore needs at least three independent axes:

1. decision/ranking quality;
2. probability calibration/reliability;
3. latency and compute cost.

A classifier or contrastive scorer can win top-1 while still be badly overconfident. Conversely, temperature calibration can materially improve proper scoring metrics without changing top-1 accuracy at all.
