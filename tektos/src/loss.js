// Supervised loss without gradients.
//
// Used for evaluation and for finite-difference gradient checks, where running
// a full backward pass per perturbation is wasted work. tests/gradient-check.mjs
// asserts it agrees exactly with the loss returned by lossAndGrads, so the two
// cannot drift apart.

import {softmax} from "./core/tensor.js";
import {FieldKind} from "./core/schema.js";
import {forwardCached} from "./train-forward.js";

export function supervisedLoss(model, stream, targets, {weight = 1, fieldWeight = 1, consequenceWeight = 1} = {}) {
  const cache = forwardCached(model, stream);
  let loss = 0;

  const choiceTarget = Array.isArray(targets.choices) ? targets.choices : null;
  if (!choiceTarget) throw new Error("targets.choices is required");
  if (choiceTarget.length !== cache.choiceScores.length) throw new Error("choices target length must match candidate count");
  const choiceProbs = softmax(cache.choiceScores, 1);
  for (let i = 0; i < choiceTarget.length; i++) {
    if (choiceTarget[i] > 0) loss -= weight * choiceTarget[i] * Math.log(Math.max(1e-12, choiceProbs[i]));
  }

  for (const spec of model.fieldSpecs) {
    const {field, key} = spec;
    const target = targets.fields?.[key];
    if (target == null) continue;
    const out = cache.fieldOut[key];
    if (field.kind === FieldKind.REF) {
      const positions = out.entityPositions;
      if (!positions.length) continue;
      const index = positions.findIndex(pos => String(cache.tokens[pos].id) === String(target));
      if (index < 0) continue;
      const probs = softmax(out.scores, 1);
      loss -= weight * fieldWeight * Math.log(Math.max(1e-12, probs[index]));
    } else if (field.kind === FieldKind.NUMBER) {
      const span = field.max != null && field.min != null ? field.max - field.min : 1;
      const lo = field.min != null ? field.min : -1;
      const normalized = Math.min(1 - 1e-6, Math.max(1e-6, (Number(target) - lo) / span));
      const logit = Math.log(normalized / (1 - normalized));
      const err = Number(out.logits[0]) - logit;
      loss += weight * fieldWeight * 0.5 * err * err;
    } else {
      const classes = field.kind === FieldKind.ENUM
        ? field.values.findIndex(v => v.id === String(target))
        : (target ? 1 : 0);
      if (classes < 0) continue;
      const probs = softmax(Array.from(out.logits), 1);
      loss -= weight * fieldWeight * Math.log(Math.max(1e-12, probs[classes]));
    }
  }

  if (cache.consequence && Array.isArray(targets.consequence)) {
    for (let h = 0; h < model.consequenceHorizons.length; h++) {
      const err = Number(cache.consequence[h]) - Number(targets.consequence[h] || 0);
      loss += weight * consequenceWeight * 0.5 * err * err;
    }
  }

  return loss;
}
