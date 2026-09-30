// Supervised training for DecisionModel.
//
// Scope: the supervised surface only (choice scoring, argument construction,
// optional consequence regression). Advantage weighting and teacher
// distributions enter through the same interface: a target distribution over
// choices, optionally weighted per example by measured advantage.
//
// The forward pass is duplicated here in cached form so the inference path in
// model.js stays readable. tests/gradient-check.mjs asserts the two produce
// identical logits, so they cannot silently diverge.

import {forwardCached, layerNorm, softmaxRow} from "./train-forward.js";
import {TokenType} from "./core/tokens.js";
import {FieldKind} from "./core/schema.js";

function layerNormBackward(dY, cache) {
  const {mu, inv, x, n} = cache;
  const dx = new Float32Array(n);
  let sumDy = 0, sumDyX = 0;
  for (let i = 0; i < n; i++) {
    const centered = x[i] - mu;
    sumDy += dY[i];
    sumDyX += dY[i] * centered;
  }
  const dVar = sumDyX * -0.5 * inv * inv * inv;
  const dMu = -sumDy * inv + dVar * (-2 / n) * 0;
  for (let i = 0; i < n; i++) {
    const centered = x[i] - mu;
    dx[i] = dY[i] * inv + dVar * 2 * centered / n + dMu / n;
  }
  return dx;
}

function zeroGrads(model) {
  const grads = {};
  for (const [key, value] of Object.entries(model.params)) if (value) grads[key] = new Float32Array(value.length);
  grads.__blocks = model.blocks.map(block => Object.fromEntries(Object.entries(block).map(([k, v]) => [k, new Float32Array(v.length)])));
  return grads;
}

// Supervised loss and gradients.
//   targets.choices  : soft distribution over the candidate list (required)
//   targets.fields   : {fieldId: value}          (optional)
//   targets.consequence : array of numbers       (optional)
//   weight           : per-example scalar (advantage weighting)
export function lossAndGrads(model, stream, targets, {weight = 1, fieldWeight = 1, consequenceWeight = 1} = {}) {
  const cache = forwardCached(model, stream);
  const d = model.d;
  const grads = zeroGrads(model);
  let loss = 0;

  const choiceDist = Array.isArray(targets.choices) ? targets.choices : null;
  if (!choiceDist) throw new Error("targets.choices is required");
  if (choiceDist.length !== cache.choiceScores.length) throw new Error("choices target length must match candidate count");

  const probs = softmaxRow(cache.choiceScores);
  for (let i = 0; i < choiceDist.length; i++) if (choiceDist[i] > 0) loss -= weight * choiceDist[i] * Math.log(Math.max(1e-12, probs[i]));

  const dScores = new Float32Array(choiceDist.length);
  for (let i = 0; i < choiceDist.length; i++) dScores[i] = weight * (probs[i] - choiceDist[i]);

  const dStates = new Array(cache.tokens.length);
  for (let i = 0; i < cache.tokens.length; i++) dStates[i] = new Float32Array(d);
  const addToState = (index, vector, scale = 1) => {
    if (!dStates[index]) dStates[index] = new Float32Array(d);
    for (let c = 0; c < d; c++) dStates[index][c] += vector[c] * scale;
  };

  // Accumulated gradient with respect to the pooled context vector. Declared
  // here because both the choice head and the field heads contribute to it.
  const dContext = new Float32Array(d);

  // Choice head: probe term plus the bilinear option-context interaction.
  const dContextKey = new Float32Array(d);
  const scale = 1 / Math.sqrt(d);
  for (let i = 0; i < cache.choicePositions.length; i++) {
    const pos = cache.choicePositions[i];
    const probe = model.params.scoreProbe;
    const query = cache.choiceQueries[i];
    const g = dScores[i];
    for (let c = 0; c < d; c++) {
      grads.scoreProbe[c] += g * cache.states[pos][c];
      dStates[pos][c] += g * probe[c];
      // interaction = (Wq * choice) . (Wk * context) / sqrt(d)
      const dQ = g * scale * cache.contextKey[c];
      dContextKey[c] += g * scale * query[c];
      for (let k = 0; k < d; k++) {
        grads.scoreQueryW[c * d + k] += dQ * cache.states[pos][k];
        dStates[pos][k] += dQ * model.params.scoreQueryW[c * d + k];
      }
    }
  }
  for (let c = 0; c < d; c++) {
    const grad = dContextKey[c];
    if (grad === 0) continue;
    for (let k = 0; k < d; k++) {
      grads.scoreKeyW[c * d + k] += grad * cache.context[k];
      dContext[c] += grad * model.params.scoreKeyW[c * d + k];
    }
  }
  grads.scoreBias[0] += dScores.reduce((a, b) => a + b, 0);

  const addToContext = (vector, scale = 1) => {
    for (let c = 0; c < d; c++) dContext[c] += vector[c] * scale;
  };

  for (const spec of model.fieldSpecs) {
    const {field, key, slots} = spec;
    const out = cache.fieldOut[key];
    const target = targets.fields?.[key];
    if (target == null) continue;
    const probeVector = model.params[`fieldProbe_${key}`];
    let dot = 0;
    for (let c = 0; c < d; c++) dot += probeVector[c] * cache.context[c];

    if (field.kind === FieldKind.REF) {
      const positions = out.entityPositions;
      if (!positions.length) continue;
      const classProbs = softmaxRow(out.scores);
      const index = positions.findIndex((_, i2) => {
        const token = cache.tokens[positions[i2]];
        return String(token.id) === String(target);
      });
      if (index < 0) continue;
      loss -= weight * fieldWeight * Math.log(Math.max(1e-12, classProbs[index]));
      const dScoresF = new Float32Array(classProbs.length);
      for (let i = 0; i < classProbs.length; i++) dScoresF[i] = weight * fieldWeight * (classProbs[i] - (i === index ? 1 : 0));
      // The reference head scores each entity token with a query built from the
      // context; it therefore has gradients into the entity states, the query
      // projection, and the context.
      const dQuery = new Float32Array(d);
      for (let i = 0; i < positions.length; i++) {
        const pos = positions[i], scaleRow = dScoresF[i] / Math.sqrt(d);
        for (let c = 0; c < d; c++) dQuery[c] += scaleRow * cache.states[pos][c];
      }
      const wRef = model.params[`fieldClass_${key}`];
      for (let o = 0; o < d; o++) {
        const g = dQuery[o];
        if (g === 0) continue;
        for (let c = 0; c < d; c++) {
          grads[`fieldClass_${key}`][o * d + c] += g * cache.context[c];
          dContext[c] += g * wRef[o * d + c];
        }
      }
      const totalScale = dScoresF.reduce((a, b) => a + b, 0);
      for (let c = 0; c < d; c++) grads[`fieldProbe_${key}`][c] += totalScale * cache.context[c];
      for (let c = 0; c < d; c++) dContext[c] += totalScale * probeVector[c];
      continue;
    }

    const logits = Array.from(out.logits);
    const classProbs = softmaxRow(logits);
    let targetIndex;
    if (field.kind === FieldKind.ENUM) {
      targetIndex = field.values.findIndex(v => v.id === String(target));
      if (targetIndex < 0) continue;
    } else if (field.kind === FieldKind.BOOL) {
      targetIndex = target ? 1 : 0;
    } else {
      targetIndex = 0;
    }
    if (field.kind === FieldKind.NUMBER) {
      const span = field.max != null && field.min != null ? field.max - field.min : 1;
      const lo = field.min != null ? field.min : -1;
      const normalized = Math.min(1 - 1e-6, Math.max(1e-6, (Number(target) - lo) / span));
      const logit = Math.log(normalized / (1 - normalized));
      const predicted = logits[0];
      const err = predicted - logit;
      loss += weight * fieldWeight * 0.5 * err * err;
      const w = model.params[`fieldClass_${key}`];
      for (let c = 0; c < d; c++) {
        grads[`fieldClass_${key}`][c] += weight * fieldWeight * err * cache.context[c];
        dContext[c] += weight * fieldWeight * err * w[c];
      }
      grads[`fieldBias_${key}`][0] += weight * fieldWeight * err;
      for (let c = 0; c < d; c++) {
        grads[`fieldProbe_${key}`][c] += weight * fieldWeight * err * cache.context[c];
        dContext[c] += weight * fieldWeight * err * model.params[`fieldProbe_${key}`][c];
      }
      continue;
    }

    loss -= weight * fieldWeight * Math.log(Math.max(1e-12, classProbs[targetIndex]));
    const w = model.params[`fieldClass_${key}`];
    for (let cls = 0; cls < slots; cls++) {
      const g = weight * fieldWeight * (classProbs[cls] - (cls === targetIndex ? 1 : 0));
      if (g === 0) continue;
      const base = cls * d;
      for (let c = 0; c < d; c++) {
        grads[`fieldClass_${key}`][base + c] += g * cache.context[c];
        dContext[c] += g * w[base + c];
      }
      grads[`fieldBias_${key}`][cls] += g;
      for (let c = 0; c < d; c++) {
        grads[`fieldProbe_${key}`][c] += g * cache.context[c];
        dContext[c] += g * model.params[`fieldProbe_${key}`][c];
      }
    }
  }

  if (cache.consequence && Array.isArray(targets.consequence)) {
    for (let h = 0; h < model.consequenceHorizons.length; h++) {
      const predicted = cache.consequence[h];
      const err = predicted - Number(targets.consequence[h] || 0);
      loss += weight * consequenceWeight * 0.5 * err * err;
      for (let c = 0; c < d; c++) {
        grads.consequenceOut[h * d + c] += weight * consequenceWeight * err * cache.context[c];
        dContext[c] += weight * consequenceWeight * err * model.params.consequenceOut[h * d + c];
      }
      grads.consequenceBias[h] += weight * consequenceWeight * err;
    }
  }

  if (cache.contextCount) for (let c = 0; c < d; c++) dContext[c] /= cache.contextCount;
  for (let i = 0; i < cache.tokens.length; i++) {
    if (cache.tokens[i].type === TokenType.CHOICE) continue;
    addToState(i, dContext);
  }

  // ---- backprop through blocks ----
  let dState = dStates;
  for (let layer = cache.blocks.length - 1; layer >= 0; layer--) {
    const block = model.blocks[layer];
    const c = cache.blocks[layer];
    const g = grads.__blocks[layer];
    const dResidual = new Array(cache.tokens.length);
    const dAttended = new Array(cache.tokens.length);
    for (let i = 0; i < cache.tokens.length; i++) {
      const dMerged = dState[i] || new Float32Array(d);
      dResidual[i] = Float32Array.from(dMerged);
      const dOut = dMerged;
      const dHidden = new Float32Array(2 * d);
      for (let o = 0; o < d; o++) {
        const grad = dOut[o];
        g.b2[o] += grad;
        const base = o * 2 * d;
        for (let hI = 0; hI < 2 * d; hI++) {
          g.w2[base + hI] += grad * c.hidden[i][hI];
          dHidden[hI] += grad * block.w2[base + hI];
        }
      }
      const dPre = new Float32Array(2 * d);
      for (let hI = 0; hI < 2 * d; hI++) dPre[hI] = dHidden[hI] * (1 - c.hidden[i][hI] * c.hidden[i][hI]);
      const dLn2 = new Float32Array(d);
      // ln2 output is needed for the w1 gradient; recompute it rather than
      // caching another tensor per token.
      const ln2out = layerNorm(c.residual[i], {});
      for (let hI = 0; hI < 2 * d; hI++) {
        const grad = dPre[hI];
        if (grad === 0) continue;
        const base = hI * d;
        for (let cIn = 0; cIn < d; cIn++) g.w1[base + cIn] += grad * ln2out[cIn];
        for (let cIn = 0; cIn < d; cIn++) dLn2[cIn] += grad * block.w1[base + cIn];
      }
      const dResidualFromLn2 = layerNormBackward(dLn2, c.ln2Cache[i]);
      for (let cIn = 0; cIn < d; cIn++) dResidual[i][cIn] += dResidualFromLn2[cIn];
      // `attended` reaches the output twice: directly through the residual, and
      // through the second layer norm feeding the feed-forward block. It must
      // receive the whole residual gradient, not just the direct term.
      dAttended[i] = dResidual[i];
    }

    const dMixed = new Array(cache.tokens.length);
    for (let i = 0; i < cache.tokens.length; i++) {
      const dM = new Float32Array(d);
      for (let o = 0; o < d; o++) {
        const grad = dAttended[i][o];
        g.wo[o * d + 0] += 0;
        if (grad === 0) continue;
        const base = o * d;
        for (let cIn = 0; cIn < d; cIn++) {
          g.wo[base + cIn] += grad * c.mixed[i][cIn];
          dM[cIn] += grad * block.wo[base + cIn];
        }
      }
      dMixed[i] = dM;
    }

    const dQ = new Array(cache.tokens.length);
    const dK = new Array(cache.tokens.length);
    const dV = new Array(cache.tokens.length);
    for (let i = 0; i < cache.tokens.length; i++) {
      dQ[i] = new Float32Array(d);
      dK[i] = new Float32Array(d);
      dV[i] = new Float32Array(d);
    }
    const scale = 1 / Math.sqrt(model.headDim);
    const dW = new Array(cache.tokens.length);
    for (let i = 0; i < cache.tokens.length; i++) {
      const row = new Float32Array(cache.tokens.length);
      for (let j = 0; j < cache.tokens.length; j++) {
        let dot = 0;
        for (let cIn = 0; cIn < d; cIn++) dot += dMixed[i][cIn] * c.v[j][cIn];
        row[j] = dot;
        if (dot !== 0) for (let cIn = 0; cIn < d; cIn++) dV[j][cIn] += c.weights[i][j] * dMixed[i][cIn];
      }
      dW[i] = row;
    }
    for (let i = 0; i < cache.tokens.length; i++) {
      let weighted = 0;
      for (let j = 0; j < cache.tokens.length; j++) weighted += c.weights[i][j] * dW[i][j];
      for (let j = 0; j < cache.tokens.length; j++) {
        const ds = c.weights[i][j] * (dW[i][j] - weighted);
        if (ds === 0) continue;
        for (let h = 0; h < model.heads; h++) {
          const base = h * model.headDim;
          for (let cIn = 0; cIn < model.headDim; cIn++) {
            const qv = c.q[i][base + cIn], kv = c.k[j][base + cIn];
            dQ[i][base + cIn] += ds * scale * kv;
            dK[j][base + cIn] += ds * scale * qv;
          }
        }
      }
    }
    const dLn = new Array(cache.tokens.length);
    for (let i = 0; i < cache.tokens.length; i++) {
      dLn[i] = new Float32Array(d);
      for (let o = 0; o < d; o++) {
        const base = o * d;
        const gq = dQ[i][o], gk = dK[i][o], gv = dV[i][o];
        for (let cIn = 0; cIn < d; cIn++) {
          const x = c.lnOut[i][cIn];
          if (gq) g.wq[base + cIn] += gq * x;
          if (gk) g.wk[base + cIn] += gk * x;
          if (gv) g.wv[base + cIn] += gv * x;
          if (gq) dLn[i][cIn] += gq * block.wq[base + cIn];
          if (gk) dLn[i][cIn] += gk * block.wk[base + cIn];
          if (gv) dLn[i][cIn] += gv * block.wv[base + cIn];
        }
      }
    }
    const prevState = new Array(cache.tokens.length);
    for (let i = 0; i < cache.tokens.length; i++) {
      const dFromLn = layerNormBackward(dLn[i], c.lnCache[i]);
      const out = new Float32Array(d);
      for (let cIn = 0; cIn < d; cIn++) out[cIn] = dResidual[i][cIn] + dFromLn[cIn];
      prevState[i] = out;
    }
    dState = prevState;
  }

  // ---- input projection and type embedding ----
  for (let i = 0; i < cache.tokens.length; i++) {
    const grad = dState[i];
    if (!grad) continue;
    const token = cache.tokens[i];
    const input = new Float32Array(model.inDim);
    if (token.embedding) input.set(token.embedding.subarray(0, model.embeddingDim));
    input[model.embeddingDim + 0] = Number.isFinite(token.value) ? token.value : 0;
    input[model.embeddingDim + 1] = Number.isFinite(token.numeric) ? Number(token.numeric) : 0;
    input[model.embeddingDim + 2] = Number.isFinite(token.count) ? Math.tanh(Number(token.count) / 8) : 0;
    input[model.embeddingDim + 3] = token.type === TokenType.CHOICE ? 1 : 0;
    const typeOffset = (model.typeIndex.get(token.type) ?? 0) * d;
    for (let o = 0; o < d; o++) {
      const gg = grad[o];
      if (gg === 0) continue;
      grads.inBias[o] += gg;
      grads.typeEmbedding[typeOffset + o] += gg;
      const base = o * model.inDim;
      for (let cIn = 0; cIn < model.inDim; cIn++) grads.inProj[base + cIn] += gg * input[cIn];
    }
  }

  return {loss, grads, cache};
}

export class Trainer {
  constructor(model, {lr = 0.02, l2 = 1e-4, clip = 1} = {}) {
    this.model = model;
    this.lr = lr;
    this.l2 = l2;
    this.clip = clip;
    this.updates = 0;
    this.momentum = new Map();
  }

  // One supervised step on a single example.
  step(stream, targets, options = {}) {
    const {loss, grads} = lossAndGrads(this.model, stream, targets, options);
    const apply = (tensor, grad) => {
      for (let i = 0; i < tensor.length; i++) {
        let g = grad[i] + this.l2 * tensor[i];
        if (g > this.clip) g = this.clip;
        else if (g < -this.clip) g = -this.clip;
        tensor[i] -= this.lr * g;
      }
    };
    for (const [key, value] of Object.entries(this.model.params)) {
      if (value) apply(value, grads[key]);
    }
    this.model.blocks.forEach((block, i) => {
      for (const key of Object.keys(block)) apply(block[key], grads.__blocks[i][key]);
    });
    this.updates++;
    this.model.updates++;
    return loss;
  }

  // Mean loss over a batch, averaged over examples.
  evaluate(batch, options = {}) {
    if (!batch.length) return 0;
    let total = 0;
    for (const example of batch) total += lossAndGrads(this.model, example.stream, example.targets, options).loss;
    return total / batch.length;
  }

  fit(batch, {epochs = 1, shuffle = null, onEpoch = null} = {}) {
    const history = [];
    for (let epoch = 0; epoch < epochs; epoch++) {
      const order = batch.map((_, i) => i);
      if (shuffle) {
        for (let i = order.length - 1; i > 0; i--) {
          const j = Math.floor(shuffle() * (i + 1));
          [order[i], order[j]] = [order[j], order[i]];
        }
      }
      let total = 0;
      for (const index of order) total += this.step(batch[index].stream, batch[index].targets);
      const meanLoss = total / Math.max(1, order.length);
      history.push(meanLoss);
      if (onEpoch) onEpoch(epoch, meanLoss);
    }
    return history;
  }
}
