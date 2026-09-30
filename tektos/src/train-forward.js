// Cached forward pass, shared by the trainer (which needs intermediates for
// backprop) and by the loss-only evaluator, so the two cannot drift apart.

import {addBias, matmul} from "./core/tensor.js";
import {TokenType} from "./core/tokens.js";
import {FieldKind} from "./core/schema.js";

export function layerNorm(x, cache) {
  const n = x.length;
  let mu = 0;
  for (const v of x) mu += v;
  mu /= n;
  let variance = 0;
  for (const v of x) variance += (v - mu) * (v - mu);
  variance /= n;
  const inv = 1 / Math.sqrt(variance + 1e-5);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = (x[i] - mu) * inv;
  if (cache) {
    cache.mu = mu;
    cache.inv = inv;
    cache.x = x;
    cache.n = n;
  }
  return out;
}

export function softmaxRow(scores) {
  let peak = -Infinity;
  for (const s of scores) if (s > peak) peak = s;
  const out = new Float32Array(scores.length);
  let sum = 0;
  for (let i = 0; i < scores.length; i++) {
    out[i] = Math.exp(scores[i] - peak);
    sum += out[i];
  }
  for (let i = 0; i < out.length; i++) out[i] /= sum || 1;
  return out;
}

export function forwardCached(model, stream) {
  const tokens = stream.tokens;
  const n = tokens.length;
  const d = model.d;
  const initial = tokens.map(t => model.encodeToken(t));
  const blocks = [];
  let states = initial;

  for (const block of model.blocks) {
    const lnCache = [];
    const lnOut = states.map(s => layerNorm(s, (lnCache[lnCache.length] = {})));
    const q = lnOut.map(x => matmul(x, block.wq, d, d));
    const k = lnOut.map(x => matmul(x, block.wk, d, d));
    const v = lnOut.map(x => matmul(x, block.wv, d, d));
    const headDim = model.headDim;
    const scale = 1 / Math.sqrt(headDim);
    const weights = new Array(n);
    const mixed = new Array(n);
    for (let i = 0; i < n; i++) {
      const scores = new Float32Array(n);
      for (let j = 0; j < n; j++) {
        let dot = 0;
        for (let h = 0; h < model.heads; h++) {
          const base = h * headDim;
          let s = 0;
          for (let c = 0; c < headDim; c++) s += q[i][base + c] * k[j][base + c];
          dot += s * scale;
        }
        scores[j] = dot;
      }
      weights[i] = softmaxRow(scores);
      const m = new Float32Array(d);
      for (let j = 0; j < n; j++) {
        const w = weights[i][j], vj = v[j];
        for (let c = 0; c < d; c++) m[c] += w * vj[c];
      }
      mixed[i] = m;
    }
    const attended = mixed.map(m => matmul(m, block.wo, d, d));
    const residual = new Array(n);
    const hidden = new Array(n);
    const hiddenPre = new Array(n);
    const ln2Cache = new Array(n);
    const next = new Array(n);
    for (let i = 0; i < n; i++) {
      const r = new Float32Array(d);
      for (let c = 0; c < d; c++) r[c] = states[i][c] + attended[i][c];
      residual[i] = r;
      const pre = addBias(matmul(layerNorm(r, (ln2Cache[i] = {})), block.w1, 2 * d, d), block.b1);
      hiddenPre[i] = pre;
      const h = new Float32Array(pre.length);
      for (let c = 0; c < pre.length; c++) h[c] = Math.tanh(pre[c]);
      hidden[i] = h;
      const out = addBias(matmul(h, block.w2, d, 2 * d), block.b2);
      const merged = new Float32Array(d);
      for (let c = 0; c < d; c++) merged[c] = r[c] + out[c];
      next[i] = merged;
    }
    blocks.push({lnCache, lnOut, q, k, v, weights, mixed, attended, residual, hiddenPre, hidden, ln2Cache});
    states = next;
  }

  const context = new Float32Array(d);
  let contextCount = 0;
  for (let i = 0; i < n; i++) {
    if (tokens[i].type === TokenType.CHOICE) continue;
    for (let c = 0; c < d; c++) context[c] += states[i][c];
    contextCount++;
  }
  if (contextCount) for (let c = 0; c < d; c++) context[c] /= contextCount;

  const choicePositions = [];
  const choiceScores = [];
  const choiceQueries = [];
  const contextKey = matmul(context, model.params.scoreKeyW, d, d);
  for (let i = 0; i < n; i++) {
    if (tokens[i].type !== TokenType.CHOICE) continue;
    choicePositions.push(i);
    const query = matmul(states[i], model.params.scoreQueryW, d, d);
    choiceQueries.push(query);
    // Accumulate in the same order as the inference path so the two forward
    // implementations agree to Float32 precision.
    let interaction = 0;
    for (let c = 0; c < d; c++) interaction += query[c] * contextKey[c];
    let dot = 0;
    for (let c = 0; c < d; c++) dot += model.params.scoreProbe[c] * states[i][c];
    choiceScores.push(interaction / Math.sqrt(d) + dot + model.params.scoreBias[0]);
  }

  const entityPositions = [];
  for (let i = 0; i < n; i++) if (tokens[i].type === TokenType.ENTITY) entityPositions.push(i);

  const fieldOut = {};
  for (const spec of model.fieldSpecs) {
    const {field, key} = spec;
    const probeVector = model.params[`fieldProbe_${key}`];
    let dot = 0;
    for (let c = 0; c < d; c++) dot += probeVector[c] * context[c];
    if (field.kind === FieldKind.REF) {
      const query = matmul(context, model.params[`fieldClass_${key}`], d, d);
      const scores = entityPositions.map(pos => {
        let s = 0;
        for (let c = 0; c < d; c++) s += query[c] * states[pos][c];
        return s / Math.sqrt(d) + dot;
      });
      fieldOut[key] = {kind: field.kind, scores, entityPositions, query};
    } else {
      const w = model.params[`fieldClass_${key}`], bias = model.params[`fieldBias_${key}`];
      const logits = new Float32Array(spec.slots);
      for (let cls = 0; cls < spec.slots; cls++) {
        const base = cls * d;
        let s = bias[cls];
        for (let c = 0; c < d; c++) s += w[base + c] * context[c];
        logits[cls] = s + dot;
      }
      fieldOut[key] = {kind: field.kind, logits};
    }
  }

  const consequence = model.params.consequenceOut
    ? addBias(matmul(context, model.params.consequenceOut, model.consequenceHorizons.length, d), model.params.consequenceBias)
    : null;

  return {tokens, initial, blocks, states, context, contextCount, choicePositions, choiceScores, choiceQueries, contextKey, entityPositions, fieldOut, consequence};
}
