// The decision model: one shared encoder, one runtime scoring path.
//
//   tokens ──► [ shared encoder ] ──┬─► choice scoring        (runtime)
//                                   ├─► argument construction (runtime)
//                                   └─► consequence prediction (auxiliary)
//
// Scoring is candidate-conditioned: a score is produced for a supplied choice
// by reading that choice's own token, so nothing is compiled against a fixed
// vocabulary of actions. This is what lets a new, renamed or reordered option
// set be handled without retraining.

import {addBias, argmax, matmul, mulberry32, softmax} from "./tensor.js";
import {TokenType} from "./tokens.js";
import {FieldKind} from "./schema.js";

function heInit(size, fanIn, rng) {
  const w = new Float32Array(size);
  const scale = Math.sqrt(2 / Math.max(1, fanIn));
  for (let i = 0; i < size; i++) w[i] = (rng() * 2 - 1) * scale;
  return w;
}

function layerNorm(x) {
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
  return out;
}

export class DecisionModel {
  constructor({schema, d = 64, layers = 2, heads = 4, embeddingDim = 32, consequenceHorizons = [], seed = 20260930} = {}) {
    if (!schema) throw new Error("DecisionModel requires a schema");
    if (d % heads !== 0) throw new Error("d must be divisible by heads");
    this.schema = schema;
    this.d = d;
    this.layers = layers;
    this.heads = heads;
    this.headDim = d / heads;
    this.embeddingDim = embeddingDim;
    // Accept either explicit horizons or a count, so callers can write
    // `consequenceHorizons: 2` for "two future measurements".
    this.consequenceHorizons = Array.isArray(consequenceHorizons)
      ? [...consequenceHorizons]
      : Array.from({length: Math.max(0, Math.floor(Number(consequenceHorizons) || 0))}, (_, i) => i + 1);
    this.updates = 0;
    const rng = mulberry32(seed);
    this.params = {};

    const types = Object.values(TokenType);
    this.typeIndex = new Map(types.map((t, i) => [t, i]));
    this.params.typeEmbedding = heInit(types.length * d, d, rng);

    // Per-token content projection: text embedding + typed numeric slots.
    this.inDim = embeddingDim + 4;
    this.params.inProj = heInit(d * this.inDim, this.inDim, rng);
    this.params.inBias = new Float32Array(d);

    this.blocks = [];
    for (let l = 0; l < layers; l++) {
      this.blocks.push({
        wq: heInit(d * d, d, rng), wk: heInit(d * d, d, rng), wv: heInit(d * d, d, rng), wo: heInit(d * d, d, rng),
        w1: heInit(2 * d * d, d, rng), b1: new Float32Array(2 * d),
        w2: heInit(d * 2 * d, 2 * d, rng), b2: new Float32Array(d)
      });
    }

    // Choice scoring: one learned probe over the final token states.
    this.params.scoreProbe = new Float32Array(d);
    for (let i = 0; i < d; i++) this.params.scoreProbe[i] = (rng() * 2 - 1) * 0.5;
    this.params.scoreBias = new Float32Array(1);
    // Bilinear option-context interaction. A single probe over the option token
    // can only express "how option-like is this token"; relating an option to
    // the *state* (this option turns left because the target is to the left)
    // needs the option and the context to interact. This is that term.
    this.params.scoreQueryW = heInit(d * d, d, rng);
    this.params.scoreKeyW = heInit(d * d, d, rng);

    // Argument construction: one probe per option field, plus per-class weights.
    this.fieldSpecs = schema.optionFields.map(field => {
      const slots = field.kind === FieldKind.ENUM
        ? field.values.length + 1 // trailing "unset"
        : field.kind === FieldKind.BOOL ? 2 : 1;
      const key = field.id;
      const probe = new Float32Array(d);
      for (let i = 0; i < d; i++) probe[i] = (rng() * 2 - 1) * 0.5;
      this.params[`fieldProbe_${key}`] = probe;
      this.params[`fieldClass_${key}`] = heInit(d * d, d, rng);
      this.params[`fieldBias_${key}`] = new Float32Array(slots);
      return {field, key, slots};
    });

    // Optional consequence branch: predict future measurements from context.
    const horizonCount = this.consequenceHorizons.length;
    this.params.consequenceOut = horizonCount ? heInit(horizonCount * d, d, rng) : null;
    this.params.consequenceBias = horizonCount ? new Float32Array(horizonCount) : null;
  }

  parameterCount() {
    let n = 0;
    for (const value of Object.values(this.params)) if (value) n += value.length;
    for (const block of this.blocks) for (const value of Object.values(block)) n += value.length;
    return n;
  }

  encodeToken(token) {
    const input = new Float32Array(this.inDim);
    if (token.embedding) input.set(token.embedding.subarray(0, this.embeddingDim));
    input[this.embeddingDim + 0] = Number.isFinite(token.value) ? token.value : 0;
    input[this.embeddingDim + 1] = Number.isFinite(token.numeric) ? Number(token.numeric) : 0;
    input[this.embeddingDim + 2] = Number.isFinite(token.count) ? Math.tanh(Number(token.count) / 8) : 0;
    input[this.embeddingDim + 3] = token.type === TokenType.CHOICE ? 1 : 0;
    const projected = addBias(matmul(input, this.params.inProj, this.d, this.inDim), this.params.inBias);
    const typeOffset = (this.typeIndex.get(token.type) ?? 0) * this.d;
    for (let i = 0; i < this.d; i++) projected[i] += this.params.typeEmbedding[typeOffset + i];
    return projected;
  }

  encode(tokens) {
    const n = tokens.length;
    let states = tokens.map(t => this.encodeToken(t));
    for (const block of this.blocks) {
      const projected = states.map(s => {
        const nx = layerNorm(s);
        return {q: matmul(nx, block.wq, this.d, this.d), k: matmul(nx, block.wk, this.d, this.d), v: matmul(nx, block.wv, this.d, this.d)};
      });
      const scale = 1 / Math.sqrt(this.headDim);
      const next = new Array(n);
      for (let i = 0; i < n; i++) {
        const scores = new Float32Array(n);
        let peak = -Infinity;
        for (let j = 0; j < n; j++) {
          let dot = 0;
          for (let h = 0; h < this.heads; h++) {
            const base = h * this.headDim;
            let s = 0;
            for (let k = 0; k < this.headDim; k++) s += projected[i].q[base + k] * projected[j].k[base + k];
            dot += s * scale;
          }
          scores[j] = dot;
          if (dot > peak) peak = dot;
        }
        let sum = 0;
        for (let j = 0; j < n; j++) {
          scores[j] = Math.exp(scores[j] - peak);
          sum += scores[j];
        }
        const mixed = new Float32Array(this.d);
        for (let j = 0; j < n; j++) {
          const w = scores[j] / (sum || 1), vj = projected[j].v;
          for (let k = 0; k < this.d; k++) mixed[k] += w * vj[k];
        }
        const attended = matmul(mixed, block.wo, this.d, this.d);
        const residual = new Float32Array(this.d);
        for (let k = 0; k < this.d; k++) residual[k] = states[i][k] + attended[k];
        const hidden = addBias(matmul(layerNorm(residual), block.w1, 2 * this.d, this.d), block.b1);
        for (let k = 0; k < hidden.length; k++) hidden[k] = Math.tanh(hidden[k]);
        const out = addBias(matmul(hidden, block.w2, this.d, 2 * this.d), block.b2);
        const merged = new Float32Array(this.d);
        for (let k = 0; k < this.d; k++) merged[k] = residual[k] + out[k];
        next[i] = merged;
      }
      states = next;
    }
    return states;
  }

  contextVector(states, tokens) {
    const context = new Float32Array(this.d);
    let count = 0;
    for (let i = 0; i < tokens.length; i++) {
      if (tokens[i].type === TokenType.CHOICE) continue;
      const s = states[i];
      for (let k = 0; k < this.d; k++) context[k] += s[k];
      count++;
    }
    if (count) for (let k = 0; k < this.d; k++) context[k] /= count;
    return context;
  }

  forward(stream) {
    const states = this.encode(stream.tokens);
    const context = this.contextVector(states, stream.tokens);
    const probe = this.params.scoreProbe;
    const contextKey = matmul(context, this.params.scoreKeyW, this.d, this.d);
    const choiceScores = [];
    for (let i = 0; i < stream.tokens.length; i++) {
      if (stream.tokens[i].type !== TokenType.CHOICE) continue;
      const state = states[i];
      const query = matmul(state, this.params.scoreQueryW, this.d, this.d);
      let interaction = 0;
      for (let k = 0; k < this.d; k++) interaction += query[k] * contextKey[k];
      let dot = 0;
      for (let k = 0; k < this.d; k++) dot += probe[k] * state[k];
      choiceScores.push(interaction / Math.sqrt(this.d) + dot + this.params.scoreBias[0]);
    }
    const entityTokens = [];
    for (let i = 0; i < stream.tokens.length; i++) {
      if (stream.tokens[i].type === TokenType.ENTITY) entityTokens.push({token: stream.tokens[i], state: states[i]});
    }
    const fields = {};
    for (const spec of this.fieldSpecs) {
      const {field, key, slots} = spec;
      const probeVector = this.params[`fieldProbe_${key}`];
      let dot = 0;
      for (let k = 0; k < this.d; k++) dot += probeVector[k] * context[k];
      if (field.kind === FieldKind.REF) {
        const query = matmul(context, this.params[`fieldClass_${key}`], this.d, this.d);
        const scores = entityTokens.map(entry => {
          let s = 0;
          for (let k = 0; k < this.d; k++) s += query[k] * entry.state[k];
          return s / Math.sqrt(this.d) + dot;
        });
        fields[key] = {kind: field.kind, scores, entityIds: entityTokens.map(e => e.token.id)};
      } else {
        const w = this.params[`fieldClass_${key}`], bias = this.params[`fieldBias_${key}`];
        const logits = new Float32Array(slots);
        for (let c = 0; c < slots; c++) {
          const base = c * this.d;
          let s = bias[c];
          for (let k = 0; k < this.d; k++) s += w[base + k] * context[k];
          logits[c] = s + dot;
        }
        fields[key] = {kind: field.kind, logits};
      }
    }
    const consequence = this.params.consequenceOut
      ? Array.from(addBias(matmul(context, this.params.consequenceOut, this.consequenceHorizons.length, this.d), this.params.consequenceBias))
      : null;
    return {context, choiceScores, fields, consequence, states};
  }

  // Runtime decision. One pass; no teacher, no search.
  decide(stream, {allowed = null, temperature = 1} = {}) {
    const out = this.forward(stream);
    const probs = softmax(out.choiceScores, temperature);
    const args = [];
    for (const spec of this.fieldSpecs) {
      const {field, key} = spec;
      const entry = out.fields[key];
      if (field.kind === FieldKind.REF) {
        const p = entry.scores.length ? softmax(entry.scores, 1) : [];
        const best = entry.scores.length ? argmax(p) : -1;
        args.push({
          id: field.id,
          kind: field.kind,
          value: best >= 0 ? entry.entityIds[best] : null,
          confidence: best >= 0 ? p[best] : 0,
          provenance: best >= 0 ? {collection: field.collection, entity: entry.entityIds[best]} : null
        });
      } else if (field.kind === FieldKind.ENUM) {
        const p = softmax(Array.from(entry.logits), 1);
        const best = argmax(p);
        const unset = best === field.values.length;
        args.push({id: field.id, kind: field.kind, value: unset ? null : field.values[best].id, confidence: p[best], provenance: null});
      } else if (field.kind === FieldKind.BOOL) {
        const p = softmax(Array.from(entry.logits), 1);
        args.push({id: field.id, kind: field.kind, value: p[1] > 0.5, confidence: Math.max(p[0], p[1]), provenance: null});
      } else {
        const raw = Number(entry.logits[0]);
        const span = field.max != null && field.min != null ? field.max - field.min : 1;
        const value = field.min != null ? field.min + span / (1 + Math.exp(-raw)) : Math.tanh(raw);
        args.push({id: field.id, kind: field.kind, value, confidence: 0.5, provenance: null});
      }
    }
    let chosen = out.choiceScores.length ? argmax(probs) : -1;
    if (allowed && chosen >= 0 && allowed[chosen] === false) {
      let best = -1;
      for (let i = 0; i < probs.length; i++) {
        if (allowed[i] === false) continue;
        if (best < 0 || probs[i] > probs[best]) best = i;
      }
      chosen = best;
    }
    return {
      choiceIndex: chosen,
      choice: chosen >= 0 ? stream.candidates[chosen] : null,
      probs,
      confidence: probs.length ? Math.max(...probs) : 0,
      arguments: args,
      consequence: out.consequence,
      context: out.context
    };
  }

  exportWeights() {
    const weights = {};
    for (const [key, value] of Object.entries(this.params)) if (value) weights[key] = Array.from(value);
    weights.__blocks = this.blocks.map(b => Object.fromEntries(Object.entries(b).map(([k, v]) => [k, Array.from(v)])));
    return {
      version: 1,
      schemaSignature: this.schema.signature(),
      config: {d: this.d, layers: this.layers, heads: this.heads, embeddingDim: this.embeddingDim, consequenceHorizons: this.consequenceHorizons},
      updates: this.updates,
      weights
    };
  }

  importWeights(checkpoint) {
    if (!checkpoint || checkpoint.version !== 1) return false;
    if (checkpoint.schemaSignature !== this.schema.signature()) return false;
    const {config} = checkpoint;
    if (config.d !== this.d || config.layers !== this.layers || config.heads !== this.heads) return false;
    for (const [key, value] of Object.entries(checkpoint.weights)) {
      if (key === "__blocks") continue;
      if (!this.params[key] || this.params[key].length !== value.length) return false;
      this.params[key].set(value);
    }
    if (checkpoint.weights.__blocks) {
      if (checkpoint.weights.__blocks.length !== this.blocks.length) return false;
      this.blocks.forEach((block, i) => {
        for (const key of Object.keys(block)) block[key].set(checkpoint.weights.__blocks[i][key]);
      });
    }
    this.updates = Number(checkpoint.updates || 0);
    return true;
  }
}
