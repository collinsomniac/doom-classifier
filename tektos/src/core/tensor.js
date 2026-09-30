// Shared numeric utilities, written by hand so the model trains in plain JS or
// WebGPU without a framework dependency.

export function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function matmul(x, w, outSize, inSize) {
  const out = new Float32Array(outSize);
  for (let o = 0; o < outSize; o++) {
    let sum = 0;
    const base = o * inSize;
    for (let i = 0; i < inSize; i++) sum += w[base + i] * x[i];
    out[o] = sum;
  }
  return out;
}

export function addBias(x, b) {
  for (let i = 0; i < x.length; i++) x[i] += b[i];
  return x;
}

export function softmax(scores, temperature = 1) {
  if (!scores.length) return [];
  const t = Math.max(1e-6, temperature);
  let peak = -Infinity;
  for (const s of scores) if (s > peak) peak = s;
  const exps = new Array(scores.length);
  let sum = 0;
  for (let i = 0; i < scores.length; i++) {
    const e = Math.exp((scores[i] - peak) / t);
    exps[i] = e;
    sum += e;
  }
  if (!(sum > 0)) return scores.map(() => 1 / scores.length);
  return exps.map(e => e / sum);
}

export function logSoftmax(scores, temperature = 1) {
  return softmax(scores, temperature).map(v => Math.log(Math.max(1e-12, v)));
}

export function argmax(values) {
  let best = 0;
  for (let i = 1; i < values.length; i++) if (values[i] > values[best]) best = i;
  return best;
}

export function clamp(v, lo, hi) {
  return v < lo ? lo : v > hi ? hi : v;
}

export function crossEntropy(target, predicted) {
  let loss = 0;
  for (let i = 0; i < target.length; i++) {
    if (target[i] > 0) loss -= target[i] * Math.log(Math.max(1e-12, predicted[i]));
  }
  return loss;
}

export function mean(values) {
  if (!values.length) return 0;
  return values.reduce((a, b) => a + b, 0) / values.length;
}

export function sampleCategorical(probs, rng) {
  let r = rng();
  for (let i = 0; i < probs.length; i++) {
    r -= probs[i];
    if (r <= 0) return i;
  }
  return probs.length - 1;
}
