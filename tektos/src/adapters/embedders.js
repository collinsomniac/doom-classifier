// Text embedders. The tokenizer accepts any object with `{dim, embed(text)}`,
// so the decision core never depends on a particular encoder.
//
// Production supplies a frozen sentence encoder (cached per string). The two
// here exist so the architecture can be tested without one:
//
//   hashEmbedder  — no semantics; each word is an independent direction.
//   stemEmbedder  — crudely semantic: inflections and the common derivational
//                   suffixes are stripped, so "leftward" ~ "left" ~ "lefts".
//
// The distinction matters. Paraphrase invariance is a property of the encoder,
// not of the decision architecture, so tests that demand it must run with an
// encoder that can represent it.

import {hashEmbedder} from "../core/tokens.js";

const SUFFIXES = ["wards", "ward", "wise", "ness", "ingly", "edly", "ing", "ers", "er", "ed", "es", "ly", "s"];

export function stem(word) {
  let w = String(word).toLowerCase();
  for (const suffix of SUFFIXES) {
    if (w.length > suffix.length + 2 && w.endsWith(suffix)) {
      w = w.slice(0, -suffix.length);
      break;
    }
  }
  return w;
}

export function stemEmbedder(dim = 32) {
  const cache = new Map();
  const hash = str => {
    let h = 2166136261 >>> 0;
    for (let i = 0; i < str.length; i++) {
      h ^= str.charCodeAt(i);
      h = Math.imul(h, 16777619) >>> 0;
    }
    return h >>> 0;
  };
  return {
    dim,
    name: "stem",
    embed(text) {
      const key = String(text ?? "");
      const hit = cache.get(key);
      if (hit) return hit;
      const out = new Float32Array(dim);
      const words = key.toLowerCase().match(/[a-z0-9]+/g) || [];
      for (const word of words) {
        const h = hash(stem(word));
        out[h % dim] += 1;
        out[(h >>> 8) % dim] += 0.5;
      }
      let norm = 0;
      for (const v of out) norm += v * v;
      norm = Math.sqrt(norm) || 1;
      for (let i = 0; i < dim; i++) out[i] /= norm;
      cache.set(key, out);
      return out;
    }
  };
}

export {hashEmbedder};
