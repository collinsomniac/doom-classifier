// Typed tokenization: turn (information, choices, schema) into a token stream.
//
// The token stream is the model's entire input surface. Two rules matter:
//
//   1. Numbers stay numeric. Forcing a bearing through a text encoder would
//      trade away precision for nothing.
//   2. Text goes through a *supplied* embedder. This module has no model
//      dependency, so the same tokenizer serves a hash embedder in tests and a
//      frozen pretrained encoder in production.

export const TokenType = Object.freeze({
  BOS: "bos",
  OBJECTIVE: "objective",
  FIELD: "field",               // observation scalar / enum / bool
  COLLECTION: "collection",     // section marker carrying its member count
  ENTITY: "entity",             // one member of a collection
  CHOICE_FIELD: "choice_field", // a template field description
  CHOICE: "choice",             // a candidate option
  TEACHER: "teacher",
  SEP: "sep"
});

// Deterministic string -> unit vector fallback. The production path swaps this
// for cached embeddings of the same strings; tests use it so the tokenizer is
// verifiable without any model.
export function hashEmbedder(dim = 32) {
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
    embed(text) {
      const key = String(text ?? "");
      const hit = cache.get(key);
      if (hit) return hit;
      const out = new Float32Array(dim);
      const words = key.toLowerCase().match(/[a-z0-9]+/g) || [];
      for (const word of words) {
        const h = hash(word);
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

export function tokenize({schema, information = {}, choices = [], teacherText = null, embedder = hashEmbedder()}) {
  const tokens = [];
  const push = token => {
    tokens.push(token);
    return token;
  };
  push({type: TokenType.BOS});
  if (schema.objective) push({type: TokenType.OBJECTIVE, text: schema.objective, embedding: embedder.embed(schema.objective)});

  for (const field of schema.fields) {
    if (!Object.prototype.hasOwnProperty.call(information, field.id)) continue;
    const raw = information[field.id];
    const token = {type: TokenType.FIELD, id: field.id, label: field.label, raw};
    const descriptor = `${field.label}${field.description ? " " + field.description : ""}`;
    token.embedding = embedder.embed(descriptor);
    if (field.kind === "number") {
      const n = Number(raw);
      token.numeric = Number.isFinite(n) ? n : 0;
      const {min: lo, max: hi, scale} = field;
      token.value = Number.isFinite(lo) && Number.isFinite(hi) && hi > lo
        ? (Math.min(hi, Math.max(lo, token.numeric)) - lo) / (hi - lo)
        : Number.isFinite(scale) && scale > 0
          ? Math.tanh(token.numeric / scale)
          : 0;
    } else if (field.kind === "enum") {
      let index = field.values.findIndex(v => v.id === String(raw));
      if (index < 0) index = field.values.length; // trailing "unset" slot
      token.value = index;
      token.descriptor = `${field.label} ${field.values[index]?.label ?? "unset"}`;
    } else if (field.kind === "bool") {
      token.value = raw ? 1 : 0;
    }
    push(token);
  }

  for (const collection of schema.collections) {
    const members = information?._collections?.[collection.id] || [];
    push({type: TokenType.COLLECTION, id: collection.id, count: members.length, embedding: embedder.embed(collection.label)});
    for (const member of members) {
      const token = {
        type: TokenType.ENTITY,
        collection: collection.id,
        id: member?.engine_record_id ?? member?.id ?? null,
        numeric: [],
        fieldIds: []
      };
      for (const field of collection.fields) {
        const value = Number(member?.[field.id]);
        token.fieldIds.push(field.id);
        token.numeric.push(Number.isFinite(value) ? value : 0);
      }
      token.embedding = embedder.embed(collection.label + " " + (member?.label || collection.label));
      push(token);
    }
  }

  for (const field of schema.optionFields) {
    const descriptor = `${field.label} (${field.kind})${field.description ? ": " + field.description : ""}`;
    push({type: TokenType.CHOICE_FIELD, id: field.id, embedding: embedder.embed(descriptor)});
  }

  const candidates = choices.map((choice, index) => {
    const text = String(choice?.text ?? choice?.description ?? choice?.label ?? choice?.id ?? index);
    const token = {
      type: TokenType.CHOICE,
      index,
      id: choice?.id != null ? String(choice.id) : String(index),
      text,
      embedding: embedder.embed(text)
    };
    // A candidate may arrive partially filled; unset fields are still predicted.
    if (choice?.params) token.params = choice.params;
    push(token);
    return {index, id: token.id, text};
  });

  if (teacherText) push({type: TokenType.TEACHER, text: String(teacherText), embedding: embedder.embed(teacherText)});
  push({type: TokenType.SEP});
  return {tokens, candidates, dim: embedder.dim, schemaSignature: schema.signature()};
}
