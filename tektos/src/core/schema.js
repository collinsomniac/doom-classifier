// Schema: the single source of truth for typed decisions.
//
// A Schema describes observation fields, observation collections (entity sets),
// and the *option template*: the fields a concrete choice may set. Everything
// downstream (tokenization, heads, decoding, validation) is generated from it,
// so nothing in the model is compiled against a particular environment.

export const FieldKind = Object.freeze({
  NUMBER: "number", // continuous, optionally with min/max or a scale
  ENUM: "enum",     // finite labelled set
  BOOL: "bool",
  REF: "ref"        // reference into a named observation collection
});

const KINDS = new Set(Object.values(FieldKind));

function fail(message) {
  throw new Error("schema: " + message);
}

function normalizeField(raw, {name, pointer = false} = {}) {
  if (!raw || !raw.id) fail(`${name} has a field without an id`);
  const kind = String(raw.kind || "").toLowerCase();
  if (!KINDS.has(kind)) fail(`${name}.${raw.id} has unknown kind "${raw.kind}"`);
  const field = {id: String(raw.id), kind, label: raw.label || String(raw.id), description: raw.description || ""};
  if (kind === FieldKind.ENUM) {
    const values = raw.values;
    if (!Array.isArray(values) || values.length < 1) fail(`${name}.${raw.id} enum needs values`);
    field.values = values.map(v => {
      if (v != null && typeof v === "object") {
        const id = v.id ?? v.value ?? v.label;
        if (id == null) fail(`${name}.${raw.id} enum value without an id`);
        return {id: String(id), label: String(v.label ?? id)};
      }
      return {id: String(v), label: String(v)};
    });
  }
  if (kind === FieldKind.NUMBER) {
    field.min = Number.isFinite(raw.min) ? Number(raw.min) : null;
    field.max = Number.isFinite(raw.max) ? Number(raw.max) : null;
    field.scale = Number.isFinite(raw.scale) ? Number(raw.scale) : null;
    // A number in the option template must be bounded: the model produces it
    // through a bounded scalar head, so an unbounded value has no defined
    // output representation.
    if (pointer && !(Number.isFinite(field.min) && Number.isFinite(field.max) && field.max > field.min)) {
      fail(`${name}.${raw.id}: option number fields require finite min and max`);
    }
  }
  if (kind === FieldKind.REF) {
    field.collection = raw.collection ? String(raw.collection) : null;
    if (!field.collection) fail(`${name}.${raw.id} ref needs a collection`);
  }
  // Option fields only: whether a concrete choice MUST set this field. A model
  // may emit a choice missing optional fields; validation decides validity.
  if (pointer) field.required = raw.required !== false;
  return field;
}

export class Schema {
  constructor(raw) {
    if (!raw || !Array.isArray(raw.fields)) fail("raw schema needs a fields array");
    const version = raw.version == null ? 1 : Number(raw.version);
    if (!Number.isInteger(version) || version < 1) fail("version must be a positive integer");
    this.version = version;
    this.id = String(raw.id || "schema");
    this.objective = raw.objective ? String(raw.objective) : "";
    this.fields = raw.fields.map(f => normalizeField(f, {name: "observation"}));
    this.collections = (raw.collections || []).map(c => {
      if (!c || !c.id) fail("collection without an id");
      return {
        id: String(c.id),
        label: c.label || String(c.id),
        description: c.description || "",
        fields: (c.fields || []).map(f => normalizeField(f, {name: `collection ${c.id}`}))
      };
    });
    this.optionFields = (raw.optionFields || []).map(f => normalizeField(f, {name: "option", pointer: true}));
    if (!this.optionFields.length) fail("optionFields must be non-empty");
    const seen = new Set();
    for (const f of this.optionFields) {
      if (seen.has(f.id)) fail(`duplicate option field ${f.id}`);
      seen.add(f.id);
      if (f.kind === FieldKind.REF && !this.collections.some(c => c.id === f.collection)) {
        fail(`option field ${f.id} references unknown collection ${f.collection}`);
      }
    }
    this._index = new Map(this.optionFields.map(f => [f.id, f]));
  }

  field(id) {
    return this.fields.find(f => f.id === id) || null;
  }

  optionField(id) {
    return this._index.get(id) || null;
  }

  collection(id) {
    return this.collections.find(c => c.id === id) || null;
  }

  // How many distinct values a field can take, for sizing heads. Numbers are
  // handled by a continuous head, so they report null.
  cardinality(field) {
    if (field.kind === FieldKind.ENUM) return field.values.length;
    if (field.kind === FieldKind.BOOL) return 2;
    return null;
  }

  // A stable, order-independent description of what this schema can express.
  // Used to invalidate checkpoints when a schema changes shape.
  signature() {
    const part = f => {
      const bits = [f.id, f.kind];
      if (f.values) bits.push("[" + f.values.map(v => v.id).join(",") + "]");
      if (f.min != null) bits.push("min" + f.min);
      if (f.max != null) bits.push("max" + f.max);
      if (f.collection) bits.push("->" + f.collection);
      return bits.join(":");
    };
    return [
      "v" + this.version,
      "f(" + this.fields.map(part).join(";") + ")",
      "c(" + this.collections.map(c => c.id + ":" + c.fields.map(part).join(";")).join("|") + ")",
      "o(" + this.optionFields.map(part).join(";") + ")"
    ].join("#");
  }
}

export function defineSchema(raw) {
  return new Schema(raw);
}

// Validate a concrete choice against the option template.
// Structural only: it must never encode tactical policy.
export function validateChoice(schema, choice, {observation = null, partial = false} = {}) {
  const errors = [];
  const unchecked = [];
  const given = new Set();
  for (const field of schema.optionFields) {
    const has = choice && Object.prototype.hasOwnProperty.call(choice, field.id);
    if (!has) {
      if (field.required && !partial) errors.push(`missing required field ${field.id}`);
      continue;
    }
    given.add(field.id);
    const value = choice[field.id];
    switch (field.kind) {
      case FieldKind.BOOL:
        if (typeof value !== "boolean") errors.push(`${field.id} must be boolean`);
        break;
      case FieldKind.ENUM:
        if (!field.values.some(v => v.id === String(value))) errors.push(`${field.id} has unknown value ${JSON.stringify(value)}`);
        break;
      case FieldKind.NUMBER: {
        const n = Number(value);
        if (!Number.isFinite(n)) errors.push(`${field.id} must be a finite number`);
        else {
          if (field.min != null && n < field.min) errors.push(`${field.id} below min ${field.min}`);
          if (field.max != null && n > field.max) errors.push(`${field.id} above max ${field.max}`);
        }
        break;
      }
      case FieldKind.REF: {
        // A reference can only be checked against the observation it must point
        // into. Rather than silently reporting "valid", record it as unchecked.
        if (!observation) { unchecked.push(field.id); break; }
        const members = observation?._collections?.[field.collection] || [];
        const ok = members.some(m => String(m?.engine_record_id ?? m?.id) === String(value));
        if (!ok) errors.push(`${field.id} references an object not present in ${field.collection}`);
        break;
      }
      default:
        break;
    }
  }
  return {valid: errors.length === 0, errors, fields: [...given], unchecked};
}
