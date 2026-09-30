// With identity blocks, the only path into the initial states is the pooled
// context. Split its consumers to find which one accumulates a wrong gradient.
import {defineSchema} from "../src/core/schema.js";
import {tokenize} from "../src/core/tokens.js";
import {DecisionModel} from "../src/core/model.js";
import {lossAndGrads} from "../src/train.js";
import {supervisedLoss} from "../src/loss.js";

const schema = defineSchema({
  id: "x", objective: "pick",
  fields: [{id: "bearing", kind: "number", min: -1, max: 1}],
  collections: [],
  optionFields: [{id: "turn", kind: "enum", values: ["none", "left", "right"]}]
});
const stream = tokenize({schema, information: {bearing: 0.4}, choices: [{id: "a", text: "left"}, {id: "b", text: "right"}, {id: "c", text: "hold"}]});
const targets = {choices: [1, 0, 0], fields: {turn: "left"}};

const zeroFieldHeads = model => {
  for (const spec of model.fieldSpecs) {
    model.params[`fieldClass_${spec.key}`].fill(0);
    model.params[`fieldProbe_${spec.key}`].fill(0);
    model.params[`fieldBias_${spec.key}`].fill(0);
  }
};
const identity = model => { model.blocks[0].wo.fill(0); model.blocks[0].w2.fill(0); };

const measure = (label, mutate) => {
  const model = new DecisionModel({schema, d: 16, layers: 1, heads: 4, embeddingDim: 8, seed: 3});
  identity(model);
  mutate(model);
  const {grads} = lossAndGrads(model, stream, targets);
  const w = model.params.inProj;
  const out = [];
  for (const idx of [0, 20]) {
    const g = () => supervisedLoss(model, stream, targets);
    const keep = w[idx];
    w[idx] = keep + 1e-3; const p1 = g();
    w[idx] = keep - 1e-3; const m1 = g();
    w[idx] = keep + 5e-4; const p2 = g();
    w[idx] = keep - 5e-4; const m2 = g();
    w[idx] = keep;
    const richardson = (4 * ((p2 - m2) / 1e-3) - (p1 - m1) / 2e-3) / 3;
    const analytic = grads.inProj[idx];
    const rel = Math.abs(richardson - analytic) / Math.max(1e-12, Math.abs(analytic));
    out.push(`${idx} an ${analytic.toExponential(3)} num ${richardson.toExponential(3)} rel ${(rel * 100).toFixed(1)}%`);
  }
  console.log(label.padEnd(34), out.join(" | "));
};

measure("identity, all consumers", () => {});
measure("identity, no scoreKeyW path", m => m.params.scoreKeyW.fill(0));
measure("identity, no field heads", m => zeroFieldHeads(m));
measure("identity, neither", m => { m.params.scoreKeyW.fill(0); zeroFieldHeads(m); });
measure("identity, context only via 1 field", m => {
  // keep a single class weight so the field contribution is unambiguous
  for (const spec of model.fieldSpecs) { m.params[`fieldClass_${spec.key}`].fill(0); m.params[`fieldProbe_${spec.key}`].fill(0); }
});
