// Isolate the encoder backward: null one component at a time and compare the
// inProj gradient against finite differences.
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
const targets = {choices: [1, 0, 0]};

const run = (label, mutate) => {
  const model = new DecisionModel({schema, d: 16, layers: 1, heads: 4, embeddingDim: 8, seed: 3});
  if (mutate) mutate(model);
  const {grads} = lossAndGrads(model, stream, targets);
  const w = model.params.inProj;
  const rows = [];
  for (const idx of [0, 20]) {
    const keep = w[idx];
    w[idx] = keep + 1e-3; const plus = supervisedLoss(model, stream, targets);
    w[idx] = keep - 1e-3; const minus = supervisedLoss(model, stream, targets);
    w[idx] = keep;
    const numeric = (plus - minus) / 2e-3, analytic = grads.inProj[idx];
    const rel = Math.abs(numeric - analytic) / Math.max(1e-12, Math.abs(analytic));
    rows.push(`${idx}: num ${numeric.toExponential(2)} an ${analytic.toExponential(2)} rel ${(rel * 100).toFixed(1)}%`);
  }
  console.log(label.padEnd(22), rows.join(" | "));
};

run("full block");
run("attention output = 0", m => m.blocks[0].wo.fill(0));
run("feed-forward out = 0", m => m.blocks[0].w2.fill(0));
run("both nulled", m => { m.blocks[0].wo.fill(0); m.blocks[0].w2.fill(0); });
