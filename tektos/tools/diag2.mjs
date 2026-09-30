// Does the inProj gradient mismatch shrink with epsilon (noise) or persist (bug)?
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
const model = new DecisionModel({schema, d: 16, layers: 2, heads: 4, embeddingDim: 8, seed: 3});
const {grads, loss} = lossAndGrads(model, stream, targets);
const w = model.params.inProj;
for (const idx of [0, 5, 20, 100]) {
  const line = [`idx ${idx} analytic ${grads.inProj[idx].toExponential(3)}`];
  for (const eps of [1e-2, 1e-3, 1e-4]) {
    const keep = w[idx];
    w[idx] = keep + eps; const plus = supervisedLoss(model, stream, targets);
    w[idx] = keep - eps; const minus = supervisedLoss(model, stream, targets);
    w[idx] = keep;
    line.push(`eps=${eps} numeric ${((plus - minus) / (2 * eps)).toExponential(3)}`);
  }
  console.log(line.join(" | "));
}
console.log("loss", loss.toFixed(5));
