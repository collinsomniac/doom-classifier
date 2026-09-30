// Richardson-extrapolated finite differences to separate truncation error from
// a genuine gradient error, on the exact configuration the gradient check uses.
import {defineSchema} from "../src/core/schema.js";
import {tokenize} from "../src/core/tokens.js";
import {DecisionModel} from "../src/core/model.js";
import {lossAndGrads} from "../src/train.js";
import {supervisedLoss} from "../src/loss.js";

const schema = defineSchema({
  id: "gradcheck",
  objective: "Choose the tool that satisfies the request.",
  fields: [
    {id: "intent", kind: "enum", values: ["weather", "reminder", "math"]},
    {id: "urgency", kind: "number", min: 0, max: 1}
  ],
  collections: [{id: "contacts", fields: [{id: "rank", kind: "number"}, {id: "name_length", kind: "number"}]}],
  optionFields: [
    {id: "tool", kind: "enum", values: ["get_weather", "set_reminder", "compute"]},
    {id: "notify", kind: "bool"},
    {id: "contact", kind: "ref", collection: "contacts", required: false}
  ]
});
const stream = tokenize({
  schema,
  information: {intent: "weather", urgency: 0.7, _collections: {contacts: [{engine_record_id: 7, rank: 1, name_length: 4}, {engine_record_id: 9, rank: 2, name_length: 6}]}},
  choices: [{id: "a", text: "Get the current weather for a city"}, {id: "b", text: "Create a reminder for later"}, {id: "c", text: "Evaluate an arithmetic expression"}]
});
const targets = {choices: [0.7, 0.2, 0.1], fields: {tool: "get_weather", notify: true, contact: 9}, consequence: [0.35, -0.1]};
const model = new DecisionModel({schema, d: 24, layers: 2, heads: 3, embeddingDim: 12, consequenceHorizons: 2, seed: 7});
const {grads} = lossAndGrads(model, stream, targets, {weight: 0.85});

const derivative = (tensor, idx, grad, eps) => {
  const keep = tensor[idx];
  tensor[idx] = keep + eps; const plus = grad();
  tensor[idx] = keep - eps; const minus = grad();
  tensor[idx] = keep;
  return (plus - minus) / (2 * eps);
};

const check = (name, tensor, grad, indices) => {
  for (const idx of indices) {
    const g = () => supervisedLoss(model, stream, targets, {weight: 0.85});
    const d1 = derivative(tensor, idx, g, 1e-3);
    const d2 = derivative(tensor, idx, g, 5e-4);
    const richardson = (4 * d2 - d1) / 3;
    const analytic = grad[idx];
    const rel = Math.abs(richardson - analytic) / Math.max(1e-12, Math.abs(analytic));
    console.log(`${name}[${idx}] analytic ${analytic.toExponential(4)} richardson ${richardson.toExponential(4)} rel ${(rel * 100).toFixed(2)}%`);
  }
};

check("inProj", model.params.inProj, grads.inProj, [0, 20]);
check("scoreQueryW", model.params.scoreQueryW, grads.scoreQueryW, [0, 30]);
check("scoreKeyW", model.params.scoreKeyW, grads.scoreKeyW, [0, 30]);
check("typeEmbedding", model.params.typeEmbedding, grads.typeEmbedding, [3]);
