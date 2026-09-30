// Bisect the gradient discrepancy by enabling pieces one at a time.
import {defineSchema} from "../src/core/schema.js";
import {tokenize} from "../src/core/tokens.js";
import {DecisionModel} from "../src/core/model.js";
import {lossAndGrads} from "../src/train.js";
import {supervisedLoss} from "../src/loss.js";

const schema = defineSchema({
  id: "x", objective: "pick",
  fields: [{id: "bearing", kind: "number", min: -1, max: 1}],
  collections: [{id: "ents", fields: [{id: "rank", kind: "number"}]}],
  optionFields: [
    {id: "turn", kind: "enum", values: ["none", "left", "right"]},
    {id: "engage", kind: "bool"},
    {id: "pick", kind: "ref", collection: "ents", required: false}
  ]
});
const stream = tokenize({
  schema,
  information: {bearing: 0.4, _collections: {ents: [{engine_record_id: 3, rank: 1}, {engine_record_id: 4, rank: 2}]}},
  choices: [{id: "a", text: "left"}, {id: "b", text: "right"}, {id: "c", text: "hold"}]
});

const cases = [
  {name: "choice only, 1 layer", layers: 1, fields: false, consequence: false, ref: false},
  {name: "choice only, 2 layers", layers: 2, fields: false, consequence: false, ref: false},
  {name: "choice + enum field", layers: 1, fields: true, consequence: false, ref: false},
  {name: "choice + ref field", layers: 1, fields: false, consequence: false, ref: true},
  {name: "choice + consequence", layers: 1, fields: false, consequence: true, ref: false}
];

for (const test of cases) {
  const model = new DecisionModel({schema, d: 16, layers: test.layers, heads: 4, embeddingDim: 8, consequenceHorizons: test.consequence ? 2 : 0, seed: 3});
  const targets = {choices: [1, 0, 0]};
  if (test.fields) targets.fields = {turn: "left", engage: true};
  if (test.ref) targets.fields = {pick: 4};
  if (test.consequence) targets.consequence = [0.5, -0.25];
  const {grads} = lossAndGrads(model, stream, targets);
  const w = model.params.inProj;
  const rows = [];
  for (const idx of [0, 5, 20]) {
    const keep = w[idx];
    w[idx] = keep + 1e-3; const plus = supervisedLoss(model, stream, targets);
    w[idx] = keep - 1e-3; const minus = supervisedLoss(model, stream, targets);
    w[idx] = keep;
    const numeric = (plus - minus) / 2e-3, analytic = grads.inProj[idx];
    rows.push(`${idx}:${numeric.toExponential(2)}/${analytic.toExponential(2)}`);
  }
  console.log(test.name.padEnd(24), rows.join("  "), "| numeric/analytic");
}
