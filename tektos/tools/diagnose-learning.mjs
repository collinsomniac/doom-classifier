// Diagnostic: can the model overfit a handful of examples at all?
import {defineSchema} from "../src/core/schema.js";
import {tokenize, hashEmbedder} from "../src/core/tokens.js";
import {DecisionModel} from "../src/core/model.js";
import {Trainer} from "../src/train.js";
import {supervisedLoss} from "../src/loss.js";

const schema = defineSchema({
  id: "diag",
  objective: "Choose the appropriate typed option.",
  fields: [
    {id: "target_visible", kind: "bool", label: "target visible"},
    {id: "bearing", kind: "number", min: -1, max: 1, label: "signed bearing"}
  ],
  collections: [],
  optionFields: [
    {id: "turn", kind: "enum", values: ["none", "left", "right"]},
    {id: "engage", kind: "bool"}
  ]
});
const OPTIONS = [
  {id: "hold", text: "hold position and keep heading", params: {turn: "none", engage: false}},
  {id: "left", text: "rotate the view toward the left", params: {turn: "left", engage: false}},
  {id: "right", text: "rotate the view toward the right", params: {turn: "right", engage: false}}
];
const embedder = hashEmbedder(24);
const data = [
  {information: {target_visible: true, bearing: 0.4}, index: 1},
  {information: {target_visible: true, bearing: -0.4}, index: 2},
  {information: {target_visible: false, bearing: 0}, index: 0}
].map(d => ({
  stream: tokenize({schema, information: d.information, choices: OPTIONS, embedder}),
  targets: {choices: OPTIONS.map((_, i) => (i === d.index ? 1 : 0)), fields: {turn: OPTIONS[d.index].params.turn, engage: OPTIONS[d.index].params.engage}},
  index: d.index
}));

for (const lr of [0.01, 0.05]) {
  const model = new DecisionModel({schema, d: 32, layers: 2, heads: 4, embeddingDim: 24, seed: 5});
  const trainer = new Trainer(model, {lr, l2: 1e-5, clip: 2});
  const losses = [];
  for (let i = 0; i < 120; i++) {
    let total = 0;
    for (const example of data) total += trainer.step(example.stream, example.targets);
    if (i % 20 === 0 || i === 119) losses.push(+total.toFixed(3));
  }
  const predictions = data.map(e => {
    const d = model.decide(e.stream);
    const argmax = d.probs.indexOf(Math.max(...d.probs));
    return `${OPTIONS[e.index].id}->${OPTIONS[argmax].id}${argmax === e.index ? "" : "(WRONG)"}`;
  });
  let norm = 0;
  for (const v of Object.values(model.params)) if (v) for (const x of v) norm += x * x;
  console.log(`lr=${lr} losses`, losses.join(" "), "| preds", predictions.join(" "), "| |W|", Math.sqrt(norm).toFixed(2));
}
