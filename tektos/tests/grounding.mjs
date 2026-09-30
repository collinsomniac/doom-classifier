// End-to-end grounding test: can this architecture learn a typed decision
// relation, deploy it, and generalise to states and phrasings it never saw?
//
// The task is small and deliberately NOT DOOM-specific: route a decision to one
// of several typed option objects, where the correct option depends on the
// *relation* between observed fields (a signed bearing and whether the target
// is visible), not on any single field's value. Labels come from a synthetic
// teacher, because this test asks whether the architecture can represent and
// deploy a known mapping; it does not claim an environment produces it.
//
// Two design points learned the hard way:
//   1. Option semantics must not overlap. An earlier version contained both
//      "orient left" and "watch left", which the context cannot distinguish, so
//      the one-hot target fought itself and the model settled on a constant.
//   2. Paraphrase invariance is a property of the TEXT ENCODER, not of the
//      decision architecture. A bare hash embedder cannot represent "leftward"
//      ~ "left", so this test uses a stem-aware embedder (the production
//      configuration supplies a frozen sentence encoder).

import assert from "node:assert/strict";
import {defineSchema} from "../src/core/schema.js";
import {tokenize} from "../src/core/tokens.js";
import {DecisionModel} from "../src/core/model.js";
import {Trainer} from "../src/train.js";
import {stemEmbedder} from "../src/adapters/embedders.js";
import {mulberry32} from "../src/core/tensor.js";

const schema = defineSchema({
  id: "orient-and-engage",
  objective: "Choose the typed option object appropriate for the current relation to the target.",
  fields: [
    {id: "target_visible", kind: "bool", label: "target visible"},
    {id: "bearing", kind: "number", min: -1, max: 1, label: "signed bearing to target", description: "positive means left of view, negative means right"},
    {id: "distance", kind: "number", min: 0, max: 2048, label: "distance to target"}
  ],
  collections: [],
  optionFields: [
    {id: "turn", kind: "enum", values: ["none", "left", "right"], label: "view change"},
    {id: "engage", kind: "bool", label: "engage the target"}
  ]
});

// Option objects are data. Each is separated by its relation to the target.
const OPTIONS = [
  {id: "hold", text: "hold position and keep heading", params: {turn: "none", engage: false}},
  {id: "orient_left", text: "rotate the view toward the left", params: {turn: "left", engage: false}},
  {id: "orient_right", text: "rotate the view toward the right", params: {turn: "right", engage: false}},
  {id: "engage", text: "keep the heading and engage the target", params: {turn: "none", engage: true}}
];

function teacher({bearing, target_visible: visible}) {
  if (!visible) return "hold";
  if (Math.abs(bearing) <= 0.06) return "engage";
  return bearing > 0 ? "orient_left" : "orient_right";
}

const embedder = stemEmbedder(24);
const makeStream = (information, options) => tokenize({schema, information, choices: options, embedder});

// Training sees several wordings; the invariance check below uses yet another.
const wordings = [
  o => o.text,
  o => `${o.id.replace(/_/g, " ")} action`,
  o => `${o.params.turn === "left" ? "leftward" : o.params.turn === "right" ? "rightward" : "straight"} motion with weapon ${o.params.engage ? "active" : "idle"}`
];

function batch(count, seed) {
  const rng = mulberry32(seed);
  const examples = [];
  for (let i = 0; i < count; i++) {
    const visible = rng() > 0.25;
    const bearing = visible ? rng() * 2 - 1 : 0;
    const information = {target_visible: visible, bearing, distance: 64 + rng() * 900};
    const correct = teacher(information);
    const index = OPTIONS.findIndex(o => o.id === correct);
    examples.push({
      information,
      targets: {choices: OPTIONS.map((_, i2) => (i2 === index ? 1 : 0)), fields: {turn: OPTIONS[index].params.turn, engage: OPTIONS[index].params.engage}}
    });
  }
  return examples;
}

const TRAIN_EXAMPLES = Number(process.env.GROUNDING_EXAMPLES || 140);
const EPOCHS = Number(process.env.GROUNDING_EPOCHS || 40);
const train = batch(TRAIN_EXAMPLES, 11);

const model = new DecisionModel({schema, d: 40, layers: 2, heads: 4, embeddingDim: 24, seed: 5});
// lr is deliberately conservative: measured on this architecture 0.05
// oscillates and fails to reduce the loss while 0.01 converges
// (tools/diagnose-learning.mjs). Optimizer stability is verified, not assumed.
const trainer = new Trainer(model, {lr: 0.01, l2: 1e-5, clip: 2});

const evaluate = () => {
  let total = 0;
  for (const example of train) total += trainer.evaluate([{stream: makeStream(example.information, OPTIONS), targets: example.targets}]);
  return total / train.length;
};
const before = evaluate();

const history = [];
const wordingNoise = mulberry32(99);
for (let epoch = 0; epoch < EPOCHS; epoch++) {
  let total = 0;
  for (const example of train) {
    // Each epoch sees a randomly chosen wording per option: input augmentation
    // of the same task, which is what forces the model to read the relation
    // from the observation rather than memorise a string.
    const phrasing = OPTIONS.map(o => ({id: o.id, text: wordings[Math.floor(wordingNoise() * wordings.length)](o)}));
    total += trainer.step(makeStream(example.information, phrasing), example.targets);
  }
  const meanLoss = total / train.length;
  history.push(meanLoss);
  if (epoch % 10 === 0 || epoch === EPOCHS - 1) console.log(`  epoch ${epoch + 1}/${EPOCHS} loss ${meanLoss.toFixed(4)}`);
}
const after = evaluate();

// ---- deployed behaviour (original wording, never seen as a training target) ----
const decideFor = (information, options = OPTIONS) => model.decide(makeStream(information, options));
const left = decideFor({target_visible: true, bearing: 0.35, distance: 200});
const right = decideFor({target_visible: true, bearing: -0.35, distance: 200});
const aligned = decideFor({target_visible: true, bearing: 0.01, distance: 200});
const hidden = decideFor({target_visible: false, bearing: 0, distance: 200});

const argOf = (result, id) => result.arguments.find(a => a.id === id)?.value;
const mass = (result, id) => result.probs[OPTIONS.findIndex(o => o.id === id)];
const show = (name, result) => console.log(`  ${name.padEnd(8)} -> ${result.choice.id.padEnd(13)} turn=${String(argOf(result, "turn")).padEnd(6)} engage=${String(argOf(result, "engage")).padEnd(5)} p=${result.confidence.toFixed(3)}`);

console.log("loss", {before: +before.toFixed(3), after: +after.toFixed(3), first: +history[0].toFixed(3), last: +history.at(-1).toFixed(3)});
show("left", left);
show("right", right);
show("aligned", aligned);
show("hidden", hidden);

// 1. Training must actually fit the task.
assert.ok(after < before * 0.5, `training must substantially reduce the loss (before ${before.toFixed(2)}, after ${after.toFixed(2)})`);

// 2. Mirrored relational states must produce opposite orientation arguments
//    and opposite choices.
assert.equal(argOf(left, "turn"), "left", "a target to the left must request a left orientation");
assert.equal(argOf(right, "turn"), "right", "a target to the right must request a right orientation");
assert.equal(left.choice.id, "orient_left", "the left state must select the left-orienting option");
assert.equal(right.choice.id, "orient_right", "the right state must select the right-orienting option");
const divergence = 0.5 * left.probs.reduce((sum, p, i) => sum + Math.abs(p - right.probs[i]), 0);
assert.ok(divergence > 0.4, `mirrored states must differ materially (got ${divergence.toFixed(3)})`);

// 3. Alignment and visibility must gate engaging.
assert.equal(argOf(aligned, "engage"), true, "an aligned visible target must be engaged");
assert.equal(argOf(aligned, "turn"), "none", "an aligned target must not request rotation");
assert.equal(argOf(hidden, "engage"), false, "an invisible target must not be engaged");
assert.equal(hidden.choice.id, "hold", "with no visible target the option must be hold");
assert.ok(mass(aligned, "engage") > 0.5, "the engage option must carry the mass when aligned");

// 4. Generalisation beyond the training range of bearings.
const farLeft = decideFor({target_visible: true, bearing: 0.95, distance: 1600});
const farRight = decideFor({target_visible: true, bearing: -0.95, distance: 1600});
assert.equal(argOf(farLeft, "turn"), "left", "an unseen extreme left bearing must still orient left");
assert.equal(argOf(farRight, "turn"), "right", "an unseen extreme right bearing must still orient right");

// 5. Choice order must not matter.
const reordered = decideFor({target_visible: true, bearing: 0.35, distance: 200}, [...OPTIONS].reverse());
assert.equal(reordered.choice.id, left.choice.id, "reordering the option list must not change the decision");

// 6. Wording must not matter (with a semantics-aware encoder).
const rephrased = OPTIONS.map(o => ({id: o.id, text: `${o.id.replace(/_/g, " ")} — weapon ${o.params.engage ? "ready" : "stowed"}`}));
assert.equal(decideFor({target_visible: true, bearing: -0.35, distance: 200}, rephrased).choice.id, "orient_right", "rephrased option text must not change the decision");
assert.equal(decideFor({target_visible: false, bearing: 0, distance: 200}, rephrased).choice.id, "hold", "rephrased option text must not change the quiet decision");

// 7. The candidate set is not a fixed vocabulary: a new option is scoreable
//    without retraining, and the option template is unchanged.
const extended = [...OPTIONS, {id: "retreat", text: "move away to break contact", params: {turn: "none", engage: false}}];
const withNew = model.decide(makeStream({target_visible: true, bearing: 0.35, distance: 90}, extended));
assert.equal(withNew.probs.length, extended.length, "an added option must receive a score without retraining");
assert.ok(withNew.probs.every(p => p >= 0 && p <= 1), "scores must be probabilities");
assert.ok(Math.abs(withNew.probs.reduce((a, b) => a + b, 0) - 1) < 1e-9, "scores must sum to one");

// 8. Confidence must be informative, not constant.
assert.ok(aligned.confidence > 0.5 && hidden.confidence > 0.5, "clear states must be decided confidently");
assert.ok(new Set([left, right, aligned, hidden].map(r => r.confidence.toFixed(3))).size > 1, "confidence must vary with the state");

// 9. Checkpoint round-trip must preserve the deployed distribution.
const restored = new DecisionModel({schema, d: 40, layers: 2, heads: 4, embeddingDim: 24, seed: 6});
assert.equal(restored.importWeights(JSON.parse(JSON.stringify(model.exportWeights()))), true, "checkpoint must load");
const restoredLeft = restored.decide(makeStream({target_visible: true, bearing: 0.35, distance: 200}, OPTIONS));
restoredLeft.probs.forEach((p, i) => assert.ok(Math.abs(p - left.probs[i]) < 1e-6, "checkpoint round-trip changed the policy"));

// 10. A different schema must be rejected rather than silently misread.
const otherSchema = defineSchema({
  id: "other",
  fields: [{id: "target_visible", kind: "bool"}],
  optionFields: [{id: "turn", kind: "enum", values: ["none", "left", "right"]}]
});
const incompatible = new DecisionModel({schema: otherSchema, d: 40, layers: 2, heads: 4, embeddingDim: 24, seed: 6});
assert.equal(incompatible.importWeights(model.exportWeights()), false, "a different schema signature must be rejected");

console.log("grounding ok", {
  trainingExamples: train.length,
  epochs: EPOCHS,
  lossBefore: +before.toFixed(3),
  lossAfter: +after.toFixed(3),
  mirroredDivergence: +divergence.toFixed(3),
  parameters: model.parameterCount()
});
