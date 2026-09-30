// Core invariants: everything that must hold regardless of training.
//
// These run in CI. They cover the parts of the architecture that are verified
// today: schema compilation, schema-driven tokenization, the candidate-
// conditioned forward pass, typed argument construction, checkpoint round-trip,
// and the invariances the design commits to (option order, candidate-set
// extension, schema mismatch rejection).

import assert from "node:assert/strict";
import {defineSchema, validateChoice, FieldKind} from "../src/core/schema.js";
import {tokenize} from "../src/core/tokens.js";
import {stemEmbedder} from "../src/adapters/embedders.js";
import {DecisionModel} from "../src/core/model.js";
import {supervisedLoss} from "../src/loss.js";
import {lossAndGrads} from "../src/train.js";

const schema = defineSchema({
  id: "tool-routing",
  objective: "Route the request to the appropriate tool and fill its arguments.",
  fields: [
    {id: "intent", kind: "enum", values: ["weather", "reminder", "arithmetic"]},
    {id: "urgency", kind: "number", min: 0, max: 1},
    {id: "polite", kind: "bool"}
  ],
  collections: [{id: "contacts", label: "contacts", fields: [{id: "rank", kind: "number"}, {id: "name_length", kind: "number"}]}],
  optionFields: [
    {id: "tool", kind: "enum", values: ["get_weather", "set_reminder", "compute"]},
    {id: "notify", kind: "bool"},
    {id: "recipient", kind: "ref", collection: "contacts", required: false},
    {id: "delay_minutes", kind: "number", min: 0, max: 1440, required: false}
  ]
});

const information = {
  intent: "weather",
  urgency: 0.7,
  polite: true,
  _collections: {contacts: [{engine_record_id: 7, rank: 1, name_length: 4}, {engine_record_id: 9, rank: 2, name_length: 6}]}
};
const choices = [
  {id: "a", text: "Get the current weather for a city"},
  {id: "b", text: "Create a reminder to be delivered later"},
  {id: "c", text: "Evaluate an arithmetic expression"}
];
const embedder = stemEmbedder(24);
const stream = tokenize({schema, information, choices, embedder});
const model = new DecisionModel({schema, d: 32, layers: 2, heads: 4, embeddingDim: 24, consequenceHorizons: 2, seed: 7});

// ---- 1. schema ----
assert.equal(schema.optionFields.length, 4);
assert.equal(schema.collection("contacts").fields.length, 2);
assert.equal(schema.optionField("recipient").kind, FieldKind.REF);
assert.throws(() => defineSchema({fields: [], optionFields: [{id: "x", kind: "ref", collection: "missing"}]}), /unknown collection/);
assert.throws(() => defineSchema({fields: [{id: "a", kind: "nonsense"}], optionFields: [{id: "b", kind: "bool"}]}), /unknown kind/);

// structural validity only: a well-formed but tactically wrong choice is valid
assert.equal(validateChoice(schema, {tool: "get_weather", notify: true, recipient: 7}).valid, true);
assert.equal(validateChoice(schema, {tool: "teleport", notify: true}).valid, false, "unknown enum value must be rejected");
const withoutObservation = validateChoice(schema, {tool: "compute", notify: true, recipient: 42});
assert.equal(withoutObservation.valid, true, "without an observation a reference cannot be judged invalid");
assert.deepEqual(withoutObservation.unchecked, ["recipient"], "an unverifiable reference must be reported as unchecked, not silently accepted");
assert.equal(validateChoice(schema, {tool: "compute", notify: true, recipient: 42}, {observation: information}).valid, false);
assert.equal(validateChoice(schema, {tool: "compute", notify: true, recipient: 9}, {observation: information}).valid, true);
assert.equal(validateChoice(schema, {tool: "compute"}, {partial: true}).valid, true, "partial choices are allowed when asked");

// ---- 2. tokenization ----
const kinds = stream.tokens.map(t => t.type);
assert.ok(kinds.includes("objective") && kinds.includes("field") && kinds.includes("entity") && kinds.includes("choice"));
assert.equal(stream.tokens.filter(t => t.type === "choice").length, 3);
assert.equal(stream.tokens.filter(t => t.type === "entity").length, 2);
assert.equal(stream.tokens.filter(t => t.type === "choice_field").length, schema.optionFields.length);
assert.equal(stream.schemaSignature, schema.signature());
const bearing = stream.tokens.find(t => t.type === "field" && t.id === "urgency");
assert.ok(Math.abs(bearing.value - 0.7) < 1e-9, "a bounded number must be normalized, not embedded as text");

// a different option order must produce the same set of choice tokens, reordered
const reordered = tokenize({schema, information, choices: [...choices].reverse(), embedder});
assert.deepEqual(reordered.tokens.filter(t => t.type === "choice").map(t => t.id), ["c", "b", "a"]);

// ---- 3. forward pass ----
const decision = model.decide(stream);
assert.equal(decision.probs.length, 3);
assert.ok(Math.abs(decision.probs.reduce((a, b) => a + b, 0) - 1) < 1e-9, "scores must be a distribution");
assert.ok(decision.arguments.length === schema.optionFields.length, "every option field must receive a value or an explicit null");
const toolArgument = decision.arguments.find(a => a.id === "tool");
assert.ok(toolArgument.value === null || schema.optionField("tool").values.some(v => v.id === toolArgument.value), "enum argument must be a declared value or null");
const recipient = decision.arguments.find(a => a.id === "recipient");
assert.ok(recipient.value === null || [7, 9].includes(Number(recipient.value)), "reference argument must point at a supplied object");
assert.equal(recipient.provenance?.collection ?? "contacts", "contacts");
assert.ok(recipient.confidence >= 0 && recipient.confidence <= 1);

// candidate-count independence: adding an option adds one score, one pass
const extended = model.decide(tokenize({schema, information, choices: [...choices, {id: "d", text: "Search the web"}], embedder}));
assert.equal(extended.probs.length, 4);
assert.ok(Math.abs(extended.probs.reduce((a, b) => a + b, 0) - 1) < 1e-9);

// the model must not rely on candidate position: a lone candidate scores without error
const single = model.decide(tokenize({schema, information, choices: [choices[1]], embedder}));
assert.equal(single.choice.id, "b");

// ---- 4. training plumbing consistency ----
const targets = {choices: [0.6, 0.3, 0.1], fields: {tool: "get_weather", notify: true, recipient: 9, delay_minutes: 15}, consequence: [0.4, -0.2]};
const {loss} = lossAndGrads(model, stream, targets, {weight: 0.9});
assert.ok(Math.abs(supervisedLoss(model, stream, targets, {weight: 0.9}) - loss) < 1e-6, "the loss-only evaluator must equal the trainer's loss");
assert.throws(() => lossAndGrads(model, stream, {choices: [1, 0]}, {}), /length must match/);

// ---- 5. checkpoint ----
const checkpoint = JSON.parse(JSON.stringify(model.exportWeights()));
const restored = new DecisionModel({schema, d: 32, layers: 2, heads: 4, embeddingDim: 24, consequenceHorizons: 2, seed: 99});
assert.equal(restored.importWeights(checkpoint), true);
const restoredDecision = restored.decide(stream);
restoredDecision.probs.forEach((p, i) => assert.ok(Math.abs(p - decision.probs[i]) < 1e-6, "checkpoint must reproduce the distribution"));
assert.equal(restored.importWeights({...checkpoint, schemaSignature: "different"}), false, "a mismatched schema signature must be refused, not misread");

// a model of the wrong shape must refuse the checkpoint
const wrongShape = new DecisionModel({schema, d: 16, layers: 2, heads: 4, embeddingDim: 24, seed: 1});
assert.equal(wrongShape.importWeights(checkpoint), false, "shape mismatch must be rejected");

console.log("core invariants ok", {
  tokens: stream.tokens.length,
  parameters: model.parameterCount(),
  argumentsConstructed: decision.arguments.length
});
