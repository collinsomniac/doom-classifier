import assert from "node:assert/strict";
import {NeuralSetResidualQ} from "../src/core/neural-set-residual.js";

const makeVector=(index,size=384)=>Array.from({length:size},(_,i)=>i===index?1:0);
const schema={
  objective:"choose an appropriate operation",
  objectiveSemanticVector:makeVector(3),
  fields:[{id:"x",label:"measurement",description:"current measurement",min:0,max:1,semanticVector:makeVector(7)}],
  collections:[]
};
const actions=[
  {id:"a",label:"operate",description:"perform the operation",semanticVector:makeVector(11)},
  {id:"b",label:"operate",description:"perform the operation",semanticVector:makeVector(19)}
];
const obs={x:.6,_collections:{}};
const net=new NeuralSetResidualQ(schema,actions,{seed:41});
assert.notDeepEqual(Array.from(net.actionEmbeddings[0]),Array.from(net.actionEmbeddings[1]),"semantic action vectors must affect compiled candidate embeddings");
const scores=net.scoresObservation(obs);
assert.ok(Math.abs(scores[0]-scores[1])>1e-7,"semantically distinct candidates with identical surface text must remain independently scoreable");

const changedSchema={...schema,objectiveSemanticVector:makeVector(29),fields:[{...schema.fields[0],semanticVector:makeVector(31)}]};
const netChanged=new NeuralSetResidualQ(changedSchema,actions,{seed:41});
const changed=netChanged.scoresObservation(obs);
const diff=Math.max(...scores.map((v,i)=>Math.abs(v-changed[i])));
assert.ok(diff>1e-6,"compiled schema semantics must influence the neural state representation");
console.log(JSON.stringify({ok:true,params:net.parameterCount(),scores,changed,diff}));
