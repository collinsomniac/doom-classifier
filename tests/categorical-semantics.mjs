import assert from "node:assert/strict";
import {NeuralSetResidualQ} from "../src/core/neural-set-residual.js";
import {TransformersNLIAdapter} from "../src/model-adapters/transformers-nli.js";

const schema={
  objective:"choose an appropriate operation",
  fields:[{id:"tool",label:"equipped tool",description:"tool currently available",enum:{0:"none",1:"pistol",2:"shotgun"}}],
  collections:[]
};
const actions=[{id:"fire",label:"fire",description:"activate the equipped weapon"},{id:"wait",label:"wait",description:"do nothing"}];
const net=new NeuralSetResidualQ(schema,actions,{seed:55});
const pistol=net.scoresObservation({tool:1,_collections:{}});
const shotgun=net.scoresObservation({tool:2,_collections:{}});
assert.ok(Math.max(...pistol.map((v,i)=>Math.abs(v-shotgun[i])))>1e-7,"categorical labels must change neural state even when represented by numeric engine codes");

const teacher=new TransformersNLIAdapter();
teacher.compile(schema,actions);
const text=teacher.stateText({tool:1,_collections:{}}),payload=JSON.parse(text);
assert.deepEqual(payload.observation.scalars.tool,{value:1,label:"pistol"});
assert.equal(payload.objective,"choose an appropriate operation");
assert.equal(payload.projection.bounded,true);
console.log(JSON.stringify({ok:true,pistol,shotgun,text}));
