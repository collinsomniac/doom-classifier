import assert from "node:assert/strict";
import {SemanticResidualPolicy} from "../src/core/policy.js";

class ZeroSemantic{
  constructor(){this.name="zero"}compile(schema,actions){this.actions=actions}async score(){return this.actions.map(()=>0)}
}
const schema={fields:[{id:"x",label:"signal",description:"generic scalar signal",min:0,max:1}],collections:[]};
const actions=[{id:"a",label:"act",description:"perform the action"},{id:"b",label:"hold",description:"hold state"}];
const policy=new SemanticResidualPolicy({schema,actions,semantic:new ZeroSemantic(),residual:"neural-set",inferenceMode:"neural",seed:3,replayCapacity:4,replayBatch:2,nStep:1});
const transition=i=>({
  observation:{x:(i%2)*.8,_collections:{}},
  temporal:new Float32Array([0,0]),
  features:new Float32Array([1,0,0,0]),
  actionIndex:i%2,
  reward:i%2?.4:-.1,
  nextObservation:{x:((i+1)%2)*.8,_collections:{}},
  nextTemporal:new Float32Array([0,0]),
  nextFeatures:new Float32Array([1,0,0,0]),
  done:false
});
const start=policy.q.updates;
const results=[];
for(let i=0;i<6;i++)results.push(policy.learn(transition(i)));
assert.equal(policy.replay.length,4,"replay buffer must respect capacity");
assert.ok(policy.q.updates-start>6,"replay should produce extra neural updates beyond one update per transition");
assert.ok(results.some(r=>r.replayUpdates===2),"configured replay batch should be used once history exists");
policy.resetLearning();
assert.equal(policy.replay.length,0,"resetLearning must clear incompatible experience");
console.log(JSON.stringify({ok:true,totalUpdates:policy.q.updates-start,results:results.map(r=>({replayUpdates:r.replayUpdates,replaySize:r.replaySize}))}));
