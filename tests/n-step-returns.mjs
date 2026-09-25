import assert from "node:assert/strict";
import {SemanticResidualPolicy} from "../src/core/policy.js";

const schema={fields:[{id:"x",label:"x",min:0,max:1}],collections:[]};
const actions=[{id:"a",label:"a"},{id:"b",label:"b"}];
const semantic={name:"stub",backend:"test",compile(){},async score(){return[0,0]}};
const q={
  gamma:.9,updates:0,seen:[],
  updateTransition(t){this.updates++;this.seen.push(t);return{td:0,target:t.reward,q:0}},
  reset(){this.updates=0;this.seen=[]},parameterCount(){return 0}
};
const p=new SemanticResidualPolicy({schema,actions,semantic,residual:q,nStep:3,replayCapacity:0,replayBatch:0,seed:41,inferenceMode:"neural"});
const tr=(id,reward,done=false)=>({
  observation:{x:id/10,_collections:{}},temporal:null,features:new Float32Array([1]),
  actionIndex:id%2,reward,nextObservation:{x:(id+1)/10,_collections:{}},nextTemporal:null,nextFeatures:new Float32Array([1]),done
});

let r=p.learn(tr(1,1));assert.equal(r.pending,true);assert.equal(q.updates,0);
r=p.learn(tr(2,2));assert.equal(r.pending,true);assert.equal(q.updates,0);
r=p.learn(tr(3,3));
assert.equal(q.updates,1);
assert.equal(q.seen[0].nStepHorizon,3);
assert.ok(Math.abs(q.seen[0].reward-(1+.9*2+.81*3))<1e-9);
assert.ok(Math.abs(q.seen[0].bootstrapDiscount-.729)<1e-12);
assert.equal(q.seen[0].done,false);

r=p.learn(tr(4,4,true));
assert.equal(q.updates,4,"terminal transition must flush all shorter pending prefixes");
assert.equal(p.nStepBuffer.length,0);
assert.ok(Math.abs(q.seen[1].reward-(2+.9*3+.81*4))<1e-9);
assert.equal(q.seen[1].done,true);assert.equal(q.seen[1].bootstrapDiscount,0);
assert.ok(Math.abs(q.seen[2].reward-(3+.9*4))<1e-9);
assert.equal(q.seen[2].nStepHorizon,2);
assert.equal(q.seen[3].reward,4);assert.equal(q.seen[3].nStepHorizon,1);

const verified={updates:q.updates,rewards:q.seen.map(x=>x.reward),horizons:q.seen.map(x=>x.nStepHorizon),discounts:q.seen.map(x=>x.bootstrapDiscount)};
p.resetLearning();assert.equal(p.nStepBuffer.length,0);
console.log(JSON.stringify({ok:true,...verified}));
