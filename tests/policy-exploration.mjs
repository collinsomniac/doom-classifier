import assert from "node:assert/strict";
import {SemanticResidualPolicy} from "../src/core/policy.js";

const schema={fields:[{id:"x",label:"x",min:0,max:1}],collections:[]};
const actions=[{id:"a",label:"a"},{id:"b",label:"b"},{id:"c",label:"c"}];
const semantic={name:"stub",backend:"test",compile(){},async score(){return[0,0,0]}};
const logits=[Math.log(.6),Math.log(.3),Math.log(.1)];
const q={
  scoreStatsObservation(){return{scores:[...logits],memberScores:null,semanticScores:[...logits],valueScores:[0,0,0]}},
  reset(){},parameterCount(){return 0}
};
const p=new SemanticResidualPolicy({schema,actions,semantic,residual:q,temperature:1,epsilon:0,seed:5,inferenceMode:"neural"});

p.rng=()=>.65;
let d=await p.decide({x:.5,_collections:{}},{useResidual:true,memory:false,explore:true});
assert.equal(d.action.id,"b","policy-proportional exploration must sample from the distribution, not force argmax");
assert.equal(d.explorationStrategy,"policy-proportional");

p.rng=()=>.99;
d=await p.decide({x:.5,_collections:{}},{useResidual:true,memory:false,explore:true});
assert.equal(d.action.id,"c");

d=await p.decide({x:.5,_collections:{}},{useResidual:true,memory:false,explore:false});
assert.equal(d.action.id,"a","frozen/non-exploratory decoding must remain deterministic argmax");
assert.equal(d.explorationStrategy,"greedy");

p.epsilon=1;p.rng=()=>.5;
d=await p.decide({x:.5,_collections:{}},{useResidual:true,memory:false,explore:true});
assert.equal(d.action.id,"b","uniform-mix=1 should sample a uniform categorical policy");
console.log(JSON.stringify({ok:true,last:d.action.id,strategy:d.explorationStrategy}));
