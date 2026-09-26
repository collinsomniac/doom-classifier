import assert from "node:assert/strict";
import {NeuralSetResidualQ} from "../src/core/neural-set-residual.js";

const schema={fields:[{id:"signal",label:"threat signal",description:"current threat evidence",min:0,max:1}],collections:[]};
const actions=[{id:"act",label:"act",description:"respond to the threat"},{id:"hold",label:"hold",description:"hold position"}];
const obs={signal:.8,_collections:{}},next={signal:.4,_collections:{}};
const net=new NeuralSetResidualQ(schema,actions,{seed:211,targetSyncInterval:4});
assert.ok(net.valueHeadLayer,"consequence branch should have an independent action-conditioned adapter");
assert.ok(net.parameterCount()<8000,"low-rank critic decoupling should preserve the project sub-8k model ceiling");

for(let i=0;i<40;i++)net.distill(obs,[3,-2],{strength:.6});
net.syncTarget();
const before=net.scoreStatsObservation(obs),semanticBefore=[...before.semanticScores],valueBefore=[...before.valueScores],combinedBefore=[...before.scores];

for(let i=0;i<12;i++)net.updateTransition({observation:obs,actionIndex:1,reward:.8,nextObservation:next,done:true});
const afterReward=net.scoreStatsObservation(obs);
const semanticDrift=Math.max(...semanticBefore.map((v,i)=>Math.abs(v-afterReward.semanticScores[i])));
const valueMove=Math.max(...valueBefore.map((v,i)=>Math.abs(v-afterReward.valueScores[i])));
const combinedMove=Math.max(...combinedBefore.map((v,i)=>Math.abs(v-afterReward.scores[i])));
assert.ok(semanticDrift<1e-8,"reward TD must not overwrite semantic-prior logits");
assert.ok(valueMove>1e-5,"reward TD must update consequence-value branch");
assert.ok(combinedMove>1e-5,"learned value must influence final action scores");

const semanticPreSecond=[...afterReward.semanticScores];
for(let i=0;i<20;i++)net.distill(obs,[-2,3],{strength:.6});
const afterTeacher=net.scoreStatsObservation(obs);
const teacherMove=Math.max(...semanticPreSecond.map((v,i)=>Math.abs(v-afterTeacher.semanticScores[i])));
assert.ok(teacherMove>1e-5,"teacher distillation must update semantic-prior branch");
console.log(JSON.stringify({ok:true,params:net.parameterCount(),semanticDrift,valueMove,combinedMove,teacherMove}));
