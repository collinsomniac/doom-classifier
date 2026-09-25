import assert from "node:assert/strict";
import {NeuralSetResidualQ} from "../src/core/neural-set-residual.js";
import {SemanticResidualPolicy} from "../src/core/policy.js";
import {HashSemanticAdapter} from "../src/model-adapters/hash-semantic.js";

const schema={objective:"respond appropriately",fields:[{id:"signal",label:"signal",description:"current signal",min:0,max:1}],collections:[]};
const actions=[
  {id:"left",label:"left",description:"move left",params:{}},
  {id:"right",label:"right",description:"move right",params:{}}
];
const net=new NeuralSetResidualQ(schema,actions,{seed:313,targetSyncInterval:3});
const obs={signal:.7,_collections:{}};

for(let i=0;i<30;i++)net.distill(obs,[2,-1],{strength:.6});
const none=net.scoreStatsObservation(obs,{history:{previousActionIndex:null,streak:0}});
const repeated=net.scoreStatsObservation(obs,{history:{previousActionIndex:0,streak:6}});
const switched=net.scoreStatsObservation(obs,{history:{previousActionIndex:1,streak:1}});

assert.deepEqual(none.semanticScores,repeated.semanticScores,"previous action must not affect semantic-prior logits");
assert.deepEqual(repeated.semanticScores,switched.semanticScores,"semantic prior must remain history-free");
const valueHistoryDiff=Math.max(...repeated.valueScores.map((v,i)=>Math.abs(v-switched.valueScores[i])));
assert.ok(valueHistoryDiff>1e-9,"value branch should receive previous-action context");
assert.ok(net.parameterCount()<8000,"history-aware fast policy must remain under 8k parameters");

const beforeSemantic=[...repeated.semanticScores];
for(let i=0;i<8;i++)net.updateTransition({
  observation:obs,history:{previousActionIndex:0,streak:5},actionIndex:0,reward:.5,
  nextObservation:obs,nextHistory:{previousActionIndex:0,streak:6},done:false
});
const after=net.scoreStatsObservation(obs,{history:{previousActionIndex:0,streak:6}});
const semanticDrift=Math.max(...beforeSemantic.map((v,i)=>Math.abs(v-after.semanticScores[i])));
assert.equal(semanticDrift,0,"history-conditioned reward learning must not overwrite semantic prior");

const policy=new SemanticResidualPolicy({schema,actions,semantic:new HashSemanticAdapter(),residual:net,seed:313});
assert.equal(policy.currentActionHistory().previousActionIndex,null);
assert.deepEqual(policy.nextActionHistory(0),{previousActionIndex:0,streak:1});
policy.commitAction(0);policy.commitAction(0);policy.commitAction(0);
assert.deepEqual(policy.currentActionHistory(),{previousActionIndex:0,streak:3});
assert.deepEqual(policy.nextActionHistory(1),{previousActionIndex:1,streak:1});
policy.resetEpisode();
assert.deepEqual(policy.currentActionHistory(),{previousActionIndex:null,streak:0});

console.log(JSON.stringify({ok:true,params:net.parameterCount(),valueHistoryDiff,semanticDrift}));
