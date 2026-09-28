import assert from "node:assert/strict";
import {SemanticResidualPolicy} from "../src/core/policy.js";
import {HashSemanticAdapter} from "../src/core/semantic.js";
import {softmax} from "../src/core/math.js";

const schema={
  objective:"choose a useful typed control from structured state",
  fields:[{id:"pressure",label:"pressure",description:"current pressure",min:0,max:1}],
  collections:[{id:"objects",label:"objects",description:"variable structured objects",fields:[
    {id:"distance",label:"distance",description:"distance from controller",min:0,max:100},
    {id:"salience",label:"salience",description:"measured object salience",min:0,max:1}
  ]}],
  actionFields:[{id:"engage",label:"engage control",description:"whether engagement is active",min:0,max:1}]
};
const actions=[
  {id:"hold",label:"hold",description:"hold the current state",params:{engage:0}},
  {id:"engage",label:"engage",description:"engage the current opportunity",params:{engage:1}}
];
const obs={pressure:.8,_collections:{objects:[{distance:12,salience:1},{distance:80,salience:.1}]}};
const policy=new SemanticResidualPolicy({schema,actions,semantic:new HashSemanticAdapter(),residual:"neural-set",seed:515,inferenceMode:"neural"});
for(let i=0;i<30;i++)policy.q.distill(obs,[2.5,-1.5],{strength:.5});

const before=policy.q.scoreStatsObservation(obs),semanticBefore=[...before.semanticScores],valueBefore=softmax(before.valueScores,1);
const probeState=policy.q.encodeState(obs,{cache:false}),probeForward=policy.q.actionForward(probeState,1);
policy.q.zeroGrad();policy.q.backwardValue(probeForward,1,{bootstrap:false});
const queryGradient=Math.max(...policy.q.valueQueryLayer.gw.map(Math.abs));
assert.ok(queryGradient>1e-10,"critic must expose a trainable private action-conditioned attention path");
policy.q.zeroGrad();

const result=policy.fitCounterfactualValueDistributions([{observation:obs,target:[.02,.98]}],{steps:48,strength:.35});
const after=policy.q.scoreStatsObservation(obs),valueAfter=softmax(after.valueScores,1);
const semanticDrift=Math.max(...semanticBefore.map((v,i)=>Math.abs(v-after.semanticScores[i])));

assert.ok(result?.updates>0,"measured target fit must update critic");
assert.ok(semanticDrift<1e-8,"counterfactual consequence fitting must not overwrite semantic prior");
assert.ok(valueAfter[1]>valueBefore[1],"measured best action must gain critic probability");
assert.ok(policy.q.parameterCount()<10000);
console.log(JSON.stringify({ok:true,params:policy.q.parameterCount(),semanticDrift,queryGradient,valueBefore,valueAfter,result}));
