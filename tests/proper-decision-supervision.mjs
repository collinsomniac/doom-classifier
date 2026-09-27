import assert from "node:assert/strict";
import {SemanticResidualPolicy} from "../src/core/policy.js";
import {HashSemanticAdapter} from "../src/core/semantic.js";
import {softmax} from "../src/core/math.js";

const schema={
  objective:"route work using queue urgency and available budget",
  fields:[{id:"pressure",label:"queue pressure",description:"normalized work pressure",min:0,max:1}],
  collections:[],
  actionFields:[{id:"mode",label:"routing mode",description:"typed routing choice",enum:{0:"defer",1:"balanced",2:"urgent"}}]
};
const actions=[
  {id:"defer",label:"defer",description:"delay work briefly",params:{mode:0}},
  {id:"balanced",label:"balanced",description:"use balanced service",params:{mode:1}},
  {id:"urgent",label:"urgent",description:"prioritize immediate service",params:{mode:2}}
];
const policy=new SemanticResidualPolicy({schema,actions,semantic:new HashSemanticAdapter(),residual:"neural-set",seed:909,inferenceMode:"neural"});
const low={pressure:.12,_collections:{}},high={pressure:.92,_collections:{}};
const lowTarget=[.82,.14,.04],highTarget=[.04,.16,.80];

function probs(obs){return softmax(policy.q.scoreStatsObservation(obs).semanticScores,1)}
function ce(target,p){return-target.reduce((s,y,i)=>s+y*Math.log(Math.max(1e-9,p[i])),0)}
const beforeLow=probs(low),beforeHigh=probs(high),before=(ce(lowTarget,beforeLow)+ce(highTarget,beforeHigh))/2;

for(let epoch=0;epoch<240;epoch++){
  policy.superviseDecisionDistribution(low,lowTarget,{strength:.8});
  policy.superviseDecisionDistribution(high,highTarget,{strength:.8});
}
const afterLow=probs(low),afterHigh=probs(high),after=(ce(lowTarget,afterLow)+ce(highTarget,afterHigh))/2;
assert.ok(after<before*.8,"proper-scoring supervision should reduce cross-entropy materially");
assert.ok(afterLow[0]>beforeLow[0],"low-pressure defer probability should move toward target");
assert.ok(afterHigh[2]>beforeHigh[2],"high-pressure urgent probability should move toward target");
assert.equal(afterLow.indexOf(Math.max(...afterLow)),0);
assert.equal(afterHigh.indexOf(Math.max(...afterHigh)),2);
assert.equal(policy.q.properScoreUpdates,480);
assert.equal(policy.probabilityCalibrator.fitted,false,"changing model logits must invalidate old post-hoc calibration");
const checkpoint=policy.exportCheckpoint();
assert.equal(checkpoint.q.properScoreUpdates,480);
console.log(JSON.stringify({ok:true,before,after,beforeLow,beforeHigh,afterLow,afterHigh,properScoreUpdates:policy.q.properScoreUpdates,params:policy.q.parameterCount()}));
