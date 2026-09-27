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
const lowTarget=[.80,.16,.04],highTarget=[.05,.18,.77];

function probs(obs){return softmax(policy.q.scoreStatsObservation(obs).semanticScores,1)}
function ce(target,p){return-target.reduce((s,y,i)=>s+y*Math.log(Math.max(1e-9,p[i])),0)}
function brier(target,p){return p.reduce((s,v,i)=>s+(v-target[i])**2,0)}
const initialHigh=probs(high),initialCE=ce(highTarget,initialHigh);

// The differentiable proper-score path must make a small local step in the correct direction.
policy.superviseDecisionDistribution(high,highTarget,{steps:1,strength:.04});
const oneStep=probs(high);
assert.ok(ce(highTarget,oneStep)<initialCE,"single proper-score gradient step must decrease cross-entropy");

// The fast fitting path solves the small semantic head against probability targets in one batch.
const beforeLow=probs(low),beforeHigh=probs(high),before=(ce(lowTarget,beforeLow)+ce(highTarget,beforeHigh))/2;
const fit=policy.fitDecisionDistributions([
  {observation:low,target:lowTarget},
  {observation:high,target:highTarget}
],{ridge:.0002,refineSteps:0});
const afterLow=probs(low),afterHigh=probs(high),after=(ce(lowTarget,afterLow)+ce(highTarget,afterHigh))/2;

assert.ok(fit?.rows>=6);
console.log("PROPER_SCORE_DIAGNOSTIC "+JSON.stringify({initialCE,oneStepCE:ce(highTarget,oneStep),before,after,beforeLow,beforeHigh,afterLow,afterHigh,fit}));
assert.ok(after<before*.82,"closed-form probability fitting should reduce batch cross-entropy materially");
assert.ok((brier(lowTarget,afterLow)+brier(highTarget,afterHigh))<(brier(lowTarget,beforeLow)+brier(highTarget,beforeHigh))*.75,"Brier error should improve materially");
assert.ok(afterLow[0]>beforeLow[0],"low-pressure defer probability should move toward target");
assert.ok(afterHigh[2]>beforeHigh[2],"high-pressure urgent probability should move toward target");
assert.equal(afterLow.indexOf(Math.max(...afterLow)),0);
assert.equal(afterHigh.indexOf(Math.max(...afterHigh)),2);
assert.equal(policy.q.properScoreUpdates,3);
assert.equal(policy.probabilityCalibrator.fitted,false,"changing model logits must invalidate old post-hoc calibration");
const checkpoint=policy.exportCheckpoint();
assert.equal(checkpoint.q.properScoreUpdates,policy.q.properScoreUpdates);
console.log(JSON.stringify({ok:true,initialCE,oneStepCE:ce(highTarget,oneStep),before,after,beforeLow,beforeHigh,afterLow,afterHigh,properScoreUpdates:policy.q.properScoreUpdates,params:policy.q.parameterCount()}));
