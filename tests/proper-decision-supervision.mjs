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
const obs={pressure:.92,_collections:{}},target=[.06,.19,.75];

function probs(){return softmax(policy.q.scoreStatsObservation(obs).semanticScores,1)}
function ce(p){return-target.reduce((s,y,i)=>s+y*Math.log(Math.max(1e-9,p[i])),0)}
function brier(p){return p.reduce((s,v,i)=>s+(v-target[i])**2,0)}

const before=probs(),beforeCE=ce(before),beforeBrier=brier(before);
for(let i=0;i<320;i++)policy.superviseDecisionDistribution(obs,target,{strength:.22});
const after=probs(),afterCE=ce(after),afterBrier=brier(after);

assert.ok(afterCE<beforeCE*.72,"proper-scoring supervision should reduce target cross-entropy materially");
assert.ok(afterBrier<beforeBrier*.5,"proper-scoring supervision should reduce target Brier error materially");
assert.ok(Math.abs(after[2]-target[2])<Math.abs(before[2]-target[2]),"target action probability should move toward its requested probability");
assert.equal(after.indexOf(Math.max(...after)),2);
assert.equal(policy.q.properScoreUpdates,320);
assert.equal(policy.probabilityCalibrator.fitted,false,"changing model logits must invalidate old post-hoc calibration");
const checkpoint=policy.exportCheckpoint();
assert.equal(checkpoint.q.properScoreUpdates,320);
console.log(JSON.stringify({ok:true,beforeCE,afterCE,beforeBrier,afterBrier,before,after,target,properScoreUpdates:policy.q.properScoreUpdates,params:policy.q.parameterCount()}));
