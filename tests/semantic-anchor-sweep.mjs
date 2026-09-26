import assert from "node:assert/strict";
import {SemanticResidualPolicy} from "../src/core/policy.js";
import {softmax} from "../src/core/math.js";

const schema={objective:"adapt to structured state without forgetting prior regimes",fields:[{id:"x",label:"x",min:0,max:1},{id:"y",label:"y",min:0,max:1}],collections:[]};
const actions=[{id:"anchor",label:"anchor"},{id:"later",label:"later"},{id:"neutral",label:"neutral"}];
const semantic={name:"stub",backend:"test",compile(){},async score(){return[0,0,0]}};
const anchorObs={x:.05,y:.95,_collections:{}},anchorScores=[4,-1,-2],anchorTarget=softmax(anchorScores,1);
const stream=Array.from({length:14},(_,i)=>({obs:{x:.35+(i%5)*.13,y:.05+(i%3)*.1,_collections:{}},scores:[-1,4-(i%2)*.25,-2]}));
function kl(target,pred){let out=0;for(let i=0;i<target.length;i++)if(target[i]>0)out+=target[i]*Math.log(target[i]/Math.max(1e-9,pred[i]));return out}
function probs(policy,obs){return softmax(policy.q.scoreStatsObservation(obs).semanticScores,1)}
function run({steps,strength}){
  const p=new SemanticResidualPolicy({schema,actions,semantic,residual:"neural-set",seed:733,teacherReplayCapacity:4,teacherReplayBatch:2,teacherReplayStrength:.45,teacherAnchorCapacity:1,teacherAnchorRehearsal:steps,teacherAnchorStrength:strength,inferenceMode:"neural"});
  p.applyTeacherScores(anchorObs,anchorScores,8,null,{strength:1.5,anchor:true});
  for(const item of stream)p.applyTeacherScores(item.obs,item.scores,4,null,{strength:.65});
  const anchorPred=probs(p,anchorObs),latest=stream.at(-1),latestTarget=softmax(latest.scores,1),latestPred=probs(p,latest.obs);
  return{steps,strength,anchorP:anchorPred[0],anchorKL:kl(anchorTarget,anchorPred),latestP:latestPred[1],latestKL:kl(latestTarget,latestPred),jointKL:kl(anchorTarget,anchorPred)+kl(latestTarget,latestPred)};
}
const configs=[
  {steps:0,strength:0},
  {steps:1,strength:.25},{steps:1,strength:.45},{steps:1,strength:.7},
  {steps:2,strength:.25},{steps:2,strength:.45},{steps:2,strength:.7},
  {steps:3,strength:.45},{steps:3,strength:.7},
  {steps:4,strength:.45}
];
const results=configs.map(run),best=[...results].sort((a,b)=>a.jointKL-b.jointKL)[0];
assert.ok(results.every(r=>Number.isFinite(r.jointKL)));
assert.ok(best.anchorP>.02&&best.latestP>.2,"best rehearsal balance must retain both old and new regimes");
console.log("SEMANTIC_ANCHOR_SWEEP "+JSON.stringify({best,results}));
