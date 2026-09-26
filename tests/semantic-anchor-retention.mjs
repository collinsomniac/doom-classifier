import assert from "node:assert/strict";
import {SemanticResidualPolicy} from "../src/core/policy.js";
import {softmax} from "../src/core/math.js";

const schema={
  objective:"choose the action best supported by current structured state",
  fields:[
    {id:"x",label:"signal x",description:"primary task signal",min:0,max:1},
    {id:"y",label:"signal y",description:"secondary task signal",min:0,max:1}
  ],
  collections:[]
};
const actions=[
  {id:"anchor_action",label:"anchor action",description:"action preferred in the bootstrap regime"},
  {id:"later_action",label:"later action",description:"action preferred in later regimes"},
  {id:"neutral_action",label:"neutral action",description:"neutral fallback"}
];
const semantic={name:"stub",backend:"test",compile(){},async score(){return[0,0,0]}};
const anchorObs={x:.05,y:.95,_collections:{}},anchorScores=[4,-1,-2];
const laterStates=Array.from({length:14},(_,i)=>({
  obs:{x:.35+(i%5)*.13,y:.05+(i%3)*.1,_collections:{}},
  scores:[-1,4-(i%2)*.25,-2]
}));

function make(anchorCapacity){
  return new SemanticResidualPolicy({
    schema,actions,semantic,residual:"neural-set",seed:733,
    teacherReplayCapacity:4,teacherReplayBatch:2,teacherReplayStrength:.45,teacherAnchorCapacity:anchorCapacity,
    inferenceMode:"neural"
  });
}
function semanticProbs(policy,obs){
  return softmax(policy.q.scoreStatsObservation(obs).semanticScores,1);
}
function kl(target,pred){
  let out=0;for(let i=0;i<target.length;i++)if(target[i]>0)out+=target[i]*Math.log(target[i]/Math.max(1e-9,pred[i]));return out;
}

const anchored=make(1),unanchored=make(0);
anchored.applyTeacherScores(anchorObs,anchorScores,8,null,{strength:1.5,anchor:true});
unanchored.applyTeacherScores(anchorObs,anchorScores,8,null,{strength:1.5,anchor:true});
const target=softmax(anchorScores,1),initialA=semanticProbs(anchored,anchorObs),initialU=semanticProbs(unanchored,anchorObs);

for(const item of laterStates){
  anchored.applyTeacherScores(item.obs,item.scores,4,null,{strength:.65});
  unanchored.applyTeacherScores(item.obs,item.scores,4,null,{strength:.65});
}

const finalA=semanticProbs(anchored,anchorObs),finalU=semanticProbs(unanchored,anchorObs);
const anchoredKL=kl(target,finalA),unanchoredKL=kl(target,finalU);
assert.equal(anchored.teacherAnchors.length,1);
assert.equal(unanchored.teacherAnchors.length,0);
assert.ok(finalA[0]>finalU[0],"pinned semantic rehearsal should preserve more probability on the bootstrap action");
assert.ok(anchoredKL<unanchoredKL,"pinned semantic rehearsal should reduce bootstrap-state KL drift");

console.log(JSON.stringify({
  ok:true,
  initialAnchorProb:initialA[0],
  initialNoAnchorProb:initialU[0],
  finalAnchorProb:finalA[0],
  finalNoAnchorProb:finalU[0],
  anchoredKL,
  unanchoredKL,
  replaySize:anchored.teacherReplay.length,
  anchors:anchored.teacherAnchors.length
}));
