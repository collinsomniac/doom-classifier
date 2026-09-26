import assert from "node:assert/strict";
import {NeuralSetResidualQ} from "../src/core/neural-set-residual.js";

const schema={
  objective:"Choose actions that maximize observed reward.",
  actionFields:[{id:"kind",label:"kind",description:"generic action kind",enum:{0:"good",1:"neutral",2:"bad"}}],
  fields:[{id:"signal",label:"signal",description:"current task signal",min:0,max:1}],
  collections:[]
};
const actions=[
  {id:"good",label:"good",description:"high-return action",params:{kind:0}},
  {id:"neutral",label:"neutral",description:"medium-return action",params:{kind:1}},
  {id:"bad",label:"bad",description:"low-return action",params:{kind:2}}
];
const obs={signal:.7,_collections:{}},rewards=[1,.15,-.75];

function make(alpha){return new NeuralSetResidualQ(schema,actions,{seed:77,useTargetNetwork:false,bootstrapProbability:1,lr:.012,advantageGapAlpha:alpha})}
function train(model){
  for(let epoch=0;epoch<120;epoch++)for(let actionIndex=0;actionIndex<actions.length;actionIndex++){
    model.updateTransition({observation:obs,actionIndex,reward:rewards[actionIndex],nextObservation:obs,done:true});
  }
  return model.valueScoresObservation(obs);
}
const ordinary=make(0),gapModel=make(.5);
const base=train(ordinary),gap=train(gapModel);
const baseGap=base[0]-base[1],learnedGap=gap[0]-gap[1];
assert.ok(gap[0]>gap[1]&&gap[1]>gap[2]);
assert.ok(learnedGap>baseGap*1.15,`gap operator should enlarge good-vs-neutral margin: base=${baseGap}, gap=${learnedGap}`);

const neutralIndex=1,neutralBefore=gapModel.valueScoresObservation(obs);
const info=gapModel.updateTransition({observation:obs,actionIndex:neutralIndex,reward:rewards[neutralIndex],nextObservation:obs,done:true});
assert.ok(info.actionGap>0);
assert.ok(info.gapPenalty>0);
assert.ok(Math.abs(info.gapPenalty-info.advantageGapAlpha*info.actionGap)<1e-9);

const greedyIndex=gapModel.valueScoresObservation(obs).reduce((best,v,i,a)=>v>a[best]?i:best,0);
const greedyInfo=gapModel.updateTransition({observation:obs,actionIndex:greedyIndex,reward:rewards[greedyIndex],nextObservation:obs,done:true});
assert.ok(greedyInfo.gapPenalty<1e-7,"greedy action should retain ordinary Bellman target");

console.log(JSON.stringify({
  ok:true,params:gapModel.parameterCount(),updates:gapModel.updates,alpha:gapModel.advantageGapAlpha,
  ordinary:actions.map((a,i)=>({id:a.id,q:base[i]})),
  gap:actions.map((a,i)=>({id:a.id,q:gap[i]})),
  baseGap,learnedGap,ratio:learnedGap/baseGap,
  probe:{actionGap:info.actionGap,gapPenalty:info.gapPenalty}
}));
