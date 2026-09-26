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
const alphas=[0,.05,.1,.2,.3,.4,.5],results=[];
for(const alpha of alphas){
  const model=make(alpha),scores=train(model),runnerUp=Math.max(...scores.slice(1)),topCorrect=scores[0]===Math.max(...scores),topGap=scores[0]-runnerUp;
  results.push({alpha,topCorrect,scores:[...scores],topGap,gapGoodNeutral:scores[0]-scores[1],gapGoodBad:scores[0]-scores[2]});
}
const baseline=results[0],safe=results.slice(1).filter(x=>x.topCorrect&&x.topGap>baseline.topGap),best=[...safe].sort((a,b)=>b.topGap-a.topGap)[0]||null;
console.log("ACTION_GAP_SWEEP "+JSON.stringify({params:make(0).parameterCount(),baseline,results,best}));
assert.ok(best,"at least one positive alpha should preserve the greedy action and enlarge its runner-up gap");

// Diagnostic gate: the implementation default must match the empirically safe alpha before DOOM runs.
const configured=make(undefined).advantageGapAlpha;
assert.equal(configured,best.alpha,`set branch default alpha to safe sweep winner ${best.alpha}, currently ${configured}`);
