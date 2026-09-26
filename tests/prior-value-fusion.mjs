import assert from "node:assert/strict";
import {SemanticResidualPolicy} from "../src/core/policy.js";
import {softmax} from "../src/core/math.js";

const schema={fields:[{id:"x",label:"x",min:0,max:1}],collections:[]};
const actions=[{id:"prior",label:"prior"},{id:"value",label:"value"},{id:"other",label:"other"}];
const semantic={name:"stub",backend:"test",compile(){},async score(){return[0,0,0]}};
const q={
  updates:0,
  scoreStatsObservation(){
    return{
      scores:[0,0,0],
      semanticScores:[.8,0,-.4],
      valueScores:[-.2,1,0],
      valueMemberScores:[[-.25,-.2,-.15],[.9,1,1.1],[-.05,0,.05]],
      memberScores:[[0,0,0],[0,0,0],[0,0,0]]
    };
  },
  reset(){this.updates=0},parameterCount(){return 0}
};
const p=new SemanticResidualPolicy({
  schema,actions,semantic,residual:q,temperature:.5,seed:11,
  priorKlBudget:.08,valueBetaMax:64,valueTrustUpdates:100,valueEpistemicBudget:1,criticKlExpansion:0,inferenceMode:"neural"
});
const obs={x:.5,_collections:{}};

let d=await p.decide(obs,{useResidual:true,memory:false,explore:false});
assert.equal(d.action.id,"prior");
assert.equal(d.valueBeta,0,"untrained value branch must have zero policy authority");
assert.equal(d.priorKL,0);
assert.equal(d.valueTrust,0);

q.updates=100;
d=await p.decide(obs,{useResidual:true,memory:false,explore:false});
assert.ok(d.valueBeta>0,"trained value branch should receive bounded policy influence");
assert.ok(d.priorKL<=.0805,"fused policy must remain inside configured KL budget");
assert.ok(d.priorKL>.01,"fusion should spend meaningful KL budget when Q disagrees with prior");
assert.equal(d.valueTrust,1);

const prior=softmax([.8,0,-.4],.5),fused=d.probs;
const manualKL=fused.reduce((s,x,i)=>s+(x>0?x*Math.log(x/prior[i]):0),0);
assert.ok(Math.abs(manualKL-d.priorKL)<1e-6,"reported prior KL must match executed distribution");
assert.equal(d.uncertainty.epistemic>=0,true);

const permissive=new SemanticResidualPolicy({
  schema,actions,semantic,residual:q,temperature:.5,seed:12,
  priorKlBudget:.8,valueBetaMax:64,valueTrustUpdates:1,valueEpistemicBudget:1,criticKlExpansion:0,inferenceMode:"neural"
});
const moved=await permissive.decide(obs,{useResidual:true,memory:false,explore:false});
assert.equal(moved.action.id,"value","larger KL budget must allow learned consequence value to override the prior");


const tinyQ={...q,updates:100,scoreStatsObservation(){return{
  scores:[0,0,0],semanticScores:[.8,0,-.4],valueScores:[-.0002,.001,0],
  valueMemberScores:[[-.00022,-.0002,-.00018],[.0009,.001,.0011],[-.00005,0,.00005]],memberScores:[[0,0,0],[0,0,0],[0,0,0]]
}}};
const adaptive=new SemanticResidualPolicy({
  schema,actions,semantic,residual:tinyQ,temperature:.5,seed:13,
  priorKlBudget:.08,valueBetaMax:4096,valueTrustUpdates:100,valueEpistemicBudget:1,criticKlExpansion:0,inferenceMode:"neural"
});
const scaled=await adaptive.decide(obs,{useResidual:true,memory:false,explore:false});
assert.ok(scaled.valueBeta>64,"adaptive fusion must expand beyond the old arbitrary beta cap when value scores are small");
assert.ok(scaled.priorKL>.07&&scaled.priorKL<=.0805,"adaptive fusion should spend nearly all available KL budget");
assert.ok(scaled.klUtilization>.85,"KL utilization should expose that the trust budget was actually used");
assert.equal(scaled.valueBetaSaturated,false,"beta should stop on KL boundary, not the safety ceiling");

const typedSchema={fields:[{id:"x",label:"x",min:0,max:1}],collections:[],actionFields:[{id:"forward",label:"forward",min:0,max:1},{id:"fire",label:"fire",min:0,max:1}]};
const typedActions=[
  {id:"wait",label:"wait",params:{forward:0,fire:0}},
  {id:"forward",label:"forward",params:{forward:1,fire:0}},
  {id:"fire",label:"fire",params:{forward:0,fire:1}},
  {id:"forward_fire",label:"forward + fire",params:{forward:1,fire:1}}
];
const typedQ={updates:100,scoreStatsObservation(){return{
  scores:[0,0,0,0],semanticScores:[0,.2,.4,.45],valueScores:[0,.5,1,.55],
  valueMemberScores:[[0,.01,-.01],[.48,.5,.52],[.98,1,1.02],[.53,.55,.57]],memberScores:[[0,0,0],[0,0,0],[0,0,0],[0,0,0]]
}},reset(){},parameterCount(){return 0}};
const typedSemantic={name:"stub",backend:"test",compile(){},async score(){return[0,0,0,0]}};
const typedPolicy=new SemanticResidualPolicy({schema:typedSchema,actions:typedActions,semantic:typedSemantic,residual:typedQ,temperature:.7,priorKlBudget:.08,valueTrustUpdates:1,valueEpistemicBudget:1,typedValueBlend:1,criticKlExpansion:0,inferenceMode:"neural"});
const typedDecision=await typedPolicy.decide({x:.5,_collections:{}},{useResidual:true,memory:false,explore:false});
assert.ok(typedDecision.typedValueFit>.4,"policy fusion should expose meaningful typed-action value fit");
assert.ok(typedDecision.typedValueBlendUsed>.4,"typed value smoothing should activate in proportion to fit quality");
assert.equal(typedDecision.typedValueScores.length,typedActions.length);
assert.ok(typedDecision.typedFieldCoefficients.some(x=>x.id==="fire"&&x.weight>0),"typed fusion should recover positive fire contribution");

const confidentAuthority=new SemanticResidualPolicy({
  schema,actions,semantic,residual:q,temperature:.5,seed:121,
  priorKlBudget:.08,valueBetaMax:4096,valueTrustUpdates:100,valueEpistemicBudget:1,criticKlExpansion:1,inferenceMode:"neural"
});
const authorityDecision=await confidentAuthority.decide(obs,{useResidual:true,memory:false,explore:false});
assert.ok(authorityDecision.criticTopAgreement>.99,"all bootstrap heads should agree in the confident fixture");
assert.ok(authorityDecision.criticRankingConfidence>.8,"stable critic ranking should earn high state confidence");
assert.ok(authorityDecision.criticAuthority>.8,"fully trained critic should convert confidence into authority");
assert.ok(authorityDecision.priorKlBudget>.14&&authorityDecision.priorKlBudget<=.1605,"confident critic may expand the base KL budget toward 2x");
assert.ok(authorityDecision.criticKlMultiplier>1.8&&authorityDecision.criticKlMultiplier<=2.001);

const uncertainQ={...q,updates:100,scoreStatsObservation(){return{
  scores:[0,0,0],semanticScores:[.8,0,-.4],valueScores:[0,1,0],
  valueMemberScores:[[-.1,0,.1],[-2,1,4],[-.1,0,.1]],memberScores:[[0,0,0],[0,0,0],[0,0,0]]
}}};
const guarded=new SemanticResidualPolicy({
  schema,actions,semantic,residual:uncertainQ,temperature:.5,seed:14,
  priorKlBudget:.8,valueBetaMax:4096,valueTrustUpdates:1,valueEpistemicBudget:.01,criticKlExpansion:1,inferenceMode:"neural"
});
const guardedDecision=await guarded.decide(obs,{useResidual:true,memory:false,explore:false});
assert.ok(guardedDecision.valueBeta>0,"confident value influence should not be disabled outright");
assert.ok(guardedDecision.valueEpistemic<=.0105,"value authority must stop at the epistemic disagreement budget");
assert.ok(guardedDecision.epistemicUtilization>.8,"uncertain critic should spend the epistemic trust budget");
assert.ok(guardedDecision.priorKL<guardedDecision.priorKlBudget,"epistemic guard should become the active constraint before the permissive KL budget");
assert.ok(guardedDecision.criticTopAgreement<.99,"split bootstrap heads should not look fully confident");
assert.ok(guardedDecision.criticAuthority<.15,"split critic ranking must not materially expand KL authority");

const flatQ={...q,updates:100,scoreStatsObservation(){return{
  scores:[0,0,0],semanticScores:[.8,0,-.4],valueScores:[1,1,1],
  valueMemberScores:[[1,1,1],[1,1,1],[1,1,1]],memberScores:[[0,0,0],[0,0,0],[0,0,0]]
}}};
const invariant=new SemanticResidualPolicy({schema,actions,semantic,residual:flatQ,temperature:.5,priorKlBudget:.8,valueTrustUpdates:1,inferenceMode:"neural"});
const same=await invariant.decide(obs,{useResidual:true,memory:false,explore:false});
assert.ok(same.probs.every((x,i)=>Math.abs(x-prior[i])<1e-8),"action-invariant value offsets must not alter semantic prior");
assert.ok(same.priorKL<1e-10);

console.log(JSON.stringify({ok:true,beta:d.valueBeta,priorKL:d.priorKL,authorityBudget:authorityDecision.priorKlBudget,authority:authorityDecision.criticAuthority,rankingConfidence:authorityDecision.criticRankingConfidence,adaptiveBeta:scaled.valueBeta,adaptiveKL:scaled.priorKL,adaptiveUtilization:scaled.klUtilization,guardedBeta:guardedDecision.valueBeta,guardedEpistemic:guardedDecision.valueEpistemic,guardedAuthority:guardedDecision.criticAuthority,guardedEpistemicUtilization:guardedDecision.epistemicUtilization,moved:moved.action.id,invariantKL:same.priorKL}));
