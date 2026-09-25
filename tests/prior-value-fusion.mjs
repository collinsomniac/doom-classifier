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
  priorKlBudget:.08,valueBetaMax:64,valueTrustUpdates:100,inferenceMode:"neural"
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
  priorKlBudget:.8,valueBetaMax:64,valueTrustUpdates:1,inferenceMode:"neural"
});
const moved=await permissive.decide(obs,{useResidual:true,memory:false,explore:false});
assert.equal(moved.action.id,"value","larger KL budget must allow learned consequence value to override the prior");

const flatQ={...q,updates:100,scoreStatsObservation(){return{
  scores:[0,0,0],semanticScores:[.8,0,-.4],valueScores:[1,1,1],
  valueMemberScores:[[1,1,1],[1,1,1],[1,1,1]],memberScores:[[0,0,0],[0,0,0],[0,0,0]]
}}};
const invariant=new SemanticResidualPolicy({schema,actions,semantic,residual:flatQ,temperature:.5,priorKlBudget:.8,valueTrustUpdates:1,inferenceMode:"neural"});
const same=await invariant.decide(obs,{useResidual:true,memory:false,explore:false});
assert.ok(same.probs.every((x,i)=>Math.abs(x-prior[i])<1e-8),"action-invariant value offsets must not alter semantic prior");
assert.ok(same.priorKL<1e-10);

console.log(JSON.stringify({ok:true,beta:d.valueBeta,priorKL:d.priorKL,moved:moved.action.id,invariantKL:same.priorKL}));
