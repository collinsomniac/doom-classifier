import assert from "node:assert/strict";
import {SemanticResidualPolicy} from "../src/core/policy.js";
import {HashSemanticAdapter} from "../src/core/semantic.js";

const schema={objective:"choose useful typed action",fields:[{id:"x",label:"signal",description:"generic state signal",min:0,max:1}],collections:[]};
const actions=[{id:"a",label:"a",description:"first action"},{id:"b",label:"b",description:"second action"}];
const obs={x:.7,_collections:{}};
const policy=new SemanticResidualPolicy({schema,actions,semantic:new HashSemanticAdapter(),residual:"neural-set",seed:72,inferenceMode:"neural"});

function fit(){
  policy.fitProbabilityCalibration([
    {logits:[2,-1],label:0},{logits:[1.8,-.8],label:0},{logits:[-1,2],label:1},{logits:[-.8,1.8],label:1}
  ],{steps:48});
  assert.equal(policy.probabilityCalibrator.fitted,true);
}

fit();
policy.superviseDecisionDistribution(obs,[.8,.2],{steps:1,strength:.1});
assert.equal(policy.probabilityCalibrator.fitted,false,"proper-score update must invalidate calibration");

fit();
for(let i=0;i<4;i++)policy.learn({observation:obs,temporal:null,features:null,actionIndex:0,reward:.2,nextObservation:obs,nextTemporal:null,nextFeatures:null,done:false,nStepHorizon:1});
assert.equal(policy.probabilityCalibrator.fitted,false,"reward/value update must invalidate calibration");

fit();
policy.applyTeacherScores(obs,[2,-1],1,null,{strength:.1});
assert.equal(policy.probabilityCalibrator.fitted,false,"semantic teacher update must invalidate calibration");

fit();
policy.setSemantic(new HashSemanticAdapter());
assert.equal(policy.probabilityCalibrator.fitted,false,"changing semantic teacher must invalidate calibration");

console.log(JSON.stringify({ok:true}));
