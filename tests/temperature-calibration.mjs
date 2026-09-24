import assert from "node:assert/strict";
import {SemanticResidualPolicy} from "../src/core/policy.js";

class SharpTeacher{
  constructor(){this.name="sharp"}compile(schema,actions){this.actions=actions}
  async score(){return this.actions.map((_,i)=>i===0?Math.log(.92):Math.log(.08/(this.actions.length-1)))}
}
const schema={fields:[{id:"x",label:"signal",description:"generic signal",min:0,max:1}],collections:[]};
const actions=[
  {id:"a",label:"act",description:"perform the useful operation"},
  {id:"b",label:"hold",description:"hold state"},
  {id:"c",label:"other",description:"perform another operation"}
];
const obs={x:.5,_collections:{}};
const policy=new SemanticResidualPolicy({schema,actions,semantic:new SharpTeacher(),residual:"neural-set",inferenceMode:"adaptive",temperature:.8,seed:17});
const before=policy.temperature;
const prime=await policy.primeTeacher(obs,{steps:20});
assert.equal(prime.stale,false);
assert.ok(prime.calibration);
assert.ok(policy.temperature>=.3&&policy.temperature<=1.5);
assert.ok(Number.isFinite(prime.calibration.loss));

policy.setInferenceMode("hybrid");
const teacherOnly=await policy.decide(obs,{useResidual:false});
const expected=[.92,.04,.04];
for(let i=0;i<expected.length;i++)assert.ok(Math.abs(teacherOnly.probs[i]-expected[i])<1e-5,"teacher-only mode must preserve teacher probabilities");

policy.resetLearning();
assert.equal(policy.temperature,policy.baseTemperature);
console.log(JSON.stringify({ok:true,before,calibrated:prime.calibration.temperature,teacherOnly:teacherOnly.probs}));
