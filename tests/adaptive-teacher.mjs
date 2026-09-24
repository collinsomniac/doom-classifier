import assert from "node:assert/strict";
import {SemanticResidualPolicy} from "../src/core/policy.js";

class CountingSemantic{
  constructor(){this.name="CountingSemantic";this.calls=0}
  compile(schema,actions){this.actions=actions}
  async score(){this.calls++;return this.actions.map((_,i)=>i===0?2:-1)}
}
const schema={fields:[{id:"x",label:"signal",description:"generic signal",min:0,max:1}],collections:[{id:"items",label:"items",description:"candidate records",fields:[{id:"v",label:"value",description:"candidate value",min:0,max:1}]}]};
const actions=[{id:"go",label:"go",description:"perform the useful operation"},{id:"hold",label:"hold",description:"do nothing for now"}];
const semantic=new CountingSemantic();
const policy=new SemanticResidualPolicy({schema,actions,semantic,residual:"neural-set",inferenceMode:"adaptive",teacherInterval:3,teacherMinGap:3,teacherEntropy:2,teacherMargin:-1,teacherNovelty:2,distillSteps:2,seed:12});
const obs={x:.5,_collections:{items:[{v:.7},{v:.2}]}};
const used=[];
for(let i=0;i<7;i++){const d=await policy.decide(obs,{useResidual:true});used.push(d.teacherUsed)}
assert.deepEqual(used,[true,false,false,true,false,false,true]);
assert.equal(semantic.calls,3);
const before=semantic.calls;policy.setInferenceMode("neural");
for(let i=0;i<5;i++)await policy.decide(obs,{useResidual:true});
assert.equal(semantic.calls,before,"neural-only path must not invoke semantic teacher");
policy.setInferenceMode("hybrid");await policy.decide(obs,{useResidual:true});assert.equal(semantic.calls,before+1);
console.log(JSON.stringify({ok:true,teacherPattern:used,semanticCalls:semantic.calls,distillUpdates:policy.q.distillUpdates}));
