import assert from "node:assert/strict";
import {SemanticResidualPolicy} from "../src/core/policy.js";

class DeferredSemantic{
  constructor(){this.name="DeferredSemantic";this.calls=0;this.pending=[]}
  compile(schema,actions){this.actions=actions}
  score(){
    this.calls++;
    return new Promise(resolve=>this.pending.push(()=>resolve(this.actions.map((_,i)=>i===0?2:-1))));
  }
  resolveNext(){const fn=this.pending.shift();if(fn)fn()}
}
const schema={fields:[{id:"x",label:"signal",description:"generic signal",min:0,max:1}],collections:[{id:"items",label:"items",description:"candidate records",fields:[{id:"v",label:"value",description:"candidate value",min:0,max:1}]}]};
const actions=[{id:"go",label:"go",description:"perform the useful operation"},{id:"hold",label:"hold",description:"do nothing for now"}];
const semantic=new DeferredSemantic();
const policy=new SemanticResidualPolicy({schema,actions,semantic,residual:"neural-set",inferenceMode:"adaptive",teacherInterval:3,teacherMinGap:3,teacherEntropy:2,teacherMargin:-1,teacherNovelty:2,distillSteps:2,seed:12});
const obs={x:.5,_collections:{items:[{v:.7},{v:.2}]}};

const first=await policy.decide(obs,{useResidual:true});
assert.equal(first.teacherUsed,true);
assert.equal(first.semanticLatencyMs,0,"adaptive tick must not await semantic teacher");
assert.equal(semantic.calls,1);
assert.equal(policy.teacherCalls,0);
assert.ok(policy.teacherPromise,"teacher must continue asynchronously");
semantic.resolveNext();await policy.awaitTeacher();
assert.equal(policy.teacherCalls,1);
assert.equal(policy.q.distillUpdates,2);

const cadence=[];
for(let i=0;i<6;i++){
  const d=await policy.decide(obs,{useResidual:true});cadence.push(d.teacherUsed);
  if(d.teacherUsed){semantic.resolveNext();await policy.awaitTeacher()}
}
assert.deepEqual(cadence,[false,false,true,false,false,true]);
const before=semantic.calls;
policy.setInferenceMode("neural");
for(let i=0;i<4;i++)await policy.decide(obs,{useResidual:true});
assert.equal(semantic.calls,before,"neural-only path must not invoke semantic teacher");

const primeSemantic=new DeferredSemantic(),primePolicy=new SemanticResidualPolicy({schema,actions,semantic:primeSemantic,residual:"neural-set",inferenceMode:"adaptive",seed:9});
const primePromise=primePolicy.primeTeacher(obs,{steps:3});
assert.equal(primeSemantic.calls,1);primeSemantic.resolveNext();const prime=await primePromise;
assert.equal(prime.stale,false);assert.equal(primePolicy.teacherCalls,1);assert.equal(primePolicy.q.distillUpdates,3);
const primedTick=await primePolicy.decide(obs,{useResidual:true});
assert.equal(primedTick.teacherUsed,false,"bootstrap should keep the first control tick on the neural fast path");

console.log(JSON.stringify({ok:true,cadence,semanticCalls:semantic.calls,distillUpdates:policy.q.distillUpdates,primeDistill:primePolicy.q.distillUpdates}));
