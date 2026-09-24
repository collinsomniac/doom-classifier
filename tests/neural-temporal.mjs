import assert from "node:assert/strict";
import {SemanticResidualPolicy} from "../src/core/policy.js";

class ZeroSemantic{
  constructor(){this.name="ZeroSemantic"}
  compile(schema,actions){this.actions=actions}
  async score(){return this.actions.map(()=>0)}
}

const schema={fields:[
  {id:"signal",label:"signal level",description:"current normalized operating signal",min:0,max:1}
],collections:[]};
const actions=[
  {id:"increase",label:"increase",description:"increase the controlled quantity"},
  {id:"hold",label:"hold",description:"keep the current setting"}
];

function makePolicy(){
  return new SemanticResidualPolicy({schema,actions,semantic:new ZeroSemantic(),residual:"neural-set",inferenceMode:"neural",seed:77});
}

const withMemory=makePolicy();
await withMemory.decide({signal:0},{useResidual:true,memory:true});
const remembered=await withMemory.decide({signal:1},{useResidual:true,memory:true});

const withoutMemory=makePolicy();
await withoutMemory.decide({signal:0},{useResidual:true,memory:false});
const instantaneous=await withoutMemory.decide({signal:1},{useResidual:true,memory:false});

assert.ok(remembered.temporal.some((v,i)=>i>0&&Math.abs(v)>1e-8),"enabled memory must produce a non-zero temporal delta");
assert.ok(instantaneous.temporal.every(v=>Math.abs(v)<1e-8),"disabled memory must zero the temporal channel");
const diff=remembered.qScores.reduce((m,v,i)=>Math.max(m,Math.abs(v-instantaneous.qScores[i])),0);
assert.ok(diff>1e-6,"neural scores must depend on temporal context, not only the instantaneous observation");

const next=withMemory.encode({signal:.5},true,false);
const learned=withMemory.learn({
  observation:{signal:1},temporal:remembered.temporal,features:remembered.features,actionIndex:remembered.actionIndex,reward:.5,
  nextObservation:{signal:.5},nextTemporal:next.temporal,nextFeatures:next.features,done:false
});
assert.ok(Number.isFinite(learned.td));
console.log(JSON.stringify({ok:true,temporal:Array.from(remembered.temporal),scoreDiff:diff,td:learned.td}));
