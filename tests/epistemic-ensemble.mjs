import assert from "node:assert/strict";
import {NeuralSetResidualQ} from "../src/core/neural-set-residual.js";
import {SemanticResidualPolicy} from "../src/core/policy.js";

const schema={fields:[{id:"x",label:"signal",description:"generic system signal",min:0,max:1}],collections:[{
  id:"items",label:"items",description:"variable candidate records",fields:[{id:"v",label:"value",description:"candidate utility",min:0,max:1}]
}]};
const actions=[
  {id:"a",label:"act",description:"perform the candidate operation"},
  {id:"b",label:"hold",description:"preserve current state"},
  {id:"c",label:"inspect",description:"inspect the environment"}
];
const obs={x:.4,_collections:{items:[{v:.2},{v:.7},{v:.9}]}};
const net=new NeuralSetResidualQ(schema,actions,{seed:123,ensembleSize:3});
const stats=net.scoreStatsObservation(obs);
assert.equal(stats.scores.length,actions.length);
assert.equal(stats.memberScores.length,actions.length);
assert.ok(stats.memberScores.every(row=>row.length===3));
const spread=Math.max(...stats.memberScores.flat())-Math.min(...stats.memberScores.flat());
assert.ok(spread>1e-6,"independently initialized ensemble heads must not be identical");
assert.ok(net.parameterCount()<10000);

class ZeroSemantic{
  constructor(){this.name="zero"}compile(schema,actions){this.actions=actions}async score(){return this.actions.map(()=>0)}
}
const policy=new SemanticResidualPolicy({schema,actions,semantic:new ZeroSemantic(),residual:net,inferenceMode:"neural",seed:123});
const decision=await policy.decide(obs,{useResidual:true});
assert.ok(Number.isFinite(decision.uncertainty.epistemic));
assert.ok(decision.uncertainty.epistemic>=0&&decision.uncertainty.epistemic<=1);
assert.ok(decision.uncertainty.epistemic>0,"untrained ensemble should expose non-zero epistemic disagreement");

for(let i=0;i<80;i++)net.distill(obs,[4,-2,-2],{strength:.5});
const trained=await policy.decide(obs,{useResidual:true});
assert.ok(Number.isFinite(trained.uncertainty.epistemic));
console.log(JSON.stringify({ok:true,params:net.parameterCount(),initialEpistemic:decision.uncertainty.epistemic,trainedEpistemic:trained.uncertainty.epistemic,memberScores:stats.memberScores}));
