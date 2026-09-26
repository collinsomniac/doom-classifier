import assert from "node:assert/strict";
import {SemanticResidualPolicy} from "../src/core/policy.js";
import {HashSemanticScorer} from "../src/core/semantic.js";

const schema={objective:"choose a controller input",fields:[{id:"x",label:"x",description:"state value",min:0,max:1}],collections:[],actionFields:[{id:"go",label:"go",description:"go control",min:0,max:1}]};
const actions=[{id:"go",label:"go",description:"apply go control",params:{go:1}},{id:"wait",label:"wait",description:"apply no control",params:{go:0}}];
const semantic=new HashSemanticScorer();const source=new SemanticResidualPolicy({schema,actions,semantic,residual:"neural-set",seed:44,inferenceMode:"neural"});
const obs={x:.75,_collections:{}};
for(let i=0;i<20;i++)source.q.distill(obs,[2,-1],{strength:.7});
for(let i=0;i<15;i++)source.q.updateTransition({observation:obs,actionIndex:0,reward:.6,nextObservation:obs,done:true});
source.temperature=.61;
const before=source.q.scoreStatsObservation(obs),checkpoint=source.exportCheckpoint();
assert.equal(checkpoint.q.params,source.q.parameterCount());
const target=new SemanticResidualPolicy({schema:{...schema,objective:"placeholder"},actions:[...actions].reverse(),semantic:new HashSemanticScorer(),residual:"neural-set",seed:999,inferenceMode:"adaptive"});
target.importCheckpoint(checkpoint);
const after=target.q.scoreStatsObservation(obs);
for(const key of ["scores","semanticScores","valueScores"])for(let i=0;i<before[key].length;i++)assert.ok(Math.abs(before[key][i]-after[key][i])<1e-7,key+" round-trip mismatch");
assert.deepEqual(target.actions.map(a=>a.id),["go","wait"]);assert.equal(target.inferenceMode,"neural");assert.equal(target.temperature,.61);
console.log(JSON.stringify({ok:true,params:target.q.parameterCount(),bytes:JSON.stringify(checkpoint).length}));
