import assert from "node:assert/strict";
import {NeuralSetResidualQ} from "../src/core/neural-set-residual.js";

const schema={
  fields:[
    {id:"energy",label:"energy",description:"remaining operating energy",min:0,max:100},
    {id:"load",label:"load",description:"current system load",min:0,max:1}
  ],
  collections:[{
    id:"candidates",label:"candidate objects",description:"variable nearby candidates",
    fields:[
      {id:"distance",label:"distance",description:"distance from controller",scale:100},
      {id:"quality",label:"quality",description:"candidate utility quality",min:0,max:1},
      {id:"active",label:"active",description:"candidate is currently active",min:0,max:1}
    ]
  }]
};
const actions=[
  {id:"a",label:"approach",description:"move toward a useful candidate"},
  {id:"b",label:"hold",description:"preserve current position"}
];
const obs={energy:70,load:.4,_collections:{candidates:[{distance:30,quality:.9,active:1},{distance:80,quality:.2,active:0}]}};
const permuted={energy:70,load:.4,_collections:{candidates:[...obs._collections.candidates].reverse()}};
const net=new NeuralSetResidualQ(schema,actions,{seed:7});
const a=net.scoresObservation(obs),b=net.scoresObservation(permuted);
assert.equal(a.length,2);
assert.ok(a.every(Number.isFinite));
assert.ok(a.every((v,i)=>Math.abs(v-b[i])<1e-6),"set encoder must be permutation invariant");
const before=[...a];
for(let i=0;i<100;i++)net.updateTransition({observation:obs,actionIndex:0,reward:1,nextObservation:obs,done:true});
const after=net.scoresObservation(obs);
assert.ok(after[0]>before[0],"positive terminal reward should raise chosen action score");
assert.ok(net.parameterCount()<10000);
const teacher=net.distill(obs,[3,-2]);
assert.ok(Number.isFinite(teacher.loss));
console.log(JSON.stringify({ok:true,params:net.parameterCount(),before,after,distillLoss:teacher.loss}));
