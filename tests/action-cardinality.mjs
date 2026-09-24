import {performance} from "node:perf_hooks";
import {NeuralSetResidualQ} from "../src/core/neural-set-residual.js";

const schema={
  fields:Array.from({length:8},(_,i)=>({id:"g"+i,label:"global signal "+i,description:"generic global numeric signal "+i,scale:10})),
  collections:[{id:"items",label:"candidate records",description:"variable structured candidates",fields:Array.from({length:8},(_,i)=>({id:"v"+i,label:"record signal "+i,description:"generic record numeric attribute "+i,scale:10}))}]
};
const obs={_collections:{items:[]}};for(let i=0;i<8;i++)obs["g"+i]=i;
for(let r=0;r<128;r++)obs._collections.items.push(Object.fromEntries(Array.from({length:8},(_,i)=>["v"+i,Math.sin(r*.13+i)*10])));
const baseActions=Array.from({length:8},(_,i)=>({id:"a"+i,label:"operation "+i,description:"perform typed operation family "+i}));
const net=new NeuralSetResidualQ(schema,baseActions,{seed:4}),params=net.parameterCount(),results=[];
for(const count of [8,32,128,256,512,1024]){
  const actions=Array.from({length:count},(_,i)=>({id:"choice_"+i,label:"choice "+i,description:"perform candidate typed operation "+i}));
  const c0=performance.now();net.setActions(actions);const compileMs=performance.now()-c0;
  for(let i=0;i<3;i++)net.scoresObservation(obs);
  const times=[];for(let i=0;i<20;i++){const t=performance.now();const scores=net.scoresObservation(obs);times.push(performance.now()-t);if(scores.length!==count)throw new Error("bad action count")}
  times.sort((a,b)=>a-b);const p=q=>times[Math.floor((times.length-1)*q)];
  results.push({actions:count,compileMs,p50Ms:p(.5),p95Ms:p(.95),params:net.parameterCount()});
}
if(results.some(r=>r.params!==params))throw new Error("trainable parameter count changed with action cardinality");
if(results.at(-1).p95Ms>100)throw new Error("1024-choice fast path exceeded guardrail: "+JSON.stringify(results.at(-1)));
console.log(JSON.stringify({ok:true,params,results},null,2));
