import {performance} from "node:perf_hooks";
import {NeuralSetResidualQ} from "../src/core/neural-set-residual.js";

const fields=Array.from({length:12},(_,i)=>({id:"f"+i,label:"sensor field "+i,description:"generic numeric record attribute "+i,scale:100}));
const schema={fields:Array.from({length:16},(_,i)=>({id:"g"+i,label:"global field "+i,description:"generic global state attribute "+i,scale:100})),collections:[
  {id:"objects",label:"objects",description:"dynamic object records",fields},
  {id:"geometry",label:"geometry",description:"static geometry records",fields:fields.slice(0,8)}
]};
const actions=Array.from({length:12},(_,i)=>({id:"action_"+i,label:"action "+i,description:"typed candidate operation "+i}));
const observation={_collections:{objects:[],geometry:[]}};
for(let i=0;i<256;i++)observation._collections.objects.push(Object.fromEntries(fields.map((f,j)=>[f.id,Math.sin(i*.17+j)*100])));
for(let i=0;i<512;i++)observation._collections.geometry.push(Object.fromEntries(fields.slice(0,8).map((f,j)=>[f.id,Math.cos(i*.09+j)*100])));
for(let i=0;i<16;i++)observation["g"+i]=i*3;
const net=new NeuralSetResidualQ(schema,actions,{seed:11});
for(let i=0;i<10;i++)net.scoresObservation(observation);
const times=[];
for(let i=0;i<100;i++){const t=performance.now();net.scoresObservation(observation);times.push(performance.now()-t)}
times.sort((a,b)=>a-b);
const p=q=>times[Math.floor((times.length-1)*q)];
const result={params:net.parameterCount(),records:768,actions:12,p50Ms:p(.5),p95Ms:p(.95),p99Ms:p(.99)};
if(result.p95Ms>50)throw new Error("Neural set inference unexpectedly slow: "+JSON.stringify(result));
console.log(JSON.stringify(result,null,2));
