import assert from "node:assert/strict";
import {NeuralSetResidualQ} from "../src/core/neural-set-residual.js";

const schema={fields:[{id:"energy",label:"energy",description:"remaining energy",min:0,max:1}],collections:[{id:"objects",label:"objects",description:"dynamic candidate objects",fields:[
  {id:"slot",label:"slot identity",description:"stable numeric record marker",min:0,max:10},
  {id:"distance",label:"distance",description:"distance from controller",min:0,max:100},
  {id:"utility",label:"utility",description:"usefulness of object",min:0,max:1},
  {id:"hostile",label:"hostile",description:"whether object is hostile",min:0,max:1}
]}]};
const actions=[{id:"engage",label:"engage",description:"interact aggressively with a hostile object"},{id:"collect",label:"collect",description:"acquire a useful object"}];
const obs={energy:.8,_collections:{objects:[
  {slot:1,distance:10,utility:.1,hostile:1},
  {slot:2,distance:20,utility:.95,hostile:0},
  {slot:3,distance:80,utility:.3,hostile:0}
]}};
const net=new NeuralSetResidualQ(schema,actions,{seed:33});
assert.ok(net.parameterCount()<7000);
for(let a=0;a<actions.length;a++){
  const top=net.inspectAttention(obs,a,{topK:3}),sum=top.reduce((s,x)=>s+x.weight,0);
  assert.ok(Math.abs(sum-1)<1e-6);
  assert.ok(top.every(x=>Number.isFinite(x.weight)&&x.weight>=0));
  const reversed={...obs,_collections:{objects:[...obs._collections.objects].reverse()}};
  const top2=net.inspectAttention(reversed,a,{topK:3});
  const bySlot=Object.fromEntries(top.map(x=>[x.record.slot,x.weight]));
  for(const item of top2)assert.ok(Math.abs(item.weight-bySlot[item.record.slot])<1e-6,"attention must be permutation invariant");
}
const lowEnergy={...obs,energy:.05};
const highAttention=net.inspectAttention(obs,0,{topK:3}),lowAttention=net.inspectAttention(lowEnergy,0,{topK:3});
const highBySlot=Object.fromEntries(highAttention.map(x=>[x.record.slot,x.weight]));
const stateAttentionDiff=Math.max(...lowAttention.map(x=>Math.abs(x.weight-highBySlot[x.record.slot])));
assert.ok(stateAttentionDiff>1e-8,"attention for the same action must be able to change with global state");

const before=net.scoresObservation(obs);
for(let i=0;i<80;i++)net.distill(obs,[4,-3],{strength:.6});
const after=net.scoresObservation(obs);
assert.ok(after[0]-after[1]>before[0]-before[1],"teacher distillation should move action margin");
console.log(JSON.stringify({ok:true,params:net.parameterCount(),stateAttentionDiff,before,after,attention:actions.map((_,i)=>net.inspectAttention(obs,i,{topK:3}).map(x=>({slot:x.record.slot,weight:x.weight})))}));
