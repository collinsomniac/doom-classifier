import assert from "node:assert/strict";
import {counterfactualTarget,probeCounterfactualActions} from "../src/core/counterfactual.js";

const equal=counterfactualTarget([1,1,1],[.6,.3,.1]);
assert.deepEqual(equal.target.map(x=>Number(x.toFixed(10))),[.6,.3,.1],"equal returns must preserve the prior");

const shifted=counterfactualTarget([.1,.7,.2],[.6,.2,.2],{temperature:.5});
assert.ok(shifted.target[1]>.2,"measured advantage should overcome some prior mass");
assert.equal(shifted.target.indexOf(Math.max(...shifted.target)),1);

class ForkableRoutingArena{
  constructor(){
    this.actions=[
      {id:"fast",label:"fast",params:{urgency:1}},
      {id:"efficient",label:"efficient",params:{urgency:.4}},
      {id:"defer",label:"defer",params:{urgency:0}}
    ];
    this.state={queue:.8,energy:.55,t:0};this.lastObservation=this.observe();
  }
  observe(){return{queue:this.state.queue,energy:this.state.energy,t:this.state.t,_collections:{}}}
  saveSnapshot(){this.snapshot=structuredClone(this.state);return{state:structuredClone(this.state)}}
  restoreSnapshot(token){this.state=structuredClone(token.state);this.lastObservation=this.observe();return this.lastObservation}
  async step(id){
    this.state.t++;
    let reward=0;
    if(id==="fast"){this.state.queue-=.25;this.state.energy-=.12;reward=.35}
    else if(id==="efficient"){this.state.queue-=.14;this.state.energy-=.035;reward=.24}
    else{this.state.queue+=.06;this.state.energy+=.01;reward=-.08}
    this.lastObservation=this.observe();
    return{observation:this.lastObservation,reward,done:false,info:{outcome:{queue:this.state.queue,energy:this.state.energy}}};
  }
}

const env=new ForkableRoutingArena(),before=structuredClone(env.state),prior=[.2,.55,.25];
const probe=await probeCounterfactualActions({environment:env,prior,horizon:1,temperature:.45});
assert.deepEqual(env.state,before,"probing must restore the environment after branching");
assert.deepEqual(probe.returns.map(v=>Number(v.toFixed(2))),[.35,.24,-.08]);
assert.ok(Math.abs(probe.target.reduce((a,b)=>a+b,0)-1)<1e-9);
assert.ok(probe.target[0]>prior[0],"measured best branch must gain probability");
assert.ok(probe.target[2]<prior[2],"measured worst branch must lose probability");
assert.equal(probe.target.indexOf(Math.max(...probe.target)),1,"moderate update may preserve a strong incumbent prior");

const sharp=await probeCounterfactualActions({environment:env,prior,horizon:1,temperature:.2});
assert.equal(sharp.target.indexOf(Math.max(...sharp.target)),0,"low-temperature improvement should let clear measured advantage override the prior");
assert.ok(sharp.target[0]>probe.target[0]);

let deterministicCalls=0;
const deterministic=await probeCounterfactualActions({
  environment:env,prior,candidateIndices:[0,1],temperature:.5,
  stepper:async id=>{deterministicCalls++;return env.step(id)}
});
assert.equal(deterministicCalls,2);
assert.deepEqual(deterministic.returns.slice(0,2).map(v=>Number(v.toFixed(2))),[.35,.24]);

const subset=await probeCounterfactualActions({environment:env,prior,candidateIndices:[0,1],temperature:.6});
assert.equal(subset.target[2],0);
assert.ok(Math.abs(subset.target[0]+subset.target[1]-1)<1e-9);

console.log("COUNTERFACTUAL_CORE "+JSON.stringify({
  ok:true,returns:probe.returns,target:probe.target,sharpTarget:sharp.target,prior,spread:probe.spread,scale:probe.scale
}));
