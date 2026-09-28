import assert from "node:assert/strict";
import {SemanticResidualPolicy} from "../src/core/policy.js";
import {HashSemanticAdapter} from "../src/core/semantic.js";
import {ExperimentController} from "../src/core/controller.js";
import {measureCounterfactualValues,trainWithMeasuredForks} from "../src/core/measured-fork-training.js";

class ForkArena{
  constructor(){
    this.actions=[
      {id:"advance",label:"advance",description:"advance toward useful state",params:{advance:1}},
      {id:"retreat",label:"retreat",description:"move away from useful state",params:{advance:0}},
      {id:"wait",label:"wait",description:"apply no movement",params:{advance:0}}
    ];
    this.schema={
      environment:"small deterministic routing arena",objective:"increase progress while preserving energy",
      actionFields:[{id:"advance",label:"advance control",description:"whether progress control is active",min:0,max:1}],
      fields:[
        {id:"progress",label:"progress",description:"current normalized progress",min:0,max:1},
        {id:"energy",label:"energy",description:"remaining normalized energy",min:0,max:1}
      ],collections:[]
    };
    this.state={progress:.2,energy:.8};this.lastObservation=this.observe();
  }
  observe(){this.lastObservation={...this.state,_collections:{}};return this.lastObservation}
  async reset(){this.state={progress:.2,energy:.8};return this.observe()}
  saveSnapshot(){return{state:{...this.state}}}
  restoreSnapshot(token){this.state={...token.state};return this.observe()}
  stepTics(id,_tics=1){return this.step(id)}
  async step(id){
    let reward=-.01;
    if(id==="advance"){this.state.progress=Math.min(1,this.state.progress+.18);this.state.energy-=.03;reward=.3}
    else if(id==="retreat"){this.state.progress=Math.max(0,this.state.progress-.12);this.state.energy-=.01;reward=-.18}
    else{this.state.energy=Math.min(1,this.state.energy+.005);reward=-.03}
    return{observation:this.observe(),reward,done:false,info:{outcome:{}}};
  }
}

const env=new ForkArena(),policy=new SemanticResidualPolicy({schema:env.schema,actions:env.actions,semantic:new HashSemanticAdapter(),residual:"neural-set",seed:818,inferenceMode:"neural"});
const controller=new ExperimentController({environment:env,policy,hz:20});controller.training=false;controller.explore=false;controller.memory=true;controller.useResidual=true;

const measured=await measureCounterfactualValues({environment:env,policy,tics:3,fallbackTics:5,fit:true,fitSteps:8,fitStrength:.2});
assert.equal(env.actions[measured.probe.returns.indexOf(Math.max(...measured.probe.returns))].id,"advance");
assert.ok(measured.probe.spread>0);assert.ok(measured.example);assert.ok(measured.fitResult?.trustUpdates===1);
assert.deepEqual(env.state,{progress:.2,energy:.8},"counterfactual measurement must restore exact arena state");

const startUpdates=policy.q.updates,probes=[];
const result=await trainWithMeasuredForks({
  controller,policy,environment:env,steps:8,stageSize:4,epsilon:.2,tics:3,fallbackTics:5,fitSteps:4,fitStrength:.15,resetEvery:0,bootstrapProbe:true,
  onProbe:x=>probes.push({phase:x.phase,spread:x.probe.spread,fit:!!x.fitResult})
});
assert.equal(result.completed,8);assert.equal(result.measuredProbes,3);assert.ok(result.measuredFits>=2);
assert.ok(policy.q.updates>startUpdates);assert.ok(probes.some(x=>x.phase==="bootstrap"));assert.ok(probes.filter(x=>x.phase==="post-stage").length===2);
console.log(JSON.stringify({ok:true,params:policy.q.parameterCount(),result:{completed:result.completed,measuredFits:result.measuredFits,measuredProbes:result.measuredProbes,meanMeasuredSpread:result.meanMeasuredSpread,actionDiversity:result.actionDiversity},probes}));
