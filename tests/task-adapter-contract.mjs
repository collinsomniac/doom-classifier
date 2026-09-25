import assert from "node:assert/strict";
import {HashSemanticAdapter} from "../src/core/semantic.js";
import {SemanticResidualPolicy} from "../src/core/policy.js";
import {ExperimentController} from "../src/core/controller.js";

class WorkloadRouterArena{
  constructor(){
    this.actions=[
      {id:"dispatch_low_latency",label:"dispatch low latency",description:"send the next job to a fast worker",params:{speed:1,conserve:0,defer:0}},
      {id:"dispatch_efficient",label:"dispatch energy efficient",description:"send the next job to an energy-efficient worker",params:{speed:.35,conserve:1,defer:0}},
      {id:"defer_batch",label:"defer batch",description:"hold the job briefly for later batching",params:{speed:0,conserve:1,defer:1}}
    ];
    this.schema={
      objective:"Keep request latency low while preserving enough energy to continue serving future work.",
      actionFields:[
        {id:"speed",label:"dispatch urgency",description:"relative preference for immediate low-latency service",min:0,max:1},
        {id:"conserve",label:"resource conservation",description:"whether the action prioritizes conserving energy",min:0,max:1},
        {id:"defer",label:"defer work",description:"whether the action intentionally delays a request",min:0,max:1}
      ],
      fields:[
        {id:"queue_pressure",label:"queue pressure",description:"fraction of request queue currently occupied",min:0,max:1},
        {id:"energy_budget",label:"energy budget",description:"remaining normalized compute energy",min:0,max:1},
        {id:"deadline_pressure",label:"deadline pressure",description:"urgency of the oldest queued request",min:0,max:1}
      ],
      collections:[{
        id:"workers",label:"available workers",description:"variable compute workers that may serve work",
        fields:[
          {id:"latency",label:"worker latency",description:"normalized service latency",min:0,max:1},
          {id:"efficiency",label:"worker efficiency",description:"normalized energy efficiency",min:0,max:1},
          {id:"capacity",label:"worker capacity",description:"available normalized capacity",min:0,max:1}
        ]
      }]
    };
    this.reset();
  }
  reset(){
    this.t=0;this.queue=.62;this.energy=.78;this.deadline=.35;this.lastObservation=this.observe();this.lastOutcome=null;return this.lastObservation;
  }
  observe(){
    return{queue_pressure:this.queue,energy_budget:this.energy,deadline_pressure:this.deadline,_collections:{workers:[
      {latency:.18,efficiency:.42,capacity:.72},
      {latency:.55,efficiency:.92,capacity:.88},
      {latency:.34,efficiency:.68,capacity:.52}
    ]}};
  }
  step(actionId){
    this.t++;
    let reward=-.01;
    if(actionId==="dispatch_low_latency"){this.queue=Math.max(0,this.queue-.18);this.energy=Math.max(0,this.energy-.09);reward+=this.deadline>.55?.38:.16}
    else if(actionId==="dispatch_efficient"){this.queue=Math.max(0,this.queue-.1);this.energy=Math.max(0,this.energy-.025);reward+=this.energy<.42?.32:.18}
    else{this.queue=Math.min(1,this.queue+.07);this.energy=Math.min(1,this.energy+.015);reward+=this.queue<.35?.08:-.08}
    this.deadline=Math.min(1,Math.max(0,this.deadline+this.queue*.09-.035));
    if(this.queue>.92)reward-=.45;if(this.energy<.08)reward-=.35;
    const observation=this.observe(),done=this.t>=96;
    this.lastObservation=observation;this.lastOutcome={reward,queue:this.queue,energy:this.energy,deadline:this.deadline};
    return{observation,reward,done,info:{outcome:this.lastOutcome}};
  }
}

const env=new WorkloadRouterArena(),semantic=new HashSemanticAdapter();
const rawScore=semantic.score.bind(semantic);semantic.score=async obs=>rawScore(obs);
const policy=new SemanticResidualPolicy({schema:env.schema,actions:env.actions,semantic,residual:"neural-set",seed:404,inferenceMode:"neural"});
const controller=new ExperimentController({environment:env,policy,hz:20});
controller.training=true;controller.explore=true;controller.memory=true;controller.useResidual=true;policy.epsilon=.2;

assert.ok(env.schema.fields.every(f=>!["health","ammo","weapon","kills"].includes(f.id)));
assert.ok(env.actions.every(a=>!["fire","forward","back","use"].includes(a.id)));
const params=policy.q.parameterCount();
for(let i=0;i<72;i++)assert.equal(await controller.tick(),true);
assert.equal(controller.steps,72);
assert.ok(policy.q.updates>0);
assert.ok(policy.replay.length>0);
assert.equal(policy.actions.length,3);
assert.equal(policy.q.parameterCount(),params);

controller.training=false;controller.explore=false;policy.setInferenceMode("neural");
const teacherBefore=policy.teacherCalls;
for(let i=0;i<12;i++)assert.equal(await controller.tick(),true);
assert.equal(policy.teacherCalls,teacherBefore,"frozen generic task inference must not require a teacher");
assert.ok(controller.lastDecision?.probs.every(Number.isFinite));
assert.ok(Math.abs(controller.lastDecision.probs.reduce((a,b)=>a+b,0)-1)<1e-5);

console.log(JSON.stringify({ok:true,task:"workload-routing",params,updates:policy.q.updates,replay:policy.replay.length,lastAction:controller.lastDecision.action.id,teacherCalls:policy.teacherCalls}));
