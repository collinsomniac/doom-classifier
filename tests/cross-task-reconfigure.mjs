import assert from "node:assert/strict";
import {HashSemanticAdapter} from "../src/core/semantic.js";
import {SemanticResidualPolicy} from "../src/core/policy.js";
import {ExperimentController} from "../src/core/controller.js";

class TinyTask{
  constructor(schema,actions,stepFn){this.schema=schema;this.actions=actions;this.stepFn=stepFn;this.reset()}
  reset(){this.t=0;this.state={...this.schema.initial,_collections:structuredClone(this.schema.initialCollections||{})};this.lastObservation=this.observe();this.lastOutcome=null;return this.lastObservation}
  observe(){return structuredClone(this.state)}
  step(actionId){
    const result=this.stepFn(this.state,actionId,this.t++),done=this.t>=64;
    this.state=result.state;this.lastObservation=this.observe();this.lastOutcome={reward:result.reward};
    return{observation:this.lastObservation,reward:result.reward,done,info:{outcome:this.lastOutcome}};
  }
}

const routingSchema={
  objective:"Keep queued work moving without exhausting the available compute reserve.",
  actionFields:[
    {id:"urgency",label:"urgency",description:"how aggressively to service work",min:0,max:1},
    {id:"conserve",label:"conserve",description:"whether to conserve compute reserve",min:0,max:1}
  ],
  fields:[
    {id:"queue",label:"queue pressure",description:"fraction of work waiting",min:0,max:1},
    {id:"reserve",label:"compute reserve",description:"remaining compute reserve",min:0,max:1}
  ],
  collections:[{id:"workers",label:"workers",description:"available compute workers",fields:[
    {id:"latency",label:"latency",description:"relative service latency",min:0,max:1},
    {id:"capacity",label:"capacity",description:"available capacity",min:0,max:1}
  ]}],
  initial:{queue:.7,reserve:.8},initialCollections:{workers:[{latency:.2,capacity:.7},{latency:.6,capacity:.95}]}
};
const routingActions=[
  {id:"fast",label:"fast dispatch",description:"dispatch immediately",params:{urgency:1,conserve:0}},
  {id:"efficient",label:"efficient dispatch",description:"use efficient service",params:{urgency:.4,conserve:1}},
  {id:"defer",label:"defer",description:"briefly defer service",params:{urgency:0,conserve:1}}
];
const routing=new TinyTask(routingSchema,routingActions,(s,a)=>{
  let queue=s.queue,reserve=s.reserve,reward=-.01;
  if(a==="fast"){queue=Math.max(0,queue-.2);reserve=Math.max(0,reserve-.08);reward+=queue>.55?.3:.12}
  else if(a==="efficient"){queue=Math.max(0,queue-.11);reserve=Math.max(0,reserve-.025);reward+=.16}
  else{queue=Math.min(1,queue+.06);reserve=Math.min(1,reserve+.01);reward+=queue<.25?.06:-.08}
  return{state:{...s,queue,reserve},reward};
});

const climateSchema={
  objective:"Keep plant zones near healthy temperature and moisture while limiting resource use.",
  actionFields:[
    {id:"water",label:"water",description:"irrigation intensity",min:0,max:1},
    {id:"vent",label:"ventilation",description:"ventilation intensity",min:0,max:1},
    {id:"heat",label:"heating",description:"heating intensity",min:0,max:1}
  ],
  fields:[
    {id:"temperature",label:"temperature",description:"normalized greenhouse temperature",min:0,max:1},
    {id:"moisture",label:"moisture",description:"normalized soil moisture",min:0,max:1},
    {id:"energy",label:"energy reserve",description:"remaining actuator energy",min:0,max:1}
  ],
  collections:[{id:"zones",label:"plant zones",description:"individual growing zones",fields:[
    {id:"stress",label:"plant stress",description:"normalized plant stress",min:0,max:1},
    {id:"dryness",label:"dryness",description:"normalized local dryness",min:0,max:1}
  ]}],
  initial:{temperature:.72,moisture:.28,energy:.9},initialCollections:{zones:[{stress:.35,dryness:.7},{stress:.2,dryness:.55},{stress:.4,dryness:.8}]}
};
const climateActions=[
  {id:"water",label:"irrigate",description:"irrigate the zones",params:{water:1,vent:0,heat:0}},
  {id:"vent",label:"ventilate",description:"ventilate the greenhouse",params:{water:0,vent:1,heat:0}},
  {id:"heat",label:"heat",description:"raise greenhouse temperature",params:{water:0,vent:0,heat:1}},
  {id:"water_vent",label:"irrigate + ventilate",description:"irrigate while ventilating",params:{water:1,vent:1,heat:0}},
  {id:"hold",label:"hold",description:"apply no actuator input",params:{water:0,vent:0,heat:0}}
];
const climate=new TinyTask(climateSchema,climateActions,(s,a)=>{
  let temperature=s.temperature,moisture=s.moisture,energy=s.energy;
  const spec=climateActions.find(x=>x.id===a)?.params||{};
  moisture=Math.max(0,Math.min(1,moisture+.12*(spec.water||0)-.018));
  temperature=Math.max(0,Math.min(1,temperature+.08*(spec.heat||0)-.07*(spec.vent||0)+.008));
  energy=Math.max(0,energy-.025*((spec.water||0)+(spec.vent||0)+(spec.heat||0)));
  const stress=Math.abs(temperature-.55)+Math.abs(moisture-.58),reward=.25-stress-.05*(1-energy);
  return{state:{...s,temperature,moisture,energy},reward};
});

const semantic=new HashSemanticAdapter(),policy=new SemanticResidualPolicy({schema:routing.schema,actions:routing.actions,semantic,residual:"neural-set",seed:733,inferenceMode:"neural"});
const controller=new ExperimentController({environment:routing,policy,hz:20});
controller.training=true;controller.explore=true;controller.memory=true;controller.useResidual=true;policy.epsilon=.18;
const params=policy.q.parameterCount();
for(let i=0;i<40;i++)assert.equal(await controller.tick(),true);
assert.ok(policy.q.updates>0);

controller.environment=climate;policy.reconfigure({schema:climate.schema,actions:climate.actions});await controller.reset({learning:false});
assert.equal(policy.q.parameterCount(),params,"schema/action reconfiguration must not resize the neural core");
assert.equal(policy.actions.length,5);
controller.training=true;controller.explore=true;policy.epsilon=.18;
for(let i=0;i<48;i++)assert.equal(await controller.tick(),true);
assert.ok(policy.q.updates>0);
controller.training=false;controller.explore=false;policy.setInferenceMode("neural");
const teacherBefore=policy.teacherCalls;
for(let i=0;i<12;i++)assert.equal(await controller.tick(),true);
assert.equal(policy.teacherCalls,teacherBefore);
assert.ok(controller.lastDecision.probs.every(Number.isFinite));
assert.ok(Math.abs(controller.lastDecision.probs.reduce((a,b)=>a+b,0)-1)<1e-5);
assert.ok(!policy.schema.fields.some(f=>["health","ammo","weapon","kills"].includes(f.id)));

console.log(JSON.stringify({ok:true,from:"workload-routing",to:"greenhouse-control",params,actions:policy.actions.length,updates:policy.q.updates,lastAction:controller.lastDecision.action.id,teacherCalls:policy.teacherCalls}));
