import assert from "node:assert/strict";
import {TransformersNLIAdapter} from "../src/model-adapters/transformers-nli.js";

const schema={
  objective:"select literal controls",
  fields:[],collections:[],
  actionFields:[
    {id:"left",label:"turn left control",description:"rotate left",min:0,max:1},
    {id:"fire",label:"weapon fire control",description:"fire weapon",min:0,max:1}
  ]
};
const actions=[
  {id:"wait",label:"wait",description:"no controls",params:{left:0,fire:0}},
  {id:"left",label:"left",description:"turn left",params:{left:1,fire:0}},
  {id:"fire",label:"fire",description:"fire",params:{left:0,fire:1}},
  {id:"left_fire",label:"left + fire",description:"turn left and fire",params:{left:1,fire:1}}
];
const adapter=new TransformersNLIAdapter();adapter.compile(schema,actions);
assert.equal(adapter.labels.length,2,"teacher work should scale with typed primitive fields, not joint packets");
adapter.classifier=async(_text,labels,options)=>{
  assert.equal(options.multi_label,true);
  assert.equal(labels.length,2);
  return{labels:[labels[0],labels[1]],scores:[.9,.2]};
};
const scores=await adapter.score({_collections:{}});
assert.equal(scores.length,4);
const top=scores.indexOf(Math.max(...scores));
assert.equal(actions[top].id,"left","high left / low fire evidence should compose to left without a joint-label query");
assert.ok(scores[3]<scores[1],"adding an independently disfavored fire bit should lower the joint packet score");
console.log(JSON.stringify({ok:true,teacherQuestions:adapter.labels.length,actions:actions.length,scores}));


const axisSchema={
  objective:"select one option on each typed control axis",
  fields:[],collections:[],
  actionFields:[
    {id:"forward",label:"forward",description:"move forward",axis:"translation",axisLabel:"translation",neutralLabel:"no translation",min:0,max:1},
    {id:"back",label:"backward",description:"move backward",axis:"translation",axisLabel:"translation",neutralLabel:"no translation",min:0,max:1},
    {id:"fire",label:"fire",description:"fire weapon",axis:"trigger",axisLabel:"trigger",neutralLabel:"do not fire",min:0,max:1}
  ]
};
const axisActions=[
  {id:"wait",params:{forward:0,back:0,fire:0}},
  {id:"forward",params:{forward:1,back:0,fire:0}},
  {id:"back",params:{forward:0,back:1,fire:0}},
  {id:"fire",params:{forward:0,back:0,fire:1}},
  {id:"forward_fire",params:{forward:1,back:0,fire:1}},
  {id:"back_fire",params:{forward:0,back:1,fire:1}}
];
const axisAdapter=new TransformersNLIAdapter();axisAdapter.compile(axisSchema,axisActions);
assert.equal(axisAdapter.labels.length,5,"translation neutral/forward/back + trigger neutral/fire should require five semantic questions");
axisAdapter.classifier=async(_text,labels)=>({
  labels,
  // translation: neutral .15, forward .85, back .08; trigger: neutral .75, fire .25
  scores:[.15,.85,.08,.75,.25]
});
const axisScores=await axisAdapter.score({_collections:{}});
assert.equal(axisScores.length,axisActions.length);
assert.equal(axisActions[axisScores.indexOf(Math.max(...axisScores))].id,"forward","axis-wise normalization should compose forward + neutral trigger");
assert.ok(axisScores[axisActions.findIndex(a=>a.id==="forward")]>axisScores[axisActions.findIndex(a=>a.id==="back")]);
assert.ok(axisScores[axisActions.findIndex(a=>a.id==="wait")]>axisScores[axisActions.findIndex(a=>a.id==="fire")],"explicit neutral trigger evidence must suppress gratuitous firing");
console.log(JSON.stringify({ok:true,axisTeacherQuestions:axisAdapter.labels.length,axisScores}));
