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
