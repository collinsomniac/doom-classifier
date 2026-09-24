import assert from "node:assert/strict";
import {TransformersNLIAdapter} from "../src/model-adapters/transformers-nli.js";

const schema={objective:"choose a justified action",fields:[],collections:[]};
const actions=[
  {id:"strong",label:"strong",description:"strongly supported action"},
  {id:"weak",label:"weak",description:"weakly supported action"},
  {id:"neutral",label:"neutral",description:"neutral action"}
];
const adapter=new TransformersNLIAdapter();
adapter.compile(schema,actions);
adapter.classifier=async(_text,labels,options)=>{
  assert.equal(options.multi_label,true,"teacher must request independent NLI scoring");
  return{labels:[labels[1],labels[2],labels[0]],scores:[.1,.5,.9]};
};
const scores=await adapter.score({_collections:{}});
const expected=[Math.log(.9/.1),Math.log(.1/.9),0];
for(let i=0;i<scores.length;i++)assert.ok(Math.abs(scores[i]-expected[i])<1e-9);
assert.ok(scores[0]>scores[2]&&scores[2]>scores[1]);
console.log(JSON.stringify({ok:true,scores}));
