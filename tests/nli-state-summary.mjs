import assert from "node:assert/strict";
import {TransformersNLIAdapter,summarizeCollection} from "../src/model-adapters/transformers-nli.js";

const schema={
  objective:"pick an appropriate typed action",
  fields:[{id:"energy",label:"energy",description:"remaining energy",min:0,max:100}],
  collections:[{
    id:"objects",label:"objects",description:"variable structured objects",
    fields:[
      {id:"distance",label:"distance",description:"distance from controller",scale:100},
      {id:"quality",label:"quality",description:"utility quality",min:0,max:1},
      {id:"active",label:"active",description:"currently active",min:0,max:1},
      {id:"kind",label:"object kind",description:"categorical object kind",enum:{1:"hostile actor",2:"projectile"}}
    ]
  }]
};
const actions=[{id:"use",label:"use",description:"use a suitable object"},{id:"wait",label:"wait",description:"do nothing"}];
const obs={energy:55,_collections:{objects:[
  {distance:5,quality:.9,active:1,kind:1},{distance:80,quality:.2,active:1,kind:2},{distance:40,quality:.5,active:1,kind:1}
]}};
const summary=summarizeCollection(schema.collections[0],obs._collections.objects);
assert.match(summary,/3 records/);
assert.match(summary,/distance/);
assert.match(summary,/active active=3\/3 \(100%\)/,"constant active binary flag must survive variance-based compression");
assert.match(summary,/object kind categories=hostile actor:2, projectile:1/,"categorical collection values must be summarized by semantic name");
assert.match(summary,/object kind:(hostile actor|projectile)/,"representative records must render categorical names");
assert.match(summary,/representative 1/);
const adapter=new TransformersNLIAdapter({maxStateChars:2000});
adapter.compile(schema,actions);
const text=adapter.stateText(obs);
const parsed=JSON.parse(text);assert.equal(parsed.projection.bounded,true);assert.equal(parsed.observation.collections.objects.count,3);assert.match(parsed.observation.collections.objects.summary,/3 records/);assert.equal(adapter.fullStateObject(obs).observation.collections.objects.length,3);assert.ok(text.length<=2000);
console.log(JSON.stringify({ok:true,length:text.length,summary}));
