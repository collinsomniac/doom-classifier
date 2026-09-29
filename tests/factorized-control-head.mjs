import assert from "node:assert/strict";
import {FactorizedControlHead} from "../src/core/factorized-control-head.js";

const schema={
  fields:[
    {id:"bearing",decisionFeature:true,min:-1,max:1},
    {id:"visible",decisionFeature:true,min:0,max:1},
    {id:"bearing_zone",decisionFeature:true,enum:{0:"none",1:"center",2:"left",3:"right"}},
    {id:"distance_zone",decisionFeature:true,enum:{0:"none",1:"near",2:"far"}}
  ],
  actionFields:[
    {id:"view",enum:{0:"none",1:"left",2:"right"}},
    {id:"trigger",enum:{0:"hold",1:"fire"}}
  ]
};
const actions=[
  {id:"wait",params:{view:0,trigger:0}},
  {id:"left",params:{view:1,trigger:0}},
  {id:"right",params:{view:2,trigger:0}},
  {id:"fire",params:{view:0,trigger:1}},
  {id:"left_fire",params:{view:1,trigger:1}},
  {id:"right_fire",params:{view:2,trigger:1}}
];
const head=new FactorizedControlHead(schema,actions,{lr:.12});
const one=id=>actions.map(a=>a.id===id?1:0);
const states=[
  [{bearing:.7,visible:1,bearing_zone:2,distance_zone:1},one("left")],
  [{bearing:-.7,visible:1,bearing_zone:3,distance_zone:1},one("right")],
  [{bearing:0,visible:1,bearing_zone:1,distance_zone:1},one("fire")],
  [{bearing:0,visible:0,bearing_zone:0,distance_zone:0},one("wait")]
];
for(let epoch=0;epoch<220;epoch++)for(const [obs,target] of states)head.supervise(obs,target,{strength:1});

const score=obs=>head.evaluate(obs);
const left=score(states[0][0]),right=score(states[1][0]),ahead=score(states[2][0]),quiet=score(states[3][0]);
const axis=(result,id)=>result.axis.find(x=>x.id===id);
const probability=(result,id,value)=>{
  const a=axis(result,id),i=a.values.indexOf(String(value));return a.probs[i];
};
assert.ok(probability(left,"view",1)>.8,"left bearing should strongly prefer left view");
assert.ok(probability(right,"view",2)>.8,"right bearing should strongly prefer right view");
assert.ok(probability(ahead,"trigger",1)>.8,"aligned visible target should strongly prefer fire");
assert.ok(probability(quiet,"trigger",0)>.8,"no target should strongly prefer hold");
assert.ok(head.parameterCount()<500,"factor head should stay tiny");

const saved=head.exportCheckpoint(),restored=new FactorizedControlHead(schema,actions);
assert.equal(restored.importCheckpoint(saved),true);
assert.deepEqual(restored.evaluate(states[0][0]).scores.map(x=>Number(x.toFixed(8))),left.scores.map(x=>Number(x.toFixed(8))));

console.log("factorized control head ok",{params:head.parameterCount(),updates:head.updates});
