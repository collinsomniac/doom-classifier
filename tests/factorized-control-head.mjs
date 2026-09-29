import assert from "node:assert/strict";
import {FactorizedControlHead} from "../src/core/factorized-control-head.js";

const schema={
  fields:[
    {id:"bearing",decisionFeature:true,min:-1,max:1},
    {id:"visible",decisionFeature:true,min:0,max:1},
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
  [{bearing:.7,visible:1,distance_zone:1},one("left")],
  [{bearing:-.7,visible:1,distance_zone:1},one("right")],
  [{bearing:0,visible:1,distance_zone:1},one("fire")],
  [{bearing:0,visible:0,distance_zone:0},one("wait")]
];
for(let epoch=0;epoch<220;epoch++)for(const [obs,target] of states)head.supervise(obs,target,{strength:1});

const score=obs=>head.evaluate(obs);
const left=score(states[0][0]),right=score(states[1][0]),ahead=score(states[2][0]),quiet=score(states[3][0]);
const index=id=>actions.findIndex(a=>a.id===id);
assert.ok(left.scores[index("left")] > left.scores[index("right")]+1,"left bearing should prefer left control");
assert.ok(right.scores[index("right")] > right.scores[index("left")]+1,"right bearing should prefer right control");
assert.ok(ahead.scores[index("fire")] > ahead.scores[index("wait")]+1,"aligned visible target should prefer fire");
assert.ok(quiet.scores[index("wait")] > quiet.scores[index("fire")]+1,"no target should prefer hold");
assert.ok(head.parameterCount()<500,"factor head should stay tiny");

const saved=head.exportCheckpoint(),restored=new FactorizedControlHead(schema,actions);
assert.equal(restored.importCheckpoint(saved),true);
assert.deepEqual(restored.evaluate(states[0][0]).scores.map(x=>Number(x.toFixed(8))),left.scores.map(x=>Number(x.toFixed(8))));

console.log("factorized control head ok",{params:head.parameterCount(),updates:head.updates});
