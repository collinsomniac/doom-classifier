import assert from "node:assert/strict";
import {NeuralSetResidualQ} from "../src/core/neural-set-residual.js";

const schema={
  fields:[{id:"urgency",label:"urgency",description:"current request urgency",min:0,max:1}],
  collections:[],
  actionFields:[
    {id:"x",label:"horizontal target",description:"normalized horizontal target coordinate",min:0,max:1},
    {id:"y",label:"vertical target",description:"normalized vertical target coordinate",min:0,max:1},
    {id:"strength",label:"command strength",description:"normalized command magnitude",min:0,max:1}
  ]
};
const actions=[
  {id:"p1",label:"point",description:"act at the proposed coordinate",params:{x:.1,y:.2,strength:.4}},
  {id:"p2",label:"point",description:"act at the proposed coordinate",params:{x:.8,y:.7,strength:.9}}
];
const obs={urgency:.6,_collections:{}};
const net=new NeuralSetResidualQ(schema,actions,{seed:19});
const params=net.parameterCount(),scores=net.scoresObservation(obs);
assert.notDeepEqual(Array.from(net.actionEmbeddings[0]),Array.from(net.actionEmbeddings[1]),"numeric action parameters must affect action representation");
assert.ok(Math.abs(scores[0]-scores[1])>1e-8,"parameterized candidates should be independently scoreable");

const schemaScaled={...schema,actionFields:[
  {id:"x2",label:"horizontal target",description:"normalized horizontal target coordinate",min:0,max:100},
  {id:"y2",label:"vertical target",description:"normalized vertical target coordinate",min:0,max:100},
  {id:"power",label:"command strength",description:"normalized command magnitude",min:0,max:10}
]};
const scaled=[
  {id:"q2",label:"point",description:"act at the proposed coordinate",params:{x2:80,y2:70,power:9}},
  {id:"q1",label:"point",description:"act at the proposed coordinate",params:{x2:10,y2:20,power:4}}
];
net.setSchema(schemaScaled).setActions(scaled);
const scaledScores=net.scoresObservation(obs);
assert.equal(net.parameterCount(),params);
assert.ok(Math.abs(scaledScores[0]-scores[1])<1e-5);
assert.ok(Math.abs(scaledScores[1]-scores[0])<1e-5);
console.log(JSON.stringify({ok:true,params,scores,scaledScores}));
