import assert from "node:assert/strict";
import {NeuralSetResidualQ} from "../src/core/neural-set-residual.js";

const fieldsA=[
  {id:"energy",label:"energy reserve",description:"remaining operating energy",min:0,max:100},
  {id:"load",label:"system load",description:"current fractional workload",min:0,max:1}
];
const collectionFieldsA=[
  {id:"distance",label:"distance",description:"distance from controller",min:0,max:100},
  {id:"quality",label:"quality",description:"candidate utility quality",min:0,max:1}
];
const schemaA={fields:fieldsA,collections:[{id:"items",label:"candidate objects",description:"variable nearby candidates",fields:collectionFieldsA}]};
const actionsA=[
  {id:"approach",label:"approach",description:"move toward a useful candidate"},
  {id:"hold",label:"hold",description:"preserve current position"},
  {id:"inspect",label:"inspect",description:"gather more information about a candidate"}
];
const obsA={energy:60,load:.35,_collections:{items:[{distance:20,quality:.9},{distance:80,quality:.2}]}};

const net=new NeuralSetResidualQ(schemaA,actionsA,{seed:91});
const params=net.parameterCount(),scoresA=net.scoresObservation(obsA);

const schemaB={
  fields:[
    {id:"work_fraction",label:"system load",description:"current fractional workload",min:0,max:100},
    {id:"battery_units",label:"energy reserve",description:"remaining operating energy",min:0,max:1000}
  ],
  collections:[{
    id:"objects_v2",label:"candidate objects",description:"variable nearby candidates",
    fields:[
      {id:"utility",label:"quality",description:"candidate utility quality",min:0,max:100},
      {id:"range_cm",label:"distance",description:"distance from controller",min:0,max:1000}
    ]
  }]
};
const actionsB=[
  {id:"new_inspect_id",label:"inspect",description:"gather more information about a candidate"},
  {id:"new_approach_id",label:"approach",description:"move toward a useful candidate"},
  {id:"new_hold_id",label:"hold",description:"preserve current position"}
];
const obsB={work_fraction:35,battery_units:600,_collections:{objects_v2:[{utility:20,range_cm:800},{utility:90,range_cm:200}]}};

net.setSchema(schemaB).setActions(actionsB);
const scoresB=net.scoresObservation(obsB);
assert.equal(net.parameterCount(),params,"schema/action recompilation must not resize trainable network");
const byLabelA=Object.fromEntries(actionsA.map((a,i)=>[a.label,scoresA[i]]));
for(let i=0;i<actionsB.length;i++)assert.ok(Math.abs(scoresB[i]-byLabelA[actionsB[i].label])<1e-5,"semantic-equivalent relabel/reorder/scale should preserve score for "+actionsB[i].label);

console.log(JSON.stringify({ok:true,params,scoresA,scoresB}));
