import assert from "node:assert/strict";
import {compileActionFieldProjector,projectValueToActionFields} from "../src/core/action-factorization.js";

const schema={actionFields:[
  {id:"forward",label:"forward",min:0,max:1},
  {id:"fire",label:"fire",min:0,max:1}
]};
const actions=[
  {id:"wait",params:{forward:0,fire:0}},
  {id:"forward",params:{forward:1,fire:0}},
  {id:"fire",params:{forward:0,fire:1}},
  {id:"forward_fire",params:{forward:1,fire:1}}
];

const noisy=[0,.5,1,.55];
const projected=projectValueToActionFields(schema,actions,noisy,{ridge:.01,maxBlend:1});
assert.ok(projected.fitQuality>.45,"typed fields should explain a useful fraction of additive value structure");
const idx=Object.fromEntries(actions.map((a,i)=>[a.id,i]));
assert.ok(projected.projected[idx.forward_fire]>projected.projected[idx.fire],"field projection should recover positive forward contribution inside the compound action");
assert.ok(projected.projected[idx.fire]>projected.projected[idx.forward],"fire contribution should remain stronger than forward");
assert.ok(projected.coefficients.find(x=>x.id==="forward").weight>0);
assert.ok(projected.coefficients.find(x=>x.id==="fire").weight>0);

const compiled=compileActionFieldProjector(schema,actions,{ridge:.01,maxBlend:1}),compiledResult=compiled.project(noisy);
assert.equal(compiled.basisSize,3);
assert.ok(compiledResult.scores.every((v,i)=>Math.abs(v-projected.scores[i])<1e-10),"compiled projector must match one-shot projection");
assert.ok(compiledResult.projected.every((v,i)=>Math.abs(v-projected.projected[i])<1e-10));

const enumSchema={actionFields:[{id:"mode",label:"mode",enum:{0:"slow",1:"fast"}}]};
const enumActions=[{id:"slow",params:{mode:0}},{id:"fast",params:{mode:1}}];
const enumFit=projectValueToActionFields(enumSchema,enumActions,[-.2,.8],{ridge:.001,maxBlend:1});
assert.ok(enumFit.projected[1]>enumFit.projected[0],"categorical typed fields must participate in the projection");

const flat=projectValueToActionFields(schema,actions,[1,1,1,1]);
assert.equal(flat.fitQuality,0);
assert.ok(flat.scores.every(v=>Math.abs(v)<1e-12));

console.log(JSON.stringify({ok:true,fitQuality:projected.fitQuality,blendUsed:projected.blendUsed,coefficients:projected.coefficients,projected:projected.projected,enumProjected:enumFit.projected}));
