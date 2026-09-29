// Decompose the deployed neural-mode logits of the CI checkpoint on mirrored probe states.
import {readFileSync} from "node:fs";
import {SemanticResidualPolicy} from "../src/core/policy.js";
import {DoomWasmArena} from "../src/env/doom-wasm.js";
const path=process.argv[2]||"/root/art/doom-causal-checkpoint.json";
const ck=JSON.parse(readFileSync(path,"utf8")),arena=new DoomWasmArena({});
const semantic={name:"stub",backend:"test",compile(){},async score(){return new Array(arena.actions.length).fill(0)}};
const p=new SemanticResidualPolicy({schema:arena.schema,actions:arena.actions,semantic,residual:"neural-set",inferenceMode:"neural"});
try{p.importCheckpoint(ck)}catch(e){console.log("import error",e.message)}
const ids=arena.actions.map(a=>a.id);
const make=b=>{const has=b!==null,d=192,r=(b??0)*Math.PI;return{health:100,armor:0,bullets:50,shells:0,weapon:1,heading:0,
  visible_hostile_count:has?1:0,targeting_hostile_count:has?1:0,nearest_hostile_distance:has?d:4096,nearest_hostile_bearing:b??0,
  nearest_visible_hostile_distance:has?d:4096,nearest_visible_hostile_bearing:b??0,nearest_visible_hostile_health:has?60:0,
  visible_hostile_bearing_zone:!has?0:Math.abs(b)<=.06?1:b>0?2:3,visible_hostile_distance_zone:has?1:0,aim_alignment:has?1-Math.min(1,Math.abs(b)*8):0,
  nearest_targeting_hostile_distance:has?d:4096,nearest_targeting_hostile_bearing:b??0,visible_projectile_count:0,nearest_projectile_distance:4096,nearest_projectile_bearing:0,
  _collections:{entities:has?[{engine_record_id:999,type:11,kind:1,relative_x:d*Math.cos(r),relative_y:d*Math.sin(r),distance:d,relative_angle:b,visible:1,countkill:1,health:60,radius:20,height:56}]:[],geometry:[]}};};
const sd=v=>{const m=v.reduce((a,b)=>a+b,0)/v.length;return Math.sqrt(v.reduce((s,x)=>s+(x-m)**2,0)/v.length)};
const tv=(a,b)=>.5*a.reduce((s,x,i)=>s+Math.abs(x-b[i]),0);
const sm=(v,t)=>{const m=Math.max(...v),e=v.map(x=>Math.exp((x-m)/t)),z=e.reduce((a,b)=>a+b,0);return e.map(x=>x/z)};
const rows={};
for(const [n,b] of [["left",.24],["ahead",0],["right",-.24],["quiet",null]]){
  const o=make(b),enc=p.encode(o,false,false),ev=p.residualEvaluation(o,enc.features,enc.temporal,[]);
  rows[n]={final:ev.scores,factor:ev.factorScores||[],sem:ev.rawScores?ev.semanticScores.map((v,i)=>v-(ev.factorScores?.[i]||0)):[],value:ev.valueScores||[]};
  const pr=sm(ev.scores,p.temperature),top=ids.map((id,i)=>[id,pr[i]]).sort((a,b)=>b[1]-a[1]).slice(0,3);
  console.log(n.padEnd(6),"sd final",sd(ev.scores).toFixed(3),"factor",sd(rows[n].factor).toFixed(3),"sem",sd(rows[n].sem).toFixed(3),"value",sd(rows[n].value).toFixed(3),"beta",Number(ev.valueBeta||0).toFixed(3),"top",top.map(([i,v])=>i+":"+v.toFixed(2)).join(" "));
}
for(const k of ["final","factor","sem"])if(rows.left[k].length)console.log("TV left/right on",k.padEnd(6),tv(sm(rows.left[k],p.temperature),sm(rows.right[k],p.temperature)).toFixed(3));
