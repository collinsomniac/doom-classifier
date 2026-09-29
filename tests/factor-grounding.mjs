// Unit-level relational grounding test (runs before any DOOM training).
//
// Trains ONLY through the production supervision entry point
// (SemanticResidualPolicy.superviseDecisionDistribution) on the real DOOM
// schema and 15-action interface, then measures the deployed neural-mode
// distribution produced by residualEvaluation. Labels are synthetic by design:
// this checks whether the architecture can REPRESENT and DEPLOY a known
// relational mapping, not whether DOOM produces it.
//
// Label design mirrors real causal labels: mass is split across several
// composites that share one semantic factor (e.g. "turn left" and
// "turn left + fire"), and "wait" is never labelled.
import assert from "node:assert/strict";
import {SemanticResidualPolicy} from "../src/core/policy.js";
import {FactorizedControlHead} from "../src/core/factorized-control-head.js";
import {DoomWasmArena} from "../src/env/doom-wasm.js";
import {softmax} from "../src/core/math.js";

const arena=new DoomWasmArena({}),schema=arena.schema,actions=arena.actions,ids=actions.map(a=>a.id);
const semantic={name:"stub",backend:"test",compile(){},async score(){return new Array(actions.length).fill(0)}};
// Uses the production neural-set residual: the test covers the whole deployed fusion.
const residual="neural-set"; // the production residual

function state(bearing,{distance=192,health=100}={}){
  const has=bearing!==null,b=has?bearing:0,r=b*Math.PI;
  return{health,armor:0,bullets:50,shells:0,weapon:1,heading:0,recent_damage:0,under_fire:0,
    visible_hostile_count:has?1:0,targeting_hostile_count:has?1:0,
    nearest_hostile_distance:has?distance:4096,nearest_hostile_bearing:b,
    nearest_visible_hostile_distance:has?distance:4096,nearest_visible_hostile_bearing:b,nearest_visible_hostile_health:has?60:0,
    visible_hostile_bearing_zone:!has?0:Math.abs(b)<=.06?1:b>0?2:3,visible_hostile_distance_zone:!has?0:distance<256?1:distance<768?2:3,
    aim_alignment:has?1-Math.min(1,Math.abs(b)*8):0,nearest_targeting_hostile_distance:has?distance:4096,nearest_targeting_hostile_bearing:b,
    visible_projectile_count:0,nearest_projectile_distance:4096,nearest_projectile_bearing:0,
    _collections:{entities:has?[{engine_record_id:999,type:11,kind:1,relative_x:distance*Math.cos(r),relative_y:distance*Math.sin(r),distance,relative_angle:b,visible:1,countkill:1,health:60,radius:20,height:56}]:[],geometry:[]}};
}
const mass=(...xs)=>ids.map(id=>xs.includes(id)?1/xs.length:0);
// Deterministic training set: several bearings/distances per regime.
const data=[];
for(const d of [128,320,900]){
  for(const b of [.14,.24,.4]){
    data.push([state(b,{distance:d}),mass("turn_left","turn_left_fire","strafe_left")]);
    data.push([state(-b,{distance:d}),mass("turn_right","turn_right_fire","strafe_right")]);
  }
  for(const b of [0,.03,-.03])data.push([state(b,{distance:d}),mass("fire","forward_fire","back_fire","strafe_left_fire","strafe_right_fire")]);
}
for(let k=0;k<3;k++)data.push([state(null),mass("forward","turn_left","turn_right","strafe_left","strafe_right")]);

function build(){
  return new SemanticResidualPolicy({schema,actions,semantic,residual,inferenceMode:"neural",seed:7});
}
function measure(p){
  const score=obs=>{
    const enc=p.encode(obs,false,false),ev=p.residualEvaluation(obs,enc.features,enc.temporal,[]),probs=softmax(ev.scores,p.temperature);
    const m=pred=>actions.reduce((s,a,i)=>s+(pred(a)?probs[i]:0),0);
    return{probs,top:ids[probs.indexOf(Math.max(...probs))],fire:m(a=>a.params.trigger===1),turnL:m(a=>a.params.view===1),turnR:m(a=>a.params.view===2),wait:probs[ids.indexOf("wait")]};
  };
  const L=score(state(.24)),R=score(state(-.24)),A=score(state(0)),Q=score(state(null));
  const tv=.5*L.probs.reduce((s,v,i)=>s+Math.abs(v-R.probs[i]),0);
  const lo=p=>Math.log(Math.max(1e-9,p)/Math.max(1e-9,1-p));
  return{L,R,A,Q,tv,turnMargin:.5*((L.turnL-L.turnR)+(R.turnR-R.turnL)),fireLogOdds:lo(A.fire)-lo(Q.fire)};
}

// 1) Reproduce the pre-fix failure mode on the head alone: marginal-per-axis
// training composes into a wait-dominated joint even with perfect axis fit.
{
  const legacy=new FactorizedControlHead(schema,actions,{lr:.08});
  const marginalSupervise=(obs,target)=>{ // legacy objective, kept only as a regression witness
    const x=legacy.encode(obs),logits=legacy.axisLogits(obs);
    legacy.axes.forEach((axis,ai)=>{
      const marg=new Array(axis.values.length).fill(0);
      actions.forEach((a,k)=>{const vi=axis.index.get(String(a.params[axis.id]));if(vi!=null)marg[vi]+=target[k]});
      const pred=softmax(Array.from(logits[ai]),1);
      axis.values.forEach((_,vi)=>{const row=legacy.weights[ai][vi];for(let j=0;j<legacy.inputDim;j++)row[j]-=legacy.lr*(pred[vi]-marg[vi])*x[j]});
    });
  };
  for(let e=0;e<200;e++)for(const [o,t] of data)marginalSupervise(o,t);
  const j=softmax(legacy.jointScores(legacy.axisLogits(state(.24))),1);
  console.log("legacy marginal objective, left state: wait",j[ids.indexOf("wait")].toFixed(3),"turn_left",j[ids.indexOf("turn_left")].toFixed(3));
}

// 2) Production pathway must learn and DEPLOY the relation.
const p=build(),before=measure(p);
for(let epoch=0;epoch<60;epoch++)for(const [obs,target] of data)p.superviseDecisionDistribution(obs,target,{steps:1,strength:.6});
const after=measure(p);
const fmt=m=>({top:[m.L.top,m.A.top,m.R.top,m.Q.top],tv:+m.tv.toFixed(3),turnMargin:+m.turnMargin.toFixed(3),fireLogOdds:+m.fireLogOdds.toFixed(2),waitLeft:+m.L.wait.toFixed(3)});
console.log("before",fmt(before));console.log("after ",fmt(after));

assert.ok(after.L.turnL>after.L.turnR+.35,"hostile left must raise left-turn mass over right-turn mass by >.35");
assert.ok(after.R.turnR>after.R.turnL+.35,"mirrored right state must reverse the preference by >.35");
assert.ok(after.turnMargin>.4,"signed directional margin must exceed .4");
assert.ok(after.tv>.4,"mirrored states must produce materially different policies (TV>.4)");
assert.ok(after.fireLogOdds>2,"aligned hostile must raise fire log-odds over no-hostile by >2");
assert.ok(after.A.fire>.6&&after.Q.fire<.3,"fire should dominate when aligned and be unlikely when quiet");
assert.ok(after.L.wait<.05&&after.Q.wait<.05,"unlabelled wait must not become the mode");

// 3) Serialization must preserve the deployed distribution exactly.
const restored=build();restored.importCheckpoint(JSON.parse(JSON.stringify(p.exportCheckpoint())));
const again=measure(restored);
for(const k of ["L","R","A","Q"])again[k].probs.forEach((v,i)=>assert.ok(Math.abs(v-after[k].probs[i])<1e-9,"checkpoint round-trip changed "+k));
console.log("factor grounding ok",{residual,params:p.factorHead.parameterCount(),examples:data.length});
