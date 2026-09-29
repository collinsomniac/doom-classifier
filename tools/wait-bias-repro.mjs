// Minimal reproduction: marginal-per-axis training of a factorized head on
// multimodal composite targets makes the all-neutral composite (wait) win.
import {FactorizedControlHead} from "../src/core/factorized-control-head.js";
import {DoomWasmArena} from "../src/env/doom-wasm.js";
const arena=new DoomWasmArena({}),head=new FactorizedControlHead(arena.schema,arena.actions,{lr:.08});
const ids=arena.actions.map(a=>a.id),mix=(...xs)=>ids.map(id=>xs.includes(id)?1/xs.length:0);
const obs=b=>({visible_hostile_count:b===null?0:1,nearest_visible_hostile_distance:b===null?4096:192,nearest_visible_hostile_bearing:b??0,
  visible_hostile_bearing_zone:b===null?0:Math.abs(b)<=.06?1:b>0?2:3,visible_hostile_distance_zone:b===null?0:1,aim_alignment:b===null?0:1-Math.min(1,Math.abs(b)*8)});
// A hostile on the left: "turn left" and "strafe left" are equally good; wait is never labelled.
const data=[[obs(.3),mix("turn_left","strafe_left")],[obs(-.3),mix("turn_right","strafe_right")],[obs(0),mix("fire","forward_fire")],[obs(null),mix("forward","turn_left")]];
for(let e=0;e<400;e++)for(const [o,t] of data)head.supervise(o,t);
for(const [o] of data){const p=head.evaluate(o).scores,e=p.map(Math.exp),z=e.reduce((a,b)=>a+b,0);
  console.log(ids.map((id,i)=>[id,e[i]/z]).sort((a,b)=>b[1]-a[1]).slice(0,3).map(([id,v])=>id+":"+v.toFixed(2)).join(" "));}
