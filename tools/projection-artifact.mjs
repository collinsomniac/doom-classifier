// Does the typed pre-projection of causal labels invent a "use" preference?
import {projectValueToActionFields} from "../src/core/action-factorization.js";
import {DoomWasmArena} from "../src/env/doom-wasm.js";
import {softmax,argmax} from "../src/core/math.js";
const a=new DoomWasmArena({}),ids=a.actions.map(x=>x.id);
// Typical probe: firing branches lose slightly (ammo/noise), everything else ties.
const returns=a.actions.map(x=>x.params.trigger?-.02:0);
const t=softmax(returns.map(r=>r/.28),1),log=t.map(Math.log);
const pr=projectValueToActionFields(a.schema,a.actions,log,{ridge:.035,maxBlend:.7});
const blended=softmax(pr.scores,1);
console.log("measured top",ids[argmax(t)],"(tie among non-fire)");
console.log("projected top",ids[argmax(blended)],blended.map((p,i)=>ids[i]+":"+p.toFixed(3)).filter((_,i)=>!a.actions[i].params.trigger).join(" "));
