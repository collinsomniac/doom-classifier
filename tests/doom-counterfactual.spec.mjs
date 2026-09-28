import {test,expect} from "@playwright/test";

test.setTimeout(180000);

test("exact same-state DOOM probes expose useful local action separation",async({page})=>{
  await page.goto("http://127.0.0.1:8000/doom.html",{waitUntil:"domcontentloaded"});
  const result=await page.evaluate(async()=>{
    const [{DoomWasmArena},{probeCounterfactualActions}]=await Promise.all([
      import("/src/env/doom-wasm.js"),import("/src/core/counterfactual.js")
    ]);
    const canvas=document.createElement("canvas");canvas.width=640;canvas.height=400;canvas.style.display="none";document.body.appendChild(canvas);
    const base="https://raw.githubusercontent.com/collinsomniac/doom-classifier/engine-runtime";
    const env=await DoomWasmArena.boot({canvas,runtimeBase:base,runtimeInfo:{owned:true,base,source:"counterfactual-benchmark"},actionMs:110});
    if(!env.supportsSnapshots()||!env.supportsExactTics())throw new Error("owned runtime missing deterministic fork support");
    const actions=env.actions,prior=actions.map(()=>1/actions.length),out=[];
    for(const tics of [6,12,24,35]){
      await env.reset();
      const before=JSON.stringify(env.lastObservation);
      const started=performance.now();
      const probe=await probeCounterfactualActions({
        environment:env,actions,prior,horizon:1,temperature:.45,priorStrength:1,
        stepper:id=>env.stepTics(id,tics)
      });
      const elapsed=performance.now()-started,after=JSON.stringify(env.lastObservation);
      if(before!==after)throw new Error("probe failed to restore flattened environment state");
      const ranked=probe.trials.map(t=>({
        id:t.id,return:Number(t.return),damage:Number(t.outcome?.damageDealt||0),
        kills:Number(t.outcome?.playerKillDelta||0),pickups:Number(t.outcome?.playerPickupDelta||0),
        healthDelta:Number(t.outcome?.healthDelta||0)
      })).sort((a,b)=>b.return-a.return);
      out.push({
        tics,elapsed,spread:probe.spread,best:ranked[0],worst:ranked.at(-1),
        positive:ranked.filter(x=>x.return>0).length,uniqueReturns:new Set(ranked.map(x=>x.return.toFixed(6))).size,
        top:ranked.slice(0,5),targetTop:actions[probe.target.indexOf(Math.max(...probe.target))].id
      });
    }
    return out;
  });

  expect(result).toHaveLength(4);
  for(const row of result){
    expect(Number.isFinite(row.spread)).toBe(true);
    expect(row.elapsed).toBeGreaterThan(0);
  }
  expect(result.some(row=>row.spread>0)).toBe(true);
  console.log("DOOM_COUNTERFACTUAL_HORIZONS "+JSON.stringify(result));
});
