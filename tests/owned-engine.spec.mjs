import {test,expect} from "@playwright/test";

test.setTimeout(180000);

test("owned Chocolate Doom runtime exposes causal event counters",async({page})=>{
  const consoleErrors=[];
  page.on("console",msg=>{if(msg.type()==="error")consoleErrors.push(msg.text())});
  page.on("pageerror",err=>consoleErrors.push(String(err)));

  await page.goto("http://127.0.0.1:8000/doom.html",{waitUntil:"domcontentloaded"});

  const result=await page.evaluate(async()=>{
    const {DoomWasmArena}=await import("/src/env/doom-wasm.js");
    const canvas=document.createElement("canvas");
    canvas.width=640;canvas.height=400;canvas.style.display="none";document.body.appendChild(canvas);

    const env=await DoomWasmArena.boot({
      canvas,
      actionMs:350,
      runtimeBase:"/engine/dist",
      runtimeInfo:{owned:true,source:"owned-engine-ci",base:"/engine/dist"}
    });

    const before=env.readRaw(),initial={...(before.events||{})};
    let after=before,pulses=0;
    while(pulses<12&&Number(after.events?.player_damage_dealt||0)===0){
      await env.step("fire");
      after=env.readRaw();
      pulses++;
    }

    const observed={...(after.events||{})};
    await env.reset();
    const reset={...(env.readRaw().events||{})};
    env.setActionMs(110);

    return{runtime:env.runtime,initial,observed,reset,pulses};
  });

  expect(result.runtime.owned).toBe(true);
  for(const key of ["player_damage_dealt","player_kills","player_pickups","level_completions","secret_exits"]){
    expect(Object.prototype.hasOwnProperty.call(result.initial,key),key+" missing from initial event ABI").toBe(true);
    expect(typeof result.initial[key]).toBe("number");
    expect(result.reset[key],key+" must reset for a controlled episode").toBe(0);
  }
  expect(result.observed.player_damage_dealt).toBeGreaterThan(0);
  expect(result.observed.player_kills).toBeGreaterThanOrEqual(0);
  if(consoleErrors.length)throw new Error(consoleErrors.join(" | "));

  console.log("OWNED_ENGINE_ABI "+JSON.stringify(result));
});
