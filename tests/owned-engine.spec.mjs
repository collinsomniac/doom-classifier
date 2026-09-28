import {test,expect} from "@playwright/test";

test.setTimeout(180000);

test("owned Chocolate Doom runtime exposes causal event counters",async({page})=>{
  const consoleErrors=[];
  page.on("console",msg=>{if(msg.type()==="error")consoleErrors.push(msg.text())});
  page.on("pageerror",err=>consoleErrors.push(String(err)));

  await page.goto("http://127.0.0.1:8000/doom.html?starter=off",{waitUntil:"domcontentloaded"});

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
    let after=before,pulses=0,lastOutcome=null;
    while(pulses<12&&Number(after.events?.player_damage_dealt||0)===0){
      const step=await env.step("fire");
      lastOutcome=step.info?.outcome||null;
      after=env.readRaw();
      pulses++;
    }

    const observed={...(after.events||{})};
    await env.reset();
    env.setActionMs(180);
    const simplify=raw=>({
      player:{
        health:Number(raw.player?.health||0),armor:Number(raw.player?.armor||0),weapon:Number(raw.player?.weapon||0),
        bullets:Number(raw.player?.ammo?.bullets||0),shells:Number(raw.player?.ammo?.shells||0),rockets:Number(raw.player?.ammo?.rockets||0),cells:Number(raw.player?.ammo?.cells||0),
        x:Number(raw.player?.x||0),y:Number(raw.player?.y||0),z:Number(raw.player?.z||0),vx:Number(raw.player?.vx||0),vy:Number(raw.player?.vy||0),angle:Number(raw.player?.angle||0),kills:Number(raw.player?.kills||0)
      },
      events:{...(raw.events||{})},
      enemies:(raw.world?.entities||[]).filter(e=>e.enemy).map(e=>({
        type:Number(e.type||0),x:Number(e.x||0),y:Number(e.y||0),z:Number(e.z||0),
        vx:Number(e.vx||0),vy:Number(e.vy||0),health:Number(e.health||0),targeting:!!e.targeting_player
      })).sort((a,b)=>JSON.stringify(a).localeCompare(JSON.stringify(b)))
    });
    const snapshotSupported=env.supportsSnapshots(),snapshotToken=snapshotSupported?env.saveSnapshot():null;
    const snapshotBefore=simplify(env.readRaw());
    if(snapshotSupported){
      await env.step("turn_right_fire");
      env.restoreSnapshot(snapshotToken);
    }
    const snapshotRestored=simplify(env.readRaw());

    await env.reset();
    const reset={...(env.readRaw().events||{})};
    env.setActionMs(110);

    return{runtime:env.runtime,initial,observed,reset,pulses,lastOutcome,snapshotSupported,snapshotBefore,snapshotRestored};
  });

  expect(result.runtime.owned).toBe(true);
  for(const key of ["player_damage_dealt","player_kills","player_pickups","level_completions","secret_exits"]){
    expect(Object.prototype.hasOwnProperty.call(result.initial,key),key+" missing from initial event ABI").toBe(true);
    expect(typeof result.initial[key]).toBe("number");
    expect(result.reset[key],key+" must reset for a controlled episode").toBe(0);
  }
  expect(result.observed.player_damage_dealt).toBeGreaterThan(0);
  expect(result.observed.player_kills).toBeGreaterThanOrEqual(0);
  expect(result.lastOutcome?.combatAttributionAvailable).toBe(true);
  expect(result.lastOutcome?.damageAttributed).toBe(true);
  expect(result.lastOutcome?.damageDealt).toBeGreaterThan(0);
  expect(result.lastOutcome?.attributedCombatReward).toBeGreaterThan(0);
  expect(result.lastOutcome?.reward).toBeGreaterThan(0);
  expect(result.snapshotSupported).toBe(true);
  expect(result.snapshotRestored).toEqual(result.snapshotBefore);
  if(consoleErrors.length)throw new Error(consoleErrors.join(" | "));

  console.log("OWNED_ENGINE_ABI "+JSON.stringify(result));
});
