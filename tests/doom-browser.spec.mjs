import {test,expect} from "@playwright/test";

test.setTimeout(420000);

test("real DOOM verifies firing, prepares semantics, and runs the neural fast path",async({page})=>{
  const consoleErrors=[];page.on("pageerror",error=>consoleErrors.push("pageerror: "+String(error)));page.on("console",message=>{if(message.type()==="error")consoleErrors.push("console: "+message.text())});
  await page.goto("http://127.0.0.1:8000/doom.html?starter=off",{waitUntil:"domcontentloaded"});await page.locator("#bootBtn").click();
  await page.waitForFunction(()=>{const text=document.querySelector("#runtimeStatus")?.textContent;return text==="ENGINE READY"||text==="BOOT FAILED"},null,{timeout:90000});
  if((await page.locator("#runtimeStatus").textContent())!=="ENGINE READY")throw new Error("DOOM boot failed. "+(await page.locator("#bootStatus").textContent()));
  await expect(page.locator("#stateTable .state-row")).toHaveCount(33);
  await expect(page.locator("#playTab")).toBeVisible();
  await expect(page.locator("#inspectTab")).toBeHidden();
  await expect(page.locator("#decisionCircuit")).toBeVisible();
  const desktopPlay=await page.evaluate(()=>({canvas:document.querySelector("#doomCanvas")?.getBoundingClientRect().width||0,viewport:innerWidth}));
  expect(desktopPlay.canvas).toBeGreaterThan(desktopPlay.viewport*.70);
  await expect(page.locator('[data-arch="environment"]')).toHaveClass(/active/);
  expect(await page.evaluate(()=>window.__doomLab.env.runtime?.owned)).toBe(true);
  await expect(page.locator("#weaponState")).toContainText("pistol");
  await expect(page.locator("#manualActionSelect")).toHaveValue("fire");

  const fire=await page.evaluate(async()=>{
    const lab=window.__doomLab,env=lab.env;env.setActionMs(350);
    const before=env.observe(),ammoBefore=before.bullets+before.shells+before.rockets+before.cells;
    let ammoAfter=ammoBefore,damage=0,pulses=0;
    for(;pulses<8&&ammoAfter===ammoBefore&&damage===0;pulses++){
      const step=await env.step("fire"),after=step.observation;
      ammoAfter=after.bullets+after.shells+after.rockets+after.cells;damage+=step.info?.outcome?.hostileHpLoss||0;
    }
    await env.reset();env.setActionMs(110);
    return{ammoBefore,ammoAfter,damage,pulses,mask:env.actionMasks.fire,combo:env.actionMasks.strafe_left_fire,forwardTurnFire:env.actionMasks.forward_turn_left_fire,strafeTurnFire:env.actionMasks.strafe_right_turn_right_fire,useCombo:env.actionMasks.forward_turn_left_fire_use,actionCount:env.actions.length};
  });
  expect(fire.mask).toBe(64);expect(fire.combo).toBe(80);expect(fire.forwardTurnFire).toBe(69);expect(fire.strafeTurnFire).toBe(104);expect(fire.useCombo).toBe(197);expect(fire.actionCount).toBe(60);expect(fire.ammoAfter<fire.ammoBefore||fire.damage>0).toBeTruthy();

  await page.locator("#prepareBtn").click();
  await page.waitForFunction(()=>{const text=document.querySelector("#prepareStatus")?.textContent||"";return text.includes("READY TO PLAY")||text.includes("Preparation failed")},null,{timeout:300000});
  const prepare=(await page.locator("#prepareStatus").textContent())||"";
  if(prepare.includes("failed"))throw new Error("Preparation failed: "+prepare+" browser="+consoleErrors.join(" | "));
  await expect(page.locator("#schemaChip")).toContainText("schema compiled");await expect(page.locator("#teacherChip")).toContainText("teacher loaded");await expect(page.locator("#policyChip")).toContainText("ready to play");
  await expect(page.locator("#backboneName")).toContainText("params");
  await page.locator("#inspectTabBtn").click();
  await expect(page.locator("#inspectTab")).toBeVisible();
  await expect(page.locator("#architectureFlow")).toBeVisible();
  await expect(page.locator("#teacherTranscript .teacher-item").first()).toBeVisible();
  await expect(page.locator('[data-arch="teacher"]')).toHaveClass(/active/);
  const prepareFit=await page.evaluate(()=>window.__doomLab.policy.lastTeacherResult);console.log("TEACHER_BOOTSTRAP_FIT "+JSON.stringify(prepareFit));
  expect(Number(prepareFit?.distillation?.kl)).toBeLessThan(.12);

  await page.locator("#stepBtn").click();await expect(page.locator("#steps")).toHaveText("1",{timeout:8000});
  await expect(page.locator("#chosenAction")).not.toHaveText("—");await expect(page.locator("#eventLog")).toContainText("s0001");
  await expect(page.locator("#circuitChosen")).not.toContainText("waiting");
  await expect(page.locator("#circuitOutputLabel")).not.toContainText("waiting");
  await expect(page.locator("#circuitPriorTop")).not.toHaveText("—");
  await expect(page.locator("#circuitValueTop")).not.toHaveText("—");
  await expect(page.locator("#circuitActions .circuit-action")).toHaveCount(60);
  await expect(page.locator("#circuitActions .circuit-action.chosen")).toHaveCount(1);
  await expect(page.locator("#attentionList .attention-row").first()).toBeVisible();
  await expect(page.locator("#trainingOutput")).toContainText("s0001");
  expect(await page.evaluate(()=>window.__doomLab.env.supportsSnapshots()&&window.__doomLab.env.supportsExactTics())).toBe(true);
  await page.selectOption("#counterfactualTics","6");
  await page.click("#counterfactualProbeBtn");
  await expect(page.locator("#counterfactualResults .counterfactual-row")).toHaveCount(60);
  await expect(page.locator("#counterfactualStatus")).toContainText("state restored");
  expect(await page.evaluate(()=>window.__doomLab.counterfactual?.trials?.length)).toBe(60);
  await expect(page.locator('[data-arch="actions"]')).toHaveClass(/hot/);
  await expect(page.locator("#architectureFlow")).toHaveClass(/tick-pulse/);
  const pulseAnimation=await page.evaluate(()=>getComputedStyle(document.querySelector('[data-arch="actions"]')).animationName);
  expect(pulseAnimation).toContain("archTravelPulse");
  await expect(page.locator("#typedTrace")).toBeVisible();
  await expect(page.locator("#typedRawTop")).not.toHaveText("—");
  await expect(page.locator("#typedProjector")).toContainText("fit");
  await expect(page.locator("#typedFused")).not.toHaveText("—");
  await expect(page.locator("#typedFieldGrid .typed-field")).toHaveCount(8);

  const semanticProbes=await page.evaluate(async()=>{
    const {env,policy}=window.__doomLab,base=env.observe(),actions=policy.actions;
    const mkEntity=(overrides={})=>({engine_record_id:99,type:1,x:base.player_x+128,y:base.player_y,z:base.player_z,relative_x:128,relative_y:0,relative_z:0,velocity_x:0,velocity_y:0,radius:20,height:56,health:40,distance:128,relative_angle:0,visible:1,countkill:1,pickup:0,targeting_player:1,...overrides});
    const noPickup={nearest_pickup_distance:8192,nearest_pickup_relative_angle:0};
    const cases={
      enemy_ahead:{...base,health:100,recent_damage:0,recent_hostile_hp_loss:0,under_fire:0,bullets:50,hostile_count:1,visible_hostile_count:1,targeting_player_count:1,nearest_hostile_distance:128,nearest_hostile_relative_angle:0,nearest_hostile_visible:1,...noPickup,_collections:{entities:[mkEntity()],geometry:[]}},
      under_fire_side:{...base,health:35,recent_damage:20,recent_hostile_hp_loss:0,under_fire:1,bullets:50,hostile_count:1,visible_hostile_count:1,targeting_player_count:1,nearest_hostile_distance:116,nearest_hostile_relative_angle:.22,nearest_hostile_visible:1,...noPickup,_collections:{entities:[mkEntity({relative_x:64,relative_y:96,distance:116,relative_angle:.22})],geometry:[]}},
      quiet_room:{...base,health:100,recent_damage:0,recent_hostile_hp_loss:0,under_fire:0,bullets:50,hostile_count:0,visible_hostile_count:0,targeting_player_count:0,nearest_hostile_distance:8192,nearest_hostile_relative_angle:0,nearest_hostile_visible:0,...noPickup,_collections:{entities:[],geometry:[]}}
    };
    const out={};
    for(const [name,obs] of Object.entries(cases)){
      const scores=await policy.semantic.score(obs),exp=scores.map(Math.exp),z=exp.reduce((a,b)=>a+b,0)||1,probs=exp.map(v=>v/z);
      const ranked=actions.map((a,i)=>({id:a.id,p:probs[i]})).sort((a,b)=>b.p-a.p);
      const marginal=field=>actions.reduce((sum,a,i)=>sum+(a.params?.[field]?probs[i]:0),0);
      out[name]={top:ranked.slice(0,5),fire:marginal("fire"),strafe:marginal("strafe_left")+marginal("strafe_right"),turn:marginal("turn_left")+marginal("turn_right"),forward:marginal("forward"),back:marginal("back"),use:marginal("use")};
    }
    return out;
  });
  console.log("SEMANTIC_DOOM_PROBES "+JSON.stringify(semanticProbes));

  await page.setViewportSize({width:390,height:844});
  await page.locator("#inspectTabBtn").click();
  const inspectMobile=await page.evaluate(()=>{
    const flow=document.querySelector("#architectureFlow"),fr=flow?.getBoundingClientRect();
    const offenders=[...document.querySelectorAll("#inspectTab *")].map(el=>{const r=el.getBoundingClientRect();return{tag:el.tagName,id:el.id||"",right:r.right,left:r.left,width:r.width}}).filter(x=>x.right>innerWidth+1||x.left<-1).sort((a,b)=>b.width-a.width).slice(0,12);
    return{viewport:innerWidth,scrollWidth:document.documentElement.scrollWidth,flowWidth:fr?.width||0,offenders};
  });
  console.log("MOBILE_INSPECT_LAYOUT "+JSON.stringify(inspectMobile));
  expect(inspectMobile.scrollWidth,"overflow offenders: "+JSON.stringify(inspectMobile.offenders)).toBeLessThanOrEqual(inspectMobile.viewport+1);
  expect(inspectMobile.flowWidth).toBeLessThanOrEqual(inspectMobile.viewport);
  await page.locator("#playTabBtn").click();
  const playMobile=await page.evaluate(()=>{
    const canvas=document.querySelector("#doomCanvas")?.getBoundingClientRect(),play=document.querySelector("#playTab")?.getBoundingClientRect();
    return{viewport:innerWidth,canvasWidth:canvas?.width||0,playWidth:play?.width||0,inspectHidden:document.querySelector("#inspectTab")?.hidden};
  });
  console.log("MOBILE_PLAY_LAYOUT "+JSON.stringify(playMobile));
  expect(playMobile.inspectHidden).toBe(true);
  expect(playMobile.canvasWidth).toBeGreaterThan(playMobile.viewport*.75);
  expect(playMobile.canvasWidth).toBeLessThanOrEqual(playMobile.playWidth+1);
  if(consoleErrors.length)throw new Error("Browser errors after successful prepared-model step: "+consoleErrors.join(" | "));
});
