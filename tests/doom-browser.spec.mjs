import {test,expect} from "@playwright/test";

test.setTimeout(420000);

test("real DOOM verifies firing, prepares semantics, and runs the neural fast path",async({page})=>{
  const consoleErrors=[];page.on("pageerror",error=>consoleErrors.push("pageerror: "+String(error)));page.on("console",message=>{if(message.type()==="error")consoleErrors.push("console: "+message.text())});
  await page.goto("http://127.0.0.1:8000/doom.html",{waitUntil:"domcontentloaded"});await page.locator("#bootBtn").click();
  await page.waitForFunction(()=>{const text=document.querySelector("#runtimeStatus")?.textContent;return text==="ENGINE READY"||text==="BOOT FAILED"},null,{timeout:90000});
  if((await page.locator("#runtimeStatus").textContent())!=="ENGINE READY")throw new Error("DOOM boot failed. "+(await page.locator("#bootStatus").textContent()));
  await expect(page.locator("#stateTable .state-row")).toHaveCount(25);
  await expect(page.locator("#architectureFlow")).toBeVisible();
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
    return{ammoBefore,ammoAfter,damage,pulses,mask:env.actionMasks.fire,combo:env.actionMasks.strafe_left_fire};
  });
  expect(fire.mask).toBe(64);expect(fire.combo).toBe(80);expect(fire.ammoAfter<fire.ammoBefore||fire.damage>0).toBeTruthy();

  await page.locator("#prepareBtn").click();
  await page.waitForFunction(()=>{const text=document.querySelector("#prepareStatus")?.textContent||"";return text.includes("READY TO PLAY")||text.includes("Preparation failed")},null,{timeout:300000});
  const prepare=(await page.locator("#prepareStatus").textContent())||"";
  if(prepare.includes("failed"))throw new Error("Preparation failed: "+prepare+" browser="+consoleErrors.join(" | "));
  await expect(page.locator("#schemaChip")).toContainText("schema compiled");await expect(page.locator("#teacherChip")).toContainText("teacher loaded");await expect(page.locator("#policyChip")).toContainText("ready to play");
  await expect(page.locator("#backboneName")).toContainText("params");
  await expect(page.locator("#teacherTranscript .teacher-item").first()).toBeVisible();
  await expect(page.locator('[data-arch="teacher"]')).toHaveClass(/active/);

  await page.locator("#stepBtn").click();await expect(page.locator("#steps")).toHaveText("1",{timeout:8000});
  await expect(page.locator("#chosenAction")).not.toHaveText("—");await expect(page.locator("#eventLog")).toContainText("s0001");
  await expect(page.locator("#attentionList .attention-row").first()).toBeVisible();
  await expect(page.locator("#trainingOutput")).toContainText("s0001");
  await expect(page.locator('[data-arch="actions"]')).toHaveClass(/hot/);

  const semanticProbes=await page.evaluate(async()=>{
    const {env,policy}=window.__doomLab,base=env.observe(),actions=policy.actions;
    const mkEntity=(overrides={})=>({engine_record_id:99,type:1,x:base.player_x+128,y:base.player_y,z:base.player_z,relative_x:128,relative_y:0,relative_z:0,velocity_x:0,velocity_y:0,radius:20,height:56,health:40,distance:128,relative_angle:0,visible:1,countkill:1,pickup:0,targeting_player:1,...overrides});
    const cases={
      enemy_ahead:{...base,health:100,recent_damage:0,recent_hostile_hp_loss:0,under_fire:0,bullets:50,_collections:{entities:[mkEntity()],geometry:[]}},
      under_fire_side:{...base,health:35,recent_damage:20,recent_hostile_hp_loss:0,under_fire:1,bullets:50,_collections:{entities:[mkEntity({relative_x:64,relative_y:96,distance:116,relative_angle:.22})],geometry:[]}},
      quiet_room:{...base,health:100,recent_damage:0,recent_hostile_hp_loss:0,under_fire:0,bullets:50,_collections:{entities:[],geometry:[]}}
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
  const mobileLayout=await page.evaluate(()=>{
    const flow=document.querySelector("#architectureFlow"),env=document.querySelector('[data-arch="environment"]'),state=document.querySelector('[data-arch="state"]');
    const fr=flow?.getBoundingClientRect(),er=env?.getBoundingClientRect(),sr=state?.getBoundingClientRect();
    const offenders=[...document.querySelectorAll("body *")].map(el=>{const r=el.getBoundingClientRect();return{tag:el.tagName,id:el.id||"",className:typeof el.className==="string"?el.className:"",left:r.left,right:r.right,width:r.width}}).filter(x=>x.right>innerWidth+1||x.left<-1).sort((a,b)=>b.width-a.width).slice(0,12);
    return{viewport:innerWidth,scrollWidth:document.documentElement.scrollWidth,flowWidth:fr?.width||0,envTop:er?.top||0,stateTop:sr?.top||0,offenders};
  });
  console.log("MOBILE_LAYOUT "+JSON.stringify(mobileLayout));
  expect(mobileLayout.scrollWidth,"overflow offenders: "+JSON.stringify(mobileLayout.offenders)).toBeLessThanOrEqual(mobileLayout.viewport+1);
  expect(mobileLayout.flowWidth).toBeLessThanOrEqual(mobileLayout.viewport);
  expect(mobileLayout.stateTop).toBeGreaterThan(mobileLayout.envTop);
  if(consoleErrors.length)throw new Error("Browser errors after successful prepared-model step: "+consoleErrors.join(" | "));
});
