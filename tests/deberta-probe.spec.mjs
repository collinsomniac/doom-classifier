import {test,expect} from "@playwright/test";

test.setTimeout(420000);

test("probe DeBERTa xsmall as an evidence-conditioned DOOM teacher",async({page})=>{
  await page.goto("http://127.0.0.1:8000/doom.html",{waitUntil:"domcontentloaded"});
  await page.locator("#bootBtn").click();
  await page.waitForFunction(()=>document.querySelector("#runtimeStatus")?.textContent==="ENGINE READY",null,{timeout:90000});
  await page.locator("#modelSelect").selectOption("deberta");
  await page.locator("#loadModelBtn").click();
  await page.waitForFunction(()=>{const t=document.querySelector("#modelStatus")?.textContent||"";return t.includes("bootstrap")||t.includes("failed")},null,{timeout:240000});
  const status=(await page.locator("#modelStatus").textContent())||"";
  if(status.includes("failed"))throw new Error(status);

  const result=await page.evaluate(async()=>{
    const {env,policy}=window.__doomLab,base=env.observe(),actions=policy.actions;
    const entity=(overrides={})=>({engine_record_id:99,type:1,x:base.player_x+128,y:base.player_y,z:base.player_z,relative_x:128,relative_y:0,relative_z:0,velocity_x:0,velocity_y:0,radius:20,height:56,health:40,distance:128,relative_angle:0,visible:1,countkill:1,pickup:0,targeting_player:1,...overrides});
    const cases={
      enemy_ahead:{...base,health:100,recent_damage:0,under_fire:0,bullets:50,_collections:{entities:[entity()],geometry:[]}},
      under_fire_side:{...base,health:35,recent_damage:20,under_fire:1,bullets:50,_collections:{entities:[entity({relative_y:96,distance:116,relative_angle:.22})],geometry:[]}},
      quiet_room:{...base,health:100,recent_damage:0,under_fire:0,bullets:50,_collections:{entities:[],geometry:[]}}
    };
    const out={};
    for(const [name,obs] of Object.entries(cases)){
      const scores=await policy.semantic.score(obs),e=scores.map(Math.exp),z=e.reduce((a,b)=>a+b,0)||1,p=e.map(v=>v/z);
      const marginal=f=>actions.reduce((sum,a,i)=>sum+(a.params?.[f]?p[i]:0),0);
      out[name]={top:actions.map((a,i)=>({id:a.id,p:p[i]})).sort((a,b)=>b.p-a.p).slice(0,5),fire:marginal("fire"),strafe:marginal("strafe_left")+marginal("strafe_right"),turn:marginal("turn_left")+marginal("turn_right"),use:marginal("use")};
    }
    return{teacher:policy.semantic.name,backend:policy.semantic.backend,cases:out};
  });
  console.log("DOOM_DEBERTA_PROBE "+JSON.stringify(result));
  expect(result.teacher).toContain("DeBERTa");
});
