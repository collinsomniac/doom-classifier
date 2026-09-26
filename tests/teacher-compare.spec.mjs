import {test,expect} from "@playwright/test";

test.setTimeout(600000);

test("compare browser NLI teachers on identical structured DOOM probes",async({page})=>{
  await page.goto("http://127.0.0.1:8000/doom.html",{waitUntil:"domcontentloaded"});
  await page.locator("#bootBtn").click();
  await page.waitForFunction(()=>document.querySelector("#runtimeStatus")?.textContent==="ENGINE READY",null,{timeout:90000});
  await page.locator("#prepareBtn").click();
  await page.waitForFunction(()=>{const t=document.querySelector("#prepareStatus")?.textContent||"";return t.includes("READY TO PLAY")||t.includes("Preparation failed")},null,{timeout:300000});
  if((await page.locator("#prepareStatus").textContent())?.includes("failed"))throw new Error("recommended preparation failed");

  const probe=()=>page.evaluate(async()=>{
    const {env,policy}=window.__doomLab,base=env.observe(),actions=policy.actions,teacher=policy.semantic;
    const teacherCanonical=await teacher.scoreRaw(base),studentCanonical=policy.q.scoreStatsObservation(base,{temporal:null}).semanticScores;
    const nullBias=await teacher.nullBias();
    const softmax=s=>{const peak=Math.max(...s),e=s.map(v=>Math.exp(v-peak)),z=e.reduce((a,b)=>a+b,0)||1;return e.map(v=>v/z)};
    const tp=softmax(teacherCanonical),sp=softmax(studentCanonical),studentKL=tp.reduce((sum,p,i)=>sum+(p>0?p*Math.log(p/Math.max(1e-9,sp[i])):0),0),bootstrap=policy.lastTeacherResult?.distillation||null;
    const entity=(overrides={})=>({engine_record_id:99,type:1,x:base.player_x+128,y:base.player_y,z:base.player_z,relative_x:128,relative_y:0,relative_z:0,velocity_x:0,velocity_y:0,radius:20,height:56,health:40,distance:128,relative_angle:0,visible:1,countkill:1,pickup:0,targeting_player:1,...overrides});
    const cases={
      enemy_ahead:{...base,health:100,recent_damage:0,under_fire:0,bullets:50,_collections:{entities:[entity()],geometry:[]}},
      under_fire_side:{...base,health:35,recent_damage:20,under_fire:1,bullets:50,_collections:{entities:[entity({relative_y:96,distance:116,relative_angle:.22})],geometry:[]}},
      quiet_room:{...base,health:100,recent_damage:0,under_fire:0,bullets:50,_collections:{entities:[],geometry:[]}}
    };
    const summarize=scores=>{
      const e=scores.map(Math.exp),z=e.reduce((a,b)=>a+b,0)||1,p=e.map(v=>v/z);
      const marginal=f=>actions.reduce((sum,a,i)=>sum+(a.params?.[f]?p[i]:0),0);
      return{
        top:actions.map((a,i)=>({id:a.id,p:p[i]})).sort((a,b)=>b.p-a.p).slice(0,5),
        fire:marginal("fire"),strafe:marginal("strafe_left")+marginal("strafe_right"),turn:marginal("turn_left")+marginal("turn_right"),use:marginal("use")
      };
    };
    const raw={},calibrated={};
    for(const [name,obs] of Object.entries(cases)){
      const scores=await teacher.scoreRaw(obs);
      raw[name]=summarize(scores);
      calibrated[name]=summarize(teacher.calibrateScores(scores,nullBias,{strength:1}));
    }
    const discrimination=result=>({
      fireEnemyMinusQuiet:result.enemy_ahead.fire-result.quiet_room.fire,
      strafeUnderFireMinusQuiet:result.under_fire_side.strafe-result.quiet_room.strafe,
      useQuiet:result.quiet_room.use
    });
    return{teacher:teacher.name,backend:teacher.backend,raw,calibrated,rawDiscrimination:discrimination(raw),calibratedDiscrimination:discrimination(calibrated),nullBias:actions.map((a,i)=>({id:a.id,logit:nullBias[i]})).sort((x,y)=>y.logit-x.logit).slice(0,6),studentFit:{canonicalKL:studentKL,bootstrapKL:Number(bootstrap?.kl??NaN),steps:Number(bootstrap?.stepsUsed||0)}};
  });

  const mobile=await probe();
  await page.locator(".advanced-panel").evaluate(el=>{el.open=true});
  await page.locator("#modelSelect").selectOption("distilbert");await page.locator("#loadModelBtn").click();
  await page.waitForFunction(()=>{const t=document.querySelector("#modelStatus")?.textContent||"";return t.includes("bootstrap")||t.includes("failed")},null,{timeout:300000});
  const status=(await page.locator("#modelStatus").textContent())||"";
  if(status.includes("failed"))throw new Error(status);
  const distil=await probe();

  await page.locator("#modelSelect").selectOption("deberta");await page.locator("#loadModelBtn").click();
  await page.waitForFunction(()=>{const t=document.querySelector("#modelStatus")?.textContent||"";return t.includes("bootstrap")||t.includes("failed")},null,{timeout:300000});
  const debertaStatus=(await page.locator("#modelStatus").textContent())||"";
  if(debertaStatus.includes("failed"))throw new Error(debertaStatus);
  const deberta=await probe();

  console.log("DOOM_TEACHER_COMPARISON "+JSON.stringify({scoring:"raw-vs-null-state-calibrated-entailment-logodds",mobile,distil,deberta}));
  expect(mobile.teacher).toContain("MobileBERT");expect(distil.teacher).toContain("DistilBERT");expect(deberta.teacher).toContain("DeBERTa");
});
