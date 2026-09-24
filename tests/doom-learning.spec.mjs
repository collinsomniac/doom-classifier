import {test,expect} from "@playwright/test";

test.setTimeout(600000);

test("prepared real-Doom policy reports pre/post short fine-tune behavior",async({page})=>{
  const errors=[];page.on("pageerror",e=>errors.push("pageerror: "+String(e)));page.on("console",m=>{if(m.type()==="error")errors.push("console: "+m.text())});
  await page.goto("http://127.0.0.1:8000/doom.html",{waitUntil:"domcontentloaded"});
  await page.locator("#bootBtn").click();
  await page.waitForFunction(()=>document.querySelector("#runtimeStatus")?.textContent==="ENGINE READY",null,{timeout:90000});
  await page.locator("#prepareBtn").click();
  await page.waitForFunction(()=>{const t=document.querySelector("#prepareStatus")?.textContent||"";return t.includes("READY TO PLAY")||t.includes("Preparation failed")},null,{timeout:300000});
  const status=await page.locator("#prepareStatus").textContent();
  if(status?.includes("failed"))throw new Error(status);

  const result=await page.evaluate(async()=>{
    const lab=window.__doomLab,{controller:c,policy:p}=lab;
    const evaluate=async(steps)=>{
      c.pause();c.training=false;c.explore=false;c.memory=true;c.useResidual=true;p.setInferenceMode("neural");
      await c.reset({learning:false});
      const teacherBefore=p.teacherCalls,counts={};let reward=0,damage=0,received=0,kills=0,maxP=0,entropy=0,fire=0;
      for(let i=0;i<steps;i++){
        await c.tick();const d=c.lastDecision;if(!d)continue;
        counts[d.action.id]=(counts[d.action.id]||0)+1;
        reward+=d.reward||0;damage+=d.outcome?.damageDealt||0;received+=Math.max(0,-(d.outcome?.healthDelta||0));kills+=d.outcome?.killDelta||0;
        maxP+=Math.max(...d.probs);entropy+=d.uncertainty.entropy;if(d.action.id.includes("fire"))fire++;
      }
      const lat=c.latencySummary(),dominant=Object.entries(counts).sort((a,b)=>b[1]-a[1])[0]||["none",0];
      return{steps,reward,damage,received,kills,fire,diversity:Object.keys(counts).length,dominant,counts,meanMaxP:maxP/steps,meanEntropy:entropy/steps,p95Ms:lat.p95,teacherCalls:p.teacherCalls-teacherBefore,temperature:p.temperature};
    };
    const before=await evaluate(24);
    await c.reset({learning:false});c.training=true;c.explore=true;c.memory=true;c.useResidual=true;p.setInferenceMode("adaptive");
    const updatesBefore=p.q.updates,teacherBefore=p.teacherCalls;
    const train=await c.trainBurst({steps:64,epsilon:.16});
    const training={...train,neuralUpdates:p.q.updates-updatesBefore,teacherCalls:p.teacherCalls-teacherBefore,replaySize:p.replay?.length||0,temperature:p.temperature};
    const after=await evaluate(24);
    return{before,training,after,params:p.q.parameterCount()};
  });

  console.log("DOOM_LEARNING_BENCHMARK "+JSON.stringify(result));
  expect(result.params).toBeGreaterThan(0);
  expect(result.before.teacherCalls).toBe(0);
  expect(result.after.teacherCalls).toBe(0);
  expect(result.training.neuralUpdates).toBeGreaterThan(64);
  expect(result.training.replaySize).toBeGreaterThan(0);
  expect(result.before.steps).toBe(24);expect(result.after.steps).toBe(24);
  if(errors.length)throw new Error(errors.join(" | "));
});
