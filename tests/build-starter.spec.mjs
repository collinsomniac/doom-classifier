import {test,expect} from "@playwright/test";
import {mkdirSync,writeFileSync} from "node:fs";
import {dirname,resolve} from "node:path";

test.setTimeout(720000);

async function boot(page){
  await page.goto("http://127.0.0.1:8000/doom.html?starter=off",{waitUntil:"domcontentloaded"});
  await page.locator("#bootBtn").click();
  await expect(page.locator("#runtimeStatus")).toContainText("ENGINE READY",{timeout:120000});
}
async function canonicalProbe(page,observation=null){
  return page.evaluate(async supplied=>{
    const {policy:p,controller:c,env}=window.__doomLab;
    c.pause();c.training=false;c.explore=false;c.memory=false;c.useResidual=true;p.setInferenceMode("neural");
    if(!supplied)await c.reset({learning:false});p.resetEpisode();
    const teacherBefore=p.teacherCalls,obs=supplied||env.observe(),d=await p.decide(obs,{useResidual:true,memory:false,explore:false});
    return{observation:obs,action:d.action.id,probs:[...d.probs],qScores:[...d.qScores],semantic:[...(d.semanticPriorScores||[])],value:[...(d.valueScores||[])],teacherCalls:p.teacherCalls-teacherBefore};
  },observation);
}

async function frozenEval(page,steps=24){
  return page.evaluate(async steps=>{
    const {policy:p,controller:c}=window.__doomLab;
    c.pause();c.training=false;c.explore=false;c.memory=true;c.useResidual=true;p.setInferenceMode("neural");
    await c.reset({learning:false});const teacherBefore=p.teacherCalls,actions={};let reward=0,damage=0,kills=0;
    for(let i=0;i<steps;i++){
      const ok=await c.tick();if(!ok&&c.state==="ERROR")throw new Error("controller error during frozen evaluation");
      const d=c.lastDecision;if(!d)continue;
      reward+=Number(d.reward||0);damage+=Number(d.outcome?.damageDealt||0);kills+=Number(d.outcome?.playerKillDelta||0);
      actions[d.action.id]=(actions[d.action.id]||0)+1;
    }
    const dominant=Object.entries(actions).sort((a,b)=>b[1]-a[1])[0]||["none",0];
    return{steps,reward,damage,kills,teacherCalls:p.teacherCalls-teacherBefore,dominant,params:p.q.parameterCount(),updates:p.q.updates};
  },steps);
}

test("build and round-trip a quality-gated teacher-free starter checkpoint",async({page,context})=>{
  await boot(page);
  await page.locator("#prepareBtn").click();
  await expect(page.locator("#policyChip")).toContainText("ready to play",{timeout:360000});

  const before=await frozenEval(page,24);
  const trained=await page.evaluate(async()=>{
    const {policy:p,controller:c}=window.__doomLab;
    p.setInferenceMode("adaptive");c.memory=true;c.useResidual=true;
    const result=await c.trainBurst({steps:256,epsilon:.16,rolloutHorizon:64});
    await p.awaitTeacher?.();
    return{...result,params:p.q.parameterCount(),updates:p.q.updates,teacherCalls:p.teacherCalls};
  });
  const after=await frozenEval(page,24);
  const checkpoint=await page.evaluate(()=>window.__doomLab.policy.exportCheckpoint());
  const probe=await canonicalProbe(page);

  expect(checkpoint.format).toBe("doom-classifier-policy");
  expect(checkpoint.q.params).toBe(after.params);
  expect(after.teacherCalls,"candidate playback must be teacher-free").toBe(0);
  expect(trained.updates).toBeGreaterThan(0);
  expect(trained.completed).toBe(256);
  expect(after.kills,"starter must retain the baseline kill count").toBeGreaterThanOrEqual(before.kills);
  expect(after.reward,"starter return regressed too far").toBeGreaterThanOrEqual(before.reward*.70);
  expect(after.damage,"starter attributed damage regressed too far").toBeGreaterThanOrEqual(before.damage*.60);

  const output=resolve(process.env.STARTER_OUTPUT||"artifacts/doom-starter.json");
  mkdirSync(dirname(output),{recursive:true});
  writeFileSync(output,JSON.stringify(checkpoint));

  await page.evaluate(cp=>localStorage.setItem("doom-classifier-checkpoint-v1",JSON.stringify(cp)),checkpoint);
  await page.close();
  const fresh=await context.newPage();
  await boot(fresh);
  await fresh.locator("#loadSavedBtn").click();
  await expect(fresh.locator("#checkpointStatus")).toContainText("loaded",{timeout:30000});
  await expect(fresh.locator("#teacherChip")).toContainText("teacher-free checkpoint");
  const replayProbe=await canonicalProbe(fresh,probe.observation);
  expect(replayProbe.teacherCalls).toBe(0);
  expect(replayProbe.action).toBe(probe.action);
  expect(replayProbe.probs.length).toBe(probe.probs.length);
  expect(Math.max(...replayProbe.probs.map((v,i)=>Math.abs(v-probe.probs[i])))).toBeLessThan(1e-6);
  expect(Math.max(...replayProbe.qScores.map((v,i)=>Math.abs(v-probe.qScores[i])))).toBeLessThan(1e-6);

  const replay=await frozenEval(fresh,24);
  expect(replay.teacherCalls).toBe(0);
  expect(replay.params).toBe(after.params);
  expect(replay.damage,"fresh-engine replay should retain substantial combat behavior").toBeGreaterThanOrEqual(after.damage*.5);
  expect(replay.reward).toBeGreaterThan(0);

  console.log("STARTER_CHECKPOINT "+JSON.stringify({before,trained,after,probe,replayProbe,replay,bytes:JSON.stringify(checkpoint).length,output}));
});
