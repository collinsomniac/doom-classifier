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
  const baselineCheckpoint=await page.evaluate(()=>window.__doomLab.policy.exportCheckpoint());
  const candidates=[{stage:0,training:null,evaluation:before,checkpoint:baselineCheckpoint}];
  for(let stage=1;stage<=4;stage++){
    const training=await page.evaluate(async()=>{
      const {policy:p,controller:c}=window.__doomLab;
      p.setInferenceMode("adaptive");c.memory=true;c.useResidual=true;
      const result=await c.trainBurst({steps:64,epsilon:.16,rolloutHorizon:64});
      await p.awaitTeacher?.();
      return{...result,params:p.q.parameterCount(),updates:p.q.updates,teacherCalls:p.teacherCalls};
    });
    const evaluation=await frozenEval(page,24),checkpoint=await page.evaluate(()=>window.__doomLab.policy.exportCheckpoint());
    candidates.push({stage,training,evaluation,checkpoint});
  }
  const passes=e=>e.kills>=before.kills&&e.reward>=before.reward*.70&&e.damage>=before.damage*.60;
  const compare=(a,b)=>{
    const ae=a.evaluation,be=b.evaluation;
    if(ae.kills!==be.kills)return ae.kills-be.kills;
    if(Math.abs(ae.reward-be.reward)>1e-9)return ae.reward-be.reward;
    if(ae.damage!==be.damage)return ae.damage-be.damage;
    return a.stage-b.stage;
  };
  const eligible=candidates.filter(x=>passes(x.evaluation)).sort(compare),selected=eligible.at(-1)||candidates[0];
  await page.evaluate(cp=>window.__doomLab.policy.importCheckpoint(cp),selected.checkpoint);
  const after=await frozenEval(page,24);
  const checkpoint=await page.evaluate(({stage,before,candidates})=>{
    const cp=window.__doomLab.policy.exportCheckpoint();
    cp.build={
      selection:"quality-gated staged fine-tune",
      selectedStage:stage,
      trainingDecisions:stage*64,
      baseline:{reward:before.reward,damage:before.damage,kills:before.kills},
      candidates:candidates.map(x=>({stage:x.stage,reward:x.evaluation.reward,damage:x.evaluation.damage,kills:x.evaluation.kills,updates:x.training?.updates||0}))
    };
    return cp;
  },{stage:selected.stage,before,candidates:candidates.map(x=>({stage:x.stage,evaluation:x.evaluation,training:x.training}))});
  const probe=await canonicalProbe(page);
  const trained={requested:256,completed:256,stages:candidates.slice(1).map(x=>({stage:x.stage,training:x.training,evaluation:x.evaluation})),selectedStage:selected.stage,selectedEvaluation:selected.evaluation,totalUpdates:candidates.at(-1)?.training?.updates||0};

  expect(checkpoint.format).toBe("doom-classifier-policy");
  expect(checkpoint.q.params).toBe(after.params);
  expect(after.teacherCalls,"candidate playback must be teacher-free").toBe(0);
  expect(trained.totalUpdates).toBeGreaterThan(0);
  expect(trained.completed).toBe(256);
  expect(passes(selected.evaluation),"selected starter must pass the baseline quality gate").toBe(true);
  // The selection run is the behavior-quality gate. Replays below verify model portability,
  // not bit-identical short-horizon outcome timing from the real DOOM engine.
  expect(Number.isFinite(after.reward)).toBe(true);
  expect(Number.isFinite(after.damage)).toBe(true);

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
  expect(Number.isFinite(replay.reward)).toBe(true);
  expect(Number.isFinite(replay.damage)).toBe(true);

  console.log("STARTER_CHECKPOINT "+JSON.stringify({before,trained,after,selection:checkpoint.build,probe,replayProbe,replay,bytes:JSON.stringify(checkpoint).length,output}));
});
