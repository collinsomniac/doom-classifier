import {test,expect} from "@playwright/test";
import {mkdirSync,writeFileSync} from "node:fs";
import {resolve} from "node:path";

test.setTimeout(720000);

const causalSteps=Number(process.env.CAUSAL_TRAIN_STEPS||768);
const replayRollouts=Number(process.env.CAUSAL_REPLAY_ROLLOUTS||16);
const replaySteps=Number(process.env.CAUSAL_REPLAY_STEPS||64);
const killTarget=Number(process.env.CAUSAL_KILL_TARGET||20);
const outputDir=resolve(process.env.CAUSAL_OUTPUT_DIR||"artifacts/causal-combat");

async function bootAndPrepare(page){
  await page.goto("http://127.0.0.1:8000/doom.html?starter=off",{waitUntil:"domcontentloaded"});
  await page.locator("#bootBtn").click();
  await expect(page.locator("#runtimeStatus")).toContainText("ENGINE READY",{timeout:120000});
  await page.locator("#prepareBtn").click();
  await expect(page.locator("#policyChip")).toContainText("ready to play",{timeout:360000});
}

async function evaluateExact(page,{rollouts=replayRollouts,stepsPerRollout=replaySteps,actionTics=4}={}){
  return page.evaluate(async({rollouts,stepsPerRollout,actionTics})=>{
    const {policy:p,controller:c,env}=window.__doomLab;
    c.pause();c.training=false;c.explore=false;c.memory=true;c.useResidual=true;
    await c.quiesce({teacher:true});p.setInferenceMode("neural");
    const runs=[];let totalKills=0,totalDamage=0,totalReward=0,totalPickups=0,totalDecisions=0;
    for(let r=0;r<rollouts;r++){
      await env.reset();p.resetEpisode();let kills=0,damage=0,reward=0,pickups=0;const actions=[];
      for(let i=0;i<stepsPerRollout;i++){
        const obs=env.observe(),d=await p.decide(obs,{useResidual:true,memory:true,explore:false});
        const step=env.stepTics(d.action.id,actionTics),outcome=step.info?.outcome||null;
        p.commitActionOutcome?.({actionIndex:d.actionIndex,reward:step.reward,outcome,done:step.done});
        kills+=Number(outcome?.playerKillDelta||0);damage+=Number(outcome?.damageDealt||0);
        pickups+=Number(outcome?.playerPickupDelta||0);reward+=Number(step.reward||0);totalDecisions++;
        actions.push({
          action:d.action.id,reward:Number(step.reward||0),
          kills:Number(outcome?.playerKillDelta||0),damage:Number(outcome?.damageDealt||0),
          healthDelta:Number(outcome?.healthDelta||0),pickup:Number(outcome?.playerPickupDelta||0),
          p:Number(d.probs?.[d.actionIndex]||0),entropy:Number(d.uncertainty?.entropy||0)
        });
        if(step.done)break;
      }
      const run={index:r,steps:actions.length,kills,damage,reward,pickups,actions};
      runs.push(run);totalKills+=kills;totalDamage+=damage;totalReward+=reward;totalPickups+=pickups;
    }
    return{rollouts,stepsPerRollout,actionTics,totalDecisions,totalKills,totalDamage,totalReward,totalPickups,runs};
  },{rollouts,stepsPerRollout,actionTics});
}

async function replayActions(page,runs,{actionTics=4}={}){
  return page.evaluate(async({runs,actionTics})=>{
    const {policy:p,env}=window.__doomLab;
    const replayed=[];let totalKills=0,totalDamage=0,totalReward=0;
    for(const source of runs){
      await env.reset();p.resetEpisode();let kills=0,damage=0,reward=0;const actions=[];
      for(const item of source.actions){
        const step=env.stepTics(item.action,actionTics),outcome=step.info?.outcome||null;
        const index=p.actions.findIndex(a=>a.id===item.action);
        p.commitActionOutcome?.({actionIndex:index,reward:step.reward,outcome,done:step.done});
        kills+=Number(outcome?.playerKillDelta||0);damage+=Number(outcome?.damageDealt||0);reward+=Number(step.reward||0);
        actions.push(item.action);if(step.done)break;
      }
      replayed.push({index:source.index,steps:actions.length,kills,damage,reward,actions});
      totalKills+=kills;totalDamage+=damage;totalReward+=reward;
    }
    return{totalKills,totalDamage,totalReward,runs:replayed};
  },{runs,actionTics});
}

test("causal policy curriculum trains, saves, and exactly replays combat runs",async({page})=>{
  mkdirSync(outputDir,{recursive:true});
  await bootAndPrepare(page);

  const baseline=await evaluateExact(page,{rollouts:4,stepsPerRollout:64,actionTics:4});
  const training=await page.evaluate(async steps=>{
    const {controller:c,policy:p}=window.__doomLab;
    p.setInferenceMode("neural");
    return c.trainCausalPolicy({
      steps,probeTics:24,actionTics:4,rolloutHorizon:64,
      targetTemperature:.22,priorStrength:.03,superviseSteps:3,superviseStrength:.65,
      batchRefitEvery:64,batchWindow:steps,ridge:.02
    });
  },causalSteps);

  const checkpoint=await page.evaluate(()=>window.__doomLab.policy.exportCheckpoint());
  checkpoint.build={
    selection:"exact-state causal policy curriculum",
    causalSteps,probeTics:24,actionTics:4,rolloutHorizon:64,
    training:{kills:training.kills,damage:training.damage,return:training.return,examples:training.examples,informative:training.informative}
  };
  writeFileSync(resolve(outputDir,"doom-causal-checkpoint.json"),JSON.stringify(checkpoint));

  const evaluation=await evaluateExact(page,{rollouts:replayRollouts,stepsPerRollout:replaySteps,actionTics:4});
  const replay=await replayActions(page,evaluation.runs,{actionTics:4});
  const report={version:"causal-policy-v1",killTarget,baseline,training,evaluation,replay,checkpoint:{params:checkpoint.q?.params,bytes:JSON.stringify(checkpoint).length}};
  writeFileSync(resolve(outputDir,"combat-report.json"),JSON.stringify(report,null,2));
  writeFileSync(resolve(outputDir,"combat-replays.json"),JSON.stringify({version:1,actionTics:4,runs:evaluation.runs},null,2));

  console.log("CAUSAL_COMBAT_RESULT "+JSON.stringify({
    target:killTarget,
    baseline:{kills:baseline.totalKills,damage:baseline.totalDamage,reward:baseline.totalReward},
    training:{kills:training.kills,damage:training.damage,reward:training.return,examples:training.examples,informative:training.informative,diversity:training.actionDiversity},
    evaluation:{kills:evaluation.totalKills,damage:evaluation.totalDamage,reward:evaluation.totalReward,decisions:evaluation.totalDecisions},
    replay:{kills:replay.totalKills,damage:replay.totalDamage,reward:replay.totalReward},
    params:checkpoint.q?.params
  }));

  expect(training.examples).toBe(causalSteps);
  expect(training.informative).toBeGreaterThan(causalSteps*.15);
  expect(replay.totalKills).toBe(evaluation.totalKills);
  expect(replay.totalDamage).toBe(evaluation.totalDamage);
  expect(Math.abs(replay.totalReward-evaluation.totalReward)).toBeLessThan(1e-6);
  expect(evaluation.totalKills).toBeGreaterThanOrEqual(killTarget);
});
