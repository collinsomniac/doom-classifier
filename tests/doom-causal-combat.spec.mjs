import {test,expect} from "@playwright/test";
import {mkdirSync,writeFileSync} from "node:fs";
import {resolve} from "node:path";

test.setTimeout(900000);

const causalSteps=Number(process.env.CAUSAL_TRAIN_STEPS||768);
const replayRollouts=Number(process.env.CAUSAL_REPLAY_ROLLOUTS||16);
const replaySteps=Number(process.env.CAUSAL_REPLAY_STEPS||64);
const killTarget=Number(process.env.CAUSAL_KILL_TARGET||20);
const minDiversity=Number(process.env.CAUSAL_MIN_DIVERSITY||4);
const minSwitches=Number(process.env.CAUSAL_MIN_SWITCHES||8);
const maxStreakLimit=Number(process.env.CAUSAL_MAX_STREAK||24);
const outputDir=resolve(process.env.CAUSAL_OUTPUT_DIR||"artifacts/causal-combat");

async function bootAndPrepare(page){
  await page.goto("http://127.0.0.1:8000/doom.html?starter=off",{waitUntil:"domcontentloaded"});
  await page.locator("#bootBtn").click();
  await expect(page.locator("#runtimeStatus")).toContainText("ENGINE READY",{timeout:120000});

  // Causal training needs semantic schema bindings but no runtime NLI teacher.
  // Compile MiniLM schema embeddings directly and keep the slow teacher out of
  // the offline simulator loop.
  await page.locator("#inspectTabBtn").click();
  await page.locator(".advanced-panel").evaluate(el=>{el.open=true});
  await page.locator("#schemaCompileBtn").click();
  await expect(page.locator("#schemaStatus")).toContainText("compiled",{timeout:300000});
  await page.locator("#playTabBtn").click();
}

async function evaluateExact(page,{rollouts=replayRollouts,stepsPerRollout=replaySteps,actionTics=4,verifyReplay=false}={}){
  return page.evaluate(async({rollouts,stepsPerRollout,actionTics,verifyReplay})=>{
    const {policy:p,controller:c,env}=window.__doomLab;
    c.pause();c.training=false;c.explore=false;c.memory=true;c.useResidual=true;
    await c.quiesce({teacher:true});p.setInferenceMode("neural");
    const runs=[],replayRuns=[];let totalKills=0,totalDamage=0,totalReward=0,totalPickups=0,totalDecisions=0;
    let replayKills=0,replayDamage=0,replayReward=0;
    for(let r=0;r<rollouts;r++){
      await env.reset();p.resetEpisode();
      // A DOOM savegame is a deterministic canonicalization boundary, not a
      // byte-for-byte continuation of every transient engine cache. Capture
      // and restore once before the measured run so evaluation and replay
      // begin from the same complete simulator state (including RNG cursors).
      const startSnapshot=env.saveSnapshot();
      env.restoreSnapshot(startSnapshot);p.resetEpisode();
      let kills=0,damage=0,reward=0,pickups=0;const actions=[];
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
          p:Number(d.probs?.[d.actionIndex]||0),entropy:Number(d.uncertainty?.entropy||0),
          decisionRule:d.decisionRule||"argmax"
        });
        if(step.done)break;
      }
      const counts={},sequence=actions.map(x=>x.action);let switches=0,maxStreak=0,last=null,streak=0;
      for(const id of sequence){
        counts[id]=(counts[id]||0)+1;
        if(id===last)streak++;else{if(last!==null)switches++;last=id;streak=1}
        maxStreak=Math.max(maxStreak,streak);
      }
      const run={index:r,steps:actions.length,kills,damage,reward,pickups,actions,counts,diversity:Object.keys(counts).length,switches,maxStreak};
      runs.push(run);totalKills+=kills;totalDamage+=damage;totalReward+=reward;totalPickups+=pickups;

      if(verifyReplay){
        env.restoreSnapshot(startSnapshot);p.resetEpisode();
        let rk=0,rd=0,rr=0;const replayedActions=[];
        for(const item of actions){
          const actionIndex=p.actions.findIndex(a=>a.id===item.action),step=env.stepTics(item.action,actionTics),outcome=step.info?.outcome||null;
          p.commitActionOutcome?.({actionIndex,reward:step.reward,outcome,done:step.done});
          rk+=Number(outcome?.playerKillDelta||0);rd+=Number(outcome?.damageDealt||0);rr+=Number(step.reward||0);
          replayedActions.push(item.action);if(step.done)break;
        }
        replayRuns.push({
          index:r,steps:replayedActions.length,kills:rk,damage:rd,reward:rr,actions:replayedActions,
          exact:rk===kills&&rd===damage&&Math.abs(rr-reward)<1e-9
        });
        replayKills+=rk;replayDamage+=rd;replayReward+=rr;
      }
    }
    const actionCounts={};let totalSwitches=0,maxStreak=0,minRunDiversity=Infinity;
    for(const run of runs){
      totalSwitches+=run.switches;maxStreak=Math.max(maxStreak,run.maxStreak);minRunDiversity=Math.min(minRunDiversity,run.diversity);
      for(const [id,n] of Object.entries(run.counts))actionCounts[id]=(actionCounts[id]||0)+n;
    }
    return{
      rollouts,stepsPerRollout,actionTics,totalDecisions,totalKills,totalDamage,totalReward,totalPickups,runs,
      actionCounts,actionDiversity:Object.keys(actionCounts).length,minRunDiversity:Number.isFinite(minRunDiversity)?minRunDiversity:0,
      totalSwitches,maxStreak,
      replay:verifyReplay?{
        sameStartSnapshot:true,canonicalStart:true,totalKills:replayKills,totalDamage:replayDamage,totalReward:replayReward,runs:replayRuns,
        exact:replayRuns.length===runs.length&&replayRuns.every(x=>x.exact)
      }:null
    };
  },{rollouts,stepsPerRollout,actionTics,verifyReplay});
}


async function groundingProbe(page){
  return page.evaluate(()=>{
    const {env,policy:p}=window.__doomLab,base=env.observe(),actions=p.actions;
    const clone=value=>structuredClone(value);
    const softmax=s=>{const peak=Math.max(...s),e=s.map(v=>Math.exp(v-peak)),z=e.reduce((a,b)=>a+b,0)||1;return e.map(v=>v/z)};
    const entity=(bearing,distance=192)=>{
      const rad=bearing*Math.PI;
      return{engine_record_id:999,type:11,kind:1,x:base.player_x+distance*Math.cos(rad),y:base.player_y+distance*Math.sin(rad),z:base.player_z,
        relative_x:distance*Math.cos(rad),relative_y:distance*Math.sin(rad),relative_z:0,velocity_x:0,velocity_y:0,radius:20,height:56,health:60,
        distance,relative_angle:bearing,visible:1,countkill:1,pickup:0,targeting_player:1};
    };
    const make=(bearing=null)=>{
      const obs=clone(base),has=bearing!==null,e=has?entity(bearing):null;
      Object.assign(obs,{
        visible_hostile_count:has?1:0,targeting_hostile_count:has?1:0,
        nearest_hostile_distance:has?e.distance:4096,nearest_hostile_bearing:has?bearing:0,
        nearest_visible_hostile_distance:has?e.distance:4096,nearest_visible_hostile_bearing:has?bearing:0,nearest_visible_hostile_health:has?60:0,
        visible_hostile_bearing_zone:!has?0:Math.abs(bearing)<=.06?1:bearing>0?2:3,
        visible_hostile_distance_zone:!has?0:e.distance<256?1:e.distance<768?2:3,
        aim_alignment:has?1-Math.min(1,Math.abs(bearing)*8):0,
        nearest_targeting_hostile_distance:has?e.distance:4096,nearest_targeting_hostile_bearing:has?bearing:0,
        visible_projectile_count:0,nearest_projectile_distance:4096,nearest_projectile_bearing:0,
        _collections:{entities:has?[e]:[],geometry:[]}
      });
      return obs;
    };
    const score=obs=>{
      const encoded=p.encode(obs,false,false),ev=p.residualEvaluation(obs,encoded.features,encoded.temporal,[]),probs=softmax(ev.scores.map(v=>Number(v)/Math.max(.05,Number(p.temperature)||1)));
      const mass=predicate=>actions.reduce((sum,a,i)=>sum+(predicate(a)?probs[i]:0),0);
      return{
        probs,
        top:actions.map((a,i)=>({id:a.id,p:probs[i]})).sort((a,b)=>b.p-a.p).slice(0,5),
        fire:mass(a=>Number(a.params?.trigger||0)===1),
        left:mass(a=>Number(a.params?.view||0)===1||Number(a.params?.movement||0)===3),
        right:mass(a=>Number(a.params?.view||0)===2||Number(a.params?.movement||0)===4),
        turnLeft:mass(a=>Number(a.params?.view||0)===1),
        turnRight:mass(a=>Number(a.params?.view||0)===2)
      };
    };
    // Native DOOM bridge uses relative_angle = bearing - player_angle.
    // Left-turn input increases player_angle, so positive relative_angle is left.
    const left=score(make(.24)),ahead=score(make(0)),right=score(make(-.24)),quiet=score(make(null));
    const tv=.5*left.probs.reduce((sum,p,i)=>sum+Math.abs(p-right.probs[i]),0);
    const directional=.5*((left.left-left.right)+(right.right-right.left));
    const turnDirectional=.5*((left.turnLeft-left.turnRight)+(right.turnRight-right.turnLeft));
    return{
      left:{top:left.top,left:left.left,right:left.right,fire:left.fire},
      ahead:{top:ahead.top,fire:ahead.fire},
      right:{top:right.top,left:right.left,right:right.right,fire:right.fire},
      quiet:{top:quiet.top,fire:quiet.fire},
      totalVariationLeftRight:tv,directionalPreference:directional,turnDirectionalPreference:turnDirectional,
      fireContrastAheadVsQuiet:ahead.fire-quiet.fire
    };
  });
}

async function fiveMinuteBenchmark(page,{totalTics=10500,actionTics=4}={}){
  return page.evaluate(async({totalTics,actionTics})=>{
    const {policy:p,controller:c,env}=window.__doomLab;
    c.pause();c.training=false;c.explore=false;c.memory=true;c.useResidual=true;await c.quiesce({teacher:true});p.setInferenceMode("neural");
    const started=performance.now(),actions={},bearingAction={left:{},center:{},right:{},none:{}};
    let tics=0,decisions=0,kills=0,damage=0,reward=0,deaths=0,resets=0;
    let visible=0,aligned=0,alignedFire=0,offAxis=0,correctTurn=0,noVisible=0,noVisibleFire=0;

    const canonicalReset=async()=>{
      await env.reset();p.resetEpisode();const snap=env.saveSnapshot();env.restoreSnapshot(snap);p.resetEpisode();resets++;
    };
    await canonicalReset();

    while(tics<totalTics){
      const obs=env.observe(),d=await p.decide(obs,{useResidual:true,memory:true,explore:false}),id=d.action.id,params=d.action.params||{};
      const hasVisible=Number(obs.visible_hostile_count||0)>0&&Number(obs.nearest_visible_hostile_distance||4096)<4096;
      const bearing=Number(obs.nearest_visible_hostile_bearing||0),fire=Number(params.trigger||0)===1;
      const bin=!hasVisible?"none":bearing>.08?"left":bearing<-.08?"right":"center";
      bearingAction[bin][id]=(bearingAction[bin][id]||0)+1;
      if(hasVisible){
        visible++;
        if(Math.abs(bearing)<=.06){aligned++;if(fire)alignedFire++}
        if(Math.abs(bearing)>=.10){
          offAxis++;
          if((bearing>0&&Number(params.view||0)===1)||(bearing<0&&Number(params.view||0)===2))correctTurn++;
        }
      }else{noVisible++;if(fire)noVisibleFire++}

      const step=env.stepTics(id,Math.min(actionTics,totalTics-tics)),outcome=step.info?.outcome||null;
      p.commitActionOutcome?.({actionIndex:d.actionIndex,reward:step.reward,outcome,done:step.done});
      tics+=Number(step.info?.exactTics||actionTics);decisions++;actions[id]=(actions[id]||0)+1;
      kills+=Number(outcome?.playerKillDelta||0);damage+=Number(outcome?.damageDealt||0);reward+=Number(step.reward||0);
      if(step.done){
        if(Number(step.observation?.health||0)<=0)deaths++;
        if(tics<totalTics)await canonicalReset();
      }
    }

    const stateTotals=Object.fromEntries(Object.entries(bearingAction).map(([k,row])=>[k,Object.values(row).reduce((a,b)=>a+b,0)]));
    const total=decisions||1,actionTotals=actions;
    let mi=0;
    for(const [state,row] of Object.entries(bearingAction)){
      const ps=(stateTotals[state]||0)/total;if(ps<=0)continue;
      for(const [action,n] of Object.entries(row)){
        const pxy=n/total,pa=(actionTotals[action]||0)/total;
        if(pxy>0&&pa>0)mi+=pxy*Math.log(pxy/(ps*pa));
      }
    }
    const hState=-Object.values(stateTotals).reduce((sum,n)=>{const q=n/total;return q>0?sum+q*Math.log(q):sum},0);
    return{
      simulatedSeconds:tics/35,tics,decisions,kills,damage,reward,deaths,resets,
      wallMs:performance.now()-started,actionCounts:actions,actionDiversity:Object.keys(actions).length,
      visibleDecisions:visible,alignedDecisions:aligned,alignedFireRate:aligned?alignedFire/aligned:0,
      offAxisDecisions:offAxis,correctTurnRate:offAxis?correctTurn/offAxis:0,
      noVisibleDecisions:noVisible,noVisibleFireRate:noVisible?noVisibleFire/noVisible:0,
      bearingAction,stateActionMutualInformation:mi,normalizedStateActionMI:hState>1e-9?mi/hState:0
    };
  },{totalTics,actionTics});
}

test("causal policy curriculum trains, saves, and exactly replays combat runs",async({page})=>{
  mkdirSync(outputDir,{recursive:true});
  await bootAndPrepare(page);

  const baseline=await evaluateExact(page,{rollouts:4,stepsPerRollout:64,actionTics:4});
  const training=await page.evaluate(async steps=>{
    const {controller:c,policy:p}=window.__doomLab;
    await c.reset({learning:false});
    const canonical=c.environment.saveSnapshot();c.environment.restoreSnapshot(canonical);p.resetEpisode();
    p.setInferenceMode("neural");
    return c.trainCausalPolicy({
      steps,probeTics:24,actionTics:4,plannerDepth:1,continuationTics:8,continuationCandidates:4,rolloutHorizon:128,
      targetTemperature:.28,priorStrength:0,superviseSteps:3,superviseStrength:.58,
      supervisionReplay:2,replayStrength:.20,behaviorCoverage:.55,
      batchRefitEvery:0,batchWindow:steps,finalRefit:false,ridge:.02
    });
  },causalSteps);

  const checkpoint=await page.evaluate(()=>window.__doomLab.policy.exportCheckpoint());

  const runtime=await page.evaluate(()=>({
    commit:window.__doomLab.env.runtime?.commit||null,
    sourceCommit:window.__doomLab.env.runtime?.sourceCommit||null,
    base:window.__doomLab.env.runtime?.base||null
  }));
  const evaluated=await evaluateExact(page,{rollouts:replayRollouts,stepsPerRollout:replaySteps,actionTics:4,verifyReplay:true});
  const {replay,...evaluation}=evaluated;
  const grounding=await groundingProbe(page);
  const fiveMinute=await fiveMinuteBenchmark(page,{totalTics:10500,actionTics:4});

  checkpoint.build={
    selection:"canonical exact-state typed causal curriculum",
    causalSteps,probeTics:24,actionTics:4,plannerDepth:1,continuationTics:8,continuationCandidates:4,rolloutHorizon:128,
    training:{
      kills:training.kills,damage:training.damage,return:training.return,examples:training.examples,informative:training.informative,
      meanTypedTargetFit:training.meanTypedTargetFit,meanTypedTargetBlend:training.meanTypedTargetBlend,
      behaviorCoverage:training.behaviorCoverage,coverageActions:training.coverageActions,
      behaviorDiversity:training.behaviorDiversity,behaviorActionCounts:training.behaviorActionCounts
    },
    validation:{kills:evaluation.totalKills,damage:evaluation.totalDamage,reward:evaluation.totalReward},
    grounding,
    fiveMinute:{kills:fiveMinute.kills,damage:fiveMinute.damage,simulatedSeconds:fiveMinute.simulatedSeconds,normalizedStateActionMI:fiveMinute.normalizedStateActionMI}
  };
  writeFileSync(resolve(outputDir,"doom-causal-checkpoint.json"),JSON.stringify(checkpoint));

  const report={
    version:"causal-policy-v6-factorized-prior-free",
    gate:{killTarget,minDiversity,minSwitches,maxStreakLimit},
    runtime,baseline,training,evaluation,replay,grounding,fiveMinute,
    checkpoint:{params:checkpoint.q?.params,bytes:JSON.stringify(checkpoint).length}
  };
  writeFileSync(resolve(outputDir,"combat-report.json"),JSON.stringify(report,null,2));
  writeFileSync(resolve(outputDir,"combat-replays.json"),JSON.stringify({version:4,actionTics:4,verification:"same canonical native start snapshot with restored RNG state",runs:evaluation.runs},null,2));

  console.log("CAUSAL_COMBAT_RESULT "+JSON.stringify({
    target:killTarget,runtime,
    baseline:{kills:baseline.totalKills,damage:baseline.totalDamage,reward:baseline.totalReward,diversity:baseline.actionDiversity,maxStreak:baseline.maxStreak},
    training:{kills:training.kills,damage:training.damage,reward:training.return,examples:training.examples,informative:training.informative,diversity:training.actionDiversity,replaySupervisionUpdates:training.replaySupervisionUpdates,fitPasses:training.fitPasses,typedFit:training.meanTypedTargetFit,typedBlend:training.meanTypedTargetBlend,behaviorCoverage:training.behaviorCoverage,coverageActions:training.coverageActions,behaviorDiversity:training.behaviorDiversity,behaviorCounts:training.behaviorActionCounts},
    evaluation:{kills:evaluation.totalKills,damage:evaluation.totalDamage,reward:evaluation.totalReward,decisions:evaluation.totalDecisions,diversity:evaluation.actionDiversity,minRunDiversity:evaluation.minRunDiversity,switches:evaluation.totalSwitches,maxStreak:evaluation.maxStreak,counts:evaluation.actionCounts},
    replay:{kills:replay.totalKills,damage:replay.totalDamage,reward:replay.totalReward},
    grounding,
    fiveMinute:{kills:fiveMinute.kills,damage:fiveMinute.damage,reward:fiveMinute.reward,deaths:fiveMinute.deaths,decisions:fiveMinute.decisions,wallMs:fiveMinute.wallMs,diversity:fiveMinute.actionDiversity,alignedFireRate:fiveMinute.alignedFireRate,correctTurnRate:fiveMinute.correctTurnRate,noVisibleFireRate:fiveMinute.noVisibleFireRate,normalizedStateActionMI:fiveMinute.normalizedStateActionMI,counts:fiveMinute.actionCounts},
    params:checkpoint.q?.params
  }));

  expect(training.examples).toBe(training.informative);
  expect(training.informative).toBeGreaterThan(causalSteps*.55);
  expect(replay.sameStartSnapshot).toBe(true);
  expect(replay.exact).toBe(true);
  expect(replay.totalKills).toBe(evaluation.totalKills);
  expect(replay.totalDamage).toBe(evaluation.totalDamage);
  expect(Math.abs(replay.totalReward-evaluation.totalReward)).toBeLessThan(1e-6);
  expect(evaluation.totalKills).toBeGreaterThanOrEqual(killTarget);
  expect(evaluation.actionDiversity).toBeGreaterThanOrEqual(minDiversity);
  expect(evaluation.minRunDiversity).toBeGreaterThanOrEqual(minDiversity);
  expect(evaluation.totalSwitches).toBeGreaterThanOrEqual(minSwitches*replayRollouts);
  expect(evaluation.maxStreak).toBeLessThanOrEqual(maxStreakLimit);
});
