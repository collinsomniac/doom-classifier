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
      const teacherBefore=p.teacherCalls,counts={},semanticCounts={},valueCounts={};let reward=0,hostileHpLoss=0,received=0,kills=0,maxP=0,entropy=0,fire=0,switches=0,maxStreak=0,lastAction=null,streak=0,chosenSemantic=0,chosenValue=0,valueBeta=0,priorKL=0,valueTrust=0,klUtilization=0,betaSaturated=0;
      for(let i=0;i<steps;i++){
        await c.tick();const d=c.lastDecision;if(!d)continue;
        counts[d.action.id]=(counts[d.action.id]||0)+1;
        if(d.action.id===lastAction)streak++;else{if(lastAction!==null)switches++;streak=1;lastAction=d.action.id}maxStreak=Math.max(maxStreak,streak);
        reward+=d.reward||0;hostileHpLoss+=d.outcome?.hostileHpLoss??d.outcome?.damageDealt??0;received+=Math.max(0,-(d.outcome?.healthDelta||0));kills+=d.outcome?.killDelta||0;
        maxP+=Math.max(...d.probs);entropy+=d.uncertainty.entropy;if(d.action.id.includes("fire"))fire++;
        chosenSemantic+=Number(d.semanticPriorScores?.[d.actionIndex]||0);chosenValue+=Number(d.valueScores?.[d.actionIndex]||0);valueBeta+=Number(d.valueBeta||0);priorKL+=Number(d.priorKL||0);valueTrust+=Number(d.valueTrust||0);klUtilization+=Number(d.klUtilization||0);if(d.valueBetaSaturated)betaSaturated++;
        if(d.semanticPriorScores){const si=d.semanticPriorScores.reduce((best,v,i,a)=>v>a[best]?i:best,0),id=p.actions[si].id;semanticCounts[id]=(semanticCounts[id]||0)+1}
        if(d.valueScores){const vi=d.valueScores.reduce((best,v,i,a)=>v>a[best]?i:best,0),id=p.actions[vi].id;valueCounts[id]=(valueCounts[id]||0)+1}
      }
      const lat=c.latencySummary(),dominant=Object.entries(counts).sort((a,b)=>b[1]-a[1])[0]||["none",0],semanticDominant=Object.entries(semanticCounts).sort((a,b)=>b[1]-a[1])[0]||["none",0],valueDominant=Object.entries(valueCounts).sort((a,b)=>b[1]-a[1])[0]||["none",0];
      return{steps,reward,hostileHpLoss,received,kills,fire,diversity:Object.keys(counts).length,switches,maxStreak,dominant,semanticDominant,valueDominant,counts,semanticCounts,valueCounts,meanChosenSemantic:chosenSemantic/steps,meanChosenValue:chosenValue/steps,meanValueBeta:valueBeta/steps,meanPriorKL:priorKL/steps,meanValueTrust:valueTrust/steps,meanKLUtilization:klUtilization/steps,betaSaturated,betaSaturationRate:betaSaturated/steps,meanMaxP:maxP/steps,meanEntropy:entropy/steps,p95Ms:lat.p95,teacherCalls:p.teacherCalls-teacherBefore,temperature:p.temperature};
    };
    const distribution=async()=>{
      await c.reset({learning:false});p.resetEpisode();
      const obs=lab.env.observe(),labels=p.actions.map(a=>a.id);
      const teacherDecision=await p.decide(obs,{useResidual:false,memory:false,explore:false});
      p.resetEpisode();
      const neuralDecision=await p.decide(obs,{useResidual:true,memory:false,explore:false});
      const rank=probs=>labels.map((id,i)=>({id,p:probs[i]})).sort((a,b)=>b.p-a.p);
      return{teacher:rank(teacherDecision.probs).slice(0,8),neural:rank(neuralDecision.probs).slice(0,8),temperature:p.temperature,teacherState:p.semantic.stateText?.(obs)?.slice(0,3500)||null};
    };
    const initialDistribution=await distribution();
    const before=await evaluate(24);
    await c.reset({learning:false});c.training=true;c.explore=true;c.memory=true;c.useResidual=true;p.setInferenceMode("adaptive");
    const updatesBefore=p.q.updates,distillBefore=p.q.distillUpdates||0,teacherBefore=p.teacherCalls,traceStart=c.trace.length;
    const train=await c.trainBurst({steps:64,epsilon:.16});
    const trainingTrace=c.trace.slice(traceStart),trainingCounts={},rewardByAction={};let trainingSwitches=0,trainingMaxStreak=0,trainingLast=null,trainingStreak=0;
    for(const t of trainingTrace){
      trainingCounts[t.action]=(trainingCounts[t.action]||0)+1;rewardByAction[t.action]=(rewardByAction[t.action]||0)+t.reward;
      if(t.action===trainingLast)trainingStreak++;else{if(trainingLast!==null)trainingSwitches++;trainingStreak=1;trainingLast=t.action}trainingMaxStreak=Math.max(trainingMaxStreak,trainingStreak);
    }
    const training={...train,neuralUpdates:p.q.updates-updatesBefore,semanticDistillUpdates:(p.q.distillUpdates||0)-distillBefore,teacherCalls:p.teacherCalls-teacherBefore,replaySize:p.replay?.length||0,teacherReplaySize:p.teacherReplay?.length||0,temperature:p.temperature,counts:trainingCounts,rewardByAction,switches:trainingSwitches,maxStreak:trainingMaxStreak,explorationStrategies:[...new Set(trainingTrace.map(t=>t.explorationStrategy))]};
    const trainedDistribution=await distribution();
    const after=await evaluate(24);
    return{version:"attributed-reward-nstep4-kl-64",initialDistribution,before,training,trainedDistribution,after,params:p.q.parameterCount(),model:p.q.name,targetSyncs:p.q.targetSyncs??0,splitHeads:typeof p.q.valueScoresObservation==="function"};
  });

  console.log("DOOM_LEARNING_BENCHMARK "+JSON.stringify(result));
  expect(result.params).toBeGreaterThan(0);expect(result.params).toBeLessThan(8000);expect(result.splitHeads).toBe(true);expect(result.model).toContain("SemanticValue");
  expect(result.before.teacherCalls).toBe(0);
  expect(result.after.teacherCalls).toBe(0);
  expect(result.training.neuralUpdates).toBeGreaterThan(64);
  expect(result.training.replaySize).toBeGreaterThan(0);
  expect(result.before.steps).toBe(24);expect(result.after.steps).toBe(24);
  if(errors.length)throw new Error(errors.join(" | "));
});
