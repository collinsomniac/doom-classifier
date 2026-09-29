import {test,expect} from "@playwright/test";

test.setTimeout(240000);

async function bootPublished(page){
  await page.goto("http://127.0.0.1:8000/doom.html?starter=auto",{waitUntil:"domcontentloaded"});
  await page.locator("#bootBtn").click();
  await expect(page.locator("#runtimeStatus")).toContainText(/ENGINE READY|CHECKPOINT READY/,{timeout:120000});
  await expect(page.locator("#checkpointStatus")).toContainText("Validated starter",{timeout:60000});
}

test("compare greedy and seeded uncertainty-aware decoders on the published causal model",async({page})=>{
  await bootPublished(page);
  const result=await page.evaluate(async()=>{
    const {policy:p,controller:c,env}=window.__doomLab;
    const {mulberry32}=await import("/src/core/math.js");
    c.pause();c.training=false;c.explore=false;c.memory=true;c.useResidual=true;p.setInferenceMode("neural");

    const modes=[
      {id:"greedy",temperature:0,mix:0},
      {id:"sample_045",temperature:.45,mix:1},
      {id:"mix_010",temperature:.60,mix:.10},
      {id:"mix_020",temperature:.60,mix:.20},
      {id:"mix_030",temperature:.60,mix:.30},
      {id:"adaptive",temperature:.60,adaptiveMax:.35}
    ];
    const sample=(probs,temp,rng)=>{
      const power=1/Math.max(.05,temp),weights=probs.map(v=>Math.pow(Math.max(1e-12,Number(v)||0),power));
      const sum=weights.reduce((a,b)=>a+b,0)||1,target=rng()*sum;let acc=0;
      for(let i=0;i<weights.length;i++){acc+=weights[i];if(target<=acc)return i}
      return weights.length-1;
    };
    const argmax=xs=>xs.reduce((best,v,i,a)=>v>a[best]?i:best,0);
    const evaluate=async mode=>{
      const runs=[],globalCounts={};let totalKills=0,totalDamage=0,totalReward=0,totalSwitches=0,globalMaxStreak=0,totalFire=0,totalActions=0,totalSampled=0,totalNonGreedy=0;
      for(let r=0;r<8;r++){
        const rng=mulberry32((0x6d2b79f5^(r*0x9e3779b9)^(Math.round(mode.temperature*1000)<<8)^(Math.round((mode.mix||mode.adaptiveMax||0)*1000)<<16))>>>0);
        await env.reset();p.resetEpisode();let kills=0,damage=0,reward=0,last=null,streak=0,maxStreak=0,switches=0;const counts={},actions=[];
        for(let i=0;i<64;i++){
          const obs=env.observe(),d=await p.decide(obs,{useResidual:true,memory:true,explore:false}),greedy=argmax(d.probs);
          let sampleChance=Number(mode.mix||0);
          if(mode.adaptiveMax){
            const entropy=Number(d.uncertainty?.entropy||0),margin=Number(d.uncertainty?.margin||0);
            const entropyGate=Math.max(0,Math.min(1,(entropy-.65)/.35)),marginGate=Math.max(0,Math.min(1,(.15-margin)/.15));
            sampleChance=mode.adaptiveMax*entropyGate*marginGate;
          }
          let index=greedy;
          if(sampleChance>0&&rng()<sampleChance){index=sample(d.probs,mode.temperature,rng);totalSampled++}
          if(index!==greedy)totalNonGreedy++;
          const action=p.actions[index],step=env.stepTics(action.id,4),outcome=step.info?.outcome||null;
          p.commitActionOutcome?.({actionIndex:index,reward:step.reward,outcome,done:step.done});
          kills+=Number(outcome?.playerKillDelta||0);damage+=Number(outcome?.damageDealt||0);reward+=Number(step.reward||0);
          counts[action.id]=(counts[action.id]||0)+1;globalCounts[action.id]=(globalCounts[action.id]||0)+1;
          totalActions++;if(Number(action.params?.trigger||0)===1)totalFire++;
          if(action.id===last)streak++;else{if(last!==null)switches++;last=action.id;streak=1}maxStreak=Math.max(maxStreak,streak);
          actions.push(action.id);if(step.done)break;
        }
        runs.push({index:r,kills,damage,reward,diversity:Object.keys(counts).length,switches,maxStreak,counts,actions});
        totalKills+=kills;totalDamage+=damage;totalReward+=reward;totalSwitches+=switches;globalMaxStreak=Math.max(globalMaxStreak,maxStreak);
      }
      return{
        ...mode,totalKills,totalDamage,totalReward,diversity:Object.keys(globalCounts).length,
        minRunDiversity:Math.min(...runs.map(x=>x.diversity)),totalSwitches,maxStreak:globalMaxStreak,
        fireRate:totalActions?totalFire/totalActions:0,sampleRate:totalActions?totalSampled/totalActions:0,
        nonGreedyRate:totalActions?totalNonGreedy/totalActions:0,counts:globalCounts,runs
      };
    };
    const out=[];for(const mode of modes)out.push(await evaluate(mode));
    return out;
  });

  console.log("DECODE_BENCHMARK "+JSON.stringify(result.map(x=>({
    id:x.id,temperature:x.temperature,mix:x.mix,adaptiveMax:x.adaptiveMax,kills:x.totalKills,damage:x.totalDamage,reward:x.totalReward,
    diversity:x.diversity,minRunDiversity:x.minRunDiversity,switches:x.totalSwitches,maxStreak:x.maxStreak,fireRate:x.fireRate,sampleRate:x.sampleRate,nonGreedyRate:x.nonGreedyRate,counts:x.counts
  }))));
  const greedy=result.find(x=>x.id==="greedy"),mixed=result.filter(x=>x.id.startsWith("mix_")||x.id==="adaptive");
  expect(greedy?.totalKills).toBeGreaterThanOrEqual(12);
  expect(mixed.some(x=>x.totalKills>=12&&x.diversity>=4&&x.minRunDiversity>=3&&x.totalSwitches>=48&&x.maxStreak<=32)).toBe(true);
});
