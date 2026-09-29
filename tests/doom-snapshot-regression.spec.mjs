import {test,expect} from "@playwright/test";

test.setTimeout(180000);

const genericSequence=["back_fire","back_fire","back_fire","back_fire","back_fire","back_fire","forward","back_fire","back_fire","back_fire","back_fire","forward_fire","back_fire","turn_right_fire","strafe_left_fire","forward_fire","strafe_left_fire","forward_fire","forward_fire","forward_fire","forward","forward_fire","forward_fire","forward_fire","forward_fire","forward_fire","forward_fire","forward_fire","forward_fire","forward_fire","forward_fire","forward_fire","forward_fire","back_fire","forward_fire","forward_fire","forward_fire","forward_fire","back","turn_right_fire","forward_fire","strafe_left_fire","back","forward_fire","forward_fire","fire","strafe_right_fire","forward_fire","forward_fire","forward_fire","forward_fire","strafe_left_fire","forward_fire","forward_fire","back_fire","forward_fire","forward_fire","forward_fire","back","forward_fire","forward_fire","turn_right","forward_fire","forward_fire"];

const combatSequence=["strafe_left_fire","strafe_left_fire","strafe_left_fire","strafe_left_fire","back_fire","strafe_left_fire","strafe_left_fire","strafe_left_fire","strafe_left_fire","turn_left_fire","strafe_left_fire","strafe_left_fire","strafe_left_fire","strafe_left_fire","strafe_left_fire","strafe_right_fire","strafe_left_fire","strafe_left_fire","strafe_left_fire","strafe_left_fire","strafe_left_fire","strafe_left_fire","strafe_left_fire","strafe_left_fire","strafe_left_fire","strafe_left_fire","strafe_right_fire","strafe_left_fire","strafe_left_fire","strafe_left","strafe_left_fire","strafe_left_fire","strafe_left_fire","strafe_left_fire","forward_fire","strafe_left_fire","strafe_left_fire","strafe_left_fire","strafe_left_fire","forward_fire","strafe_left_fire","strafe_left_fire","strafe_left_fire","strafe_left_fire","strafe_left","forward_fire","strafe_left_fire","strafe_left_fire","strafe_left_fire","strafe_left_fire","strafe_right_fire","strafe_left_fire","strafe_left_fire","strafe_right_fire","strafe_left_fire","strafe_left_fire","strafe_left_fire","strafe_left_fire","strafe_left_fire","strafe_left_fire","strafe_left_fire","strafe_left_fire","strafe_left_fire","strafe_left_fire"];

test("generic and combat exact-tic sequences are identical after snapshot restore",async({page})=>{
  await page.goto("http://127.0.0.1:8000/doom.html?starter=off",{waitUntil:"domcontentloaded"});
  const result=await page.evaluate(async({genericSequence,combatSequence})=>{
    const {DoomWasmArena}=await import("/src/env/doom-wasm.js");
    const canvas=document.createElement("canvas");canvas.width=640;canvas.height=480;canvas.style.display="none";document.body.appendChild(canvas);
    const pointer=await fetch("/runtime/engine-runtime.json",{cache:"no-store"}).then(r=>r.json());
    const commit=String(pointer.asset_commit||"");
    if(!/^[0-9a-f]{40}$/i.test(commit))throw new Error("invalid engine runtime pointer");
    const base="https://raw.githubusercontent.com/collinsomniac/doom-classifier/"+commit;
    const env=await DoomWasmArena.boot({canvas,runtimeBase:base,runtimeInfo:{owned:true,base,commit,sourceCommit:pointer.source_commit,source:"long-snapshot-regression"},actionMs:110});
    const simplify=raw=>({
      player:{
        health:Number(raw.player?.health||0),armor:Number(raw.player?.armor||0),weapon:Number(raw.player?.weapon||0),
        bullets:Number(raw.player?.ammo?.bullets||0),shells:Number(raw.player?.ammo?.shells||0),rockets:Number(raw.player?.ammo?.rockets||0),cells:Number(raw.player?.ammo?.cells||0),
        x:Number(raw.player?.x||0),y:Number(raw.player?.y||0),z:Number(raw.player?.z||0),vx:Number(raw.player?.vx||0),vy:Number(raw.player?.vy||0),angle:Number(raw.player?.angle||0),kills:Number(raw.player?.kills||0)
      },
      engine:{gametic:Number(raw.engine_state?.gametic||0),gamestate:Number(raw.engine_state?.gamestate||0),paused:!!raw.engine_state?.paused,controls:Number(raw.engine_state?.controls||0)},
      events:{...(raw.events||{})},
      enemies:(raw.world?.entities||[]).filter(e=>e.enemy).map(e=>({
        id:Number(e.id||0),type:Number(e.type||0),x:Number(e.x||0),y:Number(e.y||0),z:Number(e.z||0),
        vx:Number(e.vx||0),vy:Number(e.vy||0),health:Number(e.health||0),angle:Number(e.angle||0),
        target:!!e.targeting_player,visible:!!e.visible
      })).sort((a,b)=>a.id-b.id)
    });
    const check=async(sequence,{precondition=false}={})=>{
      await env.reset();

      // The causal trainer performs hundreds of branch/restore cycles before
      // frozen evaluation. Reproduce that hidden-state pressure here rather
      // than validating only a pristine fresh boot.
      if(precondition){
        const branchActions=["forward_fire","strafe_left_fire","back_fire","turn_left_fire","strafe_right_fire","fire"];
        for(let i=0;i<72;i++){
          const branch=env.saveSnapshot();
          env.stepTics(branchActions[i%branchActions.length],24);
          env.restoreSnapshot(branch);
        }
      }

      const snap=env.saveSnapshot();
      const run=()=>{
        env.restoreSnapshot(snap);
        const rows=[];
        for(let i=0;i<sequence.length;i++){
          const before=simplify(env.readRaw()),step=env.stepTics(sequence[i],4),after=simplify(env.readRaw());
          rows.push({i,action:sequence[i],before,after,reward:Number(step.reward||0),outcome:step.info?.outcome||null});
        }
        return rows;
      };
      const a=run(),b=run();let first=-1;
      for(let i=0;i<a.length;i++)if(JSON.stringify(a[i])!==JSON.stringify(b[i])){first=i;break}
      return{
        first,length:sequence.length,
        diff:first<0?null:{first,action:sequence[first],a:a[first],b:b[first],previous:first>0?{a:a[first-1],b:b[first-1]}:null}
      };
    };

    const checkLiveVsRestored=async(sequence,{precondition=false,canonicalizeFirst=false,observeBeforeAction=true}={})=>{
      await env.reset();
      if(canonicalizeFirst){
        const canonical=env.saveSnapshot();
        env.restoreSnapshot(canonical);
      }
      if(precondition){
        const branchActions=["forward_fire","strafe_left_fire","back_fire","turn_left_fire","strafe_right_fire","fire"];
        for(let i=0;i<72;i++){
          const branch=env.saveSnapshot();
          env.stepTics(branchActions[i%branchActions.length],24);
          env.restoreSnapshot(branch);
        }
      }

      const snap=env.saveSnapshot();
      const startLive=simplify(env.readRaw());
      const live=[];
      for(let i=0;i<sequence.length;i++){
        if(observeBeforeAction)env.observe();
        const before=simplify(env.readRaw()),step=env.stepTics(sequence[i],4),after=simplify(env.readRaw());
        live.push({i,action:sequence[i],before,after,reward:Number(step.reward||0),outcome:step.info?.outcome||null});
      }

      env.restoreSnapshot(snap);
      const startRestored=simplify(env.readRaw());
      const replay=[];
      for(let i=0;i<sequence.length;i++){
        if(observeBeforeAction)env.observe();
        const before=simplify(env.readRaw()),step=env.stepTics(sequence[i],4),after=simplify(env.readRaw());
        replay.push({i,action:sequence[i],before,after,reward:Number(step.reward||0),outcome:step.info?.outcome||null});
      }

      let first=-1;
      for(let i=0;i<live.length;i++){
        if(JSON.stringify(live[i])!==JSON.stringify(replay[i])){first=i;break}
      }
      return{
        first,length:sequence.length,startEqual:JSON.stringify(startLive)===JSON.stringify(startRestored),
        startLive,startRestored,
        diff:first<0?null:{first,action:sequence[first],live:live[first],restored:replay[first],previous:first>0?{live:live[first-1],restored:replay[first-1]}:null}
      };
    };

    const checkObservationParity=async sequence=>{
      await env.reset();
      const snap=env.saveSnapshot();

      const run=withPolicyObservation=>{
        env.restoreSnapshot(snap);
        const rows=[];
        for(let i=0;i<sequence.length;i++){
          if(withPolicyObservation)env.observe();
          const step=env.stepTics(sequence[i],4),after=simplify(env.readRaw());
          rows.push({i,action:sequence[i],after,reward:Number(step.reward||0),outcome:step.info?.outcome||null});
        }
        return rows;
      };

      const observed=run(true),actionOnly=run(false);let first=-1;
      for(let i=0;i<observed.length;i++){
        if(JSON.stringify(observed[i])!==JSON.stringify(actionOnly[i])){first=i;break}
      }
      return{
        first,length:sequence.length,
        diff:first<0?null:{first,action:sequence[first],observed:observed[first],actionOnly:actionOnly[first],previous:first>0?{observed:observed[first-1],actionOnly:actionOnly[first-1]}:null}
      };
    };

    return{
      generic:await check(genericSequence),
      combat:await check(combatSequence),
      combatAfterForkPressure:await check(combatSequence,{precondition:true}),
      observationParity:await checkObservationParity(combatSequence),
      // Fresh live state versus a savegame restore is diagnostic only:
      // classic DOOM save/load canonicalizes transient state. The actual
      // determinism contract begins after a canonical restore boundary.
      freshLiveVsRestored:await checkLiveVsRestored(combatSequence),
      canonicalLiveVsRestored:await checkLiveVsRestored(combatSequence,{canonicalizeFirst:true}),
      liveVsRestoredAfterForkPressure:await checkLiveVsRestored(combatSequence,{precondition:true}),
      runtime:env.runtime
    };
  },{genericSequence,combatSequence});

  console.log("SNAPSHOT_DIVERGENCE "+JSON.stringify(result));
  expect(result.generic.first).toBe(-1);
  expect(result.combat.first).toBe(-1);
  expect(result.combatAfterForkPressure.first).toBe(-1);
  expect(result.observationParity.first).toBe(-1);
  expect(result.canonicalLiveVsRestored.first).toBe(-1);
  expect(result.liveVsRestoredAfterForkPressure.first).toBe(-1);
});
