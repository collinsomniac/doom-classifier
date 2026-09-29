import {test,expect} from "@playwright/test";

test.setTimeout(180000);

const sequence=["back_fire","back_fire","back_fire","back_fire","back_fire","back_fire","forward","back_fire","back_fire","back_fire","back_fire","forward_fire","back_fire","turn_right_fire","strafe_left_fire","forward_fire","strafe_left_fire","forward_fire","forward_fire","forward_fire","forward","forward_fire","forward_fire","forward_fire","forward_fire","forward_fire","forward_fire","forward_fire","forward_fire","forward_fire","forward_fire","forward_fire","forward_fire","back_fire","forward_fire","forward_fire","forward_fire","forward_fire","back","turn_right_fire","forward_fire","strafe_left_fire","back","forward_fire","forward_fire","fire","strafe_right_fire","forward_fire","forward_fire","forward_fire","forward_fire","strafe_left_fire","forward_fire","forward_fire","back_fire","forward_fire","forward_fire","forward_fire","back","forward_fire","forward_fire","turn_right","forward_fire","forward_fire"];

test("long mixed exact-tic sequence is identical after snapshot restore",async({page})=>{
  await page.goto("http://127.0.0.1:8000/doom.html?starter=off",{waitUntil:"domcontentloaded"});
  const result=await page.evaluate(async sequence=>{
    const {DoomWasmArena}=await import("/src/env/doom-wasm.js");
    const canvas=document.createElement("canvas");canvas.width=640;canvas.height=480;canvas.style.display="none";document.body.appendChild(canvas);
    const base="https://raw.githubusercontent.com/collinsomniac/doom-classifier/engine-runtime";
    const env=await DoomWasmArena.boot({canvas,runtimeBase:base,runtimeInfo:{owned:true,base,source:"long-snapshot-regression"},actionMs:110});
    await env.reset();
    const snap=env.saveSnapshot();
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
    const run=()=>{
      env.restoreSnapshot(snap);
      const rows=[];
      for(let i=0;i<sequence.length;i++){
        const before=simplify(env.readRaw()),step=env.stepTics(sequence[i],4),after=simplify(env.readRaw());
        rows.push({i,action:sequence[i],before,after,reward:Number(step.reward||0),outcome:step.info?.outcome||null});
      }
      return rows;
    };
    const a=run(),b=run();
    let first=-1;
    for(let i=0;i<a.length;i++)if(JSON.stringify(a[i])!==JSON.stringify(b[i])){first=i;break}
    const diff=first<0?null:{
      first,action:sequence[first],
      a:a[first],b:b[first],
      previous:first>0?{a:a[first-1],b:b[first-1]}:null
    };
    return{first,diff,length:sequence.length};
  },sequence);

  console.log("SNAPSHOT_DIVERGENCE "+JSON.stringify(result));
  expect(result.first).toBe(-1);
});
