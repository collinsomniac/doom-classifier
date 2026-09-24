import {MockArena} from "./env/mock-arena.js";
import {HashSemanticAdapter} from "./core/semantic.js";
import {SemanticResidualPolicy} from "./core/policy.js";
import {ExperimentController} from "./core/controller.js";
import {TransformersNLIAdapter,NLI_PRESETS} from "./model-adapters/transformers-nli.js";

const env=new MockArena(1337);
const hashSemantic=new HashSemanticAdapter();
hashSemantic.backend="local-js";
const policy=new SemanticResidualPolicy({schema:env.schema,actions:env.actions,semantic:hashSemantic});
const controller=new ExperimentController({environment:env,policy,hz:20});

const $=id=>document.getElementById(id);
const ui={
  start:$("startBtn"),step:$("stepBtn"),reset:$("resetBtn"),residual:$("residualToggle"),training:$("trainingToggle"),memory:$("memoryToggle"),explore:$("exploreToggle"),
  rate:$("rateInput"),rateOut:$("rateOut"),fsm:$("fsmState"),dot:$("statusDot"),bars:$("actionBars"),chosen:$("chosenAction"),
  entropy:$("entropy"),margin:$("margin"),novelty:$("novelty"),latLast:$("latLast"),latSemantic:$("latSemantic"),latP95:$("latP95"),latP99:$("latP99"),
  steps:$("steps"),episodes:$("episodes"),ret:$("return"),updates:$("updates"),state:$("stateTable"),objective:$("objectiveText"),log:$("eventLog"),export:$("exportBtn"),
  arena:$("arena"),chart:$("rewardChart"),backbone:$("backboneName"),modelSelect:$("modelSelect"),loadModel:$("loadModelBtn"),modelProgress:$("modelProgress"),modelStatus:$("modelStatus")
};

function setControlsDisabled(disabled){for(const el of [ui.start,ui.step,ui.reset,ui.loadModel,ui.modelSelect])el.disabled=disabled}

function buildBars(){
  ui.bars.innerHTML="";
  for(const a of env.actions){
    const row=document.createElement("div");row.className="bar-row";row.dataset.action=a.id;
    const label=document.createElement("span");label.textContent=a.label;
    const track=document.createElement("div");track.className="bar-track";const fill=document.createElement("div");fill.className="bar-fill";track.append(fill);
    const val=document.createElement("span");val.textContent="0.000";row.append(label,track,val);ui.bars.append(row);
  }
}
buildBars();ui.objective.textContent=env.schema.objective;

function renderArena(){
  const c=ui.arena,ctx=c.getContext("2d"),w=c.width,h=c.height;ctx.clearRect(0,0,w,h);ctx.fillStyle="#090d12";ctx.fillRect(0,0,w,h);ctx.strokeStyle="#1d2630";ctx.lineWidth=1;
  for(let x=0;x<w;x+=40){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,h);ctx.stroke()}for(let y=0;y<h;y+=40){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(w,y);ctx.stroke()}
  ctx.fillStyle="#26313a";ctx.fillRect((.48-.16)*w,(.58-.11)*h,.32*w,.22*h);
  ctx.beginPath();ctx.arc(env.goal.x*w,env.goal.y*h,12,0,Math.PI*2);ctx.fillStyle="#66e3ff";ctx.fill();
  ctx.beginPath();ctx.arc(env.enemy.x*w,env.enemy.y*h,13,0,Math.PI*2);ctx.fillStyle="#ff6b6b";ctx.fill();
  const px=env.player.x*w,py=env.player.y*h;ctx.save();ctx.translate(px,py);ctx.rotate(env.player.angle);
  ctx.beginPath();ctx.moveTo(18,0);ctx.lineTo(-11,-10);ctx.lineTo(-6,0);ctx.lineTo(-11,10);ctx.closePath();ctx.fillStyle="#d9ff5a";ctx.fill();ctx.restore();
}
function renderChart(){
  const c=ui.chart,ctx=c.getContext("2d"),xs=controller.returns.slice(-60);ctx.clearRect(0,0,c.width,c.height);ctx.strokeStyle="#303946";ctx.beginPath();ctx.moveTo(0,c.height/2);ctx.lineTo(c.width,c.height/2);ctx.stroke();
  if(xs.length<2)return;const min=Math.min(-1,...xs),max=Math.max(1,...xs);ctx.strokeStyle="#d9ff5a";ctx.lineWidth=2;ctx.beginPath();xs.forEach((v,i)=>{const x=i/(xs.length-1)*c.width,y=c.height-(v-min)/(max-min)*c.height;i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.stroke();
}
function render(){
  renderArena();renderChart();const obs=env.observe();
  ui.state.innerHTML=Object.entries(obs).map(([k,v])=>'<div class="state-row"><span>'+k+'</span><strong>'+Number(v).toFixed(3)+'</strong></div>').join("");
  ui.steps.textContent=controller.steps;ui.episodes.textContent=controller.episodes;ui.ret.textContent=controller.episodeReturn.toFixed(2);ui.updates.textContent=policy.q.updates;
  ui.backbone.textContent=policy.semantic.name+(policy.semantic.backend?" · "+policy.semantic.backend:"");
  const lat=controller.latencySummary();ui.latLast.textContent=lat.last.toFixed(2)+" ms";ui.latP95.textContent=lat.p95.toFixed(2)+" ms";ui.latP99.textContent=lat.p99.toFixed(2)+" ms";
  const d=controller.lastDecision;
  if(d){
    ui.latSemantic.textContent=d.semanticLatencyMs.toFixed(2)+" ms";ui.chosen.textContent=d.action.label;ui.entropy.textContent=d.uncertainty.entropy.toFixed(3);ui.margin.textContent=d.uncertainty.margin.toFixed(3);ui.novelty.textContent=d.uncertainty.novelty.toFixed(3);
    [...ui.bars.children].forEach((row,i)=>{row.querySelector(".bar-fill").style.width=(d.probs[i]*100).toFixed(1)+"%";row.lastElementChild.textContent=d.probs[i].toFixed(3)});
    ui.log.textContent=controller.trace.slice(-10).reverse().map(t=>"s"+String(t.step).padStart(4,"0")+"  "+t.action.padEnd(13)+" p="+Math.max(...Object.values(t.probabilities)).toFixed(3)+" r="+t.reward.toFixed(3)+"  "+t.latencyMs.toFixed(2)+"ms").join("\n");
  }
}

async function switchBackbone(){
  controller.pause();setControlsDisabled(true);ui.modelProgress.value=0;
  const selected=ui.modelSelect.value;
  const previous=policy.semantic;
  try{
    if(selected==="hash"){
      if(previous!==hashSemantic&&previous.dispose)await previous.dispose();
      policy.setSemantic(hashSemantic);policy.resetLearning();controller.reset({learning:false});
      ui.modelStatus.textContent="hash baseline ready · residual weights cleared";
      render();return;
    }

    const preset=NLI_PRESETS[selected];
    ui.modelStatus.textContent="loading "+preset.label+" · "+preset.approx;
    const candidate=new TransformersNLIAdapter({
      preset:selected,
      onProgress:info=>{
        if(Number.isFinite(info.normalizedProgress))ui.modelProgress.value=info.normalizedProgress;
        const file=info.file?" · "+info.file:"";
        ui.modelStatus.textContent=(info.status||"loading")+file;
      }
    });
    candidate.compile(env.schema,env.actions);
    if(previous!==hashSemantic&&previous.dispose)await previous.dispose();
    policy.setSemantic(hashSemantic);
    await candidate.load();
    policy.setSemantic(candidate);policy.resetLearning();controller.reset({learning:false});
    ui.modelProgress.value=100;ui.modelStatus.textContent=preset.label+" ready · "+candidate.backend+" · residual weights cleared";
    render();
  }catch(error){
    policy.setSemantic(hashSemantic);policy.resetLearning();controller.reset({learning:false});
    ui.modelProgress.value=0;ui.modelSelect.value="hash";ui.modelStatus.textContent="model load failed; hash restored · "+String(error?.message||error);
    render();
  }finally{setControlsDisabled(false)}
}

controller.addEventListener("tick",render);
controller.addEventListener("state",e=>{ui.fsm.textContent=e.detail;ui.start.textContent=e.detail==="RUNNING"?"Pause":"Start";ui.dot.style.background=e.detail==="ERROR"?"#ff6b6b":"#d9ff5a"});
controller.addEventListener("error",e=>{ui.log.textContent="ERROR: "+e.detail.message+"\n"+ui.log.textContent});

ui.start.addEventListener("click",()=>controller.state==="RUNNING"?controller.pause():controller.start());
ui.step.addEventListener("click",async()=>{if(controller.state==="RUNNING")controller.pause();await controller.tick()});
ui.reset.addEventListener("click",()=>controller.reset({learning:confirm("Reset learned residual weights too? OK = yes, Cancel = keep learning.")}));
ui.residual.addEventListener("change",()=>controller.useResidual=ui.residual.checked);ui.training.addEventListener("change",()=>controller.training=ui.training.checked);ui.memory.addEventListener("change",()=>controller.memory=ui.memory.checked);ui.explore.addEventListener("change",()=>controller.explore=ui.explore.checked);
ui.rate.addEventListener("input",()=>{ui.rateOut.value=ui.rate.value;controller.setHz(Number(ui.rate.value))});ui.loadModel.addEventListener("click",switchBackbone);
ui.export.addEventListener("click",()=>{const blob=new Blob([controller.exportTrace()],{type:"application/json"}),a=document.createElement("a");a.href=URL.createObjectURL(blob);a.download="doom-classifier-trace-"+Date.now()+".json";a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)});
render();
