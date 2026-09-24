import {DoomWasmArena,DOOM_RUNTIME_PROVENANCE} from "./env/doom-wasm.js";
import {HashSemanticAdapter} from "./core/semantic.js";
import {SemanticResidualPolicy} from "./core/policy.js";
import {ExperimentController} from "./core/controller.js";
import {TransformersNLIAdapter,NLI_PRESETS} from "./model-adapters/transformers-nli.js";

const $=id=>document.getElementById(id);
const ui={boot:$("bootBtn"),start:$("startBtn"),step:$("stepBtn"),reset:$("resetBtn"),canvas:$("doomCanvas"),runtime:$("runtimeStatus"),bootStatus:$("bootStatus"),residual:$("residualToggle"),training:$("trainingToggle"),memory:$("memoryToggle"),explore:$("exploreToggle"),actionMs:$("actionMs"),actionMsOut:$("actionMsOut"),bars:$("actionBars"),chosen:$("chosenAction"),entropy:$("entropy"),margin:$("margin"),novelty:$("novelty"),latLast:$("latLast"),latSemantic:$("latSemantic"),latP95:$("latP95"),latP99:$("latP99"),backbone:$("backboneName"),modelSelect:$("modelSelect"),loadModel:$("loadModelBtn"),modelProgress:$("modelProgress"),modelStatus:$("modelStatus"),state:$("stateTable"),objective:$("objectiveText"),steps:$("steps"),episodes:$("episodes"),ret:$("return"),updates:$("updates"),lastReward:$("lastReward"),log:$("eventLog"),export:$("exportBtn"),dot:$("statusDot")};
let env=null,policy=null,controller=null,hashSemantic=null;

function setReadyControls(ready){for(const el of [ui.start,ui.step,ui.reset,ui.loadModel,ui.export])el.disabled=!ready}
function setRuntime(text,error=false){ui.runtime.textContent=text;ui.dot.style.background=error?"#ff6b6b":"#d9ff5a"}
function buildBars(){ui.bars.innerHTML="";if(!env)return;for(const a of env.actions){const row=document.createElement("div");row.className="bar-row";const label=document.createElement("span");label.textContent=a.label;const track=document.createElement("div");track.className="bar-track";const fill=document.createElement("div");fill.className="bar-fill";track.append(fill);const value=document.createElement("span");value.textContent="0.000";row.append(label,track,value);ui.bars.append(row)}}
function render(){
  if(!env||!controller||!policy)return;
  const obs=env.lastObservation||env.observe();ui.objective.textContent=env.schema.objective;
  ui.state.innerHTML=Object.entries(obs).map(([key,value])=>'<div class="state-row"><span>'+key+'</span><strong>'+Number(value).toFixed(3)+'</strong></div>').join("");
  ui.steps.textContent=controller.steps;ui.episodes.textContent=controller.episodes;ui.ret.textContent=controller.episodeReturn.toFixed(3);ui.updates.textContent=policy.q.updates;
  ui.backbone.textContent=policy.semantic.name+(policy.semantic.backend?" · "+policy.semantic.backend:"");
  const lat=controller.latencySummary();ui.latLast.textContent=lat.last.toFixed(2)+" ms";ui.latP95.textContent=lat.p95.toFixed(2)+" ms";ui.latP99.textContent=lat.p99.toFixed(2)+" ms";
  const d=controller.lastDecision;
  if(d){ui.latSemantic.textContent=d.semanticLatencyMs.toFixed(2)+" ms";ui.lastReward.textContent=d.reward.toFixed(3);ui.chosen.textContent=d.action.label;ui.entropy.textContent=d.uncertainty.entropy.toFixed(3);ui.margin.textContent=d.uncertainty.margin.toFixed(3);ui.novelty.textContent=d.uncertainty.novelty.toFixed(3);[...ui.bars.children].forEach((row,i)=>{row.querySelector(".bar-fill").style.width=(d.probs[i]*100).toFixed(1)+"%";row.lastElementChild.textContent=d.probs[i].toFixed(3)});ui.log.textContent=controller.trace.slice(-12).reverse().map(t=>"s"+String(t.step).padStart(4,"0")+" "+t.action.padEnd(13)+" p="+Math.max(...Object.values(t.probabilities)).toFixed(3)+" r="+t.reward.toFixed(3)+" policy="+t.latencyMs.toFixed(1)+"ms").join("\n")}
}
function bindController(){controller.addEventListener("tick",render);controller.addEventListener("state",event=>{ui.start.textContent=event.detail==="RUNNING"?"Pause agent":"Start agent";setRuntime(event.detail)});controller.addEventListener("error",event=>{setRuntime("ERROR",true);ui.log.textContent="ERROR: "+event.detail.message+"\n"+ui.log.textContent})}

async function boot(){
  ui.boot.disabled=true;setRuntime("LOADING");ui.bootStatus.textContent="Fetching pinned engine + Freedoom runtime…";
  try{
    env=await DoomWasmArena.boot({canvas:ui.canvas,actionMs:Number(ui.actionMs.value),onProgress:message=>{ui.bootStatus.textContent=message}});
    hashSemantic=new HashSemanticAdapter();hashSemantic.backend="local-js";policy=new SemanticResidualPolicy({schema:env.schema,actions:env.actions,semantic:hashSemantic,seed:1993});controller=new ExperimentController({environment:env,policy,hz:8});
    controller.useResidual=ui.residual.checked;controller.training=ui.training.checked;controller.memory=ui.memory.checked;controller.explore=ui.explore.checked;
    bindController();buildBars();setReadyControls(true);setRuntime("READY");ui.bootStatus.textContent=DOOM_RUNTIME_PROVENANCE.engine+" + "+DOOM_RUNTIME_PROVENANCE.content+" · pinned "+DOOM_RUNTIME_PROVENANCE.commit.slice(0,10);ui.modelStatus.textContent="hash baseline ready";render();
  }catch(error){ui.boot.disabled=false;setRuntime("BOOT FAILED",true);ui.bootStatus.textContent=String(error?.message||error);ui.log.textContent=String(error?.stack||error)}
}
async function switchBackbone(){
  if(!policy||!controller)return;controller.pause();setReadyControls(false);ui.modelProgress.value=0;const selected=ui.modelSelect.value,previous=policy.semantic;
  try{
    if(selected==="hash"){if(previous!==hashSemantic&&previous.dispose)await previous.dispose();policy.setSemantic(hashSemantic);policy.resetLearning();await controller.reset({learning:false});ui.modelStatus.textContent="hash baseline ready · residual weights cleared";render();return}
    const preset=NLI_PRESETS[selected];ui.modelStatus.textContent="loading "+preset.label+" · "+preset.approx;
    const candidate=new TransformersNLIAdapter({preset:selected,onProgress:info=>{if(Number.isFinite(info.normalizedProgress))ui.modelProgress.value=info.normalizedProgress;ui.modelStatus.textContent=(info.status||"loading")+(info.file?" · "+info.file:"")}});
    candidate.compile(env.schema,env.actions);if(previous!==hashSemantic&&previous.dispose)await previous.dispose();policy.setSemantic(hashSemantic);await candidate.load();policy.setSemantic(candidate);policy.resetLearning();await controller.reset({learning:false});ui.modelProgress.value=100;ui.modelStatus.textContent=preset.label+" ready · "+candidate.backend+" · residual weights cleared";render();
  }catch(error){policy.setSemantic(hashSemantic);policy.resetLearning();await controller.reset({learning:false});ui.modelSelect.value="hash";ui.modelProgress.value=0;ui.modelStatus.textContent="load failed; hash restored · "+String(error?.message||error);render()}
  finally{setReadyControls(true)}
}

ui.boot.addEventListener("click",boot);ui.start.addEventListener("click",()=>controller?.state==="RUNNING"?controller.pause():controller?.start());ui.step.addEventListener("click",async()=>{if(!controller)return;if(controller.state==="RUNNING")controller.pause();await controller.tick()});ui.reset.addEventListener("click",async()=>{if(controller)await controller.reset({learning:false});render()});ui.loadModel.addEventListener("click",switchBackbone);
ui.residual.addEventListener("change",()=>{if(controller)controller.useResidual=ui.residual.checked});ui.training.addEventListener("change",()=>{if(controller)controller.training=ui.training.checked});ui.memory.addEventListener("change",()=>{if(controller)controller.memory=ui.memory.checked});ui.explore.addEventListener("change",()=>{if(controller)controller.explore=ui.explore.checked});ui.actionMs.addEventListener("input",()=>{ui.actionMsOut.value=ui.actionMs.value+" ms";env?.setActionMs(ui.actionMs.value)});
ui.export.addEventListener("click",()=>{if(!controller)return;const blob=new Blob([controller.exportTrace()],{type:"application/json"}),a=document.createElement("a");a.href=URL.createObjectURL(blob);a.download="doom-real-trace-"+Date.now()+".json";a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)});
