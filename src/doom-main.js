import {DoomWasmArena,DOOM_RUNTIME_PROVENANCE} from "./env/doom-wasm.js";
import {HashSemanticAdapter} from "./core/semantic.js";
import {SemanticResidualPolicy} from "./core/policy.js";
import {ExperimentController} from "./core/controller.js";
import {TransformersNLIAdapter,NLI_PRESETS} from "./model-adapters/transformers-nli.js";

const $=id=>document.getElementById(id);
const ui={boot:$("bootBtn"),start:$("startBtn"),step:$("stepBtn"),reset:$("resetBtn"),canvas:$("doomCanvas"),runtime:$("runtimeStatus"),bootStatus:$("bootStatus"),path:$("pathSelect"),residual:$("residualToggle"),training:$("trainingToggle"),memory:$("memoryToggle"),explore:$("exploreToggle"),actionMs:$("actionMs"),actionMsOut:$("actionMsOut"),bars:$("actionBars"),chosen:$("chosenAction"),entropy:$("entropy"),margin:$("margin"),novelty:$("novelty"),latLast:$("latLast"),latSemantic:$("latSemantic"),latP95:$("latP95"),attention:$("attentionList"),attentionCount:$("attentionCount"),teacherCalls:$("teacherCalls"),backbone:$("backboneName"),modelSelect:$("modelSelect"),loadModel:$("loadModelBtn"),modelProgress:$("modelProgress"),modelStatus:$("modelStatus"),state:$("stateTable"),objective:$("objectiveText"),steps:$("steps"),episodes:$("episodes"),ret:$("return"),updates:$("updates"),lastReward:$("lastReward"),log:$("eventLog"),export:$("exportBtn"),dot:$("statusDot")};
let env=null,policy=null,controller=null,hashSemantic=null;

function setReadyControls(ready){for(const el of [ui.start,ui.step,ui.reset,ui.loadModel,ui.export,ui.path])el.disabled=!ready}
function setRuntime(text,error=false){ui.runtime.textContent=text;ui.dot.style.background=error?"#ff6b6b":"#d9ff5a"}
function buildBars(){ui.bars.innerHTML="";if(!env)return;for(const a of env.actions){const row=document.createElement("div");row.className="bar-row";const label=document.createElement("span");label.textContent=a.label;const track=document.createElement("div");track.className="bar-track";const fill=document.createElement("div");fill.className="bar-fill";track.append(fill);const value=document.createElement("span");value.textContent="0.000";row.append(label,track,value);ui.bars.append(row)}}
function render(){
  if(!env||!controller||!policy)return;const obs=env.lastObservation||env.observe();ui.objective.textContent=env.schema.objective;
  ui.state.innerHTML=env.schema.fields.map(({id})=>'<div class="state-row"><span>'+id+'</span><strong>'+Number(obs[id]??0).toFixed(3)+'</strong></div>').join("");
  ui.steps.textContent=controller.steps;ui.episodes.textContent=controller.episodes;ui.ret.textContent=controller.episodeReturn.toFixed(3);ui.updates.textContent=policy.q.updates;ui.teacherCalls.textContent=policy.teacherCalls;
  ui.backbone.textContent=policy.semantic.name+(policy.semantic.backend?" · "+policy.semantic.backend:"")+" + "+(policy.q.name||policy.q.constructor.name);
  const lat=controller.latencySummary();ui.latLast.textContent=lat.last.toFixed(2)+" ms";ui.latP95.textContent=lat.p95.toFixed(2)+" ms";
  const d=controller.lastDecision;
  if(d){
    const decisionObs=controller.trace.at(-1)?.observation||obs;
    const attended=policy.q.inspectAttention?.(decisionObs,d.actionIndex,{topK:6})||[];
    ui.attentionCount.textContent=attended.length+" records";
    ui.attention.innerHTML=attended.length?attended.map(item=>{
      const interesting=Object.entries(item.record||{}).filter(([,value])=>typeof value==="number"&&Number.isFinite(value)&&value!==0).sort((a,b)=>Math.abs(Number(b[1]))-Math.abs(Number(a[1]))).slice(0,5);
      const detail=interesting.map(([key,value])=>key+"="+Number(value).toFixed(Math.abs(value)<10?2:0)).join(" · ");
      return '<div class="attention-row"><div><strong>'+item.collectionId+'['+item.index+']</strong><small>'+detail+'</small></div><span>'+(item.weight*100).toFixed(1)+'%</span></div>';
    }).join(""):'<div class="attention-empty">No structured records in this observation.</div>';
    ui.latSemantic.textContent=(d.semanticLatencyMs||policy.lastTeacherLatencyMs||0).toFixed(2)+" ms";ui.lastReward.textContent=d.reward.toFixed(3);ui.chosen.textContent=d.action.label;ui.entropy.textContent=d.uncertainty.entropy.toFixed(3);ui.margin.textContent=d.uncertainty.margin.toFixed(3);ui.novelty.textContent=d.uncertainty.novelty.toFixed(3);
    [...ui.bars.children].forEach((row,i)=>{row.querySelector(".bar-fill").style.width=(d.probs[i]*100).toFixed(1)+"%";row.lastElementChild.textContent=d.probs[i].toFixed(3)});
    ui.log.textContent=controller.trace.slice(-12).reverse().map(t=>"s"+String(t.step).padStart(4,"0")+" "+(t.teacherUsed?"Q":t.teacherPending?"…":"·")+" "+t.action.padEnd(13)+" p="+Math.max(...Object.values(t.probabilities)).toFixed(3)+" r="+t.reward.toFixed(3)+" neural="+t.residualLatencyMs.toFixed(2)+"ms total="+t.latencyMs.toFixed(1)+"ms").join("\n");
  }
}
function bindController(){controller.addEventListener("tick",render);controller.addEventListener("state",event=>{ui.start.textContent=event.detail==="RUNNING"?"Pause agent":"Start agent";setRuntime(event.detail)});controller.addEventListener("error",event=>{setRuntime("ERROR",true);ui.log.textContent="ERROR: "+event.detail.message+"\n"+ui.log.textContent})}
async function primeCurrentTeacher(label){
  const obs=env.lastObservation||env.observe();ui.modelStatus.textContent=label+" · semantic bootstrap in progress";
  const prime=await policy.primeTeacher(obs,{steps:8});
  if(prime?.stale)throw new Error("Semantic bootstrap became stale");
  return prime;
}
async function boot(){
  ui.boot.disabled=true;setRuntime("LOADING");ui.bootStatus.textContent="Fetching pinned engine + Freedoom runtime…";
  try{
    env=await DoomWasmArena.boot({canvas:ui.canvas,actionMs:Number(ui.actionMs.value),onProgress:message=>{ui.bootStatus.textContent=message}});
    hashSemantic=new HashSemanticAdapter();hashSemantic.backend="local-js";
    policy=new SemanticResidualPolicy({schema:env.schema,actions:env.actions,semantic:hashSemantic,residual:"neural-set",seed:1993,inferenceMode:ui.path.value});
    controller=new ExperimentController({environment:env,policy,hz:8});controller.useResidual=ui.residual.checked;controller.training=ui.training.checked;controller.memory=ui.memory.checked;controller.explore=ui.explore.checked;
    await primeCurrentTeacher("hash teacher");bindController();buildBars();setReadyControls(true);setRuntime("READY");
    const counts=env.lastObservation?._collections||{};ui.bootStatus.textContent=DOOM_RUNTIME_PROVENANCE.engine+" + "+DOOM_RUNTIME_PROVENANCE.content+" · "+policy.q.parameterCount()+" neural params · "+(counts.entities?.length||0)+" entities + "+(counts.geometry?.length||0)+" geometry records";
    ui.modelStatus.textContent="adaptive hash teacher ready · bootstrap "+policy.lastTeacherLatencyMs.toFixed(1)+" ms";render();
  }catch(error){ui.boot.disabled=false;setRuntime("BOOT FAILED",true);ui.bootStatus.textContent=String(error?.message||error);ui.log.textContent=String(error?.stack||error)}
}
async function switchBackbone(){
  if(!policy||!controller)return;controller.pause();setReadyControls(false);ui.modelProgress.value=0;const selected=ui.modelSelect.value,previous=policy.semantic;
  try{
    if(selected==="hash"){
      if(previous!==hashSemantic&&previous.dispose)await previous.dispose();policy.setSemantic(hashSemantic);policy.resetLearning();await controller.reset({learning:false});
      const prime=await primeCurrentTeacher("hash teacher");ui.modelStatus.textContent=ui.path.value+" hash teacher ready · bootstrap "+prime.ms.toFixed(1)+" ms · neural weights primed";render();return;
    }
    const preset=NLI_PRESETS[selected];ui.modelStatus.textContent="loading "+preset.label+" · "+preset.approx;
    const candidate=new TransformersNLIAdapter({preset:selected,maxStateChars:2600,onProgress:info=>{
      if(Number.isFinite(info.normalizedProgress))ui.modelProgress.value=info.normalizedProgress;
      const status=info.status==="ready"?"model loaded":(info.status||"loading");
      ui.modelStatus.textContent=status+(info.file?" · "+info.file:"");
    }});
    candidate.compile(env.schema,env.actions);if(previous!==hashSemantic&&previous.dispose)await previous.dispose();policy.setSemantic(hashSemantic);await candidate.load();
    policy.setSemantic(candidate);policy.resetLearning();await controller.reset({learning:false});const prime=await primeCurrentTeacher(preset.label);
    ui.modelProgress.value=100;ui.modelStatus.textContent=preset.label+" "+ui.path.value+" teacher ready · bootstrap "+prime.ms.toFixed(1)+" ms · fast ticks are neural";render();
  }catch(error){
    policy.setSemantic(hashSemantic);policy.resetLearning();await controller.reset({learning:false});await primeCurrentTeacher("hash teacher").catch(()=>null);
    ui.modelSelect.value="hash";ui.modelProgress.value=0;ui.modelStatus.textContent="load/bootstrap failed; hash restored · "+String(error?.message||error);render();
  }finally{setReadyControls(true)}
}

ui.boot.addEventListener("click",boot);ui.start.addEventListener("click",()=>controller?.state==="RUNNING"?controller.pause():controller?.start());
ui.step.addEventListener("click",async()=>{if(!controller)return;if(controller.state==="RUNNING")controller.pause();await controller.tick()});
ui.reset.addEventListener("click",async()=>{if(controller)await controller.reset({learning:false});render()});ui.loadModel.addEventListener("click",switchBackbone);
ui.path.addEventListener("change",()=>{if(policy){policy.setInferenceMode(ui.path.value);ui.modelStatus.textContent=ui.path.value+" inference · "+policy.semantic.name}});
ui.residual.addEventListener("change",()=>{if(controller)controller.useResidual=ui.residual.checked});ui.training.addEventListener("change",()=>{if(controller)controller.training=ui.training.checked});ui.memory.addEventListener("change",()=>{if(controller)controller.memory=ui.memory.checked});ui.explore.addEventListener("change",()=>{if(controller)controller.explore=ui.explore.checked});ui.actionMs.addEventListener("input",()=>{ui.actionMsOut.value=ui.actionMs.value+" ms";env?.setActionMs(ui.actionMs.value)});
ui.export.addEventListener("click",()=>{if(!controller)return;const blob=new Blob([controller.exportTrace()],{type:"application/json"}),a=document.createElement("a");a.href=URL.createObjectURL(blob);a.download="doom-real-trace-"+Date.now()+".json";a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)});
