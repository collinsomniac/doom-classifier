import {DoomWasmArena,DOOM_RUNTIME_PROVENANCE} from "./env/doom-wasm.js";
import {HashSemanticAdapter} from "./core/semantic.js";
import {SemanticResidualPolicy} from "./core/policy.js";
import {ExperimentController,ControllerState} from "./core/controller.js";
import {TransformersNLIAdapter,NLI_PRESETS} from "./model-adapters/transformers-nli.js";
import {MiniLMSchemaCompiler,SCHEMA_EMBEDDING_PRESET} from "./model-adapters/schema-embedding-compiler.js";
import {probeCounterfactualActions} from "./core/counterfactual.js";
import {softmax} from "./core/math.js";

const OWNED_RUNTIME_POINTER="./runtime/engine-runtime.json";
const OWNED_RUNTIME_RAW_ROOT="https://raw.githubusercontent.com/collinsomniac/doom-classifier";
const STARTER_MODEL_BASE="https://raw.githubusercontent.com/collinsomniac/doom-classifier/model-runtime";
const GENERIC_STARTER_BASE="https://raw.githubusercontent.com/collinsomniac/doom-classifier/starter-runtime";
const query=new URLSearchParams(globalThis.location?.search||""),requestedRuntime=query.get("runtime")||"owned",starterMode=query.get("starter")||"auto";
async function resolveOwnedRuntime(){
  const response=await fetch(OWNED_RUNTIME_POINTER,{cache:"no-store"});
  if(!response.ok)throw new Error("owned runtime pointer HTTP "+response.status);
  const pointer=await response.json(),commit=String(pointer?.asset_commit||"").trim();
  if(!/^[0-9a-f]{40}$/i.test(commit))throw new Error("owned runtime pointer is missing an exact asset commit");
  const base=OWNED_RUNTIME_RAW_ROOT+"/"+commit;
  return{
    base,
    info:{
      owned:true,repository:String(pointer.repository||"collinsomniac/doom-classifier"),
      branch:String(pointer.branch||"engine-runtime"),commit,
      sourceCommit:String(pointer.source_commit||""),base
    }
  };
}
const $=id=>document.getElementById(id);
const ui={
  boot:$("bootBtn"),prepare:$("prepareBtn"),start:$("startBtn"),step:$("stepBtn"),reset:$("resetBtn"),canvas:$("doomCanvas"),stage:$("doomStage"),teacherPause:$("teacherPauseOverlay"),teacherPauseReason:$("teacherPauseReason"),runtime:$("runtimeStatus"),runtimeTitle:$("runtimeTitle"),bootStatus:$("bootStatus"),
  iwad:$("iwadInput"),iwadStatus:$("iwadStatus"),engineChip:$("engineChip"),schemaChip:$("schemaChip"),teacherChip:$("teacherChip"),policyChip:$("policyChip"),prepareStatus:$("prepareStatus"),
  profile:$("profileSelect"),applyProfile:$("applyProfileBtn"),profileHint:$("profileHint"),teacherMode:$("teacherModeSelect"),useNeural:$("useNeuralToggle"),learn:$("learnToggle"),memory:$("memoryToggle"),explore:$("exploreToggle"),
  tune:$("tuneBtn"),eval:$("evalBtn"),evalResults:$("evalResults"),tuneSteps:$("tuneSteps"),tuneProgress:$("tuneProgress"),tuneStatus:$("tuneStatus"),tuneBadge:$("tuneBadge"),loadStarter:$("loadStarterBtn"),loadSaved:$("loadSavedBtn"),saveCheckpoint:$("saveCheckpointBtn"),exportCheckpoint:$("exportCheckpointBtn"),checkpointStatus:$("checkpointStatus"),publishedReplaySelect:$("publishedReplaySelect"),publishedReplayBtn:$("publishedReplayBtn"),publishedReplayStatus:$("publishedReplayStatus"),
  cfProbe:$("counterfactualProbeBtn"),cfFit:$("counterfactualFitBtn"),cfClear:$("counterfactualClearBtn"),cfTics:$("counterfactualTics"),cfBadge:$("counterfactualBadge"),cfStatus:$("counterfactualStatus"),cfResults:$("counterfactualResults"),cfExamples:$("counterfactualExamples"),cfSpread:$("counterfactualSpread"),cfPriorTop:$("counterfactualPriorTop"),cfMeasuredTop:$("counterfactualMeasuredTop"),cfTargetTop:$("counterfactualTargetTop"),
  actionMs:$("actionMs"),actionMsOut:$("actionMsOut"),manualAction:$("manualActionSelect"),manual:$("manualBtn"),manualStatus:$("manualStatus"),weaponState:$("weaponState"),
  circuit:$("decisionCircuit"),circuitActions:$("circuitActions"),circuitChosen:$("circuitChosen"),circuitRate:$("circuitRate"),circuitLatency:$("circuitLatency"),circuitState:$("circuitState"),circuitPriorTop:$("circuitPriorTop"),circuitPriorMeta:$("circuitPriorMeta"),circuitValueTop:$("circuitValueTop"),circuitValueMeta:$("circuitValueMeta"),circuitSearchTop:$("circuitSearchTop"),circuitSearchMeta:$("circuitSearchMeta"),circuitFusionMeta:$("circuitFusionMeta"),circuitOutputGlyph:$("circuitOutputGlyph"),circuitOutputLabel:$("circuitOutputLabel"),circuitOutputProb:$("circuitOutputProb"),circuitBackend:$("circuitBackend"),circuitAgreement:$("circuitAgreement"),
  bars:$("actionBars"),chosen:$("chosenAction"),chosenSemantic:$("chosenSemantic"),chosenValue:$("chosenValue"),chosenScore:$("chosenScore"),probabilityCalibration:$("probabilityCalibration"),intentFire:$("intentFire"),intentStrafe:$("intentStrafe"),intentTurn:$("intentTurn"),intentForward:$("intentForward"),intentBack:$("intentBack"),intentUse:$("intentUse"),entropy:$("entropy"),margin:$("margin"),epistemic:$("epistemic"),novelty:$("novelty"),latLast:$("latLast"),latSemantic:$("latSemantic"),latP95:$("latP95"),
  attention:$("attentionList"),attentionCount:$("attentionCount"),teacherCalls:$("teacherCalls"),decodeTemp:$("decodeTemp"),replaySize:$("replaySize"),teacherReplaySize:$("teacherReplaySize"),valueBeta:$("valueBeta"),priorKL:$("priorKL"),klUtilization:$("klUtilization"),backbone:$("backboneName"),modelSelect:$("modelSelect"),loadModel:$("loadModelBtn"),modelProgress:$("modelProgress"),modelStatus:$("modelStatus"),
  schemaCompile:$("schemaCompileBtn"),schemaStatus:$("schemaStatus"),state:$("stateTable"),objective:$("objectiveText"),steps:$("steps"),episodes:$("episodes"),ret:$("return"),updates:$("updates"),lastReward:$("lastReward"),damageDealt:$("damageDealt"),hostileHpLoss:$("hostileHpLoss"),damageReceived:$("damageReceived"),combatAttribution:$("combatAttribution"),
  architecture:$("architectureFlow"),archMode:$("archMode"),archFeedback:$("archFeedback"),typedTrace:$("typedTrace"),typedRawTop:$("typedRawTop"),typedRawMeta:$("typedRawMeta"),typedProjector:$("typedProjector"),typedProjectorMeta:$("typedProjectorMeta"),typedStrongField:$("typedStrongField"),typedStrongFieldMeta:$("typedStrongFieldMeta"),typedTrust:$("typedTrust"),typedTrustMeta:$("typedTrustMeta"),typedFused:$("typedFused"),typedFusedMeta:$("typedFusedMeta"),typedFieldGrid:$("typedFieldGrid"),trainingModeBadge:$("trainingModeBadge"),trainingOutput:$("trainingOutput"),teacherTranscript:$("teacherTranscript"),
  log:$("eventLog"),export:$("exportBtn"),dot:$("statusDot"),
  playTabBtn:$("playTabBtn"),inspectTabBtn:$("inspectTabBtn"),playTab:$("playTab"),inspectTab:$("inspectTab")
};
let env=null,policy=null,controller=null,hashSemantic=null;
let engineReady=false,schemaCompiled=false,teacherReady=false,checkpointReady=false,busy=false,prepared=false,checkpointInfo=null;
let lastArchPulseDecision=-1;
let lastCounterfactualProbe=null,counterfactualExamples=[],counterfactualBusy=false,lastCounterfactualFit=null,publishedReplayBundle=null,replayBusy=false;

function setLabTab(name,{updateHash=true}={}){
  const active=name==="inspect"?"inspect":"play";
  for(const [key,button,panel] of [["play",ui.playTabBtn,ui.playTab],["inspect",ui.inspectTabBtn,ui.inspectTab]]){
    const on=key===active;
    button?.classList.toggle("active",on);button?.setAttribute("aria-selected",String(on));
    panel?.classList.toggle("active",on);if(panel)panel.hidden=!on;
  }
  if(updateHash){
    const url=new URL(location.href);url.hash=active==="inspect"?"inspect":"play";
    history.replaceState(null,"",url);
  }
}
ui.playTabBtn?.addEventListener("click",()=>setLabTab("play"));
ui.inspectTabBtn?.addEventListener("click",()=>setLabTab("inspect"));
setLabTab(location.hash==="#inspect"?"inspect":"play",{updateHash:false});

const PROFILES={
  assisted:{label:"Adaptive assisted",teacher:"adaptive",neural:true,learning:false,memory:true,explore:false,hint:"Fast neural decisions normally; when confidence is low, the simulator freezes on the exact state while the semantic teacher resolves and distills it."},
  teacher:{label:"Teacher-only zero-shot",teacher:"every",neural:false,learning:false,memory:true,explore:false,hint:"MobileBERT scores every decision while simulation time is frozen. Useful semantic baseline; wall-clock latency never becomes in-game hesitation."},
  learning:{label:"Online learning",teacher:"adaptive",neural:true,learning:true,memory:true,explore:true,hint:"The neural controller samples from its own uncertainty (with a small uniform floor), learns real consequences, and receives adaptive semantic supervision."},
  frozen:{label:"Frozen neural evaluation",teacher:"off",neural:true,learning:false,memory:true,explore:false,hint:"No teacher calls, no weight updates, no random exploration. This is the clean small-model evaluation mode."}
};

function setChip(element,state,text){
  element.className="ready-chip "+state;element.textContent=text;
}
function isLearnedTeacher(){return teacherReady&&policy?.semantic&&policy.semantic!==hashSemantic}
function updateReadiness(){
  const teacherPrepared=schemaCompiled&&isLearnedTeacher();
  prepared=engineReady&&(checkpointReady||teacherPrepared);
  setChip(ui.engineChip,engineReady?"ready":"off",engineReady?"engine ready":"engine");
  setChip(ui.schemaChip,(schemaCompiled||checkpointReady)?"ready":"off",checkpointReady?"checkpoint schema":schemaCompiled?"schema compiled":"schema semantics");
  setChip(ui.teacherChip,isLearnedTeacher()?"ready":checkpointReady?"ready":"off",isLearnedTeacher()?"teacher loaded":checkpointReady?"teacher-free checkpoint":"teacher");
  setChip(ui.policyChip,prepared?"ready":"warn",prepared?"ready to play":"not prepared");
  const lock=busy||!engineReady;
  ui.prepare.disabled=lock;ui.loadModel.disabled=lock;ui.schemaCompile.disabled=lock;ui.manual.disabled=lock;ui.manualAction.disabled=lock;ui.reset.disabled=lock;ui.export.disabled=lock;
  if(ui.loadStarter)ui.loadStarter.disabled=lock;if(ui.loadSaved)ui.loadSaved.disabled=lock||!localStorage.getItem("doom-classifier-checkpoint-v1");
  if(ui.saveCheckpoint)ui.saveCheckpoint.disabled=busy||!prepared;if(ui.exportCheckpoint)ui.exportCheckpoint.disabled=busy||!prepared;
  if(ui.publishedReplaySelect)ui.publishedReplaySelect.disabled=busy||replayBusy||!publishedReplayBundle?.runs?.length;
  if(ui.publishedReplayBtn)ui.publishedReplayBtn.disabled=busy||replayBusy||!publishedReplayBundle?.runs?.length||!engineReady;
  const forkReady=!!env?.supportsSnapshots?.()&&!!env?.supportsExactTics?.();if(ui.cfProbe)ui.cfProbe.disabled=busy||!prepared||!forkReady;if(ui.cfTics)ui.cfTics.disabled=busy||!forkReady;if(ui.cfFit)ui.cfFit.disabled=busy||!prepared||!counterfactualExamples.length;if(ui.cfClear)ui.cfClear.disabled=busy||!counterfactualExamples.length;
  for(const element of [ui.profile,ui.applyProfile,ui.teacherMode,ui.useNeural,ui.learn,ui.memory,ui.explore,ui.tune,ui.eval,ui.tuneSteps,ui.start,ui.step])element.disabled=busy||!prepared;
  if(controller?.state==="RUNNING")ui.start.disabled=false;
}
function setBusy(value){busy=value;updateReadiness()}
function setRuntime(text,error=false){ui.runtime.textContent=text;ui.dot.style.background=error?"#ff6b6b":"#d9ff5a"}
function displayField(field,value){
  const category=field.enum?.[String(value)];
  if(category!==undefined)return category+" ("+value+")";
  const n=Number(value);return Number.isFinite(n)?n.toFixed(Math.abs(n)<10?3:1):String(value??"—");
}
function weaponLabel(obs){
  const field=env?.schema.fields.find(f=>f.id==="weapon"),value=obs?.weapon,category=field?.enum?.[String(value)]||("weapon "+value);
  const ammo=value===1||value===3?obs.bullets:value===2||value===8?obs.shells:value===4?obs.rockets:value===5||value===6?obs.cells:"∞";
  return category+" · ammo "+ammo;
}
function actionGlyph(action){
  const p=action?.params||{},bits=[];
  if(Number(p.strafe_left||0)>0)bits.push("←");
  if(Number(p.strafe_right||0)>0)bits.push("→");
  if(Number(p.turn_left||0)>0)bits.push("↶");
  if(Number(p.turn_right||0)>0)bits.push("↷");
  if(Number(p.forward||0)>0)bits.push("↑");
  if(Number(p.back||0)>0)bits.push("↓");
  if(Number(p.fire||0)>0)bits.push("◎");
  if(Number(p.use||0)>0)bits.push("◇");
  return bits.join("")||"·";
}
function topScoreIndex(values){
  if(!values?.length)return -1;
  let best=0;for(let i=1;i<values.length;i++)if(Number(values[i])>Number(values[best]))best=i;
  return best;
}
function measuredDecisionRate(){
  const rows=controller?.trace?.slice(-16)||[];
  if(rows.length<2)return 0;
  const elapsed=(Number(rows.at(-1)?.t||0)-Number(rows[0]?.t||0))/1000;
  return elapsed>0?(rows.length-1)/elapsed:0;
}
function setCircuitHot(name,hot){
  const node=ui.circuit?.querySelector('[data-circuit="'+name+'"]');
  if(node)node.classList.toggle("hot",!!hot);
}
function renderDecisionCircuit(obs,d){
  if(!ui.circuit||!policy||!env)return;
  const entities=obs?._collections?.entities?.length||0,geometry=obs?._collections?.geometry?.length||0;
  ui.circuitState.textContent=(policy.schema?.fields?.length||0)+"F · "+entities+"E · "+geometry+"G · h"+Number(d?.history?.length||0);
  const fastBackend=policy.q?.backend||"local-js",teacherBackend=isLearnedTeacher()?(policy.semantic?.backend||"loaded"):"off";
  ui.circuitBackend.textContent="engine WASM · fast "+fastBackend+" · teacher "+teacherBackend;
  const rate=measuredDecisionRate(),lat=Number(d?.latencyMs||controller?.latencySummary?.().last||0);
  ui.circuitRate.textContent=(rate||Number(controller?.hz||0)).toFixed(1)+" decisions/s";
  ui.circuitLatency.textContent=lat.toFixed(1)+" ms / decision";
  setCircuitHot("state",!!d);setCircuitHot("prior",!!d);setCircuitHot("value",!!d&&Number(d.valueBeta||0)>0);
  setCircuitHot("search",counterfactualBusy||!!(lastCounterfactualProbe&&Date.now()-Number(lastCounterfactualProbe.t||0)<1800));
  setCircuitHot("fusion",!!d);setCircuitHot("output",!!d);
  if(!d){
    ui.circuitChosen.textContent="waiting for a decision";
    ui.circuitPriorTop.textContent="—";ui.circuitValueTop.textContent="—";
    ui.circuitOutputGlyph.textContent="·";ui.circuitOutputLabel.textContent="waiting";ui.circuitOutputProb.textContent="p 0.000";
    ui.circuitFusionMeta.textContent="β 0";ui.circuitAgreement.textContent="prior / critic / fused waiting";
    return;
  }
  const semanticTop=topScoreIndex(d.semanticPriorScores),valueTop=topScoreIndex(d.valueScores);
  const semanticAction=semanticTop>=0?policy.actions[semanticTop]:null,valueAction=valueTop>=0?policy.actions[valueTop]:null;
  const chosen=d.action,chosenP=Number(d.probs?.[d.actionIndex]||0);
  ui.circuitChosen.textContent=actionGlyph(chosen)+" "+(chosen?.label||chosen?.id||"waiting");
  ui.circuitPriorTop.textContent=semanticAction?(actionGlyph(semanticAction)+" "+semanticAction.id):"—";
  ui.circuitPriorMeta.textContent=semanticTop>=0?("score "+Number(d.semanticPriorScores?.[semanticTop]||0).toFixed(3)+(d.teacherUsed?" · teacher tick":d.teacherPending?" · teacher pending":" · fast prior")):"waiting";
  ui.circuitValueTop.textContent=valueAction?(actionGlyph(valueAction)+" "+valueAction.id):"—";
  ui.circuitValueMeta.textContent=valueTop>=0?("Q "+Number(d.valueScores?.[valueTop]||0).toFixed(3)+" · authority "+(Number(d.criticAuthority||0)*100).toFixed(0)+"%"):"waiting";
  if(lastCounterfactualProbe?.returns?.length){
    const measuredTop=topScoreIndex(lastCounterfactualProbe.returns),a=policy.actions[measuredTop];
    ui.circuitSearchTop.textContent=a?(actionGlyph(a)+" "+a.id):"—";
    ui.circuitSearchMeta.textContent="same state · "+Number(lastCounterfactualProbe.tics||0)+" tics · ΔR "+Number(lastCounterfactualProbe.spread||0).toFixed(3);
  }else{
    ui.circuitSearchTop.textContent=env.supportsSnapshots?.()&&env.supportsExactTics?.()?"fork ready":"unavailable";
    ui.circuitSearchMeta.textContent=env.supportsSnapshots?.()&&env.supportsExactTics?.()?"snapshot + deterministic tics":"runtime lacks exact fork";
  }
  ui.circuitFusionMeta.textContent="β "+Number(d.valueBeta||0).toFixed(1)+" · KL "+Number(d.priorKL||0).toFixed(3)+" · "+String(d.decisionRule||"argmax");
  ui.circuitOutputGlyph.textContent=actionGlyph(chosen);ui.circuitOutputLabel.textContent=chosen?.id||"waiting";ui.circuitOutputProb.textContent="p "+chosenP.toFixed(3);
  const semanticId=semanticAction?.id||"—",valueId=valueAction?.id||"—",chosenId=chosen?.id||"—";
  ui.circuitAgreement.textContent="prior "+semanticId+(semanticId===valueId?" = ":" ≠ ")+"critic "+valueId+" → fused "+chosenId;
  [...(ui.circuitActions?.children||[])].forEach((chip,i)=>{
    chip.classList.toggle("chosen",i===d.actionIndex);
    chip.classList.toggle("semantic-top",i===semanticTop);
    chip.classList.toggle("value-top",i===valueTop);
    const prob=chip.querySelector('[data-role="prob"]');if(prob)prob.textContent=Number(d.probs?.[i]||0).toFixed(2);
  });
}
function buildBars(){
  ui.bars.innerHTML="";ui.manualAction.innerHTML="";
  if(!env)return;
  for(const a of env.actions){
    const row=document.createElement("div");row.className="bar-row";
    const label=document.createElement("span");label.textContent=a.label;
    const track=document.createElement("div");track.className="bar-track";const fill=document.createElement("div");fill.className="bar-fill";track.append(fill);
    const value=document.createElement("span");value.textContent="0.000";row.append(label,track,value);ui.bars.append(row);
    const option=document.createElement("option");option.value=a.id;option.textContent=a.label;ui.manualAction.append(option);
  }
  if(ui.circuitActions)ui.circuitActions.innerHTML=env.actions.map((a,i)=>'<div class="circuit-action" data-index="'+i+'" title="'+escapeHtml(a.label||a.id)+'"><b>'+escapeHtml(actionGlyph(a))+'</b><span>'+escapeHtml(a.id)+'</span><small data-role="prob">0.00</small></div>').join("");
  if([...ui.manualAction.options].some(o=>o.value==="fire"))ui.manualAction.value="fire";
}
function syncRuntimeConfig(){
  if(!controller||!policy)return;
  if(ui.teacherMode.value==="off"&&!ui.useNeural.checked)ui.useNeural.checked=true;
  controller.useResidual=ui.useNeural.checked;controller.training=ui.learn.checked;controller.memory=ui.memory.checked;controller.explore=ui.explore.checked;
  policy.setInferenceMode(ui.teacherMode.value==="off"?"neural":ui.teacherMode.value==="every"?"hybrid":"adaptive");
}
function applyProfile(id=ui.profile.value){
  const cfg=PROFILES[id]||PROFILES.assisted;ui.profile.value=id;ui.teacherMode.value=cfg.teacher;ui.useNeural.checked=cfg.neural;ui.learn.checked=cfg.learning;ui.memory.checked=cfg.memory;ui.explore.checked=cfg.explore;
  ui.profileHint.textContent=cfg.label+" — "+cfg.hint;syncRuntimeConfig();
}
function escapeHtml(value){return String(value??"").replace(/[&<>"']/g,ch=>({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[ch]))}
function localSoftmax(scores,temp=1){
  if(!scores?.length)return[];const t=Math.max(.05,Number(temp)||1),peak=Math.max(...scores),ex=scores.map(v=>Math.exp((Number(v)-peak)/t)),sum=ex.reduce((a,b)=>a+b,0)||1;return ex.map(v=>v/sum)
}
function normalizeFieldValue(field,raw){
  if(field?.enum)return String(field.enum?.[String(raw)]??raw);
  const v=Number(raw??0),min=Number.isFinite(Number(field?.min))?Number(field.min):0,max=Number.isFinite(Number(field?.max))?Number(field.max):1;
  return max===min?0:Math.max(0,Math.min(1,(v-min)/(max-min)));
}
function fieldExpectation(field,dist){
  if(field?.enum){
    const bins=new Map();policy.actions.forEach((action,i)=>{const key=String(action.params?.[field.id]??"unset");bins.set(key,(bins.get(key)||0)+(dist[i]||0))});
    const [key,p]=[...bins.entries()].sort((a,b)=>b[1]-a[1])[0]||["unset",0];
    return{label:(field.enum?.[key]??key)+" "+(p*100).toFixed(0)+"%",strength:Number(p||0)};
  }
  let sum=0;policy.actions.forEach((action,i)=>{sum+=(dist[i]||0)*normalizeFieldValue(field,action.params?.[field.id]??0)});
  return{label:sum.toFixed(2),strength:Math.max(0,Math.min(1,sum))};
}
function renderTypedFields(d){
  if(!ui.typedFieldGrid||!policy)return;const fields=policy.schema.actionFields||[];
  if(!d||!fields.length){
    ui.typedFieldGrid.innerHTML='<div class="typed-field-empty">Field-level agreement appears after the first decision.</div>';
    for(const element of [ui.typedRawTop,ui.typedProjector,ui.typedStrongField,ui.typedTrust,ui.typedFused])if(element)element.textContent="—";
    return
  }
  const semantic=localSoftmax(d.semanticPriorScores||[],Math.max(.25,policy.temperature||1));
  const valueScores=d.valueScores||[],mean=valueScores.reduce((a,b)=>a+Number(b||0),0)/Math.max(1,valueScores.length),variance=valueScores.reduce((a,b)=>a+(Number(b||0)-mean)**2,0)/Math.max(1,valueScores.length),scale=Math.max(.02,Math.sqrt(variance));
  const value=localSoftmax(valueScores,scale),typedScores=d.typedValueScores||valueScores,typedMean=typedScores.reduce((a,b)=>a+Number(b||0),0)/Math.max(1,typedScores.length),typedVariance=typedScores.reduce((a,b)=>a+(Number(b||0)-typedMean)**2,0)/Math.max(1,typedScores.length),typedScale=Math.max(.02,Math.sqrt(typedVariance)),typed=localSoftmax(typedScores,typedScale),fused=d.probs||[];
  const branch=(name,item,cls="")=>'<div class="typed-branch '+cls+'"><span>'+name+'</span><div class="typed-meter"><i style="width:'+(item.strength*100).toFixed(1)+'%"></i></div><b>'+escapeHtml(item.label)+'</b></div>';
  const coefficientFor=field=>Number((d.typedFieldCoefficients||[]).find(x=>x.id===field.id)?.weight||0);
  const rawTopIndex=valueScores.length?valueScores.reduce((best,v,i,a)=>Number(v)>Number(a[best])?i:best,0):-1;
  const rawTop=rawTopIndex>=0?policy.actions[rawTopIndex]:null;
  const strongest=[...(d.typedFieldCoefficients||[])].sort((a,b)=>Math.abs(Number(b.weight||0))-Math.abs(Number(a.weight||0)))[0]||null;
  if(ui.typedRawTop)ui.typedRawTop.textContent=rawTop?rawTop.id:"—";
  if(ui.typedRawMeta)ui.typedRawMeta.textContent=rawTop?("Q "+Number(valueScores[rawTopIndex]||0).toFixed(4)+" · gap "+Number(d.criticGap||0).toFixed(4)):"exact-action values";
  if(ui.typedProjector)ui.typedProjector.textContent=(Number(d.typedValueFit||0)*100).toFixed(0)+"% fit";
  if(ui.typedProjectorMeta)ui.typedProjectorMeta.textContent=(Number(d.typedValueBlendUsed||0)*100).toFixed(0)+"% structured blend";
  if(ui.typedStrongField)ui.typedStrongField.textContent=strongest?(strongest.id+" "+(Number(strongest.weight||0)>=0?"+":"")+Number(strongest.weight||0).toFixed(3)):"—";
  if(ui.typedStrongFieldMeta)ui.typedStrongFieldMeta.textContent=strongest?"learned primitive coefficient":"primitive coefficient";
  if(ui.typedTrust)ui.typedTrust.textContent=(Number(d.criticAuthority||0)*100).toFixed(0)+"% authority";
  if(ui.typedTrustMeta)ui.typedTrustMeta.textContent="heads "+(Number(d.criticTopAgreement||0)*100).toFixed(0)+"% · SNR "+Number(d.criticMarginSnr||0).toFixed(1)+" · KL "+Number(d.priorKlBudget||0).toFixed(3);
  if(ui.typedFused)ui.typedFused.textContent=d.action?.id||"—";
  if(ui.typedFusedMeta)ui.typedFusedMeta.textContent="p "+Number(d.probs?.[d.actionIndex]||0).toFixed(3)+" · β "+Number(d.valueBeta||0).toFixed(2);
  ui.typedTrace?.classList.toggle("typed-trace-disagree",!!rawTop&&rawTop.id!==d.action?.id);
  ui.typedFieldGrid.innerHTML=fields.map(field=>{
    const prior=fieldExpectation(field,semantic),raw=fieldExpectation(field,value),structured=fieldExpectation(field,typed),final=fieldExpectation(field,fused),spread=Math.max(prior.strength,structured.strength,final.strength)-Math.min(prior.strength,structured.strength,final.strength),coef=coefficientFor(field);
    const cls=spread>.25?"field-disagree":spread<.08?"field-agree":"";
    return '<div class="typed-field '+cls+'" title="branch spread '+spread.toFixed(3)+'"><div class="typed-field-title"><strong>'+escapeHtml(field.label||field.id)+'</strong><em>ΔQ '+(coef>=0?"+":"")+coef.toFixed(3)+'</em></div><div class="typed-field-values">'+branch("prior",prior)+branch("raw Q",raw,"field-raw")+branch("typed Q",structured,"field-value")+branch("fused",final,"field-fused")+'</div></div>';
  }).join("");
}
function setArchNode(id,{active=false,hot=false,pending=false,learning=false,value=null}={}){
  const node=ui.architecture?.querySelector('[data-arch="'+id+'"]');if(!node)return;
  node.classList.toggle("active",!!active);node.classList.toggle("hot",!!hot);node.classList.toggle("pending",!!pending);node.classList.toggle("learning",!!learning);
  if(value!==null){const target=node.querySelector(".arch-value");if(target)target.textContent=value}
}
function renderArchitecture(obs,d,outcome){
  if(!ui.architecture)return;
  const pulseDecision=Number(policy?.decisionCount||0);
  if(d&&pulseDecision!==lastArchPulseDecision){
    lastArchPulseDecision=pulseDecision;
    ui.architecture.classList.remove("tick-pulse");
    void ui.architecture.offsetWidth;
    ui.architecture.classList.add("tick-pulse");
  }
  const entities=obs?._collections?.entities?.length||0,geometry=obs?._collections?.geometry?.length||0,lastTeacher=policy?.lastTeacherResult;
  const teacherFresh=!!lastTeacher&&Math.abs(Number(policy?.decisionCount||0)-Number(lastTeacher.step||0))<=2;
  const learning=!!controller?.training||!!d?.learningInfo;
  setArchNode("environment",{active:engineReady,hot:!!d,value:engineReady?(env.runtime?.owned?"DOOM · owned causal ABI":"DOOM · borrowed telemetry"):"not mounted"});
  setArchNode("state",{active:engineReady,hot:!!d,value:engineReady?(policy.schema.fields.length+" fields · "+entities+" entities · "+geometry+" lines"):"facts + records"});
  setArchNode("encoder",{active:prepared||!!d,hot:!!d,value:policy?(policy.q.parameterCount()+" params · memory "+(controller?.memory?"on":"off")):"shared representation"});
  const teacherTop=lastTeacher?.top?.[0],semanticTop=d?.semanticPriorScores?.length?d.semanticPriorScores.reduce((best,v,i,a)=>v>a[best]?i:best,0):-1,valueTop=d?.valueScores?.length?d.valueScores.reduce((best,v,i,a)=>v>a[best]?i:best,0):-1;
  const learnInfo=d?.learningInfo||controller?.trace?.at(-1)?.learning;
  setArchNode("teacher",{active:isLearnedTeacher(),hot:!!d?.teacherUsed||teacherFresh,pending:!!d?.teacherPending,value:isLearnedTeacher()?(policy.semantic.name+(teacherTop?" → "+teacherTop.id+" "+(teacherTop.probability*100).toFixed(0)+"%":"")+" · "+Number(policy.lastTeacherLatencyMs||0).toFixed(0)+" ms"):"off"});
  setArchNode("semantic",{active:prepared||!!d,hot:!!d?.semanticUsed||teacherFresh,value:semanticTop>=0?("prior → "+policy.actions[semanticTop].id+" · "+Number(d.semanticPriorScores[semanticTop]||0).toFixed(3)):"unprepared"});
  setArchNode("replay",{active:(policy?.replay?.length||0)>0||learning,hot:learning,learning,value:(policy?.replay?.length||0)+" transitions · H"+Number(learnInfo?.nStepHorizon||policy?.nStep||1)+(learnInfo?" · TD "+Number(learnInfo.td||0).toFixed(3):"")});
  const cfSupport=!!env?.supportsSnapshots?.()&&!!env?.supportsExactTics?.(),cfSpread=Number(lastCounterfactualProbe?.spread||0);
  setArchNode("counterfactual",{active:cfSupport,hot:counterfactualBusy,value:counterfactualBusy?"branching exact state…":lastCounterfactualProbe?(lastCounterfactualProbe.trials.length+" actions · "+lastCounterfactualProbe.tics+" tics · ΔR "+cfSpread.toFixed(3)):cfSupport?"exact fork ready":"unavailable"});
  setArchNode("properfit",{active:counterfactualExamples.length>0||!!lastCounterfactualFit,hot:!!lastCounterfactualFit&&Date.now()-lastCounterfactualFit.t<1800,value:lastCounterfactualFit?("fit "+lastCounterfactualFit.examples+" states · "+lastCounterfactualFit.rows+" rows"):(counterfactualExamples.length+" collected targets")});
  const rankedValues=d?.valueScores?.length?[...d.valueScores].sort((a,b)=>b-a):[],valueGap=Number(d?.criticGap??(rankedValues.length>1?rankedValues[0]-rankedValues[1]:0));
  setArchNode("value",{active:(policy?.q?.updates||0)>0||!!d,hot:learning,learning,value:valueTop>=0?("value → "+policy.actions[valueTop].id+" · ΔQ "+valueGap.toFixed(4)+" ("+(Number(d?.criticGapShare||0)*100).toFixed(0)+"% spread) · heads "+(Number(d?.criticTopAgreement||0)*100).toFixed(0)+"% · SNR "+Number(d?.criticMarginSnr||0).toFixed(1)):"untrained"});
  setArchNode("fusion",{active:!!d,hot:!!d&&Number(d.valueBeta||0)>0,value:d?("β "+Number(d.valueBeta||0).toFixed(2)+" · conf "+(Number(d.criticRankingConfidence||0)*100).toFixed(0)+"% · gate "+(Number(d.criticKlGate??1)*100).toFixed(0)+"% · KL "+Number(d.basePriorKlBudget||0).toFixed(3)+"→"+Number(d.priorKlBudget||0).toFixed(3)+" · used "+(Number(d.klUtilization||0)*100).toFixed(0)+"% · → "+d.action.id):"β 0 · critic gate waiting"});
  setArchNode("calibration",{active:!!d,hot:!!d?.probabilityCalibrated,value:d?.probabilityCalibrated?("fitted · T×"+Number(d.probabilityCalibrationTemperature||1).toFixed(3)+" · effective T "+Number(d.probabilityTemperature||0).toFixed(3)):"unverified · normalized scores only"});
  setArchNode("actions",{active:!!d,hot:!!d,value:d?(d.action.label+" · p "+Number(d.probs?.[d.actionIndex]||0).toFixed(3)):"waiting"});
  setArchNode("actuator",{active:!!d,hot:!!d,value:d?("primitive mask "+String(env.actionMasks?.[d.action.id]??"—")):"idle"});
  const starterSummary=checkpointReady?checkpointSummary(checkpointInfo):null;ui.archMode.textContent=!engineReady?"waiting for engine":learning?"learning":starterSummary?("validated stage "+starterSummary.stage+" · "+starterSummary.decisions+" decisions · teacher off"):policy?.inferenceMode==="neural"?"frozen / neural":policy?.inferenceMode==="hybrid"?"teacher in decode path":"adaptive supervision";
  const reward=Number(d?.reward||0),dmg=Number(outcome?.damageDealt||0),kills=Number(outcome?.playerKillDelta||0),pickups=Number(outcome?.playerPickupDelta||0);
  ui.archFeedback.textContent=d?("r "+reward.toFixed(3)+" · attributed damage "+dmg.toFixed(0)+" · kills "+kills.toFixed(0)+" · pickups "+pickups.toFixed(0)):"reward/events feed the consequence branch; teacher supervision feeds only the semantic prior";
}
function renderTrainingStream(){
  if(!ui.trainingOutput||!controller)return;
  const rows=controller.trace.slice(-24);
  ui.trainingModeBadge.textContent=controller.training?"learning · "+(policy.q.updates||0)+" updates":policy.inferenceMode==="neural"?"frozen neural":"observing";
  if(!rows.length){ui.trainingOutput.textContent="No decisions yet.";return}
  ui.trainingOutput.textContent=rows.map(t=>{
    const maxP=Math.max(...Object.values(t.probabilities||{x:0})),learn=t.learning;
    const eventBits=[];if(Number(t.outcome?.damageDealt||0)>0)eventBits.push("dmg+"+Number(t.outcome.damageDealt).toFixed(0));if(Number(t.outcome?.playerKillDelta||0)>0)eventBits.push("kill+"+Number(t.outcome.playerKillDelta).toFixed(0));if(Number(t.outcome?.playerPickupDelta||0)>0)eventBits.push("pickup+"+Number(t.outcome.playerPickupDelta).toFixed(0));
    const bootstrap=learn?.doubleDqn&&Number(learn.bootstrapActionIndex)>=0?" boot→"+(policy.actions[learn.bootstrapActionIndex]?.id||learn.bootstrapActionIndex)+"@"+Number(learn.bootstrapValue||0).toFixed(3):"";
    const td=learn?" td="+Number(learn.td||0).toFixed(3)+"→"+Number(learn.target||0).toFixed(3)+" H"+Number(learn.nStepHorizon||0)+bootstrap:"";
    return "s"+String(t.step).padStart(4,"0")+" "+(t.mode?.training?"TRAIN":"PLAY ")+" "+String(t.action).padEnd(19)+" p="+maxP.toFixed(3)+" r="+Number(t.reward||0).toFixed(3)+td+" β="+Number(t.valueBeta||0).toFixed(2)+" KL="+(Number(t.klUtilization||0)*100).toFixed(0)+"% "+(eventBits.join(",")||"—");
  }).join("\n");
  ui.trainingOutput.scrollTop=ui.trainingOutput.scrollHeight;
}
function renderTeacherTranscript(){
  if(!ui.teacherTranscript||!policy)return;
  const history=policy.teacherHistory||[];if(!history.length){ui.teacherTranscript.innerHTML='<div class="teacher-empty">No semantic-teacher response yet. Prepare Recommended or enable teacher supervision.</div>';return}
  ui.teacherTranscript.innerHTML=history.slice(-6).reverse().map((item,index)=>{
    const actions=(item.top||[]).slice(0,5).map(a=>'<span class="teacher-action">'+escapeHtml(a.label||a.id)+' <strong>'+(Number(a.probability||0)*100).toFixed(1)+'%</strong></span>').join("");
    const fit=item.distillation?('<span>distill '+Number(item.distillation.stepsUsed||0)+' steps · KL '+Number(item.distillation.kl||0).toFixed(3)+'</span>'):"";
    const calibration=item.calibration?('<span>student-fit T '+Number(item.calibration.temperature||0).toFixed(3)+'</span>'):"";
    return '<article class="teacher-item '+(index===0?"latest":"")+'"><div class="teacher-item-head"><strong>'+escapeHtml(item.model)+' · '+escapeHtml(item.kind)+'</strong><span>s'+String(item.step).padStart(4,"0")+' · '+Number(item.ms||0).toFixed(0)+' ms</span></div><div class="teacher-reason">'+escapeHtml(item.reason)+'</div><div class="teacher-actions">'+actions+'</div><div class="teacher-meta">'+fit+calibration+'</div><details><summary>state sent to teacher</summary><pre class="teacher-state">'+escapeHtml(item.stateText||"state serializer unavailable")+'</pre></details></article>';
  }).join("");
}
function semanticPriorAtState(observation){
  const stats=policy.q.scoreStatsObservation(observation,{temporal:null});
  return softmax(stats.semanticScores,Math.max(.05,Number(policy.temperature)||1));
}
function renderCounterfactual(){
  if(!ui.cfResults)return;
  const count=counterfactualExamples.length;ui.cfExamples.textContent=String(count);
  if(!lastCounterfactualProbe){
    ui.cfBadge.textContent=counterfactualBusy?"probing":"idle";ui.cfSpread.textContent="—";ui.cfPriorTop.textContent="—";ui.cfMeasuredTop.textContent="—";ui.cfTargetTop.textContent="—";
    if(!counterfactualBusy)ui.cfResults.innerHTML='<div class="counterfactual-empty">Run a probe to compare the current semantic prior against outcomes measured from an identical state.</div>';
    return;
  }
  const p=lastCounterfactualProbe,priorTop=p.prior.indexOf(Math.max(...p.prior)),measuredTop=p.returns.reduce((best,v,i,a)=>v>a[best]?i:best,0),targetTop=p.target.indexOf(Math.max(...p.target));
  ui.cfBadge.textContent=counterfactualBusy?"probing":"state restored";ui.cfSpread.textContent=Number(p.spread||0).toFixed(3);ui.cfPriorTop.textContent=policy.actions[priorTop]?.id||"—";ui.cfMeasuredTop.textContent=policy.actions[measuredTop]?.id||"—";ui.cfTargetTop.textContent=policy.actions[targetTop]?.id||"—";
  const rows=p.trials.map((trial,i)=>({trial,index:trial.actionIndex,prior:p.prior[trial.actionIndex]||0,target:p.target[trial.actionIndex]||0,ret:p.returns[trial.actionIndex]})).sort((a,b)=>b.ret-a.ret);
  ui.cfResults.innerHTML=rows.map((row,rank)=>{
    const o=row.trial.outcome||{},bits=[];if(Number(o.damageDealt||0)>0)bits.push("dmg+"+Number(o.damageDealt).toFixed(0));if(Number(o.playerKillDelta||0)>0)bits.push("kill+"+Number(o.playerKillDelta).toFixed(0));if(Number(o.playerPickupDelta||0)>0)bits.push("pickup+"+Number(o.playerPickupDelta).toFixed(0));if(Number(o.healthDelta||0)<0)bits.push("hp"+Number(o.healthDelta).toFixed(0));
    return '<div class="counterfactual-row '+(rank===0?"top-measured":"")+'"><strong>'+(rank+1)+'. '+escapeHtml(row.trial.label||row.trial.id)+'</strong><div class="counterfactual-cell"><span>prior</span><b>'+(row.prior*100).toFixed(1)+'%</b></div><div class="counterfactual-cell"><span>return</span><b>'+Number(row.ret||0).toFixed(3)+'</b></div><div class="counterfactual-cell"><span>target</span><b>'+(row.target*100).toFixed(1)+'%</b></div><div class="counterfactual-outcome">'+(bits.join(" · ")||"no causal event")+'</div></div>';
  }).join("");
}
async function probeSameState(){
  if(!prepared||!env?.supportsSnapshots?.()||!env?.supportsExactTics?.())return;
  controller.pause();counterfactualBusy=true;setBusy(true);ui.cfStatus.textContent="Forking one exact state across all typed actions…";renderCounterfactual();renderArchitecture(env.lastObservation||env.observe(),controller.lastDecision,env.lastOutcome);await Promise.resolve();
  const started=performance.now();
  try{
    const observation=env.lastObservation||env.observe(),prior=semanticPriorAtState(observation),tics=Math.max(1,Number(ui.cfTics.value)||24);
    const probe=await probeCounterfactualActions({
      environment:env,actions:policy.actions,prior,horizon:1,temperature:.45,priorStrength:1,
      stepper:id=>env.stepTics(id,tics)
    });
    lastCounterfactualProbe={...probe,prior,tics,ms:performance.now()-started,t:Date.now()};
    if(Number(probe.spread||0)>1e-9){
      counterfactualExamples.push({observation:probe.observation,target:probe.target,temporal:null,meta:{tics,spread:probe.spread}});
      if(counterfactualExamples.length>12)counterfactualExamples.shift();
    }
    const best=probe.returns.reduce((bi,v,i,a)=>v>a[bi]?i:bi,0);
    ui.cfStatus.textContent="PROBE COMPLETE · "+probe.trials.length+" exact branches × "+tics+" tics · "+(performance.now()-started).toFixed(1)+" ms · best "+policy.actions[best].id+" · state restored"+(probe.spread<=1e-9?" · no return separation, not collected":"");
  }catch(error){ui.cfStatus.textContent="probe failed · "+String(error?.message||error)}
  finally{counterfactualBusy=false;setBusy(false);render()}
}
function fitCounterfactualTargets(){
  if(!prepared||!counterfactualExamples.length)return;
  controller.pause();setBusy(true);
  try{
    const teacherBefore=policy.teacherCalls,result=policy.fitDecisionDistributions(counterfactualExamples,{ridge:.04,refineSteps:2,strength:.04});
    if(!result)throw new Error("No valid counterfactual targets to fit");
    lastCounterfactualFit={...result,t:Date.now(),examples:counterfactualExamples.length};
    ui.cfStatus.textContent="MEASURED TARGET FIT · "+counterfactualExamples.length+" states · "+Number(result.rows||0)+" action rows · "+Number(result.refineSteps||0)+" refinement passes · teacher calls "+(policy.teacherCalls-teacherBefore);
    checkpointReady=false;checkpointInfo=null;setRuntime("COUNTERFACTUAL FIT · FROZEN READY");
  }catch(error){ui.cfStatus.textContent="counterfactual fit failed · "+String(error?.message||error)}
  finally{setBusy(false);render()}
}
function clearCounterfactualTargets(){
  counterfactualExamples=[];lastCounterfactualProbe=null;lastCounterfactualFit=null;ui.cfStatus.textContent="Collected same-state supervision cleared; model weights unchanged.";render();
}
function render(){
  if(!env||!controller||!policy)return;
  const obs=env.lastObservation||env.observe();ui.objective.textContent=policy.schema.objective||env.schema.objective;ui.weaponState.textContent=weaponLabel(obs);
  ui.state.innerHTML=(policy.schema.fields||env.schema.fields).map(field=>'<div class="state-row"><span>'+field.label+'</span><strong>'+displayField(field,obs[field.id])+'</strong></div>').join("");
  ui.steps.textContent=controller.steps;ui.episodes.textContent=controller.episodes;ui.ret.textContent=controller.episodeReturn.toFixed(3);ui.updates.textContent=policy.q.updates;ui.teacherCalls.textContent=policy.teacherCalls;ui.decodeTemp.textContent=policy.temperature.toFixed(3);ui.replaySize.textContent=String(policy.replay?.length||0);ui.teacherReplaySize.textContent=String(policy.teacherReplay?.length||0);ui.valueBeta.textContent=Number(controller.lastDecision?.valueBeta||0).toFixed(3);ui.priorKL.textContent=Number(controller.lastDecision?.priorKL||0).toFixed(3);ui.klUtilization.textContent=(Number(controller.lastDecision?.klUtilization||0)*100).toFixed(0)+"%"+(controller.lastDecision?.valueBetaSaturated?" cap":"");
  ui.backbone.textContent=(policy.q.name||"neural")+" · "+policy.q.parameterCount()+" params";
  const lat=controller.latencySummary();ui.latLast.textContent=lat.last.toFixed(2)+" ms";ui.latP95.textContent=lat.p95.toFixed(2)+" ms";ui.latSemantic.textContent=(policy.lastTeacherLatencyMs||0).toFixed(1)+" ms";
  const outcome=controller.lastDecision?.outcome||env.lastOutcome;ui.damageDealt.textContent=Number(outcome?.damageDealt||0).toFixed(0);ui.hostileHpLoss.textContent=Number(outcome?.hostileHpLoss||0).toFixed(0);ui.damageReceived.textContent=Math.max(0,-Number(outcome?.healthDelta||0)).toFixed(0);ui.combatAttribution.textContent=outcome?.combatAttributionAvailable?"native":"unavailable";
  const d=controller.lastDecision;
  ui.probabilityCalibration.textContent=d?.probabilityCalibrated?("fitted · T×"+Number(d.probabilityCalibrationTemperature||1).toFixed(3)):"unverified on DOOM";
  if(d){
    const decisionObs=controller.trace.at(-1)?.observation||obs;
    const attended=policy.q.inspectAttention?.(decisionObs,d.actionIndex,{topK:6,temporal:d.temporal,history:d.history})||[];
    ui.attentionCount.textContent=attended.length+" records";
    ui.attention.innerHTML=attended.length?attended.map(item=>{
      const interesting=Object.entries(item.record||{}).filter(([,value])=>typeof value==="number"&&Number.isFinite(value)&&value!==0).sort((a,b)=>Math.abs(Number(b[1]))-Math.abs(Number(a[1]))).slice(0,5);
      const detail=interesting.map(([key,value])=>key+"="+Number(value).toFixed(Math.abs(value)<10?2:0)).join(" · ");
      return '<div class="attention-row"><div><strong>'+item.collectionId+'['+item.index+']</strong><small>'+detail+'</small></div><span>'+(item.weight*100).toFixed(1)+'%</span></div>';
    }).join(""):'<div class="attention-empty">No structured records in this observation.</div>';
    ui.lastReward.textContent=d.reward.toFixed(3);ui.chosen.textContent=d.action.label;ui.chosenSemantic.textContent=Number(d.semanticPriorScores?.[d.actionIndex]??0).toFixed(3);ui.chosenValue.textContent=Number(d.valueScores?.[d.actionIndex]??0).toFixed(3);ui.chosenScore.textContent=Number(d.qScores?.[d.actionIndex]??0).toFixed(3);ui.entropy.textContent=d.uncertainty.entropy.toFixed(3);ui.margin.textContent=d.uncertainty.margin.toFixed(3);ui.epistemic.textContent=(d.uncertainty.epistemic||0).toFixed(3);ui.novelty.textContent=d.uncertainty.novelty.toFixed(3);
    const marginal=field=>policy.actions.reduce((sum,action,i)=>sum+(Number(action.params?.[field]||0)>0?d.probs[i]:0),0);
    ui.intentFire.textContent=marginal("fire").toFixed(3);
    ui.intentStrafe.textContent=(marginal("strafe_left")+marginal("strafe_right")).toFixed(3);
    ui.intentTurn.textContent=(marginal("turn_left")+marginal("turn_right")).toFixed(3);
    ui.intentForward.textContent=marginal("forward").toFixed(3);
    ui.intentBack.textContent=marginal("back").toFixed(3);
    ui.intentUse.textContent=marginal("use").toFixed(3);
    [...ui.bars.children].forEach((row,i)=>{row.querySelector(".bar-fill").style.width=(d.probs[i]*100).toFixed(1)+"%";row.lastElementChild.textContent=d.probs[i].toFixed(3)});
    ui.log.textContent=controller.trace.slice(-14).reverse().map(t=>"s"+String(t.step).padStart(4,"0")+" "+(t.teacherUsed?"Q":t.teacherPending?"…":"·")+" "+t.action.padEnd(19)+" p="+Math.max(...Object.values(t.probabilities)).toFixed(3)+" r="+t.reward.toFixed(3)+" dmg="+Number(t.outcome?.damageDealt||0).toFixed(0)+" "+t.residualLatencyMs.toFixed(1)+"ms").join("\n");
  }
  renderDecisionCircuit(obs,d);renderArchitecture(obs,d,outcome);renderTypedFields(d);renderTrainingStream();renderTeacherTranscript();renderCounterfactual();
  updateReadiness();
}
function bindController(){
  controller.addEventListener("tick",render);
  controller.addEventListener("state",event=>{ui.start.textContent=event.detail==="RUNNING"?"Pause":"Play";setRuntime(event.detail);updateReadiness()});
  controller.addEventListener("teacherwait",event=>{
    const detail=event.detail||{},active=!!detail.active;
    if(ui.teacherPause){
      ui.teacherPause.hidden=!active;
      if(ui.teacherPauseReason)ui.teacherPauseReason.textContent=active?(detail.reason||"low confidence"):"resolved";
    }
    if(active)setRuntime(detail.simulationFrozen?"TEACHER PAUSE · SIM FROZEN":"TEACHER WAIT");
    else if(controller.state===ControllerState.RUNNING)setRuntime("RUNNING");
    renderTeacherTranscript();
  });
  controller.addEventListener("error",event=>{if(ui.teacherPause)ui.teacherPause.hidden=true;setRuntime("ERROR",true);ui.log.textContent="ERROR: "+event.detail.message+"\n"+ui.log.textContent});
}
async function primeCurrentTeacher(label,steps=12,{converge=false}={}){
  const obs=env.lastObservation||env.observe();ui.modelStatus.textContent=label+" · distilling current state";
  const prime=await policy.primeTeacher(obs,{steps,maxSteps:converge?256:steps,targetKL:converge?.02:null});
  if(prime?.stale)throw new Error("Semantic bootstrap became stale");
  return prime;
}
async function compileSchemaOnly(){
  ui.schemaStatus.textContent="loading "+SCHEMA_EMBEDDING_PRESET.label+" · "+SCHEMA_EMBEDDING_PRESET.approx;
  const compiler=new MiniLMSchemaCompiler({onProgress:info=>{ui.schemaStatus.textContent=(info.status||"working")+(info.message?" · "+info.message:"")}});
  try{
    await compiler.load();const compiled=await compiler.compile(env.schema,env.actions);policy.reconfigure({schema:compiled.schema,actions:compiled.actions});schemaCompiled=true;
    ui.schemaStatus.textContent="compiled · "+compiled.meta.bindings+" semantic bindings · "+compiled.meta.backend;return compiled.meta;
  }finally{await compiler.dispose().catch(()=>{})}
}
async function loadTeacherOnly(selected){
  if(selected==="hash"){
    if(policy.semantic!==hashSemantic&&policy.semantic?.dispose)await policy.semantic.dispose();policy.setSemantic(hashSemantic);teacherReady=false;return null;
  }
  const preset=NLI_PRESETS[selected];ui.modelStatus.textContent="loading "+preset.label+" · "+preset.approx;
  const labelBiasCalibration=selected==="mobilebert";
  const candidate=new TransformersNLIAdapter({preset:selected,maxStateChars:3000,labelBiasCalibration,onProgress:info=>{
    if(Number.isFinite(info.normalizedProgress))ui.modelProgress.value=info.normalizedProgress;
    ui.modelStatus.textContent=(info.status||"loading")+(info.file?" · "+info.file:"");
  }});
  candidate.compile(policy.schema,policy.actions);await candidate.load();
  if(policy.semantic!==hashSemantic&&policy.semantic?.dispose)await policy.semantic.dispose();
  policy.setSemantic(candidate);teacherReady=true;ui.modelProgress.value=100;
  ui.modelStatus.textContent=preset.label+" · "+candidate.backend+(labelBiasCalibration?" · null-state label-bias corrected":"");
  return candidate;
}
async function boot(){
  setBusy(true);ui.boot.disabled=true;setRuntime("LOADING");ui.bootStatus.textContent=requestedRuntime!=="borrowed"?"Fetching project-owned Chocolate Doom runtime…":"Fetching borrowed pinned Chocolate Doom runtime…";
  try{
    const file=ui.iwad.files?.[0]||null;if(file&&file.size>128*1024*1024)throw new Error("IWAD is larger than the 128 MB browser safety limit");
    const iwadFile=file?await file.arrayBuffer():null;
    const runtimeOptions=requestedRuntime!=="borrowed"
      ?await resolveOwnedRuntime().then(runtime=>({runtimeBase:runtime.base,runtimeInfo:runtime.info}))
      :{};
    env=await DoomWasmArena.boot({canvas:ui.canvas,actionMs:Number(ui.actionMs.value),iwadFile,contentName:file?.name||null,...runtimeOptions,onProgress:message=>{ui.bootStatus.textContent=message}});
    hashSemantic=new HashSemanticAdapter();hashSemantic.backend="local-js";
    policy=new SemanticResidualPolicy({schema:env.schema,actions:env.actions,semantic:hashSemantic,residual:"neural-set",seed:1993,inferenceMode:"adaptive"});
    policy.onTeacherResult=()=>{if(env&&controller){const obs=env.lastObservation||env.observe(),d=controller.lastDecision,outcome=d?.outcome||env.lastOutcome;renderTeacherTranscript();renderArchitecture(obs,d,outcome)}};
    controller=new ExperimentController({environment:env,policy,hz:8});controller.training=false;controller.explore=false;controller.memory=true;controller.useResidual=true;controller.freezeOnTeacher=true;
    bindController();buildBars();engineReady=true;ui.runtimeTitle.textContent=(env.runtime?.owned?"Owned ":"")+"Chocolate Doom · "+env.contentName;
    const counts=env.lastObservation?._collections||{},runtimeId=env.runtime?.commit?(" · runtime "+String(env.runtime.commit).slice(0,8)):"";ui.bootStatus.textContent=DOOM_RUNTIME_PROVENANCE.engine+" · "+env.contentName+runtimeId+" · "+policy.q.parameterCount()+" params · "+(counts.entities?.length||0)+" entities · "+(counts.geometry?.length||0)+" lines";
    ui.prepareStatus.textContent="Engine ready. Prepare Recommended before model-controlled play.";ui.schemaStatus.textContent="Lexical feature hash only.";ui.modelStatus.textContent="No learned teacher loaded.";
    applyProfile("assisted");setRuntime("ENGINE READY");window.__doomLab={get env(){return env},get policy(){return policy},get controller(){return controller},get counterfactual(){return lastCounterfactualProbe},get counterfactualExamples(){return counterfactualExamples}};render();
    if(starterMode==="auto")void maybeAutoLoadStarter();
  }catch(error){ui.boot.disabled=false;setRuntime("BOOT FAILED",true);ui.bootStatus.textContent=String(error?.message||error);ui.log.textContent=String(error?.stack||error)}
  finally{setBusy(false)}
}
async function importPortableCheckpoint(checkpoint,label="checkpoint"){
  if(!engineReady||!policy||!controller)return;
  controller.pause();setBusy(true);
  try{
    if(policy.semantic&&policy.semantic!==hashSemantic&&policy.semantic?.dispose)await policy.semantic.dispose().catch(()=>{});
    policy.setSemantic(hashSemantic);teacherReady=false;
    policy.importCheckpoint(checkpoint);schemaCompiled=true;checkpointReady=true;checkpointInfo=checkpoint.build||null;
    await controller.reset({learning:false});applyProfile("frozen");ui.profile.value="frozen";
    ui.modelSelect.value="hash";ui.modelStatus.textContent="Teacher not required · loaded frozen checkpoint.";
    ui.schemaStatus.textContent="Compiled semantic schema restored from checkpoint.";
    ui.prepareStatus.textContent="READY TO PLAY · "+label+" · "+policy.q.parameterCount()+" params · teacher-free neural fast path";
    const summary=checkpointSummary(checkpointInfo);ui.checkpointStatus.textContent=summary?(label+" loaded · "+(summary.kind==="causal"?"causal policy":"quality-gated stage "+summary.stage)+" · "+summary.decisions+" training decisions · validation "+summary.damage.toFixed(0)+" damage / "+summary.kills.toFixed(0)+" kills / return "+summary.reward.toFixed(3)+" · "+policy.q.parameterCount()+" params"):(label+" loaded · "+policy.q.parameterCount()+" params · "+policy.q.updates+" consequence updates");
    setRuntime("CHECKPOINT READY");render();
  }catch(error){ui.checkpointStatus.textContent="checkpoint load failed · "+String(error?.message||error);setRuntime("CHECKPOINT ERROR",true)}
  finally{setBusy(false)}
}
function checkpointSummary(build){
  if(!build)return null;
  if(build.selection==="exact-state causal policy curriculum"||build.selection==="canonical exact-state causal policy curriculum"||Number.isFinite(Number(build.causalSteps))){
    const validation=build.validation||{},training=build.training||{};
    return{
      kind:"causal",stage:null,decisions:Number(build.causalSteps||0),
      reward:Number(validation.reward??training.return??0),
      damage:Number(validation.damage??training.damage??0),
      kills:Number(validation.kills??training.kills??0)
    };
  }
  const stage=Number(build.selectedStage),selected=build.candidates?.find?.(x=>Number(x.stage)===stage),decisions=Number(build.trainingDecisions||stage*64||0);
  if(!Number.isFinite(stage)||!selected)return null;
  return{kind:"staged",stage,decisions,reward:Number(selected.reward||0),damage:Number(selected.damage||0),kills:Number(selected.kills||0)};
}
function starterCheckpointUsable(checkpoint){
  const summary=checkpointSummary(checkpoint?.build);
  return !!summary&&summary.reward>0&&(summary.kills>=1||summary.damage>=20);
}
async function fetchPublishedCombatReport(){
  const response=await fetch(STARTER_MODEL_BASE+"/combat-report.json",{cache:"no-cache"});
  if(!response.ok)throw new Error("combat report HTTP "+response.status);
  return response.json();
}
async function fetchPublishedReplayBundle(){
  const response=await fetch(STARTER_MODEL_BASE+"/combat-replays.json",{cache:"no-cache"});
  if(!response.ok)throw new Error("combat replay HTTP "+response.status);
  const bundle=await response.json();
  if(!Array.isArray(bundle?.runs)||!bundle.runs.length)throw new Error("combat replay bundle is empty");
  return bundle;
}
function installReplayBundle(bundle){
  publishedReplayBundle=bundle;
  if(ui.publishedReplaySelect){
    ui.publishedReplaySelect.innerHTML=bundle.runs.map((run,i)=>'<option value="'+i+'">run '+(i+1)+' · '+Number(run.kills||0)+' kills · '+Number(run.damage||0)+' dmg · '+Number(run.steps||run.actions?.length||0)+' decisions</option>').join("");
    ui.publishedReplaySelect.value="0";
  }
  if(ui.publishedReplayStatus){
    const kills=bundle.runs.reduce((s,r)=>s+Number(r.kills||0),0),damage=bundle.runs.reduce((s,r)=>s+Number(r.damage||0),0);
    ui.publishedReplayStatus.textContent="Published replay bundle · "+bundle.runs.length+" runs · "+kills+" kills · "+damage+" damage · exact "+Number(bundle.actionTics||4)+"-tic packets";
  }
  updateReadiness();
}
async function fetchStarterCheckpoint(){
  const urls=[STARTER_MODEL_BASE+"/doom-starter.json",GENERIC_STARTER_BASE+"/doom-starter.json","./models/doom-starter.json"];
  let last=null;
  for(const url of urls){
    try{
      const response=await fetch(url,{cache:"no-cache"});
      if(response.ok){
        const checkpoint=await response.json();
        if(url.includes("model-runtime")){
          try{
            const report=await fetchPublishedCombatReport();
            if(report?.evaluation&&checkpoint?.build){
              checkpoint.build.validation={
                reward:Number(report.evaluation.totalReward||0),
                damage:Number(report.evaluation.totalDamage||0),
                kills:Number(report.evaluation.totalKills||0)
              };
            }
          }catch{}
        }
        if(!starterCheckpointUsable(checkpoint))throw new Error("published starter failed minimum combat-quality gate");
        return{checkpoint,url};
      }
      last=new Error("HTTP "+response.status+" from "+url);
    }catch(error){last=error}
  }
  throw last||new Error("validated starter not published yet");
}
async function loadBundledCheckpoint({silent=false}={}){
  try{
    if(!silent)ui.checkpointStatus.textContent="Fetching validated starter…";
    const {checkpoint,url}=await fetchStarterCheckpoint();
    await importPortableCheckpoint(checkpoint,"validated starter");
    const summary=checkpointSummary(checkpointInfo);ui.checkpointStatus.textContent=(summary?("Validated starter · "+(summary.kind==="causal"?"causal policy":"stage "+summary.stage)+" / "+summary.decisions+" decisions · "+summary.damage.toFixed(0)+" damage · "+summary.kills.toFixed(0)+" kills · return "+summary.reward.toFixed(3)):"Validated starter loaded")+" · "+policy.q.parameterCount()+" params · source "+(url.includes("model-runtime")?"model-runtime causal snapshot":url.includes("starter-runtime")?"starter-runtime staged snapshot":"local bundle");
    if(url.includes("model-runtime"))try{installReplayBundle(await fetchPublishedReplayBundle())}catch(error){if(ui.publishedReplayStatus)ui.publishedReplayStatus.textContent="Replay bundle unavailable · "+String(error?.message||error)}
    return true;
  }catch(error){
    if(!silent)ui.checkpointStatus.textContent="Validated starter unavailable · "+String(error?.message||error);
    return false;
  }
}
async function maybeAutoLoadStarter(){
  if(starterMode==="off"||!engineReady||checkpointReady)return false;
  ui.checkpointStatus.textContent="Checking for validated starter checkpoint…";
  const loaded=await loadBundledCheckpoint({silent:true});
  if(!loaded)ui.checkpointStatus.textContent="No published starter yet · use Prepare Recommended to bootstrap from the generic core.";
  return loaded;
}
async function replayPublishedRun(){
  if(replayBusy||!engineReady)return;
  try{
    if(!publishedReplayBundle)installReplayBundle(await fetchPublishedReplayBundle());
    const index=Math.max(0,Math.min(publishedReplayBundle.runs.length-1,Number(ui.publishedReplaySelect?.value||0))),run=publishedReplayBundle.runs[index],tics=Number(publishedReplayBundle.actionTics||4);
    if(!run?.actions?.length)throw new Error("selected replay has no actions");
    replayBusy=true;controller.pause();await controller.quiesce({teacher:true});updateReadiness();
    await env.reset();policy.resetEpisode();
    const canonicalStart=Number(publishedReplayBundle?.version||0)>=3||/canonical/i.test(String(publishedReplayBundle?.verification||""));
    if(canonicalStart&&env.supportsSnapshots?.()){
      const startSnapshot=env.saveSnapshot();
      env.restoreSnapshot(startSnapshot);policy.resetEpisode();
    }
    if(ui.publishedReplayStatus)ui.publishedReplayStatus.textContent="Replaying run "+(index+1)+" · 0 / "+run.actions.length;
    let kills=0,damage=0;
    for(let i=0;i<run.actions.length;i++){
      const item=run.actions[i],actionId=typeof item==="string"?item:item.action,actionIndex=policy.actions.findIndex(a=>a.id===actionId);
      const step=env.stepTics(actionId,tics),outcome=step.info?.outcome||null;
      policy.commitActionOutcome?.({actionIndex,reward:step.reward,outcome,done:step.done});
      kills+=Number(outcome?.playerKillDelta||0);damage+=Number(outcome?.damageDealt||0);
      if(i%2===0||i+1===run.actions.length){
        if(ui.publishedReplayStatus)ui.publishedReplayStatus.textContent="Replaying run "+(index+1)+" · "+(i+1)+" / "+run.actions.length+" · "+kills+" kills · "+damage+" damage";
        render();await new Promise(resolve=>setTimeout(resolve,70));
      }
      if(step.done)break;
    }
    if(ui.publishedReplayStatus)ui.publishedReplayStatus.textContent="Replay complete · run "+(index+1)+" · "+kills+" kills · "+damage+" damage · expected "+Number(run.kills||0)+" / "+Number(run.damage||0);
  }catch(error){
    if(ui.publishedReplayStatus)ui.publishedReplayStatus.textContent="Replay failed · "+String(error?.message||error);
  }finally{
    replayBusy=false;updateReadiness();render();
  }
}
ui.publishedReplayBtn?.addEventListener("click",replayPublishedRun);

async function loadBrowserCheckpoint(){
  try{const raw=localStorage.getItem("doom-classifier-checkpoint-v1");if(!raw)throw new Error("no browser-saved checkpoint");await importPortableCheckpoint(JSON.parse(raw),"browser-saved checkpoint")}
  catch(error){ui.checkpointStatus.textContent="browser checkpoint load failed · "+String(error?.message||error)}
}
function saveBrowserCheckpoint(){
  try{const checkpoint=policy.exportCheckpoint();localStorage.setItem("doom-classifier-checkpoint-v1",JSON.stringify(checkpoint));ui.checkpointStatus.textContent="Saved in this browser · "+checkpoint.q.params+" params";updateReadiness()}
  catch(error){ui.checkpointStatus.textContent="checkpoint save failed · "+String(error?.message||error)}
}
function exportPortableCheckpoint(){
  try{
    const checkpoint=policy.exportCheckpoint(),blob=new Blob([JSON.stringify(checkpoint)],{type:"application/json"}),a=document.createElement("a");
    a.href=URL.createObjectURL(blob);a.download="doom-classifier-checkpoint-"+Date.now()+".json";a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);
    ui.checkpointStatus.textContent="Checkpoint exported · "+checkpoint.q.params+" params";
  }catch(error){ui.checkpointStatus.textContent="checkpoint export failed · "+String(error?.message||error)}
}
async function prepareRecommended(){
  if(!engineReady)return;controller.pause();setBusy(true);ui.prepareStatus.textContent="Preparing semantic schema…";
  try{
    if(!schemaCompiled)await compileSchemaOnly();
    ui.prepareStatus.textContent="Loading MobileBERT semantic teacher…";ui.modelSelect.value="mobilebert";
    if(!isLearnedTeacher()||policy.semantic.presetKey!=="mobilebert")await loadTeacherOnly("mobilebert");
    checkpointReady=false;checkpointInfo=null;policy.resetLearning();await controller.reset({learning:false});const prime=await primeCurrentTeacher("MobileBERT",16,{converge:true});
    const fit=prime.distillation?" · fit "+prime.distillation.stepsUsed+" steps · KL "+prime.distillation.kl.toFixed(3):"";
    applyProfile("assisted");ui.prepareStatus.textContent="READY TO PLAY · schema compiled · MobileBERT "+prime.ms.toFixed(0)+" ms"+fit+" · fast path is neural";
    ui.tuneStatus.textContent="Optional: run a short real-Doom training burst, then evaluate teacher-off.";setRuntime("READY TO PLAY");render();
  }catch(error){ui.prepareStatus.textContent="Preparation failed · "+String(error?.message||error);setRuntime("PREP FAILED",true)}
  finally{setBusy(false)}
}
async function compileSchemaSemantics(){
  if(!engineReady)return;controller.pause();setBusy(true);
  try{
    await compileSchemaOnly();policy.resetLearning();await controller.reset({learning:false});
    if(isLearnedTeacher())await primeCurrentTeacher("compiled schema",12);
    ui.prepareStatus.textContent=isLearnedTeacher()?"Schema recompiled and teacher re-primed.":"Schema compiled. Load a learned teacher to complete preparation.";render();
  }catch(error){ui.schemaStatus.textContent="compile failed · "+String(error?.message||error)}
  finally{setBusy(false)}
}
async function switchTeacher(){
  if(!engineReady)return;controller.pause();setBusy(true);ui.modelProgress.value=0;
  try{
    await loadTeacherOnly(ui.modelSelect.value);policy.resetLearning();await controller.reset({learning:false});
    if(isLearnedTeacher()){const prime=await primeCurrentTeacher(policy.semantic.name,12,{converge:true});ui.modelStatus.textContent=policy.semantic.name+" ready · bootstrap "+prime.ms.toFixed(0)+" ms · fit "+(prime.distillation?.stepsUsed||12)+" steps"}
    else ui.modelStatus.textContent="Hash baseline selected — useful only for ablation.";
    render();
  }catch(error){policy.setSemantic(hashSemantic);teacherReady=false;ui.modelSelect.value="hash";ui.modelProgress.value=0;ui.modelStatus.textContent="teacher load failed · "+String(error?.message||error);render()}
  finally{setBusy(false)}
}
async function tuneAgent(){
  if(!prepared)return;controller.pause();setBusy(true);applyProfile("learning");ui.tuneProgress.value=0;ui.tuneBadge.textContent="training";
  const steps=Number(ui.tuneSteps.value)||128,startUpdates=policy.q.updates,startTeacher=policy.teacherCalls;
  try{
    const result=await controller.trainBurst({steps,epsilon:.16,rolloutHorizon:64,onProgress:p=>{ui.tuneProgress.value=p.ratio*100;ui.tuneStatus.textContent="training "+p.completed+"/"+p.total+" · updates "+(p.updates-startUpdates)+" · rollout restarts "+p.rolloutRestarts}});
    applyProfile("frozen");ui.profile.value="frozen";ui.tuneBadge.textContent="frozen neural";
    ui.tuneStatus.textContent="TRAINING COMPLETE · "+result.completed+" decisions across "+(result.rolloutRestarts+1)+" rollouts · "+result.actionDiversity+" actions explored · "+result.updates+" reward updates · "+(policy.teacherCalls-startTeacher)+" teacher refreshes · ready for teacher-off playback";
    setRuntime("TRAINED · FROZEN");render();
  }catch(error){ui.tuneStatus.textContent="training failed · "+String(error?.message||error);setRuntime("TRAINING ERROR",true)}
  finally{setBusy(false)}
}
async function evaluateFrozen(){
  if(!prepared||!controller)return;
  controller.pause();setBusy(true);applyProfile("frozen");ui.profile.value="frozen";ui.evalResults.innerHTML="<span>Running 48 teacher-off decisions from a fresh encounter…</span>";
  try{
    await controller.reset({learning:false});syncRuntimeConfig();
    const teacherBefore=policy.teacherCalls,actions={},samples=[];let reward=0,damageDealt=0,damageReceived=0,kills=0,fireActions=0;
    for(let i=0;i<48;i++){
      const ok=await controller.tick();if(!ok&&controller.state==="ERROR")throw new Error("controller error during evaluation");
      const d=controller.lastDecision;if(!d)continue;
      reward+=Number(d.reward||0);damageDealt+=Number(d.outcome?.damageDealt||0);damageReceived+=Math.max(0,-Number(d.outcome?.healthDelta||0));kills+=Number(d.outcome?.killDelta||0);
      actions[d.action.id]=(actions[d.action.id]||0)+1;if(d.action.id.includes("fire"))fireActions++;
      samples.push({maxP:Math.max(...d.probs),entropy:d.uncertainty.entropy});
    }
    const mean=key=>samples.length?samples.reduce((s,x)=>s+x[key],0)/samples.length:0,lat=controller.latencySummary();
    const dominant=Object.entries(actions).sort((a,b)=>b[1]-a[1])[0]||["—",0],diversity=Object.keys(actions).length,teacherDelta=policy.teacherCalls-teacherBefore;
    ui.evalResults.innerHTML=
      '<div><span>return</span><strong>'+reward.toFixed(3)+'</strong></div>'+
      '<div><span>damage dealt / received</span><strong>'+damageDealt.toFixed(0)+' / '+damageReceived.toFixed(0)+'</strong></div>'+
      '<div><span>kills</span><strong>'+kills.toFixed(0)+'</strong></div>'+
      '<div><span>fire-capable choices</span><strong>'+fireActions+' / 48</strong></div>'+
      '<div><span>action diversity</span><strong>'+diversity+' / '+policy.actions.length+'</strong></div>'+
      '<div><span>dominant action</span><strong>'+dominant[0]+' · '+dominant[1]+'</strong></div>'+
      '<div><span>mean max probability</span><strong>'+mean("maxP").toFixed(3)+'</strong></div>'+
      '<div><span>mean entropy</span><strong>'+mean("entropy").toFixed(3)+'</strong></div>'+
      '<div><span>p95 decision</span><strong>'+lat.p95.toFixed(2)+' ms</strong></div>'+
      '<div><span>teacher calls</span><strong>'+teacherDelta+'</strong></div>';
    ui.tuneStatus.textContent="Frozen evaluation complete. Teacher calls must remain 0; compare this panel before/after training.";
    setRuntime("FROZEN EVALUATION COMPLETE");render();
  }catch(error){ui.evalResults.innerHTML="<span>Evaluation failed · "+String(error?.message||error)+"</span>";setRuntime("EVALUATION ERROR",true)}
  finally{setBusy(false)}
}

async function manualPrimitive(){
  if(!engineReady)return;controller.pause();setBusy(true);
  try{
    const id=ui.manualAction.value,result=await env.step(id);policy.resetEpisode();
    ui.manualStatus.textContent=id+" executed · reward "+result.reward.toFixed(3)+" · damage dealt "+Number(result.info?.outcome?.damageDealt||0).toFixed(0);
    if(result.done)await env.reset();render();
  }catch(error){ui.manualStatus.textContent="manual action failed · "+String(error?.message||error)}
  finally{setBusy(false)}
}

ui.iwad.addEventListener("change",()=>{const file=ui.iwad.files?.[0];ui.iwadStatus.textContent=file?"Selected local IWAD: "+file.name+" · "+(file.size/1048576).toFixed(1)+" MB":"Default: Freedoom 0.13.0"});
ui.boot.addEventListener("click",boot);ui.prepare.addEventListener("click",prepareRecommended);ui.tune.addEventListener("click",tuneAgent);ui.eval.addEventListener("click",evaluateFrozen);ui.manual.addEventListener("click",manualPrimitive);ui.cfProbe.addEventListener("click",probeSameState);ui.cfFit.addEventListener("click",fitCounterfactualTargets);ui.cfClear.addEventListener("click",clearCounterfactualTargets);ui.loadStarter.addEventListener("click",loadBundledCheckpoint);ui.loadSaved.addEventListener("click",loadBrowserCheckpoint);ui.saveCheckpoint.addEventListener("click",saveBrowserCheckpoint);ui.exportCheckpoint.addEventListener("click",exportPortableCheckpoint);
ui.start.addEventListener("click",()=>{if(!controller)return;if(controller.state==="RUNNING")controller.pause();else{syncRuntimeConfig();controller.start()}});
ui.step.addEventListener("click",async()=>{if(!controller||!prepared)return;if(controller.state==="RUNNING")controller.pause();syncRuntimeConfig();await controller.tick()});
ui.reset.addEventListener("click",async()=>{if(controller){await controller.reset({learning:false});render()}});
ui.applyProfile.addEventListener("click",()=>applyProfile());ui.profile.addEventListener("change",()=>{ui.profileHint.textContent=(PROFILES[ui.profile.value]||PROFILES.assisted).hint});
for(const element of [ui.teacherMode,ui.useNeural,ui.learn,ui.memory,ui.explore])element.addEventListener("change",syncRuntimeConfig);
ui.loadModel.addEventListener("click",switchTeacher);ui.schemaCompile.addEventListener("click",compileSchemaSemantics);
ui.actionMs.addEventListener("input",()=>{ui.actionMsOut.value=ui.actionMs.value+" ms";env?.setActionMs(ui.actionMs.value)});
ui.export.addEventListener("click",()=>{if(!controller)return;const blob=new Blob([controller.exportTrace()],{type:"application/json"}),a=document.createElement("a");a.href=URL.createObjectURL(blob);a.download="doom-neural-trace-"+Date.now()+".json";a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)});
updateReadiness();
