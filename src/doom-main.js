import {DoomWasmArena,DOOM_RUNTIME_PROVENANCE} from "./env/doom-wasm.js";
import {HashSemanticAdapter} from "./core/semantic.js";
import {SemanticResidualPolicy} from "./core/policy.js";
import {ExperimentController} from "./core/controller.js";
import {TransformersNLIAdapter,NLI_PRESETS} from "./model-adapters/transformers-nli.js";
import {MiniLMSchemaCompiler,SCHEMA_EMBEDDING_PRESET} from "./model-adapters/schema-embedding-compiler.js";

const OWNED_RUNTIME_BASE="https://raw.githubusercontent.com/collinsomniac/doom-classifier/engine-runtime";
const requestedRuntime=new URLSearchParams(globalThis.location?.search||"").get("runtime")||"owned";
const $=id=>document.getElementById(id);
const ui={
  boot:$("bootBtn"),prepare:$("prepareBtn"),start:$("startBtn"),step:$("stepBtn"),reset:$("resetBtn"),canvas:$("doomCanvas"),runtime:$("runtimeStatus"),runtimeTitle:$("runtimeTitle"),bootStatus:$("bootStatus"),
  iwad:$("iwadInput"),iwadStatus:$("iwadStatus"),engineChip:$("engineChip"),schemaChip:$("schemaChip"),teacherChip:$("teacherChip"),policyChip:$("policyChip"),prepareStatus:$("prepareStatus"),
  profile:$("profileSelect"),applyProfile:$("applyProfileBtn"),profileHint:$("profileHint"),teacherMode:$("teacherModeSelect"),useNeural:$("useNeuralToggle"),learn:$("learnToggle"),memory:$("memoryToggle"),explore:$("exploreToggle"),
  tune:$("tuneBtn"),eval:$("evalBtn"),evalResults:$("evalResults"),tuneSteps:$("tuneSteps"),tuneProgress:$("tuneProgress"),tuneStatus:$("tuneStatus"),tuneBadge:$("tuneBadge"),
  actionMs:$("actionMs"),actionMsOut:$("actionMsOut"),manualAction:$("manualActionSelect"),manual:$("manualBtn"),manualStatus:$("manualStatus"),weaponState:$("weaponState"),
  bars:$("actionBars"),chosen:$("chosenAction"),chosenSemantic:$("chosenSemantic"),chosenValue:$("chosenValue"),chosenScore:$("chosenScore"),intentFire:$("intentFire"),intentStrafe:$("intentStrafe"),intentTurn:$("intentTurn"),intentForward:$("intentForward"),intentBack:$("intentBack"),intentUse:$("intentUse"),entropy:$("entropy"),margin:$("margin"),epistemic:$("epistemic"),novelty:$("novelty"),latLast:$("latLast"),latSemantic:$("latSemantic"),latP95:$("latP95"),
  attention:$("attentionList"),attentionCount:$("attentionCount"),teacherCalls:$("teacherCalls"),decodeTemp:$("decodeTemp"),replaySize:$("replaySize"),teacherReplaySize:$("teacherReplaySize"),valueBeta:$("valueBeta"),priorKL:$("priorKL"),klUtilization:$("klUtilization"),backbone:$("backboneName"),modelSelect:$("modelSelect"),loadModel:$("loadModelBtn"),modelProgress:$("modelProgress"),modelStatus:$("modelStatus"),
  schemaCompile:$("schemaCompileBtn"),schemaStatus:$("schemaStatus"),state:$("stateTable"),objective:$("objectiveText"),steps:$("steps"),episodes:$("episodes"),ret:$("return"),updates:$("updates"),lastReward:$("lastReward"),damageDealt:$("damageDealt"),hostileHpLoss:$("hostileHpLoss"),damageReceived:$("damageReceived"),combatAttribution:$("combatAttribution"),
  architecture:$("architectureFlow"),archMode:$("archMode"),archFeedback:$("archFeedback"),trainingModeBadge:$("trainingModeBadge"),trainingOutput:$("trainingOutput"),teacherTranscript:$("teacherTranscript"),
  log:$("eventLog"),export:$("exportBtn"),dot:$("statusDot")
};
let env=null,policy=null,controller=null,hashSemantic=null;
let engineReady=false,schemaCompiled=false,teacherReady=false,busy=false,prepared=false;

const PROFILES={
  assisted:{label:"Adaptive assisted",teacher:"adaptive",neural:true,learning:false,memory:true,explore:false,hint:"Fast neural decisions every tick; MobileBERT is scheduled only when uncertainty/novelty warrants it."},
  teacher:{label:"Teacher-only zero-shot",teacher:"every",neural:false,learning:false,memory:true,explore:false,hint:"MobileBERT scores every decision. Useful semantic baseline, but it pays teacher latency every tick."},
  learning:{label:"Online learning",teacher:"adaptive",neural:true,learning:true,memory:true,explore:true,hint:"The neural controller samples from its own uncertainty (with a small uniform floor), learns real consequences, and receives adaptive semantic supervision."},
  frozen:{label:"Frozen neural evaluation",teacher:"off",neural:true,learning:false,memory:true,explore:false,hint:"No teacher calls, no weight updates, no random exploration. This is the clean small-model evaluation mode."}
};

function setChip(element,state,text){
  element.className="ready-chip "+state;element.textContent=text;
}
function isLearnedTeacher(){return teacherReady&&policy?.semantic&&policy.semantic!==hashSemantic}
function updateReadiness(){
  prepared=engineReady&&schemaCompiled&&isLearnedTeacher();
  setChip(ui.engineChip,engineReady?"ready":"off",engineReady?"engine ready":"engine");
  setChip(ui.schemaChip,schemaCompiled?"ready":"off",schemaCompiled?"schema compiled":"schema semantics");
  setChip(ui.teacherChip,isLearnedTeacher()?"ready":"off",isLearnedTeacher()?"teacher loaded":"teacher");
  setChip(ui.policyChip,prepared?"ready":"warn",prepared?"ready to play":"not prepared");
  const lock=busy||!engineReady;
  ui.prepare.disabled=lock;ui.loadModel.disabled=lock;ui.schemaCompile.disabled=lock;ui.manual.disabled=lock;ui.manualAction.disabled=lock;ui.reset.disabled=lock;ui.export.disabled=lock;
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
function setArchNode(id,{active=false,hot=false,pending=false,learning=false,value=null}={}){
  const node=ui.architecture?.querySelector('[data-arch="'+id+'"]');if(!node)return;
  node.classList.toggle("active",!!active);node.classList.toggle("hot",!!hot);node.classList.toggle("pending",!!pending);node.classList.toggle("learning",!!learning);
  if(value!==null){const target=node.querySelector(".arch-value");if(target)target.textContent=value}
}
function renderArchitecture(obs,d,outcome){
  if(!ui.architecture)return;
  const entities=obs?._collections?.entities?.length||0,geometry=obs?._collections?.geometry?.length||0,lastTeacher=policy?.lastTeacherResult;
  const teacherFresh=!!lastTeacher&&Math.abs(Number(policy?.decisionCount||0)-Number(lastTeacher.step||0))<=2;
  const learning=!!controller?.training||!!d?.learningInfo;
  setArchNode("environment",{active:engineReady,hot:!!d,value:engineReady?(env.runtime?.owned?"DOOM · owned causal ABI":"DOOM · borrowed telemetry"):"not mounted"});
  setArchNode("state",{active:engineReady,hot:!!d,value:engineReady?(policy.schema.fields.length+" fields · "+entities+" entities · "+geometry+" lines"):"facts + records"});
  setArchNode("encoder",{active:prepared||!!d,hot:!!d,value:policy?(policy.q.parameterCount()+" params · memory "+(controller?.memory?"on":"off")):"shared representation"});
  setArchNode("teacher",{active:isLearnedTeacher(),hot:!!d?.teacherUsed||teacherFresh,pending:!!d?.teacherPending,value:isLearnedTeacher()?(policy.semantic.name+" · "+Number(policy.lastTeacherLatencyMs||0).toFixed(0)+" ms"):"off"});
  setArchNode("semantic",{active:prepared||!!d,hot:!!d?.semanticUsed||teacherFresh,value:d?"chosen "+Number(d.semanticPriorScores?.[d.actionIndex]??0).toFixed(3):"unprepared"});
  setArchNode("replay",{active:(policy?.replay?.length||0)>0||learning,hot:learning,learning,value:(policy?.replay?.length||0)+" transitions · n="+(policy?.nStep||1)});
  setArchNode("value",{active:(policy?.q?.updates||0)>0||!!d,hot:learning,learning,value:d?"chosen "+Number(d.valueScores?.[d.actionIndex]??0).toFixed(3)+" · "+(policy.q.updates||0)+" updates":"untrained"});
  setArchNode("fusion",{active:!!d,hot:!!d&&Number(d.valueBeta||0)>0,value:d?"β "+Number(d.valueBeta||0).toFixed(2)+" · KL "+(Number(d.klUtilization||0)*100).toFixed(0)+"%":"β 0 · KL 0%"});
  setArchNode("actions",{active:!!d,hot:!!d,value:d?(d.action.label+" · p "+Number(d.probs?.[d.actionIndex]||0).toFixed(3)):"waiting"});
  setArchNode("actuator",{active:!!d,hot:!!d,value:d?("primitive mask "+String(env.actionMasks?.[d.action.id]??"—")):"idle"});
  ui.archMode.textContent=!engineReady?"waiting for engine":learning?"learning":policy?.inferenceMode==="neural"?"frozen / neural":policy?.inferenceMode==="hybrid"?"teacher in decode path":"adaptive supervision";
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
    const td=learn?" td="+Number(learn.td||0).toFixed(3)+" H"+Number(learn.nStepHorizon||0):"";
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
    const calibration=item.calibration?('<span>T '+Number(item.calibration.temperature||0).toFixed(3)+'</span>'):"";
    return '<article class="teacher-item '+(index===0?"latest":"")+'"><div class="teacher-item-head"><strong>'+escapeHtml(item.model)+' · '+escapeHtml(item.kind)+'</strong><span>s'+String(item.step).padStart(4,"0")+' · '+Number(item.ms||0).toFixed(0)+' ms</span></div><div class="teacher-reason">'+escapeHtml(item.reason)+'</div><div class="teacher-actions">'+actions+'</div><div class="teacher-meta">'+fit+calibration+'</div><details><summary>state sent to teacher</summary><pre class="teacher-state">'+escapeHtml(item.stateText||"state serializer unavailable")+'</pre></details></article>';
  }).join("");
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
  if(d){
    const decisionObs=controller.trace.at(-1)?.observation||obs;
    const attended=policy.q.inspectAttention?.(decisionObs,d.actionIndex,{topK:6,temporal:d.temporal})||[];
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
  renderArchitecture(obs,d,outcome);renderTrainingStream();renderTeacherTranscript();
  updateReadiness();
}
function bindController(){
  controller.addEventListener("tick",render);
  controller.addEventListener("state",event=>{ui.start.textContent=event.detail==="RUNNING"?"Pause":"Play";setRuntime(event.detail);updateReadiness()});
  controller.addEventListener("error",event=>{setRuntime("ERROR",true);ui.log.textContent="ERROR: "+event.detail.message+"\n"+ui.log.textContent});
}
async function primeCurrentTeacher(label,steps=12,{converge=false}={}){
  const obs=env.lastObservation||env.observe();ui.modelStatus.textContent=label+" · distilling current state";
  const prime=await policy.primeTeacher(obs,{steps,maxSteps:converge?128:steps,targetKL:converge?.02:null});
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
  const candidate=new TransformersNLIAdapter({preset:selected,maxStateChars:3000,onProgress:info=>{
    if(Number.isFinite(info.normalizedProgress))ui.modelProgress.value=info.normalizedProgress;
    ui.modelStatus.textContent=(info.status||"loading")+(info.file?" · "+info.file:"");
  }});
  candidate.compile(policy.schema,policy.actions);await candidate.load();
  if(policy.semantic!==hashSemantic&&policy.semantic?.dispose)await policy.semantic.dispose();
  policy.setSemantic(candidate);teacherReady=true;ui.modelProgress.value=100;return candidate;
}
async function boot(){
  setBusy(true);ui.boot.disabled=true;setRuntime("LOADING");ui.bootStatus.textContent=requestedRuntime!=="borrowed"?"Fetching project-owned Chocolate Doom runtime…":"Fetching borrowed pinned Chocolate Doom runtime…";
  try{
    const file=ui.iwad.files?.[0]||null;if(file&&file.size>128*1024*1024)throw new Error("IWAD is larger than the 128 MB browser safety limit");
    const iwadFile=file?await file.arrayBuffer():null;
    const runtimeOptions=requestedRuntime!=="borrowed"?{runtimeBase:OWNED_RUNTIME_BASE,runtimeInfo:{owned:true,repository:"collinsomniac/doom-classifier",branch:"engine-runtime",base:OWNED_RUNTIME_BASE}}:{};
    env=await DoomWasmArena.boot({canvas:ui.canvas,actionMs:Number(ui.actionMs.value),iwadFile,contentName:file?.name||null,...runtimeOptions,onProgress:message=>{ui.bootStatus.textContent=message}});
    hashSemantic=new HashSemanticAdapter();hashSemantic.backend="local-js";
    policy=new SemanticResidualPolicy({schema:env.schema,actions:env.actions,semantic:hashSemantic,residual:"neural-set",seed:1993,inferenceMode:"adaptive"});
    controller=new ExperimentController({environment:env,policy,hz:8});controller.training=false;controller.explore=false;controller.memory=true;controller.useResidual=true;
    bindController();buildBars();engineReady=true;ui.runtimeTitle.textContent=(env.runtime?.owned?"Owned ":"")+"Chocolate Doom · "+env.contentName;
    const counts=env.lastObservation?._collections||{};ui.bootStatus.textContent=DOOM_RUNTIME_PROVENANCE.engine+" · "+env.contentName+" · "+policy.q.parameterCount()+" params · "+(counts.entities?.length||0)+" entities · "+(counts.geometry?.length||0)+" lines";
    ui.prepareStatus.textContent="Engine ready. Prepare Recommended before model-controlled play.";ui.schemaStatus.textContent="Lexical feature hash only.";ui.modelStatus.textContent="No learned teacher loaded.";
    applyProfile("assisted");setRuntime("ENGINE READY");window.__doomLab={get env(){return env},get policy(){return policy},get controller(){return controller}};render();
  }catch(error){ui.boot.disabled=false;setRuntime("BOOT FAILED",true);ui.bootStatus.textContent=String(error?.message||error);ui.log.textContent=String(error?.stack||error)}
  finally{setBusy(false)}
}
async function prepareRecommended(){
  if(!engineReady)return;controller.pause();setBusy(true);ui.prepareStatus.textContent="Preparing semantic schema…";
  try{
    if(!schemaCompiled)await compileSchemaOnly();
    ui.prepareStatus.textContent="Loading MobileBERT semantic teacher…";ui.modelSelect.value="mobilebert";
    if(!isLearnedTeacher()||policy.semantic.presetKey!=="mobilebert")await loadTeacherOnly("mobilebert");
    policy.resetLearning();await controller.reset({learning:false});const prime=await primeCurrentTeacher("MobileBERT",16,{converge:true});
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
    const result=await controller.trainBurst({steps,epsilon:.16,onProgress:p=>{ui.tuneProgress.value=p.ratio*100;ui.tuneStatus.textContent="training "+p.completed+"/"+p.total+" · updates "+(p.updates-startUpdates)+" · episodes "+p.episodes}});
    applyProfile("frozen");ui.profile.value="frozen";ui.tuneBadge.textContent="frozen neural";
    ui.tuneStatus.textContent="TRAINING COMPLETE · "+result.completed+" decisions · "+result.actionDiversity+" actions explored · "+result.switches+" switches · max streak "+result.maxStreak+" · "+result.updates+" reward updates · "+(policy.teacherCalls-startTeacher)+" teacher refreshes · ready for teacher-off playback";
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
ui.boot.addEventListener("click",boot);ui.prepare.addEventListener("click",prepareRecommended);ui.tune.addEventListener("click",tuneAgent);ui.eval.addEventListener("click",evaluateFrozen);ui.manual.addEventListener("click",manualPrimitive);
ui.start.addEventListener("click",()=>{if(!controller)return;if(controller.state==="RUNNING")controller.pause();else{syncRuntimeConfig();controller.start()}});
ui.step.addEventListener("click",async()=>{if(!controller||!prepared)return;if(controller.state==="RUNNING")controller.pause();syncRuntimeConfig();await controller.tick()});
ui.reset.addEventListener("click",async()=>{if(controller){await controller.reset({learning:false});render()}});
ui.applyProfile.addEventListener("click",()=>applyProfile());ui.profile.addEventListener("change",()=>{ui.profileHint.textContent=(PROFILES[ui.profile.value]||PROFILES.assisted).hint});
for(const element of [ui.teacherMode,ui.useNeural,ui.learn,ui.memory,ui.explore])element.addEventListener("change",syncRuntimeConfig);
ui.loadModel.addEventListener("click",switchTeacher);ui.schemaCompile.addEventListener("click",compileSchemaSemantics);
ui.actionMs.addEventListener("input",()=>{ui.actionMsOut.value=ui.actionMs.value+" ms";env?.setActionMs(ui.actionMs.value)});
ui.export.addEventListener("click",()=>{if(!controller)return;const blob=new Blob([controller.exportTrace()],{type:"application/json"}),a=document.createElement("a");a.href=URL.createObjectURL(blob);a.download="doom-neural-trace-"+Date.now()+".json";a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)});
updateReadiness();
