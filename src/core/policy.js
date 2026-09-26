import {argmax,clamp,entropyNormalized,mulberry32,sampleCategorical,softmax} from "./math.js";
import {TemporalMemory} from "./memory.js";
import {LinearResidualQ} from "./residual.js";
import {NeuralSetResidualQ} from "./neural-set-residual.js";
import {projectValueToActionFields} from "./action-factorization.js";

function numericFeatures(schema,obs){
  const out=[1];
  for(const f of schema.fields){
    const v=Number(obs[f.id]??0),min=f.min??0,max=f.max??1;
    out.push(clamp(max===min?0:(v-min)/(max-min),0,1)*2-1);
  }
  return new Float32Array(out);
}
class NoveltyTracker{
  constructor(size){this.n=0;this.mean=new Float32Array(size);this.m2=new Float32Array(size)}
  reset(){this.n=0;this.mean.fill(0);this.m2.fill(0)}
  observe(x){
    if(this.n<4){this.n++;for(let i=0;i<x.length;i++){const d=x[i]-this.mean[i];this.mean[i]+=d/this.n;this.m2[i]+=d*(x[i]-this.mean[i])}return 1}
    let z=0;
    for(let i=0;i<x.length;i++){const variance=this.m2[i]/Math.max(1,this.n-1)+.02,d=x[i]-this.mean[i];z+=Math.min(9,d*d/variance)}
    const score=Math.tanh(Math.sqrt(z/x.length)/2);this.n++;
    for(let i=0;i<x.length;i++){const d=x[i]-this.mean[i];this.mean[i]+=d/this.n;this.m2[i]+=d*(x[i]-this.mean[i])}
    return score;
  }
}
function ensembleDisagreement(memberScores,temperature=1){
  if(!memberScores||!memberScores.length)return 0;
  const members=memberScores[0]?.length||0,actions=memberScores.length;
  if(members<2||actions<2)return 0;
  const distributions=Array.from({length:members},(_,m)=>softmax(memberScores.map(row=>row[m]),temperature));
  const mean=new Float32Array(actions);
  for(const p of distributions)for(let a=0;a<actions;a++)mean[a]+=p[a]/members;
  let js=0;
  for(const p of distributions){
    for(let a=0;a<actions;a++)if(p[a]>0)js+=p[a]*Math.log(p[a]/Math.max(1e-9,mean[a]));
  }
  return Math.max(0,Math.min(1,(js/members)/Math.log(actions)));
}
function klDivergence(target,predicted){
  let value=0;for(let i=0;i<target.length;i++)if(target[i]>0)value+=target[i]*Math.log(target[i]/Math.max(1e-9,predicted[i]));return value;
}
function crossEntropy(target,predicted){
  let loss=0;for(let i=0;i<target.length;i++)if(target[i]>0)loss-=target[i]*Math.log(Math.max(1e-9,predicted[i]));return loss;
}
function confidenceStats(probs){
  const sorted=[...probs].sort((a,b)=>b-a);
  return{entropy:entropyNormalized(probs),margin:(sorted[0]??0)-(sorted[1]??0)};
}

export class SemanticResidualPolicy{
  constructor({
    schema,actions,semantic,residual="linear",residualWeight=.9,temperature=.8,epsilon=.08,seed=2026,
    inferenceMode="hybrid",teacherInterval=32,teacherMinGap=8,teacherEntropy=.78,teacherMargin=.10,teacherNovelty=.85,teacherEpistemic=.025,distillSteps=4,replayCapacity=96,replayBatch=2,
    teacherReplayCapacity=8,teacherReplayBatch=2,teacherReplayStrength=.2,nStep=4,
    priorKlBudget=.08,valueBetaMax=4096,valueBetaSearchSteps=12,valueTrustUpdates=192,valueEpistemicBudget=.025,typedValueBlend=.65,typedValueRidge=.05,
    criticKlExpansion=1,criticKlFloor=.10,criticAgreementFloor=.67,criticSnrFloor=.75,criticSnrTarget=2,criticGapShareTarget=.25
  }){
    this.schema=schema;this.actions=actions;this.residualWeight=residualWeight;this.baseTemperature=temperature;this.temperature=temperature;this.epsilon=epsilon;this.seed=seed;this.rng=mulberry32(seed);
    this.baseSize=1+schema.fields.length;this.memory=new TemporalMemory(this.baseSize);this.featureSize=this.baseSize*2;
    if(typeof residual==="object")this.q=residual;
    else if(residual==="neural-set")this.q=new NeuralSetResidualQ(schema,actions,{seed});
    else this.q=new LinearResidualQ(actions.length,this.featureSize);
    this.inferenceMode=inferenceMode;this.teacherInterval=teacherInterval;this.teacherMinGap=teacherMinGap;this.teacherEntropy=teacherEntropy;this.teacherMargin=teacherMargin;this.teacherNovelty=teacherNovelty;this.teacherEpistemic=teacherEpistemic;this.distillSteps=distillSteps;this.replayCapacity=replayCapacity;this.replayBatch=replayBatch;this.replay=[];this.replayRng=mulberry32((seed^0x517cc1b7)>>>0);
    this.teacherReplayCapacity=Math.max(0,Math.floor(teacherReplayCapacity));this.teacherReplayBatch=Math.max(0,Math.floor(teacherReplayBatch));this.teacherReplayStrength=teacherReplayStrength;this.teacherReplay=[];this.teacherReplayRng=mulberry32((seed^0xa341316c)>>>0);this.nStep=Math.max(1,Math.floor(nStep));this.nStepBuffer=[];
    this.priorKlBudget=Math.max(0,Number(priorKlBudget)||0);this.valueBetaMax=Math.max(0,Number(valueBetaMax)||0);this.valueBetaSearchSteps=Math.max(1,Math.floor(valueBetaSearchSteps));this.valueTrustUpdates=Math.max(1,Math.floor(valueTrustUpdates));this.valueEpistemicBudget=clamp(Number(valueEpistemicBudget??.025),0,1);this.typedValueBlend=clamp(Number(typedValueBlend??.65),0,1);this.typedValueRidge=Math.max(1e-6,Number(typedValueRidge)||.05);
    this.criticKlExpansion=Math.max(0,Number(criticKlExpansion)||0);this.criticKlFloor=clamp(Number(criticKlFloor??.10),0,1);this.criticAgreementFloor=clamp(Number(criticAgreementFloor??.67),0,.99);this.criticSnrFloor=Math.max(0,Number(criticSnrFloor??.75));this.criticSnrTarget=Math.max(this.criticSnrFloor+1e-6,Number(criticSnrTarget??2));this.criticGapShareTarget=Math.max(1e-6,Number(criticGapShareTarget??.25));
    this.novelty=new NoveltyTracker(this.baseSize);
    this.decisionCount=0;this.lastTeacherStep=-1e9;this.teacherCalls=0;this.semanticCalls=0;this.teacherGeneration=0;this.teacherPromise=null;this.teacherScheduled=0;this.lastTeacherLatencyMs=0;this.lastTeacherError=null;this.lastTeacherResult=null;this.teacherHistory=[];this.onTeacherResult=null;
    this.setSemantic(semantic);
  }
  setSemantic(semantic){
    this.teacherGeneration++;this.teacherPromise=null;this.semantic=semantic;semantic.compile(this.schema,this.actions);this.lastTeacherStep=-1e9;this.lastTeacherError=null;this.lastTeacherResult=null;this.teacherHistory=[];this.teacherReplay=[];
  }
  setInferenceMode(mode){
    if(!["hybrid","adaptive","neural"].includes(mode))throw new Error("Unknown inference mode: "+mode);
    this.inferenceMode=mode;
  }
  reconfigure({schema=this.schema,actions=this.actions}={}){
    this.teacherGeneration++;this.teacherPromise=null;
    this.schema=schema;this.actions=actions;
    this.baseSize=1+schema.fields.length;this.memory=new TemporalMemory(this.baseSize);this.featureSize=this.baseSize*2;this.novelty=new NoveltyTracker(this.baseSize);
    if(this.q.setSchema&&this.q.setActions){this.q.setSchema(schema);this.q.setActions(actions)}
    else if(this.q instanceof LinearResidualQ){this.q=new LinearResidualQ(actions.length,this.featureSize)}
    this.semantic.compile(schema,actions);this.lastTeacherStep=-1e9;this.lastTeacherError=null;this.lastTeacherResult=null;this.teacherHistory=[];this.replay=[];this.teacherReplay=[];this.nStepBuffer=[];
    return this;
  }
  resetEpisode(){this.memory.reset();this.nStepBuffer=[]}
  resetLearning(){
    this.teacherGeneration++;this.teacherPromise=null;this.q.reset();this.novelty.reset();this.rng=mulberry32(this.seed);this.replayRng=mulberry32((this.seed^0x517cc1b7)>>>0);this.teacherReplayRng=mulberry32((this.seed^0xa341316c)>>>0);this.replay=[];this.teacherReplay=[];this.nStepBuffer=[];this.temperature=this.baseTemperature;
    this.decisionCount=0;this.lastTeacherStep=-1e9;this.teacherCalls=0;this.semanticCalls=0;this.teacherScheduled=0;this.lastTeacherLatencyMs=0;this.lastTeacherError=null;this.lastTeacherResult=null;this.teacherHistory=[];
  }
  exportCheckpoint(){
    if(!this.q?.exportCheckpoint)throw new Error("Residual model does not support checkpoints");
    const clone=value=>JSON.parse(JSON.stringify(value,(_key,v)=>ArrayBuffer.isView(v)?Array.from(v):v));
    return{
      format:"doom-classifier-policy",version:1,createdAt:new Date().toISOString(),
      schema:clone(this.schema),actions:clone(this.actions),temperature:this.temperature,baseTemperature:this.baseTemperature,
      inferenceMode:"neural",q:this.q.exportCheckpoint()
    };
  }
  importCheckpoint(checkpoint){
    if(checkpoint?.format!=="doom-classifier-policy"||checkpoint.version!==1)throw new Error("Unsupported policy checkpoint");
    if(!checkpoint.schema||!Array.isArray(checkpoint.actions))throw new Error("Checkpoint is missing schema/actions");
    this.reconfigure({schema:checkpoint.schema,actions:checkpoint.actions});
    if(!this.q?.importCheckpoint)throw new Error("Residual model does not support checkpoints");
    this.q.importCheckpoint(checkpoint.q);
    this.temperature=clamp(Number(checkpoint.temperature||this.baseTemperature),.05,2);
    this.baseTemperature=clamp(Number(checkpoint.baseTemperature||this.temperature),.05,2);
    this.inferenceMode="neural";this.teacherGeneration++;this.teacherPromise=null;this.lastTeacherStep=-1e9;this.lastTeacherError=null;this.lastTeacherResult=null;this.teacherHistory=[];
    return this;
  }
  encode(obs,memoryEnabled=true,commit=true){
    const base=numericFeatures(this.schema,obs),temporal=commit?this.memory.update(base,memoryEnabled):this.memory.preview(base,memoryEnabled);
    const x=new Float32Array(this.featureSize);x.set(base,0);x.set(temporal,this.baseSize);return{base,temporal,features:x};
  }
  fusePriorValue(priorScores,valueScores,valueMemberScores=null,{temperature=this.temperature}={}){
    if(!priorScores?.length||!valueScores?.length||priorScores.length!==valueScores.length){
      return{scores:priorScores||[],memberScores:null,valueBeta:0,priorKL:0,valueTrust:0,priorKlBudget:0,klUtilization:0,valueEpistemic:0,valueEpistemicBudget:this.valueEpistemicBudget,epistemicUtilization:0,valueBetaSaturated:false};
    }
    const valueTrust=clamp(Number(this.q?.updates||0)/this.valueTrustUpdates,0,1),baseBudget=this.priorKlBudget*valueTrust,temp=Math.max(.05,Number(temperature)||1),epistemicBudget=this.valueEpistemicBudget;
    const prior=softmax(priorScores,temp),typed=projectValueToActionFields(this.schema,this.actions,valueScores,{ridge:this.typedValueRidge,maxBlend:this.typedValueBlend}),effectiveValue=typed.scores;
    const center=effectiveValue.reduce((a,b)=>a+b,0)/effectiveValue.length,centered=effectiveValue.map(v=>v-center);
    const hasMembers=valueMemberScores?.length===priorScores.length&&Array.isArray(valueMemberScores[0])&&valueMemberScores[0].length>1;
    let effectiveMembers=null;
    if(hasMembers){
      const memberCount=valueMemberScores[0].length,projected=Array.from({length:memberCount},(_,member)=>projectValueToActionFields(this.schema,this.actions,valueMemberScores.map(row=>Number(row?.[member]||0)),{ridge:this.typedValueRidge,maxBlend:this.typedValueBlend}).scores);
      effectiveMembers=priorScores.map((_,a)=>projected.map(scores=>scores[a]));
    }
    const criticRank=[...effectiveValue.keys()].sort((a,b)=>effectiveValue[b]-effectiveValue[a]),criticTop=criticRank[0]??0,criticRunner=criticRank[1]??criticTop,criticGap=Math.max(0,Number(effectiveValue[criticTop]||0)-Number(effectiveValue[criticRunner]||0));
    const criticMin=Math.min(...effectiveValue),criticMax=Math.max(...effectiveValue),criticSpread=Math.max(0,criticMax-criticMin),criticGapShare=criticSpread>1e-9?clamp(criticGap/criticSpread,0,1):0;
    let criticTopAgreement=0,criticMarginMean=criticGap,criticMarginStd=0,criticMarginSnr=0;
    if(effectiveMembers?.[criticTop]?.length){
      const members=effectiveMembers[criticTop].length,margins=[];let agree=0;
      for(let member=0;member<members;member++){
        let memberTop=0;for(let action=1;action<effectiveMembers.length;action++)if(Number(effectiveMembers[action]?.[member]||0)>Number(effectiveMembers[memberTop]?.[member]||0))memberTop=action;
        if(memberTop===criticTop)agree++;
        margins.push(Number(effectiveMembers[criticTop]?.[member]||0)-Number(effectiveMembers[criticRunner]?.[member]||0));
      }
      criticTopAgreement=agree/members;criticMarginMean=margins.reduce((a,b)=>a+b,0)/members;
      criticMarginStd=Math.sqrt(margins.reduce((s,v)=>s+(v-criticMarginMean)**2,0)/members);
      criticMarginSnr=Math.abs(criticMarginMean)/(criticMarginStd+1e-6);
    }
    const agreementGate=hasMembers?clamp((criticTopAgreement-this.criticAgreementFloor)/(1-this.criticAgreementFloor),0,1):0;
    const snrGate=hasMembers?clamp((criticMarginSnr-this.criticSnrFloor)/(this.criticSnrTarget-this.criticSnrFloor),0,1):0;
    const gapGate=clamp(criticGapShare/this.criticGapShareTarget,0,1);
    const criticRankingConfidence=agreementGate*snrGate*gapGate,criticAuthority=valueTrust*criticRankingConfidence;
    const criticKlGate=this.criticKlFloor+(1-this.criticKlFloor)*criticRankingConfidence;
    const criticKlMultiplier=criticKlGate*(1+this.criticKlExpansion*criticAuthority),budget=baseBudget*criticKlMultiplier;
    const evaluate=beta=>{
      const scores=priorScores.map((v,i)=>v+beta*centered[i]),probs=softmax(scores,temp),memberScores=hasMembers?priorScores.map((semantic,a)=>Array.from({length:valueMemberScores[0].length},(_,m)=>semantic+beta*Number(effectiveMembers[a]?.[m]||0))):null;
      const epistemic=memberScores?ensembleDisagreement(memberScores,temp):0;
      return{beta,scores,probs,memberScores,kl:klDivergence(probs,prior),epistemic};
    };
    const allowed=x=>x.kl<=budget+1e-12&&x.epistemic<=epistemicBudget+1e-12;
    let chosen=evaluate(0),valueBetaSaturated=false;
    if(budget>0&&this.valueBetaMax>0){
      let lo=0,hi=Math.min(1,this.valueBetaMax),candidate=evaluate(hi);
      while(allowed(candidate)&&hi<this.valueBetaMax){
        chosen=candidate;lo=hi;
        const next=Math.min(this.valueBetaMax,hi*2);
        if(next===hi)break;
        hi=next;candidate=evaluate(hi);
      }
      if(allowed(candidate)){
        chosen=candidate;valueBetaSaturated=hi>=this.valueBetaMax&&candidate.kl<budget*.98&&candidate.epistemic<epistemicBudget*.98;
      }else{
        for(let i=0;i<this.valueBetaSearchSteps;i++){
          const mid=(lo+hi)/2,test=evaluate(mid);
          if(allowed(test)){lo=mid;chosen=test}else hi=mid;
        }
      }
    }
    const klUtilization=budget>0?clamp(chosen.kl/budget,0,1):0,epistemicUtilization=epistemicBudget>0?clamp(chosen.epistemic/epistemicBudget,0,1):0;
    return{scores:chosen.scores,memberScores:chosen.memberScores,valueBeta:chosen.beta,priorKL:chosen.kl,valueTrust,basePriorKlBudget:baseBudget,priorKlBudget:budget,klUtilization,valueEpistemic:chosen.epistemic,valueEpistemicBudget:epistemicBudget,epistemicUtilization,valueBetaSaturated,priorProbs:prior,typedValueScores:effectiveValue,typedValueProjected:typed.projected,typedValueFit:typed.fitQuality,typedValueBlendUsed:typed.blendUsed,typedFieldCoefficients:typed.coefficients,criticTopIndex:criticTop,criticRunnerIndex:criticRunner,criticGap,criticSpread,criticGapShare,criticTopAgreement,criticMarginMean,criticMarginStd,criticMarginSnr,criticRankingConfidence,criticAuthority,criticKlGate,criticKlMultiplier};
  }
  residualEvaluation(obs,features,temporal=null){
    if(this.q.scoreStatsObservation){
      const raw=this.q.scoreStatsObservation(obs,{temporal});
      if(raw.semanticScores&&raw.valueScores){
        const fused=this.fusePriorValue(raw.semanticScores,raw.valueScores,raw.valueMemberScores,{temperature:this.temperature});
        return{...raw,...fused,rawScores:raw.scores};
      }
      return raw;
    }
    const scores=this.q.scoresObservation?this.q.scoresObservation(obs,{temporal}):this.q.scores(features);
    return{scores,memberScores:null,valueBeta:0,priorKL:0,valueTrust:0,priorKlBudget:0,klUtilization:0,valueBetaSaturated:false};
  }
  residualScores(obs,features,temporal=null){return this.residualEvaluation(obs,features,temporal).scores}
  async semanticScores(obs,semantic=this.semantic){
    const t=performance.now(),scores=await semantic.score(obs);this.semanticCalls++;return{scores,ms:performance.now()-t};
  }
  teacherTriggerReason(provisional,novelty){
    if(this.teacherPromise)return null;
    const gap=this.decisionCount-this.lastTeacherStep;
    if(this.teacherCalls===0)return "initial supervision";
    if(gap>=this.teacherInterval)return "scheduled refresh";
    if(gap<this.teacherMinGap)return null;
    const reasons=[];
    if(provisional.entropy>=this.teacherEntropy)reasons.push("high entropy");
    if(provisional.margin<=this.teacherMargin)reasons.push("small margin");
    if(novelty>=this.teacherNovelty)reasons.push("novel state");
    if(provisional.epistemic>=this.teacherEpistemic)reasons.push("value disagreement");
    return reasons.length?reasons.join(" + "):null;
  }
  shouldTeacher(provisional,novelty){return !!this.teacherTriggerReason(provisional,novelty)}
  recordTeacherResult(obs,scores,{kind="supervision",reason="teacher call",ms=0,distillation=null,calibration=null,step=this.decisionCount}={}){
    if(!scores?.length)return null;
    const probs=softmax(scores,1),ranked=this.actions.map((action,i)=>({id:action.id,label:action.label,score:Number(scores[i]||0),probability:Number(probs[i]||0)})).sort((a,b)=>b.probability-a.probability);
    let stateText="";
    try{if(typeof this.semantic?.stateText==="function")stateText=String(this.semantic.stateText(obs)||"")}catch{}
    const item={
      id:(this.teacherHistory.at(-1)?.id||0)+1,t:Date.now(),step:Number(step||0),kind,reason,
      model:this.semantic?.name||"semantic teacher",ms:Number(ms||0),top:ranked.slice(0,6),
      distillation:distillation?{stepsUsed:Number(distillation.stepsUsed||0),headFitKL:Number(distillation.headFitKL??NaN),kl:Number(distillation.kl||0),loss:Number(distillation.loss||0),teacherReplayUpdates:Number(distillation.teacherReplayUpdates||0)}:null,
      calibration:calibration?{temperature:Number(calibration.temperature||0),bestTemperature:Number(calibration.bestTemperature||0),loss:Number(calibration.loss||0)}:null,
      stateText:stateText.slice(0,1800)
    };
    this.lastTeacherResult=item;this.teacherHistory.push(item);if(this.teacherHistory.length>24)this.teacherHistory.shift();try{this.onTeacherResult?.(item)}catch{}return item;
  }
  calibrateTemperature(obs,teacherScores,temporal=null,{blend=.35}={}){
    if(!teacherScores?.length)return null;
    const encoded=this.encode(obs,true,false),raw=this.q.scoreStatsObservation?.(obs,{temporal}),logits=raw?.semanticScores||this.residualScores(obs,encoded.features,temporal),target=softmax(teacherScores,1);
    let bestT=this.temperature,bestLoss=Infinity;
    const candidates=[.30,.40,.50,.65,.80,1.0,1.25,1.5];
    for(const t of candidates){const loss=crossEntropy(target,softmax(logits,t));if(loss<bestLoss){bestLoss=loss;bestT=t}}
    this.temperature=clamp(this.temperature*(1-blend)+bestT*blend,.30,1.5);
    return{temperature:this.temperature,bestTemperature:bestT,loss:bestLoss};
  }
  snapshotTeacherExample(obs,scores,temporal=null){
    const observation=globalThis.structuredClone?globalThis.structuredClone(obs):JSON.parse(JSON.stringify(obs));
    return{observation,scores:[...scores],temporal:temporal?new Float32Array(temporal):null};
  }
  replayTeacherDistillation(){
    if(!this.q.distill||this.teacherReplayBatch<=0||!this.teacherReplay.length)return{updates:0,meanLoss:0};
    const count=Math.min(this.teacherReplayBatch,this.teacherReplay.length);let loss=0;
    for(let i=0;i<count;i++){
      const sample=this.teacherReplay[Math.floor(this.teacherReplayRng()*this.teacherReplay.length)];
      const result=this.q.distill(sample.observation,sample.scores,{strength:this.teacherReplayStrength,temporal:sample.temporal});
      loss+=Number(result?.loss||0);
    }
    return{updates:count,meanLoss:count?loss/count:0};
  }
  rememberTeacherExample(obs,scores,temporal=null){
    if(this.teacherReplayCapacity<=0)return;
    this.teacherReplay.push(this.snapshotTeacherExample(obs,scores,temporal));
    if(this.teacherReplay.length>this.teacherReplayCapacity)this.teacherReplay.shift();
  }
  applyTeacherScores(obs,scores,steps=this.distillSteps,temporal=null,{maxSteps=steps,targetKL=null,strength=.5}={}){
    if(!this.q.distill)return null;
    const replay=this.replayTeacherDistillation(),teacher=softmax(scores,1),current=this.snapshotTeacherExample(obs,scores,temporal);
    const headFit=this.q.fitSemanticHead?.([...this.teacherReplay,current])||null;
    let result=null,used=0,kl=Infinity,headFitKL=Infinity;
    if(headFit&&this.q.scoreStatsObservation){
      const student=softmax(this.q.scoreStatsObservation(obs,{temporal}).semanticScores,1);
      headFitKL=klDivergence(teacher,student);kl=headFitKL;result={teacher,student,loss:crossEntropy(teacher,student),headFit};
    }
    const cap=Math.max(steps,Math.floor(maxSteps||steps)),alreadyFit=targetKL!=null&&Number.isFinite(kl)&&kl<=targetKL;
    if(!alreadyFit){
      for(let i=0;i<cap;i++){
        result=this.q.distill(obs,scores,{strength,temporal});used=i+1;
        if(result?.student){kl=klDivergence(teacher,result.student);if(used>=steps&&targetKL!=null&&kl<=targetKL)break}
        if(used>=steps&&targetKL==null)break;
      }
    }
    this.rememberTeacherExample(obs,scores,temporal);
    this.q.syncTarget?.({value:false});
    return result?{...result,headFit,headFitKL,stepsUsed:used,kl,teacherReplayUpdates:replay.updates,teacherReplayMeanLoss:replay.meanLoss,teacherReplaySize:this.teacherReplay.length}:null;
  }
  async primeTeacher(obs,{steps=Math.max(4,this.distillSteps),maxSteps=steps,targetKL=null,temporal=null}={}){
    const semantic=this.semantic,generation=this.teacherGeneration,requestedStep=this.decisionCount;
    const result=await this.semanticScores(obs,semantic);
    if(generation!==this.teacherGeneration||semantic!==this.semantic)return{stale:true,ms:result.ms};
    const distillation=this.applyTeacherScores(obs,result.scores,steps,temporal,{maxSteps,targetKL,strength:2});const calibration=this.calibrateTemperature(obs,result.scores,temporal,{blend:.65});
    this.teacherCalls++;this.lastTeacherStep=requestedStep;this.lastTeacherLatencyMs=result.ms;this.lastTeacherError=null;
    this.recordTeacherResult(obs,result.scores,{kind:"bootstrap",reason:"semantic bootstrap",ms:result.ms,distillation,calibration,step:requestedStep});
    return{stale:false,ms:result.ms,scores:result.scores,calibration,distillation};
  }
  scheduleTeacher(obs,temporal=null,reason="adaptive refresh"){
    if(this.teacherPromise)return false;
    const semantic=this.semantic,generation=this.teacherGeneration,requestedStep=this.decisionCount;
    this.teacherScheduled++;
    let task;
    task=(async()=>{
      try{
        const result=await this.semanticScores(obs,semantic);
        if(generation!==this.teacherGeneration||semantic!==this.semantic)return{stale:true,ms:result.ms};
        const distillation=this.applyTeacherScores(obs,result.scores,this.distillSteps,temporal),calibration=this.calibrateTemperature(obs,result.scores,temporal,{blend:.18});
        this.teacherCalls++;this.lastTeacherStep=requestedStep;this.lastTeacherLatencyMs=result.ms;this.lastTeacherError=null;
        this.recordTeacherResult(obs,result.scores,{kind:"adaptive",reason,ms:result.ms,distillation,calibration,step:requestedStep});
        return{stale:false,ms:result.ms,scores:result.scores,distillation,calibration};
      }catch(error){
        if(generation===this.teacherGeneration){this.lastTeacherError=error}
        return{stale:generation!==this.teacherGeneration,error};
      }finally{
        if(this.teacherPromise===task)this.teacherPromise=null;
      }
    })();
    this.teacherPromise=task;
    return true;
  }
  async awaitTeacher(){return this.teacherPromise?this.teacherPromise:null}

  async decide(obs,{useResidual=true,memory=true,explore=false}={}){
    const t0=performance.now(),encoded=this.encode(obs,memory,true),novelty=this.novelty.observe(encoded.base);
    const residualStart=performance.now(),residualEval=this.residualEvaluation(obs,encoded.features,encoded.temporal);let q=residualEval.scores,residualMs=performance.now()-residualStart;
    let activeFusion=residualEval,epistemic=ensembleDisagreement(residualEval.memberScores,this.temperature);
    let sem=new Array(this.actions.length).fill(0),semanticMs=0,teacherUsed=false,semanticUsed=false,logits;
    const mode=useResidual?this.inferenceMode:"hybrid";

    if(mode==="hybrid"){
      const result=await this.semanticScores(obs);sem=result.scores;semanticMs=result.ms;semanticUsed=true;
      this.teacherCalls++;this.lastTeacherStep=this.decisionCount;this.lastTeacherLatencyMs=result.ms;this.lastTeacherError=null;
      this.recordTeacherResult(obs,sem,{kind:"decode",reason:"teacher in decode path",ms:result.ms,step:this.decisionCount});
      if(useResidual&&residualEval.valueScores){
        const direct=this.fusePriorValue(sem,residualEval.valueScores,residualEval.valueMemberScores,{temperature:this.temperature});
        activeFusion={...residualEval,...direct};logits=direct.scores;epistemic=ensembleDisagreement(direct.memberScores,this.temperature);
      }else logits=useResidual?sem.map((v,i)=>v+this.residualWeight*q[i]):sem;
    }else if(mode==="adaptive"){
      const provisional={...confidenceStats(softmax(q,this.temperature)),epistemic},teacherReason=this.teacherTriggerReason(provisional,novelty);
      teacherUsed=teacherReason?this.scheduleTeacher(obs,encoded.temporal,teacherReason):false;
      logits=q;
    }else{
      logits=q;
    }

    const decodeTemperature=mode==="hybrid"&&!useResidual?1:this.temperature;
    const probs=softmax(logits,decodeTemperature);let chosen=argmax(probs),explorationStrategy="greedy";
    if(explore){
      const mix=clamp(this.epsilon,0,1),uniform=1/Math.max(1,probs.length);
      const sampling=probs.map(p=>(1-mix)*p+mix*uniform);
      chosen=sampleCategorical(sampling,this.rng);explorationStrategy="policy-proportional";
    }
    const stats=confidenceStats(probs);this.decisionCount++;
    return{
      actionIndex:chosen,action:this.actions[chosen],probs,semanticScores:sem,qScores:logits,semanticPriorScores:activeFusion.semanticScores||residualEval.semanticScores||null,valueScores:residualEval.valueScores||null,features:encoded.features,temporal:encoded.temporal,
      valueBeta:Number(activeFusion.valueBeta||0),priorKL:Number(activeFusion.priorKL||0),valueTrust:Number(activeFusion.valueTrust||0),basePriorKlBudget:Number(activeFusion.basePriorKlBudget||0),priorKlBudget:Number(activeFusion.priorKlBudget||0),klUtilization:Number(activeFusion.klUtilization||0),valueEpistemic:Number(activeFusion.valueEpistemic||0),valueEpistemicBudget:Number(activeFusion.valueEpistemicBudget??this.valueEpistemicBudget),epistemicUtilization:Number(activeFusion.epistemicUtilization||0),valueBetaSaturated:!!activeFusion.valueBetaSaturated,typedValueScores:activeFusion.typedValueScores||null,typedValueFit:Number(activeFusion.typedValueFit||0),typedValueBlendUsed:Number(activeFusion.typedValueBlendUsed||0),typedFieldCoefficients:activeFusion.typedFieldCoefficients||null,criticTopIndex:Number(activeFusion.criticTopIndex??-1),criticRunnerIndex:Number(activeFusion.criticRunnerIndex??-1),criticGap:Number(activeFusion.criticGap||0),criticSpread:Number(activeFusion.criticSpread||0),criticGapShare:Number(activeFusion.criticGapShare||0),criticTopAgreement:Number(activeFusion.criticTopAgreement||0),criticMarginMean:Number(activeFusion.criticMarginMean||0),criticMarginStd:Number(activeFusion.criticMarginStd||0),criticMarginSnr:Number(activeFusion.criticMarginSnr||0),criticRankingConfidence:Number(activeFusion.criticRankingConfidence||0),criticAuthority:Number(activeFusion.criticAuthority||0),criticKlMultiplier:Number(activeFusion.criticKlMultiplier||1),
      uncertainty:{...stats,novelty,epistemic},latencyMs:performance.now()-t0,semanticLatencyMs:semanticMs,residualLatencyMs:residualMs,
      teacherUsed,teacherPending:!!this.teacherPromise,semanticUsed,inferenceMode:mode,explorationStrategy,teacherCalls:this.teacherCalls,teacherScheduled:this.teacherScheduled,lastTeacherLatencyMs:this.lastTeacherLatencyMs
    };
  }
  aggregateNStep(count=Math.min(this.nStep,this.nStepBuffer.length)){
    if(!this.nStepBuffer.length)return null;
    const gamma=Number(this.q?.gamma??.96),first=this.nStepBuffer[0];let reward=0,discount=1,last=first,horizon=0;
    for(let i=0;i<count&&i<this.nStepBuffer.length;i++){
      const t=this.nStepBuffer[i];reward+=discount*Number(t.reward||0);discount*=gamma;last=t;horizon++;
      if(t.done)break;
    }
    return{...first,reward,nextObservation:last.nextObservation,nextTemporal:last.nextTemporal,nextFeatures:last.nextFeatures,done:!!last.done,bootstrapDiscount:last.done?0:discount,nStepHorizon:horizon};
  }
  applyLearningTransition(transition){
    const primary=this.q.updateTransition(transition);
    if(this.replayCapacity>0){
      this.replay.push(transition);if(this.replay.length>this.replayCapacity)this.replay.shift();
    }
    let replayUpdates=0,replayAbsTd=0;
    const candidates=Math.max(0,this.replay.length-1),count=Math.min(this.replayBatch,candidates);
    for(let i=0;i<count;i++){
      const index=Math.floor(this.replayRng()*candidates),sample=this.replay[index],result=this.q.updateTransition(sample);
      replayUpdates++;replayAbsTd+=Math.abs(Number(result?.td||0));
    }
    return{...primary,replayUpdates,replayMeanAbsTd:replayUpdates?replayAbsTd/replayUpdates:0,replaySize:this.replay.length,nStepHorizon:transition.nStepHorizon||1};
  }
  drainNStep({flush=false}={}){
    let primaryUpdates=0,replayUpdates=0,replayAbsTd=0,last=null;
    while(this.nStepBuffer.length&&(flush||this.nStepBuffer.length>=this.nStep)){
      const aggregate=this.aggregateNStep(Math.min(this.nStep,this.nStepBuffer.length));if(!aggregate)break;
      const result=this.applyLearningTransition(aggregate);last=result;primaryUpdates++;replayUpdates+=result.replayUpdates||0;replayAbsTd+=(result.replayMeanAbsTd||0)*(result.replayUpdates||0);
      this.nStepBuffer.shift();
      if(!flush&&this.nStepBuffer.length<this.nStep)break;
    }
    return last?{...last,primaryUpdates,replayUpdates,replayMeanAbsTd:replayUpdates?replayAbsTd/replayUpdates:0,nStepBufferSize:this.nStepBuffer.length,nStep:this.nStep}:{primaryUpdates:0,replayUpdates:0,replayMeanAbsTd:0,replaySize:this.replay.length,nStepBufferSize:this.nStepBuffer.length,nStep:this.nStep,pending:true};
  }
  flushLearning(){return this.q.updateTransition?this.drainNStep({flush:true}):null}
  learn(transition){
    if(!this.q.updateTransition)return this.q.update(transition.features,transition.actionIndex,transition.reward,transition.nextFeatures,transition.done);
    this.nStepBuffer.push(transition);
    return this.drainNStep({flush:!!transition.done});
  }
  distill(observation,teacherScores,options){return this.q.distill?.(observation,teacherScores,options)||null}
}
