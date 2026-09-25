import {argmax,clamp,entropyNormalized,mulberry32,softmax} from "./math.js";
import {TemporalMemory} from "./memory.js";
import {LinearResidualQ} from "./residual.js";
import {NeuralSetResidualQ} from "./neural-set-residual.js";

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
    inferenceMode="hybrid",teacherInterval=32,teacherMinGap=8,teacherEntropy=.78,teacherMargin=.10,teacherNovelty=.85,teacherEpistemic=.025,distillSteps=4,replayCapacity=96,replayBatch=2
  }){
    this.schema=schema;this.actions=actions;this.residualWeight=residualWeight;this.baseTemperature=temperature;this.temperature=temperature;this.epsilon=epsilon;this.seed=seed;this.rng=mulberry32(seed);
    this.baseSize=1+schema.fields.length;this.memory=new TemporalMemory(this.baseSize);this.featureSize=this.baseSize*2;
    if(typeof residual==="object")this.q=residual;
    else if(residual==="neural-set")this.q=new NeuralSetResidualQ(schema,actions,{seed});
    else this.q=new LinearResidualQ(actions.length,this.featureSize);
    this.inferenceMode=inferenceMode;this.teacherInterval=teacherInterval;this.teacherMinGap=teacherMinGap;this.teacherEntropy=teacherEntropy;this.teacherMargin=teacherMargin;this.teacherNovelty=teacherNovelty;this.teacherEpistemic=teacherEpistemic;this.distillSteps=distillSteps;this.replayCapacity=replayCapacity;this.replayBatch=replayBatch;this.replay=[];this.replayRng=mulberry32((seed^0x517cc1b7)>>>0);
    this.novelty=new NoveltyTracker(this.baseSize);this.actionHistory={previousActionIndex:null,streak:0};
    this.decisionCount=0;this.lastTeacherStep=-1e9;this.teacherCalls=0;this.semanticCalls=0;this.teacherGeneration=0;this.teacherPromise=null;this.teacherScheduled=0;this.lastTeacherLatencyMs=0;this.lastTeacherError=null;
    this.setSemantic(semantic);
  }
  setSemantic(semantic){
    this.teacherGeneration++;this.teacherPromise=null;this.semantic=semantic;semantic.compile(this.schema,this.actions);this.lastTeacherStep=-1e9;this.lastTeacherError=null;
  }
  setInferenceMode(mode){
    if(!["hybrid","adaptive","neural"].includes(mode))throw new Error("Unknown inference mode: "+mode);
    this.inferenceMode=mode;
  }
  reconfigure({schema=this.schema,actions=this.actions}={}){
    this.teacherGeneration++;this.teacherPromise=null;
    this.schema=schema;this.actions=actions;
    this.baseSize=1+schema.fields.length;this.memory=new TemporalMemory(this.baseSize);this.featureSize=this.baseSize*2;this.novelty=new NoveltyTracker(this.baseSize);this.actionHistory={previousActionIndex:null,streak:0};
    if(this.q.setSchema&&this.q.setActions){this.q.setSchema(schema);this.q.setActions(actions)}
    else if(this.q instanceof LinearResidualQ){this.q=new LinearResidualQ(actions.length,this.featureSize)}
    this.semantic.compile(schema,actions);this.lastTeacherStep=-1e9;this.lastTeacherError=null;this.replay=[];
    return this;
  }
  resetEpisode(){this.memory.reset();this.actionHistory={previousActionIndex:null,streak:0}}
  currentActionHistory(){return{previousActionIndex:Number.isInteger(this.actionHistory?.previousActionIndex)?this.actionHistory.previousActionIndex:null,streak:Math.max(0,Number(this.actionHistory?.streak||0))}}
  nextActionHistory(actionIndex){
    const previous=this.currentActionHistory(),same=previous.previousActionIndex===actionIndex;
    return{previousActionIndex:actionIndex,streak:same?Math.min(255,previous.streak+1):1};
  }
  commitAction(actionIndex){this.actionHistory=this.nextActionHistory(actionIndex);return this.currentActionHistory()}
  resetLearning(){
    this.teacherGeneration++;this.teacherPromise=null;this.q.reset();this.novelty.reset();this.actionHistory={previousActionIndex:null,streak:0};this.rng=mulberry32(this.seed);this.replayRng=mulberry32((this.seed^0x517cc1b7)>>>0);this.replay=[];this.temperature=this.baseTemperature;
    this.decisionCount=0;this.lastTeacherStep=-1e9;this.teacherCalls=0;this.semanticCalls=0;this.teacherScheduled=0;this.lastTeacherLatencyMs=0;this.lastTeacherError=null;
  }
  encode(obs,memoryEnabled=true,commit=true){
    const base=numericFeatures(this.schema,obs),temporal=commit?this.memory.update(base,memoryEnabled):this.memory.preview(base,memoryEnabled);
    const x=new Float32Array(this.featureSize);x.set(base,0);x.set(temporal,this.baseSize);return{base,temporal,features:x};
  }
  residualEvaluation(obs,features,temporal=null,history=this.actionHistory){
    if(this.q.scoreStatsObservation)return this.q.scoreStatsObservation(obs,{temporal,history});
    const scores=this.q.scoresObservation?this.q.scoresObservation(obs,{temporal,history}):this.q.scores(features);
    return{scores,memberScores:null};
  }
  residualScores(obs,features,temporal=null,history=this.actionHistory){return this.residualEvaluation(obs,features,temporal,history).scores}
  async semanticScores(obs,semantic=this.semantic){
    const t=performance.now(),scores=await semantic.score(obs);this.semanticCalls++;return{scores,ms:performance.now()-t};
  }
  shouldTeacher(provisional,novelty){
    if(this.teacherPromise)return false;
    const gap=this.decisionCount-this.lastTeacherStep;
    if(this.teacherCalls===0)return true;
    if(gap>=this.teacherInterval)return true;
    return gap>=this.teacherMinGap&&(provisional.entropy>=this.teacherEntropy||provisional.margin<=this.teacherMargin||novelty>=this.teacherNovelty||provisional.epistemic>=this.teacherEpistemic);
  }
  calibrateTemperature(obs,teacherScores,temporal=null,{blend=.35}={}){
    if(!teacherScores?.length)return null;
    const encoded=this.encode(obs,true,false),logits=this.residualScores(obs,encoded.features,temporal),target=softmax(teacherScores,1);
    let bestT=this.temperature,bestLoss=Infinity;
    const candidates=[.30,.40,.50,.65,.80,1.0,1.25,1.5];
    for(const t of candidates){const loss=crossEntropy(target,softmax(logits,t));if(loss<bestLoss){bestLoss=loss;bestT=t}}
    this.temperature=clamp(this.temperature*(1-blend)+bestT*blend,.30,1.5);
    return{temperature:this.temperature,bestTemperature:bestT,loss:bestLoss};
  }
  applyTeacherScores(obs,scores,steps=this.distillSteps,temporal=null,{maxSteps=steps,targetKL=null}={}){
    if(!this.q.distill)return null;
    const teacher=softmax(scores,1);let result=null,used=0,kl=Infinity;
    const cap=Math.max(steps,Math.floor(maxSteps||steps));
    for(let i=0;i<cap;i++){
      result=this.q.distill(obs,scores,{strength:.5,temporal});used=i+1;
      if(result?.student){kl=klDivergence(teacher,result.student);if(used>=steps&&targetKL!=null&&kl<=targetKL)break}
      if(used>=steps&&targetKL==null)break;
    }
    this.q.syncTarget?.({value:false});
    return result?{...result,stepsUsed:used,kl}:null;
  }
  async primeTeacher(obs,{steps=Math.max(4,this.distillSteps),maxSteps=steps,targetKL=null,temporal=null}={}){
    const semantic=this.semantic,generation=this.teacherGeneration,requestedStep=this.decisionCount;
    const result=await this.semanticScores(obs,semantic);
    if(generation!==this.teacherGeneration||semantic!==this.semantic)return{stale:true,ms:result.ms};
    const distillation=this.applyTeacherScores(obs,result.scores,steps,temporal,{maxSteps,targetKL});const calibration=this.calibrateTemperature(obs,result.scores,temporal,{blend:.65});
    this.teacherCalls++;this.lastTeacherStep=requestedStep;this.lastTeacherLatencyMs=result.ms;this.lastTeacherError=null;
    return{stale:false,ms:result.ms,scores:result.scores,calibration,distillation};
  }
  scheduleTeacher(obs,temporal=null){
    if(this.teacherPromise)return false;
    const semantic=this.semantic,generation=this.teacherGeneration,requestedStep=this.decisionCount;
    this.teacherScheduled++;
    let task;
    task=(async()=>{
      try{
        const result=await this.semanticScores(obs,semantic);
        if(generation!==this.teacherGeneration||semantic!==this.semantic)return{stale:true,ms:result.ms};
        this.applyTeacherScores(obs,result.scores,this.distillSteps,temporal);this.calibrateTemperature(obs,result.scores,temporal,{blend:.18});
        this.teacherCalls++;this.lastTeacherStep=requestedStep;this.lastTeacherLatencyMs=result.ms;this.lastTeacherError=null;
        return{stale:false,ms:result.ms,scores:result.scores};
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
    const t0=performance.now(),encoded=this.encode(obs,memory,true),novelty=this.novelty.observe(encoded.base),actionHistory=this.currentActionHistory();
    const residualStart=performance.now(),residualEval=this.residualEvaluation(obs,encoded.features,encoded.temporal,actionHistory);let q=residualEval.scores,residualMs=performance.now()-residualStart;
    const epistemic=ensembleDisagreement(residualEval.memberScores,this.temperature);
    let sem=new Array(this.actions.length).fill(0),semanticMs=0,teacherUsed=false,semanticUsed=false,logits;
    const mode=useResidual?this.inferenceMode:"hybrid";

    if(mode==="hybrid"){
      const result=await this.semanticScores(obs);sem=result.scores;semanticMs=result.ms;semanticUsed=true;
      logits=sem.map((v,i)=>v+(useResidual?this.residualWeight*q[i]:0));
    }else if(mode==="adaptive"){
      const provisional={...confidenceStats(softmax(q,this.temperature)),epistemic};
      teacherUsed=this.shouldTeacher(provisional,novelty)?this.scheduleTeacher(obs,encoded.temporal):false;
      logits=q;
    }else{
      logits=q;
    }

    const decodeTemperature=mode==="hybrid"&&!useResidual?1:this.temperature;
    const probs=softmax(logits,decodeTemperature);let chosen=argmax(probs);
    if(explore&&this.rng()<this.epsilon)chosen=Math.floor(this.rng()*this.actions.length);
    const stats=confidenceStats(probs);this.decisionCount++;
    return{
      actionIndex:chosen,action:this.actions[chosen],probs,semanticScores:sem,qScores:q,semanticPriorScores:residualEval.semanticScores||null,valueScores:residualEval.valueScores||null,features:encoded.features,temporal:encoded.temporal,actionHistory,
      uncertainty:{...stats,novelty,epistemic},latencyMs:performance.now()-t0,semanticLatencyMs:semanticMs,residualLatencyMs:residualMs,
      teacherUsed,teacherPending:!!this.teacherPromise,semanticUsed,inferenceMode:mode,teacherCalls:this.teacherCalls,teacherScheduled:this.teacherScheduled,lastTeacherLatencyMs:this.lastTeacherLatencyMs
    };
  }
  learn(transition){
    if(!this.q.updateTransition)return this.q.update(transition.features,transition.actionIndex,transition.reward,transition.nextFeatures,transition.done);
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
    return{...primary,replayUpdates,replayMeanAbsTd:replayUpdates?replayAbsTd/replayUpdates:0,replaySize:this.replay.length};
  }
  distill(observation,teacherScores,options){return this.q.distill?.(observation,teacherScores,options)||null}
}
