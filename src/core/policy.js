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
function confidenceStats(probs){
  const sorted=[...probs].sort((a,b)=>b-a);
  return{entropy:entropyNormalized(probs),margin:(sorted[0]??0)-(sorted[1]??0)};
}

export class SemanticResidualPolicy{
  constructor({
    schema,actions,semantic,residual="linear",residualWeight=.9,temperature=.8,epsilon=.08,seed=2026,
    inferenceMode="hybrid",teacherInterval=32,teacherMinGap=8,teacherEntropy=.78,teacherMargin=.10,teacherNovelty=.85,teacherEpistemic=.025,distillSteps=4
  }){
    this.schema=schema;this.actions=actions;this.residualWeight=residualWeight;this.temperature=temperature;this.epsilon=epsilon;this.seed=seed;this.rng=mulberry32(seed);
    this.baseSize=1+schema.fields.length;this.memory=new TemporalMemory(this.baseSize);this.featureSize=this.baseSize*2;
    if(typeof residual==="object")this.q=residual;
    else if(residual==="neural-set")this.q=new NeuralSetResidualQ(schema,actions,{seed});
    else this.q=new LinearResidualQ(actions.length,this.featureSize);
    this.inferenceMode=inferenceMode;this.teacherInterval=teacherInterval;this.teacherMinGap=teacherMinGap;this.teacherEntropy=teacherEntropy;this.teacherMargin=teacherMargin;this.teacherNovelty=teacherNovelty;this.teacherEpistemic=teacherEpistemic;this.distillSteps=distillSteps;
    this.novelty=new NoveltyTracker(this.baseSize);
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
    this.baseSize=1+schema.fields.length;this.memory=new TemporalMemory(this.baseSize);this.featureSize=this.baseSize*2;this.novelty=new NoveltyTracker(this.baseSize);
    if(this.q.setSchema&&this.q.setActions){this.q.setSchema(schema);this.q.setActions(actions)}
    else if(this.q instanceof LinearResidualQ){this.q=new LinearResidualQ(actions.length,this.featureSize)}
    this.semantic.compile(schema,actions);this.lastTeacherStep=-1e9;this.lastTeacherError=null;
    return this;
  }
  resetEpisode(){this.memory.reset()}
  resetLearning(){
    this.teacherGeneration++;this.teacherPromise=null;this.q.reset();this.novelty.reset();this.rng=mulberry32(this.seed);
    this.decisionCount=0;this.lastTeacherStep=-1e9;this.teacherCalls=0;this.semanticCalls=0;this.teacherScheduled=0;this.lastTeacherLatencyMs=0;this.lastTeacherError=null;
  }
  encode(obs,memoryEnabled=true,commit=true){
    const base=numericFeatures(this.schema,obs),temporal=commit?this.memory.update(base,memoryEnabled):this.memory.preview(base,memoryEnabled);
    const x=new Float32Array(this.featureSize);x.set(base,0);x.set(temporal,this.baseSize);return{base,temporal,features:x};
  }
  residualEvaluation(obs,features,temporal=null){
    if(this.q.scoreStatsObservation)return this.q.scoreStatsObservation(obs,{temporal});
    const scores=this.q.scoresObservation?this.q.scoresObservation(obs,{temporal}):this.q.scores(features);
    return{scores,memberScores:null};
  }
  residualScores(obs,features,temporal=null){return this.residualEvaluation(obs,features,temporal).scores}
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
  applyTeacherScores(obs,scores,steps=this.distillSteps,temporal=null){
    if(!this.q.distill)return null;
    let result=null;
    for(let i=0;i<steps;i++)result=this.q.distill(obs,scores,{strength:.5,temporal});
    return result;
  }
  async primeTeacher(obs,{steps=Math.max(4,this.distillSteps),temporal=null}={}){
    const semantic=this.semantic,generation=this.teacherGeneration,requestedStep=this.decisionCount;
    const result=await this.semanticScores(obs,semantic);
    if(generation!==this.teacherGeneration||semantic!==this.semantic)return{stale:true,ms:result.ms};
    this.applyTeacherScores(obs,result.scores,steps,temporal);this.teacherCalls++;this.lastTeacherStep=requestedStep;this.lastTeacherLatencyMs=result.ms;this.lastTeacherError=null;
    return{stale:false,ms:result.ms,scores:result.scores};
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
        this.applyTeacherScores(obs,result.scores,this.distillSteps,temporal);
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
    const t0=performance.now(),encoded=this.encode(obs,memory,true),novelty=this.novelty.observe(encoded.base);
    const residualStart=performance.now(),residualEval=this.residualEvaluation(obs,encoded.features,encoded.temporal);let q=residualEval.scores,residualMs=performance.now()-residualStart;
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

    const probs=softmax(logits,this.temperature);let chosen=argmax(probs);
    if(explore&&this.rng()<this.epsilon)chosen=Math.floor(this.rng()*this.actions.length);
    const stats=confidenceStats(probs);this.decisionCount++;
    return{
      actionIndex:chosen,action:this.actions[chosen],probs,semanticScores:sem,qScores:q,features:encoded.features,temporal:encoded.temporal,
      uncertainty:{...stats,novelty,epistemic},latencyMs:performance.now()-t0,semanticLatencyMs:semanticMs,residualLatencyMs:residualMs,
      teacherUsed,teacherPending:!!this.teacherPromise,semanticUsed,inferenceMode:mode,teacherCalls:this.teacherCalls,teacherScheduled:this.teacherScheduled,lastTeacherLatencyMs:this.lastTeacherLatencyMs
    };
  }
  learn(transition){
    if(this.q.updateTransition)return this.q.updateTransition(transition);
    return this.q.update(transition.features,transition.actionIndex,transition.reward,transition.nextFeatures,transition.done);
  }
  distill(observation,teacherScores,options){return this.q.distill?.(observation,teacherScores,options)||null}
}
