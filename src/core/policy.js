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
function confidenceStats(probs){
  const sorted=[...probs].sort((a,b)=>b-a);
  return{entropy:entropyNormalized(probs),margin:(sorted[0]??0)-(sorted[1]??0)};
}

export class SemanticResidualPolicy{
  constructor({
    schema,actions,semantic,residual="linear",residualWeight=.9,temperature=.8,epsilon=.08,seed=2026,
    inferenceMode="hybrid",teacherInterval=32,teacherMinGap=8,teacherEntropy=.78,teacherMargin=.10,teacherNovelty=.85,distillSteps=4
  }){
    this.schema=schema;this.actions=actions;this.residualWeight=residualWeight;this.temperature=temperature;this.epsilon=epsilon;this.seed=seed;this.rng=mulberry32(seed);
    this.baseSize=1+schema.fields.length;this.memory=new TemporalMemory(this.baseSize);this.featureSize=this.baseSize*2;
    if(typeof residual==="object")this.q=residual;
    else if(residual==="neural-set")this.q=new NeuralSetResidualQ(schema,actions,{seed});
    else this.q=new LinearResidualQ(actions.length,this.featureSize);
    this.inferenceMode=inferenceMode;this.teacherInterval=teacherInterval;this.teacherMinGap=teacherMinGap;this.teacherEntropy=teacherEntropy;this.teacherMargin=teacherMargin;this.teacherNovelty=teacherNovelty;this.distillSteps=distillSteps;
    this.novelty=new NoveltyTracker(this.baseSize);this.decisionCount=0;this.lastTeacherStep=-1e9;this.teacherCalls=0;this.semanticCalls=0;
    this.setSemantic(semantic);
  }
  setSemantic(semantic){this.semantic=semantic;semantic.compile(this.schema,this.actions);this.lastTeacherStep=-1e9}
  setInferenceMode(mode){if(!["hybrid","adaptive","neural"].includes(mode))throw new Error("Unknown inference mode: "+mode);this.inferenceMode=mode;this.lastTeacherStep=-1e9}
  resetEpisode(){this.memory.reset()}
  resetLearning(){
    this.q.reset();this.novelty.reset();this.rng=mulberry32(this.seed);this.decisionCount=0;this.lastTeacherStep=-1e9;this.teacherCalls=0;this.semanticCalls=0;
  }
  encode(obs,memoryEnabled=true,commit=true){
    const base=numericFeatures(this.schema,obs),temporal=commit?this.memory.update(base,memoryEnabled):this.memory.preview(base,memoryEnabled);
    const x=new Float32Array(this.featureSize);x.set(base,0);x.set(temporal,this.baseSize);return{base,features:x};
  }
  residualScores(obs,features){return this.q.scoresObservation?this.q.scoresObservation(obs):this.q.scores(features)}
  async semanticScores(obs){
    const t=performance.now(),scores=await this.semantic.score(obs);this.semanticCalls++;return{scores,ms:performance.now()-t};
  }
  shouldTeacher(provisional,novelty){
    const gap=this.decisionCount-this.lastTeacherStep;
    if(this.decisionCount===0)return true;
    if(gap>=this.teacherInterval)return true;
    return gap>=this.teacherMinGap&&(provisional.entropy>=this.teacherEntropy||provisional.margin<=this.teacherMargin||novelty>=this.teacherNovelty);
  }
  async decide(obs,{useResidual=true,memory=true,explore=false}={}){
    const t0=performance.now(),encoded=this.encode(obs,memory,true),novelty=this.novelty.observe(encoded.base);
    const residualStart=performance.now();let q=this.residualScores(obs,encoded.features),residualMs=performance.now()-residualStart;
    let sem=new Array(this.actions.length).fill(0),semanticMs=0,teacherUsed=false,semanticUsed=false,logits;
    const mode=useResidual?this.inferenceMode:"hybrid";

    if(mode==="hybrid"){
      const result=await this.semanticScores(obs);sem=result.scores;semanticMs=result.ms;semanticUsed=true;
      logits=sem.map((v,i)=>v+(useResidual?this.residualWeight*q[i]:0));
    }else if(mode==="adaptive"){
      const provisionalProbs=softmax(q,this.temperature),provisional=confidenceStats(provisionalProbs);
      if(this.shouldTeacher(provisional,novelty)){
        const result=await this.semanticScores(obs);sem=result.scores;semanticMs=result.ms;semanticUsed=true;teacherUsed=true;this.teacherCalls++;this.lastTeacherStep=this.decisionCount;
        if(this.q.distill){
          for(let i=0;i<this.distillSteps;i++)this.q.distill(obs,sem,{strength:.5});
          const refresh=performance.now();q=this.residualScores(obs,encoded.features);residualMs+=performance.now()-refresh;
        }
      }
      logits=q;
    }else{
      logits=q;
    }

    const probs=softmax(logits,this.temperature);let chosen=argmax(probs);
    if(explore&&this.rng()<this.epsilon)chosen=Math.floor(this.rng()*this.actions.length);
    const stats=confidenceStats(probs);this.decisionCount++;
    return{
      actionIndex:chosen,action:this.actions[chosen],probs,semanticScores:sem,qScores:q,features:encoded.features,
      uncertainty:{...stats,novelty},latencyMs:performance.now()-t0,semanticLatencyMs:semanticMs,residualLatencyMs:residualMs,
      teacherUsed,semanticUsed,inferenceMode:mode,teacherCalls:this.teacherCalls
    };
  }
  learn(transition){
    if(this.q.updateTransition)return this.q.updateTransition(transition);
    return this.q.update(transition.features,transition.actionIndex,transition.reward,transition.nextFeatures,transition.done);
  }
  distill(observation,teacherScores,options){return this.q.distill?.(observation,teacherScores,options)||null}
}
