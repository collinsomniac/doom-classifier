import {argmax,clamp,entropyNormalized,mulberry32,softmax} from "./math.js";
import {TemporalMemory} from "./memory.js";
import {LinearResidualQ} from "./residual.js";

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
    const score=Math.tanh(Math.sqrt(z/x.length)/2);
    this.n++;
    for(let i=0;i<x.length;i++){const d=x[i]-this.mean[i];this.mean[i]+=d/this.n;this.m2[i]+=d*(x[i]-this.mean[i])}
    return score;
  }
}

export class SemanticResidualPolicy{
  constructor({schema,actions,semantic,residualWeight=.9,temperature=.8,epsilon=.08,seed=2026}){
    this.schema=schema;this.actions=actions;this.residualWeight=residualWeight;this.temperature=temperature;this.epsilon=epsilon;this.seed=seed;this.rng=mulberry32(seed);
    this.baseSize=1+schema.fields.length;this.memory=new TemporalMemory(this.baseSize);this.featureSize=this.baseSize*2;
    this.q=new LinearResidualQ(actions.length,this.featureSize);this.novelty=new NoveltyTracker(this.baseSize);
    this.setSemantic(semantic);
  }
  setSemantic(semantic){this.semantic=semantic;semantic.compile(this.schema,this.actions)}
  resetEpisode(){this.memory.reset()}
  resetLearning(){this.q.reset();this.novelty.reset();this.rng=mulberry32(this.seed)}
  encode(obs,memoryEnabled=true,commit=true){
    const base=numericFeatures(this.schema,obs),temporal=commit?this.memory.update(base,memoryEnabled):this.memory.preview(base,memoryEnabled);
    const x=new Float32Array(this.featureSize);x.set(base,0);x.set(temporal,this.baseSize);
    return{base,features:x};
  }
  async decide(obs,{useResidual=true,memory=true,explore=false}={}){
    const t0=performance.now(),encoded=this.encode(obs,memory,true);
    const semanticStart=performance.now();
    const sem=await this.semantic.score(obs);
    const semanticMs=performance.now()-semanticStart;
    const q=this.q.scores(encoded.features);
    const logits=sem.map((v,i)=>v+(useResidual?this.residualWeight*q[i]:0)),probs=softmax(logits,this.temperature);
    let chosen=argmax(probs);
    if(explore&&this.rng()<this.epsilon)chosen=Math.floor(this.rng()*this.actions.length);
    const sorted=[...probs].sort((a,b)=>b-a);
    return{
      actionIndex:chosen,action:this.actions[chosen],probs,semanticScores:sem,qScores:q,features:encoded.features,
      uncertainty:{entropy:entropyNormalized(probs),margin:(sorted[0]??0)-(sorted[1]??0),novelty:this.novelty.observe(encoded.base)},
      latencyMs:performance.now()-t0,semanticLatencyMs:semanticMs
    };
  }
  learn(transition){return this.q.update(transition.features,transition.actionIndex,transition.reward,transition.nextFeatures,transition.done)}
}
