import {dot,clamp} from "./math.js";

export class LinearResidualQ{
  constructor(actionCount,featureSize,{lr=.035,gamma=.94,l2=1e-5}={}){
    this.actionCount=actionCount;this.featureSize=featureSize;this.lr=lr;this.gamma=gamma;this.l2=l2;
    this.weights=Array.from({length:actionCount},()=>new Float32Array(featureSize));
    this.updates=0;
  }
  reset(){for(const w of this.weights)w.fill(0);this.updates=0}
  scores(features){return this.weights.map(w=>dot(w,features))}
  update(features,actionIndex,reward,nextFeatures,done=false){
    const q=this.scores(features)[actionIndex];
    const next=this.scores(nextFeatures);
    const target=reward+(done?0:this.gamma*Math.max(...next));
    const td=clamp(target-q,-3,3);
    const w=this.weights[actionIndex];
    for(let i=0;i<w.length;i++)w[i]+=this.lr*(td*features[i]-this.l2*w[i]);
    this.updates++;
    return{td,target,q};
  }
}
