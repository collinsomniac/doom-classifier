import {clamp,softmax} from "./math.js";

function featureSpec(schema){
  const specs=[];
  for(const field of schema?.fields||[]){
    if(!field?.decisionFeature)continue;
    if(field.enum){
      for(const key of Object.keys(field.enum))specs.push({id:field.id+"="+key,read:obs=>String(obs?.[field.id]??"__unset")===String(key)?1:0});
      continue;
    }
    specs.push({id:field.id,read:obs=>{
      const raw=Number(obs?.[field.id]??0);if(!Number.isFinite(raw))return 0;
      const lo=Number(field.min),hi=Number(field.max),scale=Math.abs(Number(field.scale)||0);
      if(Number.isFinite(lo)&&Number.isFinite(hi)&&hi>lo){
        if(lo<0&&hi>0)return clamp(raw/Math.max(Math.abs(lo),Math.abs(hi)),-1,1);
        return clamp((raw-lo)/(hi-lo),0,1);
      }
      if(scale>0)return Math.tanh(raw/scale);
      return clamp(raw,-1,1);
    }});
  }
  return specs;
}
function axisSpecs(schema,actions){
  return (schema?.actionFields||[]).map(field=>{
    const values=[];const seen=new Set();
    for(const action of actions||[]){
      const key=String(action?.params?.[field.id]??"__unset");
      if(!seen.has(key)){seen.add(key);values.push(key)}
    }
    return{id:field.id,label:field.label||field.id,values,index:new Map(values.map((v,i)=>[v,i]))};
  }).filter(axis=>axis.values.length>1);
}
function zeros(rows,cols){return Array.from({length:rows},()=>new Float64Array(cols))}
function centered(values){const m=values.reduce((a,b)=>a+Number(b||0),0)/Math.max(1,values.length);return values.map(v=>Number(v||0)-m)}

export class FactorizedControlHead{
  constructor(schema,actions,{lr=.08,l2=1e-5}={}){
    this.lr=lr;this.l2=l2;this.configure(schema,actions);
  }
  configure(schema,actions){
    this.schema=schema;this.actions=actions||[];this.features=featureSpec(schema);this.axes=axisSpecs(schema,this.actions);this.inputDim=this.features.length+1;
    this.weights=this.axes.map(axis=>zeros(axis.values.length,this.inputDim));this._avi=null;this.updates=0;return this;
  }
  reset(){return this.configure(this.schema,this.actions)}
  encode(observation){
    const x=new Float64Array(this.inputDim);x[0]=1;
    for(let i=0;i<this.features.length;i++)x[i+1]=Number(this.features[i].read(observation)||0);
    return x;
  }
  axisLogits(observation){
    const x=this.encode(observation);
    return this.axes.map((axis,ai)=>this.weights[ai].map(row=>{
      let s=0;for(let j=0;j<this.inputDim;j++)s+=row[j]*x[j];return s;
    }));
  }
  // Index of each action's value on each axis (null when the axis is unset).
  actionValueIndex(){
    if(!this._avi)this._avi=this.actions.map(action=>this.axes.map(axis=>axis.index.get(String(action?.params?.[axis.id]??"__unset"))??null));
    return this._avi;
  }
  // Joint energy of every *valid* composite action: sum of its axis logits.
  // The distribution is normalized over the composites that actually exist
  // (log-linear product of experts), NOT a product of independent marginals.
  jointScores(logits){
    return this.actionValueIndex().map(values=>{
      let s=0;for(let ai=0;ai<values.length;ai++)if(values[ai]!=null)s+=logits[ai][values[ai]];return s;
    });
  }
  evaluate(observation){
    if(!this.axes.length||!this.actions.length)return{scores:new Array(this.actions.length).fill(0),axis:[],updates:this.updates,trust:0};
    const logits=this.axisLogits(observation),scores=this.jointScores(logits),joint=softmax(scores,1),avi=this.actionValueIndex();
    // Axis probabilities are marginals of the normalized joint, so diagnostics
    // describe the distribution the policy actually decodes.
    const axis=this.axes.map((a,ai)=>{
      const probs=new Array(a.values.length).fill(0);
      for(let k=0;k<joint.length;k++){const vi=avi[k][ai];if(vi!=null)probs[vi]+=joint[k]}
      return{id:a.id,values:a.values,logits:Array.from(logits[ai]),probs};
    });
    return{scores:centered(scores),joint,axis,updates:this.updates,trust:clamp(this.updates/96,0,1)};
  }
  // Proper joint cross-entropy over valid composites. Earlier versions fitted
  // each axis marginal independently; composing marginals then made the
  // all-neutral composite ("wait") the mode whenever the label mass was split
  // across e.g. {turn_left, strafe_left}. See tests/factor-grounding.mjs.
  supervise(observation,targetDistribution,{strength=1}={}){
    const raw=Array.from(targetDistribution||[],v=>Math.max(0,Number(v)||0)),sum=raw.reduce((a,b)=>a+b,0);
    if(raw.length!==this.actions.length||sum<=0||!this.axes.length)return null;
    const target=raw.map(v=>v/sum),x=this.encode(observation),logits=this.axisLogits(observation),pred=softmax(this.jointScores(logits),1),avi=this.actionValueIndex();
    let loss=0,brier=0;
    const grads=this.axes.map(a=>new Float64Array(a.values.length));
    for(let k=0;k<pred.length;k++){
      if(target[k]>0)loss-=target[k]*Math.log(Math.max(1e-12,pred[k]));brier+=(pred[k]-target[k])**2;
      const g=pred[k]-target[k];
      for(let ai=0;ai<this.axes.length;ai++){const vi=avi[k][ai];if(vi!=null)grads[ai][vi]+=g}
    }
    const s=Number(strength||1);
    for(let ai=0;ai<this.axes.length;ai++)for(let vi=0;vi<this.axes[ai].values.length;vi++){
      const grad=grads[ai][vi]*s,row=this.weights[ai][vi];
      for(let j=0;j<this.inputDim;j++)row[j]-=this.lr*(grad*x[j]+this.l2*row[j]);
    }
    this.updates++;return{loss,brier,updates:this.updates};
  }
  parameterCount(){return this.weights.reduce((n,axis)=>n+axis.reduce((m,row)=>m+row.length,0),0)}
  exportCheckpoint(){
    return{version:1,features:this.features.map(x=>x.id),axes:this.axes.map(x=>({id:x.id,values:x.values})),updates:this.updates,weights:this.weights.map(axis=>axis.map(row=>Array.from(row)))};
  }
  importCheckpoint(checkpoint){
    if(!checkpoint||checkpoint.version!==1)return false;
    if(checkpoint.axes?.length!==this.axes.length||checkpoint.features?.join("|")!==this.features.map(x=>x.id).join("|"))return false;
    for(let ai=0;ai<this.axes.length;ai++){
      if(checkpoint.axes[ai]?.id!==this.axes[ai].id||checkpoint.axes[ai]?.values?.join("|")!==this.axes[ai].values.join("|"))return false;
      for(let vi=0;vi<this.axes[ai].values.length;vi++){
        const source=checkpoint.weights?.[ai]?.[vi];if(!Array.isArray(source)||source.length!==this.inputDim)return false;
        this.weights[ai][vi].set(source);
      }
    }
    this.updates=Math.max(0,Number(checkpoint.updates||0));return true;
  }
}
