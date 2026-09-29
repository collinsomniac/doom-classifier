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
    this.weights=this.axes.map(axis=>zeros(axis.values.length,this.inputDim));this.updates=0;return this;
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
  evaluate(observation){
    if(!this.axes.length||!this.actions.length)return{scores:new Array(this.actions.length).fill(0),axis:[],updates:this.updates,trust:0};
    const logits=this.axisLogits(observation),scores=this.actions.map(action=>{
      let sum=0,used=0;
      for(let ai=0;ai<this.axes.length;ai++){
        const axis=this.axes[ai],vi=axis.index.get(String(action?.params?.[axis.id]??"__unset"));
        if(vi==null)continue;sum+=logits[ai][vi];used++;
      }
      return used?sum/Math.sqrt(used):0;
    });
    return{scores:centered(scores),axis:this.axes.map((axis,i)=>({id:axis.id,values:axis.values,logits:Array.from(logits[i]),probs:softmax(Array.from(logits[i]),1)})),updates:this.updates,trust:clamp(this.updates/96,0,1)};
  }
  supervise(observation,targetDistribution,{strength=1}={}){
    const raw=Array.from(targetDistribution||[],v=>Math.max(0,Number(v)||0)),sum=raw.reduce((a,b)=>a+b,0);
    if(raw.length!==this.actions.length||sum<=0||!this.axes.length)return null;
    const target=raw.map(v=>v/sum),x=this.encode(observation),logits=this.axisLogits(observation);
    let loss=0,brier=0;
    for(let ai=0;ai<this.axes.length;ai++){
      const axis=this.axes[ai],marginal=new Array(axis.values.length).fill(0);
      for(let a=0;a<this.actions.length;a++){
        const vi=axis.index.get(String(this.actions[a]?.params?.[axis.id]??"__unset"));if(vi!=null)marginal[vi]+=target[a];
      }
      const z=marginal.reduce((a,b)=>a+b,0)||1;for(let i=0;i<marginal.length;i++)marginal[i]/=z;
      const pred=softmax(Array.from(logits[ai]),1);
      for(let vi=0;vi<axis.values.length;vi++){
        loss-=marginal[vi]*Math.log(Math.max(1e-9,pred[vi]));brier+=(pred[vi]-marginal[vi])**2;
        const grad=(pred[vi]-marginal[vi])*Number(strength||1);
        const row=this.weights[ai][vi];
        for(let j=0;j<this.inputDim;j++)row[j]-=this.lr*(grad*x[j]+this.l2*row[j]);
      }
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
