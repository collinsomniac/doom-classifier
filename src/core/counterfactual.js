import {clamp} from "./math.js";

function clone(value){
  if(value==null)return value;
  return globalThis.structuredClone?globalThis.structuredClone(value):JSON.parse(JSON.stringify(value));
}

function normalize(xs,floor=1e-8){
  const values=Array.from(xs||[],v=>Math.max(floor,Number(v)||0));
  const sum=values.reduce((a,b)=>a+b,0);
  if(!values.length)return[];
  if(sum<=0)return values.map(()=>1/values.length);
  return values.map(v=>v/sum);
}

function softmaxLogits(logits){
  if(!logits.length)return[];
  const peak=Math.max(...logits),weights=logits.map(v=>Math.exp(clamp(v-peak,-60,0))),sum=weights.reduce((a,b)=>a+b,0)||1;
  return weights.map(v=>v/sum);
}

export function counterfactualTarget(returns,prior,{temperature=.7,priorStrength=1,minScale=1e-4}={}){
  const values=Array.from(returns||[],Number);
  if(!values.length||values.some(v=>!Number.isFinite(v)))throw new Error("counterfactualTarget requires finite returns");
  const p=normalize(prior?.length===values.length?prior:new Array(values.length).fill(1));
  const max=Math.max(...values),min=Math.min(...values),mean=values.reduce((a,b)=>a+b,0)/values.length;
  const variance=values.reduce((s,v)=>s+(v-mean)**2,0)/values.length,std=Math.sqrt(variance);
  const scale=Math.max(Number(minScale)||1e-4,max-min,std*2);
  if(max-min<1e-12)return{target:p,scale,spread:0,advantages:values.map(()=>0)};
  const t=Math.max(.05,Number(temperature)||.7),strength=Math.max(0,Number(priorStrength)||0);
  const advantages=values.map(v=>(v-max)/scale);
  const logits=advantages.map((a,i)=>strength*Math.log(Math.max(1e-9,p[i]))+a/t);
  return{target:softmaxLogits(logits),scale,spread:max-min,advantages};
}

export async function probeCounterfactualActions({
  environment,
  actions=environment?.actions,
  prior=null,
  candidateIndices=null,
  horizon=1,
  discount=.96,
  continuation=null,
  temperature=.7,
  priorStrength=1
}={}){
  if(!environment||typeof environment.saveSnapshot!=="function"||typeof environment.restoreSnapshot!=="function"){
    throw new Error("Counterfactual probing requires a forkable environment adapter");
  }
  if(!Array.isArray(actions)||!actions.length)throw new Error("Counterfactual probing requires typed actions");
  const indices=candidateIndices?.length?[...new Set(candidateIndices.map(Number))]:actions.map((_,i)=>i);
  if(indices.some(i=>!Number.isInteger(i)||i<0||i>=actions.length))throw new Error("Invalid counterfactual candidate index");
  const baseObservation=clone(environment.lastObservation??environment.observe?.());
  const snapshot=environment.saveSnapshot();
  const returns=new Array(actions.length).fill(NaN),trials=[];
  const steps=Math.max(1,Math.floor(Number(horizon)||1)),gamma=clamp(Number(discount)||0,0,1);

  try{
    for(const actionIndex of indices){
      environment.restoreSnapshot(snapshot);
      let total=0,weight=1,done=false,lastStep=null;
      for(let depth=0;depth<steps&&!done;depth++){
        let actionId=actions[actionIndex].id;
        if(depth>0&&typeof continuation==="function"){
          const next=await continuation({
            depth,actionIndex,action:actions[actionIndex],environment,
            observation:clone(environment.lastObservation??environment.observe?.()),lastStep
          });
          if(typeof next==="number")actionId=actions[next]?.id;
          else if(typeof next==="string")actionId=next;
          else if(next?.id)actionId=next.id;
          if(!actionId)break;
        }
        const step=await environment.step(actionId);
        const reward=Number(step?.reward||0);
        total+=weight*reward;weight*=gamma;done=!!step?.done;lastStep=step;
      }
      returns[actionIndex]=total;
      trials.push({
        actionIndex,id:actions[actionIndex].id,label:actions[actionIndex].label||actions[actionIndex].id,
        return:total,done,steps:steps-(done?Math.max(0,steps-1):0),
        outcome:clone(lastStep?.info?.outcome||null)
      });
    }
  }finally{
    environment.restoreSnapshot(snapshot);
  }

  const measured=indices.map(i=>returns[i]);
  const measuredPrior=normalize(indices.map(i=>prior?.[i]??1));
  const improved=counterfactualTarget(measured,measuredPrior,{temperature,priorStrength});
  const target=new Array(actions.length).fill(0);
  indices.forEach((index,j)=>{target[index]=improved.target[j]});
  return{
    observation:baseObservation,target,returns,trials,candidateIndices:indices,
    measuredPrior,temperature,priorStrength,scale:improved.scale,spread:improved.spread,advantages:improved.advantages
  };
}
