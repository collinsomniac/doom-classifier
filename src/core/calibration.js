import {argmax,clamp,softmax} from "./math.js";

function normalized(probs){
  const xs=Array.from(probs||[],x=>Math.max(0,Number(x)||0)),sum=xs.reduce((a,b)=>a+b,0);
  if(!xs.length)return[];
  if(sum<=0)return xs.map(()=>1/xs.length);
  return xs.map(x=>x/sum);
}

export function calibrationMetrics(samples,{bins=10}={}){
  const valid=(samples||[]).filter(s=>Array.isArray(s?.probs)&&s.probs.length>1&&Number.isInteger(s.label)&&s.label>=0&&s.label<s.probs.length);
  if(!valid.length)return{count:0,accuracy:0,meanConfidence:0,ece:0,mce:0,brier:0,nll:0,bins:[]};
  const nBins=Math.max(2,Math.floor(bins)),bucket=Array.from({length:nBins},()=>({count:0,confidence:0,correct:0}));
  let correct=0,confidence=0,brier=0,nll=0;
  for(const sample of valid){
    const p=normalized(sample.probs),top=argmax(p),conf=clamp(Number(p[top]||0),0,1),hit=top===sample.label?1:0;
    correct+=hit;confidence+=conf;nll-=Math.log(Math.max(1e-12,p[sample.label]||0));
    for(let i=0;i<p.length;i++){const y=i===sample.label?1:0;brier+=(p[i]-y)**2}
    const bi=Math.min(nBins-1,Math.floor(conf*nBins)),b=bucket[bi];b.count++;b.confidence+=conf;b.correct+=hit;
  }
  let ece=0,mce=0;
  const reliability=bucket.map((b,index)=>{
    const meanConfidence=b.count?b.confidence/b.count:0,accuracy=b.count?b.correct/b.count:0,gap=Math.abs(accuracy-meanConfidence);
    if(b.count){ece+=gap*b.count/valid.length;mce=Math.max(mce,gap)}
    return{index,low:index/nBins,high:(index+1)/nBins,count:b.count,meanConfidence,accuracy,gap};
  });
  return{count:valid.length,accuracy:correct/valid.length,meanConfidence:confidence/valid.length,ece,mce,brier:brier/valid.length,nll:nll/valid.length,bins:reliability};
}

export function logitsToProbabilities(logits,temperature=1){
  return softmax(Array.from(logits||[],Number),Math.max(.02,Number(temperature)||1));
}

export function evaluateLogitCalibration(samples,temperature=1,options={}){
  return calibrationMetrics((samples||[]).map(s=>({probs:logitsToProbabilities(s.logits,temperature),label:s.label})),options);
}

export function fitTemperature(samples,{min=.15,max=8,steps=160,bins=10}={}){
  const valid=(samples||[]).filter(s=>Array.isArray(s?.logits)&&s.logits.length>1&&Number.isInteger(s.label)&&s.label>=0&&s.label<s.logits.length);
  if(!valid.length)return{temperature:1,before:evaluateLogitCalibration([],1,{bins}),after:evaluateLogitCalibration([],1,{bins})};
  const lo=Math.log(Math.max(.02,min)),hi=Math.log(Math.max(min+.001,max)),count=Math.max(16,Math.floor(steps));
  let bestT=1,bestNll=Infinity;
  for(let i=0;i<count;i++){
    const t=Math.exp(lo+(hi-lo)*(i/(count-1))),metrics=evaluateLogitCalibration(valid,t,{bins});
    if(metrics.nll<bestNll){bestNll=metrics.nll;bestT=t}
  }
  // Local refinement around the best grid point.
  let span=(hi-lo)/Math.max(1,count-1),center=Math.log(bestT);
  for(let round=0;round<3;round++){
    let localBest=bestT,localLoss=bestNll;
    for(let j=-8;j<=8;j++){
      const t=Math.exp(center+j*span/8),metrics=evaluateLogitCalibration(valid,t,{bins});
      if(metrics.nll<localLoss){localLoss=metrics.nll;localBest=t}
    }
    bestT=localBest;bestNll=localLoss;center=Math.log(bestT);span/=4;
  }
  return{temperature:bestT,before:evaluateLogitCalibration(valid,1,{bins}),after:evaluateLogitCalibration(valid,bestT,{bins})};
}

export class TemperatureCalibrator{
  constructor(temperature=1){this.temperature=Math.max(.02,Number(temperature)||1);this.fitted=false;this.fitMetrics=null}
  fit(samples,options={}){const result=fitTemperature(samples,options);this.temperature=result.temperature;this.fitted=result.after.count>0;this.fitMetrics=result;return result}
  probabilities(logits){return logitsToProbabilities(logits,this.temperature)}
  metrics(samples,options={}){return calibrationMetrics((samples||[]).map(s=>({probs:this.probabilities(s.logits),label:s.label})),options)}
  export(){return{type:"temperature",version:1,temperature:this.temperature,fitted:this.fitted}}
  import(data){if(data?.type!=="temperature"||data.version!==1)throw new Error("Unsupported calibrator checkpoint");this.temperature=Math.max(.02,Number(data.temperature)||1);this.fitted=!!data.fitted;return this}
}
