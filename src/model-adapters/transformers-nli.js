const TRANSFORMERS_CDN="https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.0";

export const NLI_PRESETS=Object.freeze({
  mobilebert:{label:"MobileBERT-MNLI",modelId:"Xenova/mobilebert-uncased-mnli",gpuDtype:"fp16",wasmDtype:"int8",approx:"~50 MB WebGPU / ~26 MB WASM"},
  distilbert:{label:"DistilBERT-MNLI",modelId:"Xenova/distilbert-base-uncased-mnli",gpuDtype:"fp16",wasmDtype:"int8",approx:"~134 MB WebGPU / ~67 MB WASM"},
  deberta:{label:"DeBERTa-v3-xsmall NLI",modelId:"Xenova/nli-deberta-v3-xsmall",gpuDtype:"fp16",wasmDtype:"int8",approx:"~143 MB WebGPU / ~90 MB WASM"}
});
function normalizeProgress(info){
  if(!info||typeof info!=="object")return null;
  if(Number.isFinite(info.progress))return Math.max(0,Math.min(100,info.progress));
  if(Number.isFinite(info.loaded)&&Number.isFinite(info.total)&&info.total>0)return Math.max(0,Math.min(100,100*info.loaded/info.total));
  return null;
}
function fieldValueText(field,value){
  const category=field?.enum?.[String(value)];
  return category!==undefined?String(category)+" (code "+value+")":Number(value).toFixed(3);
}
function normalized(v,field={}){
  v=Number(v);if(!Number.isFinite(v))v=0;
  if(Number.isFinite(field.min)&&Number.isFinite(field.max)&&field.max!==field.min)return Math.max(-1,Math.min(1,((v-field.min)/(field.max-field.min))*2-1));
  const scale=Math.max(1e-6,Number(field.scale)||1),x=v/scale;return x/(1+Math.abs(x));
}
function fieldStats(field,records){
  const raw=[],norm=[];
  for(const record of records){
    if(!record||!Object.prototype.hasOwnProperty.call(record,field.id))continue;
    const v=Number(record[field.id]);if(!Number.isFinite(v))continue;
    raw.push(v);norm.push(normalized(v,field));
  }
  if(!raw.length)return null;
  const mean=raw.reduce((a,b)=>a+b,0)/raw.length,nmean=norm.reduce((s,v)=>s+v,0)/norm.length;
  const variance=norm.reduce((s,v)=>s+(v-nmean)*(v-nmean),0)/norm.length;
  const categorical=!!field.enum;
  const categoryCounts=categorical?raw.reduce((map,v)=>{
    const label=field.enum?.[String(v)]??("code "+v);map.set(label,(map.get(label)||0)+1);return map;
  },new Map()):null;
  const explicitBoolean=field.type==="boolean"||field.kind==="boolean";
  const binaryValues=!categorical&&field.min===0&&field.max===1&&raw.every(v=>Math.abs(v)<1e-9||Math.abs(v-1)<1e-9);
  const booleanish=explicitBoolean||binaryValues;
  const active=booleanish?raw.filter(v=>v>.5).length:0;
  return{field,mean,min:Math.min(...raw),max:Math.max(...raw),variance,categorical,categoryCounts,booleanish,active,count:raw.length};
}
function representativeRecords(records,stats,count=2){
  if(!records.length||!stats.length)return[];
  const vectors=records.map(record=>stats.map(s=>normalized(record?.[s.field.id],s.field)));
  const selected=[],selectedIndices=[];
  let first=0,best=-1;
  for(let i=0;i<vectors.length;i++){const norm=vectors[i].reduce((s,v)=>s+v*v,0);if(norm>best){best=norm;first=i}}
  selected.push(records[first]);selectedIndices.push(first);
  while(selected.length<Math.min(count,records.length)){
    let pick=-1,pickScore=-1;
    for(let i=0;i<vectors.length;i++){
      if(selectedIndices.includes(i))continue;
      let minDist=Infinity;
      for(const j of selectedIndices){let d=0;for(let k=0;k<stats.length;k++){const delta=vectors[i][k]-vectors[j][k];d+=delta*delta}minDist=Math.min(minDist,d)}
      if(minDist>pickScore){pickScore=minDist;pick=i}
    }
    if(pick<0)break;selected.push(records[pick]);selectedIndices.push(pick);
  }
  return selected;
}
export function summarizeCollection(collection,records,{maxFields=8,representatives=2,maxFlags=4}={}){
  const all=(collection.fields||[]).map(field=>fieldStats(field,records)).filter(Boolean);
  const categories=all.filter(s=>s.categorical).sort((a,b)=>b.variance-a.variance);
  const flags=all.filter(s=>!s.categorical&&s.booleanish&&s.active>0).sort((a,b)=>(b.active/b.count)-(a.active/a.count)||String(a.field.id).localeCompare(String(b.field.id))).slice(0,maxFlags);
  const chosen=new Set([...categories,...flags].map(s=>s.field.id));
  const numeric=all.filter(s=>!chosen.has(s.field.id)).sort((a,b)=>b.variance-a.variance);
  const stats=[...categories,...flags,...numeric].slice(0,maxFields);
  const label=collection.label||collection.id,description=collection.description?" ("+collection.description+")":"";
  const lines=[label+description+": "+records.length+" records."];
  if(stats.length)lines.push("stats: "+stats.map(s=>{
    const name=s.field.label||s.field.id;
    if(s.categorical){
      const top=[...s.categoryCounts.entries()].sort((a,b)=>b[1]-a[1]).slice(0,5);
      return name+" categories="+top.map(([label,count])=>label+":"+count).join(", ");
    }
    if(s.booleanish)return name+" active="+s.active+"/"+s.count+" ("+Math.round(100*s.active/s.count)+"%)";
    return name+" mean="+s.mean.toFixed(2)+" min="+s.min.toFixed(2)+" max="+s.max.toFixed(2);
  }).join("; "));
  const reps=representativeRecords(records,stats,representatives);
  reps.forEach((record,i)=>lines.push("representative "+(i+1)+": {"+stats.map(s=>{
    const value=record[s.field.id]??0;return(s.field.label||s.field.id)+":"+fieldValueText(s.field,value);
  }).join(", ")+"}"));
  return lines.join("\n");
}

export class TransformersNLIAdapter{
  constructor({preset="mobilebert",onProgress=()=>{},maxStateChars=5000,maxPremiseTokens=384,labelBiasCalibration=false,labelBiasStrength=1}={}){
    if(!NLI_PRESETS[preset])throw new Error("Unknown NLI preset: "+preset);
    this.presetKey=preset;this.preset=NLI_PRESETS[preset];this.name=this.preset.label;this.onProgress=onProgress;this.maxStateChars=maxStateChars;this.maxPremiseTokens=maxPremiseTokens;
    this.labelBiasCalibration=!!labelBiasCalibration;this.labelBiasStrength=Math.max(0,Number(labelBiasStrength)||0);
    this.schema=null;this.actions=null;this.labels=null;this.classifier=null;this.backend="unloaded";this.nullBiasLogits=null;this.nullBiasPromise=null;
  }
  compile(schema,actions){
    this.schema=schema;this.actions=actions;this.nullBiasLogits=null;this.nullBiasPromise=null;
    this.labels=actions.map(a=>{
      const active=Object.entries(a.params||{}).filter(([,v])=>Number(v)>0).map(([k])=>k.replaceAll("_"," "));
      return a.label+" — controller buttons: "+(active.length?active.join(", "):"none")+"; "+a.description;
    });
  }
  report(info){this.onProgress({...info,normalizedProgress:normalizeProgress(info)})}
  async load(){
    if(this.classifier)return this;
    this.report({status:"loading-library",file:"Transformers.js"});
    const mod=await import(TRANSFORMERS_CDN);mod.env.allowLocalModels=false;
    const canWebGPU=typeof navigator!=="undefined"&&!!navigator.gpu;
    const attempts=canWebGPU?[{device:"webgpu",dtype:this.preset.gpuDtype},{device:"wasm",dtype:this.preset.wasmDtype}]:[{device:"wasm",dtype:this.preset.wasmDtype}];
    let lastError=null;
    for(const attempt of attempts){
      try{
        this.report({status:"loading-model",file:this.preset.label+" "+attempt.device+" "+attempt.dtype});
        this.classifier=await mod.pipeline("zero-shot-classification",this.preset.modelId,{device:attempt.device,dtype:attempt.dtype,progress_callback:info=>this.report(info)});
        this.backend=attempt.device+" / "+attempt.dtype;this.report({status:"ready",file:this.backend,progress:100});return this;
      }catch(error){
        lastError=error;this.report({status:"backend-failed",file:attempt.device+" / "+attempt.dtype,error:String(error?.message||error)});
        if(this.classifier?.dispose)await this.classifier.dispose();this.classifier=null;
      }
    }
    throw lastError||new Error("Unable to load NLI model");
  }
  fullStateObject(observation){
    const scalars={};
    for(const field of this.schema?.fields||[]){
      const raw=observation?.[field.id]??0,category=field.enum?.[String(raw)];
      scalars[field.id]=category===undefined?raw:{value:raw,label:String(category)};
    }
    const collections={};
    for(const collection of this.schema?.collections||[])collections[collection.id]=observation?._collections?.[collection.id]||[];
    return{
      environment:this.schema?.environment||"structured interactive environment",
      objective:this.schema?.objective||"choose a controller input from the observed state",
      control_interval_ms:Number.isFinite(Number(this.schema?.controlHorizonMs))?Number(this.schema.controlHorizonMs):null,
      observation:{scalars,collections}
    };
  }
  stateObject(observation){
    const full=this.fullStateObject(observation),collections={};
    for(const collection of this.schema?.collections||[]){
      const records=full.observation.collections[collection.id]||[];
      collections[collection.id]={count:records.length,summary:summarizeCollection(collection,records)};
    }
    return{
      environment:full.environment,objective:full.objective,control_interval_ms:full.control_interval_ms,
      observation:{scalars:full.observation.scalars,collections},
      projection:{bounded:true,reason:"semantic teacher token budget; fast neural core receives the full structured observation"}
    };
  }
  stateText(observation){
    const text=JSON.stringify(this.stateObject(observation));
    return text.length<=this.maxStateChars?text:text.slice(0,Math.max(0,this.maxStateChars-32))+'..."truncated":true}';
  }
  boundedPremise(observation){
    const text=this.stateText(observation),tokenizer=this.classifier?.tokenizer;
    if(!tokenizer)return text;
    try{
      const encoded=tokenizer(text,{return_tensor:false,truncation:true,max_length:this.maxPremiseTokens});
      let ids=encoded?.input_ids??encoded;
      if(Array.isArray(ids)&&Array.isArray(ids[0]))ids=ids[0];
      if(Array.isArray(ids)&&typeof tokenizer.decode==="function"){
        return tokenizer.decode(ids,{skip_special_tokens:true});
      }
    }catch(error){
      console.warn("[semantic teacher] tokenizer pre-truncation failed; using character-bounded premise",error);
    }
    return text;
  }
  outputLogits(output){
    const independent=new Map((output?.labels||[]).map((label,i)=>[label,output.scores[i]])),eps=1e-6;
    return this.labels.map(label=>{
      const p=Math.max(eps,Math.min(1-eps,Number(independent.get(label)??eps)));
      return Math.log(p/(1-p));
    });
  }
  async classifyPremise(premise){
    if(!this.classifier)await this.load();
    const output=await this.classifier(premise,this.labels,{multi_label:true,hypothesis_template:"Given only the observed game state and literal controller meanings, using {} for the next control interval is contextually appropriate."});
    return this.outputLogits(output);
  }
  nullStateText(){
    return JSON.stringify({
      environment:this.schema?.environment||"structured interactive environment",
      objective:this.schema?.objective||"choose a controller input from the observed state",
      control_interval_ms:Number.isFinite(Number(this.schema?.controlHorizonMs))?Number(this.schema.controlHorizonMs):null,
      observation:{status:"withheld for state-independent action-label calibration"},
      calibration:{purpose:"estimate action-label preference that exists without current-state evidence"}
    });
  }
  boundedText(text){
    const tokenizer=this.classifier?.tokenizer;if(!tokenizer)return text;
    try{
      const encoded=tokenizer(text,{return_tensor:false,truncation:true,max_length:this.maxPremiseTokens});
      let ids=encoded?.input_ids??encoded;if(Array.isArray(ids)&&Array.isArray(ids[0]))ids=ids[0];
      if(Array.isArray(ids)&&typeof tokenizer.decode==="function")return tokenizer.decode(ids,{skip_special_tokens:true});
    }catch{}
    return text;
  }
  async scoreRaw(observation){return this.classifyPremise(this.boundedPremise(observation))}
  async nullBias(){
    if(this.nullBiasLogits)return this.nullBiasLogits;
    if(!this.nullBiasPromise)this.nullBiasPromise=(async()=>this.classifyPremise(this.boundedText(this.nullStateText())))();
    try{this.nullBiasLogits=await this.nullBiasPromise;return this.nullBiasLogits}
    finally{this.nullBiasPromise=null}
  }
  calibrateScores(raw,bias,{strength=this.labelBiasStrength}={}){
    if(!raw?.length||!bias?.length||raw.length!==bias.length)return raw?[...raw]:[];
    const alpha=Math.max(0,Number(strength)||0),corrected=raw.map((v,i)=>Number(v||0)-alpha*Number(bias[i]||0));
    const mean=corrected.reduce((x,y)=>x+y,0)/Math.max(1,corrected.length);
    return corrected.map(v=>v-mean);
  }
  async scoreCalibrated(observation,{strength=this.labelBiasStrength}={}){
    const [raw,bias]=await Promise.all([this.scoreRaw(observation),this.nullBias()]);
    return this.calibrateScores(raw,bias,{strength});
  }
  async score(observation){
    return this.labelBiasCalibration?this.scoreCalibrated(observation):this.scoreRaw(observation);
  }
  async dispose(){if(this.classifier?.dispose)await this.classifier.dispose();this.classifier=null;this.backend="disposed"}
}
