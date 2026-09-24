const TRANSFORMERS_CDN="https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.0";

export const NLI_PRESETS=Object.freeze({
  mobilebert:{label:"MobileBERT-MNLI",modelId:"Xenova/mobilebert-uncased-mnli",gpuDtype:"fp16",wasmDtype:"int8",approx:"~50 MB WebGPU / ~26 MB WASM"},
  distilbert:{label:"DistilBERT-MNLI",modelId:"Xenova/distilbert-base-uncased-mnli",gpuDtype:"fp16",wasmDtype:"int8",approx:"~134 MB WebGPU / ~67 MB WASM"}
});
function normalizeProgress(info){
  if(!info||typeof info!=="object")return null;
  if(Number.isFinite(info.progress))return Math.max(0,Math.min(100,info.progress));
  if(Number.isFinite(info.loaded)&&Number.isFinite(info.total)&&info.total>0)return Math.max(0,Math.min(100,100*info.loaded/info.total));
  return null;
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
  const mean=raw.reduce((a,b)=>a+b,0)/raw.length,nmean=norm.reduce((a,b)=>a+b,0)/norm.length;
  const variance=norm.reduce((s,v)=>s+(v-nmean)*(v-nmean),0)/norm.length;
  return{field,mean,min:Math.min(...raw),max:Math.max(...raw),variance};
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
export function summarizeCollection(collection,records,{maxFields=6,representatives=2}={}){
  const stats=(collection.fields||[]).map(field=>fieldStats(field,records)).filter(Boolean).sort((a,b)=>b.variance-a.variance).slice(0,maxFields);
  const label=collection.label||collection.id,description=collection.description?" ("+collection.description+")":"";
  const lines=[label+description+": "+records.length+" records."];
  if(stats.length)lines.push("stats: "+stats.map(s=>(s.field.label||s.field.id)+" mean="+s.mean.toFixed(2)+" min="+s.min.toFixed(2)+" max="+s.max.toFixed(2)).join("; "));
  const reps=representativeRecords(records,stats,representatives);
  reps.forEach((record,i)=>lines.push("representative "+(i+1)+": {"+stats.map(s=>(s.field.label||s.field.id)+":"+Number(record[s.field.id]??0).toFixed(2)).join(", ")+"}"));
  return lines.join("\n");
}

export class TransformersNLIAdapter{
  constructor({preset="mobilebert",onProgress=()=>{},maxStateChars=5000}={}){
    if(!NLI_PRESETS[preset])throw new Error("Unknown NLI preset: "+preset);
    this.presetKey=preset;this.preset=NLI_PRESETS[preset];this.name=this.preset.label;this.onProgress=onProgress;this.maxStateChars=maxStateChars;
    this.schema=null;this.actions=null;this.labels=null;this.classifier=null;this.backend="unloaded";
  }
  compile(schema,actions){this.schema=schema;this.actions=actions;this.labels=actions.map(a=>a.label+" — "+a.description)}
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
  stateText(observation){
    const objective=this.schema?.objective||"choose the action that best advances the environment objective";
    const lines=["Objective: "+objective,"Global state:"];
    for(const field of this.schema.fields){
      const value=Number(observation[field.id]??0),label=field.label||field.id;
      let relative="";
      if(Number.isFinite(field.min)&&Number.isFinite(field.max)&&field.max!==field.min)relative=" ["+Math.round(((value-field.min)/(field.max-field.min))*100)+"% of declared range]";
      lines.push("- "+label+"="+value.toFixed(3)+relative+(field.description?" ("+field.description+")":""));
    }
    for(const collection of this.schema.collections||[]){
      const records=observation?._collections?.[collection.id]||[];
      lines.push("Collection: "+summarizeCollection(collection,records));
    }
    const text=lines.join("\n");
    return text.length<=this.maxStateChars?text:text.slice(0,this.maxStateChars)+"\n[bounded teacher synopsis truncated]";
  }
  async score(observation){
    if(!this.classifier)await this.load();
    const output=await this.classifier(this.stateText(observation),this.labels,{multi_label:false,hypothesis_template:"For the stated objective and current state, choosing {} is an appropriate next action."});
    const scores=new Map(output.labels.map((label,i)=>[label,output.scores[i]])),floor=1e-7;
    return this.labels.map(label=>Math.log(Math.max(floor,scores.get(label)??floor)));
  }
  async dispose(){if(this.classifier?.dispose)await this.classifier.dispose();this.classifier=null;this.backend="disposed"}
}
