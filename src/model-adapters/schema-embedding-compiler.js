const TRANSFORMERS_CDN="https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.0";

export const SCHEMA_EMBEDDING_PRESET=Object.freeze({
  label:"MiniLM schema compiler",
  modelId:"Xenova/all-MiniLM-L6-v2",
  gpuDtype:"fp16",
  wasmDtype:"int8",
  approx:"~45 MB WebGPU / ~23 MB WASM"
});

function describeField(field){
  return [field.label,field.description,field.unit].filter(Boolean).join(" ")||String(field.id||"value");
}
function describeCollection(collection){
  return [collection.label,collection.description].filter(Boolean).join(" ")||String(collection.id||"collection");
}
function describeAction(action){
  return [action.label,action.description].filter(Boolean).join(" ")||String(action.id||"action");
}
function rowsFromTensor(output,count){
  if(output?.tolist){
    const rows=output.tolist();
    if(Array.isArray(rows)&&rows.length===count&&Array.isArray(rows[0]))return rows;
  }
  const dims=output?.dims||[],data=output?.data;
  if(!data||dims.length<2)throw new Error("Unexpected embedding tensor shape");
  const width=dims[dims.length-1],rows=[];
  for(let i=0;i<count;i++)rows.push(Array.from(data.slice(i*width,(i+1)*width)));
  return rows;
}
function cloneSchema(schema){
  return{
    ...schema,
    fields:(schema.fields||[]).map(field=>({...field})),
    actionFields:(schema.actionFields||[]).map(field=>({...field})),
    collections:(schema.collections||[]).map(collection=>({...collection,fields:(collection.fields||[]).map(field=>({...field}))}))
  };
}

export class MiniLMSchemaCompiler{
  constructor({onProgress=()=>{}}={}){
    this.name=SCHEMA_EMBEDDING_PRESET.label;
    this.onProgress=onProgress;
    this.extractor=null;
    this.backend="unloaded";
  }
  report(info){this.onProgress(info)}
  async load(){
    if(this.extractor)return this;
    this.report({status:"loading-library",message:"loading Transformers.js"});
    const mod=await import(TRANSFORMERS_CDN);
    mod.env.allowLocalModels=false;
    const canWebGPU=typeof navigator!=="undefined"&&!!navigator.gpu;
    const attempts=canWebGPU
      ? [{device:"webgpu",dtype:SCHEMA_EMBEDDING_PRESET.gpuDtype},{device:"wasm",dtype:SCHEMA_EMBEDDING_PRESET.wasmDtype}]
      : [{device:"wasm",dtype:SCHEMA_EMBEDDING_PRESET.wasmDtype}];
    let lastError=null;
    for(const attempt of attempts){
      try{
        this.report({status:"loading-model",message:SCHEMA_EMBEDDING_PRESET.label+" "+attempt.device+" "+attempt.dtype});
        this.extractor=await mod.pipeline("feature-extraction",SCHEMA_EMBEDDING_PRESET.modelId,{
          device:attempt.device,
          dtype:attempt.dtype,
          progress_callback:info=>this.report({...info,status:info.status||"loading-model"})
        });
        this.backend=attempt.device+" / "+attempt.dtype;
        this.report({status:"ready",message:this.backend});
        return this;
      }catch(error){
        lastError=error;
        if(this.extractor?.dispose)await this.extractor.dispose();
        this.extractor=null;
        this.report({status:"backend-failed",message:attempt.device+" / "+attempt.dtype+" · "+String(error?.message||error)});
      }
    }
    throw lastError||new Error("Unable to load schema embedding model");
  }
  async embed(texts){
    if(!this.extractor)await this.load();
    if(!texts.length)return[];
    const output=await this.extractor(texts,{pooling:"mean",normalize:true});
    return rowsFromTensor(output,texts.length);
  }
  async compile(schema,actions){
    if(!this.extractor)await this.load();
    const nextSchema=cloneSchema(schema);
    const nextActions=(actions||[]).map(action=>({...action,params:action.params?{...action.params}:action.params,values:action.values?{...action.values}:action.values}));
    const jobs=[];
    const add=(target,key,text)=>{const normalized=String(text||"").trim();if(normalized)jobs.push({target,key,text:normalized})};

    add(nextSchema,"objectiveSemanticVector",nextSchema.objective||"environment objective");
    for(const field of nextSchema.fields)add(field,"semanticVector",describeField(field));
    for(const field of nextSchema.actionFields)add(field,"semanticVector",describeField(field));
    for(const collection of nextSchema.collections){
      add(collection,"semanticVector",describeCollection(collection));
      for(const field of collection.fields)add(field,"semanticVector",describeField(field));
    }
    for(const action of nextActions)add(action,"semanticVector",describeAction(action));

    const unique=[...new Set(jobs.map(job=>job.text))];
    this.report({status:"embedding",message:"embedding "+unique.length+" unique schema/action descriptions"});
    const rows=await this.embed(unique),byText=new Map(unique.map((text,i)=>[text,rows[i]]));
    for(const job of jobs)job.target[job.key]=byText.get(job.text);
    const dimensions=rows[0]?.length||0;
    this.report({status:"compiled",message:jobs.length+" semantic bindings · "+dimensions+" dimensions"});
    return{schema:nextSchema,actions:nextActions,meta:{uniqueTexts:unique.length,bindings:jobs.length,dimensions,backend:this.backend,modelId:SCHEMA_EMBEDDING_PRESET.modelId}};
  }
  async dispose(){
    if(this.extractor?.dispose)await this.extractor.dispose();
    this.extractor=null;
    this.backend="disposed";
  }
}
