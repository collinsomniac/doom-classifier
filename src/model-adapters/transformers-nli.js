const TRANSFORMERS_CDN="https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.0";

export const NLI_PRESETS=Object.freeze({
  mobilebert:{
    label:"MobileBERT-MNLI",
    modelId:"Xenova/mobilebert-uncased-mnli",
    gpuDtype:"fp16",
    wasmDtype:"int8",
    approx:"~50 MB WebGPU / ~26 MB WASM"
  },
  distilbert:{
    label:"DistilBERT-MNLI",
    modelId:"Xenova/distilbert-base-uncased-mnli",
    gpuDtype:"fp16",
    wasmDtype:"int8",
    approx:"~134 MB WebGPU / ~67 MB WASM"
  }
});

function normalizeProgress(info){
  if(!info||typeof info!=="object")return null;
  if(Number.isFinite(info.progress))return Math.max(0,Math.min(100,info.progress));
  if(Number.isFinite(info.loaded)&&Number.isFinite(info.total)&&info.total>0)return Math.max(0,Math.min(100,100*info.loaded/info.total));
  return null;
}

export class TransformersNLIAdapter{
  constructor({preset="mobilebert",onProgress=()=>{}}={}){
    if(!NLI_PRESETS[preset])throw new Error("Unknown NLI preset: "+preset);
    this.presetKey=preset;
    this.preset=NLI_PRESETS[preset];
    this.name=this.preset.label;
    this.onProgress=onProgress;
    this.schema=null;
    this.actions=null;
    this.labels=null;
    this.classifier=null;
    this.backend="unloaded";
  }

  compile(schema,actions){
    this.schema=schema;
    this.actions=actions;
    this.labels=actions.map(a=>a.label+" — "+a.description);
  }

  report(info){
    this.onProgress({...info,normalizedProgress:normalizeProgress(info)});
  }

  async load(){
    if(this.classifier)return this;
    this.report({status:"loading-library",file:"Transformers.js"});
    const mod=await import(TRANSFORMERS_CDN);
    mod.env.allowLocalModels=false;

    const canWebGPU=typeof navigator!=="undefined"&&!!navigator.gpu;
    const attempts=canWebGPU
      ? [{device:"webgpu",dtype:this.preset.gpuDtype},{device:"wasm",dtype:this.preset.wasmDtype}]
      : [{device:"wasm",dtype:this.preset.wasmDtype}];

    let lastError=null;
    for(const attempt of attempts){
      try{
        this.report({status:"loading-model",file:this.preset.label+" "+attempt.device+" "+attempt.dtype});
        this.classifier=await mod.pipeline("zero-shot-classification",this.preset.modelId,{
          device:attempt.device,
          dtype:attempt.dtype,
          progress_callback:info=>this.report(info)
        });
        this.backend=attempt.device+" / "+attempt.dtype;
        this.report({status:"ready",file:this.backend,progress:100});
        return this;
      }catch(error){
        lastError=error;
        this.report({status:"backend-failed",file:attempt.device+" / "+attempt.dtype,error:String(error?.message||error)});
        if(this.classifier?.dispose)await this.classifier.dispose();
        this.classifier=null;
      }
    }
    throw lastError||new Error("Unable to load NLI model");
  }

  stateText(observation){
    const objective=this.schema?.objective||"choose the action that best advances the environment objective";
    const lines=["Objective: "+objective,"Current structured environment state:"];
    for(const field of this.schema.fields){
      const value=Number(observation[field.id]??0);
      let suffix="";
      if(Number.isFinite(field.min)&&Number.isFinite(field.max)&&field.max!==field.min){
        const relative=(value-field.min)/(field.max-field.min);
        suffix="; relative position "+Math.round(relative*100)+"% within declared range ["+field.min+", "+field.max+"]";
      }
      lines.push("- "+field.label+" ("+field.description+"): "+value.toFixed(4)+suffix+(field.unit?"; unit "+field.unit:""));
    }
    return lines.join("\n");
  }

  async score(observation){
    if(!this.classifier)await this.load();
    const output=await this.classifier(this.stateText(observation),this.labels,{
      multi_label:false,
      hypothesis_template:"For the stated objective and current state, choosing {} is an appropriate next action."
    });
    const scores=new Map(output.labels.map((label,i)=>[label,output.scores[i]]));
    const floor=1e-7;
    return this.labels.map(label=>Math.log(Math.max(floor,scores.get(label)??floor)));
  }

  async dispose(){
    if(this.classifier?.dispose)await this.classifier.dispose();
    this.classifier=null;
    this.backend="disposed";
  }
}
