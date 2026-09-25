import {clamp,hash32,mulberry32,softmax} from "./math.js";

function tokens(text){return String(text||"").toLowerCase().match(/[a-z0-9]+/g)||[]}
function descriptor(field){
  const rich=[field?.label,field?.description,field?.unit].filter(Boolean).join(" ");
  return rich||String(field?.id||"value");
}
function compileSparse(text,dim,baseScale=1){
  const ts=tokens(text),map=new Map();if(!ts.length)return[];
  const norm=baseScale/Math.sqrt(ts.length),add=(i,v)=>map.set(i,(map.get(i)||0)+v);
  for(const token of ts){
    const h=hash32(token);
    add(h%dim,((h>>>24)&1?1:-1)*norm);
    add((h>>>8)%dim,.5*((h>>>23)&1?1:-1)*norm);
    add((h>>>16)%dim,.25*norm);
  }
  return[...map].map(([index,value])=>({index,value}));
}
function addSparse(out,sparse,scale=1){for(const {index,value} of sparse)out[index]+=value*scale}
function addDense(out,dense,scale=1){if(!dense)return;for(let i=0;i<Math.min(out.length,dense.length);i++)out[i]+=dense[i]*scale}
function projectSemanticVector(vector,dim){
  if(!vector||!vector.length)return null;
  const out=new Float32Array(dim);
  for(let i=0;i<vector.length;i++){
    const value=Number(vector[i]);if(!Number.isFinite(value)||value===0)continue;
    let h=Math.imul((i+1)>>>0,0x9e3779b1)>>>0;h^=h>>>16;h=Math.imul(h,0x85ebca6b)>>>0;h^=h>>>13;
    out[h%dim]+=((h>>>31)?1:-1)*value;
  }
  let n=0;for(const v of out)n+=v*v;n=Math.sqrt(n)||1;for(let i=0;i<out.length;i++)out[i]/=n;
  return out;
}
function normalizeValue(value,field={}){
  let v=Number(value);if(!Number.isFinite(v))v=0;
  if(Number.isFinite(field.min)&&Number.isFinite(field.max)&&field.max!==field.min)return clamp(((v-field.min)/(field.max-field.min))*2-1,-1,1);
  const scale=Math.max(1e-6,Number(field.scale)||1),x=v/scale;return x/(1+Math.abs(x));
}
function compileFields(fields,dim){
  return(fields||[]).map(field=>({
    field,
    valueSparse:compileSparse(descriptor(field),dim,1),
    presentSparse:compileSparse(descriptor(field)+" present",dim,.04),
    semanticDense:projectSemanticVector(field.semanticVector,dim),
    enumValues:Object.fromEntries(Object.entries(field.enum||{}).map(([key,label])=>[String(key),{
      label,
      sparse:compileSparse(descriptor(field)+" category "+label,dim,1),
      semanticDense:projectSemanticVector(field.enumSemanticVectors?.[key],dim)
    }]))
  }));
}
function vectorFromCompiled(compiledFields,record,dim,prefixSparse=null,prefixDense=null){
  const out=new Float32Array(dim);if(prefixSparse)addSparse(out,prefixSparse,1);if(prefixDense)addDense(out,prefixDense,.9);
  for(const compiled of compiledFields){
    const {field}=compiled;if(record==null||!Object.prototype.hasOwnProperty.call(record,field.id))continue;
    const raw=record[field.id],enumValue=compiled.enumValues?.[String(raw)];
    addSparse(out,compiled.presentSparse,1);
    if(enumValue){
      addSparse(out,compiled.valueSparse,.2);addDense(out,compiled.semanticDense,.2);
      addSparse(out,enumValue.sparse,1);addDense(out,enumValue.semanticDense,1.1);
    }else{
      const normalized=normalizeValue(raw,field);addSparse(out,compiled.valueSparse,normalized);addDense(out,compiled.semanticDense,normalized*.9);
    }
  }
  return out;
}
function actionVector(action,dim,compiledActionFields=[]){
  const text=[action.label,action.description].filter(Boolean).join(" ")||action.id,out=new Float32Array(dim);
  addSparse(out,compileSparse(text,dim,1));addDense(out,projectSemanticVector(action.semanticVector,dim),1.15);
  const params=action.params||action.values||{};
  for(const compiled of compiledActionFields){
    const {field}=compiled;if(!Object.prototype.hasOwnProperty.call(params,field.id))continue;
    const raw=params[field.id],enumValue=compiled.enumValues?.[String(raw)];
    addSparse(out,compiled.presentSparse,1);
    if(enumValue){
      addSparse(out,enumValue.sparse,1);addDense(out,enumValue.semanticDense,1.05);
    }else{
      const normalized=normalizeValue(raw,field);
      addSparse(out,compiled.valueSparse,normalized);addDense(out,compiled.semanticDense,normalized*.95);
    }
  }
  let n=0;for(const x of out)n+=x*x;n=Math.sqrt(n)||1;for(let i=0;i<out.length;i++)out[i]/=n;return out;
}
class Dense{
  constructor(inputSize,outputSize,rng,{activation="tanh"}={}){
    this.inputSize=inputSize;this.outputSize=outputSize;this.activation=activation;
    this.w=new Float32Array(inputSize*outputSize);this.b=new Float32Array(outputSize);this.gw=new Float32Array(this.w.length);this.gb=new Float32Array(outputSize);
    const scale=Math.sqrt(6/(inputSize+outputSize));for(let i=0;i<this.w.length;i++)this.w[i]=(rng()*2-1)*scale;
  }
  forward(input){
    const out=new Float32Array(this.outputSize);
    for(let o=0;o<this.outputSize;o++){let s=this.b[o],base=o*this.inputSize;for(let i=0;i<this.inputSize;i++)s+=this.w[base+i]*input[i];out[o]=this.activation==="tanh"?Math.tanh(s):s}
    return{input,out};
  }
  zeroGrad(){this.gw.fill(0);this.gb.fill(0)}
  backward(cache,gradOut){
    const gradIn=new Float32Array(this.inputSize);
    for(let o=0;o<this.outputSize;o++){
      let g=gradOut[o];if(this.activation==="tanh")g*=1-cache.out[o]*cache.out[o];this.gb[o]+=g;const base=o*this.inputSize;
      for(let i=0;i<this.inputSize;i++){gradIn[i]+=this.w[base+i]*g;this.gw[base+i]+=g*cache.input[i]}
    }
    return gradIn;
  }
  step(lr,l2=0,clip=1){
    for(let i=0;i<this.w.length;i++)this.w[i]-=lr*(clamp(this.gw[i],-clip,clip)+l2*this.w[i]);
    for(let i=0;i<this.b.length;i++)this.b[i]-=lr*clamp(this.gb[i],-clip,clip);
  }
  count(){return this.w.length+this.b.length}
}
function zeroEntityGrads(n,dim){return Array.from({length:n},()=>new Float32Array(dim))}

export class NeuralSetResidualQ{
  constructor(schema,actions,{seed=2026,hashDim=48,globalDim=16,temporalDim=8,entityHidden=24,entityDim=16,actionDim=24,headDim=32,valueHidden=16,valueWeight=.5,ensembleSize=3,bootstrapProbability=.8,lr=.008,gamma=.96,l2=1e-6,useTargetNetwork=true,targetSyncInterval=24}={}){
    this.schema=schema;this.actions=actions;this.seed=seed;this.hashDim=hashDim;this.globalDim=globalDim;this.temporalDim=temporalDim;this.entityHidden=entityHidden;this.entityDim=entityDim;this.actionDim=actionDim;this.headDim=headDim;this.valueHidden=valueHidden;this.valueWeight=valueWeight;this.ensembleSize=ensembleSize;this.bootstrapProbability=bootstrapProbability;
    this.lr=lr;this.gamma=gamma;this.l2=l2;this.useTargetNetwork=useTargetNetwork;this.targetSyncInterval=Math.max(1,Math.floor(targetSyncInterval));this.name="SchemaSemanticValueSetNet";
    this.attentionStateDim=globalDim+temporalDim;this.queryInputDim=actionDim+this.attentionStateDim;this.contextDim=globalDim+temporalDim+entityDim*2+2;this.headInputDim=this.contextDim+entityDim+actionDim;
    this.setSchema(schema);this.setActions(actions);this.initialize();
  }
  initialize(){
    const rng=mulberry32(this.seed);
    this.globalLayer=new Dense(this.hashDim,this.globalDim,rng);
    this.temporalLayer=new Dense(this.hashDim,this.temporalDim,rng);
    this.entityLayer1=new Dense(this.hashDim,this.entityHidden,rng);
    this.entityLayer2=new Dense(this.entityHidden,this.entityDim,rng);
    this.queryLayer=new Dense(this.queryInputDim,this.entityDim,rng,{activation:"tanh"});
    this.headLayer=new Dense(this.headInputDim,this.headDim,rng);
    this.semanticLayer=new Dense(this.headDim,1,rng,{activation:"linear"});
    this.valueLayer=new Dense(this.headDim,this.valueHidden,rng,{activation:"tanh"});
    this.valueOutLayers=Array.from({length:this.ensembleSize},()=>new Dense(this.valueHidden,1,rng,{activation:"linear"}));
    for(const layer of this.valueOutLayers)for(let i=0;i<layer.w.length;i++)layer.w[i]*=.02;
    this.bootstrapRng=mulberry32((this.seed^0x9e3779b9)>>>0);
    this.semanticLayers=[this.globalLayer,this.temporalLayer,this.entityLayer1,this.entityLayer2,this.queryLayer,this.headLayer,this.semanticLayer];
    this.valueLayers=[this.valueLayer,...this.valueOutLayers];
    this.layers=[...this.semanticLayers,...this.valueLayers];
    this.updates=0;this.distillUpdates=0;this.targetSyncs=0;
    if(this.useTargetNetwork){
      if(!this.targetNet){
        this.targetNet=new NeuralSetResidualQ(this.schema,this.actions,{
          seed:this.seed,hashDim:this.hashDim,globalDim:this.globalDim,temporalDim:this.temporalDim,entityHidden:this.entityHidden,entityDim:this.entityDim,actionDim:this.actionDim,headDim:this.headDim,valueHidden:this.valueHidden,valueWeight:this.valueWeight,
          ensembleSize:this.ensembleSize,bootstrapProbability:this.bootstrapProbability,lr:this.lr,gamma:this.gamma,l2:this.l2,useTargetNetwork:false,targetSyncInterval:this.targetSyncInterval
        });
      }else{
        this.targetNet.setSchema(this.schema);this.targetNet.setActions(this.actions);
      }
      this.syncTarget();
    }
  }
  reset(){this.initialize()}
  copyParametersFrom(source){
    if(!source?.layers||source.layers.length!==this.layers.length)throw new Error("Target network shape mismatch");
    for(let i=0;i<this.layers.length;i++){
      const from=source.layers[i],to=this.layers[i];
      if(from.w.length!==to.w.length||from.b.length!==to.b.length)throw new Error("Target layer shape mismatch");
      to.w.set(from.w);to.b.set(from.b);to.zeroGrad();
    }
    return this;
  }
  syncTarget({value=true}={}){
    if(!this.targetNet)return false;
    if(value){
      this.targetNet.copyParametersFrom(this);this.targetSyncs++;
    }else{
      for(let i=0;i<this.semanticLayers.length;i++){
        const from=this.semanticLayers[i],to=this.targetNet.semanticLayers[i];
        to.w.set(from.w);to.b.set(from.b);to.zeroGrad();
      }
    }
    return true;
  }
  setSchema(schema){
    this.schema=schema;this.compiledGlobals=compileFields(schema.fields,this.hashDim);this.compiledGlobalPrefix=compileSparse("global state objective "+(schema.objective||""),this.hashDim,.25);this.compiledGlobalSemantic=projectSemanticVector(schema.objectiveSemanticVector,this.hashDim);
    this.compiledTemporal=(schema.fields||[]).map(field=>({field,valueSparse:compileSparse("recent change in "+descriptor(field),this.hashDim,1),semanticDense:projectSemanticVector(field.semanticVector,this.hashDim)}));this.compiledTemporalPrefix=compileSparse("recent temporal state change",this.hashDim,.2);
    this.compiledCollections=(schema.collections||[]).map(collection=>({id:collection.id,fields:compileFields(collection.fields,this.hashDim),prefixSparse:compileSparse([collection.label,collection.description].filter(Boolean).join(" ")||collection.id,this.hashDim,.2),prefixDense:projectSemanticVector(collection.semanticVector,this.hashDim)}));
    this.compiledActionFields=compileFields(schema.actionFields||[],this.actionDim);
    if(this.actions)this.actionEmbeddings=this.actions.map(a=>actionVector(a,this.actionDim,this.compiledActionFields));
    if(this.targetNet)this.targetNet.setSchema(schema);
    return this;
  }
  setActions(actions){this.actions=actions;this.actionEmbeddings=actions.map(a=>actionVector(a,this.actionDim,this.compiledActionFields));if(this.targetNet)this.targetNet.setActions(actions);return this}
  parameterCount(){return this.layers.reduce((n,l)=>n+l.count(),0)}
  zeroGrad(){for(const layer of this.layers)layer.zeroGrad()}
  applyLayers(layers,lr=this.lr){for(const layer of layers)layer.step(lr,this.l2)}
  apply(lr=this.lr){this.applyLayers(this.layers,lr)}

  encodeState(observation,{cache=false,temporal=null}={}){
    const globalRaw=vectorFromCompiled(this.compiledGlobals,observation,this.hashDim,this.compiledGlobalPrefix,this.compiledGlobalSemantic),globalCache=this.globalLayer.forward(globalRaw);
    const temporalRaw=new Float32Array(this.hashDim);addSparse(temporalRaw,this.compiledTemporalPrefix,1);
    if(temporal)for(let i=0;i<this.compiledTemporal.length;i++){const v=Number(temporal[i+1]??0);if(Number.isFinite(v)&&v!==0){const x=clamp(v,-1,1);addSparse(temporalRaw,this.compiledTemporal[i].valueSparse,x);addDense(temporalRaw,this.compiledTemporal[i].semanticDense,x*.9)}}
    const temporalCache=this.temporalLayer.forward(temporalRaw);
    const mean=new Float32Array(this.entityDim),max=new Float32Array(this.entityDim),maxIndex=new Int32Array(this.entityDim),latents=[],records=[],recordCaches=cache?[]:null;
    max.fill(-Infinity);maxIndex.fill(-1);let collectionCount=0;
    for(const collection of this.compiledCollections){
      const source=observation?._collections?.[collection.id]||[];if(source.length)collectionCount++;
      for(let ri=0;ri<source.length;ri++){
        const record=source[ri],raw=vectorFromCompiled(collection.fields,record,this.hashDim,collection.prefixSparse,collection.prefixDense),c1=this.entityLayer1.forward(raw),c2=this.entityLayer2.forward(c1.out),z=c2.out,r=latents.length;
        latents.push(z);records.push({collectionId:collection.id,index:ri,record});
        for(let d=0;d<this.entityDim;d++){mean[d]+=z[d];if(z[d]>max[d]){max[d]=z[d];maxIndex[d]=r}}
        if(cache)recordCaches.push({c1,c2});
      }
    }
    const recordCount=latents.length;if(recordCount){for(let d=0;d<this.entityDim;d++)mean[d]/=recordCount}else max.fill(0);
    const context=new Float32Array(this.contextDim);let p=0;context.set(globalCache.out,p);p+=this.globalDim;context.set(temporalCache.out,p);p+=this.temporalDim;context.set(mean,p);p+=this.entityDim;context.set(max,p);p+=this.entityDim;
    context[p++]=Math.log1p(recordCount)/Math.log(1025);context[p]=Math.log1p(collectionCount)/Math.log(17);
    return{context,latents,records,cache:cache?{globalCache,temporalCache,recordCaches,maxIndex,recordCount}:null};
  }
  attention(state,actionIndex){
    const queryInput=new Float32Array(this.queryInputDim);queryInput.set(this.actionEmbeddings[actionIndex],0);queryInput.set(state.context.slice(0,this.attentionStateDim),this.actionDim);
    const queryCache=this.queryLayer.forward(queryInput),query=queryCache.out,n=state.latents.length,weights=new Float32Array(n),attended=new Float32Array(this.entityDim);
    if(!n)return{queryCache,weights,attended};
    const logits=new Float32Array(n),scale=1/Math.sqrt(this.entityDim);let peak=-Infinity;
    for(let r=0;r<n;r++){let s=0,z=state.latents[r];for(let d=0;d<this.entityDim;d++)s+=query[d]*z[d];s*=scale;logits[r]=s;if(s>peak)peak=s}
    let denom=0;for(let r=0;r<n;r++){const e=Math.exp(logits[r]-peak);weights[r]=e;denom+=e}denom=denom||1;
    for(let r=0;r<n;r++){const w=weights[r]/denom;weights[r]=w;const z=state.latents[r];for(let d=0;d<this.entityDim;d++)attended[d]+=w*z[d]}
    return{queryCache,weights,attended};
  }
  actionForward(state,actionIndex){
    const attn=this.attention(state,actionIndex),input=new Float32Array(this.headInputDim);
    input.set(state.context,0);input.set(attn.attended,this.contextDim);input.set(this.actionEmbeddings[actionIndex],this.contextDim+this.entityDim);
    const h=this.headLayer.forward(input),semanticOut=this.semanticLayer.forward(h.out),valueHidden=this.valueLayer.forward(h.out);
    const valueOuts=this.valueOutLayers.map(layer=>layer.forward(valueHidden.out)),valueMemberScores=valueOuts.map(o=>o.out[0]);
    const semanticScore=semanticOut.out[0],valueScore=valueMemberScores.reduce((a,b)=>a+b,0)/valueMemberScores.length;
    const memberScores=valueMemberScores.map(value=>semanticScore+this.valueWeight*value),score=semanticScore+this.valueWeight*valueScore;
    return{score,semanticScore,valueScore,memberScores,valueMemberScores,h,semanticOut,valueHidden,valueOuts,state,actionIndex,...attn};
  }
  scoreStatsObservation(observation,{temporal=null}={}){
    const state=this.encodeState(observation,{temporal}),scores=new Array(this.actions.length),semanticScores=new Array(this.actions.length),valueScores=new Array(this.actions.length),memberScores=new Array(this.actions.length),valueMemberScores=new Array(this.actions.length);
    for(let i=0;i<scores.length;i++){
      const f=this.actionForward(state,i);scores[i]=f.score;semanticScores[i]=f.semanticScore;valueScores[i]=f.valueScore;memberScores[i]=f.memberScores;valueMemberScores[i]=f.valueMemberScores;
    }
    return{scores,semanticScores,valueScores,memberScores,valueMemberScores};
  }
  scoresObservation(observation,{temporal=null}={}){return this.scoreStatsObservation(observation,{temporal}).scores}
  valueScoresObservation(observation,{temporal=null}={}){return this.scoreStatsObservation(observation,{temporal}).valueScores}
  inspectAttention(observation,actionIndex,{topK=8,temporal=null}={}){
    const state=this.encodeState(observation,{temporal}),forward=this.actionForward(state,actionIndex);
    return state.records.map((meta,i)=>({...meta,weight:forward.weights[i]||0})).sort((a,b)=>b.weight-a.weight).slice(0,topK);
  }

  backwardSemantic(forward,gradScore,entityExtra){
    const gh=this.semanticLayer.backward(forward.semanticOut,new Float32Array([gradScore])),gin=this.headLayer.backward(forward.h,gh);
    const gradContext=gin.slice(0,this.contextDim),gradAttended=gin.slice(this.contextDim,this.contextDim+this.entityDim),n=forward.state.latents.length;
    if(n){
      const q=forward.queryCache.out,scale=1/Math.sqrt(this.entityDim),gradQuery=new Float32Array(this.entityDim);
      for(let r=0;r<n;r++){
        const z=forward.state.latents[r],w=forward.weights[r];let centeredDot=0;
        for(let d=0;d<this.entityDim;d++)centeredDot+=gradAttended[d]*(z[d]-forward.attended[d]);
        const gradLogit=w*centeredDot;
        for(let d=0;d<this.entityDim;d++){
          entityExtra[r][d]+=w*gradAttended[d]+gradLogit*q[d]*scale;
          gradQuery[d]+=gradLogit*z[d]*scale;
        }
      }
      const gradQueryInput=this.queryLayer.backward(forward.queryCache,gradQuery);
      for(let d=0;d<this.attentionStateDim;d++)gradContext[d]+=gradQueryInput[this.actionDim+d];
    }
    return gradContext;
  }
  backwardValue(forward,gradValue,{bootstrap=false}={}){
    let active=Array.from({length:this.ensembleSize},(_,i)=>i);
    if(bootstrap){active=active.filter(()=>this.bootstrapRng()<this.bootstrapProbability);if(!active.length)active=[Math.floor(this.bootstrapRng()*this.ensembleSize)]}
    const gh=new Float32Array(this.valueHidden),share=gradValue/active.length;
    for(const i of active){
      const g=this.valueOutLayers[i].backward(forward.valueOuts[i],new Float32Array([share]));
      for(let d=0;d<gh.length;d++)gh[d]+=g[d];
    }
    this.valueLayer.backward(forward.valueHidden,gh);
  }
  backwardState(state,gradContext,entityExtra){
    const cache=state.cache;this.globalLayer.backward(cache.globalCache,gradContext.slice(0,this.globalDim));
    this.temporalLayer.backward(cache.temporalCache,gradContext.slice(this.globalDim,this.globalDim+this.temporalDim));
    const entityStart=this.globalDim+this.temporalDim,meanGrad=gradContext.slice(entityStart,entityStart+this.entityDim),maxGrad=gradContext.slice(entityStart+this.entityDim,entityStart+this.entityDim*2),n=cache.recordCount;
    for(let r=0;r<n;r++){
      const g=new Float32Array(this.entityDim);
      for(let d=0;d<this.entityDim;d++)g[d]=meanGrad[d]/n+(cache.maxIndex[d]===r?maxGrad[d]:0)+(entityExtra?.[r]?.[d]||0);
      const g1=this.entityLayer2.backward(cache.recordCaches[r].c2,g);this.entityLayer1.backward(cache.recordCaches[r].c1,g1);
    }
  }
  updateTransition({observation,temporal=null,actionIndex,reward,nextObservation,nextTemporal=null,done=false}){
    const current=this.encodeState(observation,{cache:false,temporal}),chosen=this.actionForward(current,actionIndex);
    const bootstrapModel=this.targetNet||this,nextScores=done?[]:bootstrapModel.valueScoresObservation(nextObservation,{temporal:nextTemporal});
    const target=reward+(done?0:this.gamma*Math.max(...nextScores)),td=clamp(target-chosen.valueScore,-4,4);
    this.zeroGrad();this.backwardValue(chosen,-td,{bootstrap:true});this.applyLayers(this.valueLayers);this.updates++;
    if(this.targetNet&&this.updates%this.targetSyncInterval===0)this.syncTarget();
    return{td,target,q:chosen.valueScore,combined:chosen.score,semantic:chosen.semanticScore,targetNetwork:!!this.targetNet,targetSyncs:this.targetSyncs};
  }
  distill(observation,teacherScores,{strength=.35,temporal=null}={}){
    if(!teacherScores||teacherScores.length!==this.actions.length)return null;
    const state=this.encodeState(observation,{cache:true,temporal}),forwards=this.actions.map((_,i)=>this.actionForward(state,i)),student=softmax(forwards.map(x=>x.semanticScore),1),teacher=softmax(teacherScores,1);
    const gradContext=new Float32Array(this.contextDim),entityExtra=zeroEntityGrads(state.latents.length,this.entityDim);this.zeroGrad();let loss=0;
    for(let i=0;i<this.actions.length;i++){
      loss-=teacher[i]*Math.log(Math.max(1e-8,student[i]));
      const grad=this.backwardSemantic(forwards[i],student[i]-teacher[i],entityExtra);for(let j=0;j<this.contextDim;j++)gradContext[j]+=grad[j];
    }
    this.backwardState(state,gradContext,entityExtra);this.applyLayers(this.semanticLayers,this.lr*strength);this.distillUpdates++;return{loss,student,teacher};
  }
}
