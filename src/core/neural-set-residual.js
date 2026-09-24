import {clamp,hash32,mulberry32,softmax} from "./math.js";

function tokens(text){return String(text||"").toLowerCase().match(/[a-z0-9]+/g)||[]}

function descriptor(field){
  const rich=[field?.label,field?.description,field?.unit].filter(Boolean).join(" ");
  return rich||String(field?.id||"value");
}

function addHashed(out,text,scale=1){
  const ts=tokens(text);
  if(!ts.length)return;
  const norm=scale/Math.sqrt(ts.length);
  for(const token of ts){
    const h=hash32(token);
    const i=h%out.length,j=(h>>>8)%out.length,k=(h>>>16)%out.length;
    out[i]+=((h>>>24)&1?1:-1)*norm;
    out[j]+=.5*((h>>>23)&1?1:-1)*norm;
    out[k]+=.25*norm;
  }
}

function normalizeValue(value,field={}){
  let v=Number(value);
  if(!Number.isFinite(v))v=0;
  if(Number.isFinite(field.min)&&Number.isFinite(field.max)&&field.max!==field.min){
    return clamp(((v-field.min)/(field.max-field.min))*2-1,-1,1);
  }
  const scale=Math.max(1e-6,Number(field.scale)||1);
  const x=v/scale;
  return x/(1+Math.abs(x));
}

function schemaVector(schemaFields,record,dim,prefix=""){
  const out=new Float32Array(dim);
  addHashed(out,prefix,.2);
  for(const field of schemaFields||[]){
    const has=record!=null&&Object.prototype.hasOwnProperty.call(record,field.id);
    if(!has)continue;
    const v=normalizeValue(record[field.id],field);
    addHashed(out,descriptor(field),v);
    addHashed(out,descriptor(field)+" present",.04);
  }
  return out;
}

function actionVector(action,dim){
  const out=new Float32Array(dim);
  const text=[action.label,action.description].filter(Boolean).join(" ")||action.id;
  addHashed(out,text,1);
  let n=0;for(const x of out)n+=x*x;n=Math.sqrt(n)||1;
  for(let i=0;i<out.length;i++)out[i]/=n;
  return out;
}

class Dense{
  constructor(inputSize,outputSize,rng,{activation="tanh"}={}){
    this.inputSize=inputSize;this.outputSize=outputSize;this.activation=activation;
    this.w=new Float32Array(inputSize*outputSize);this.b=new Float32Array(outputSize);
    this.gw=new Float32Array(this.w.length);this.gb=new Float32Array(outputSize);
    const scale=Math.sqrt(6/(inputSize+outputSize));
    for(let i=0;i<this.w.length;i++)this.w[i]=(rng()*2-1)*scale;
  }
  forward(input){
    const out=new Float32Array(this.outputSize);
    for(let o=0;o<this.outputSize;o++){
      let s=this.b[o],base=o*this.inputSize;
      for(let i=0;i<this.inputSize;i++)s+=this.w[base+i]*input[i];
      out[o]=this.activation==="tanh"?Math.tanh(s):s;
    }
    return{input,out};
  }
  zeroGrad(){this.gw.fill(0);this.gb.fill(0)}
  backward(cache,gradOut){
    const gradIn=new Float32Array(this.inputSize);
    for(let o=0;o<this.outputSize;o++){
      let g=gradOut[o];
      if(this.activation==="tanh")g*=1-cache.out[o]*cache.out[o];
      this.gb[o]+=g;
      const base=o*this.inputSize;
      for(let i=0;i<this.inputSize;i++){
        gradIn[i]+=this.w[base+i]*g;
        this.gw[base+i]+=g*cache.input[i];
      }
    }
    return gradIn;
  }
  step(lr,l2=0,clip=1){
    for(let i=0;i<this.w.length;i++){
      const g=clamp(this.gw[i],-clip,clip)+l2*this.w[i];
      this.w[i]-=lr*g;
    }
    for(let i=0;i<this.b.length;i++)this.b[i]-=lr*clamp(this.gb[i],-clip,clip);
  }
  count(){return this.w.length+this.b.length}
}

export class NeuralSetResidualQ{
  constructor(schema,actions,{seed=2026,hashDim=48,globalDim=16,entityHidden=24,entityDim=16,actionDim=24,headDim=32,lr=.008,gamma=.96,l2=1e-6}={}){
    this.schema=schema;this.actions=actions;this.seed=seed;this.hashDim=hashDim;this.globalDim=globalDim;this.entityHidden=entityHidden;this.entityDim=entityDim;this.actionDim=actionDim;this.headDim=headDim;
    this.lr=lr;this.gamma=gamma;this.l2=l2;this.name="SchemaHashSetNet";
    this.actionEmbeddings=actions.map(a=>actionVector(a,actionDim));
    this.contextDim=globalDim+entityDim*2+2;
    this.headInputDim=this.contextDim+actionDim;
    this.initialize();
  }
  initialize(){
    const rng=mulberry32(this.seed);
    this.globalLayer=new Dense(this.hashDim,this.globalDim,rng);
    this.entityLayer1=new Dense(this.hashDim,this.entityHidden,rng);
    this.entityLayer2=new Dense(this.entityHidden,this.entityDim,rng);
    this.headLayer=new Dense(this.headInputDim,this.headDim,rng);
    this.outLayer=new Dense(this.headDim,1,rng,{activation:"linear"});
    this.layers=[this.globalLayer,this.entityLayer1,this.entityLayer2,this.headLayer,this.outLayer];
    this.updates=0;this.distillUpdates=0;
  }
  reset(){this.initialize()}
  parameterCount(){return this.layers.reduce((n,l)=>n+l.count(),0)}
  zeroGrad(){for(const layer of this.layers)layer.zeroGrad()}
  apply(lr=this.lr){for(const layer of this.layers)layer.step(lr,this.l2)}

  encodeState(observation,{cache=false}={}){
    const globalRaw=schemaVector(this.schema.fields,observation,this.hashDim,"global state");
    const globalCache=this.globalLayer.forward(globalRaw);
    const recordCaches=[],latents=[];
    let collectionCount=0;
    for(const collection of this.schema.collections||[]){
      const records=observation?._collections?.[collection.id]||[];
      if(records.length)collectionCount++;
      const prefix=[collection.label,collection.description].filter(Boolean).join(" ")||collection.id;
      for(const record of records){
        const raw=schemaVector(collection.fields,record,this.hashDim,prefix);
        const c1=this.entityLayer1.forward(raw),c2=this.entityLayer2.forward(c1.out);
        latents.push(c2.out);
        if(cache)recordCaches.push({c1,c2});
      }
    }
    const mean=new Float32Array(this.entityDim),max=new Float32Array(this.entityDim),maxIndex=new Int32Array(this.entityDim);
    max.fill(-Infinity);maxIndex.fill(-1);
    for(let r=0;r<latents.length;r++){
      const z=latents[r];
      for(let d=0;d<this.entityDim;d++){
        mean[d]+=z[d];
        if(z[d]>max[d]){max[d]=z[d];maxIndex[d]=r}
      }
    }
    if(latents.length){
      for(let d=0;d<this.entityDim;d++)mean[d]/=latents.length;
    }else max.fill(0);
    const context=new Float32Array(this.contextDim);
    let p=0;context.set(globalCache.out,p);p+=this.globalDim;context.set(mean,p);p+=this.entityDim;context.set(max,p);p+=this.entityDim;
    context[p++]=Math.log1p(latents.length)/Math.log(1025);
    context[p]=Math.log1p(collectionCount)/Math.log(17);
    return{context,cache:cache?{globalCache,recordCaches,maxIndex,recordCount:latents.length}:null};
  }

  actionForward(context,actionIndex){
    const input=new Float32Array(this.headInputDim);
    input.set(context,0);input.set(this.actionEmbeddings[actionIndex],this.contextDim);
    const h=this.headLayer.forward(input),o=this.outLayer.forward(h.out);
    return{score:o.out[0],h,o};
  }

  scoresObservation(observation){
    const state=this.encodeState(observation);
    const scores=new Array(this.actions.length);
    for(let i=0;i<scores.length;i++)scores[i]=this.actionForward(state.context,i).score;
    return scores;
  }

  backwardState(stateCache,gradContext){
    const globalGrad=gradContext.slice(0,this.globalDim);
    this.globalLayer.backward(stateCache.globalCache,globalGrad);
    const meanGrad=gradContext.slice(this.globalDim,this.globalDim+this.entityDim);
    const maxGrad=gradContext.slice(this.globalDim+this.entityDim,this.globalDim+this.entityDim*2);
    const n=stateCache.recordCount;
    if(!n)return;
    for(let r=0;r<n;r++){
      const g=new Float32Array(this.entityDim);
      for(let d=0;d<this.entityDim;d++)g[d]=meanGrad[d]/n+(stateCache.maxIndex[d]===r?maxGrad[d]:0);
      const g1=this.entityLayer2.backward(stateCache.recordCaches[r].c2,g);
      this.entityLayer1.backward(stateCache.recordCaches[r].c1,g1);
    }
  }

  backwardAction(forward,gradScore){
    const go=new Float32Array([gradScore]);
    const gh=this.outLayer.backward(forward.o,go);
    return this.headLayer.backward(forward.h,gh);
  }

  updateTransition({observation,actionIndex,reward,nextObservation,done=false}){
    const current=this.encodeState(observation,{cache:true});
    const chosen=this.actionForward(current.context,actionIndex);
    const nextScores=done?[]:this.scoresObservation(nextObservation);
    const target=reward+(done?0:this.gamma*Math.max(...nextScores));
    const td=clamp(target-chosen.score,-4,4);
    this.zeroGrad();
    const gradInput=this.backwardAction(chosen,-td);
    this.backwardState(current.cache,gradInput.slice(0,this.contextDim));
    this.apply();
    this.updates++;
    return{td,target,q:chosen.score};
  }

  distill(observation,teacherScores,{strength=.35}={}){
    if(!teacherScores||teacherScores.length!==this.actions.length)return null;
    const state=this.encodeState(observation,{cache:true});
    const forwards=this.actions.map((_,i)=>this.actionForward(state.context,i));
    const student=softmax(forwards.map(x=>x.score),1);
    const teacher=softmax(teacherScores,1);
    const gradContext=new Float32Array(this.contextDim);
    this.zeroGrad();
    let loss=0;
    for(let i=0;i<this.actions.length;i++){
      loss-=teacher[i]*Math.log(Math.max(1e-8,student[i]));
      const grad=this.backwardAction(forwards[i],student[i]-teacher[i]);
      for(let j=0;j<this.contextDim;j++)gradContext[j]+=grad[j];
    }
    this.backwardState(state.cache,gradContext);
    this.apply(this.lr*strength);
    this.distillUpdates++;
    return{loss,student,teacher};
  }
}
