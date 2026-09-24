export class TemporalMemory{
  constructor(size,{decay=.82}={}){this.size=size;this.decay=decay;this.value=new Float32Array(size);this.prev=null}
  reset(){this.value.fill(0);this.prev=null}
  project(features,enabled=true,commit=true){
    const delta=new Float32Array(this.size);
    if(this.prev)for(let i=0;i<this.size;i++)delta[i]=features[i]-this.prev[i];
    const next=new Float32Array(this.size);
    if(enabled)for(let i=0;i<this.size;i++)next[i]=this.decay*this.value[i]+(1-this.decay)*delta[i];
    if(commit){this.value.set(next);this.prev=new Float32Array(features)}
    return next;
  }
  update(features,enabled=true){return this.project(features,enabled,true)}
  preview(features,enabled=true){return this.project(features,enabled,false)}
}
