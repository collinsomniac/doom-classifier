export class TemporalMemory{
  constructor(size,{decay=.82}={}){this.size=size;this.decay=decay;this.value=new Float32Array(size);this.prev=null}
  reset(){this.value.fill(0);this.prev=null}
  update(features,enabled=true){
    const delta=new Float32Array(this.size);
    if(this.prev)for(let i=0;i<this.size;i++)delta[i]=features[i]-this.prev[i];
    if(enabled){for(let i=0;i<this.size;i++)this.value[i]=this.decay*this.value[i]+(1-this.decay)*delta[i]}
    else this.value.fill(0);
    this.prev=new Float32Array(features);
    return this.value;
  }
}
