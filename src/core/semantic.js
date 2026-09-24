import {dot,hash32} from "./math.js";

function tokenize(text){return String(text).toLowerCase().match(/[a-z0-9_]+/g)||[]}

function hashEmbedding(text,dim){
  const v=new Float32Array(dim);
  const toks=tokenize(text);
  for(const tok of toks){
    const h=hash32(tok);
    const i=h%dim;
    v[i]+=((h>>>8)&1)?1:-1;
    const j=(h>>>16)%dim;
    v[j]+=.35;
  }
  let n=0;for(const x of v)n+=x*x;n=Math.sqrt(n)||1;
  for(let i=0;i<v.length;i++)v[i]/=n;
  return v;
}

export class HashSemanticAdapter{
  constructor({dim=96,scale=.55}={}){this.dim=dim;this.scale=scale;this.name="HashSemanticAdapter";this.compiled=null}
  compile(schema,actions){
    const fields=schema.fields.map(f=>({...f,vector:hashEmbedding([f.id,f.label,f.description,f.unit||""].join(" "),this.dim)}));
    const actionVectors=actions.map(a=>hashEmbedding([a.id,a.label,a.description].join(" "),this.dim));
    this.compiled={fields,actionVectors,actions};
  }
  stateVector(observation){
    if(!this.compiled)throw new Error("semantic adapter not compiled");
    const out=new Float32Array(this.dim);
    for(const field of this.compiled.fields){
      const raw=Number(observation[field.id]??0);
      const min=field.min??0,max=field.max??1;
      const norm=max===min?0:(raw-min)/(max-min);
      const centered=Math.max(-1,Math.min(1,norm*2-1));
      for(let i=0;i<out.length;i++)out[i]+=field.vector[i]*centered;
    }
    let n=0;for(const x of out)n+=x*x;n=Math.sqrt(n)||1;
    for(let i=0;i<out.length;i++)out[i]/=n;
    return out;
  }
  score(observation){
    const s=this.stateVector(observation);
    return this.compiled.actionVectors.map(a=>dot(s,a)*this.scale);
  }
}
