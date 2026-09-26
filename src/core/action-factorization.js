import {clamp} from "./math.js";

function solveLinear(matrix,vector,n){
  const rows=Array.from({length:n},(_,r)=>{const out=new Float64Array(n+1);for(let c=0;c<n;c++)out[c]=matrix[r*n+c];out[n]=vector[r];return out});
  for(let c=0;c<n;c++){
    let pivot=c,best=Math.abs(rows[c][c]);for(let r=c+1;r<n;r++){const v=Math.abs(rows[r][c]);if(v>best){best=v;pivot=r}}
    if(best<1e-12)continue;if(pivot!==c){const tmp=rows[c];rows[c]=rows[pivot];rows[pivot]=tmp}
    const scale=rows[c][c];for(let j=c;j<=n;j++)rows[c][j]/=scale;
    for(let r=0;r<n;r++){if(r===c)continue;const factor=rows[r][c];if(Math.abs(factor)<1e-15)continue;for(let j=c;j<=n;j++)rows[r][j]-=factor*rows[c][j]}
  }
  const out=new Float64Array(n);for(let i=0;i<n;i++)out[i]=Number.isFinite(rows[i][n])?rows[i][n]:0;return out;
}
function fieldColumns(schema){
  const columns=[{id:"__bias",label:"bias",value:()=>1}];
  for(const field of schema?.actionFields||[]){
    if(field.enum){
      for(const [key,label] of Object.entries(field.enum))columns.push({id:field.id+"="+key,label:(field.label||field.id)+"="+label,value:action=>String(action?.params?.[field.id]??"__unset")===String(key)?1:0});
    }else{
      columns.push({id:field.id,label:field.label||field.id,value:action=>{
        const raw=Number(action?.params?.[field.id]??0);if(!Number.isFinite(raw))return 0;const lo=Number(field.min),hi=Number(field.max);
        if(Number.isFinite(lo)&&Number.isFinite(hi)&&hi!==lo)return clamp((raw-lo)/(hi-lo),0,1);return clamp(raw,-1,1);
      }});
    }
  }
  return columns;
}
function center(values){const mean=values.reduce((a,b)=>a+Number(b||0),0)/Math.max(1,values.length);return values.map(v=>Number(v||0)-mean)}

export function compileActionFieldProjector(schema,actions,{ridge=.05,maxBlend=.7}={}){
  const valid=Array.isArray(actions)&&actions.length>=2,columns=fieldColumns(schema),n=columns.length,rows=valid?actions.map(action=>columns.map(col=>Number(col.value(action)||0))):[];
  const gram=new Float64Array(n*n);
  for(const x of rows)for(let i=0;i<n;i++)for(let j=0;j<n;j++)gram[i*n+j]+=x[i]*x[j];
  for(let i=0;i<n;i++)gram[i*n+i]+=i===0?ridge*.1:ridge;
  const inverse=new Float64Array(n*n);
  if(valid){
    for(let col=0;col<n;col++){
      const basis=new Float64Array(n);basis[col]=1;const solution=solveLinear(gram,basis,n);
      for(let row=0;row<n;row++)inverse[row*n+col]=solution[row];
    }
  }
  const project=(valueScores,{maxBlend:blend=maxBlend}={})=>{
    if(!valid||!Array.isArray(valueScores)||valueScores.length!==actions.length)return{scores:valueScores?center(valueScores):[],projected:valueScores?center(valueScores):[],fitQuality:0,blendUsed:0,coefficients:[]};
    const y=center(valueScores),rhs=new Float64Array(n);
    for(let r=0;r<rows.length;r++){const x=rows[r],target=y[r];for(let i=0;i<n;i++)rhs[i]+=x[i]*target}
    const weights=new Float64Array(n);
    for(let i=0;i<n;i++)for(let j=0;j<n;j++)weights[i]+=inverse[i*n+j]*rhs[j];
    const projected=center(rows.map(x=>x.reduce((sum,v,i)=>sum+v*weights[i],0)));
    const sst=y.reduce((s,v)=>s+v*v,0),sse=y.reduce((s,v,i)=>s+(v-projected[i])**2,0),fitQuality=sst>1e-12?clamp(1-sse/sst,0,1):0,blendUsed=clamp(Number(blend)||0,0,1)*fitQuality;
    const scores=y.map((v,i)=>(1-blendUsed)*v+blendUsed*projected[i]);
    return{scores,projected,fitQuality,blendUsed,coefficients:columns.map((col,i)=>({id:col.id,label:col.label,weight:Number(weights[i])}))};
  };
  return{project,columns:columns.map(({id,label})=>({id,label})),actionCount:actions?.length||0,basisSize:n,ridge};
}

export function projectValueToActionFields(schema,actions,valueScores,{ridge=.05,maxBlend=.7}={}){
  return compileActionFieldProjector(schema,actions,{ridge,maxBlend}).project(valueScores);
}
