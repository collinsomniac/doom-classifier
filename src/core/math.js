export function clamp(v,min,max){return Math.max(min,Math.min(max,v))}
export function dot(a,b){let s=0;for(let i=0;i<a.length;i++)s+=a[i]*b[i];return s}
export function softmax(values,temperature=1){
  const t=Math.max(.05,temperature);
  const m=Math.max(...values);
  const xs=values.map(v=>Math.exp((v-m)/t));
  const z=xs.reduce((a,b)=>a+b,0)||1;
  return xs.map(v=>v/z);
}
export function argmax(xs){let k=0;for(let i=1;i<xs.length;i++)if(xs[i]>xs[k])k=i;return k}
export function sampleCategorical(probabilities,rng=Math.random){const r=rng();let c=0;for(let i=0;i<probabilities.length;i++){c+=probabilities[i];if(r<=c)return i}return Math.max(0,probabilities.length-1)}
export function entropyNormalized(p){
  if(p.length<2)return 0;
  let h=0;for(const x of p)if(x>0)h-=x*Math.log(x);
  return h/Math.log(p.length);
}
export function percentile(xs,p){
  if(!xs.length)return 0;
  const s=[...xs].sort((a,b)=>a-b);
  const i=(s.length-1)*p,lo=Math.floor(i),hi=Math.ceil(i);
  return lo===hi?s[lo]:s[lo]+(s[hi]-s[lo])*(i-lo);
}
export function mulberry32(seed){
  let a=seed>>>0;
  return function(){a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296}
}
export function hash32(text){
  let h=2166136261>>>0;
  for(let i=0;i<text.length;i++){h^=text.charCodeAt(i);h=Math.imul(h,16777619)}
  return h>>>0;
}
