import assert from "node:assert/strict";
import {mulberry32,softmax} from "../src/core/math.js";
import {calibrationMetrics,fitTemperature,logitsToProbabilities,TemperatureCalibrator} from "../src/core/calibration.js";

// Generic structured decision task: route work among three typed actions.
// Ground-truth probabilities are known and labels are sampled from them, so
// probability calibration is measurable rather than inferred from top-1 rank.
const rng=mulberry32(20260926);
function caseAt(i){
  const queue=((i*37)%101)/100,energy=((i*61+17)%103)/102,deadline=((i*43+29)%107)/106;
  const truthLogits=[
    1.9*queue+2.3*deadline-.9*(1-energy),
    1.15*queue+.65*deadline+1.8*(1-energy),
    1.75*(1-queue)+1.35*(1-deadline)+.35*(1-energy)
  ];
  const truth=softmax(truthLogits,1);
  const r=rng();let cumulative=0,label=truth.length-1;
  for(let a=0;a<truth.length;a++){cumulative+=truth[a];if(r<=cumulative){label=a;break}}
  // Same ranking signal, deliberately overconfident scale.
  return{logits:truthLogits.map(v=>v*2.8),label,truth};
}
const all=Array.from({length:6000},(_,i)=>caseAt(i)),fit=all.slice(0,4000),held=all.slice(4000);
const result=fitTemperature(fit,{bins:12}),before=calibrationMetrics(held.map(s=>({probs:logitsToProbabilities(s.logits,1),label:s.label})),{bins:12}),after=calibrationMetrics(held.map(s=>({probs:logitsToProbabilities(s.logits,result.temperature),label:s.label})),{bins:12});

assert.ok(result.temperature>2&&result.temperature<3.7,"temperature should recover the deliberately overconfident logit scale");
assert.equal(after.accuracy,before.accuracy,"temperature calibration must not change top-1 ranking");
assert.ok(after.nll<before.nll*.9,"held-out NLL should improve materially");
assert.ok(after.brier<before.brier*.94,"held-out Brier score should improve");
assert.ok(after.ece<before.ece*.55,"held-out ECE should improve materially");

const calibrator=new TemperatureCalibrator().fit(fit,{bins:12});
assert.ok(calibrator.after.ece<calibrator.before.ece);
const model=new TemperatureCalibrator(result.temperature),roundTrip=model.export(),restored=new TemperatureCalibrator().import(roundTrip);
assert.ok(Math.abs(restored.temperature-result.temperature)<1e-12);
console.log(JSON.stringify({ok:true,task:"stochastic-workload-routing",fitTemperature:result.temperature,before:{accuracy:before.accuracy,ece:before.ece,brier:before.brier,nll:before.nll,meanConfidence:before.meanConfidence},after:{accuracy:after.accuracy,ece:after.ece,brier:after.brier,nll:after.nll,meanConfidence:after.meanConfidence}}));
