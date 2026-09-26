import assert from "node:assert/strict";
import {SemanticResidualPolicy} from "../src/core/policy.js";

const schema={fields:[{id:"x",label:"x",min:0,max:1}],collections:[]};
const actions=[{id:"prior",label:"prior"},{id:"critic",label:"critic"}];
const semantic={name:"stub",backend:"test",compile(){},async score(){return[0,0]}};
const q={updates:100,reset(){},parameterCount(){return 0}};
const p=new SemanticResidualPolicy({
  schema,actions,semantic,residual:q,inferenceMode:"neural",
  replayEvidenceBandwidth:.08,replayEvidenceTarget:3
});
const sig=new Float32Array([0,0,0,0]);

function sample(actionIndex,reward,offset){
  return{actionIndex,reward,stateSignature:new Float32Array([offset,0,0,0])};
}

p.replay=[
  sample(1,.8,.01),sample(1,.7,-.02),sample(1,.9,.03),
  sample(0,.10,.01),sample(0,.15,-.02),sample(0,.05,.025)
];
const supported=p.replayOverrideEvidence(sig,1,0);
assert.ok(supported.critic.localSupport>2.9);
assert.ok(supported.prior.localSupport>2.9);
assert.ok(supported.returnDelta>.6);
assert.ok(supported.confidence>.9,"well-supported empirical critic advantage should score high");

p.replay=[
  sample(1,-.25,.01),sample(1,-.20,-.02),sample(1,-.30,.03),
  sample(0,.20,.01),sample(0,.25,-.02),sample(0,.15,.025)
];
const contradicted=p.replayOverrideEvidence(sig,1,0);
assert.ok(contradicted.returnDelta<-.35);
assert.ok(contradicted.confidence<.1,"nearby replay contradicting the critic override should score low");

p.replay=[
  sample(1,.8,1),sample(1,.7,.95),sample(1,.9,1.05),
  sample(0,.1,1),sample(0,.15,.95),sample(0,.05,1.05)
];
const distant=p.replayOverrideEvidence(sig,1,0);
assert.ok(distant.critic.localSupport<.2);
assert.ok(distant.supportGate<.1,"distant transitions must not masquerade as local support");
assert.ok(distant.confidence<.1);

const agrees=p.replayOverrideEvidence(sig,0,0);
assert.equal(agrees.confidence,1);
assert.equal(agrees.agrees,true);

console.log(JSON.stringify({
  ok:true,
  supported:{confidence:supported.confidence,delta:supported.returnDelta,support:supported.critic.localSupport},
  contradicted:{confidence:contradicted.confidence,delta:contradicted.returnDelta},
  distant:{confidence:distant.confidence,support:distant.critic.localSupport}
}));
