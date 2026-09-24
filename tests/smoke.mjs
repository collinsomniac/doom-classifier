import assert from "node:assert/strict";
import {MockArena} from "../src/env/mock-arena.js";
import {HashSemanticAdapter} from "../src/core/semantic.js";
import {SemanticResidualPolicy} from "../src/core/policy.js";

const env=new MockArena(7);
const policy=new SemanticResidualPolicy({schema:env.schema,actions:env.actions,semantic:new HashSemanticAdapter(),seed:9});
let episodes=0,total=0;
for(let i=0;i<3000;i++){
  const obs=env.observe();
  const decision=policy.decide(obs,{useResidual:true,memory:true,explore:true});
  assert.equal(decision.probs.length,env.actions.length);
  assert.ok(decision.probs.every(Number.isFinite));
  assert.ok(Math.abs(decision.probs.reduce((a,b)=>a+b,0)-1)<1e-5);
  const step=env.step(decision.action.id);
  const next=policy.encode(step.observation,true,false);
  policy.learn({features:decision.features,actionIndex:decision.actionIndex,reward:step.reward,nextFeatures:next.features,done:step.done});
  total+=step.reward;
  if(step.done){episodes++;env.reset();policy.resetEpisode()}
}
assert.equal(policy.q.updates,3000);
assert.ok(policy.q.weights.some(w=>Array.from(w).some(v=>Math.abs(v)>1e-8)));
assert.ok(Number.isFinite(total));
console.log(JSON.stringify({ok:true,steps:3000,episodes,totalReward:total,updates:policy.q.updates}));
