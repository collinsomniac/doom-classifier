import assert from "node:assert/strict";
import {NeuralSetResidualQ} from "../src/core/neural-set-residual.js";

const schema={
  objective:"Choose actions that maximize observed reward.",
  actionFields:[{id:"effort",label:"effort",description:"normalized action effort",min:0,max:1}],
  fields:[
    {id:"signal",label:"signal",description:"current normalized task signal",min:0,max:1},
    {id:"budget",label:"budget",description:"remaining normalized budget",min:0,max:1}
  ],
  collections:[]
};
const actions=[
  {id:"good",label:"good action",description:"take the high-return option",params:{effort:.5}},
  {id:"neutral",label:"neutral action",description:"take the medium-return option",params:{effort:.5}},
  {id:"bad",label:"bad action",description:"take the low-return option",params:{effort:.5}}
];
const obs={signal:.7,budget:.8,_collections:{}};
const rewards=[1,.15,-.75];

const model=new NeuralSetResidualQ(schema,actions,{seed:91,useTargetNetwork:false,duelingValue:true,lr:.012,bootstrapProbability:1});
assert.match(model.name,/Dueling/);
assert.ok(model.parameterCount()>0);

let stats=model.scoreStatsObservation(obs);
assert.equal(stats.valueMemberScores.length,actions.length);
assert.equal(stats.stateValueMemberScores.length,model.ensembleSize);

// Dueling identity: the mean Q across actions equals V(s) for every ensemble member.
for(let member=0;member<model.ensembleSize;member++){
  const meanQ=stats.valueMemberScores.reduce((sum,row)=>sum+row[member],0)/actions.length;
  assert.ok(Math.abs(meanQ-stats.stateValueMemberScores[member])<1e-6,
    `member ${member}: mean Q ${meanQ} must equal state V ${stats.stateValueMemberScores[member]}`);
}

// Deterministic terminal bandit: common state value should not prevent the advantage
// stream from learning an ordering among otherwise identical typed actions.
for(let epoch=0;epoch<120;epoch++){
  for(let actionIndex=0;actionIndex<actions.length;actionIndex++){
    model.updateTransition({
      observation:obs,actionIndex,reward:rewards[actionIndex],
      nextObservation:obs,done:true
    });
  }
}
stats=model.scoreStatsObservation(obs);
assert.ok(stats.valueScores[0]>stats.valueScores[1],"good action should outrank neutral");
assert.ok(stats.valueScores[1]>stats.valueScores[2],"neutral action should outrank bad");
const gap=stats.valueScores[0]-stats.valueScores[1];
assert.ok(gap>.02,`learned advantage gap should be material, got ${gap}`);

for(let member=0;member<model.ensembleSize;member++){
  const meanQ=stats.valueMemberScores.reduce((sum,row)=>sum+row[member],0)/actions.length;
  assert.ok(Math.abs(meanQ-stats.stateValueMemberScores[member])<1e-5);
}

console.log(JSON.stringify({
  ok:true,model:model.name,params:model.parameterCount(),updates:model.updates,
  q:actions.map((a,i)=>({id:a.id,q:stats.valueScores[i]})),
  gap,stateValue:stats.stateValueMemberScores
}));
