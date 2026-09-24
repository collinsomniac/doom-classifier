import {MockArena} from "../src/env/mock-arena.js";
import {HashSemanticAdapter} from "../src/core/semantic.js";
import {SemanticResidualPolicy} from "../src/core/policy.js";

async function phase({seed,steps,useResidual,train,explore,policy=null}){
  const env=new MockArena(seed);
  policy??=new SemanticResidualPolicy({schema:env.schema,actions:env.actions,semantic:new HashSemanticAdapter(),seed});
  policy.resetEpisode();
  let total=0,episodes=0,episodeReturn=0,completed=[];
  for(let i=0;i<steps;i++){
    const obs=env.observe(),d=await policy.decide(obs,{useResidual,memory:true,explore});
    const s=env.step(d.action.id),next=policy.encode(s.observation,true,false);
    if(train)policy.learn({features:d.features,actionIndex:d.actionIndex,reward:s.reward,nextFeatures:next.features,done:s.done});
    total+=s.reward;episodeReturn+=s.reward;
    if(s.done){episodes++;completed.push(episodeReturn);episodeReturn=0;env.reset();policy.resetEpisode()}
  }
  return{policy,total,episodes,meanStepReward:total/steps,meanEpisodeReturn:completed.length?completed.reduce((a,b)=>a+b,0)/completed.length:null};
}

const seed=42;
const baseline=await phase({seed,steps:6000,useResidual:false,train:false,explore:false});
const learner=new SemanticResidualPolicy({schema:new MockArena(seed).schema,actions:new MockArena(seed).actions,semantic:new HashSemanticAdapter(),seed});
const training=await phase({seed,steps:12000,useResidual:true,train:true,explore:true,policy:learner});
const frozen=await phase({seed:seed+100,steps:6000,useResidual:true,train:false,explore:false,policy:learner});
console.log(JSON.stringify({
  baseline:{meanStepReward:baseline.meanStepReward,meanEpisodeReturn:baseline.meanEpisodeReturn,episodes:baseline.episodes},
  training:{meanStepReward:training.meanStepReward,meanEpisodeReturn:training.meanEpisodeReturn,episodes:training.episodes,updates:learner.q.updates},
  frozen:{meanStepReward:frozen.meanStepReward,meanEpisodeReturn:frozen.meanEpisodeReturn,episodes:frozen.episodes}
},null,2));
