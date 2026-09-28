import {probeCounterfactualActions} from "./counterfactual.js";
import {softmax} from "./math.js";

export function semanticPriorAtState(policy,observation){
  if(!policy?.q?.scoreStatsObservation)throw new Error("Measured-fork training requires semantic/value score statistics");
  const stats=policy.q.scoreStatsObservation(observation,{temporal:null});
  return softmax(stats.semanticScores,Math.max(.05,Number(policy.temperature)||1));
}

export function supportsMeasuredForks(environment){
  return !!environment&&typeof environment.saveSnapshot==="function"&&typeof environment.restoreSnapshot==="function"&&
    (typeof environment.stepTics==="function"||typeof environment.counterfactualStep==="function");
}

export async function measureCounterfactualValues({
  environment,policy,actions=policy?.actions,tics=24,fallbackTics=null,
  temperature=.45,priorStrength=1,fit=false,fitSteps=6,fitStrength=.08,
  stepper=null
}={}){
  if(!environment||!policy||!Array.isArray(actions)||!actions.length)throw new Error("Measured fork requires environment, policy, and actions");
  const observation=environment.lastObservation||environment.observe?.(),prior=semanticPriorAtState(policy,observation),started=performance.now();
  const doStep=stepper||(typeof environment.counterfactualStep==="function"
    ?((id,horizon)=>environment.counterfactualStep(id,{horizon}))
    :(typeof environment.stepTics==="function"?((id,horizon)=>environment.stepTics(id,horizon)):null));
  if(!doStep)throw new Error("Environment does not expose deterministic counterfactual stepping");
  const run=async horizon=>probeCounterfactualActions({
    environment,actions,prior,horizon:1,temperature,priorStrength,
    stepper:id=>doStep(id,horizon)
  });
  let usedTics=Math.max(1,Number(tics)||24),probe=await run(usedTics);
  if(Number(probe.spread||0)<=1e-9&&fallbackTics&&Number(fallbackTics)>usedTics){
    usedTics=Number(fallbackTics);probe=await run(usedTics);
  }
  const measured={...probe,prior,tics:usedTics,ms:performance.now()-started,t:Date.now()};
  const example=Number(measured.spread||0)>1e-9
    ?{observation:measured.observation,target:measured.target,temporal:null,meta:{tics:usedTics,spread:measured.spread}}
    :null;
  const fitResult=fit&&example?policy.fitCounterfactualValueDistributions([example],{steps:fitSteps,strength:fitStrength}):null;
  return{probe:measured,example,fitResult};
}

export async function trainWithMeasuredForks({
  controller,policy,environment,steps=128,stageSize=32,epsilon=.16,
  tics=24,fallbackTics=35,fitSteps=6,fitStrength=.08,resetEvery=64,
  bootstrapProbe=true,onProgress=()=>{},onProbe=()=>{}
}={}){
  if(!controller||!policy||!environment)throw new Error("Measured-fork training requires controller, policy, and environment");
  const total=Math.max(1,Math.floor(Number(steps)||128)),chunk=Math.max(1,Math.floor(Number(stageSize)||32));
  const canProbe=supportsMeasuredForks(environment),counts={};let completed=0,reward=0,measuredFits=0,measuredProbes=0,spreadSum=0,rolloutRestarts=0,lastProbe=null;

  const probe=async phase=>{
    if(!canProbe)return null;
    const measured=await measureCounterfactualValues({environment,policy,tics,fallbackTics,fit:true,fitSteps,fitStrength});
    measuredProbes++;lastProbe=measured.probe;
    if(measured.example){measuredFits++;spreadSum+=Number(measured.probe.spread||0)}
    await onProbe({...measured,phase,completed,total,measuredFits,measuredProbes});
    return measured;
  };

  if(bootstrapProbe)await probe("bootstrap");

  while(completed<total){
    const stageSteps=Math.min(chunk,total-completed),base=completed;
    const result=await controller.trainBurst({steps:stageSteps,epsilon,rolloutHorizon:0,onProgress:p=>onProgress({
      phase:"td",completed:base+p.completed,total,ratio:(base+p.completed)/total,updates:policy.q.updates,
      measuredFits,measuredProbes,rolloutRestarts
    })});
    completed+=result.completed;reward+=Number(result.return||0);
    for(const [id,n] of Object.entries(result.actionCounts||{}))counts[id]=(counts[id]||0)+Number(n||0);
    await probe("post-stage");
    if(completed<total&&resetEvery>0&&completed%resetEvery===0){
      await controller.reset({learning:false});policy.resetEpisode();rolloutRestarts++;
    }
    onProgress({phase:"stage-complete",completed,total,ratio:completed/total,updates:policy.q.updates,measuredFits,measuredProbes,rolloutRestarts});
    await Promise.resolve();
  }

  const creditFlush=policy.flushLearning?.()||null;await policy.awaitTeacher?.();
  return{
    requested:total,completed,reward,return:reward,updates:policy.q.updates,actionDiversity:Object.keys(counts).length,actionCounts:counts,
    measuredFits,measuredProbes,meanMeasuredSpread:measuredFits?spreadSum/measuredFits:0,rolloutHorizon:resetEvery,rolloutRestarts,creditFlush,lastProbe
  };
}
