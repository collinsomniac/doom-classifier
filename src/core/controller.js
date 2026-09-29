import {argmax,mulberry32,percentile} from "./math.js";
import {probeCounterfactualActions} from "./counterfactual.js";

export const ControllerState=Object.freeze({READY:"READY",RUNNING:"RUNNING",PAUSED:"PAUSED",RESETTING:"RESETTING",TUNING:"TUNING",ERROR:"ERROR"});
export class ExperimentController extends EventTarget{
  constructor({environment,policy,hz=20}){
    super();this.environment=environment;this.policy=policy;this.hz=hz;this.state=ControllerState.READY;
    this.useResidual=true;this.training=false;this.memory=true;this.explore=false;this.timer=null;this.loopVersion=0;this.inFlight=false;
    this.trace=[];this.latencies=[];this.steps=0;this.episodes=0;this.episodeReturn=0;this.returns=[];this.lastDecision=null;
  }
  setState(next){this.state=next;this.dispatchEvent(new CustomEvent("state",{detail:next}))}
  setHz(hz){this.hz=hz;if(this.state===ControllerState.RUNNING){this.pause();this.start()}}
  start(){if(this.state===ControllerState.RUNNING)return;this.loopVersion++;const token=this.loopVersion;this.setState(ControllerState.RUNNING);this.schedule(token,0)}
  pause(){this.loopVersion++;if(this.timer)clearTimeout(this.timer);this.timer=null;if(this.state!==ControllerState.ERROR&&this.state!==ControllerState.TUNING)this.setState(ControllerState.PAUSED)}
  async quiesce({teacher=true}={}){
    this.pause();
    while(this.inFlight)await new Promise(resolve=>setTimeout(resolve,0));
    if(teacher&&this.policy.awaitTeacher)await this.policy.awaitTeacher();
    return true;
  }
  schedule(token,delay){if(this.state!==ControllerState.RUNNING||token!==this.loopVersion)return;this.timer=setTimeout(async()=>{const started=performance.now();await this.tick();const remaining=Math.max(0,1000/this.hz-(performance.now()-started));this.schedule(token,remaining)},delay)}
  async reset({learning=false}={}){
    await this.quiesce({teacher:true});this.setState(ControllerState.RESETTING);
    await this.environment.reset();this.policy.resetEpisode();if(learning)this.policy.resetLearning();
    this.steps=0;this.episodes=0;this.episodeReturn=0;this.returns=[];this.trace=[];this.latencies=[];this.lastDecision=null;this.setState(ControllerState.READY);this.dispatchEvent(new Event("tick"));
  }
  async tick(){
    if(this.inFlight)return false;this.inFlight=true;
    try{
      const obs=await this.environment.observe(),decision=await this.policy.decide(obs,{useResidual:this.useResidual,memory:this.memory,explore:this.training&&this.explore});
      const step=await this.environment.step(decision.action.id),outcome=step.info?.outcome||null;
      const nextHistory=this.policy.previewActionOutcome?.({actionIndex:decision.actionIndex,reward:step.reward,outcome,done:step.done})||null;
      const nextEncoded=this.policy.encode(step.observation,this.memory,false);
      let learningInfo=null;
      if(this.training)learningInfo=this.policy.learn({observation:obs,temporal:decision.temporal,history:decision.history,features:decision.features,actionIndex:decision.actionIndex,reward:step.reward,nextObservation:step.observation,nextTemporal:nextEncoded.temporal,nextHistory,nextFeatures:nextEncoded.features,done:step.done});
      this.policy.commitActionOutcome?.({actionIndex:decision.actionIndex,reward:step.reward,outcome,done:step.done});
      this.steps++;this.episodeReturn+=step.reward;this.latencies.push(decision.latencyMs);if(this.latencies.length>1000)this.latencies.shift();this.lastDecision={...decision,reward:step.reward,learningInfo,outcome:step.info?.outcome||null};
      this.trace.push({
        t:Date.now(),step:this.steps,observation:obs,action:decision.action.id,probabilities:Object.fromEntries(this.policy.actions.map((a,i)=>[a.id,decision.probs[i]])),reward:step.reward,outcome:step.info?.outcome||null,
        uncertainty:decision.uncertainty,valueBeta:decision.valueBeta??0,priorKL:decision.priorKL??0,valueTrust:decision.valueTrust??0,basePriorKlBudget:decision.basePriorKlBudget??0,priorKlBudget:decision.priorKlBudget??0,klUtilization:decision.klUtilization??0,valueEpistemic:decision.valueEpistemic??0,valueEpistemicBudget:decision.valueEpistemicBudget??0,epistemicUtilization:decision.epistemicUtilization??0,typedValueFit:decision.typedValueFit??0,typedValueBlendUsed:decision.typedValueBlendUsed??0,criticGap:decision.criticGap??0,criticGapShare:decision.criticGapShare??0,criticTopAgreement:decision.criticTopAgreement??0,criticMarginSnr:decision.criticMarginSnr??0,criticRankingConfidence:decision.criticRankingConfidence??0,criticAuthority:decision.criticAuthority??0,criticKlGate:decision.criticKlGate??1,criticKlMultiplier:decision.criticKlMultiplier??1,probabilityCalibrated:!!decision.probabilityCalibrated,probabilityCalibrationTemperature:Number(decision.probabilityCalibrationTemperature||1),probabilityTemperature:Number(decision.probabilityTemperature||0),valueBetaSaturated:!!decision.valueBetaSaturated,semanticPrior:decision.semanticPriorScores?.[decision.actionIndex]??null,learnedValue:decision.valueScores?.[decision.actionIndex]??null,combinedScore:decision.qScores?.[decision.actionIndex]??null,latencyMs:decision.latencyMs,semanticLatencyMs:decision.semanticLatencyMs,residualLatencyMs:decision.residualLatencyMs,
        teacherUsed:decision.teacherUsed,teacherPending:decision.teacherPending,semanticUsed:decision.semanticUsed,inferenceMode:decision.inferenceMode,explorationStrategy:decision.explorationStrategy,decisionRule:decision.decisionRule||"argmax",uncertaintySampleChance:Number(decision.uncertaintySampleChance||0),teacherCalls:this.policy.teacherCalls,teacherScheduled:this.policy.teacherScheduled,lastTeacherLatencyMs:this.policy.lastTeacherLatencyMs,
        learning:learningInfo?{td:Number(learningInfo.td||0),target:Number(learningInfo.target||0),q:Number(learningInfo.q||0),combined:Number(learningInfo.combined||0),bootstrapActionIndex:Number(learningInfo.bootstrapActionIndex??-1),bootstrapValue:Number(learningInfo.bootstrapValue||0),doubleDqn:!!learningInfo.doubleDqn,replayUpdates:Number(learningInfo.replayUpdates||0),replayMeanAbsTd:Number(learningInfo.replayMeanAbsTd||0),replaySize:Number(learningInfo.replaySize||0),nStepHorizon:Number(learningInfo.nStepHorizon||0),primaryUpdates:Number(learningInfo.primaryUpdates||0)}:null,
        backbone:this.policy.semantic.name,residual:this.policy.q.name||this.policy.q.constructor.name,backend:this.policy.semantic.backend||"local-js",
        mode:{useResidual:this.useResidual,training:this.training,memory:this.memory,explore:this.explore}
      });
      if(this.trace.length>5000)this.trace.shift();
      if(step.done){this.episodes++;this.returns.push(this.episodeReturn);if(this.returns.length>200)this.returns.shift();this.episodeReturn=0;await this.environment.reset();this.policy.resetEpisode()}
      this.dispatchEvent(new Event("tick"));return true;
    }catch(err){console.error(err);this.loopVersion++;if(this.timer)clearTimeout(this.timer);this.timer=null;this.setState(ControllerState.ERROR);this.dispatchEvent(new CustomEvent("error",{detail:err}));return false}
    finally{this.inFlight=false}
  }
  async trainBurst({steps=128,epsilon=.16,rolloutHorizon=0,onProgress=()=>{}}={}){
    const previous={training:this.training,explore:this.explore,epsilon:this.policy.epsilon};
    await this.quiesce({teacher:true});
    this.training=true;this.explore=true;this.policy.epsilon=epsilon;this.setState(ControllerState.TUNING);
    const startUpdates=this.policy.q.updates,startEpisodes=this.episodes,startStep=this.steps,startTrace=this.trace.length;let rolloutRestarts=0;
    try{
      for(let i=0;i<steps;i++){
        const ok=await this.tick();if(!ok&&this.state===ControllerState.ERROR)break;
        if(rolloutHorizon>0&&i+1<steps&&(i+1)%rolloutHorizon===0){
          this.policy.flushLearning?.();await this.policy.awaitTeacher?.();
          await this.environment.reset();this.policy.resetEpisode();this.episodeReturn=0;rolloutRestarts++;
        }
        if(i===0||(i+1)%4===0||i+1===steps)onProgress({completed:i+1,total:steps,ratio:(i+1)/steps,steps:this.steps,episodes:this.episodes,updates:this.policy.q.updates,rolloutRestarts});
        await Promise.resolve();
      }
      const creditFlush=this.policy.flushLearning?.()||null;
      await this.policy.awaitTeacher?.();
      const segment=this.trace.slice(startTrace),counts={};let switches=0,maxStreak=0,last=null,streak=0;
      for(const item of segment){
        counts[item.action]=(counts[item.action]||0)+1;
        if(item.action===last)streak++;else{if(last!==null)switches++;streak=1;last=item.action}
        maxStreak=Math.max(maxStreak,streak);
      }
      const trainingReturn=segment.reduce((sum,item)=>sum+Number(item.reward||0),0);
      return{requested:steps,completed:this.steps-startStep,updates:this.policy.q.updates-startUpdates,episodes:this.episodes-startEpisodes,rolloutHorizon,rolloutRestarts,return:trainingReturn,actionDiversity:Object.keys(counts).length,switches,maxStreak,actionCounts:counts,creditFlush};
    }finally{
      this.training=previous.training;this.explore=previous.explore;this.policy.epsilon=previous.epsilon;
      if(this.state!==ControllerState.ERROR)this.setState(ControllerState.PAUSED);
      this.dispatchEvent(new Event("tick"));
    }
  }
  async trainCausalPolicy({
    steps=512,probeTics=24,actionTics=4,rolloutHorizon=128,
    targetTemperature=.30,priorStrength=.08,superviseSteps=3,superviseStrength=.55,
    supervisionReplay=2,replayStrength=.24,batchRefitEvery=0,batchWindow=192,
    finalRefit=false,ridge=.025,seed=0x51a9e,onProgress=()=>{}
  }={}){
    if(!this.environment?.supportsSnapshots?.()||!this.environment?.supportsExactTics?.())throw new Error("Causal policy training requires exact snapshots and tic stepping");
    if(!this.policy?.superviseDecisionDistribution||!this.policy?.fitDecisionDistributions)throw new Error("Policy does not support proper-score supervision");
    const previous={training:this.training,explore:this.explore,mode:this.policy.inferenceMode,memory:this.memory};
    await this.quiesce({teacher:true});
    this.training=false;this.explore=false;this.memory=true;this.policy.setInferenceMode("neural");this.setState(ControllerState.TUNING);
    const examples=[],curriculumTrace=[],counts={},startStep=this.steps,replayRng=mulberry32(Number(seed)>>>0);
    let totalReward=0,totalDamage=0,totalKills=0,totalPickups=0,totalSpread=0,informative=0,resets=0,fitPasses=0,supervisionUpdates=0,replaySupervisionUpdates=0;
    try{
      for(let i=0;i<Math.max(1,Math.floor(steps));i++){
        const obs=await this.environment.observe();
        const decision=await this.policy.decide(obs,{useResidual:true,memory:true,explore:false});
        const probe=await probeCounterfactualActions({
          environment:this.environment,actions:this.policy.actions,prior:decision.probs,horizon:1,
          temperature:targetTemperature,priorStrength,
          stepper:id=>this.environment.stepTics(id,probeTics)
        });
        const target=[...probe.target],oracleIndex=argmax(target),oracle=this.policy.actions[oracleIndex];
        const example={
          observation:globalThis.structuredClone?globalThis.structuredClone(obs):JSON.parse(JSON.stringify(obs)),
          target,temporal:decision.temporal?new Float32Array(decision.temporal):null,
          history:decision.history?JSON.parse(JSON.stringify(decision.history)):null
        };
        examples.push(example);
        const supervised=this.policy.superviseDecisionDistribution(obs,target,{
          steps:superviseSteps,strength:superviseStrength,temporal:decision.temporal,history:decision.history
        });
        supervisionUpdates+=Number(supervised?.stepsUsed||0);

        // Proper-score replay prevents the sequential curriculum from collapsing
        // onto the globally frequent action at the expense of earlier state distinctions.
        const replayCount=Math.min(Math.max(0,Math.floor(supervisionReplay)),Math.max(0,examples.length-1));
        for(let ri=0;ri<replayCount;ri++){
          const ex=examples[Math.floor(replayRng()*(examples.length-1))];
          const replayed=this.policy.superviseDecisionDistribution(ex.observation,ex.target,{
            steps:1,strength:replayStrength,temporal:ex.temporal,history:ex.history
          });
          replaySupervisionUpdates+=Number(replayed?.stepsUsed||0);
        }
        if(probe.spread>1e-9){informative++;totalSpread+=probe.spread}

        const step=this.environment.stepTics(oracle.id,actionTics),outcome=step.info?.outcome||null;
        this.policy.commitActionOutcome?.({actionIndex:oracleIndex,reward:step.reward,outcome,done:step.done});
        this.steps++;this.episodeReturn+=Number(step.reward||0);totalReward+=Number(step.reward||0);
        totalDamage+=Number(outcome?.damageDealt||0);totalKills+=Number(outcome?.playerKillDelta||0);totalPickups+=Number(outcome?.playerPickupDelta||0);
        counts[oracle.id]=(counts[oracle.id]||0)+1;
        const ranked=probe.trials.map(t=>({id:t.id,return:Number(t.return||0)})).sort((a,b)=>b.return-a.return);
        curriculumTrace.push({
          step:i+1,action:oracle.id,reward:Number(step.reward||0),probeSpread:Number(probe.spread||0),
          measuredTop:ranked[0]?.id||oracle.id,measuredTopReturn:Number(ranked[0]?.return||0),
          targetTop:oracle.id,targetTopProbability:Number(target[oracleIndex]||0),
          outcome:outcome?{
            damageDealt:Number(outcome.damageDealt||0),playerKillDelta:Number(outcome.playerKillDelta||0),
            playerPickupDelta:Number(outcome.playerPickupDelta||0),healthDelta:Number(outcome.healthDelta||0),
            levelCompletionDelta:Number(outcome.levelCompletionDelta||0)
          }:null
        });

        if(batchRefitEvery>0&&(i+1)%batchRefitEvery===0){
          const batch=examples.slice(-Math.max(batchRefitEvery,Math.floor(batchWindow||batchRefitEvery)));
          this.policy.fitDecisionDistributions(batch,{ridge,refineSteps:0});fitPasses++;
        }
        const shouldReset=step.done||(rolloutHorizon>0&&i+1<steps&&(i+1)%rolloutHorizon===0);
        if(shouldReset){
          await this.environment.reset();this.policy.resetEpisode();this.episodeReturn=0;resets++;
        }
        if(i===0||(i+1)%8===0||i+1===steps)onProgress({
          completed:i+1,total:steps,ratio:(i+1)/steps,kills:totalKills,damage:totalDamage,
          examples:examples.length,informative,resets,replaySupervisionUpdates
        });
        await Promise.resolve();
      }
      if(finalRefit&&examples.length){
        this.policy.fitDecisionDistributions(examples.slice(-Math.max(64,Math.floor(batchWindow||examples.length))),{
          ridge,refineSteps:0
        });fitPasses++;
      }
      this.policy.q.syncTarget?.({value:false});
      return{
        requested:steps,completed:this.steps-startStep,probeTics,actionTics,rolloutHorizon,
        examples:examples.length,informative,meanInformativeSpread:informative?totalSpread/informative:0,
        supervisionUpdates,replaySupervisionUpdates,fitPasses,finalRefit:!!finalRefit,resets,
        return:totalReward,damage:totalDamage,kills:totalKills,pickups:totalPickups,
        actionDiversity:Object.keys(counts).length,actionCounts:counts,trace:curriculumTrace
      };
    }finally{
      this.training=previous.training;this.explore=previous.explore;this.memory=previous.memory;this.policy.setInferenceMode(previous.mode);
      if(this.state!==ControllerState.ERROR)this.setState(ControllerState.PAUSED);
      this.dispatchEvent(new Event("tick"));
    }
  }
  latencySummary(){return{last:this.latencies.at(-1)||0,p50:percentile(this.latencies,.5),p95:percentile(this.latencies,.95),p99:percentile(this.latencies,.99)}}
  exportTrace(){return JSON.stringify({meta:{createdAt:new Date().toISOString(),steps:this.steps,episodes:this.episodes,hz:this.hz,backbone:this.policy.semantic.name,residual:this.policy.q.name||this.policy.q.constructor.name,inferenceMode:this.policy.inferenceMode,teacherCalls:this.policy.teacherCalls,teacherScheduled:this.policy.teacherScheduled,backend:this.policy.semantic.backend||"local-js"},teacherHistory:this.policy.teacherHistory||[],trace:this.trace},null,2)}
}
