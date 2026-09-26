import {percentile} from "./math.js";

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
  schedule(token,delay){if(this.state!==ControllerState.RUNNING||token!==this.loopVersion)return;this.timer=setTimeout(async()=>{const started=performance.now();await this.tick();const remaining=Math.max(0,1000/this.hz-(performance.now()-started));this.schedule(token,remaining)},delay)}
  async reset({learning=false}={}){
    this.loopVersion++;if(this.timer)clearTimeout(this.timer);this.timer=null;this.setState(ControllerState.RESETTING);
    await this.environment.reset();this.policy.resetEpisode();if(learning)this.policy.resetLearning();
    this.steps=0;this.episodes=0;this.episodeReturn=0;this.returns=[];this.trace=[];this.latencies=[];this.lastDecision=null;this.setState(ControllerState.READY);this.dispatchEvent(new Event("tick"));
  }
  async tick(){
    if(this.inFlight)return false;this.inFlight=true;
    try{
      const obs=await this.environment.observe(),decision=await this.policy.decide(obs,{useResidual:this.useResidual,memory:this.memory,explore:this.training&&this.explore});
      const step=await this.environment.step(decision.action.id),nextEncoded=this.policy.encode(step.observation,this.memory,false);
      let learningInfo=null;
      if(this.training)learningInfo=this.policy.learn({observation:obs,temporal:decision.temporal,features:decision.features,actionIndex:decision.actionIndex,reward:step.reward,nextObservation:step.observation,nextTemporal:nextEncoded.temporal,nextFeatures:nextEncoded.features,done:step.done});
      this.steps++;this.episodeReturn+=step.reward;this.latencies.push(decision.latencyMs);if(this.latencies.length>1000)this.latencies.shift();this.lastDecision={...decision,reward:step.reward,learningInfo,outcome:step.info?.outcome||null};
      this.trace.push({
        t:Date.now(),step:this.steps,observation:obs,action:decision.action.id,probabilities:Object.fromEntries(this.policy.actions.map((a,i)=>[a.id,decision.probs[i]])),reward:step.reward,outcome:step.info?.outcome||null,
        uncertainty:decision.uncertainty,valueBeta:decision.valueBeta??0,priorKL:decision.priorKL??0,valueTrust:decision.valueTrust??0,priorKlBudget:decision.priorKlBudget??0,klUtilization:decision.klUtilization??0,valueEpistemic:decision.valueEpistemic??0,valueEpistemicBudget:decision.valueEpistemicBudget??0,epistemicUtilization:decision.epistemicUtilization??0,typedValueFit:decision.typedValueFit??0,typedValueBlendUsed:decision.typedValueBlendUsed??0,criticGap:decision.criticGap??0,criticTopAgreement:decision.criticTopAgreement??0,criticMarginSnr:decision.criticMarginSnr??0,valueBetaSaturated:!!decision.valueBetaSaturated,semanticPrior:decision.semanticPriorScores?.[decision.actionIndex]??null,learnedValue:decision.valueScores?.[decision.actionIndex]??null,combinedScore:decision.qScores?.[decision.actionIndex]??null,latencyMs:decision.latencyMs,semanticLatencyMs:decision.semanticLatencyMs,residualLatencyMs:decision.residualLatencyMs,
        teacherUsed:decision.teacherUsed,teacherPending:decision.teacherPending,semanticUsed:decision.semanticUsed,inferenceMode:decision.inferenceMode,explorationStrategy:decision.explorationStrategy,teacherCalls:this.policy.teacherCalls,teacherScheduled:this.policy.teacherScheduled,lastTeacherLatencyMs:this.policy.lastTeacherLatencyMs,
        learning:learningInfo?{td:Number(learningInfo.td||0),target:Number(learningInfo.target||0),q:Number(learningInfo.q||0),combined:Number(learningInfo.combined||0),replayUpdates:Number(learningInfo.replayUpdates||0),replayMeanAbsTd:Number(learningInfo.replayMeanAbsTd||0),replaySize:Number(learningInfo.replaySize||0),nStepHorizon:Number(learningInfo.nStepHorizon||0),primaryUpdates:Number(learningInfo.primaryUpdates||0)}:null,
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
    this.loopVersion++;if(this.timer)clearTimeout(this.timer);this.timer=null;
    const previous={training:this.training,explore:this.explore,epsilon:this.policy.epsilon};
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
  latencySummary(){return{last:this.latencies.at(-1)||0,p50:percentile(this.latencies,.5),p95:percentile(this.latencies,.95),p99:percentile(this.latencies,.99)}}
  exportTrace(){return JSON.stringify({meta:{createdAt:new Date().toISOString(),steps:this.steps,episodes:this.episodes,hz:this.hz,backbone:this.policy.semantic.name,residual:this.policy.q.name||this.policy.q.constructor.name,inferenceMode:this.policy.inferenceMode,teacherCalls:this.policy.teacherCalls,teacherScheduled:this.policy.teacherScheduled,backend:this.policy.semantic.backend||"local-js"},teacherHistory:this.policy.teacherHistory||[],trace:this.trace},null,2)}
}
