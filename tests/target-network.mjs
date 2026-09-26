import assert from "node:assert/strict";
import {NeuralSetResidualQ} from "../src/core/neural-set-residual.js";

const schema={fields:[{id:"x",label:"signal",description:"generic signal",min:0,max:1}],collections:[]};
const actions=[{id:"a",label:"act",description:"perform action"},{id:"b",label:"hold",description:"hold"}];
const obs={x:.4,_collections:{}},next={x:.8,_collections:{}};
const net=new NeuralSetResidualQ(schema,actions,{seed:19,targetSyncInterval:3});
assert.ok(net.targetNet,"target network should exist for online learning");
assert.equal(net.targetNet.useTargetNetwork,false,"target network must not recursively allocate another target");

const live0=net.scoresObservation(obs),target0=net.targetNet.scoresObservation(obs);
assert.deepEqual(live0,target0,"target should begin synchronized");

const transition={observation:obs,temporal:null,actionIndex:0,reward:.7,nextObservation:next,nextTemporal:null,done:false};
const onlineValue=net.valueScoresObservation.bind(net),targetValue=net.targetNet.valueScoresObservation.bind(net);
net.valueScoresObservation=(observation,opts)=>observation===next?[1,0]:onlineValue(observation,opts);
net.targetNet.valueScoresObservation=(observation,opts)=>observation===next?[0,5]:targetValue(observation,opts);
const one=net.updateTransition(transition);
net.valueScoresObservation=onlineValue;net.targetNet.valueScoresObservation=targetValue;
assert.equal(one.doubleDqn,true);
assert.equal(one.bootstrapActionIndex,0,"online critic must select the bootstrap action");
assert.equal(one.bootstrapValue,0,"target critic must evaluate the online-selected action instead of taking its own maximum");
assert.ok(Math.abs(one.target-.7)<1e-9,"disagreed target-network maximum must not inflate the TD target");
const live1=net.scoresObservation(obs),target1=net.targetNet.scoresObservation(obs);
const drift1=Math.max(...live1.map((v,i)=>Math.abs(v-target1[i])));
assert.ok(drift1>1e-8,"online network should diverge between target syncs");
assert.equal(one.targetNetwork,true);
assert.equal(net.targetSyncs,1,"constructor sync only before interval elapses");

net.updateTransition(transition);
const beforeSyncTarget=[...net.targetNet.scoresObservation(obs)];
net.updateTransition(transition);
const afterSyncTarget=net.targetNet.scoresObservation(obs),live3=net.scoresObservation(obs);
const syncDiff=Math.max(...live3.map((v,i)=>Math.abs(v-afterSyncTarget[i])));
const targetMoved=Math.max(...beforeSyncTarget.map((v,i)=>Math.abs(v-afterSyncTarget[i])));
assert.ok(syncDiff<1e-7,"target must copy online parameters at configured interval");
assert.ok(targetMoved>1e-8,"target parameters should move when synchronized");
assert.equal(net.targetSyncs,2);

const params=net.parameterCount();
assert.ok(params<10000,"reported trainable parameter count should remain online-network only");
console.log(JSON.stringify({ok:true,params,drift1,targetMoved,syncDiff,targetSyncs:net.targetSyncs}));
