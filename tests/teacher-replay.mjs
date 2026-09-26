import assert from "node:assert/strict";
import {SemanticResidualPolicy} from "../src/core/policy.js";

const schema={fields:[{id:"x",label:"x",min:0,max:1}],collections:[]};
const actions=[{id:"a",label:"a"},{id:"b",label:"b"}];
const semantic={name:"teacher",backend:"test",compile(){},async score(){return[0,0]}};
const q={
  calls:[],syncs:0,
  distill(obs,scores,{strength,temporal}={}){this.calls.push({tag:obs.tag,strength,temporal:temporal?[...temporal]:null});return{loss:Number(obs.tag||0),student:[.5,.5]}},
  syncTarget(){this.syncs++;return true},
  reset(){},parameterCount(){return 0}
};
const p=new SemanticResidualPolicy({
  schema,actions,semantic,residual:q,seed:7,
  teacherReplayCapacity:2,teacherReplayBatch:2,teacherReplayStrength:.2,teacherAnchorCapacity:1
});

let r=p.applyTeacherScores({tag:1,x:.1,_collections:{}},[1,0],1,new Float32Array([.1]),{anchor:true});
assert.equal(r.teacherReplayUpdates,0);
assert.equal(r.teacherReplaySize,0);
assert.equal(r.teacherAnchorSize,1);
assert.equal(q.calls.length,1);
assert.equal(q.calls[0].strength,.5);
assert.equal(p.teacherAnchors[0].observation.tag,1);

r=p.applyTeacherScores({tag:2,x:.2,_collections:{}},[0,1],1,new Float32Array([.2]));
assert.equal(r.teacherReplayUpdates,1);
assert.equal(r.teacherAnchorUpdates,1);
assert.equal(r.teacherReplaySize,1);
assert.equal(r.teacherAnchorSize,1);
assert.deepEqual(q.calls.slice(1).map(x=>[x.tag,x.strength]),[[1,.2],[2,.5]],"pinned bootstrap anchor must rehearse before the newest fit");

r=p.applyTeacherScores({tag:3,x:.3,_collections:{}},[1,0],1,new Float32Array([.3]));
assert.equal(r.teacherReplayUpdates,2);
assert.equal(r.teacherAnchorUpdates,1);
assert.equal(r.teacherReplaySize,2,"adaptive teacher reservoir must remain bounded");
assert.equal(p.teacherAnchors[0].observation.tag,1,"bootstrap anchor must not be evicted by adaptive observations");

for(let tag=4;tag<=8;tag++)p.applyTeacherScores({tag,x:tag/10,_collections:{}},tag%2?[1,0]:[0,1],1,new Float32Array([tag/10]));
assert.equal(p.teacherReplay.length,2);
assert.equal(p.teacherAnchors.length,1);
assert.equal(p.teacherAnchors[0].observation.tag,1);
assert.equal(p.teacherReplaySeen,7,"reservoir must count all non-anchor teacher examples");
assert.ok(p.teacherReplay.every(sample=>sample.observation.tag!==1),"pinned anchor is stored separately from reservoir");

p.setSemantic({name:"teacher2",backend:"test",compile(){},async score(){return[0,0]}});
assert.equal(p.teacherReplay.length,0,"switching teachers must invalidate semantic replay labels");
assert.equal(p.teacherAnchors.length,0,"switching teachers must invalidate pinned semantic anchors");
assert.equal(p.teacherReplaySeen,0);

console.log(JSON.stringify({ok:true,calls:q.calls.length,syncs:q.syncs,replayCapacity:p.teacherReplayCapacity,anchorCapacity:p.teacherAnchorCapacity}));
