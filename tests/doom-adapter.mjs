import assert from "node:assert/strict";
import {DoomWasmArena} from "../src/env/doom-wasm.js";

const arena=new DoomWasmArena({},{actionMs:110});
const raw={
  ready:true,
  player:{health:87,armor:40,weapon:1,ammo:{bullets:33,shells:8,rockets:2,cells:12},recent_damage:6,under_fire:true,kills:3,x:100,y:200,z:8,vx:2,vy:-1,angle:1073741824},
  world:{
    entities:[
      {id:7,type:1,x:200,y:200,z:8,vx:1,vy:0,radius:20,height:56,health:60,distance:100,relative_x:100,relative_y:0,relative_angle:1073741824,visible:true,enemy:true,pickup:false,targeting_player:false},
      {id:8,type:2,x:120,y:180,z:8,vx:0,vy:0,radius:20,height:56,health:30,distance:28,relative_x:20,relative_y:-20,relative_angle:-536870912,visible:true,enemy:true,pickup:false,targeting_player:true},
      {id:9,type:3,x:110,y:200,z:8,vx:0,vy:0,radius:8,height:16,health:0,distance:10,relative_x:10,relative_y:0,relative_angle:0,visible:true,enemy:false,pickup:true,targeting_player:false},
      {id:10,type:31,x:150,y:200,z:8,vx:-6,vy:0,radius:6,height:8,health:1000,distance:50,relative_x:50,relative_y:0,relative_angle:0,visible:true,enemy:false,pickup:false,targeting_player:false}
    ],
    lines:[{id:4,x1:80,y1:180,x2:160,y2:180,flags:1,blocking:true,special:0,tag:0}]
  }
};
arena.visitedCells=new Map();const firstExplore=arena.commitExploration(raw);const repeatExplore=arena.commitExploration(raw);\nconst moved=structuredClone(raw);moved.player.x=300;const novelExplore=arena.commitExploration(moved);\nconst flat=arena.flatten(raw);
assert.equal(flat.health,87);assert.equal(flat.weapon,1);assert.equal(flat.under_fire,1);assert.equal(flat.player_z,8);assert.equal(flat._collections.entities.length,4);assert.equal(flat._collections.geometry.length,1);
assert.equal(flat._collections.entities[1].engine_record_id,8);assert.equal(flat._collections.entities[1].targeting_player,1);assert.equal(flat._collections.entities[2].pickup,1);
assert.equal(flat._collections.entities[1].kind,1);assert.equal(flat._collections.entities[2].kind,3);assert.equal(flat._collections.entities[3].kind,2);
assert.equal(flat._collections.geometry[0].line_id,4);assert.equal(flat._collections.geometry[0].flags,1);assert.equal(flat._collections.geometry[0].x1,-20);

const damaged=structuredClone(raw);damaged.world.entities[1].health=15;
const damageOutcome=arena.outcome(raw,damaged,{exploration:{newCell:false,visitedCells:1}});\nconst noveltyOutcome=arena.outcome(raw,raw,{exploration:{newCell:true,visitedCells:2}});
assert.equal(damageOutcome.damageDealt,15);assert.ok(damageOutcome.reward>0,"nonlethal hostile damage must create immediate positive learning signal");\nassert.equal(firstExplore.newCell,true);assert.equal(repeatExplore.newCell,false);assert.equal(novelExplore.newCell,true);assert.equal(novelExplore.visitedCells,2);\nassert.ok(noveltyOutcome.explorationBonus>0,"new spatial cells must provide a small policy-blind progress bonus");

const rewardGood=arena.reward({player:{health:87,kills:3},world:{entities:[]}},{player:{health:87,kills:4},world:{entities:[]}});
const rewardBad=arena.reward({player:{health:87,kills:3},world:{entities:[]}},{player:{health:62,kills:3},world:{entities:[]}});
const rewardDead=arena.reward({player:{health:10,kills:3},world:{entities:[]}},{player:{health:0,kills:3},world:{entities:[]}});
assert.ok(rewardGood>1);assert.ok(rewardBad<0);assert.ok(rewardDead<rewardBad);
assert.equal(arena.actionMasks.fire,64);assert.equal(arena.actionMasks.forward_fire,65);assert.equal(arena.actionMasks.strafe_left_fire,80);
assert.equal(arena.actions.find(a=>a.id==="fire").params.fire,1);assert.equal(arena.actions.find(a=>a.id==="fire").params.forward,0);
assert.equal(arena.actions.find(a=>a.id==="forward_fire").params.forward,1);assert.equal(arena.actions.find(a=>a.id==="forward_fire").params.fire,1);
assert.equal(arena.schema.actionFields.length,8);
console.log(JSON.stringify({ok:true,globals:arena.schema.fields.length,actions:arena.actions.length,damageOutcome,rewardGood,rewardBad,rewardDead}));
