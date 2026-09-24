import assert from "node:assert/strict";
import {DoomWasmArena} from "../src/env/doom-wasm.js";

const arena=new DoomWasmArena({},{actionMs:120});
const raw={
  ready:true,
  player:{health:87,armor:40,weapon:2,ammo:{bullets:33,shells:8,rockets:2,cells:12},recent_damage:6,kills:3,x:100,y:200,z:8,vx:2,vy:-1,angle:1073741824},
  world:{
    entities:[
      {type:1,x:200,y:200,z:8,vx:1,vy:0,radius:20,height:56,health:60,distance:700,relative_x:100,relative_y:0,relative_angle:1073741824,visible:true,enemy:true,pickup:false,targeting_player:false},
      {type:2,x:120,y:180,z:8,vx:0,vy:0,radius:20,height:56,health:30,distance:220,relative_x:20,relative_y:-20,relative_angle:-536870912,visible:true,enemy:true,pickup:false,targeting_player:true},
      {type:3,x:110,y:200,z:8,vx:0,vy:0,radius:8,height:16,health:0,distance:90,relative_x:10,relative_y:0,relative_angle:0,visible:true,enemy:false,pickup:true,targeting_player:false}
    ],
    lines:[{id:0,x1:80,y1:180,x2:160,y2:180,blocking:true,special:0,tag:0}]
  }
};
const flat=arena.flatten(raw);
assert.equal(flat.health,87);assert.equal(flat.weapon,2);assert.equal(flat._collections.entities.length,3);assert.equal(flat._collections.geometry.length,1);
assert.equal(flat._collections.entities[1].targeting_player,1);assert.equal(flat._collections.entities[2].pickup,1);
assert.equal(flat._collections.geometry[0].x1,-20);assert.equal(flat._collections.geometry[0].y1,-20);
const rewardGood=arena.reward({player:{health:87,kills:3}},{player:{health:87,kills:4}});
const rewardBad=arena.reward({player:{health:87,kills:3}},{player:{health:62,kills:3}});
const rewardDead=arena.reward({player:{health:10,kills:3}},{player:{health:0,kills:3}});
assert.ok(rewardGood>1);assert.ok(rewardBad<0);assert.ok(rewardDead<rewardBad);
console.log(JSON.stringify({ok:true,globals:arena.schema.fields.length,entities:flat._collections.entities.length,geometry:flat._collections.geometry.length,rewardGood,rewardBad,rewardDead}));
