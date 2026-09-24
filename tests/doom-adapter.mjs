import assert from "node:assert/strict";
import {DoomWasmArena} from "../src/env/doom-wasm.js";

const arena=new DoomWasmArena({},{actionMs:120});
const raw={
  ready:true,
  player:{health:87,armor:40,ammo:{bullets:33,shells:8,rockets:2,cells:12},recent_damage:6,kills:3},
  world:{entities:[
    {enemy:true,visible:true,health:60,distance:700,relative_angle:1073741824,targeting_player:false},
    {enemy:true,visible:true,health:30,distance:220,relative_angle:-536870912,targeting_player:true},
    {enemy:true,visible:false,health:50,distance:100,relative_angle:0,targeting_player:true},
    {pickup:true,visible:true,health:0,distance:90,relative_angle:0}
  ]}
};
const flat=arena.flatten(raw);
assert.equal(flat.health,87);
assert.equal(flat.visible_hostiles,2);
assert.equal(flat.nearest_hostile_distance,220);
assert.equal(flat.nearest_hostile_health,30);
assert.equal(flat.nearest_hostile_targeting,1);
assert.equal(flat.visible_pickups,1);
assert.ok(Math.abs(flat.nearest_hostile_bearing+0.25)<1e-9);

const rewardGood=arena.reward({player:{health:87,kills:3}},{player:{health:87,kills:4}});
const rewardBad=arena.reward({player:{health:87,kills:3}},{player:{health:62,kills:3}});
const rewardDead=arena.reward({player:{health:10,kills:3}},{player:{health:0,kills:3}});
assert.ok(rewardGood>1);
assert.ok(rewardBad<0);
assert.ok(rewardDead<rewardBad);
console.log(JSON.stringify({ok:true,flat,rewardGood,rewardBad,rewardDead}));
