const RUNTIME_REPOSITORY="lukaske/jev-doom-agent";
const RUNTIME_COMMIT="318c32a24851444c1170bf083671c38723f3a35a";
const RAW_BASE="https://raw.githubusercontent.com/"+RUNTIME_REPOSITORY+"/"+RUNTIME_COMMIT+"/public/engine";

const CONTROL=Object.freeze({forward:1,back:2,turn_left:4,turn_right:8,strafe_left:16,strafe_right:32,fire:64,use:128,wait:0});
const sleep=ms=>new Promise(resolve=>setTimeout(resolve,ms));
const clamp=(v,min,max)=>Math.max(min,Math.min(max,v));

async function fetchAsset(name,type="arrayBuffer"){
  const response=await fetch(RAW_BASE+"/"+name,{cache:"force-cache"});
  if(!response.ok)throw new Error("Failed to fetch "+name+": HTTP "+response.status);
  return type==="text"?response.text():response.arrayBuffer();
}
function mkdir(fs,path){try{fs.mkdir(path)}catch{}}

export class DoomWasmArena{
  constructor(module,{actionMs=120,settleMs=20}={}){
    this.module=module;this.actionMs=actionMs;this.settleMs=settleMs;
    this.runtime={repository:RUNTIME_REPOSITORY,commit:RUNTIME_COMMIT};
    this.actions=[
      {id:"forward",label:"forward",description:"move straight ahead using the current view direction"},
      {id:"back",label:"back",description:"move backward away from the current view direction"},
      {id:"turn_left",label:"turn left",description:"rotate the player's view to the left"},
      {id:"turn_right",label:"turn right",description:"rotate the player's view to the right"},
      {id:"strafe_left",label:"strafe left",description:"move sideways to the left without intentionally changing view direction"},
      {id:"strafe_right",label:"strafe right",description:"move sideways to the right without intentionally changing view direction"},
      {id:"fire",label:"fire",description:"fire the currently equipped weapon"},
      {id:"use",label:"use",description:"activate or interact with something directly in front of the player"},
      {id:"wait",label:"wait",description:"apply no movement, turning, firing, or use input for this decision interval"}
    ];
    this.schema={
      objective:"Stay alive, avoid unnecessary damage, neutralize hostile threats when useful, use resources efficiently, and continue making progress through the environment.",
      fields:[
        {id:"health",label:"health",description:"remaining player vitality",min:0,max:200},
        {id:"armor",label:"armor",description:"remaining protective armor",min:0,max:200},
        {id:"bullets",label:"bullets",description:"available bullet ammunition",min:0,max:400},
        {id:"shells",label:"shells",description:"available shotgun shell ammunition",min:0,max:100},
        {id:"rockets",label:"rockets",description:"available rocket ammunition",min:0,max:100},
        {id:"cells",label:"cells",description:"available energy-cell ammunition",min:0,max:600},
        {id:"recent_damage",label:"recent damage",description:"damage registered by the engine in the recent combat window",min:0,max:100},
        {id:"visible_hostiles",label:"visible hostiles",description:"number of living hostile actors currently visible to the player",min:0,max:16},
        {id:"nearest_hostile_distance",label:"nearest hostile distance",description:"world-space distance to the nearest visible hostile actor; larger means farther away",min:0,max:2048},
        {id:"nearest_hostile_bearing",label:"nearest hostile bearing",description:"signed relative bearing of the nearest visible hostile; zero is directly ahead, negative and positive values are opposite turn directions",min:-1,max:1},
        {id:"nearest_hostile_health",label:"nearest hostile health",description:"remaining health of the nearest visible hostile actor",min:0,max:1000},
        {id:"nearest_hostile_targeting",label:"nearest hostile targeting player",description:"whether the nearest visible hostile currently targets the player",min:0,max:1},
        {id:"visible_pickups",label:"visible pickups",description:"number of currently visible collectible world objects",min:0,max:24},
        {id:"kills",label:"kills",description:"hostile actors defeated by the player in the current episode",min:0,max:100}
      ]
    };
    this.lastRaw=null;this.lastObservation=null;
  }

  static async boot({canvas,onProgress=()=>{},actionMs=120}={}){
    if(!canvas)throw new Error("DoomWasmArena.boot requires a canvas");
    onProgress("fetching pinned Chocolate Doom runtime");
    const [source,wasmBinary,dataPackage]=await Promise.all([
      fetchAsset("chocolate-doom.js","text"),fetchAsset("chocolate-doom.wasm"),fetchAsset("chocolate-doom.data")
    ]);
    onProgress("instantiating WebAssembly runtime");
    const blobUrl=URL.createObjectURL(new Blob([source],{type:"text/javascript"}));
    let createModule;
    try{({default:createModule}=await import(blobUrl))}finally{URL.revokeObjectURL(blobUrl)}
    if(typeof createModule!=="function")throw new Error("Chocolate Doom module factory was not exported");
    const module=await createModule({
      canvas,keyboardListeningElement:canvas,wasmBinary,getPreloadedPackage:()=>dataPackage,noInitialRun:true,
      preRun:[m=>{
        mkdir(m.FS,"/config");mkdir(m.FS,"/savegames");
        m.FS.writeFile("/config/default.cfg","fullscreen 0\ngrabmouse 0\nuse_mouse 0\n");
        m.FS.writeFile("/config/chocolate-doom.cfg","aspect_ratio_correct 1\ninteger_scaling 0\nscreenblocks 10\nsmooth_pixel_scaling 0\nforce_software_renderer 1\n");
      }],
      print:()=>{},printErr:message=>console.warn("[doom]",message)
    });
    onProgress("starting Freedoom");
    try{
      module.callMain(["-window","-iwad","/iwads/freedoom2.wad","-warp","1","-skill","1","-nomusic","-nosound","-config","/config/default.cfg","-extraconfig","/config/chocolate-doom.cfg"]);
    }catch(error){
      const message=String(error);
      if(!message.includes("unwind")&&!message.includes("SimulateInfiniteLoop"))throw error;
    }
    const arena=new DoomWasmArena(module,{actionMs});
    await arena.waitUntilReady();await arena.reset();onProgress("ready");return arena;
  }

  setActionMs(ms){this.actionMs=clamp(Number(ms)||120,35,1000)}
  readRaw(){
    const json=this.module.ccall("PromptFPS_Observation","string",[],[]);
    const raw=JSON.parse(String(json));
    if(!raw.ready)throw new Error("Doom telemetry bridge is not ready");
    return raw;
  }
  async waitUntilReady(timeoutMs=8000){
    const started=performance.now();
    while(performance.now()-started<timeoutMs){
      try{const raw=this.readRaw();if(raw.ready)return raw}catch{}
      await sleep(80);
    }
    throw new Error("Timed out waiting for Doom telemetry");
  }
  flatten(raw){
    const player=raw.player||{};
    const all=raw.world?.entities||[];
    const enemies=all.filter(entity=>entity.enemy&&entity.visible&&entity.health>0).sort((a,b)=>a.distance-b.distance);
    const nearest=enemies[0],relative=nearest?clamp(Number(nearest.relative_angle||0)/2147483648,-1,1):0;
    return{
      health:Number(player.health||0),armor:Number(player.armor||0),bullets:Number(player.ammo?.bullets||0),shells:Number(player.ammo?.shells||0),
      rockets:Number(player.ammo?.rockets||0),cells:Number(player.ammo?.cells||0),recent_damage:Number(player.recent_damage||0),
      visible_hostiles:enemies.length,nearest_hostile_distance:nearest?clamp(Number(nearest.distance||0),0,2048):2048,
      nearest_hostile_bearing:relative,nearest_hostile_health:nearest?Math.max(0,Number(nearest.health||0)):0,
      nearest_hostile_targeting:nearest?.targeting_player?1:0,visible_pickups:all.filter(entity=>entity.pickup&&entity.visible).length,kills:Number(player.kills||0)
    };
  }
  observe(){
    const raw=this.readRaw();this.lastRaw=raw;this.lastObservation=this.flatten(raw);return this.lastObservation;
  }
  reward(previous,next){
    const prev=previous?.player||{},cur=next?.player||{};
    const healthDelta=Number(cur.health||0)-Number(prev.health||0),killDelta=Math.max(0,Number(cur.kills||0)-Number(prev.kills||0));
    let reward=-0.002+killDelta*2;
    if(healthDelta<0)reward+=healthDelta/50;else if(healthDelta>0)reward+=healthDelta/200;
    if(Number(cur.health||0)<=0)reward-=2;
    return reward;
  }
  async step(actionId){
    if(!(actionId in CONTROL))throw new Error("Unsupported Doom action: "+actionId);
    const before=this.lastRaw||this.readRaw();
    this.module.ccall("PromptFPS_SetControls",null,["number"],[CONTROL[actionId]]);
    await sleep(this.actionMs);this.module.ccall("PromptFPS_SetControls",null,["number"],[0]);
    if(this.settleMs)await sleep(this.settleMs);
    const after=this.readRaw();this.lastRaw=after;this.lastObservation=this.flatten(after);
    return{observation:this.lastObservation,reward:this.reward(before,after),done:Number(after.player?.health||0)<=0,info:{raw:after,engine:after.engine||"Chocolate Doom"}};
  }
  async reset(){
    this.module.ccall("PromptFPS_SetControls",null,["number"],[0]);this.module.ccall("PromptFPS_SetStart",null,[],[]);
    await sleep(100);const raw=await this.waitUntilReady();this.lastRaw=raw;this.lastObservation=this.flatten(raw);return this.lastObservation;
  }
}

export const DOOM_RUNTIME_PROVENANCE=Object.freeze({
  repository:RUNTIME_REPOSITORY,commit:RUNTIME_COMMIT,engine:"Chocolate Doom 3.1.1",content:"Freedoom 0.13.0",
  note:"Runtime is fetched from a pinned public research build; policy/controller code is implemented independently in doom-classifier."
});
