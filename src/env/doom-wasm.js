const RUNTIME_REPOSITORY="lukaske/jev-doom-agent";
const RUNTIME_COMMIT="318c32a24851444c1170bf083671c38723f3a35a";
const RAW_BASE="https://raw.githubusercontent.com/"+RUNTIME_REPOSITORY+"/"+RUNTIME_COMMIT+"/public/engine";

const BITS=Object.freeze({FORWARD:1,BACK:2,TURN_LEFT:4,TURN_RIGHT:8,STRAFE_LEFT:16,STRAFE_RIGHT:32,FIRE:64,USE:128});
const controls=(mask)=>Object.freeze({
  forward:(mask&BITS.FORWARD)?1:0,back:(mask&BITS.BACK)?1:0,turn_left:(mask&BITS.TURN_LEFT)?1:0,turn_right:(mask&BITS.TURN_RIGHT)?1:0,
  strafe_left:(mask&BITS.STRAFE_LEFT)?1:0,strafe_right:(mask&BITS.STRAFE_RIGHT)?1:0,fire:(mask&BITS.FIRE)?1:0,use:(mask&BITS.USE)?1:0
});
const action=(id,label,description,mask)=>Object.freeze({id,label,description,mask,params:controls(mask)});
const ACTION_SPECS=Object.freeze([
  action("forward","forward","hold forward movement",BITS.FORWARD),
  action("back","back","hold backward movement",BITS.BACK),
  action("turn_left","turn left","turn the view left",BITS.TURN_LEFT),
  action("turn_right","turn right","turn the view right",BITS.TURN_RIGHT),
  action("strafe_left","strafe left","move sideways left while keeping the current view direction",BITS.STRAFE_LEFT),
  action("strafe_right","strafe right","move sideways right while keeping the current view direction",BITS.STRAFE_RIGHT),
  action("fire","fire","fire the currently equipped weapon",BITS.FIRE),
  action("forward_fire","forward + fire","hold forward movement and weapon fire at the same time",BITS.FORWARD|BITS.FIRE),
  action("back_fire","back + fire","hold backward movement and weapon fire at the same time",BITS.BACK|BITS.FIRE),
  action("strafe_left_fire","strafe left + fire","hold left strafe and weapon fire at the same time",BITS.STRAFE_LEFT|BITS.FIRE),
  action("strafe_right_fire","strafe right + fire","hold right strafe and weapon fire at the same time",BITS.STRAFE_RIGHT|BITS.FIRE),
  action("turn_left_fire","turn left + fire","turn left and fire the equipped weapon at the same time",BITS.TURN_LEFT|BITS.FIRE),
  action("turn_right_fire","turn right + fire","turn right and fire the equipped weapon at the same time",BITS.TURN_RIGHT|BITS.FIRE),
  action("use","interact / open","press the use key to open doors, activate switches, lifts, or other usable map elements directly ahead",BITS.USE),
  action("wait","wait","apply no movement, turning, firing, or use input for this decision interval",0)
]);
const ENTITY_TYPES=Object.freeze({
  0:"player",1:"zombie man",2:"shotgun guy",3:"arch-vile",4:"arch-vile fire",5:"revenant",6:"revenant tracer missile",7:"smoke",8:"mancubus",9:"mancubus fireball",
  10:"chaingunner",11:"imp",12:"demon",13:"spectre",14:"cacodemon",15:"baron of hell",16:"baron fireball",17:"hell knight",18:"lost soul",19:"spider mastermind",
  20:"arachnotron",21:"cyberdemon",22:"pain elemental",23:"Wolfenstein SS",24:"Commander Keen",25:"Icon of Sin brain",26:"boss spawn spit",27:"spawn target",
  28:"spawn cube",29:"spawn fire",30:"explosive barrel",31:"imp fireball",32:"cacodemon fireball",33:"rocket projectile",34:"plasma projectile",35:"BFG projectile",
  36:"arachnotron plasma projectile",37:"bullet impact puff",38:"blood effect",39:"teleport fog",40:"item respawn fog",41:"teleport destination",42:"BFG explosion effect"
});
const PROJECTILE_TYPES=new Set([4,6,9,16,26,28,29,31,32,33,34,35,36,42]);
const ENTITY_KINDS=Object.freeze({0:"other object",1:"hostile actor",2:"projectile or attack effect",3:"collectible item"});
function entityKind(e){
  const type=Number(e.type||0);
  if(PROJECTILE_TYPES.has(type))return 2;
  if(e.enemy)return 1;
  if(e.pickup)return 3;
  return 0;
}
const sleep=ms=>new Promise(resolve=>setTimeout(resolve,ms));
const clamp=(v,min,max)=>Math.max(min,Math.min(max,v));

async function fetchAsset(name,type="arrayBuffer"){
  const response=await fetch(RAW_BASE+"/"+name,{cache:"force-cache"});
  if(!response.ok)throw new Error("Failed to fetch "+name+": HTTP "+response.status);
  return type==="text"?response.text():response.arrayBuffer();
}
function mkdir(fs,path){try{fs.mkdir(path)}catch{}}
function hostileHealth(raw){
  return (raw?.world?.entities||[]).reduce((sum,e)=>sum+(e.enemy&&Number(e.health)>0?Number(e.health):0),0);
}

export class DoomWasmArena{
  constructor(module,{actionMs=110,settleMs=16,contentName="Freedoom 0.13.0"}={}){
    this.module=module;this.actionMs=actionMs;this.settleMs=settleMs;this.contentName=contentName;
    this.runtime={repository:RUNTIME_REPOSITORY,commit:RUNTIME_COMMIT};
    this.actions=ACTION_SPECS.map(({mask,...action})=>action);
    this.actionMasks=Object.fromEntries(ACTION_SPECS.map(action=>[action.id,action.mask]));
    this.schema={
      objective:"Stay alive, neutralize hostile threats, conserve useful resources, and make progress through the level.",
      actionFields:[
        {id:"forward",label:"forward control",description:"whether this candidate holds forward movement",min:0,max:1},
        {id:"back",label:"backward control",description:"whether this candidate holds backward movement",min:0,max:1},
        {id:"turn_left",label:"turn left control",description:"whether this candidate turns the view left",min:0,max:1},
        {id:"turn_right",label:"turn right control",description:"whether this candidate turns the view right",min:0,max:1},
        {id:"strafe_left",label:"strafe left control",description:"whether this candidate moves sideways left",min:0,max:1},
        {id:"strafe_right",label:"strafe right control",description:"whether this candidate moves sideways right",min:0,max:1},
        {id:"fire",label:"weapon fire control",description:"whether this candidate fires the equipped weapon",min:0,max:1},
        {id:"use",label:"use interaction control",description:"whether this candidate activates or uses the environment",min:0,max:1}
      ],
      fields:[
        {id:"health",label:"health",description:"remaining player vitality",min:0,max:200},
        {id:"armor",label:"armor",description:"remaining protective armor",min:0,max:200},
        {id:"bullets",label:"bullets",description:"available pistol or chaingun ammunition",min:0,max:400},
        {id:"shells",label:"shells",description:"available shotgun ammunition",min:0,max:100},
        {id:"rockets",label:"rockets",description:"available rocket launcher ammunition",min:0,max:100},
        {id:"cells",label:"cells",description:"available plasma or BFG energy ammunition",min:0,max:600},
        {id:"recent_damage",label:"recent damage received",description:"damage registered on the player in the recent engine combat window",min:0,max:100},
        {id:"recent_damage_dealt",label:"recent hostile damage dealt",description:"hostile hit points removed during the previous control interval",min:0,max:200},
        {id:"under_fire",label:"under fire",description:"whether the engine currently reports recent incoming damage",min:0,max:1},
        {id:"weapon",label:"equipped weapon",description:"weapon currently equipped by the player",enum:{
          0:"fist",1:"pistol",2:"shotgun",3:"chaingun",4:"rocket launcher",5:"plasma rifle",6:"BFG 9000",7:"chainsaw",8:"super shotgun"
        }},
        {id:"player_x",label:"player x position",description:"player world x coordinate",scale:2048},
        {id:"player_y",label:"player y position",description:"player world y coordinate",scale:2048},
        {id:"player_z",label:"player z position",description:"player world vertical coordinate",scale:256},
        {id:"velocity_x",label:"player x velocity",description:"player horizontal x momentum",scale:32},
        {id:"velocity_y",label:"player y velocity",description:"player horizontal y momentum",scale:32},
        {id:"heading",label:"player heading",description:"player view angle as a signed normalized turn",min:-1,max:1},
        {id:"kills",label:"kills",description:"hostile actors defeated by the player in the current episode",min:0,max:100},
        {id:"visited_cells",label:"visited spatial cells",description:"number of distinct coarse player-position cells visited this episode",min:0,max:500},
        {id:"cell_visits",label:"current cell visits",description:"number of control transitions ending in the current coarse spatial cell",scale:16},
        {id:"exploration_novelty",label:"exploration novelty",description:"inverse revisit count of the current spatial cell; higher means less familiar",min:0,max:1}
      ],
      collections:[
        {
          id:"entities",label:"world entities",description:"dynamic actors, objects and pickups represented with absolute and player-relative state",
          fields:[
            {id:"engine_record_id",label:"engine record id",description:"numeric record identifier supplied by the engine",scale:128},
            {id:"type",label:"entity type",description:"engine object category",enum:ENTITY_TYPES,scale:128},
            {id:"kind",label:"entity kind",description:"coarse factual engine category of this record",enum:ENTITY_KINDS},
            {id:"x",label:"entity x position",description:"entity absolute world x coordinate",scale:2048},
            {id:"y",label:"entity y position",description:"entity absolute world y coordinate",scale:2048},
            {id:"z",label:"entity z position",description:"entity absolute world vertical coordinate",scale:256},
            {id:"relative_x",label:"relative x",description:"entity x displacement from the player",scale:1024},
            {id:"relative_y",label:"relative y",description:"entity y displacement from the player",scale:1024},
            {id:"relative_z",label:"relative z",description:"entity z displacement from the player",scale:256},
            {id:"velocity_x",label:"entity x velocity",description:"entity x momentum",scale:32},
            {id:"velocity_y",label:"entity y velocity",description:"entity y momentum",scale:32},
            {id:"radius",label:"entity radius",description:"physical collision radius",scale:64},
            {id:"height",label:"entity height",description:"physical collision height",scale:128},
            {id:"health",label:"entity health",description:"remaining actor health when applicable",scale:256},
            {id:"distance",label:"distance",description:"distance from player to entity",scale:1024},
            {id:"relative_angle",label:"relative angle",description:"signed angular displacement from the player's view",min:-1,max:1},
            {id:"visible",label:"line of sight",description:"whether the engine reports direct line of sight",min:0,max:1},
            {id:"countkill",label:"hostile actor flag",description:"whether this actor counts toward the level hostile kill total",min:0,max:1},
            {id:"pickup",label:"collectible flag",description:"whether this engine object is collectible or special",min:0,max:1},
            {id:"targeting_player",label:"targets player",description:"whether this actor currently targets the player",min:0,max:1}
          ]
        },
        {
          id:"geometry",label:"world geometry",description:"map line segments represented relative to the player",
          fields:[
            {id:"line_id",label:"line id",description:"numeric map line identifier",scale:2048},
            {id:"x1",label:"line endpoint one x",description:"first endpoint x displacement from player",scale:1024},
            {id:"y1",label:"line endpoint one y",description:"first endpoint y displacement from player",scale:1024},
            {id:"x2",label:"line endpoint two x",description:"second endpoint x displacement from player",scale:1024},
            {id:"y2",label:"line endpoint two y",description:"second endpoint y displacement from player",scale:1024},
            {id:"flags",label:"line flags",description:"numeric engine bit flags attached to this map line",scale:1024},
            {id:"blocking",label:"blocking geometry",description:"whether this line blocks player movement",min:0,max:1},
            {id:"special",label:"interactive line special",description:"numeric engine special action attached to the line",scale:32},
            {id:"tag",label:"line tag",description:"numeric map linkage tag attached to the line",scale:32}
          ]
        }
      ]
    };
    this.lastRaw=null;this.lastObservation=null;this.lastDamageDealt=0;this.lastOutcome=null;this.visitedCells=new Map();this.lastExploration={visitedCells:0,cellVisits:0,novelty:1,newCell:false};
  }

  static async boot({canvas,onProgress=()=>{},actionMs=110,iwadFile=null,contentName=null}={}){
    if(!canvas)throw new Error("DoomWasmArena.boot requires a canvas");
    onProgress("fetching pinned Chocolate Doom runtime");
    const [source,wasmBinary,dataPackage]=await Promise.all([fetchAsset("chocolate-doom.js","text"),fetchAsset("chocolate-doom.wasm"),fetchAsset("chocolate-doom.data")]);
    onProgress("instantiating WebAssembly runtime");
    const blobUrl=URL.createObjectURL(new Blob([source],{type:"text/javascript"}));
    let createModule;try{({default:createModule}=await import(blobUrl))}finally{URL.revokeObjectURL(blobUrl)}
    if(typeof createModule!=="function")throw new Error("Chocolate Doom module factory was not exported");
    const useCustom=!!iwadFile;
    const module=await createModule({
      canvas,keyboardListeningElement:canvas,wasmBinary,locateFile:path=>RAW_BASE+"/"+path,getPreloadedPackage:()=>dataPackage,noInitialRun:true,
      preRun:[m=>{
        mkdir(m.FS,"/config");mkdir(m.FS,"/savegames");mkdir(m.FS,"/iwads");
        m.FS.writeFile("/config/default.cfg","fullscreen 0\ngrabmouse 0\nuse_mouse 0\n");
        m.FS.writeFile("/config/chocolate-doom.cfg","aspect_ratio_correct 1\ninteger_scaling 0\nscreenblocks 10\nsmooth_pixel_scaling 0\nforce_software_renderer 1\n");
        if(useCustom)m.FS.writeFile("/iwads/user.wad",new Uint8Array(iwadFile));
      }],
      print:()=>{},printErr:message=>console.warn("[doom]",message)
    });
    const iwadPath=useCustom?"/iwads/user.wad":"/iwads/freedoom2.wad";
    onProgress("starting "+(useCustom?(contentName||"user IWAD"):"Freedoom"));
    try{module.callMain(["-window","-iwad",iwadPath,"-warp","1","-skill","1","-nomusic","-nosound","-config","/config/default.cfg","-extraconfig","/config/chocolate-doom.cfg"])}
    catch(error){const message=String(error);if(!message.includes("unwind")&&!message.includes("SimulateInfiniteLoop"))throw error}
    const arena=new DoomWasmArena(module,{actionMs,contentName:useCustom?(contentName||"user-provided IWAD"):"Freedoom 0.13.0"});
    await arena.waitUntilReady();await arena.reset();onProgress("ready");return arena;
  }

  setActionMs(ms){this.actionMs=clamp(Number(ms)||110,35,1000)}
  readRaw(){
    const json=this.module.ccall("PromptFPS_Observation","string",[],[]);
    const raw=JSON.parse(String(json));if(!raw.ready)throw new Error("Doom telemetry bridge is not ready");return raw;
  }
  async waitUntilReady(timeoutMs=8000){
    const started=performance.now();
    while(performance.now()-started<timeoutMs){try{const raw=this.readRaw();if(raw.ready)return raw}catch{}await sleep(80)}
    throw new Error("Timed out waiting for Doom telemetry");
  }
  cellKey(raw){
    const p=raw?.player||{};return Math.floor(Number(p.x||0)/128)+","+Math.floor(Number(p.y||0)/128);
  }
  commitExploration(raw){
    const key=this.cellKey(raw),before=this.visitedCells.get(key)||0,next=before+1;
    this.visitedCells.set(key,next);
    const state={visitedCells:this.visitedCells.size,cellVisits:next,novelty:1/next,newCell:before===0};
    this.lastExploration=state;return state;
  }
  flatten(raw){
    const p=raw.player||{},all=raw.world?.entities||[],lines=raw.world?.lines||[];
    const heading=((Number(p.angle||0)>>>0)/4294967296)*2-1;
    const entities=all.map(e=>({
      engine_record_id:Number(e.id||0),type:Number(e.type||0),kind:entityKind(e),x:Number(e.x||0),y:Number(e.y||0),z:Number(e.z||0),
      relative_x:Number(e.relative_x||0),relative_y:Number(e.relative_y||0),relative_z:Number(e.z||0)-Number(p.z||0),
      velocity_x:Number(e.vx||0),velocity_y:Number(e.vy||0),radius:Number(e.radius||0),height:Number(e.height||0),health:Number(e.health||0),
      distance:Number(e.distance||0),relative_angle:clamp(Number(e.relative_angle||0)/2147483648,-1,1),visible:e.visible?1:0,countkill:e.enemy?1:0,pickup:e.pickup?1:0,targeting_player:e.targeting_player?1:0
    }));
    const geometry=lines.map(line=>({
      line_id:Number(line.id||0),x1:Number(line.x1||0)-Number(p.x||0),y1:Number(line.y1||0)-Number(p.y||0),x2:Number(line.x2||0)-Number(p.x||0),y2:Number(line.y2||0)-Number(p.y||0),
      flags:Number(line.flags||0),blocking:line.blocking?1:0,special:Number(line.special||0),tag:Number(line.tag||0)
    }));
    return{
      health:Number(p.health||0),armor:Number(p.armor||0),bullets:Number(p.ammo?.bullets||0),shells:Number(p.ammo?.shells||0),rockets:Number(p.ammo?.rockets||0),cells:Number(p.ammo?.cells||0),
      recent_damage:Number(p.recent_damage||0),recent_damage_dealt:Number(this.lastDamageDealt||0),under_fire:p.under_fire?1:0,weapon:Number(p.weapon||0),
      player_x:Number(p.x||0),player_y:Number(p.y||0),player_z:Number(p.z||0),velocity_x:Number(p.vx||0),velocity_y:Number(p.vy||0),heading,kills:Number(p.kills||0),
      visited_cells:Number(this.lastExploration?.visitedCells||0),cell_visits:Number(this.lastExploration?.cellVisits||0),exploration_novelty:Number(this.lastExploration?.novelty??1),
      _collections:{entities,geometry}
    };
  }
  observe(){const raw=this.readRaw();this.lastRaw=raw;this.lastObservation=this.flatten(raw);return this.lastObservation}
  outcome(previous,next,{exploration=this.lastExploration}={}){
    const prev=previous?.player||{},cur=next?.player||{};
    const healthDelta=Number(cur.health||0)-Number(prev.health||0);
    const killDelta=Math.max(0,Number(cur.kills||0)-Number(prev.kills||0));
    const damageDealt=Math.max(0,hostileHealth(previous)-hostileHealth(next));
    const explorationBonus=exploration?.newCell ? .015 : 0;
    let reward=-.001+damageDealt*.02+killDelta*1.25+explorationBonus;
    if(healthDelta<0)reward+=healthDelta*.03;else if(healthDelta>0)reward+=healthDelta*.005;
    if(Number(cur.health||0)<=0)reward-=2;
    return{reward,damageDealt,healthDelta,killDelta,explorationBonus,newCell:!!exploration?.newCell,visitedCells:Number(exploration?.visitedCells||0),dead:Number(cur.health||0)<=0};
  }
  reward(previous,next){return this.outcome(previous,next).reward}
  async step(actionId){
    if(!(actionId in this.actionMasks))throw new Error("Unsupported Doom action: "+actionId);
    const before=this.lastRaw||this.readRaw();
    this.module.ccall("PromptFPS_SetControls",null,["number"],[this.actionMasks[actionId]]);
    await sleep(this.actionMs);this.module.ccall("PromptFPS_SetControls",null,["number"],[0]);if(this.settleMs)await sleep(this.settleMs);
    const after=this.readRaw(),exploration=this.commitExploration(after),outcome=this.outcome(before,after,{exploration});
    this.lastDamageDealt=outcome.damageDealt;this.lastOutcome=outcome;this.lastRaw=after;this.lastObservation=this.flatten(after);
    return{observation:this.lastObservation,reward:outcome.reward,done:outcome.dead,info:{raw:after,outcome,engine:after.engine||"Chocolate Doom"}};
  }
  async reset(){
    this.module.ccall("PromptFPS_SetControls",null,["number"],[0]);this.module.ccall("PromptFPS_SetStart",null,[],[]);
    this.lastDamageDealt=0;this.lastOutcome=null;this.visitedCells=new Map();await sleep(100);const raw=await this.waitUntilReady();this.commitExploration(raw);this.lastExploration.newCell=false;this.lastRaw=raw;this.lastObservation=this.flatten(raw);return this.lastObservation;
  }
}

export const DOOM_RUNTIME_PROVENANCE=Object.freeze({repository:RUNTIME_REPOSITORY,commit:RUNTIME_COMMIT,engine:"Chocolate Doom 3.1.1",content:"Freedoom 0.13.0",note:"Runtime is fetched from a pinned public research build; policy/controller code is implemented independently in doom-classifier."});
