import {clamp,mulberry32} from "../core/math.js";

export class MockArena{
  constructor(seed=1337){
    this.seed=seed;
    this.actions=[
      {id:"forward",label:"forward",description:"move toward the current facing direction"},
      {id:"back",label:"back",description:"move away from the current facing direction"},
      {id:"strafe_left",label:"strafe left",description:"move laterally to the left while preserving facing"},
      {id:"strafe_right",label:"strafe right",description:"move laterally to the right while preserving facing"},
      {id:"turn_left",label:"turn left",description:"rotate facing toward the left"},
      {id:"turn_right",label:"turn right",description:"rotate facing toward the right"},
      {id:"fire",label:"fire",description:"activate the currently equipped ranged weapon"},
      {id:"wait",label:"wait",description:"take no movement or weapon action"}
    ];
    this.schema={fields:[
      {id:"health",label:"health",description:"remaining player vitality",min:0,max:1},
      {id:"ammo",label:"ammunition",description:"remaining ammunition for the equipped ranged weapon",min:0,max:1},
      {id:"threat_distance",label:"threat distance",description:"normalized distance from player to visible hostile threat",min:0,max:1},
      {id:"threat_bearing",label:"threat bearing",description:"signed horizontal bearing of hostile threat relative to facing",min:-1,max:1},
      {id:"incoming_damage",label:"recent damage",description:"damage received during the previous environment transition",min:0,max:1},
      {id:"goal_distance",label:"goal distance",description:"normalized distance from player to progress objective",min:0,max:1},
      {id:"cover",label:"cover",description:"local protection from hostile attack",min:0,max:1}
    ]};
    this.episode=0;this.reset();
  }
  reset(){
    this.rng=mulberry32(this.seed+this.episode);this.episode++;
    this.player={x:.18,y:.5,angle:0,health:1,ammo:1};this.enemy={x:.72,y:.22,health:1};this.goal={x:.9,y:.82};
    this.lastDamage=0;this.t=0;this.done=false;this.lastReward=0;return this.observe();
  }
  observe(){
    const dx=this.enemy.x-this.player.x,dy=this.enemy.y-this.player.y,dist=Math.hypot(dx,dy);
    const targetAngle=Math.atan2(dy,dx);let d=targetAngle-this.player.angle;
    while(d>Math.PI)d-=Math.PI*2;while(d<-Math.PI)d+=Math.PI*2;
    const gd=Math.hypot(this.goal.x-this.player.x,this.goal.y-this.player.y);
    return{health:this.player.health,ammo:this.player.ammo,threat_distance:clamp(dist/1.25,0,1),threat_bearing:clamp(d/Math.PI,-1,1),incoming_damage:clamp(this.lastDamage/.18,0,1),goal_distance:clamp(gd/1.25,0,1),cover:this.coverAt(this.player.x,this.player.y)};
  }
  coverAt(x,y){
    const cx=.48,cy=.58,w=.16,h=.11,dx=Math.max(Math.abs(x-cx)-w,0),dy=Math.max(Math.abs(y-cy)-h,0);
    return clamp(1-Math.hypot(dx,dy)*5,0,1);
  }
  step(action){
    if(this.done)return{observation:this.observe(),reward:0,done:true};
    this.t++;this.lastDamage=0;let reward=-.002;
    const speed=.045,rot=.24;let mx=0,my=0;
    if(action==="turn_left")this.player.angle-=rot;if(action==="turn_right")this.player.angle+=rot;
    if(action==="forward"){mx=Math.cos(this.player.angle)*speed;my=Math.sin(this.player.angle)*speed}
    if(action==="back"){mx=-Math.cos(this.player.angle)*speed;my=-Math.sin(this.player.angle)*speed}
    if(action==="strafe_left"){mx=Math.cos(this.player.angle-Math.PI/2)*speed;my=Math.sin(this.player.angle-Math.PI/2)*speed}
    if(action==="strafe_right"){mx=Math.cos(this.player.angle+Math.PI/2)*speed;my=Math.sin(this.player.angle+Math.PI/2)*speed}
    this.player.x=clamp(this.player.x+mx,.03,.97);this.player.y=clamp(this.player.y+my,.03,.97);

    const edx=this.enemy.x-this.player.x,edy=this.enemy.y-this.player.y,ed=Math.hypot(edx,edy);
    if(action==="fire"&&this.player.ammo>0){
      this.player.ammo=clamp(this.player.ammo-.08,0,1);
      const angle=Math.atan2(edy,edx);let da=angle-this.player.angle;while(da>Math.PI)da-=Math.PI*2;while(da<-Math.PI)da+=Math.PI*2;
      if(Math.abs(da)<.23&&ed<.72){const hit=.17+.08*(1-ed/.72);this.enemy.health-=hit;reward+=.45;if(this.enemy.health<=0){reward+=1.25;this.respawnEnemy()}}
    }

    const toward=Math.atan2(this.player.y-this.enemy.y,this.player.x-this.enemy.x),enemySpeed=.012;
    this.enemy.x=clamp(this.enemy.x+Math.cos(toward)*enemySpeed,.02,.98);this.enemy.y=clamp(this.enemy.y+Math.sin(toward)*enemySpeed,.02,.98);
    const nd=Math.hypot(this.enemy.x-this.player.x,this.enemy.y-this.player.y),cover=this.coverAt(this.player.x,this.player.y);
    if(nd<.34&&this.rng()<(0.43*(1-cover*.75))){this.lastDamage=.035+.055*(1-nd/.34);this.player.health=clamp(this.player.health-this.lastDamage,0,1);reward-=this.lastDamage*2.2}

    const gd=Math.hypot(this.goal.x-this.player.x,this.goal.y-this.player.y);
    if(gd<.08){reward+=1.7;this.goal={x:.1+.8*this.rng(),y:.1+.8*this.rng()};this.player.ammo=clamp(this.player.ammo+.35,0,1);this.player.health=clamp(this.player.health+.2,0,1)}
    if(this.t%70===0)this.player.ammo=clamp(this.player.ammo+.12,0,1);
    if(this.player.health<=0||this.t>=900){this.done=true;reward-=1}
    this.lastReward=reward;return{observation:this.observe(),reward,done:this.done};
  }
  respawnEnemy(){this.enemy={x:.15+.7*this.rng(),y:.15+.7*this.rng(),health:1}}
}
