sakuya_player=Class(player_class)

function sakuya_player:init(slot)
	LoadTexture('sakuya_player','assets/player/sakuya/sakuya.png')
	LoadTexture('sakuya_player2p','assets/player/sakuya/sakuya_2p.png')
	LoadImageGroup('sakuya_player','sakuya_player',0,0,32,48,8,3,1,1)
	LoadImageGroup('sakuya_player2p','sakuya_player2p',0,0,32,48,8,3,1,1)
	LoadImageGroup('sakuya_support','sakuya_player',128,144,16,16,8,2)
	LoadImage('sakuya_knife_blue','sakuya_player',0,160,32,16,16,16)
	LoadAnimation('sakuya_knife_blue_ef','sakuya_player',32,160,32,16,3,1,4)
	
	LoadImage('sakuya_bigknife_red','sakuya_player',0,192,64,16,16,16)
	LoadAnimation('sakuya_bigknife_red_ef','sakuya_player',0,144,32,16,4,1,4)
	
	LoadPS('sakuya_blood','assets/player/sakuya/sakuya_blood.psi','parimg1')
	
	SetImageState('sakuya_bigknife_red','',Color(0xA0FFFFFF))
	
	LoadSound('sakuya_tick','assets/player/sakuya/sakuya_tick.wav')
	
	player_class.init(self,slot)
	
	self.grazer.nopause=true
	
	self.name='Sakuya'
	self.nopause=true
	self.hspeed=4.5
	self.lspeed=2
	self.A=1 self.B=1
	self.imgs={}
	local first_player=jstg and jstg.players and jstg.players[1]
	if slot==2 and first_player and first_player.name==self.name then
		for i=1,24 do self.imgs[i]='sakuya_player2p'..i end
	else
		for i=1,24 do self.imgs[i]='sakuya_player'..i end
	end
	self.slist=
	{
		{-48,-8,-24,-16},
		{48,-8,24,-16},
		{-24,-30,-12,-30},
		{24,-30,12,-30}
	}
	self.dalist={6,-6,2,-2}
	self.dcolor=Color(180,200,200,200)
end

function sakuya_player:frame()
	player_class.frame(self)
end

function sakuya_player:shoot()
	PlaySound('plst00',0.3,self.x/1024)
	self.nextshoot=4
	if combo.on then 
		SetImageState('sakuya_bigknife_red','mul+add',Color(0xC0FFFFFF))
		for i=1,4 do 
			if self.sp[i] and self.sp[i][3]>0.5 then
				New(sakuya_bullet_main,'sakuya_bigknife_red',self.supportx+self.sp[i][1],self.supporty+self.sp[i][2],24,90,2)
			end
		end
	else 
		SetImageState('sakuya_bigknife_red','',Color(0xA0FFFFFF))
		local daf = 1
		if self.slow==1 then daf=0 end
		for i=1,4 do 
			if self.sp[i] and self.sp[i][3]>0.5 then
				for j=-2,2,4 do 
					New(sakuya_knife,'sakuya_knife_blue',self.supportx+self.sp[i][1]-j,self.supporty+self.sp[i][2],24,90+self.dalist[i]*daf+j*daf,0.5) 
				end
			end
		end 
	end
	New(sakuya_bullet_main,'sakuya_bigknife_red',self.x+6,self.y,24,90,2)
	New(sakuya_bullet_main,'sakuya_bigknife_red',self.x-6,self.y,24,90,2)
end

function sakuya_player:render()
	player_class.render(self)
	local t=int((self.timer/3)%16)+1
	for i=1,4 do
		if self.sp[i] and self.sp[i][3]>0.5 then
			Render('sakuya_support'..t,self.supportx+self.sp[i][1],self.supporty+self.sp[i][2])
		end
	end
end

sakuya_bullet_main=Class(player_bullet_straight)
function sakuya_bullet_main:init(img,x,y,v,angle,dmg)
	player_bullet_straight.init(self,img,x,y,v,angle,dmg)
	if combo.on then 
		self.a=self.a*1.5
		self.b=self.b*1.5
		self.hscale=1.5
		self.vscale=1.5
	end
end

function sakuya_bullet_main:kill()
	New(sakuya_knife_ef,self.x,self.y,self.rot,3,self.img..'_ef')
	New(sakuya_blood_ef,self.x,self.y,self.rot,4,12)
	if combo.on then combo.hit(0.03,self) end
end

sakuya_knife=Class(player_bullet_straight)

function sakuya_knife:kill()
	New(sakuya_knife_ef,self.x,self.y,self.rot,3,self.img..'_ef')
	New(sakuya_blood_ef,self.x,self.y,self.rot,4,12)
end

sakuya_knife_ef=Class(object)

function sakuya_knife_ef:init(x,y,rot,v,img)
	self.x=x
	self.y=y
	self.rot=rot
	self.vx=v*cos(rot)
	self.vy=v*sin(rot)
	self.img=img
	self.group=GROUP_GHOST
	self.layer=LAYER_PLAYER_BULLET+50
end

function sakuya_knife_ef:frame()
	if self.timer==12 then Del(self) end
end

function sakuya_knife_ef:render()
	SetAnimationState(self.img,'',Color(128-10*self.timer,255,255,255))
	object.render(self)
end

sakuya_blood_ef=Class(object)

function sakuya_blood_ef:init(x,y,a,t1,t2)
	self.x=x
	self.y=y
	self.rot=a
	self.group=GROUP_GHOST
	self.layer=LAYER_PLAYER_BULLET+60
	self.stoptime=t1
	self.deathtime=t2
	self.img='sakuya_blood'
end

function sakuya_blood_ef:frame()
	if self.timer==self.stoptime then
		ParticleStop(self)
	end
	if self.timer==self.deathtime then
		Del(self)
	end
end
