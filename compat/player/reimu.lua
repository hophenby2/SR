reimu_player=Class(player_class)

function reimu_player:init(slot)
	LoadTexture('reimu_player','assets/player/reimu/reimu.png')
	LoadTexture('reimu_player2p','assets/player/reimu/reimu_2p.png')
	LoadTexture('reimu_kekkai','assets/player/reimu/reimu_kekkai.png')
	LoadTexture('reimu_orange_ef2','assets/player/reimu/reimu_orange_eff.png')
	LoadAnimation('reimu_bullet_orange_ef2','reimu_orange_ef2',0,0,64,16,1,9,1)
	SetAnimationCenter('reimu_bullet_orange_ef2',0,8)
	LoadImageGroup('reimu_player','reimu_player',0,0,32,48,8,3,0.5,0.5)
	LoadImageGroup('reimu_player2p','reimu_player2p',0,0,32,48,8,3,0.5,0.5)
	
	LoadImage('reimu_bullet_red','reimu_player',192,160,64,16,16,16)
	SetImageState('reimu_bullet_red','',Color(0xA0FFFFFF))
	SetImageCenter('reimu_bullet_red',56,8)
	LoadAnimation('reimu_bullet_red_ef','reimu_player',0,144,16,16,4,1,4)
	SetAnimationState('reimu_bullet_red_ef','mul+add',Color(0xA0FFFFFF))
	
	LoadImage('reimu_bullet_blue','reimu_player',0,160,16,16,16,16)
	SetImageState('reimu_bullet_blue','',Color(0xFFFFE0FF))
	LoadAnimation('reimu_bullet_blue_ef','reimu_player',0,160,16,16,4,1,4)
	SetAnimationState('reimu_bullet_blue_ef','mul+add',Color(0xA0FFFFFF))
	
	LoadImage('reimu_support','reimu_player',64,144,16,16)
	LoadImage('reimu_bullet_ef_img','reimu_player',48,144,16,16)
	LoadPS('reimu_bullet_ef','assets/player/reimu/reimu_bullet_ef.psi','reimu_bullet_ef_img')
	-----------------------------------------
	LoadImage('reimu_bullet_orange','reimu_player',64,176,64,16,64,16)
	SetImageState('reimu_bullet_orange','',Color(0x80FFFFFF))
	SetImageCenter('reimu_bullet_orange',32,8)
	LoadImage('reimu_bullet_orange_ef','reimu_player',64,176,64,16,64,16)
	SetImageState('reimu_bullet_orange_ef','',Color(0x80FFFFFF))
	SetImageCenter('reimu_bullet_orange_ef',32,8)
	-----------------------------------------
	player_class.init(self,slot)
	self.name='Reimu'
	self.hspeed=4
	self.imgs={}
	self.A=0.5 self.B=0.5
	local first_player=jstg and jstg.players and jstg.players[1]
	if slot==2 and first_player and first_player.name==self.name then
		for i=1,24 do self.imgs[i]='reimu_player2p'..i end
	else
		for i=1,24 do self.imgs[i]='reimu_player'..i end
	end

	self.slist={
		{-18,30,-12,30},
		{18,30,12,30},
		{-36,0,-24,0},
		{36,0,24,0},
	}
	self.anglelist={100,80,110,70}
	self.dcolor=Color(180,240,75,75)
end
-------------------------------------------------------
function reimu_player:shoot()
	PlaySound('plst00',0.3,self.x/1024)
	self.nextshoot=4
	if combo.on then
		SetImageState('reimu_bullet_red','mul+add',Color(0xFFFFFFC0))
		for i=1,4 do
			if self.sp[i] and self.sp[i][3]>0.5 then
				New(reimu_bullet_main,'reimu_bullet_red',self.supportx+self.sp[i][1],self.supporty+self.sp[i][2],24,90,2)
			end
		end
	else
		SetImageState('reimu_bullet_red','',Color(0xA0FFFFFF))
		if self.slow==1 then
			for i=1,4 do
				if self.sp[i] and self.sp[i][3]>0.5 then
					New(reimu_bullet_orange,'reimu_bullet_orange',self.supportx+self.sp[i][1]-3,self.supporty+self.sp[i][2],24,90,0.5)
					New(reimu_bullet_orange,'reimu_bullet_orange',self.supportx+self.sp[i][1]+3,self.supporty+self.sp[i][2],24,90,0.5)
				end
			end
		else
			for i=1,4 do
				if self.sp[i] and self.sp[i][3]>0.5 then
					New(reimu_bullet_blue,'reimu_bullet_blue',self.supportx+self.sp[i][1]-3,self.supporty+self.sp[i][2],24,self.anglelist[i]+3,0.5)
					New(reimu_bullet_blue,'reimu_bullet_blue',self.supportx+self.sp[i][1]+3,self.supporty+self.sp[i][2],24,self.anglelist[i]-3,0.5)
				end
			end
		end
	end
	New(reimu_bullet_main,'reimu_bullet_red',self.x+10,self.y,24,90,2)
	New(reimu_bullet_main,'reimu_bullet_red',self.x-10,self.y,24,90,2)
end
-------------------------------------------------------

-------------------------------------------------------
function reimu_player:render()
	for i=1,4 do
		if self.sp[i] and self.sp[i][3]>0.5 then
			Render('reimu_support',self.supportx+self.sp[i][1],self.supporty+self.sp[i][2],self.timer*3)
		end
	end
	player_class.render(self)
end
-------------------------------------------------------

-------------------------------------------------------
reimu_bullet_main=Class(player_bullet_straight)

function reimu_bullet_main:init(img,x,y,v,angle,dmg)
	player_bullet_straight.init(self,img,x,y,v,angle,dmg)
	if combo.on then 
		self.a=self.a*1.5
		self.b=self.b*1.5
		self.hscale=1.5
		self.vscale=1.5
	end
end

function reimu_bullet_main:kill()
	New(reimu_bullet_red_ef,self.x,self.y,self.rot+180)
	if combo.on then combo.hit(0.03,self) end
end

-------------------------------------------------------
reimu_bullet_red_ef=Class(object)

function reimu_bullet_red_ef:init(x,y)
	self.x=x self.y=y self.rot=90 self.img='reimu_bullet_red_ef' self.layer=LAYER_PLAYER_BULLET+50 self.group=GROUP_GHOST
	self.vy=2.25
end
function reimu_bullet_red_ef:frame()
	if self.timer>14 then self.y=600 Del(self) end
end
-------------------------------------------------------
reimu_bullet_orange=Class(player_bullet_straight)

function reimu_bullet_orange:kill()
	New(reimu_bullet_orange_ef,self.x,self.y,self.rot+180+ran:Float(-15,15))
	New(reimu_bullet_orange_ef2,self.x,self.y)
end
-------------------------------------------------------
reimu_bullet_blue=Class(player_bullet_straight)

function reimu_bullet_blue:kill()
	New(reimu_bullet_blue_ef,self.x,self.y,self.rot)
end
-------------------------------------------------------
reimu_bullet_blue_ef=Class(object)

function reimu_bullet_blue_ef:init(x,y,rot)
	self.x=x self.y=y self.rot=rot self.img='reimu_bullet_blue_ef' self.layer=LAYER_PLAYER_BULLET+50 self.group=GROUP_GHOST
	self.vx=1*cos(rot) self.vy=1*sin(rot)
end

function reimu_bullet_blue_ef:frame()
	if self.timer>14 then Del(self) end
end
-------------------------------------------------------

-------------------------------------------------------
reimu_bullet_ef=Class(object)

function reimu_bullet_ef:init(x,y,rot)
	self.x=x self.y=y self.rot=rot self.img='reimu_bullet_ef' self.layer=LAYER_PLAYER_BULLET+50 self.group=GROUP_GHOST
end

function reimu_bullet_ef:frame()
	if self.timer==4 then ParticleStop(self) end
	if self.timer==30 then Del(self) end
end
-------------------------------------------------------
reimu_bullet_orange_ef=Class(object)

function reimu_bullet_orange_ef:init(x,y,rot)
	self.x=x self.y=y+32 self.rot=rot self.img='reimu_bullet_orange_ef' self.layer=LAYER_PLAYER_BULLET+50 self.group=GROUP_GHOST self.vy=2
	self.hscale=ran:Float(1.4,1.6)
end

function reimu_bullet_orange_ef:frame()
	SetImgState(self,'mul+add',255-255*self.timer/16,255,255,255)
	if self.timer>15 then self.x=600 Del(self) end
end
-------------------------------------------------------
reimu_bullet_orange_ef2=Class(object)

function reimu_bullet_orange_ef2:init(x,y)
	self.x=x self.y=y+32 self.rot=-90+ran:Float(-10,10) self.img='reimu_bullet_orange_ef2' self.layer=LAYER_PLAYER_BULLET+50 self.group=GROUP_GHOST
	self.hscale=ran:Float(1.5,1.8) self.vscale=1.5
end

function reimu_bullet_orange_ef2:frame()
	SetImgState(self,'mul+add',255,255,155,155)
	if self.timer>=9 then self.x=600 Del(self) end
end
-------------------------------------------------------
