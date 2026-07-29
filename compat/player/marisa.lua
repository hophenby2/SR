marisa_player=Class(player_class)

function marisa_player:init(slot)
	LoadTexture('marisa_player','assets/player/marisa/marisa.png')
	LoadTexture('marisa_player2p','assets/player/marisa/marisa_2p.png')
	LoadTexture('MarisaLaser','assets/player/marisa/MarisaLaser.png')
	LoadImageGroup('marisa_player','marisa_player',0,0,32,48,8,3,1,1)
	LoadImageGroup('marisa_player2p','marisa_player2p',0,0,32,48,8,3,1,1)
	LoadImage('marisa_bullet','marisa_player',0,144,32,16,16,16)
	LoadAnimation('marisa_bullet_ef','marisa_player',0,144,32,16,4,1,4)
	
	LoadImage('marisa_support','marisa_player',144,144,16,16)
	LoadImage('marisa_laser_light','marisa_player',224,224,32,32)
	SetImageState('marisa_laser_light','mul+add',Color(0xFFFFFFFF))
	---------------
	LoadPS('marisa_laser_top_red','assets/player/marisa/marisa_laser_top_red.psi','parimg11')
	LoadPS('marisa_laser_top_blue','assets/player/marisa/marisa_laser_top_blue.psi','parimg11')
	
	player_class.init(self,slot)
	self.name='Marisa'
	self.imgs={}
	self.A=1 self.B=1
	local first_player=jstg and jstg.players and jstg.players[1]
	if slot==2 and first_player and first_player.name==self.name then
		for i=1,24 do self.imgs[i]='marisa_player2p'..i end
	else
		for i=1,24 do self.imgs[i]='marisa_player'..i end
	end
	self.hspeed=5
	self.offset={600,600,600,600}
	self.laser_hit={0,0,0,0}
	
	self.slist={
		{-18,	30,	-12,	30},
		{18,	30,	12,		30},
		{-36,	0,	-24,	0},
		{36,	0,	24,		0}
	}
	self.dcolor=Color(180,120,75,240)
end

function marisa_player:frame()
	if KeyIsDown'shoot' and not combo.on then 
		for i=1,4 do 
			self.laser_hit[i]=self.laser_hit[i]-1
		end
	else
		for i=1,4 do 
			self.laser_hit[i]=0
		end
	end
	
	if self.slow==0 then 
		SetImageState('marisa_laser_light','mul+add',Color(0xFFC0C0FF))
	else
		SetImageState('marisa_laser_light','mul+add',Color(0xFFFFC0C0))
	end
	
	for i=1,4 do 
		self.offset[i]=600 
	end
	
	player_class.frame(self)
end

function marisa_player:shoot()
		if combo.on then 
			SetImageState('marisa_bullet','mul+add',Color(0x80FFFFC0))
			if self.timer%4==0 then 
				for i=1,4 do 
					New(marisa_bullet_main,'marisa_bullet',self.supportx+self.sp[i][1],self.supporty+self.sp[i][2],24,90,2)
				end
			end
		else
			SetImageState('marisa_bullet','',Color(0x80FFFFFF))
			if self.timer%12==0 then PlaySound('lazer02',0.025) end
	--		local num=30/(self.support+1)
			for i=1,4 do
				local angle=90
				if self.sp[i] and self.sp[i][3]>0.5 then
					local target=nil
					local x,y=self.supportx+self.sp[i][1],self.supporty+self.sp[i][2]
					for j,o in ObjList(GROUP_ENEMY) do
						if o.colli and IsInLaser(x,y,angle,o,24) then
							local d=Dist(o.x,o.y,x,y)
							if d<self.offset[i] then
								target=o
								self.offset[i]=d
							end
						end
					end
					for j,o in ObjList(GROUP_NONTJT) do
						if o.colli and IsInLaser(x,y,angle,o,24) then
							local d=Dist(o.x,o.y,x,y)
							if d<self.offset[i] then
								target=o
								self.offset[i]=d
							end
						end
					end
					if target then
						self.laser_hit[i]=4
						self.offset[i]=max(0,self.offset[i]-target.b)
						if target.class.base.take_damage then 
							target.class.base.take_damage(target,0.25)
							if self.timer%2==0 then lstg.var.score=lstg.var.score+100 end
						end
						if target.hp>target.maxhp*0.1 then
							PlaySound('damage00',0.3,target.x/1024)
						else
							PlaySound('damage01',0.6,target.x/1024)
						end
					end
					if self.timer%4==0 then 
						if self.slow==1 then 
							New(marisa_laser_hit,x+self.offset[i]*cos(angle),y+self.offset[i]*sin(angle),'marisa_laser_top_red')
						elseif target then
							New(marisa_laser_hit,x+self.offset[i]*cos(angle),y+self.offset[i]*sin(angle),'marisa_laser_top_blue')
						end
					end
				end
			end
		end
		if self.timer%4==0 then
			PlaySound('plst00',0.15,self.x/1024)
			New(marisa_bullet_main,'marisa_bullet',self.x+6,self.y,24,90,2)
			New(marisa_bullet_main,'marisa_bullet',self.x-6,self.y,24,90,2)
		end
end

function marisa_player:render()
	local sz=1.2+0.1*sin(self.timer*0.2)
	--support
	SetImageState('marisa_support','',Color(0xFFFFFFFF))
	for i=1,4 do if self.sp[i] then
		Render('marisa_support',self.supportx+self.sp[i][1],self.supporty+self.sp[i][2],0,self.sp[i][3],1)
	end end
	--support deco
	SetImageState('marisa_support','',Color(0x80FFFFFF))
	for i=1,4 do if self.sp[i] then
		Render('marisa_support',self.supportx+self.sp[i][1],self.supporty+self.sp[i][2],0,self.sp[i][3]*sz,sz)
	end end
	if self.fire==1 and self.nextshoot<=0 and not combo.on then
		local timer=self.timer*16
		for i=1,4 do
			local angle=90
			if self.sp[i] and self.sp[i][3]>0.5 then
				local x,y=self.supportx+self.sp[i][1],self.supporty+self.sp[i][2]
				if self.slow==0 then 
					if self.laser_hit[i]>0 then
						CreateLaser(x,y,angle,16,timer,Color(0x804040FF),self.offset[i])
					else
						CreateLaser(x,y,angle,16,timer,Color(0x80FFFFFF),self.offset[i])
					end
				else
					if self.laser_hit[i]>0 then
						CreateLaser(x,y,angle,16,timer,Color(0x80FF2010),self.offset[i])
					else
						CreateLaser(x,y,angle,16,timer,Color(0x80FF8040),self.offset[i])
					end
				end
				Render('marisa_laser_light',x,y,self.timer*5,1+0.4*sin(self.timer*45+i*90))
			end
		end
	end
	player_class.render(self)
end

marisa_bullet_main=Class(player_bullet_straight)

function marisa_bullet_main:init(img,x,y,v,angle,dmg)
	player_bullet_straight.init(self,img,x,y,v,angle,dmg)	
	if combo.on then 
		self.a=self.a*1.5
		self.b=self.b*1.5
		self.hscale=1.5
		self.vscale=1.5
	end
end

function marisa_bullet_main:kill()
	New(marisa_bullet_ef,self.x,self.y,self.rot,3)
	New(marisa_bullet_ef,self.x,self.y,self.rot,4)
	New(marisa_bullet_ef,self.x,self.y,self.rot,5)
	if combo.on then combo.hit(0.03,self) end
end

marisa_bullet_ef=Class(object)

function marisa_bullet_ef:init(x,y,rot,v)
	self.x=x
	self.y=y
	self.rot=rot
	self.vx=v*cos(rot)
	self.vy=v*sin(rot)
	self.img='marisa_bullet_ef'
	self.layer=LAYER_PLAYER_BULLET+50
end

function marisa_bullet_ef:frame()
	if self.timer==7 then Del(self) end
end
function marisa_bullet_ef:render()
	SetAnimationState('marisa_bullet_ef','',Color(128-8*self.timer,255,255,255))
	object.render(self)
end

marisa_laser_hit=Class(object)

function marisa_laser_hit:init(x,y,par)
	self.x=x
	self.y=y
	self.group=GROUP_GHOST
	self.layer=LAYER_PLAYER_BULLET+60
	self.img=par
	self.rot=90
end

function marisa_laser_hit:frame()
	if self.timer==15 then
		ParticleStop(self)
	end
	if self.timer==30 then Del(self) end
end


function IsInLaser(x0,y0,a,unit,w)
	local a1=a-Angle(x0,y0,unit.x,unit.y)
	if a%180==90 then
		if abs(unit.x-x0)<((unit.a+unit.b+w)/2) and cos(a1)>=0 then
			return true
		else
			return false
		end
	else
		local A=tan(a)
		local C=y0-A*x0
		if abs(A*unit.x-unit.y+C)/hypot(A,1)<((unit.a+unit.b+w)/2) and cos(a1)>=0 then
			return true
		else
			return false
		end
	end
end

function CreateLaser(x,y,a,w,t,c,offset)
	local width=w/2
	local n=int(offset/256)
	local length=t%256
	local endl=int(offset-n*256)
	for i=1,n do
		RenderTexture('MarisaLaser','mul+add',
				{x+(length+256*(i-1))*cos(a)-width*sin(a),y+(length+256*(i-1))*sin(a)+width*cos(a),0.5,0,         0,c},
				{x+256*i*cos(a)-width*sin(a),             y+256*i*sin(a)+width*cos(a),             0.5,256-length,0,c},
				{x+256*i*cos(a)+width*sin(a),             y+256*i*sin(a)-width*cos(a),             0.5,256-length,16,c},
				{x+(length+256*(i-1))*cos(a)+width*sin(a),y+(length+256*(i-1))*sin(a)-width*cos(a),0.5,0,         16,c}
				)
		RenderTexture('MarisaLaser','mul+add',
				{x+256*(i-1)*cos(a)-width*sin(a),         y+256*(i-1)*sin(a)+width*cos(a),         0.5,256-length,0,c},
				{x+(length+256*(i-1))*cos(a)-width*sin(a),y+(length+256*(i-1))*sin(a)+width*cos(a),0.5,256,       0,c},
				{x+(length+256*(i-1))*cos(a)+width*sin(a),y+(length+256*(i-1))*sin(a)-width*cos(a),0.5,256,       16,c},
				{x+256*(i-1)*cos(a)+width*sin(a),         y+256*(i-1)*sin(a)-width*cos(a),         0.5,256-length,16,c}
				)
	end

	if length<=endl then
		RenderTexture('MarisaLaser','mul+add',
				{x+(length+256*n)*cos(a)-width*sin(a),y+(length+256*n)*sin(a)+width*cos(a),0.5,0,   0,c},
				{x+(256*n+endl)*cos(a)-width*sin(a),  y+(256*n+endl)*sin(a)+width*cos(a),  0.5,endl-length,0,c},
				{x+(256*n+endl)*cos(a)+width*sin(a),  y+(256*n+endl)*sin(a)-width*cos(a),  0.5,endl-length,16,c},
				{x+(length+256*n)*cos(a)+width*sin(a),y+(length+256*n)*sin(a)-width*cos(a),0.5,0,   16,c}
				)
		RenderTexture('MarisaLaser','mul+add',
				{x+256*n*cos(a)-width*sin(a),         y+256*n*sin(a)+width*cos(a),         0.5,256-length,0,c},
				{x+(length+256*n)*cos(a)-width*sin(a),y+(length+256*n)*sin(a)+width*cos(a),0.5,256,       0,c},
				{x+(length+256*n)*cos(a)+width*sin(a),y+(length+256*n)*sin(a)-width*cos(a),0.5,256,       16,c},
				{x+256*n*cos(a)+width*sin(a),         y+256*n*sin(a)-width*cos(a),         0.5,256-length,16,c}
				)
	else
		RenderTexture('MarisaLaser','mul+add',
				{x+256*n*cos(a)-width*sin(a),       y+256*n*sin(a)+width*cos(a),       0.5,256-length,0,c},
				{x+(endl+256*n)*cos(a)-width*sin(a),y+(endl+256*n)*sin(a)+width*cos(a),0.5,endl+256-length,  0,c},
				{x+(endl+256*n)*cos(a)+width*sin(a),y+(endl+256*n)*sin(a)-width*cos(a),0.5,endl+256-length,  16,c},
				{x+256*n*cos(a)+width*sin(a),       y+256*n*sin(a)-width*cos(a),       0.5,256-length,16,c}
				)
	end
end
