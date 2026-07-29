stage6bg_background=Class(object)

function stage6bg_background:init()
	--
	background.init(self,false)
	stg6bg=self
	--resource
	LoadImageFromFile('stage05a','assets/background/dld06bg/stage05a.png')
	LoadImageFromFile('stage05b','assets/background/dld06bg/stage05b.png')	
	LoadImageFromFile('stage06d','assets/background/dld06bg/stage06d.png')		
	LoadImageFromFile('stage06c','assets/background/dld06bg/stage06c.png')
	
	LoadFX('sr_dld_wave','shader/dld_wave.hlsl')
	LoadFX('sr_dld_core','shader/dld_core.hlsl')
	
	SetImageState('stage06d','add+add',Color(64,255,210,0))
	--set 3d camera and fog
-----------------------------------------
------在此处设置背景行为
	self.flag=1			-- 为0时，不执行Intro部分，为1时，执行Intro部分
	--若要使用Outro,执行函数：stage6bg_background.outro(self,_time,_coretime);stage6bg_background.outro(dldstage6bg,90,60)
	self.IntroTime=240	--	Intro渐变时间
	
	--若不需要逐渐远离核心的效果，将self.coreratespeed设为0
	
	--self.OutroTime=90		--	Outro渐变时间
	--self.CoreTime=60	-- Core渐变时间
-----------------------------------------
	if self.flag~=1 then self.fog2=6.3 self.fogcolor=1
	elseif self.flag==1 then self.fog2=1.5 self.fogcolor=0 end 
	self.fog1max=1.3
	self.fog2max=6.3
	self.fogrmax=240
	self.foggmax=100
	self.fogbmax=0
	self.introeye2=2.0
	self.outroeye2=4.0
	self.introat2=1.3
	self.outroat2=0.8
-----------------------------------------
	self.corefovyinit=1.0
	self.corefovyadd=0.6
	self.corefog1init=0
	self.corefog2init=0.7
	self.corefog1=1.3
	self.corefog2=6.3
	self.coreangle=0
	self.coreomega=0.1
	self.corerate=1
	self.coreratespeed=-0.0001
	self.coreminrate=0.5

-----------------------------------------	
	dldstage6bg=self
	CreateRenderTarget("sr_dld_wave_target")
	CreateRenderTarget("sr_dld_core_target")
	Set3D('eye',0,self.introeye2,1)
	Set3D('at',1,self.introat2,0)
	Set3D('up',0,1,0)

	Set3D('z',1,100)
	--Set3D('fovy',0.8)
	Set3D('fovy',1.0)
	Set3D('fog',self.fog1max,self.fog2,Color(255,self.fogcolor*self.fogrmax,self.fogcolor*self.foggmax,self.fogcolor*self.fogbmax))

	self.fog_a=0.1
	self.fog_b=7.5
	self.fog_Fa=6.7
	self.fog_Fb=11.8
	-----
	self.eye_a=0
	self.eye_b=5
	self.eye_c=-3.7

	self.at_a=0
	self.at_b=-0.1
	self.at_c=-0.6
	-----
	self.x=0
	self.z=0
	self.alpha=0
	self.cover={}
	self.cover.alpha=0
	self.cover.r=0
	self.cover.g=0
	self.cover.b=0
	
	self.rate=0.3
	--self.rate=0.6
	self.speedx=-0.010*self.rate
	self.speedz=0.025*self.rate
	self.swingmax=20
	self.swimgspeed=0.4
	--self.swingmax=30
	--self.swimgspeed=0.4
	self.swimg=0
	self.swingcount=0
	self.AxisTilt=0.55
	------------------------------------
	self.list2={}
	self.list2_start=1
	self.list2_end=0
	self.imgs2={'stage06d'}
	
	self.interval2=1.0
	self.acc2=self.interval2
	
	self.X_Base=0.0
	self.Y_Base=0.0
	self.X_Max=1
	self.Y_Max=1
	self.speed2=-0.012
	self.door_Z_Init=-2
	self.Z_Init_Offset=-0.25
	self.Z_Speed=0.02
	self.Z_Add=self.Z_Speed*0.1
	self.imgsize2=0.05
	----------------------------------------------------	
	
------------------------------------

	self.list3={}
	self.list3_start=1
	self.list3_end=0
	self.imgs3={'stage06d'}
	
	self.interval3=0.3
	self.acc3=self.interval3

	self.X_Base_3=-1.5
	self.X_Max_3=-1.2
	self.speed=-0.01
	self.X_Speed_3_Max=self.speed*0.01
	self.Y_Init_3=0.5
	self.Y_Speed_3_Min=0.008
	self.Y_Speed_3_Max=0.025
	self.Z_Init_3=-2.0
	self.Z_Init_Offset_3=0.2
	self.Z_Speed2=0.005
	self.imgsize3=0.03
	----------------------------------------------------
end

function stage6bg_background:frame()
	task.Do(self)
	self.x=self.x+self.speedx
	self.z=self.z+self.speedz
	self.swingcount=self.swingcount+self.swimgspeed
	self.swimg=self.swingmax*sin(self.swingcount)
	
	--if self.timer>=360 and self.timer<361 then stage6bg_background.outro(self,self.OutroTime,self.CoreTime) end
	
	if self.flag==1 then
		if self.timer<=self.IntroTime then
		self.fog2=self.fog1max+(self.fog2max-self.fog1max)*self.timer/self.IntroTime
		self.fogcolor=self.timer/self.IntroTime
		Set3D('fog',self.fog1max,self.fog2,Color(255,self.fogcolor*self.fogrmax,self.fogcolor*self.foggmax,self.fogcolor*self.fogbmax))
		end
	end
	if self.flag==0 or (self.flag==1 and self.timer>self.IntroTime/2) then
			self.acc3=self.acc3-self.speed
			if self.acc3>self.interval3 then
					self.acc3=self.acc3-self.interval3
					for i=1,6 do
						self.list3_end=self.list3_end+1
						self.list3[self.list3_end]=
						{
							1,
							(self.X_Base_3+ran:Float(0,1)*self.X_Max_3)*ran:Sign(),
							self.Y_Init_3,
							self.Z_Init_3+self.Z_Init_Offset_3*i,
							ran:Float(-self.X_Speed_3_Max,self.X_Speed_3_Max),
							ran:Float(self.Y_Speed_3_Min,self.Y_Speed_3_Max),
							ran:Float(0,360),
							ran:Float(0.5,2)*ran:Sign(),
						}
					end
			end

			for i=self.list3_start,self.list3_end do
				self.list3[i][2]=self.list3[i][2]+self.list3[i][5]
				self.list3[i][3]=self.list3[i][3]+self.list3[i][6]
				self.list3[i][4]=self.list3[i][4]+self.Z_Speed2
				self.list3[i][7]=self.list3[i][7]+self.list3[i][8]
			end
			
			while next(self.list3)~=nil do
				if self.list3[self.list3_start][4]<-2 then
					self.list3[self.list3_start]=nil
					self.list3_start=self.list3_start+1
				else
					break
				end
			end
	end
	if self.flag==2 then
		self.coreangle=self.coreangle+self.coreomega
		self.corerate=max(self.corerate+self.coreratespeed,self.coreminrate)
		self.acc2=self.acc2-self.speed2
		if self.acc2>self.interval2 then
				self.acc2=self.acc2-self.interval2
				for i=1,6 do
					self.list2_end=self.list2_end+1
					local angle=ran:Float(0,360)
					self.list2[self.list2_end]=
					{
						1,
						self.X_Base+ran:Float(1,2)*cos(angle),
						self.Y_Base+ran:Float(1,2)*sin(angle),
						self.door_Z_Init+self.Z_Init_Offset*i,
						ran:Float(0,self.Z_Add*1),
						ran:Float(0,360),
						ran:Float(1,4)*ran:Sign(),
					}
				end
		end

		for i=self.list2_start,self.list2_end do
			self.list2[i][4]=self.list2[i][4]+self.list2[i][5]+self.Z_Speed
			self.list2[i][6]=self.list2[i][6]+self.list2[i][7]
		end
		
		while next(self.list2)~=nil do
			if self.list2[self.list2_start][4]>6 then
				self.list2[self.list2_start]=nil
				self.list2_start=self.list2_start+1
			else
				break
		end
	end
	--------------------------------------------------
	end
end

function stage6bg_background:render()
	SetViewMode'3d'
	local showboss = IsValid(_boss)

	if showboss then
        PostEffectCapture()
		RenderClear(lstg.view3d.fog[3])
    end
	if self.flag~=2 then
		PushRenderTarget("sr_dld_wave_target")
		local dz=6-math.mod(self.z,6)
		local dx=6-math.mod(self.x,6)
		stage6bg_background.renderground(dx,dz,self)
		PopRenderTarget()
		local cx,cy = WorldToScreen(0,0)
		local cx1=cx * screen.scale
		local cy1=(screen.height - cy) * screen.scale
		PostEffect("sr_dld_wave", "sr_dld_wave_target", 6, "mul+alpha", {
			{ cx1, cy1, self.swimg, self.AxisTilt },
		}, {})
	else
		PushRenderTarget("sr_dld_core_target")
		stage6bg_background.renderground2(self)
		PopRenderTarget()
		local cx,cy = WorldToScreen(0,0)
		local cx1=cx * screen.scale
		local cy1=(screen.height - cy) * screen.scale
		PostEffect("sr_dld_core", "sr_dld_core_target", 6, "", {
			{ cx1, cy1, 0, 0 },
			{ 255 * 100 * lstg.scale_3d * self.corerate, 192 * lstg.scale_3d, self.timer, 0 },
		}, {})
	end
	


	if showboss then
		local x,y = WorldToScreen(_boss.x,_boss.y)
		local x1 = x * screen.scale
		local y1 = (screen.height - y) * screen.scale
		local fxr = _boss.fxr or 163
		local fxg = _boss.fxg or 73
		local fxb = _boss.fxb or 164
		PostEffectApply("boss_distortion", "", {
			centerX = x1,
			centerY = y1,
			size = _boss.aura_alpha*200*lstg.scale_3d,
			color = Color(125,fxr,fxg,fxb),
			colorsize = _boss.aura_alpha*200*lstg.scale_3d,
			arg=1500*_boss.aura_alpha/128*lstg.scale_3d,
			timer = self.timer
        })
	end

	SetViewMode'world'
end

function stage6bg_background.outro(self,_time,_coretime)
	task.Clear(self)
	local initeye2=self.introeye2
	local initr=self.fogrmax
	local initg=self.foggmax
	local initb=self.fogbmax
	local initfog2=self.fog2
	local introat2=self.introat2
	local _time2=_time*0.75			--可能对颜色渐出和视角变换时间分离
	task.New(self,function()
		for j=1,_time2 do
			self.fogrmax=initr+(255-initr)*j/_time2
			self.foggmax=initg+(255-initg)*j/_time2
			self.fogbmax=initb+(255-initb)*j/_time2
			task.Wait(1)
		end
	end)
	
	task.New(self,function()
		for i=1,_time do
			Set3D('eye',0,initeye2+(self.outroeye2-initeye2)*(i/_time),1)
			Set3D('at',1,introat2+(self.outroat2-introat2)*(i/_time),0)
			self.fog2=initfog2+(self.fog1max+0.1-initfog2)*i/_time
			Set3D('fog',self.fog1max,self.fog2,Color(255,self.fogcolor*self.fogrmax,self.fogcolor*self.foggmax,self.fogcolor*self.fogbmax))
			task.Wait(1)
		end
		self.flag=2
		initeye2=self.coreeye2init
		local initfog1=self.corefog1init
		initfog2=self.corefog2init		
		Set3D('fog',initfog1,initfog2,Color(255,255,255,255))

		Set3D('eye',0,0,1.1)
		Set3D('at',0,0,1)
		Set3D('up',0,1,0)
		task.New(self,function()
			for k=1/_coretime,1+0.5/_coretime,1/_coretime do
				Set3D('fog',initfog1+(self.corefog1-initfog1)*k,initfog2+(self.corefog2-initfog2)*k,Color(255,255,255,255))
				coroutine.yield()
			end
		end)
		for s=1/_coretime,1+0.5/_coretime,1/_coretime do
			s=s*2-s*s
			Set3D('fovy',self.corefovyinit+self.corefovyadd*s)
			coroutine.yield()
		end 
	end)

end

function stage6bg_background.renderground(x,z,self)
	local r=1.2
	local x1=0-x
	local x2=r-x
	local z1=r-z
	local z2=0-z
	local t1=8
	local t2=24

	
	for j = -t1,t1 do
		for i = 0,t2 do


			Render4V('stage05b',x1+r*i,0,r*j+z1,
								x2+r*i,0,r*j+z1,
								x2+r*i,0,r*j+z2,
								x1+r*i,0,r*j+z2)		
			Render4V('stage05a',x1+r*i,0,r*j+z1,
								x2+r*i,0,r*j+z1,
								x2+r*i,0,r*j+z2,
								x1+r*i,0,r*j+z2)	
							
								
		end
	end

	for i=self.list3_end,self.list3_start,-1 do		
		local p=self.list3[i]
		local I=p[1]
		local X=p[2]
		local Y=p[3]
		local Z=p[4]
		local R=p[7]
		
		local H=self.imgsize3
		local W=self.imgsize3
		
		local Pt_0={}
		for i=1,4 do
			Pt_0[i]={X+W/2*cos(i*90+R),Y+H/2*sin(i*90+R),Z}
		end
		
		local Pt={}
		for i=1,4 do
			Pt[i]={X+W/2*cos(i*90+R),Y+H/2*sin(i*90+R),Z}
		end

	
		
		Render4V(self.imgs3[I],
			Pt[1][1],Pt[1][2],Pt[1][3],
			Pt[2][1],Pt[2][2],Pt[2][3],
			Pt[3][1],Pt[3][2],Pt[3][3],
			Pt[4][1],Pt[4][2],Pt[4][3]
			)
	end
end

function stage6bg_background.renderground2(self)
	local r=self.corerate
	local ang=self.coreangle
	local x1=-2*r
	local z1=2*r
	local x2=2*r
	local z2=2*r
	local x3=2*r
	local z3=-2*r
	local x4=-2*r
	local z4=-2*r
	local xx1=x1*cos(ang)-z1*sin(ang)
	local zz1=x1*sin(ang)+z1*cos(ang)
	local xx2=x2*cos(ang)-z2*sin(ang)
	local zz2=x2*sin(ang)+z2*cos(ang)
	local xx3=x3*cos(ang)-z3*sin(ang)
	local zz3=x3*sin(ang)+z3*cos(ang)
	local xx4=x4*cos(ang)-z4*sin(ang)
	local zz4=x4*sin(ang)+z4*cos(ang)
	Render4V('stage06c',xx1,zz1,0,
						xx2,zz2,0,
						xx3,zz3,0,
						xx4,zz4,0)	
						
	for i=self.list2_end,self.list2_start,-1 do		
		local p=self.list2[i]
		local I=p[1]
		local X=p[2]
		local Y=p[3]
		local Z=p[4]
		local R=p[6]
		
		local H=self.imgsize2
		local W=self.imgsize2
		
		local Pt_0={}
		for i=1,4 do
			Pt_0[i]={X+W/2*cos(i*90+R),Y+H/2*sin(i*90+R),Z}
		end
		
		local Pt={}
		for i=1,4 do
			Pt[i]={X+W/2*cos(i*90+R),Y+H/2*sin(i*90+R),Z}
		end
		
		Render4V(self.imgs2[I],
			Pt[1][1],Pt[1][2],Pt[1][3],
			Pt[2][1],Pt[2][2],Pt[2][3],
			Pt[3][1],Pt[3][2],Pt[3][3],
			Pt[4][1],Pt[4][2],Pt[4][3]
			)
	end
end
