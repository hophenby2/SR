stg3bg_background=Class(object)

function stg3bg_background:init()
	--
	background.init(self,false)
	stg3bg=self
	--resource
	LoadImageFromFile('stage03a','assets/background/p3/stage03a.png')
	LoadImageFromFile('stage03b','assets/background/p3/stage03b.png')
	LoadImageFromFile('stage03c','assets/background/p3/stage03c.png')
	LoadImageFromFile('stage03d','assets/background/p3/stage03d.png')
	LoadImageFromFile('stage03e','assets/background/p3/stage03e.png')
	LoadImageFromFile('stage3light','assets/background/p3/stage3light.png')
	--set 3d camera and fog
	Set3D('eye',0,9.3,-6)
	Set3D('at',0,0,-1.1)
	Set3D('up',0,1,0)
	Set3D('z',1,100)
	Set3D('fovy',0.6)
	Set3D('fog',15,24,Color(200,0,0,0))
	--
	self.speed=0.01
	self.z=0
	--
	self.eye={}
	self.eye.x=0; self.eye.y=3.5; self.eye.z=-3; 
	
	self.at={}
	self.at.x=0; self.at.y=2.5; self.at.z=-1
	
	self.up={}
	self.up.x=0; self.up.y=2.5; self.up.z=1
	
	CreateRenderTarget('player_distortion')
end

function stg3bg_background:frame()
	Set3D('eye',self.eye.x,self.eye.y,self.eye.z)
	Set3D('at',self.at.x,self.at.y,self.at.z)
	Set3D('up',self.up.x,self.up.y,self.up.z)
	self.z=self.z+self.speed
end

function stg3bg_background:render()
	SetViewMode'3d'
	local showboss = IsValid(_boss)
	if showboss then
        PostEffectCapture()
    end
	
	PushRenderTarget('player_distortion')
	
	for j=-6,6 do
		local dz=6*j-math.mod(self.z,6)
		stg3bg_background.renderground(dz)
		stg3bg_background.renderwall_left(dz)
		stg3bg_background.renderwall_right(dz)
		stg3bg_background.light_left(self.timer,dz,1)
		stg3bg_background.light_left(self.timer,dz,-1)
	end
	for j=-6,6 do
		local dz=6*j-math.mod(self.z,6)
	--	stg3bg_background.rendertop(dz,0)
		for dx=-5.4,-13.4,-4 do
			stg3bg_background.rendertop(dz,dx)
		end
		for dx=0,8,4 do
			stg3bg_background.rendertop(dz,dx)
		end
	end
	
	PopRenderTarget()
	sr_background.ApplyPlayerDistortion('player_distortion', self.timer)
	
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

function stg3bg_background.renderground(z)
	
	for dx=0,0,4 do
	Render4V('stage03e',
		-1+dx,0,z+1,
		1+dx,0,z+1,
		1+dx,0,z-1,
		-1+dx,0,z-1
	)
	Render4V('stage03a',
		-1+dx,0,z+3,
		1+dx,0,z+3,
		1+dx,0,z+1,
		-1+dx,0,z+1
	)
	Render4V('stage03a',
		-1+dx,0,z-1,
		1+dx,0,z-1,
		1+dx,0,z-3,
		-1+dx,0,z-3
	)
	end
end

function stg3bg_background.renderwall_left(z)
	for dx=0,0,4 do
	Render4V('stage03d',
		-1+dx,1,z+1,
		-1+dx,0,z+1,
		-1+dx,0,z-1,
		-1+dx,1,z-1
	)
	Render4V('stage03c',
		-1+dx,1,z+3,
		-1+dx,0,z+3,
		-1+dx,0,z+1,
		-1+dx,1,z+1
	)
	Render4V('stage03c',
		-1+dx,1,z-1,
		-1+dx,0,z-1,
		-1+dx,0,z-3,
		-1+dx,1,z-3
	)
	end
end

function stg3bg_background.renderwall_right(z)
	for dx=0,0,4 do
	Render4V('stage03d',
		1+dx,1,z+1,
		1+dx,0,z+1,
		1+dx,0,z-1,
		1+dx,1,z-1
	)
	Render4V('stage03c',
		1+dx,1,z+3,
		1+dx,0,z+3,
		1+dx,0,z+1,
		1+dx,1,z+1
	)
	Render4V('stage03c',
		1+dx,1,z-1,
		1+dx,0,z-1,
		1+dx,0,z-3,
		1+dx,1,z-3
	)
	end
end

function stg3bg_background.rendertop(z,dx)
--	for dx=-8,8,4 do
	Render4V('stage03b',
		2.70+dx,1.8,z+1,
		0.70+dx,1.4,z+1,
		0.70+dx,1.4,z-1,
		2.70+dx,1.8,z-1)
	Render4V('stage03b',
		2.70+dx,1.8,z+3,
		0.70+dx,1.4,z+3,
		0.70+dx,1.4,z+1,
		2.70+dx,1.8,z+1
	)
	Render4V('stage03b',
		2.70+dx,1.8,z-1,
		0.70+dx,1.4,z-1,
		0.70+dx,1.4,z-3,
		2.70+dx,1.8,z-3
	)
	Render4V('stage03b',
		-2.70-dx,1.8,z+1,
		-0.70-dx,1.4,z+1,
		-0.70-dx,1.4,z-1,
		-2.70-dx,1.8,z-1
	)
	Render4V('stage03b',
		-2.70-dx,1.8,z+3,
		-0.70-dx,1.4,z+3,
		-0.70-dx,1.4,z+1,
		-2.70-dx,1.8,z+1
	)
	Render4V('stage03b',
		-2.70-dx,1.8,z-1,
		-0.70-dx,1.4,z-1,
		-0.70-dx,1.4,z-3,
		-2.70-dx,1.8,z-3
	)
--	end
end

function stg3bg_background.light_left(timer,z,x)
	SetImageState('stage3light','mul+add',Color(255,255,140,0))
	if timer%1.5==0 then
		Render('stage3light',x,0.9,0,0.3,0.5,z)
	end
	SetImageState('stage3light','mul+add',Color(255,255,80,0))
	if timer%2==0 then
		Render('stage3light',x,0.9,0,0.3,0.5,z)
	end
	SetImageState('stage3light','mul+add',Color(255,255,100,0))
	if timer%1.7==0 then
		Render('stage3light',x,0.9,0,0.3,0.5,z)
	end
	SetImageState('stage3light','mul+add',Color(255,255,60,0))
	if timer%1.8==0 then
		Render('stage3light',x,0.9,0,0.3,0.5,z)
	end
end






