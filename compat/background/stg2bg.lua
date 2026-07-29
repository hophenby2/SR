stg2bg_background=Class(background)

function stg2bg_background:init()
	background.init(self,false)
	stg2bg=self
	LoadImageFromFile('2bg_1a', 'assets/background/p2/2bg_1a.png')
	LoadImageFromFile('2bg_1b', 'assets/background/p2/2bg_1b.png')
	SetImageState('2bg_1b', 'mul+add', Color(0x808080FF))
	
	LoadImageFromFile('2bg_2a', 'assets/background/p2/2bg_2a.png')
	SetImageState('2bg_2a', '', Color(0x8080FF80))
	LoadImageFromFile('2bg_2b', 'assets/background/p2/2bg_2b.png')
	
	
	Set3D('eye',0,6,-6)
	Set3D('at',0,0,20)
	Set3D('up',0,1,2)
	Set3D('z',0,24)
	Set3D('fovy',0.2)
	Set3D('fog',12,40,Color(0x00000000))
	self.yos=0
	self.speed=0.006
	
	CreateRenderTarget('player_distortion')
end

function stg2bg_background:frame()
	self.yos=self.yos+self.speed
	Set3D('eye',0,6,-6)
end

function stg2bg_background:render()
	SetViewMode'3d'
	
	local showboss = IsValid(_boss)
	if showboss then
        PostEffectCapture()
    end
	
	PushRenderTarget('player_distortion')
	
	RenderClear(lstg.view3d.fog[3])
	local y=self.yos%1
	
	--地面
	for i=-2,5 do
		--[横宽，高度，纵深]
		for h = -1,1,2 do 
			Render4V('2bg_1a',4*h,0.2,13*(0-y+i), 4*h,0.2,13*(1-y+i), 0,0,13*(1-y+i), 0,0,13*(-y+i))
		end
	end
	--瘴气
	for i=-2,5 do
		for h = -1,1,2 do 
			local yy = self.yos*(3+0.7*h)%1
			Render4V('2bg_1b',4*h,0.2,13*(0-yy+i), 4*h,0.2,13*(1-yy+i), -3*h,0,13*(1-yy+i), -3*h,0,13*(-yy+i))
		end
	end
	
	--岩壁
	for i=-2,5 do
		for h = -1,1,2 do 
			Render4V('2bg_2b',4*h,12.1,17*(0-y+i), 4*h,12.1,17*(1-y+i), 0,12.1,17*(1-y+i), 0,12.1,17*(-y+i))
		end
	end
	--瘴气
	for i=-2,5 do
		for h = -1,1,2 do 
			local yy = self.yos*(3+0.7*h)%1
			Render4V('2bg_2a',4*h,12,17*(0-yy+i), 4*h,12,17*(1-yy+i), 0,12,17*(1-yy+i), 0,12,17*(-yy+i))
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
