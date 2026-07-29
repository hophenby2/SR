stg5bg_background=Class(background)

function stg5bg_background:init()
	background.init(self,false)
	stg5bg=self
	LoadImageFromFile('5bg', 'assets/background/p5/5bg.png')
	SetImageState('5bg','add+sub',Color(255,72,48,48))
	Set3D('eye',0,6,-6)
	Set3D('at',0,0,10)
	Set3D('up',0,1,2)
	Set3D('z',0,24)
	Set3D('fovy',0.2)
	Set3D('fog',0,0,Color(0x00000000))
	self.yos=0
	self.yspeed=0.018
	self.xos=0
	self.xspeed=0
	
	CreateRenderTarget('player_distortion')
end

function stg5bg_background:frame()
	self.yos=self.yos+self.yspeed
	self.xos=self.xos+self.xspeed
end

function stg5bg_background:render()
	SetViewMode'3d'
	
	local showboss = IsValid(_boss)
	if showboss then
        PostEffectCapture()
    end
	
	PushRenderTarget('player_distortion')
	
	RenderClear(lstg.view3d.fog[3])
	local y=self.yos%1
	local x=self.xos%1
	for i=-2,5 do
		--[横宽，高度，纵深]
		for j=-2,5 do
			Render4V('5bg',
					3*(0-x+j),0,21*(0-y+i),
					3*(0-x+j),0,21*(1-y+i),
					3*(1-x+j),0,21*(1-y+i),
					3*(1-x+j),0,21*(0-y+i))
			Render4V('5bg',
					3*(0-x+j),0,16*(0-y+i),
					3*(0-x+j),0,16*(1-y+i),
					3*(1-x+j),0,16*(1-y+i),
					3*(1-x+j),0,16*(0-y+i))
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
