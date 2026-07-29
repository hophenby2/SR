stg4bg_background=Class(background)

function stg4bg_background:init()
	background.init(self,false)
	stg4bg=self
	LoadImageFromFile('4bg_a', 'assets/background/p4/4bg_a.png')
	LoadImageFromFile('4bg_b', 'assets/background/p4/4bg_b.png')
	LoadImageFromFile('4bg_c', 'assets/background/p4/4bg_c.png')
	SetImageState('4bg_b', '', Color(0, 255,255,255) )
	
	Set3D('eye',0,6,-6)
	Set3D('at',0,0,10)
	Set3D('up',0,1,2)
	Set3D('z',0,24)
	Set3D('fovy',0.2)
	Set3D('fog',0,0,Color(0x00000000))
	self.yos=0
	self.speed=0.001
	
	stg4bg.atz=10
	
	CreateRenderTarget('player_distortion')
end

function stg4bg_background:frame()
	self.yos=self.yos+self.speed
	if self.timer>=90 and self.timer<120 then
		SetImageState('4bg_c', '', Color(255*(120-self.timer)/30, 255,255,255) )
		SetImageState('4bg_b', '', Color(255*(self.timer-90)/30, 255,255,255) )
	end
	
	if IsValid(_boss) then
		self.speed=max(-0.002,self.speed-0.004/120)
	end
	self.speed=min(0.003,self.speed+0.002/120)
	
	Set3D('eye',sin(self.timer*0.13),6,-6)
	
	Set3D('at',cos(self.timer*0.23),cos(self.timer*0.31),stg4bg.atz)
end

function stg4bg_background:render()
	SetViewMode'3d'
	
	local showboss = IsValid(_boss)
	if showboss then
        PostEffectCapture()
    end
	
	PushRenderTarget('player_distortion')
	
	RenderClear(lstg.view3d.fog[3])
	local y=self.yos%1
	for i=-2,10 do
		--[横宽，高度，纵深]
		for s=-2,4,2 do
			Render4V('4bg_c',-2+s,0,4*(0-y+i),-2+s,0,4*(1-y+i),0+s,0,4*(1-y+i),0+s,0,4*(-y+i))
			Render4V('4bg_b',-2+s,0,4*(0-y+i),-2+s,0,4*(1-y+i),0+s,0,4*(1-y+i),0+s,0,4*(-y+i))
		end
	end
	
	for i=-2,8 do 
		for s=-1,1,2 do 
			for dx=-8,-4,2 do
				Render4V('4bg_a',dx*s,0,4*(0-y+i),dx*s,0,4*(1-y+i),(dx+2)*s,0,4*(1-y+i),(dx+2)*s,0,4*(-y+i))
			end
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
        }
		)
	end
	
	SetViewMode'world'
end
