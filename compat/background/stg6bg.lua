stg6bg_background=Class(background)

function stg6bg_background:init()
	background.init(self,false)
	stg6bg=self
	LoadImageFromFile('6bg', 'assets/background/p6/6bg.png')
	CopyImage('6bg_light', '6bg')
	SetImageState('6bg','add+sub',Color(255,220,152,152))
	SetImageState('6bg_light','mul+add',Color(0,220,152,152))
	Set3D('eye',0,6,-6)
	Set3D('at',0,0,10)
	Set3D('up',0,1,2)
	Set3D('z',0,24)
	Set3D('fovy',0.2)
	Set3D('fog',0,0,Color(0x00000000))
	self.yos=0
	self.speed=0.018
end

function stg6bg_background:frame()
	self.yos=self.yos+self.speed
	
	Set3D('eye',sin(self.timer/2),6,-6)
	Set3D('at',sin(self.timer/2),0,10)
end

function stg6bg_background:render()
	SetViewMode'3d'
	
	local showboss = IsValid(_boss)
	if showboss then
        PostEffectCapture()
    end
	
	RenderClear(lstg.view3d.fog[3])
	local y=self.yos%1
	for i=-2,5 do
		--[横宽，高度，纵深]
		Render4V('6bg',-3,0,21*(0-y+i),-3,0,21*(1-y+i),3,0,21*(1-y+i),3,0,21*(-y+i))
		Render4V('6bg',-3,0,16*(0-y+i),-3,0,16*(1-y+i),3,0,16*(1-y+i),3,0,16*(-y+i))
	end
	for i=-2,5 do
		--[横宽，高度，纵深]
		Render4V('6bg_light',-3,0,21*(0-y+i),-3,0,21*(1-y+i),3,0,21*(1-y+i),3,0,21*(-y+i))
		Render4V('6bg_light',-3,0,16*(0-y+i),-3,0,16*(1-y+i),3,0,16*(1-y+i),3,0,16*(-y+i))
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