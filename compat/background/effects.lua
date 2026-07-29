sr_background = sr_background or {}

function sr_background.ApplyPlayerDistortion(render_target, timer)
    if not player then
        return
    end

    local px, py = WorldToScreen(player.x, player.y)
    local center_x = px * screen.scale
    local center_y = (screen.height - py) * screen.scale
    local progress = combo.nexttoggle * 0.75
    if combo.on then
        progress = (60 - combo.nexttoggle) * 0.75
    end

    local size = max(0.001, progress * 200 * lstg.scale_3d)
    local color = player.dcolor or Color(180, 255, 255, 255)
    local a, r, g, b = color:ARGB()
    PostEffect("boss_distortion", render_target, 6, "", {
        { center_x, center_y, 0, 0 },
        { r / 255, g / 255, b / 255, a / 255 },
        { size, 1500 * progress / 128 * lstg.scale_3d, size, timer or 0 },
    }, {})
end

-- SR rotates tiled spell-card layers. The current THlib renderer only applies
-- the layer rotation to non-tiled images, so retain its renderer with that
-- one legacy behavior restored.
function _spellcard_background:render()
    SetViewMode("world")
    if self.alpha <= 0 then
        return
    end

    local showboss = lstg.tmpvar.bg and lstg.tmpvar.bg.hide == true
    if showboss then
        background.WarpEffectCapture()
    end
    for index = #self.layers, 1, -1 do
        local layer = self.layers[index]
        layer._cur_alpha = self.alpha
        SetImageState(layer.img, layer.blend, Color(layer.a * self.alpha, layer.r, layer.g, layer.b))
        local world = lstg.world
        if layer.tile then
            local width, height = GetTextureSize(layer.img)
            local left = -int((world.r + 16 + layer.x) / width + 0.5)
            local right = int((world.r + 16 - layer.x) / width + 0.5)
            local bottom = -int((world.t + 16 + layer.y) / height + 0.5)
            local top = int((world.t + 16 - layer.y) / height + 0.5)
            for x = left, right do
                for y = bottom, top do
                    Render(layer.img, layer.x + x * width, layer.y + y * height, layer.rot)
                end
            end
        else
            Render(layer.img, layer.x, layer.y, layer.rot, layer.hscale, layer.vscale)
        end
        if layer.render then
            layer.render(layer)
        end
    end
    if showboss then
        background.WarpEffectApply()
    end
end

-- Translate the old named-parameter post-effect API used by SR backgrounds
-- to the float4 parameter arrays expected by LuaSTG Sub.
local legacy_boss_render_target = "_boss_distortion_render_buffer"
local legacy_capture_active = false

function PostEffectCapture()
    legacy_capture_active = IsValid(_boss)
    if legacy_capture_active then
        background.WarpEffectCapture()
    end
end

function PostEffectApply(effect_name, blend, arguments)
    if not legacy_capture_active then
        return
    end
    legacy_capture_active = false
    if not arguments then
        background.WarpEffectApply()
        return
    end

    PopRenderTarget()
    local color = arguments.color or Color(125, 163, 73, 164)
    local a, r, g, b = color:ARGB()
    PostEffect(effect_name or "boss_distortion", legacy_boss_render_target, 6, blend or "", {
        { arguments.centerX or 0, arguments.centerY or 0, 0, 0 },
        { r / 255, g / 255, b / 255, a / 255 },
        {
            arguments.size or 0,
            arguments.arg or 0,
            arguments.colorsize or arguments.size or 0,
            arguments.timer or 0,
        },
    }, {})
end
