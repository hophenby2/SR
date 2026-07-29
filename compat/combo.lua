LoadPS("combo_aura", "assets/combo/combo_aura.psi", "parimg11")

combo = combo or {}

function combo.init()
    combo.real = 0
    combo.value = 0
    combo.max = 0
    combo.timer = 0
    combo.tmpcombo = 0
    combo.tmpcombo_gauge = 0
    combo.hitcount = 0
    combo.nexttoggle = 0
    combo.on = false
    combo.display = nil
end

function combo.tmpup(value)
    combo.tmpcombo = combo.tmpcombo + value
    combo.tmpcombo_gauge = 100
end

function combo.up(value)
    for i = 1, value do
        lstg.var.score = lstg.var.score + combo.value * 1000 + i * 1000
    end
end

function combo.add(value)
    if not combo.on then
        return
    end
    combo.real = combo.real + value
    local delta = math.floor(combo.real) - combo.value
    if delta > 0 then
        combo.up(delta)
        combo.tmpup(delta)
        combo.value = combo.value + delta
        combo.max = max(combo.value, combo.max)
    end
end

function combo.hit(value, caller)
    combo.hitcount = combo.hitcount + value
    if combo.hitcount >= 1 then
        combo.hitcount = combo.hitcount - 1
        combo.tmpup(1)
        New(item_point, caller.x, caller.y)
        combo.value = combo.value + 1
        combo.real = combo.real + 1
        combo.max = max(combo.value, combo.max)
    end
end

function combo.down(rate)
    combo.real = combo.real * rate
    combo.value = math.floor(combo.real)
end

function combo.turnon()
    combo.on = true
    combo.nexttoggle = max(60, combo.nexttoggle)
    if player then
        player.vfact = 1.2
    end
    if IsValid(combo.display) then
        ParticleFire(combo.display)
    end
end

function combo.turnoff()
    combo.on = false
    if player then
        combo.nexttoggle = max(60, combo.nexttoggle)
        player.vfact = 1.0
    end
    if IsValid(combo.display) then
        ParticleStop(combo.display)
    end
end

function combo.frame()
    combo.timer = combo.timer + 1
    if combo.nexttoggle > 0 then
        combo.nexttoggle = combo.nexttoggle - 1
    end
    if not combo.on and combo.timer % 3 == 0 then
        combo.real = max(0, combo.real - 1)
    end
    combo.value = math.floor(combo.real)
    combo.tmpcombo_gauge = combo.tmpcombo_gauge - 1
    if combo.tmpcombo_gauge <= 0 or not combo.on then
        combo.tmpcombo = 0
    end
end

function combo.render()
    local x = (lstg.world and lstg.world.scrr or 416) + 194
    local active_alpha = combo.nexttoggle > 0 and 128 or 255
    SetFontState("score3", "", Color(active_alpha, 255, 255, 255))
    RenderText("score3", "[C] TOGGLE FIRE MODE", x, 237, 0.4, "right")

    local max_alpha = combo.value < combo.max and 127 or 255
    SetFontState("score3", "", Color(max_alpha, 255, max(0, 255 - combo.max / 5), max(0, 255 - combo.max)))
    RenderText("score3", string.format("MAX %d HIT", combo.max), x, 215, 0.4, "right")

    if lstg.var.nextextend then
        SetFontState("score3", "", Color(255, 255, 255, 192))
        RenderScore("score3", lstg.var.nextextend.value, x, 363, 0.3, "right")
    end

    local value_alpha = combo.on and 255 or 127
    SetFontState("score3", "", Color(value_alpha, 255, max(0, 255 - combo.value / 5), max(0, 255 - combo.value)))
    RenderText("score3", string.format("%d HIT", combo.value), x, 193, 0.4, "right")

    if combo.tmpcombo > 0 then
        SetFontState("score3", "", Color(255, 255, max(0, 5 + 2.5 * combo.tmpcombo_gauge), 0))
        RenderText("score3", string.format("+%d", combo.tmpcombo), x, 175, 0.3, "right")
    end
end

function combo.deathbullet(hp, x, y)
    local base = tonumber(difficulty) or 1
    local threshold = base
    local count = 0
    for _ = 1, 32 do
        if threshold > hp then
            break
        end
        count = count + 1
        threshold = threshold * base
    end
    combo.add(count)
    if abs(x) > 192 or abs(y) > 224 then
        return
    end
    New(combo_db_cr, x, y, count)
end

combo_db_cr = Class(object)

function combo_db_cr:init(x, y, count)
    self.x = x
    self.y = y
    self.count = count
    task.New(self, function()
        for _ = 1, self.count do
            if Dist(self, player) > 90 then
                New(_straight, mildew, COLOR_RED, self.x, self.y, 3, ran:Float(-10, 10), true, 3 * ran:Sign(), true, true, 0, false, 0, 0, 0, false)
            else
                _drop_item(item_faith_minor, 1, self.x, self.y)
            end
            task._Wait(9)
        end
        Del(self)
    end)
end

function combo_db_cr:frame()
    task.Do(self)
end

sr_combo_ui = Class(object)

function sr_combo_ui:init()
    self.group = GROUP_GHOST
    self.layer = LAYER_TOP + 1000
    self.bound = false
    self.colli = false
end

function sr_combo_ui:render()
    SetViewMode("ui")
    combo.render()
    SetViewMode("world")
end

function combo.ensure_ui()
    if not IsValid(combo.ui) then
        combo.ui = New(sr_combo_ui)
    end
end

combo.init()
