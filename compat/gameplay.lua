local player_colors = {
    Reimu = Color(180, 240, 75, 75),
    Marisa = Color(180, 120, 75, 240),
    Sakuya = Color(180, 200, 200, 200),
}

LoadPS("sr_player_bomb_ef", "assets/player/player_bomb_ef.psi", "parimg12")

local function update_laser_head(object_)
    local length = object_.l1 + object_.l2 + object_.l3
    object_.headx = object_.x + length * cos(object_.rot)
    object_.heady = object_.y + length * sin(object_.rot)
end

local original_laser_init = laser.init
function laser:init(...)
    original_laser_init(self, ...)
    update_laser_head(self)
end

local original_laser_frame = laser.frame
function laser:frame()
    original_laser_frame(self)
    update_laser_head(self)
end

-- SR bullet clears deliberately leave straight lasers alive.
function bullet_killer:frame()
    if self.timer == 40 then
        Del(self)
    end
    for _, object_ in ObjList(GROUP_ENEMY_BULLET) do
        if Dist(self, object_) < self.timer * 20 and not object_.l2 then
            Kill(object_)
        end
    end
    if self.kill_indes then
        for _, object_ in ObjList(GROUP_INDES) do
            if Dist(self, object_) < self.timer * 20 and not object_.l2 then
                Kill(object_)
            end
        end
    end
end

lstg.var.extable = {
    value = 100000000,
    next = {
        value = 300000000,
        next = {
            value = 600000000,
            next = { value = 1000000000 },
        },
    },
}
lstg.var.hidden_route = false

player_lib.defaultFrameEvent["frame.itemCollect"] = { 95, function(owner)
    if owner.__death_state ~= 0 then
        return
    end
    if owner.y > owner.collect_line then
        for _, object_ in ObjList(GROUP_ITEM) do
            object_.attract = 8
            object_.target = owner
        end
    else
        for _, object_ in ObjList(GROUP_ITEM) do
            if Dist(owner, object_) < 224 then
                object_.attract = max(object_.attract, 6)
                object_.target = owner
            end
        end
    end
end }

player_lib.defaultFrameEvent["frame.updateVar"] = { 90, function(owner)
    owner.lh = min(1, max(0, owner.lh + (owner.slow - 0.5) * 0.3))
    if owner.nextshoot > 0 then
        owner.nextshoot = owner.nextshoot - 1
    end
    if owner.nextspell > 0 then
        owner.nextspell = owner.nextspell - 1
    end
    if owner.nextsp > 0 then
        owner.nextsp = owner.nextsp - 1
    end

    owner.support = 4
    local latency = owner.svt_latency or 0.5
    owner.supportx = owner.x + (owner.supportx - owner.x) * latency
    owner.supporty = owner.y + (owner.supporty - owner.y) * latency

    if owner.protect > 0 then
        owner.protect = owner.protect - 1
    end
    if owner.death > 0 then
        owner.death = owner.death - 1
    end
    lstg.var.pointrate = item.PointRateFunc(lstg.var)
end }

player_lib.defaultFrameEvent["frame.updateSupport"] = { 89, function(owner)
    if owner.time_stop or not owner.slist then
        return
    end
    owner.sp = {}
    for i = 1, 4 do
        if owner.slist[i] then
            owner.sp[i] = MixTable(owner.lh, owner.slist[i])
            owner.sp[i][3] = 1
        end
    end
end }

player_lib.defaultFrameEvent["frame.srExtend"] = { 98.5, function(owner)
    if lstg.var.nextextend and lstg.var.score >= lstg.var.nextextend.value then
        New(item_extend, owner.x, owner.y)
        lstg.var.nextextend = lstg.var.nextextend.next
    end
end }

sr_player_autobomb = Class(object)

function sr_player_autobomb:init(owner)
    self.owner = owner
    self.group = GROUP_GHOST
    self.layer = LAYER_PLAYER + 10
    self.bound = false
    self.img = "sr_player_bomb_ef"
    owner.protect = 120
    owner.nextspell = 60
    misc.ShakeScreen(30, 3)
    PlaySound("don00", 1, owner.x / 200)
    combo.nexttoggle = max(combo.nexttoggle, 60)
end

function sr_player_autobomb:frame()
    local owner = self.owner
    if not IsValid(owner) then
        Del(self)
        return
    end
    self.x, self.y = owner.x, owner.y
    if self.timer <= 30 then
        for _, target in ObjList(GROUP_ENEMY) do
            if BoxCheck(target, -224, 224, -256, 256) then
                Damage(target, 2)
            end
        end
        for _, target in ObjList(GROUP_NONTJT) do
            if BoxCheck(target, -224, 224, -256, 256) then
                Damage(target, 2)
            end
        end
        for _, bullet in ObjList(GROUP_ENEMY_BULLET) do
            Kill(bullet)
        end
    end
    if self.timer == 30 then
        ParticleStop(self)
    end
    if self.timer >= 120 then
        Del(self)
    end
end

sr_player_bomb = Class(object)

function sr_player_bomb:init(owner)
    self.owner = owner
    self.group = GROUP_GHOST
    self.layer = LAYER_PLAYER + 10
    self.bound = false
    self.img = "sr_player_bomb_ef"
    owner.protect = 240
    owner.nextspell = 180
    misc.ShakeScreen(150, 3)
    PlaySound("don00", 1, owner.x / 200)
    combo.nexttoggle = max(combo.nexttoggle, 180)
end

function sr_player_bomb:frame()
    local owner = self.owner
    if not IsValid(owner) then
        Del(self)
        return
    end
    self.x, self.y = owner.x, owner.y
    if self.timer <= 150 then
        for _, target in ObjList(GROUP_ENEMY) do
            if BoxCheck(target, -224, 224, -256, 256) then
                Damage(target, 2)
            end
        end
        for _, target in ObjList(GROUP_NONTJT) do
            if BoxCheck(target, -224, 224, -256, 256) then
                Damage(target, 2)
            end
        end
        for _, bullet in ObjList(GROUP_ENEMY_BULLET) do
            Kill(bullet)
        end
    end
    if self.timer == 150 then
        ParticleStop(self)
    end
    if self.timer >= 240 then
        Del(self)
    end
end

local original_player_init = player_class.init
function player_class:init(slot)
    original_player_init(self, slot)
    self.support = 4
    self.vfact = 1
    combo.init()
    combo.ensure_ui()
end

local original_player_frame = player_class.frame
function player_class:frame()
    if not self.dcolor then
        self.dcolor = player_colors[self.name] or Color(180, 255, 255, 255)
    end

    combo.frame()
    combo.ensure_ui()

    if self.death == 91 and self.nextspell <= 0 and lstg.var.bomb > 0 and not lstg.var.block_spell then
        combo.down(0.9)
        lstg.var.bomb = lstg.var.bomb - 1
        item.PlayerSpell()
        New(sr_player_autobomb, self)
        self.death = 0
    end

    local factor = combo.on and 1.2 or 1
    local hspeed, lspeed = self.hspeed, self.lspeed
    self.hspeed = hspeed * factor
    self.lspeed = lspeed * factor
    original_player_frame(self)
    self.hspeed = hspeed
    self.lspeed = lspeed
end

function player_class:special()
    if self._playersys:keyIsPressed("special") and combo.nexttoggle <= 0 then
        if combo.on then
            combo.turnoff()
            PlaySound("ophide", 0.5, self.x / 200)
        else
            combo.turnon()
            PlaySound("opshow", 0.5, self.x / 200)
        end
    end
end

reimu_player.special = player_class.special
marisa_player.special = player_class.special
sakuya_player.special = player_class.special

-- reimu_player inherited frame before this compatibility wrapper existed.
-- Marisa and Sakuya have their own frame methods that call player_class.frame.
reimu_player.frame = player_class.frame

function player_lib.system:spell()
    local owner = self.player
    item.PlayerSpell()
    lstg.var.bomb = lstg.var.bomb - 1
    New(sr_player_bomb, owner)
    owner.death = 0
    owner.nextcollect = 90
end

local original_player_init_data = item.PlayerInit
function item.PlayerInit(...)
    original_player_init_data(...)
    lstg.var.nextextend = lstg.var.extable
    combo.init()
end

local original_player_reinit_data = item.PlayerReinit
function item.PlayerReinit(...)
    local power = lstg.var.power
    original_player_reinit_data(...)
    lstg.var.power = power
    lstg.var.nextextend = lstg.var.extable
    combo.init()
end

function item.PlayerMiss(owner)
    if lstg.var.sc_bonus then
        lstg.var.sc_bonus = 0
    end
    owner.protect = 360
    lstg.var.lifeleft = lstg.var.lifeleft - 1
    lstg.var.bomb = max(lstg.var.bomb, 3)
    combo.turnoff()
end

function item.PlayerSpell()
    if lstg.var.sc_bonus then
        lstg.var.sc_bonus = 0
    end
    combo.turnoff()
end

function item.PlayerGraze()
end

function item:StartChipBonus()
end

function item:EndChipBonus()
end

function item_power:collect()
end

function item_power_large:collect()
end

function item_power_full:collect()
end

function item_chip:collect()
end

function item_bombchip:collect()
end

function item_bomb:collect()
    if lstg.var.bomb < 8 then
        lstg.var.bomb = lstg.var.bomb + 1
    end
    PlaySound("cardget", 0.8)
end

function item_faith:collect()
end

function item_faith_minor:collect()
    lstg.var.score = lstg.var.score + 10000
    New(float_text, "item", "10k", self.x, self.y + 6, 3, 15, 60, 1, 2, Color(0x80FFFFFF), Color(0x00FFFFFF))
end

function item_extend:collect()
    if lstg.var.lifeleft < 8 then
        lstg.var.lifeleft = lstg.var.lifeleft + 1
    end
    PlaySound("extend", 0.5)
    New(hinter, "hint.extend", 0.6, 0, 112, 15, 120)
end

function item_point:collect()
    local value = max(1, combo.value)
    New(float_text, "item", string.format("%dk", value), self.x, self.y + 6, 3, 15, 60, 1, 2, Color(0x80FFFF00), Color(0x00FFFF00))
    lstg.var.score = lstg.var.score + value * 1000
end

local original_enemy_frame = enemybase.frame
function enemybase:frame()
    if self.hp <= 0 and not self._sr_combo_awarded then
        self._sr_combo_awarded = true
        if combo.on then
            combo.add(1)
            combo.deathbullet(min(self.maxhp, self.maxhp - self.hp), self.x, self.y)
            New(float_text, "item", string.format("%dk", combo.value), self.x, self.y, 3, 15, 60, 1, 2, Color(0x80FFFF00), Color(0x00FFFF00))
        else
            lstg.var.score = lstg.var.score + 10000
            New(float_text, "item", "1k", self.x, self.y + 6, 3, 15, 60, 1, 2, Color(0x80FFFFFF), Color(0x00FFFFFF))
        end
    end
    original_enemy_frame(self)
end

function enemybase:colli(other)
    if other.dmg then
        lstg.var.score = lstg.var.score + 100
        local damage = other.dmg
        Damage(self, damage)
        if self._master and self._dmg_transfer and IsValid(self._master) then
            Damage(self._master, damage * self._dmg_transfer)
        end
    end
    other.killerenemy = self
    if not other.killflag then
        other.dmg = nil
        Kill(other)
    end
    if not other.mute then
        if self.dmg_factor then
            if self.hp > 100 then
                PlaySound("damage00", 0.4, self.x / 200)
            else
                PlaySound("damage01", 0.6, self.x / 200)
            end
        elseif self.hp > 60 then
            if self.hp > self.maxhp * 0.2 then
                PlaySound("damage00", 0.4, self.x / 200)
            else
                PlaySound("damage01", 0.6, self.x / 200)
            end
        else
            PlaySound("damage00", 0.35, self.x / 200, true)
        end
    end
end

-- Class(base) copies callbacks when the class is created. These THlib classes
-- already existed before the compatibility layer replaced enemybase.colli.
enemy.colli = enemybase.colli
EnemySimple.colli = enemybase.colli
boss.colli = enemybase.colli

local function is_protected(object_)
    if object_.protect == true then
        return true
    end
    return type(object_.protect) == "number" and object_.protect > 0
end

local original_editor_object_frame = _object.frame
function _object:frame()
    if type(self.protect) == "number" then
        self.protect = max(0, self.protect - 1)
    end
    original_editor_object_frame(self)
end

local original_editor_object_take_damage = _object.take_damage
function _object:take_damage(damage)
    if not is_protected(self) then
        original_editor_object_take_damage(self, damage)
    end
end

function _object:colli(other)
    if self.group ~= GROUP_ENEMY and self.group ~= GROUP_NONTJT then
        return
    end
    if other.dmg and not is_protected(self) then
        lstg.var.score = lstg.var.score + 100
        local damage = other.dmg
        Damage(self, damage)
        if self._master and self._dmg_transfer and IsValid(self._master) then
            Damage(self._master, damage * self._dmg_transfer)
        end
    end
    other.killerenemy = self
    if not other.killflag then
        other.dmg = nil
        Kill(other)
    end
    if not other.mute then
        if self.dmg_factor then
            if self.hp > 100 then
                PlaySound("damage00", 0.4, self.x / 200)
            else
                PlaySound("damage01", 0.6, self.x / 200)
            end
        elseif self.hp > 60 then
            if self.hp > self.maxhp * 0.2 then
                PlaySound("damage00", 0.4, self.x / 200)
            else
                PlaySound("damage01", 0.6, self.x / 200)
            end
        else
            PlaySound("damage00", 0.35, self.x / 200, true)
        end
    end
end

local original_editor_object_kill = _object.kill
function _object:kill()
    original_editor_object_kill(self)
    if combo.on then
        combo.add(1)
        New(float_text, "item", "x" .. string.format("%d", combo.value), self.x, self.y, 3, 15, 60, 1, 2, Color(0x80FFFF00), Color(0x00FFFF00))
    else
        New(float_text, "item", "x" .. string.format("%d", combo.value), self.x, self.y, 3, 15, 60, 1, 2, Color(0x80FFFFFF), Color(0x00FFFFFF))
    end
end

local original_boss_init = boss.init
function boss:init(...)
    original_boss_init(self, ...)
    self.sc_bonus_max = 10000000 + 100000 * combo.max
    self.sc_bonus_base = 0
end

function boss:take_damage(damage)
    if self.dmgmaxt then
        self.dmgt = self.dmgmaxt
    end
    if not is_protected(self) then
        local actual_damage = damage * self.dmg_factor * (self.DMG_factor or 1)
        self.spell_damage = self.spell_damage + actual_damage
        self.hp = self.hp - actual_damage
        lstg.var.score = lstg.var.score + 100
    end
end
