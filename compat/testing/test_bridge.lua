local script_path = arg and arg[0] or "test_bridge.lua"
local directory = string.match(script_path, "^(.*[/\\])") or ""
local Bridge = dofile(directory .. "bridge.lua")

local function check(condition, message)
    if not condition then
        error(message or "check failed", 2)
    end
end

GROUP_ENEMY_BULLET = 1
GROUP_ENEMY = 2
GROUP_GHOST = 0
GROUP_PLAYER = 4
GROUP_INDES = 5
GROUP_NONTJT = 7

setting = {
    mod = "SR_Subterrain_Reanimation_v100",
    keys = { left = 1, right = 2, up = 3, down = 4, slow = 5, shoot = 6, spell = 7 },
}

laser = { is_class = true }
local laser_child = { is_class = true, base = laser }
local player_object = {
    x = 0, y = -100, vx = 0, vy = 0, a = 2, b = 2,
    colli = true, group = GROUP_PLAYER,
    status = "normal", death = 0, protect = 0,
}
local visible_bullet = {
    x = 10, y = 20, vx = 1, vy = -2, a = 3, b = 3,
    colli = true, status = "normal", img = "ball_mid1",
}
local offscreen_bullet = {
    x = 1000, y = 1000, vx = 0, vy = 0, a = 3, b = 3,
    colli = true, status = "normal",
}
local enemy_object = {
    x = 0, y = 120, vx = 0, vy = 0, a = 16, b = 16,
    hp = 100, maxhp = 100, colli = true, status = "normal",
}
local nuke_object = {
    x = -50, y = 40, vx = 2, vy = 0, a = 40, b = 40,
    colli = true, status = "normal",
}
local laser_object = {
    class = laser_child, x = -300, y = 0, rot = 0,
    l1 = 100, l2 = 500, l3 = 100, w = 16,
    a = 0, b = 0, colli = true, status = "normal",
}

local groups = {
    [GROUP_ENEMY_BULLET] = { visible_bullet, offscreen_bullet },
    [GROUP_ENEMY] = { enemy_object },
    [GROUP_PLAYER] = { player_object },
    [GROUP_INDES] = { nuke_object, laser_object },
    [GROUP_NONTJT] = {},
}

function ObjList(group)
    local objects = groups[group] or {}
    local function next_object(_, index)
        index = index + 1
        local object = objects[index]
        if object then
            return index, object
        end
    end
    return next_object, nil, 0
end

local native_key_calls = 0
local function native_get_key(code)
    native_key_calls = native_key_calls + 1
    return code == 99
end
GetKeyState = native_get_key
function GetFPS() return 47.5 end
function GetnObj() return 123 end

KeyState = {}
function GetInput()
    for name, code in pairs(setting.keys) do
        KeyState[name] = GetKeyState(code)
    end
end
local native_get_input = GetInput

local old_frame_count = 0
function FrameFunc()
    old_frame_count = old_frame_count + 1
    GetInput()
    if stage.next_stage then
        stage.current_stage = stage.next_stage
        stage.next_stage = nil
    end
    if GetKeyState(setting.keys.left) then player_object.x = player_object.x - 1 end
    if GetKeyState(setting.keys.right) then player_object.x = player_object.x + 1 end
    if GetKeyState(setting.keys.up) then player_object.y = player_object.y + 1 end
    if GetKeyState(setting.keys.down) then player_object.y = player_object.y - 1 end
    stage.current_stage.timer = (stage.current_stage.timer or 0) + 1
    return false
end

local old_render_count = 0
function RenderFunc()
    old_render_count = old_render_count + 1
end

local seed_value
ran = {
    Seed = function(_, value) seed_value = value end,
}

lstg = {
    GetKeyState = native_get_key,
    player = player_object,
    world = { l = -192, r = 192, b = -224, t = 224, pl = -192, pr = 192, pb = -224, pt = 224 },
    var = { lifeleft = 2, bomb = 3, power = 400, score = 0 },
    FileManager = {
        CreateDirectory = function() return false end,
        DirectoryExist = function() return true end,
    },
}

function Serialize(value)
    check(value == lstg.var, "replay serialized an unexpected state table")
    return "serialized-lstg-var"
end

local saved_replays = {}
local saved_replay_bytes = {}
local replay_frame_data_position = 32
local replay_file_length_adjustment = 0
local replay_save_failures = 0
local replay_read_failures = 0
local replay_save_calls = {}
local replay_read_calls = {}
local original_bit = rawget(_G, "bit")
local original_io_open = io.open
bit = {
    bxor = function() return 0 end,
    band = function() return 0 end,
    rshift = function() return 0 end,
    bnot = function() return 0 end,
}
io.open = function(path, mode)
    if mode == "rb" and saved_replay_bytes[path] then
        local bytes = saved_replay_bytes[path]
        local position = 0
        return {
            read = function(_, count)
                if position >= #bytes then return nil end
                count = tonumber(count) or (#bytes - position)
                local value = string.sub(bytes, position + 1, position + count)
                position = position + #value
                return value
            end,
            seek = function(_, origin, offset)
                origin = origin or "cur"
                offset = offset or 0
                local base = origin == "set" and 0
                    or origin == "end" and #bytes or position
                local target = base + offset
                if target < 0 then return nil, "invalid seek" end
                position = target
                return position
            end,
            close = function() end,
        }
    end
    return original_io_open(path, mode)
end
plus = {
    ReplayFrameWriter = function()
        local writer = { count = 0, states = {} }
        function writer:Record(state)
            self.count = self.count + 1
            self.states[self.count] = {
                left = state.left == true,
                right = state.right == true,
                shoot = state.shoot == true,
            }
        end
        function writer:GetCount() return self.count end
        function writer:CopyToFileStream() end
        return writer
    end,
    ReplayManager = {},
}
function plus.ReplayManager.SaveReplayInfo(path, data)
    replay_save_calls[path] = (replay_save_calls[path] or 0) + 1
    if replay_save_failures > 0 then
        replay_save_failures = replay_save_failures - 1
        error("injected SaveReplayInfo failure")
    end
    saved_replays[path] = data
    local frame_count = data.stages[1].frameData:GetCount()
    local length = math.max(0,
        replay_frame_data_position + frame_count + replay_file_length_adjustment)
    saved_replay_bytes[path] = string.rep("\0", length)
end
function plus.ReplayManager.ReadReplayInfo(path)
    replay_read_calls[path] = (replay_read_calls[path] or 0) + 1
    if replay_read_failures > 0 then
        replay_read_failures = replay_read_failures - 1
        error("injected ReadReplayInfo failure")
    end
    local data = saved_replays[path]
    check(data ~= nil, "replay verification read an unknown path")
    return {
        fileVersion = 1,
        gameName = data.gameName,
        gameVersion = data.gameVersion,
        userName = data.userName,
        group_finish = data.group_finish,
        stages = {
            {
                stageName = data.stages[1].stageName,
                stageExtendInfo = data.stages[1].stageExtendInfo,
                randomSeed = data.stages[1].randomSeed,
                stagePlayer = data.stages[1].stagePlayer,
                frameCount = data.stages[1].frameData:GetCount(),
                frameDataPosition = replay_frame_data_position,
            },
        },
    }
end

local spell_stage = { stage_name = "Spell Practice@Spell Practice", is_menu = false, timer = 0 }
local stage4 = { stage_name = "Stage 4@Lunatic", is_menu = false, timer = 0 }
local full_stage = { stage_name = "Stage 5@Lunatic", is_menu = false, timer = 0 }
local menu_stage = { stage_name = "menu", is_menu = true, timer = 0 }
stage = {
    stages = {
        ["Spell Practice@Spell Practice"] = spell_stage,
        ["Stage 4@Lunatic"] = stage4,
        ["Stage 5@Lunatic"] = full_stage,
        ["menu"] = menu_stage,
    },
    current_stage = { stage_name = "menu", is_menu = true, timer = 0 },
}
function stage.Set(name)
    stage.next_stage = stage.stages[name]
end

task = { Clear = function() end }
ext = { pause_menu = {} }
player_list = { { "Reimu", "reimu_player", "Reimu" } }
reimu_player = { is_class = true }
_editor_class = {
    ["okuu:Lunatic"] = {
        cards = {
            { is_combat = false, t3 = 60 },
            { is_combat = true, t3 = 300 },
            { is_combat = true, t3 = 1 },
            { is_combat = true, t3 = 300 },
        },
    },
}
SR_SPELL_PRACTICE_CATALOG = {
    schema_version = 1,
    scenarios = {
        {
            scenario = "okuu:Lunatic",
            label = "Lunatic 5 Boss",
            attacks = {
                { attack = 1, card_index = 2, label = "Lunatic 5 Boss #1" },
                { attack = 2, card_index = 4, label = "Lunatic 5 Boss #2" },
            },
        },
    },
}
SR_STAGE_TEST_CATALOG = {
    schema_version = 1,
    stages = {
        {
            stage = "Stage 4@Lunatic",
            difficulty = "Lunatic",
            stage_index = 4,
            label = "Lunatic Stage 4",
        },
        {
            stage = "Stage 5@Lunatic",
            difficulty = "Lunatic",
            stage_index = 5,
            label = "Lunatic Stage 5",
        },
    },
}

local encoded_values = {}
local requests = {}
local fake_cjson = {}
fake_cjson.array_mt = {}
function fake_cjson.decode(token)
    local value = requests[token]
    if not value then error("unknown request token") end
    return value
end
function fake_cjson.encode(value)
    encoded_values[#encoded_values + 1] = value
    return "response-" .. tostring(#encoded_values)
end

local client = { input = {}, output = "", closed = false }
function client:settimeout() end
function client:setoption() end
function client:receive()
    if #self.input > 0 then
        return table.remove(self.input, 1)
    end
    return nil, "timeout", ""
end
function client:send(value)
    self.output = self.output .. value
    return #value
end
function client:close() self.closed = true end

local accepted = false
local server = {}
function server:settimeout() end
function server:getsockname() return "127.0.0.1", 24816 end
function server:accept()
    if accepted then return nil, "timeout" end
    accepted = true
    return client
end
function server:close() end

local fake_socket = {}
function fake_socket.bind(host, port)
    check(host == "127.0.0.1", "unexpected host")
    check(port == 24816, "default port must match Python client")
    return server
end

requests.RESET = {
    id = 1, command = "reset", scenario = "okuu:Lunatic", attack = 2,
    seed = 42, player = "reimu_player",
    options = {
        lifeleft = 5,
        player_protect_frames = 500,
        player_collidable = false,
        player_ghost = true,
    },
}
client.input[#client.input + 1] = "RESET"

local instance, err = Bridge.install(nil, {
    socket = fake_socket,
    cjson = fake_cjson,
    runtime_identity = {
        process_id = 1234,
        executable_path = "LuaSTGSub.exe",
        executable_crc32 = "1234abcd",
        source_crc32 = {
            ["root.lua"] = "11111111",
            ["compat/testing/bridge.lua"] = "22222222",
            ["compat/testing/init.lua"] = "33333333",
            ["compat/spell_practice.lua"] = "44444444",
            ["_editor_output.lua"] = "55555555",
        },
    },
})
check(instance, err)
check(seed_value == 1, "initial seed was not applied")
check(GetKeyState(99) == true, "native input failed before the first client accept")

instance:_accept()
instance:_handle_request({ id = -1, command = "ping" })
check(type(encoded_values[#encoded_values].process_nonce) == "string"
        and encoded_values[#encoded_values].process_nonce ~= "",
    "ping did not expose an engine-generated process nonce")
check(encoded_values[#encoded_values].protocol == 2, "ping protocol version mismatch")
check(encoded_values[#encoded_values].runtime_identity.process_id == 1234,
    "ping did not expose the operating-system process id")
instance:_handle_request({ id = 0, command = "catalog" })
check(encoded_values[#encoded_values].id == 0 and encoded_values[#encoded_values].ok, "missing catalog response")
check(encoded_values[#encoded_values].catalog.attack_count == 2, "catalog attack count mismatch")
check(encoded_values[#encoded_values].catalog.attacks[2].card_index == 4, "catalog card index mismatch")
check(encoded_values[#encoded_values].catalog.stage_count == 2, "catalog stage count mismatch")
check(encoded_values[#encoded_values].catalog.stages[2].completion_reason == "stage_complete",
    "catalog stage completion contract mismatch")
encoded_values = {}

SafetyZoneVisualizer = {
    getRuntimeStatus = function()
        return {
            schema_version = 2,
            enabled = true,
            data_source = "controller",
            controller_revision = 4,
        }
    end,
}
SR_SAFETY_ZONE_CONTROLLER_STATE = { stale = true }
FrameFunc()
check(old_frame_count == 1, "reset must advance exactly one frame")
check(SR_SAFETY_ZONE_CONTROLLER_STATE == nil,
    "attack reset did not clear stale controller overlay state")
check(seed_value == 42, "reset seed was not applied")
check(lstg.var.sc_pr_data.scene_index == 4, "attack ordinal did not resolve to card index")
check(lstg.var.sc_index == nil, "direct spell-practice reset must not leave a zero menu index")
check(lstg.var.lifeleft == 5, "reset resource override was not applied")
check(player_object.protect == 500, "player protection override was not applied")
check(player_object.colli == false, "player collision override was not applied")
check(player_object.group == GROUP_GHOST, "player ghost-group override was not applied")
check(encoded_values[#encoded_values].id == 1 and encoded_values[#encoded_values].ok, "missing reset response")
check(encoded_values[#encoded_values].observation.counts.enemy_bullets == 1, "visibility filter failed")
check(encoded_values[#encoded_values].observation.counts.lasers == 1, "laser classification failed")
check(encoded_values[#encoded_values].observation.performance.native_fps == 47.5,
    "native FPS telemetry mismatch")
check(encoded_values[#encoded_values].observation.performance.object_count == 123,
    "native object-count telemetry mismatch")
check(encoded_values[#encoded_values].observation.safety_zone_overlay.data_source
        == "controller",
    "safety-zone overlay runtime status missing")
check(getmetatable(encoded_values[#encoded_values].observation.nontjt_enemies) == fake_cjson.array_mt,
    "empty observation arrays must retain JSON array identity")

requests.STEP = {
    id = 2, command = "step",
    action = { move_x = 1, move_y = -1, slow = true, shoot = true, spell = false },
    ["repeat"] = 3,
    controller_overlay_state = {
        schema_version = 1,
        revision = 4,
        region_navigation_active = true,
    },
}
client.input[#client.input + 1] = "STEP"
FrameFunc()
local controller_overlay_state = SR_SAFETY_ZONE_CONTROLLER_STATE
check(type(controller_overlay_state) == "table"
        and controller_overlay_state.revision == 4,
    "step did not publish controller overlay state")
check(#encoded_values == 1, "step responded before repeat completed")
FrameFunc()
check(#encoded_values == 1, "step responded before final repeat")
FrameFunc()
check(old_frame_count == 4, "repeat did not advance three frames")
check(player_object.x == 3 and player_object.y == -103, "action was not held across repeat")
check(encoded_values[#encoded_values].id == 2, "step response id mismatch")
check(encoded_values[#encoded_values].observation.episode_frame == 4, "episode frame mismatch")

instance:_handle_request({
    id = 19, command = "step",
    action = { move_x = 0, move_y = 0, shoot = true },
    ["repeat"] = 1,
})
check(SR_SAFETY_ZONE_CONTROLLER_STATE == controller_overlay_state,
    "step without overlay state discarded the previous controller state")
FrameFunc()

instance:_handle_request({
    id = 20, command = "reset_stage", stage = "Stage 4@Lunatic",
    seed = 77, player = "reimu_player",
    options = {
        lifeleft = 4,
        player_collidable = true,
        player_ghost = false,
    },
})
check(SR_SAFETY_ZONE_CONTROLLER_STATE == nil,
    "stage reset did not clear controller overlay state")
FrameFunc()
check(instance.episode_kind == "stage", "stage reset did not set episode kind")
check(instance.expected_stage == "Stage 4@Lunatic", "stage reset expected-stage mismatch")
check(instance.expected_stage_successor == "Stage 5@Lunatic",
    "stage reset successor mismatch")
check(instance.expected_stage_menu == false, "nonfinal stage unexpectedly accepts a menu")
check(encoded_values[#encoded_values].reset.episode_kind == "stage",
    "stage reset response metadata mismatch")
check(encoded_values[#encoded_values].observation.stage.scenario == "Stage 4@Lunatic",
    "stage observation scenario mismatch")
check(lstg.var.lifeleft == 4, "stage reset resource override was not applied")
check(player_object.colli == true, "stage reset did not restore player collision")
check(player_object.group == GROUP_PLAYER, "stage reset did not restore player group")
instance.seen_enemy = true
groups[GROUP_ENEMY] = {}
check(instance:_check_stop(instance:collect_observation()) == nil,
    "empty gap between stage waves prematurely completed the stage")
stage.current_stage = menu_stage
check(instance:_check_stop(instance:collect_observation()) == "stage_changed",
    "a nonfinal stage accepted a menu transition")
stage.current_stage = { stage_name = "Stage 6@Lunatic", is_menu = false, timer = 0 }
check(instance:_check_stop(instance:collect_observation()) == "stage_changed",
    "a noncatalog stage transition was accepted")
stage.current_stage = full_stage
check(instance:_check_stop(instance:collect_observation()) == "stage_complete",
    "natural stage transition did not prove stage completion")

instance:_handle_request({
    id = 21, command = "reset_stage", stage = "Stage 5@Lunatic",
    seed = 78, player = "reimu_player",
})
FrameFunc()
instance.seen_enemy = true
groups[GROUP_ENEMY] = {}
check(instance.expected_stage_successor == nil, "final stage has an unexpected successor")
check(instance.expected_stage_menu == true, "final stage does not accept its menu target")
stage.current_stage = { stage_name = "Stage 6@Lunatic", is_menu = false, timer = 0 }
check(instance:_check_stop(instance:collect_observation()) == "stage_changed",
    "final stage accepted an arbitrary stage transition")
stage.current_stage = menu_stage
check(instance:_check_stop(instance:collect_observation()) == "stage_complete",
    "final stage did not accept its normal menu transition")

stage.current_stage = full_stage
instance.terminated = false
instance.termination_reason = nil
local saved_original_frame = instance.original_frame
instance.original_frame = function() return true end
instance.pending = { id = 22, command = "step", remaining = 1 }
instance.action = {}
instance:_frame()
check(instance.termination_reason == "engine_exit",
    "engine exit was incorrectly accepted as full-stage completion")
instance.original_frame = saved_original_frame
stage.current_stage = spell_stage
instance.terminated = false
instance.termination_reason = nil
instance.episode_kind = "attack"
instance.episode_scenario = "okuu:Lunatic"
instance.expected_stage = "Spell Practice@Spell Practice"
groups[GROUP_ENEMY] = { enemy_object }

RenderFunc()
check(old_render_count == 0, "headless mode did not suppress rendering while connected")
instance:_handle_request({ id = 3, command = "display", render = true, every = 5 })
check(encoded_values[#encoded_values].render == true, "display command did not enable rendering")
check(encoded_values[#encoded_values].every == 5, "display command did not preserve the sampling hint")
RenderFunc()
RenderFunc()
check(old_render_count == 2, "visible lockstep did not redraw a waiting logical frame")
instance.frame = instance.frame + 1
RenderFunc()
RenderFunc()
check(old_render_count == 4, "visible lockstep skipped a native render pass")
instance:_handle_request({ id = 4, command = "display", render = false })
check(encoded_values[#encoded_values].render == false, "display command did not disable rendering")
instance.seen_enemy = true
groups[GROUP_ENEMY] = { offscreen_bullet }
local hidden_observation = instance:collect_observation()
check(hidden_observation.counts.enemies == 0, "off-screen enemy leaked into visible observation")
check(instance:_check_stop(hidden_observation) == nil,
    "off-screen enemy was mistaken for attack completion")
groups[GROUP_ENEMY] = {}
check(instance:_check_stop(instance:collect_observation()) == "attack_complete",
    "empty authoritative enemy pool did not finish the attack")
groups[GROUP_ENEMY] = { enemy_object }

instance:_handle_request({
    id = 30, command = "reset", scenario = "okuu:Lunatic", attack = 2,
    seed = 100, player = "reimu_player", replay_name = "../invalid",
})
check(encoded_values[#encoded_values].id == 30
        and encoded_values[#encoded_values].ok == false,
    "unsafe replay filename was accepted")
check(string.find(encoded_values[#encoded_values].error, "portable filename", 1, true),
    "unsafe replay filename returned the wrong error")
check(instance.replay_capture == nil,
    "invalid replay filename unexpectedly started a capture")

instance:_handle_request({
    id = 300, command = "reset", scenario = "okuu:Lunatic", attack = 2,
    seed = 100, player = "reimu_player", replay_name = "trailing-dot.",
})
check(encoded_values[#encoded_values].id == 300
        and encoded_values[#encoded_values].ok == false,
    "Windows-trimmed trailing-dot replay filename was accepted")

for index, reserved_name in ipairs({
    "CON", "con.rep", "PrN.REP", "AUX.trace", "nul.rep",
    "COM1", "com9.rep", "LPT1.trace.rep", "lPt9.ReP",
}) do
    instance:_handle_request({
        id = 3000 + index,
        command = "reset",
        scenario = "okuu:Lunatic",
        attack = 2,
        seed = 100 + index,
        player = "reimu_player",
        replay_name = reserved_name,
    })
    local reserved_response = encoded_values[#encoded_values]
    check(reserved_response.id == 3000 + index and reserved_response.ok == false,
        "Windows reserved replay name was accepted: " .. reserved_name)
    check(string.find(
            reserved_response.error, "Windows reserved device basename", 1, true),
        "Windows reserved replay name returned the wrong error: " .. reserved_name)
    check(instance.replay_capture == nil,
        "Windows reserved replay name unexpectedly started a capture")
end

instance:_handle_request({
    id = 31, command = "reset", scenario = "okuu:Lunatic", attack = 2,
    seed = 101, player = "reimu_player", replay_name = "non-reproducible",
    options = { player_ghost = true },
})
check(encoded_values[#encoded_values].id == 31
        and encoded_values[#encoded_values].ok == false,
    "non-reproducible replay options were accepted")
check(string.find(encoded_values[#encoded_values].error, "player_ghost", 1, true),
    "non-reproducible replay options returned the wrong error")
check(instance.replay_capture == nil,
    "rejected replay options unexpectedly started a capture")

instance:_handle_request({
    id = 32, command = "reset_stage", stage = "Stage 4@Lunatic",
    seed = 102, player = "reimu_player", replay_name = "stage4-rejected",
})
check(encoded_values[#encoded_values].id == 32
        and encoded_values[#encoded_values].ok == false,
    "non-final stage replay capture was accepted")
check(string.find(encoded_values[#encoded_values].error, "final stages", 1, true),
    "non-final stage replay rejection returned the wrong error")
check(instance.replay_capture == nil,
    "rejected Stage 4 replay unexpectedly started a capture")

instance:_handle_request({
    id = 33, command = "reset", scenario = "okuu:Lunatic", attack = 2,
    seed = 20260730, player = "reimu_player", replay_name = "boss3-analysis.REP",
    options = {},
})
check(instance.replay_capture ~= nil,
    "spell-practice reset did not start replay capture")
check(instance.replay_capture.writer:GetCount() == 0,
    "replay capture recorded input before the reset frame")
FrameFunc()
local replay_reset = encoded_values[#encoded_values]
check(replay_reset.id == 33 and replay_reset.ok,
    "spell-practice replay reset failed")
check(replay_reset.reset.replay.name == "boss3-analysis",
    "replay reset did not normalize the .rep suffix")
check(replay_reset.reset.replay.random_seed == 20260730,
    "replay reset metadata seed mismatch")
check(replay_reset.reset.replay.player == "Reimu",
    "replay reset did not use the registered replay player label")
local attack_writer = instance.replay_capture.writer
check(attack_writer:GetCount() == 1,
    "reset frame was not recorded exactly once")
check(not attack_writer.states[1].left
        and not attack_writer.states[1].right
        and not attack_writer.states[1].shoot,
    "reset frame did not record neutral input")

instance:_handle_request({
    id = 34, command = "step",
    action = { move_x = 1, move_y = 0, shoot = true },
    ["repeat"] = 2,
})
FrameFunc()
FrameFunc()
check(attack_writer:GetCount() == 3,
    "held replay input did not record one sample per logical frame")
check(attack_writer.states[2].right and attack_writer.states[2].shoot
        and attack_writer.states[3].right and attack_writer.states[3].shoot,
    "replay input samples did not preserve held movement and shooting")

instance:_handle_request({
    id = 349, command = "save_replay", finish = "false",
    reason = "attack_complete",
})
local invalid_finish_type = encoded_values[#encoded_values]
check(invalid_finish_type.id == 349 and invalid_finish_type.ok == false,
    "save_replay accepted a non-Boolean finish value")
check(instance.replay_capture ~= nil,
    "invalid save_replay fields discarded the active capture")

instance:_handle_request({
    id = 350, command = "save_replay", finish = true,
    reason = "attack_complete",
})
local invalid_attack_finish = encoded_values[#encoded_values]
check(invalid_attack_finish.id == 350 and invalid_attack_finish.ok == false,
    "spell-practice replay accepted finish=true")
check(instance.replay_capture ~= nil,
    "rejected finish=true discarded the active spell-practice capture")

local attack_path = "userdata/replay/SR_Subterrain_Reanimation_v100/analysis/boss3-analysis.rep"
local attack_capture = instance.replay_capture
replay_save_failures = 1
instance:_handle_request({
    id = 351, command = "save_replay", finish = false,
    reason = "attack_complete",
})
local failed_attack_save = encoded_values[#encoded_values]
check(failed_attack_save.id == 351 and failed_attack_save.ok == false,
    "injected SaveReplayInfo failure unexpectedly succeeded")
check(string.find(failed_attack_save.error, "SaveReplayInfo failure", 1, true),
    "SaveReplayInfo failure returned the wrong error")
check(instance.replay_capture == attack_capture
        and instance.replay_capture.writer == attack_writer,
    "SaveReplayInfo failure discarded or replaced the active capture")
check(instance.replay_capture.writer:GetCount() == 3,
    "SaveReplayInfo failure changed the captured frame stream")
check(saved_replays[attack_path] == nil,
    "failed SaveReplayInfo unexpectedly published replay data")

instance:_handle_request({
    id = 35, command = "save_replay", finish = false,
    reason = "attack_complete",
})
local attack_save = encoded_values[#encoded_values]
check(attack_save.id == 35 and attack_save.ok,
    "spell-practice replay save failed")
check(attack_save.replay.saved and attack_save.replay.verified,
    "spell-practice replay was not reported as saved and verified")
check(attack_save.replay.frame_count == 3
        and attack_save.replay.frame_bytes_verified == 3
        and attack_save.replay.file_size == replay_frame_data_position + 3
        and attack_save.replay.reason == "attack_complete"
        and attack_save.replay.finish == false
        and attack_save.replay.group_finish == 0,
    "spell-practice replay result metadata mismatch")
check(instance.replay_capture == nil,
    "saved replay capture remained active")
check(replay_save_calls[attack_path] == 2,
    "SaveReplayInfo retry did not perform exactly two save attempts")
local attack_replay = saved_replays[attack_path]
check(type(attack_replay) == "table", "spell-practice replay used the wrong path")
check(attack_replay.gameName == setting.mod
        and attack_replay.gameVersion == 1
        and attack_replay.userName == "stg-lab",
    "spell-practice replay header metadata mismatch")
check(attack_replay.group_finish == 0,
    "spell-practice replay was incorrectly marked as a finished group")
check(#attack_replay.stages == 1
        and attack_replay.stages[1].stageName == "Spell Practice@Spell Practice"
        and attack_replay.stages[1].stageExtendInfo == "serialized-lstg-var"
        and attack_replay.stages[1].randomSeed == 20260730
        and attack_replay.stages[1].stagePlayer == "Reimu"
        and attack_replay.stages[1].frameData == attack_writer,
    "spell-practice replay stage metadata mismatch")

instance:_handle_request({
    id = 36, command = "reset_stage", stage = "Stage 5@Lunatic",
    seed = 20260731, player = "reimu_player", replay_name = "stage5-final",
})
check(instance.replay_capture ~= nil,
    "final-stage reset did not start replay capture")
FrameFunc()
local stage5_reset = encoded_values[#encoded_values]
check(stage5_reset.id == 36 and stage5_reset.ok,
    "final-stage replay reset failed")
check(stage5_reset.reset.replay.episode_kind == "stage"
        and stage5_reset.reset.replay.stage_name == "Stage 5@Lunatic",
    "final-stage replay reset metadata mismatch")
instance:_handle_request({
    id = 370, command = "save_replay", finish = true,
    reason = "stage_complete",
})
local premature_stage5_save = encoded_values[#encoded_values]
check(premature_stage5_save.id == 370 and premature_stage5_save.ok == false,
    "unfinished final-stage replay accepted finish=true")
check(instance.replay_capture ~= nil,
    "rejected final-stage finish discarded the active capture")
instance.terminated = true
instance.termination_reason = "stage_complete"
local stage5_path = "userdata/replay/SR_Subterrain_Reanimation_v100/analysis/stage5-final.rep"
local stage5_capture = instance.replay_capture
local stage5_writer = stage5_capture.writer
replay_read_failures = 1
instance:_handle_request({
    id = 371, command = "save_replay", finish = true,
    reason = "stage_complete",
})
local failed_stage5_read = encoded_values[#encoded_values]
check(failed_stage5_read.id == 371 and failed_stage5_read.ok == false,
    "injected ReadReplayInfo failure unexpectedly succeeded")
check(string.find(failed_stage5_read.error, "ReadReplayInfo failure", 1, true),
    "ReadReplayInfo failure returned the wrong error")
check(instance.replay_capture == stage5_capture
        and instance.replay_capture.writer == stage5_writer,
    "ReadReplayInfo failure discarded or replaced the active capture")
check(instance.replay_capture.writer:GetCount() == 1,
    "ReadReplayInfo failure changed the captured frame stream")

replay_file_length_adjustment = -1
instance:_handle_request({
    id = 372, command = "save_replay", finish = true,
    reason = "stage_complete",
})
local truncated_stage5_save = encoded_values[#encoded_values]
check(truncated_stage5_save.id == 372 and truncated_stage5_save.ok == false,
    "truncated replay frame data unexpectedly passed verification")
check(string.find(truncated_stage5_save.error, "frame data is truncated", 1, true),
    "truncated replay returned the wrong verification error")
check(instance.replay_capture == stage5_capture,
    "truncated replay verification discarded the active capture")

replay_file_length_adjustment = 1
instance:_handle_request({
    id = 373, command = "save_replay", finish = true,
    reason = "stage_complete",
})
local trailing_stage5_save = encoded_values[#encoded_values]
check(trailing_stage5_save.id == 373 and trailing_stage5_save.ok == false,
    "replay with trailing bytes unexpectedly passed verification")
check(string.find(trailing_stage5_save.error, "replay EOF mismatch", 1, true),
    "trailing replay data returned the wrong verification error")
check(instance.replay_capture == stage5_capture,
    "trailing replay verification discarded the active capture")

replay_file_length_adjustment = 0
instance:_handle_request({
    id = 37, command = "save_replay", finish = true,
    reason = "stage_complete",
})
local stage5_save = encoded_values[#encoded_values]
check(stage5_save.id == 37 and stage5_save.ok,
    "final-stage replay save failed")
check(stage5_save.replay.finish == true
        and stage5_save.replay.group_finish == 1,
    "final-stage replay result did not preserve its completion flag")
check(instance.replay_capture == nil,
    "successful ReadReplayInfo retry left the capture active")
check(replay_save_calls[stage5_path] == 4
        and replay_read_calls[stage5_path] == 4,
    "replay verification retries did not perform four save/read attempts")
local stage5_replay = saved_replays[stage5_path]
check(type(stage5_replay) == "table", "final-stage replay used the wrong path")
check(stage5_replay.group_finish == 1,
    "completed final-stage replay was not marked as a finished group")
check(stage5_replay.stages[1].stageName == "Stage 5@Lunatic"
        and stage5_replay.stages[1].randomSeed == 20260731
        and stage5_replay.stages[1].stagePlayer == "Reimu"
        and stage5_replay.stages[1].frameData:GetCount() == 1,
    "final-stage replay stage metadata mismatch")

SR_SAFETY_ZONE_CONTROLLER_STATE = { revision = 99 }
instance:_disconnect("closed")
check(SR_SAFETY_ZONE_CONTROLLER_STATE == nil,
    "disconnect did not clear stale controller overlay state")
check(GetKeyState(99) == true, "native key input did not resume after disconnect")
RenderFunc()
check(old_render_count == 5, "rendering did not resume after disconnect")

instance:uninstall()
check(FrameFunc ~= instance.frame_wrapper, "FrameFunc was not restored")
check(GetKeyState == native_get_key, "GetKeyState was not restored")
check(GetInput == native_get_input, "GetInput was not restored")

local function make_startup_transport(accept_on_call)
    local startup_client = { closed = false, timeouts = {} }
    function startup_client:settimeout(value)
        self.timeouts[#self.timeouts + 1] = value
    end
    function startup_client:setoption() end
    function startup_client:receive() return nil, "timeout", "" end
    function startup_client:send(value) return #value end
    function startup_client:close() self.closed = true end

    local startup_server = { accept_calls = 0, timeouts = {}, allow_client = false }
    function startup_server:settimeout(value)
        self.timeouts[#self.timeouts + 1] = value
    end
    function startup_server:getsockname() return "127.0.0.1", 24816 end
    function startup_server:accept()
        self.accept_calls = self.accept_calls + 1
        if self.accept_calls == accept_on_call or self.allow_client then
            return startup_client
        end
        return nil, "timeout"
    end
    function startup_server:close() end

    local startup_socket = {}
    function startup_socket.bind() return startup_server end
    return startup_socket, startup_server, startup_client
end

local startup_socket, startup_server, startup_client = make_startup_transport(1)
local startup_instance, startup_err = Bridge.install(
    { startup_accept_timeout = 2.5 }, { socket = startup_socket, cjson = fake_cjson })
check(startup_instance, startup_err)
check(startup_instance.client == startup_client, "startup client was not attached during install")
check(startup_server.accept_calls == 1, "startup accept count mismatch")
check(startup_server.timeouts[2] == 2.5, "startup accept timeout was not applied")
check(startup_server.timeouts[3] == 0, "server was not restored to non-blocking mode")
check(startup_client.timeouts[1] == 0, "startup client was not made non-blocking")
startup_instance:uninstall()

local timeout_socket, timeout_server, timeout_client = make_startup_transport(math.huge)
local timeout_instance, timeout_err = Bridge.install(
    { startup_accept_timeout = 1 }, { socket = timeout_socket, cjson = fake_cjson })
check(timeout_instance, timeout_err)
check(timeout_instance.client == nil, "startup timeout unexpectedly attached a client")
check(timeout_server.accept_calls == 1, "startup timeout did not attempt accept")
check(timeout_server.timeouts[2] == 1, "timeout test did not use configured wait")
check(timeout_server.timeouts[3] == 0, "timeout left server in blocking mode")
timeout_server.allow_client = true
check(timeout_instance:_accept(), "non-blocking accept failed after startup timeout")
check(timeout_instance.client == timeout_client, "late client was not attached")
check(timeout_server.timeouts[#timeout_server.timeouts] == 0,
    "late accept changed the server timeout")
timeout_instance:uninstall()

local original_getenv = os.getenv
local original_test_mode = rawget(_G, "SR_TEST_MODE")
SR_TEST_MODE = nil
os.getenv = function() return nil end
check(Bridge.config_from_env().startup_accept_timeout == 0,
    "non-test environment must not block during install")
os.getenv = function(name)
    if name == "SR_TEST_MODE" then return "1" end
end
check(Bridge.config_from_env().startup_accept_timeout == 30,
    "test mode startup timeout default mismatch")
os.getenv = function(name)
    if name == "SR_TEST_MODE" then return "1" end
    if name == "SR_TEST_STARTUP_ACCEPT_TIMEOUT" then return "1.25" end
end
check(Bridge.config_from_env().startup_accept_timeout == 1.25,
    "startup timeout environment override mismatch")
os.getenv = original_getenv
SR_TEST_MODE = original_test_mode
io.open = original_io_open
bit = original_bit

print("bridge stub tests passed")
