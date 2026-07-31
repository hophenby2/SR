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
GROUP_PLAYER = 4
GROUP_INDES = 5
GROUP_NONTJT = 7

setting = {
    keys = { left = 1, right = 2, up = 3, down = 4, slow = 5, shoot = 6, spell = 7 },
}

laser = { is_class = true }
local laser_child = { is_class = true, base = laser }
local player_object = {
    x = 0, y = -100, vx = 0, vy = 0, a = 2, b = 2,
    colli = true, status = "normal", death = 0, protect = 0,
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

local old_frame_count = 0
function FrameFunc()
    old_frame_count = old_frame_count + 1
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
}

local spell_stage = { stage_name = "Spell Practice@Spell Practice", is_menu = false, timer = 0 }
stage = {
    stages = { ["Spell Practice@Spell Practice"] = spell_stage },
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
    options = { lifeleft = 5, player_protect_frames = 500 },
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
encoded_values = {}

FrameFunc()
check(old_frame_count == 1, "reset must advance exactly one frame")
check(seed_value == 42, "reset seed was not applied")
check(lstg.var.sc_pr_data.scene_index == 4, "attack ordinal did not resolve to card index")
check(lstg.var.sc_index == nil, "direct spell-practice reset must not leave a zero menu index")
check(lstg.var.lifeleft == 5, "reset resource override was not applied")
check(player_object.protect == 500, "player protection override was not applied")
check(encoded_values[#encoded_values].id == 1 and encoded_values[#encoded_values].ok, "missing reset response")
check(encoded_values[#encoded_values].observation.counts.enemy_bullets == 1, "visibility filter failed")
check(encoded_values[#encoded_values].observation.counts.lasers == 1, "laser classification failed")
check(encoded_values[#encoded_values].observation.performance.native_fps == 47.5,
    "native FPS telemetry mismatch")
check(encoded_values[#encoded_values].observation.performance.object_count == 123,
    "native object-count telemetry mismatch")
check(getmetatable(encoded_values[#encoded_values].observation.nontjt_enemies) == fake_cjson.array_mt,
    "empty observation arrays must retain JSON array identity")

requests.STEP = {
    id = 2, command = "step",
    action = { move_x = 1, move_y = -1, slow = true, shoot = true, spell = false },
    ["repeat"] = 3,
}
client.input[#client.input + 1] = "STEP"
FrameFunc()
check(#encoded_values == 1, "step responded before repeat completed")
FrameFunc()
check(#encoded_values == 1, "step responded before final repeat")
FrameFunc()
check(old_frame_count == 4, "repeat did not advance three frames")
check(player_object.x == 3 and player_object.y == -103, "action was not held across repeat")
check(encoded_values[#encoded_values].id == 2, "step response id mismatch")
check(encoded_values[#encoded_values].observation.episode_frame == 4, "episode frame mismatch")

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
instance:_disconnect("closed")
check(GetKeyState(99) == true, "native key input did not resume after disconnect")
RenderFunc()
check(old_render_count == 5, "rendering did not resume after disconnect")

instance:uninstall()
check(FrameFunc ~= instance.frame_wrapper, "FrameFunc was not restored")
check(GetKeyState == native_get_key, "GetKeyState was not restored")

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

print("bridge stub tests passed")
