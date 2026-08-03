-- SR automated-test bridge. This file has no effect until Bridge.install is called.
-- LuaJIT 5.1 compatible; transport is newline-delimited JSON over TCP.

local Bridge = {
    PROTOCOL_VERSION = 2,
}

local DEFAULTS = {
    host = "127.0.0.1",
    port = 24816,
    backlog = 1,
    lockstep = true,
    exclusive_input = true,
    headless = true,
    headless_only_when_connected = true,
    render_every = 1,
    seed = 1,
    max_episode_frames = 7200,
    max_repeat = 600,
    stop_on_player_hit = true,
    stop_on_stage_change = true,
    stop_on_no_enemies = true,
    only_visible = true,
    visible_margin = 32,
    max_commands_per_poll = 8,
    max_line_bytes = 1024 * 1024,
    max_pending_bytes = 16 * 1024 * 1024,
    allow_shutdown = false,
    session_id = nil,
    startup_accept_timeout = 0,
    source_root = "game/mod/SR_Subterrain_Reanimation_v100",
}

local TEST_MODE_STARTUP_ACCEPT_TIMEOUT = 30

local ACTION_NAMES = { "move_x", "move_y", "slow", "shoot", "spell" }
local COMMAND_NAMES = {
    "ping", "catalog", "reset", "reset_stage", "step", "observe",
    "display", "save_replay", "close", "shutdown",
}
local RUNTIME_SOURCE_FILES = {
    "root.lua",
    "_editor_output.lua",
    "compat/init.lua",
    "compat/combo.lua",
    "compat/gameplay.lua",
    "compat/spell_practice.lua",
    "compat/player/marisa.lua",
    "compat/player/reimu.lua",
    "compat/player/sakuya.lua",
    "compat/background/effects.lua",
    "compat/background/stage6bg.lua",
    "compat/background/stg2bg.lua",
    "compat/background/stg3bg.lua",
    "compat/background/stg4bg.lua",
    "compat/background/stg5bg.lua",
    "compat/background/stg6bg.lua",
    "compat/testing/bridge.lua",
    "compat/testing/init.lua",
}

local function copy_table(source)
    local result = {}
    for key, value in pairs(source or {}) do
        result[key] = value
    end
    return result
end

local function merged_config(config)
    local result = copy_table(DEFAULTS)
    for key, value in pairs(config or {}) do
        result[key] = value
    end
    return result
end

local function finite_number(value)
    return type(value) == "number" and value == value
        and value ~= math.huge and value ~= -math.huge
end

local function number_or_nil(value)
    if finite_number(value) then
        return value
    end
end

local function integer(value, default, minimum, maximum)
    value = tonumber(value)
    if not finite_number(value) then
        return default
    end
    value = math.floor(value)
    if minimum and value < minimum then
        value = minimum
    end
    if maximum and value > maximum then
        value = maximum
    end
    return value
end

local function nonnegative_number(value, default)
    value = tonumber(value)
    if not finite_number(value) then
        return default
    end
    return math.max(0, value)
end

local function string_value(value)
    if type(value) == "string" then
        return value
    end
end

local function engine_metric(name)
    local metric = rawget(_G, name)
    if type(metric) ~= "function" and type(lstg) == "table" then
        metric = lstg[name]
    end
    if type(metric) ~= "function" then
        return nil
    end
    local ok, value = pcall(metric)
    if ok then
        return number_or_nil(value)
    end
end

local function truthy(value)
    if type(value) == "boolean" then
        return value
    end
    if type(value) == "number" then
        return value ~= 0
    end
    if type(value) == "string" then
        value = string.lower(value)
        return value == "1" or value == "true" or value == "yes" or value == "on"
    end
    return false
end

local function crc32_file(path)
    local bit_library = rawget(_G, "bit")
    if type(bit_library) ~= "table" then
        local ok, value = pcall(require, "bit")
        if ok then bit_library = value end
    end
    if type(bit_library) ~= "table" then return nil, "bit library unavailable" end
    local stream, open_error = io.open(path, "rb")
    if not stream then return nil, open_error end
    local crc = -1
    while true do
        local block = stream:read(65536)
        if not block then break end
        for index = 1, #block do
            crc = bit_library.bxor(crc, string.byte(block, index))
            for _ = 1, 8 do
                if bit_library.band(crc, 1) ~= 0 then
                    crc = bit_library.bxor(bit_library.rshift(crc, 1), 0xEDB88320)
                else
                    crc = bit_library.rshift(crc, 1)
                end
            end
        end
    end
    stream:close()
    crc = bit_library.bnot(crc)
    if crc < 0 then crc = crc + 4294967296 end
    return string.format("%08x", crc)
end

local function verify_replay_frame_data(path, stages)
    if type(stages) ~= "table" or #stages < 1 then
        return nil, "replay has no frame data stages"
    end
    local stream, open_error = io.open(path, "rb")
    if not stream then
        return nil, "failed to open saved replay: " .. tostring(open_error)
    end
    local operation_ok, verified, verify_error, file_size, frame_bytes = pcall(
        function()
            local preceding_end = 0
            local total_frames = 0
            for index, stage_info in ipairs(stages) do
                local position = type(stage_info) == "table"
                    and stage_info.frameDataPosition or nil
                local count = type(stage_info) == "table"
                    and stage_info.frameCount or nil
                if type(position) ~= "number" or position < 0
                        or position ~= math.floor(position) then
                    return nil, "stage " .. index
                        .. " has an invalid frame data position"
                end
                if type(count) ~= "number" or count < 0
                        or count ~= math.floor(count) then
                    return nil, "stage " .. index .. " has an invalid frame count"
                end
                if position < preceding_end then
                    return nil, "stage " .. index .. " frame data overlaps prior data"
                end
                local seek_position, seek_error = stream:seek("set", position)
                if seek_position ~= position then
                    return nil, "failed to seek to stage " .. index
                        .. " frame data: " .. tostring(seek_error)
                end
                local remaining = count
                while remaining > 0 do
                    local requested = math.min(remaining, 65536)
                    local block, read_error = stream:read(requested)
                    if type(block) ~= "string" or #block ~= requested then
                        return nil, "stage " .. index .. " frame data is truncated: "
                            .. tostring(read_error or (count - remaining +
                                (type(block) == "string" and #block or 0))
                                .. "/" .. count .. " bytes")
                    end
                    remaining = remaining - requested
                end
                preceding_end = position + count
                total_frames = total_frames + count
            end
            local eof, seek_error = stream:seek("end", 0)
            if type(eof) ~= "number" then
                return nil, "failed to determine replay EOF: " .. tostring(seek_error)
            end
            if eof ~= preceding_end then
                return nil, "replay EOF mismatch: expected " .. preceding_end
                    .. ", got " .. eof
            end
            return true, nil, eof, total_frames
        end
    )
    local close_ok, close_error = pcall(stream.close, stream)
    if not operation_ok then
        return nil, "frame data verification failed: " .. tostring(verified)
    end
    if not verified then
        return nil, verify_error
    end
    if not close_ok then
        return nil, "failed to close replay after verification: "
            .. tostring(close_error)
    end
    return {
        file_size = file_size,
        frame_bytes = frame_bytes,
    }
end

local function runtime_identity(config, dependencies)
    if type(dependencies.runtime_identity) == "table" then
        return copy_table(dependencies.runtime_identity)
    end
    local identity = { source_crc32 = {} }
    local ffi_ok, ffi = pcall(require, "ffi")
    if ffi_ok and ffi then
        pcall(ffi.cdef, [[
            unsigned long __stdcall GetCurrentProcessId(void);
            unsigned long __stdcall GetModuleFileNameA(void *module, char *filename, unsigned long size);
        ]])
        local pid_ok, pid = pcall(function() return tonumber(ffi.C.GetCurrentProcessId()) end)
        if pid_ok then identity.process_id = pid end
        local path_ok, path = pcall(function()
            local buffer = ffi.new("char[4096]")
            local length = tonumber(ffi.C.GetModuleFileNameA(nil, buffer, 4096))
            if not length or length <= 0 then return nil end
            return ffi.string(buffer, length)
        end)
        if path_ok and path then
            identity.executable_path = path
            identity.executable_crc32 = crc32_file(path)
        end
    end
    local root = tostring(config.source_root or "")
    for _, relative in ipairs(RUNTIME_SOURCE_FILES) do
        local separator = (#root > 0 and string.sub(root, -1) ~= "/") and "/" or ""
        local checksum = crc32_file(root .. separator .. relative)
        identity.source_crc32[relative] = checksum
    end
    return identity
end

local function safe_close(socket_object)
    if socket_object and socket_object.close then
        pcall(socket_object.close, socket_object)
    end
end

local function json_array(cjson_library)
    local result = {}
    local array_mt = cjson_library and cjson_library.array_mt
    if type(array_mt) == "table" then
        setmetatable(result, array_mt)
    end
    return result
end

local function class_is(class_object, target)
    if type(target) ~= "table" then
        return false
    end
    local seen = {}
    while type(class_object) == "table" and not seen[class_object] do
        if class_object == target then
            return true
        end
        seen[class_object] = true
        class_object = class_object.base
    end
    return false
end

local function call_iterator(group)
    local iterator = rawget(_G, "ObjList")
    if type(iterator) ~= "function" and type(lstg) == "table" then
        iterator = lstg.ObjList
    end
    if type(iterator) ~= "function" then
        return function() end, nil, nil
    end
    return iterator(group)
end

local Instance = {}
Instance.__index = Instance

function Instance:_log(message)
    local logger = rawget(_G, "Print") or print
    pcall(logger, "[SR test bridge] " .. tostring(message))
end

function Instance:_open_server()
    local server, err = self.socket.bind(self.config.host, self.config.port, self.config.backlog)
    if not server then
        return nil, err
    end
    server:settimeout(0)
    self.server = server
    local ok, host, port = pcall(server.getsockname, server)
    if ok then
        self.bound_host = host or self.config.host
        self.bound_port = port or self.config.port
    else
        self.bound_host = self.config.host
        self.bound_port = self.config.port
    end
    return true
end

function Instance:_disconnect(reason)
    if self.replay_capture then
        local _, replay_error = self:_finish_replay_capture(false, "disconnect")
        if replay_error then
            self:_log("failed to save replay on disconnect: " .. tostring(replay_error))
        end
    end
    if self.client then
        safe_close(self.client)
    end
    rawset(_G, "SR_SAFETY_ZONE_CONTROLLER_STATE", nil)
    self.client = nil
    self.rx_partial = ""
    self.tx_buffer = ""
    self.pending = nil
    self.action = nil
    self.terminated = false
    self.termination_reason = nil
    self.episode_kind = nil
    self.episode_scenario = nil
    self.expected_stage = nil
    self.expected_stage_successor = nil
    self.expected_stage_menu = false
    if reason and reason ~= "closed" then
        self:_log("client disconnected: " .. tostring(reason))
    end
end

function Instance:_queue(value)
    if not self.client then
        return false
    end
    local ok, encoded = pcall(self.cjson.encode, value)
    if not ok then
        self:_log("JSON encode failed: " .. tostring(encoded))
        return false
    end
    self.tx_buffer = self.tx_buffer .. encoded .. "\n"
    if #self.tx_buffer > self.config.max_pending_bytes then
        self:_disconnect("outgoing buffer limit exceeded")
        return false
    end
    return true
end

function Instance:_flush()
    if not self.client or #self.tx_buffer == 0 then
        return
    end
    local sent, err, last = self.client:send(self.tx_buffer)
    local consumed = sent or last or 0
    if consumed > 0 then
        self.tx_buffer = string.sub(self.tx_buffer, consumed + 1)
    end
    if err and err ~= "timeout" then
        self:_disconnect(err)
    end
end

function Instance:_response(id, ok, fields)
    local result = { id = id, ok = ok }
    for key, value in pairs(fields or {}) do
        result[key] = value
    end
    self:_queue(result)
end

function Instance:_attach_client(client)
    if self.client or not client then
        return false
    end
    client:settimeout(0)
    if client.setoption then
        pcall(client.setoption, client, "tcp-nodelay", true)
    end
    self.client = client
    self.rx_partial = ""
    self.tx_buffer = ""
    self:_refresh_key_map()
    return true
end

function Instance:_accept()
    if self.client then
        return true
    end
    local client = self.server:accept()
    if not client then
        return false
    end
    return self:_attach_client(client)
end

function Instance:_wait_for_startup_client()
    local timeout = nonnegative_number(self.config.startup_accept_timeout, 0)
    if self.client or timeout <= 0 then
        return false
    end

    self:_log(string.format("waiting up to %.3g seconds for startup client", timeout))
    self.server:settimeout(timeout)
    local client, err = self.server:accept()
    self.server:settimeout(0)
    if client then
        self:_attach_client(client)
        self:_log("startup client connected")
        return true
    end
    if err and err ~= "timeout" then
        self:_log("startup accept failed: " .. tostring(err))
    end
    return false
end

function Instance:_read_commands()
    if not self.client then
        return
    end
    for _ = 1, self.config.max_commands_per_poll do
        local line, err, partial = self.client:receive("*l")
        if line then
            line = self.rx_partial .. line
            self.rx_partial = ""
            if #line > self.config.max_line_bytes then
                self:_response(nil, false, { error = "request exceeds max_line_bytes" })
            else
                local ok, request = pcall(self.cjson.decode, line)
                if ok and type(request) == "table" then
                    self:_handle_request(request)
                else
                    self:_response(nil, false, { error = "invalid JSON request" })
                end
            end
        elseif partial and #partial > 0 then
            self.rx_partial = self.rx_partial .. partial
            if #self.rx_partial > self.config.max_line_bytes then
                self:_disconnect("incoming line limit exceeded")
                return
            end
        end
        if err then
            if err == "closed" then
                self:_disconnect("closed")
            elseif err ~= "timeout" then
                self:_disconnect(err)
            end
            return
        end
    end
end

function Instance:_poll_network()
    self:_accept()
    self:_flush()
    self:_read_commands()
    self:_flush()
end

function Instance:_refresh_key_map()
    self.key_names_by_code = {}
    local keys = type(setting) == "table" and setting.keys or nil
    if type(keys) ~= "table" then
        return
    end
    for name, code in pairs(keys) do
        if type(code) == "number" then
            local names = self.key_names_by_code[code]
            if not names then
                names = {}
                self.key_names_by_code[code] = names
            end
            names[#names + 1] = name
        end
    end
end

function Instance:_key_state(vkey)
    local connected = self.client ~= nil
    local names = self.key_names_by_code and self.key_names_by_code[vkey] or nil
    if connected and names then
        local action_keys = self.action and self.action.keys or nil
        for _, name in ipairs(names) do
            if action_keys and action_keys[name] then
                return true
            end
        end
        if self.config.exclusive_input or self.action then
            return false
        end
    end
    return self.original_global_get_key(vkey)
end

local function normalize_action(action)
    action = type(action) == "table" and action or {}
    local move_x = tonumber(action.move_x) or 0
    local move_y = tonumber(action.move_y) or 0
    local keys = {
        left = move_x < -0.25,
        right = move_x > 0.25,
        down = move_y < -0.25,
        up = move_y > 0.25,
        slow = truthy(action.slow),
        shoot = truthy(action.shoot),
        spell = truthy(action.spell),
    }
    return {
        move_x = math.max(-1, math.min(1, move_x)),
        move_y = math.max(-1, math.min(1, move_y)),
        slow = keys.slow,
        shoot = keys.shoot,
        spell = keys.spell,
        keys = keys,
    }
end

function Instance:_seed(seed)
    seed = integer(seed, self.config.seed, 0)
    self.config.seed = seed
    math.randomseed(seed)
    local generator = rawget(_G, "ran")
    if generator and type(generator.Seed) == "function" then
        pcall(generator.Seed, generator, seed)
    end
    if type(lstg) == "table" then
        lstg.var = lstg.var or {}
        lstg.var.ran_seed = seed
        if type(lstg.nextvar) == "table" then
            lstg.nextvar.ran_seed = seed
        end
    end
    return seed
end

local WINDOWS_RESERVED_DEVICE_NAMES = {
    CON = true,
    PRN = true,
    AUX = true,
    NUL = true,
}

local function is_windows_reserved_device_name(value)
    local basename = string.match(value, "^([^.]*)") or value
    basename = string.upper(basename)
    return WINDOWS_RESERVED_DEVICE_NAMES[basename] == true
        or string.match(basename, "^COM[1-9]$") ~= nil
        or string.match(basename, "^LPT[1-9]$") ~= nil
end

local function normalized_replay_name(value)
    if value == nil then
        return nil
    end
    if type(value) ~= "string" then
        return nil, "replay_name must be a string"
    end
    if is_windows_reserved_device_name(value) then
        return nil, "replay_name uses a Windows reserved device basename"
    end
    if #value >= 4 and string.lower(string.sub(value, -4)) == ".rep" then
        value = string.sub(value, 1, -5)
    end
    if is_windows_reserved_device_name(value) then
        return nil, "replay_name uses a Windows reserved device basename"
    end
    if #value < 1 or #value > 96
            or string.sub(value, -1) == "."
            or not string.match(value, "^[A-Za-z0-9][A-Za-z0-9_.-]*$") then
        return nil, "replay_name must be 1-96 portable filename characters"
    end
    return value
end

local function replay_directory()
    local mod_name = type(setting) == "table" and setting.mod or "unknown_mod"
    mod_name = string.gsub(tostring(mod_name), "[^A-Za-z0-9_.-]", "_")
    return "userdata/replay/" .. mod_name .. "/analysis"
end

local function create_replay_directories(directory)
    local manager = type(lstg) == "table" and lstg.FileManager or nil
    if type(manager) ~= "table" or type(manager.CreateDirectory) ~= "function" then
        return nil, "lstg.FileManager.CreateDirectory is unavailable"
    end
    for _, path in ipairs({
        "userdata",
        "userdata/replay",
        string.match(directory, "^(.*)/analysis$") or directory,
        directory,
    }) do
        local ok, created, create_error = pcall(manager.CreateDirectory, path)
        if not ok then
            return nil, "failed to create replay directory: " .. tostring(created)
        end
        -- std::filesystem::create_directories returns false when the target
        -- already exists. LuaSTG forwards that value without treating it as
        -- an error, so verify existence before reporting a failure.
        if created == false and type(manager.DirectoryExist) == "function" then
            local exists_ok, exists = pcall(manager.DirectoryExist, path)
            if not exists_ok or not exists then
                return nil, "failed to create replay directory: "
                    .. tostring(create_error or path)
            end
        elseif created == false and create_error ~= nil then
            return nil, "failed to create replay directory: "
                .. tostring(create_error)
        end
    end
    return true
end

function Instance:_start_replay_capture(
        name, stage_name, seed, player_name, episode_kind)
    local replay_name, name_error = normalized_replay_name(name)
    if name_error then
        return nil, name_error
    end
    if not replay_name then
        return nil
    end
    if seed > 4294967295 then
        return nil, "replay seed exceeds the STGR uint32 limit"
    end
    if type(plus) ~= "table"
            or plus.ReplayFrameWriter == nil
            or type(plus.ReplayManager) ~= "table"
            or type(plus.ReplayManager.SaveReplayInfo) ~= "function"
            or type(plus.ReplayManager.ReadReplayInfo) ~= "function" then
        return nil, "THlib replay writer is unavailable"
    end
    local serializer = rawget(_G, "Serialize")
    if type(serializer) ~= "function" then
        return nil, "THlib Serialize is unavailable"
    end
    local directory = replay_directory()
    local _, directory_error = create_replay_directories(directory)
    if directory_error then
        return nil, directory_error
    end
    local writer_ok, writer = pcall(plus.ReplayFrameWriter)
    if not writer_ok or type(writer) ~= "table"
            or type(writer.Record) ~= "function"
            or type(writer.GetCount) ~= "function"
            or type(writer.CopyToFileStream) ~= "function" then
        return nil, "failed to create THlib replay writer: " .. tostring(writer)
    end
    local serialize_ok, stage_extend_info = pcall(serializer, lstg.var)
    if not serialize_ok or type(stage_extend_info) ~= "string" then
        return nil, "failed to serialize replay initial state: "
            .. tostring(stage_extend_info)
    end

    local path = directory .. "/" .. replay_name .. ".rep"
    self.replay_capture = {
        schema_version = 1,
        name = replay_name,
        path = path,
        stage_name = stage_name,
        random_seed = seed,
        player = player_name,
        episode_kind = episode_kind,
        stage_extend_info = stage_extend_info,
        stage_date = os.time(),
        started_at = os.time(),
        writer = writer,
    }
    self.last_replay = nil
    return {
        schema_version = 1,
        name = replay_name,
        path = path,
        stage_name = stage_name,
        random_seed = seed,
        player = player_name,
        episode_kind = episode_kind,
        saved = false,
    }
end

function Instance:_finish_replay_capture(finish, reason)
    local capture = self.replay_capture
    if not capture then
        return nil, "no replay capture is active"
    end
    local effective_finish = finish == true
    if effective_finish and (capture.episode_kind ~= "stage"
            or self.terminated ~= true
            or self.termination_reason ~= "stage_complete"
            or self.config.stop_on_player_hit ~= true) then
        return nil, "finish=true requires a zero-death final-stage completion"
    end
    if capture.record_error then
        self.replay_capture = nil
        return nil, capture.record_error
    end

    local frame_count = capture.writer:GetCount()
    local save_data = {
        gameName = type(setting) == "table" and tostring(setting.mod or "") or "",
        gameVersion = 1,
        userName = tostring(self.config.session_id or "stg-lab"),
        group_finish = effective_finish and 1 or 0,
        stages = {
            {
                stageName = capture.stage_name,
                stageExtendInfo = capture.stage_extend_info,
                score = type(lstg) == "table" and type(lstg.var) == "table"
                    and tonumber(lstg.var.score) or 0,
                randomSeed = capture.random_seed,
                stageTime = math.max(0, os.time() - capture.started_at),
                stageDate = capture.stage_date,
                stagePlayer = capture.player,
                frameData = capture.writer,
            },
        },
    }
    local save_ok, save_error = pcall(
        plus.ReplayManager.SaveReplayInfo,
        capture.path,
        save_data
    )
    if not save_ok then
        return nil, "THlib replay save failed: " .. tostring(save_error)
    end
    local read_ok, replay_info = pcall(
        plus.ReplayManager.ReadReplayInfo,
        capture.path
    )
    local stage_info = read_ok and type(replay_info) == "table"
        and type(replay_info.stages) == "table" and replay_info.stages[1] or nil
    local verified = type(stage_info) == "table"
        and replay_info.fileVersion == 1
        and replay_info.gameName == save_data.gameName
        and replay_info.gameVersion == save_data.gameVersion
        and replay_info.userName == save_data.userName
        and #replay_info.stages == 1
        and replay_info.group_finish == (effective_finish and 1 or 0)
        and stage_info.stageName == capture.stage_name
        and stage_info.stageExtendInfo == capture.stage_extend_info
        and stage_info.randomSeed == capture.random_seed
        and stage_info.stagePlayer == capture.player
        and stage_info.frameCount == frame_count
    if not verified then
        return nil, "saved replay failed THlib metadata verification: "
            .. tostring(read_ok and "metadata mismatch" or replay_info)
    end
    local frame_verification, frame_error = verify_replay_frame_data(
        capture.path,
        replay_info.stages
    )
    if not frame_verification then
        return nil, "saved replay failed frame data verification: "
            .. tostring(frame_error)
    end
    local checksum, checksum_error = crc32_file(capture.path)
    if not checksum then
        return nil, "saved replay checksum failed: " .. tostring(checksum_error)
    end
    local result = {
        schema_version = 1,
        name = capture.name,
        path = capture.path,
        stage_name = capture.stage_name,
        random_seed = capture.random_seed,
        player = capture.player,
        episode_kind = capture.episode_kind,
        frame_count = frame_count,
        finish = effective_finish,
        group_finish = effective_finish and 1 or 0,
        reason = tostring(reason or "requested"),
        saved = true,
        verified = true,
        file_size = frame_verification.file_size,
        frame_bytes_verified = frame_verification.frame_bytes,
        crc32 = checksum,
    }
    self.replay_capture = nil
    self.last_replay = result
    return result
end

function Instance:_record_replay_input()
    local capture = self.replay_capture
    if not capture or capture.record_error then
        return
    end
    local key_state = rawget(_G, "KeyState")
    if type(key_state) ~= "table" then
        capture.record_error = "THlib KeyState is unavailable during replay capture"
        return
    end
    local ok, err = pcall(capture.writer.Record, capture.writer, key_state)
    if not ok then
        capture.record_error = "THlib replay input recording failed: " .. tostring(err)
    end
end

local function resolve_player(player_name)
    if type(player_name) ~= "string" or player_name == "" then
        return type(lstg) == "table" and lstg.var and lstg.var.player_name or nil
    end
    if type(rawget(_G, player_name)) == "table" then
        return player_name
    end
    if type(player_list) == "table" then
        for _, entry in ipairs(player_list) do
            if player_name == entry[1] or player_name == entry[2] or player_name == entry[3] then
                return entry[2]
            end
        end
    end
end

local function replay_player_label(player_name)
    if type(player_list) == "table" then
        for _, entry in ipairs(player_list) do
            if type(entry) == "table" and entry[2] == player_name then
                return tostring(entry[3] or entry[1] or player_name)
            end
        end
    end
    return tostring(player_name)
end

local function replay_options_error(options)
    for _, key in ipairs({
        "lifeleft", "bomb", "power", "faith", "score",
        "player_protect_frames", "player_collidable", "player_ghost",
    }) do
        if options[key] ~= nil then
            return "native replay cannot reproduce reset option: " .. key
        end
    end
end

local function resolve_attack(scenario, attack, options)
    if type(scenario) ~= "string" or scenario == "" then
        return nil, nil, "scenario must be a boss class name"
    end
    local editor_classes = rawget(_G, "_editor_class")
    local boss_class = type(editor_classes) == "table" and editor_classes[scenario] or nil
    if type(boss_class) ~= "table" or type(boss_class.cards) ~= "table" then
        return nil, nil, "unknown boss scenario: " .. scenario
    end
    attack = integer(attack, nil, 1)
    if not attack then
        return nil, nil, "attack must be a positive integer"
    end
    if type(options) == "table" and truthy(options.attack_is_card_index) then
        if not boss_class.cards[attack] then
            return nil, nil, "card index is out of range"
        end
        return boss_class, attack
    end
    local ordinal = 0
    for card_index, card in ipairs(boss_class.cards) do
        if card.is_combat and (tonumber(card.t3) or 0) > 60 then
            ordinal = ordinal + 1
            if ordinal == attack then
                return boss_class, card_index
            end
        end
    end
    return nil, nil, "attack ordinal is out of range"
end

local function resolve_stage(stage_name)
    if type(stage_name) ~= "string" or stage_name == "" then
        return nil, "stage must be a registered stage name"
    end
    local source = rawget(_G, "SR_STAGE_TEST_CATALOG")
    if type(source) ~= "table" or type(source.stages) ~= "table" then
        return nil, "SR stage-test catalog is not registered"
    end
    for _, entry in ipairs(source.stages) do
        if type(entry) == "table" and entry.stage == stage_name then
            if type(stage) == "table" and type(stage.stages) == "table"
                    and stage.stages[stage_name] then
                return entry
            end
            return nil, "stage catalog entry is not registered: " .. stage_name
        end
    end
    return nil, "unknown stage scenario: " .. stage_name
end

local function stage_completion_contract(stage_entry)
    local source = rawget(_G, "SR_STAGE_TEST_CATALOG")
    local entries = type(source) == "table" and source.stages or nil
    local current_index = type(stage_entry) == "table"
        and integer(stage_entry.stage_index, nil, 1) or nil
    local difficulty = type(stage_entry) == "table"
        and string_value(stage_entry.difficulty) or nil
    if type(entries) ~= "table" or not current_index or not difficulty then
        return nil, false
    end

    local maximum_index = current_index
    local successor = nil
    local successor_count = 0
    for _, candidate in ipairs(entries) do
        if type(candidate) == "table" and candidate.difficulty == difficulty then
            local candidate_index = integer(candidate.stage_index, nil, 1)
            local candidate_name = string_value(candidate.stage)
            if candidate_index then
                maximum_index = math.max(maximum_index, candidate_index)
                if candidate_index == current_index + 1 and candidate_name then
                    successor = candidate_name
                    successor_count = successor_count + 1
                end
            end
        end
    end
    if successor_count == 1 then
        return successor, false
    end
    return nil, successor_count == 0 and current_index == maximum_index
end

function Instance:_reset(request)
    local options = type(request.options) == "table" and request.options or {}
    local replay_name, replay_name_error = normalized_replay_name(request.replay_name)
    if replay_name_error then
        return nil, replay_name_error
    end
    if replay_name then
        local options_error = replay_options_error(options)
        if options_error then
            return nil, options_error
        end
    end
    if self.replay_capture then
        local _, replay_error = self:_finish_replay_capture(false, "reset")
        if replay_error then
            return nil, "failed to finalize preceding replay: " .. replay_error
        end
    end
    local _, card_index, err = resolve_attack(request.scenario, request.attack, options)
    if err then
        return nil, err
    end
    local player_name = resolve_player(request.player)
    if not player_name then
        return nil, "unknown player: " .. tostring(request.player)
    end
    if type(stage) ~= "table" or type(stage.Set) ~= "function"
            or not stage.stages or not stage.stages["Spell Practice@Spell Practice"] then
        return nil, "Spell Practice stage is not registered"
    end

    lstg.var = lstg.var or {}
    lstg.var.player_name = player_name
    -- UI.lua treats numeric zero as truthy and indexes _sc_table[0].
    lstg.var.sc_index = nil
    lstg.var.sc_pr_data = {
        class_name = request.scenario,
        scene_index = card_index,
        include_previous = truthy(options.include_previous),
    }
    lstg.var.is_practice = true
    if options.hidden_route ~= nil then
        lstg.var.hidden_route = truthy(options.hidden_route)
    end
    stage.IsSCpractice = true
    if type(ext) == "table" then
        ext.pop_pause_menu = false
        ext.pause_menu_order = nil
        ext.rep_over = false
        if type(ext.pause_menu) == "table" then
            ext.pause_menu.kill = true
            if type(task) == "table" and type(task.Clear) == "function" then
                pcall(task.Clear, ext.pause_menu)
            end
        end
    end
    stage.Set("Spell Practice@Spell Practice", "none")
    local seed = self:_seed(request.seed)
    local replay, replay_error = self:_start_replay_capture(
        request.replay_name,
        "Spell Practice@Spell Practice",
        seed,
        replay_player_label(player_name),
        "attack"
    )
    if replay_error then
        return nil, replay_error
    end

    self.episode_frame = 0
    self.terminated = false
    self.termination_reason = nil
    self.seen_enemy = false
    self.expected_stage = "Spell Practice@Spell Practice"
    self.expected_stage_successor = nil
    self.expected_stage_menu = false
    self.episode_kind = "attack"
    self.episode_scenario = request.scenario
    self.pending_options = options
    return {
        episode_kind = "attack",
        scenario = request.scenario,
        attack = request.attack,
        card_index = card_index,
        seed = seed,
        player = player_name,
        replay = replay,
    }
end

function Instance:_reset_stage(request)
    local options = type(request.options) == "table" and request.options or {}
    local replay_name, replay_name_error = normalized_replay_name(request.replay_name)
    if replay_name_error then
        return nil, replay_name_error
    end
    if replay_name then
        local options_error = replay_options_error(options)
        if options_error then
            return nil, options_error
        end
    end
    if self.replay_capture then
        local _, replay_error = self:_finish_replay_capture(false, "reset")
        if replay_error then
            return nil, "failed to finalize preceding replay: " .. replay_error
        end
    end
    local stage_entry, err = resolve_stage(request.stage)
    if err then
        return nil, err
    end
    local player_name = resolve_player(request.player)
    if not player_name then
        return nil, "unknown player: " .. tostring(request.player)
    end
    if type(stage) ~= "table" or type(stage.Set) ~= "function" then
        return nil, "stage switching is unavailable"
    end
    local expected_stage_successor, expected_stage_menu =
        stage_completion_contract(stage_entry)
    if replay_name and not expected_stage_menu then
        return nil, "native replay capture supports only attacks and final stages"
    end

    lstg.var = lstg.var or {}
    lstg.var.player_name = player_name
    lstg.var.sc_index = nil
    lstg.var.sc_pr_data = nil
    lstg.var.is_practice = false
    stage.IsSCpractice = false
    if type(ext) == "table" then
        ext.pop_pause_menu = false
        ext.pause_menu_order = nil
        ext.rep_over = false
        if type(ext.pause_menu) == "table" then
            ext.pause_menu.kill = true
            if type(task) == "table" and type(task.Clear) == "function" then
                pcall(task.Clear, ext.pause_menu)
            end
        end
    end
    stage.Set(request.stage, "none")
    local seed = self:_seed(request.seed)
    local replay, replay_error = self:_start_replay_capture(
        request.replay_name,
        request.stage,
        seed,
        replay_player_label(player_name),
        "stage"
    )
    if replay_error then
        return nil, replay_error
    end

    self.episode_frame = 0
    self.terminated = false
    self.termination_reason = nil
    self.seen_enemy = false
    self.expected_stage = request.stage
    self.expected_stage_successor = expected_stage_successor
    self.expected_stage_menu = expected_stage_menu
    self.episode_kind = "stage"
    self.episode_scenario = request.stage
    self.pending_options = options
    return {
        episode_kind = "stage",
        stage = request.stage,
        stage_index = integer(stage_entry.stage_index, nil, 1),
        difficulty = string_value(stage_entry.difficulty),
        seed = seed,
        player = player_name,
        replay = replay,
    }
end

function Instance:_catalog()
    local source = rawget(_G, "SR_SPELL_PRACTICE_CATALOG")
    if type(source) ~= "table" or type(source.scenarios) ~= "table" then
        return nil, "SR spell-practice catalog is not registered"
    end
    local scenarios = json_array(self.cjson)
    local attacks = json_array(self.cjson)
    local stages = json_array(self.cjson)
    for _, raw_scenario in ipairs(source.scenarios) do
        local scenario = string_value(raw_scenario.scenario)
        local raw_attacks = raw_scenario.attacks
        if scenario and type(raw_attacks) == "table" then
            local scenario_attacks = json_array(self.cjson)
            for _, raw_attack in ipairs(raw_attacks) do
                local attack = integer(raw_attack.attack, nil, 1)
                local card_index = integer(raw_attack.card_index, nil, 1)
                if attack and card_index then
                    local entry = {
                        episode_kind = "attack",
                        scenario = scenario,
                        attack = attack,
                        card_index = card_index,
                        label = string_value(raw_attack.label),
                        completion_reason = "attack_complete",
                    }
                    scenario_attacks[#scenario_attacks + 1] = entry
                    attacks[#attacks + 1] = entry
                end
            end
            scenarios[#scenarios + 1] = {
                scenario = scenario,
                label = string_value(raw_scenario.label),
                attack_count = #scenario_attacks,
                attacks = scenario_attacks,
            }
        end
    end
    local stage_source = rawget(_G, "SR_STAGE_TEST_CATALOG")
    if type(stage_source) == "table" and type(stage_source.stages) == "table" then
        for _, raw_stage in ipairs(stage_source.stages) do
            local stage_name = type(raw_stage) == "table"
                and string_value(raw_stage.stage) or nil
            local stage_index = type(raw_stage) == "table"
                and integer(raw_stage.stage_index, nil, 1) or nil
            if stage_name and stage_index then
                stages[#stages + 1] = {
                    episode_kind = "stage",
                    stage = stage_name,
                    stage_index = stage_index,
                    difficulty = string_value(raw_stage.difficulty),
                    label = string_value(raw_stage.label),
                    completion_reason = "stage_complete",
                }
            end
        end
    end
    return {
        schema_version = integer(source.schema_version, 1, 1),
        scenario_count = #scenarios,
        attack_count = #attacks,
        stage_count = #stages,
        scenarios = scenarios,
        attacks = attacks,
        stages = stages,
    }
end

function Instance:_handle_request(request)
    local id = request.id
    if id == nil then
        self:_response(nil, false, { error = "request id is required" })
        return
    end
    local command = request.command
    if type(command) ~= "string" then
        self:_response(id, false, { error = "command is required" })
        return
    end
    if self.pending then
        self:_response(id, false, { error = "another reset/step request is still running" })
        return
    end

    if command == "catalog" then
        local catalog, err = self:_catalog()
        if not catalog then
            self:_response(id, false, { error = err })
        else
            self:_response(id, true, { catalog = catalog })
        end
    elseif command == "reset" then
        local info, err = self:_reset(request)
        if not info then
            self:_response(id, false, { error = err })
            return
        end
        rawset(_G, "SR_SAFETY_ZONE_CONTROLLER_STATE", nil)
        self.action = normalize_action(nil)
        self.pending = { id = id, command = command, remaining = 1, reset = info }
    elseif command == "reset_stage" then
        local info, err = self:_reset_stage(request)
        if not info then
            self:_response(id, false, { error = err })
            return
        end
        rawset(_G, "SR_SAFETY_ZONE_CONTROLLER_STATE", nil)
        self.action = normalize_action(nil)
        self.pending = { id = id, command = command, remaining = 1, reset = info }
    elseif command == "step" then
        if self.terminated then
            self:_response(id, false, { error = "episode is terminated; reset is required" })
            return
        end
        self.action = normalize_action(request.action)
        if type(request.controller_overlay_state) == "table" then
            rawset(
                _G,
                "SR_SAFETY_ZONE_CONTROLLER_STATE",
                request.controller_overlay_state
            )
        end
        self.pending = {
            id = id,
            command = command,
            remaining = integer(request["repeat"], 1, 1, self.config.max_repeat),
        }
    elseif command == "observe" then
        self:_response(id, true, { observation = self:collect_observation() })
    elseif command == "display" then
        if type(request.render) ~= "boolean" then
            self:_response(id, false, { error = "display render must be a Boolean" })
        else
            local render_every = integer(request.every, self.config.render_every, 1, 600)
            self.config.headless = not request.render
            self.config.render_every = render_every
            self:_response(id, true, { render = request.render, every = render_every })
        end
    elseif command == "save_replay" then
        if type(request.finish) ~= "boolean" then
            self:_response(id, false, { error = "save_replay finish must be a Boolean" })
        elseif request.reason ~= nil
                and (type(request.reason) ~= "string" or request.reason == "") then
            self:_response(id, false, { error = "save_replay reason must be a nonempty string" })
        else
            local replay, replay_error = self:_finish_replay_capture(
                request.finish,
                request.reason or self.termination_reason or "requested"
            )
            if not replay then
                self:_response(id, false, { error = replay_error })
            else
                self:_response(id, true, { replay = replay })
            end
        end
    elseif command == "ping" then
        self:_response(id, true, {
            protocol = self.PROTOCOL_VERSION,
            frame = self.frame,
            action_names = ACTION_NAMES,
            commands = COMMAND_NAMES,
            session_id = self.config.session_id,
            process_nonce = self.process_nonce,
            runtime_identity = self.runtime_identity,
        })
    elseif command == "close" then
        self:_response(id, true)
        self.close_after_flush = true
    elseif command == "shutdown" then
        if not self.config.allow_shutdown then
            self:_response(id, false, { error = "shutdown is disabled" })
        else
            self:_response(id, true)
            self.shutdown_requested = true
        end
    else
        self:_response(id, false, { error = "unknown command: " .. command })
    end
end

function Instance:_class_name(object)
    if not self.class_names or self.frame % 300 == 0 then
        self.class_names = self.class_names or {}
        local editor_classes = rawget(_G, "_editor_class")
        if type(editor_classes) == "table" then
            for name, class_object in pairs(editor_classes) do
                if type(name) == "string" and type(class_object) == "table" then
                    self.class_names[class_object] = name
                end
            end
        end
    end
    return self.class_names[object.class] or self.class_names[object.logclass]
end

function Instance:_laser_kind(object)
    if class_is(object.class, rawget(_G, "laser_bent")) then
        return "bent_laser"
    end
    if class_is(object.class, rawget(_G, "laser")) then
        return "straight_laser"
    end
    if object.data and object.listx and object.listy then
        return "bent_laser"
    end
    if finite_number(object.l1) and finite_number(object.l2)
            and finite_number(object.l3) and finite_number(object.w) then
        return "straight_laser"
    end
end

function Instance:_visible(object, laser_kind)
    if not self.config.only_visible then
        return true
    end
    if object.hide == true or (object.status and object.status ~= "normal") then
        return false
    end
    local world = type(lstg) == "table" and lstg.world or nil
    if type(world) ~= "table" then
        return true
    end
    local x, y = number_or_nil(object.x), number_or_nil(object.y)
    if not x or not y then
        return false
    end
    local margin = self.config.visible_margin
    local radius = math.max(math.abs(tonumber(object.a) or 0), math.abs(tonumber(object.b) or 0), margin)
    local min_x, max_x, min_y, max_y = x - radius, x + radius, y - radius, y + radius
    if laser_kind == "straight_laser" then
        local length = (tonumber(object.l1) or 0) + (tonumber(object.l2) or 0) + (tonumber(object.l3) or 0)
        local angle = (tonumber(object.rot) or 0) * math.pi / 180
        local end_x, end_y = x + length * math.cos(angle), y + length * math.sin(angle)
        min_x, max_x = math.min(x, end_x) - radius, math.max(x, end_x) + radius
        min_y, max_y = math.min(y, end_y) - radius, math.max(y, end_y) + radius
    elseif laser_kind == "bent_laser" then
        for slot, px in pairs(object.listx or {}) do
            local py = object.listy[slot]
            if finite_number(px) and finite_number(py) then
                min_x, max_x = math.min(min_x, px - radius), math.max(max_x, px + radius)
                min_y, max_y = math.min(min_y, py - radius), math.max(max_y, py + radius)
            end
        end
    end
    return max_x >= world.l and min_x <= world.r and max_y >= world.b and min_y <= world.t
end

local function put_number(result, key, value)
    value = number_or_nil(value)
    if value ~= nil then
        result[key] = value
    end
end

function Instance:_object_record(id, object, kind, laser_kind)
    local result = { id = id, kind = laser_kind or kind }
    for _, key in ipairs({
        "x", "y", "dx", "dy", "vx", "vy", "ax", "ay", "rot", "omiga",
        "a", "b", "hscale", "vscale", "timer", "layer", "hp", "maxhp",
        "dmg", "index", "_index", "alpha", "w", "w0", "l1", "l2", "l3", "l",
    }) do
        put_number(result, key, object[key])
    end
    result.collidable = object.colli == true
    result.rect = object.rect == true
    result.protected = object.protect == true or (tonumber(object.protect) or 0) > 0
    result.class_name = self:_class_name(object)
    result.image = string_value(object.img)
    result.status = string_value(object.status)
    if finite_number(object.vx) and finite_number(object.vy) then
        result.speed = math.sqrt(object.vx * object.vx + object.vy * object.vy)
    end
    if laser_kind == "bent_laser" then
        local points = json_array(self.cjson)
        for slot, px in pairs(object.listx or {}) do
            local py = object.listy[slot]
            if finite_number(slot) and finite_number(px) and finite_number(py) then
                points[#points + 1] = { slot = slot, x = px, y = py }
            end
        end
        table.sort(points, function(a, b) return a.slot < b.slot end)
        result.points = points
    end
    return result
end

function Instance:_collect_group(group, kind)
    local records = json_array(self.cjson)
    local lasers = json_array(self.cjson)
    for id, object in call_iterator(group) do
        local laser_kind = self:_laser_kind(object)
        if self:_visible(object, laser_kind) then
            local record = self:_object_record(id, object, kind, laser_kind)
            records[#records + 1] = record
            if laser_kind then
                lasers[#lasers + 1] = record
            end
        end
    end
    return records, lasers
end

function Instance:_player_record()
    local player_object = type(lstg) == "table" and lstg.player or rawget(_G, "player")
    if type(player_object) ~= "table" then
        return nil
    end
    local id
    local group_player = rawget(_G, "GROUP_PLAYER") or 4
    for object_id, object in call_iterator(group_player) do
        if object == player_object then
            id = object_id
            break
        end
    end
    local result = self:_object_record(id, player_object, "player")
    for _, key in ipairs({ "death", "protect", "hspeed", "lspeed", "slow", "nextspell", "nextshoot" }) do
        put_number(result, key, player_object[key])
    end
    result.locked = player_object.lock == true
    result.dialog = player_object.dialog == true
    return result
end

function Instance:collect_observation()
    local enemy_bullets, bullet_lasers = self:_collect_group(rawget(_G, "GROUP_ENEMY_BULLET") or 1, "enemy_bullet")
    local enemies, enemy_lasers = self:_collect_group(rawget(_G, "GROUP_ENEMY") or 2, "enemy")
    local indestructibles, indes_lasers = self:_collect_group(rawget(_G, "GROUP_INDES") or 5, "indestructible")
    local nontjt, nontjt_lasers = self:_collect_group(rawget(_G, "GROUP_NONTJT") or 7, "nontjt_enemy")
    local lasers = json_array(self.cjson)
    for _, source in ipairs({ bullet_lasers, enemy_lasers, indes_lasers, nontjt_lasers }) do
        for _, record in ipairs(source) do
            lasers[#lasers + 1] = record
        end
    end

    local current_stage = type(stage) == "table" and stage.current_stage or nil
    local world = type(lstg) == "table" and lstg.world or {}
    local vars = type(lstg) == "table" and lstg.var or {}
    local observation = {
        protocol = self.PROTOCOL_VERSION,
        frame = self.frame,
        episode_frame = self.episode_frame,
        terminated = self.terminated,
        termination_reason = self.termination_reason,
        stage = {
            name = current_stage and (current_stage.stage_name or current_stage.name) or nil,
            is_menu = current_stage and current_stage.is_menu == true or false,
            timer = current_stage and number_or_nil(current_stage.timer) or nil,
            episode_kind = self.episode_kind,
            scenario = self.episode_scenario
                or (vars.sc_pr_data and vars.sc_pr_data.class_name or nil),
            card_index = vars.sc_pr_data and vars.sc_pr_data.scene_index or nil,
        },
        world = {},
        player = self:_player_record(),
        enemy_bullets = enemy_bullets,
        enemies = enemies,
        nontjt_enemies = nontjt,
        indestructibles = indestructibles,
        lasers = lasers,
        resources = {},
        performance = {
            native_fps = engine_metric("GetFPS"),
            object_count = engine_metric("GetnObj"),
        },
    }
    local visualizer = rawget(_G, "SafetyZoneVisualizer")
    if type(visualizer) == "table"
            and type(visualizer.getRuntimeStatus) == "function" then
        local ok, status = pcall(visualizer.getRuntimeStatus)
        if ok and type(status) == "table" then
            observation.safety_zone_overlay = status
        end
    end
    for _, key in ipairs({ "l", "r", "b", "t", "pl", "pr", "pb", "pt" }) do
        put_number(observation.world, key, world[key])
    end
    for _, key in ipairs({ "lifeleft", "bomb", "power", "faith", "graze", "score" }) do
        put_number(observation.resources, key, vars[key])
    end
    observation.counts = {
        enemy_bullets = #enemy_bullets,
        enemies = #enemies,
        nontjt_enemies = #nontjt,
        indestructibles = #indestructibles,
        lasers = #lasers,
    }
    return observation
end

function Instance:_apply_pending_options()
    local options = self.pending_options
    if type(options) ~= "table" then
        return
    end
    for _, key in ipairs({ "lifeleft", "bomb", "power", "faith", "score" }) do
        if finite_number(options[key]) then
            lstg.var[key] = options[key]
        end
    end
    local protect_frames = integer(options.player_protect_frames, nil, 0)
    local player_object = type(lstg) == "table" and lstg.player or rawget(_G, "player")
    if protect_frames and type(player_object) == "table" then
        player_object.protect = math.max(tonumber(player_object.protect) or 0, protect_frames)
    end
    -- Training-field capture can remove the player from collision handling so
    -- hazards pass through without being deleted. These options exist only on
    -- the opt-in test bridge; strict validation resets use the normal group.
    if options.player_collidable ~= nil and type(player_object) == "table" then
        player_object.colli = truthy(options.player_collidable)
    end
    if options.player_ghost ~= nil and type(player_object) == "table" then
        player_object.group = truthy(options.player_ghost)
            and (rawget(_G, "GROUP_GHOST") or 0)
            or (rawget(_G, "GROUP_PLAYER") or 4)
    end
    self.pending_options = nil
end

function Instance:_has_enemy_objects()
    for _ in call_iterator(rawget(_G, "GROUP_ENEMY") or 2) do
        return true
    end
    for _ in call_iterator(rawget(_G, "GROUP_NONTJT") or 7) do
        return true
    end
    return false
end

function Instance:_check_stop(observation)
    if self.config.max_episode_frames and self.episode_frame >= self.config.max_episode_frames then
        return "time_limit"
    end
    local player_state = observation.player
    if self.config.stop_on_player_hit and player_state and (player_state.death or 0) > 0 then
        return "player_hit"
    end
    if self.config.stop_on_stage_change and self.expected_stage
            and observation.stage.name ~= self.expected_stage then
        if self.episode_kind == "stage" and self.seen_enemy then
            local reached_successor = self.expected_stage_successor
                and observation.stage.name == self.expected_stage_successor
            local reached_final_menu = self.expected_stage_menu
                and observation.stage.is_menu == true
            if reached_successor or reached_final_menu then
                return "stage_complete"
            end
        end
        return "stage_changed"
    end
    -- Stop state uses the authoritative object pool. Visible observations can
    -- legitimately lose a boss for a frame while it repositions off-screen.
    if self:_has_enemy_objects() then
        self.seen_enemy = true
    elseif self.episode_kind ~= "stage"
            and self.config.stop_on_no_enemies and self.seen_enemy then
        return "attack_complete"
    end
end

function Instance:_frame(...)
    self:_poll_network()
    if self.close_after_flush then
        self.close_after_flush = false
        self:_flush()
        self:_disconnect("closed")
    end
    if self.shutdown_requested then
        self:_flush()
        return true
    end
    if not self.client or not self.pending then
        if not self.client or not self.config.lockstep then
            return self.original_frame(...)
        end
        return false
    end

    self:_refresh_key_map()
    if type(stage) == "table" and stage.next_stage then
        self:_seed(self.config.seed)
    end
    local engine_exit = self.original_frame(...)
    self.frame = self.frame + 1
    self.episode_frame = self.episode_frame + 1
    self:_apply_pending_options()

    local observation = self:collect_observation()
    local reason = self:_check_stop(observation)
    if reason or engine_exit then
        self.terminated = true
        self.termination_reason = reason or "engine_exit"
        observation.terminated = true
        observation.termination_reason = self.termination_reason
    end
    self.pending.remaining = self.pending.remaining - 1
    if self.pending.remaining <= 0 or self.terminated then
        local fields = { observation = observation }
        if self.pending.reset then
            fields.reset = self.pending.reset
        end
        self:_response(self.pending.id, true, fields)
        self.pending = nil
        self.action = nil
    end
    self:_flush()
    return engine_exit
end

function Instance:_render(...)
    local suppress = self.config.headless
        and (not self.config.headless_only_when_connected or self.client ~= nil)
    if suppress then
        return
    end
    -- LuaSTG clears and presents the swap-chain buffer even when this callback
    -- returns without drawing. Visible lockstep must therefore redraw the
    -- current logical state on every native render pass, including while it is
    -- waiting for Python. render_every remains accepted for protocol
    -- compatibility, but it cannot safely suppress drawing at the Lua layer.
    return self.original_render(...)
end

function Instance:install()
    if self.installed then
        return true
    end
    if type(rawget(_G, "FrameFunc")) ~= "function"
            or type(rawget(_G, "RenderFunc")) ~= "function"
            or type(rawget(_G, "GetKeyState")) ~= "function"
            or type(rawget(_G, "GetInput")) ~= "function" then
        return nil, "FrameFunc, RenderFunc, GetKeyState, and GetInput must exist before bridge installation"
    end
    self.original_frame = FrameFunc
    self.original_render = RenderFunc
    self.original_global_get_key = GetKeyState
    self.original_lstg_get_key = type(lstg) == "table" and lstg.GetKeyState or nil
    self.original_get_input = GetInput
    self:_refresh_key_map()

    local instance = self
    self.frame_wrapper = function(...) return instance:_frame(...) end
    self.render_wrapper = function(...) return instance:_render(...) end
    self.key_wrapper = function(vkey) return instance:_key_state(vkey) end
    self.get_input_wrapper = function(...)
        local result = instance.original_get_input(...)
        instance:_record_replay_input()
        return result
    end
    FrameFunc = self.frame_wrapper
    RenderFunc = self.render_wrapper
    GetKeyState = self.key_wrapper
    GetInput = self.get_input_wrapper
    if type(lstg) == "table" then
        lstg.GetKeyState = self.key_wrapper
    end
    self:_seed(self.config.seed)
    self.installed = true
    self:_wait_for_startup_client()
    return true
end

function Instance:uninstall()
    if self.installed then
        if FrameFunc == self.frame_wrapper then FrameFunc = self.original_frame end
        if RenderFunc == self.render_wrapper then RenderFunc = self.original_render end
        if GetKeyState == self.key_wrapper then GetKeyState = self.original_global_get_key end
        if GetInput == self.get_input_wrapper then GetInput = self.original_get_input end
        if type(lstg) == "table" and lstg.GetKeyState == self.key_wrapper then
            lstg.GetKeyState = self.original_lstg_get_key
        end
    end
    self:_disconnect("uninstall")
    safe_close(self.server)
    self.server = nil
    self.installed = false
    if Bridge.active == self then
        Bridge.active = nil
    end
end

function Bridge.new(config, dependencies)
    dependencies = dependencies or {}
    local socket_library = dependencies.socket
    local cjson_library = dependencies.cjson
    if not socket_library then
        local ok, value = pcall(require, "socket")
        if not ok then return nil, "LuaSocket unavailable: " .. tostring(value) end
        socket_library = value
    end
    if not cjson_library then
        local ok, value = pcall(require, "cjson")
        if not ok then return nil, "cjson unavailable: " .. tostring(value) end
        cjson_library = value
    end
    local instance = setmetatable({
        PROTOCOL_VERSION = Bridge.PROTOCOL_VERSION,
        config = merged_config(config),
        socket = socket_library,
        cjson = cjson_library,
        frame = 0,
        episode_frame = 0,
        rx_partial = "",
        tx_buffer = "",
    }, Instance)
    local now
    if type(socket_library.gettime) == "function" then
        local time_ok, value = pcall(socket_library.gettime)
        if time_ok and finite_number(value) then
            now = value
        end
    end
    now = now or ((os and os.time and os.time()) or 0)
    instance.process_nonce = string.format("%s@%.6f", tostring(instance), now)
    instance.runtime_identity = runtime_identity(instance.config, dependencies)
    local ok, err = instance:_open_server()
    if not ok then return nil, "failed to bind test bridge: " .. tostring(err) end
    return instance
end

function Bridge.install(config, dependencies)
    if Bridge.active and Bridge.active.installed then
        return Bridge.active
    end
    local instance, err = Bridge.new(config, dependencies)
    if not instance then return nil, err end
    local ok, install_err = instance:install()
    if not ok then
        safe_close(instance.server)
        return nil, install_err
    end
    Bridge.active = instance
    instance:_log(string.format("listening on %s:%s", instance.bound_host, instance.bound_port))
    return instance
end

function Bridge.config_from_env()
    local getenv = os and os.getenv
    local function env(name)
        if not getenv then return nil end
        local ok, value = pcall(getenv, name)
        if ok then return value end
    end
    local test_mode = truthy(rawget(_G, "SR_TEST_MODE")) or truthy(env("SR_TEST_MODE"))
    local startup_accept_default = test_mode and TEST_MODE_STARTUP_ACCEPT_TIMEOUT
        or DEFAULTS.startup_accept_timeout
    return {
        host = env("SR_TEST_HOST") or DEFAULTS.host,
        port = integer(env("SR_TEST_PORT"), DEFAULTS.port, 0, 65535),
        seed = integer(env("SR_TEST_SEED"), DEFAULTS.seed, 0),
        headless = env("SR_TEST_HEADLESS") == nil or truthy(env("SR_TEST_HEADLESS")),
        lockstep = env("SR_TEST_LOCKSTEP") == nil or truthy(env("SR_TEST_LOCKSTEP")),
        allow_shutdown = truthy(env("SR_TEST_ALLOW_SHUTDOWN")),
        session_id = string_value(env("SR_TEST_SESSION_ID")),
        max_episode_frames = integer(env("SR_TEST_MAX_FRAMES"), DEFAULTS.max_episode_frames, 1),
        startup_accept_timeout = nonnegative_number(
            env("SR_TEST_STARTUP_ACCEPT_TIMEOUT"), startup_accept_default),
        source_root = env("SR_TEST_SOURCE_ROOT") or DEFAULTS.source_root,
    }
end

Bridge.defaults = copy_table(DEFAULTS)
Bridge.Instance = Instance
SRTestBridge = Bridge

return Bridge
