-- Safe to Include unconditionally: installation only happens in explicit test mode.
if not SRTestBridge then
    Include "compat/testing/bridge.lua"
end

local function test_mode_enabled()
    if rawget(_G, "SR_TEST_MODE") == true then
        return true
    end
    if os and os.getenv then
        local ok, value = pcall(os.getenv, "SR_TEST_MODE")
        if ok and type(value) == "string" then
            value = string.lower(value)
            return value == "1" or value == "true" or value == "yes" or value == "on"
        end
    end
    return false
end

if test_mode_enabled() and SRTestBridge and not SRTestBridge.active then
    local instance, err = SRTestBridge.install(SRTestBridge.config_from_env())
    if not instance then
        error("failed to enable SR test bridge: " .. tostring(err))
    end
end

return SRTestBridge
