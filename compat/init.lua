Include "compat/combo.lua"
Include "compat/background/effects.lua"
Include "compat/player/reimu.lua"
Include "compat/player/marisa.lua"
Include "compat/player/sakuya.lua"
Include "compat/gameplay.lua"
Include "compat/background/stg2bg.lua"
Include "compat/background/stg3bg.lua"
Include "compat/background/stg4bg.lua"
Include "compat/background/stg5bg.lua"
Include "compat/background/stg6bg.lua"
Include "compat/background/stage6bg.lua"

local players = {
    { "Hakurei Reimu", "reimu_player", "Reimu" },
    { "Kirisame Marisa", "marisa_player", "Marisa" },
    { "Izayoi Sakuya", "sakuya_player", "Sakuya" },
}

for _, entry in ipairs(players) do
    local found = false
    for _, registered in ipairs(player_list) do
        if registered[1] == entry[1] then
            registered[2] = entry[2]
            registered[3] = entry[3]
            found = true
            break
        end
    end
    if not found then
        AddPlayerToPlayerList(entry[1], entry[2], entry[3])
    end
end
