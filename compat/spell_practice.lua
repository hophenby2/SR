local practice_bosses = {
    { "koishi1:Normal", "Normal 1 Mid" },
    { "yamame:Normal", "Normal 1 Boss" },
    { "koishi2:Normal", "Normal 2 Mid" },
    { "koishi:Normal", "Normal 2 Boss" },
    { "orin3:Normal", "Normal 3 Mid" },
    { "satori:Normal", "Normal 3 Boss" },
    { "orin41:Normal", "Normal 4 Mid A" },
    { "orin42:Normal", "Normal 4 Mid B" },
    { "orin:Normal", "Normal 4 Boss" },
    { "okuu:Normal", "Normal 5 Boss" },

    { "koishi1:Lunatic", "Lunatic 1 Mid" },
    { "yamame:Lunatic", "Lunatic 1 Boss" },
    { "koishi2:Lunatic", "Lunatic 2 Mid" },
    { "koishi:Lunatic", "Lunatic 2 Boss" },
    { "orin3:Lunatic", "Lunatic 3 Mid" },
    { "satori:Lunatic", "Lunatic 3 Boss" },
    { "orin41:Lunatic", "Lunatic 4 Mid A" },
    { "orin42:Lunatic", "Lunatic 4 Mid B" },
    { "orin:Lunatic", "Lunatic 4 Boss" },
    { "orin5h:Lunatic", "Lunatic 5 Hidden" },
    { "okuu:Lunatic", "Lunatic 5 Boss" },
    { "okuu_ex:Lunatic", "Lunatic 5 Extra" },
}

local practice_catalog = {
    schema_version = 1,
    scenarios = {},
}

local stage_test_catalog = {
    schema_version = 1,
    stages = {},
}

for _, practice_boss in ipairs(practice_bosses) do
    local class_name = practice_boss[1]
    local label = practice_boss[2]
    local boss_class = assert(_editor_class[class_name], "missing spell-practice boss class: " .. class_name)
    local attack_number = 0
    local attacks = {}

    for card_index, card in ipairs(boss_class.cards) do
        -- SR's generated one-second combat cards are end markers, not attacks.
        if card.is_combat and card.t3 > 60 then
            attack_number = attack_number + 1
            table.insert(_sc_table, {
                class_name,
                string.format("%s #%d", label, attack_number),
                card,
                card_index,
                false,
            })
            table.insert(attacks, {
                attack = attack_number,
                card_index = card_index,
                label = string.format("%s #%d", label, attack_number),
            })
        end
    end

    table.insert(practice_catalog.scenarios, {
        scenario = class_name,
        label = label,
        attack_count = attack_number,
        attacks = attacks,
    })
end

-- The test bridge exposes this scalar-only manifest through its catalog command.
SR_SPELL_PRACTICE_CATALOG = practice_catalog

-- Full-stage entries are kept separate from spell-practice attacks because an
-- empty enemy pool is normal between stage waves. The test bridge applies a
-- different completion contract to these entries.
for _, difficulty in ipairs({ "Normal", "Lunatic" }) do
    for stage_index = 1, 5 do
        local stage_name = string.format("Stage %d@%s", stage_index, difficulty)
        if stage and stage.stages and stage.stages[stage_name] then
            table.insert(stage_test_catalog.stages, {
                stage = stage_name,
                difficulty = difficulty,
                stage_index = stage_index,
                label = string.format("%s Stage %d", difficulty, stage_index),
            })
        end
    end
end

SR_STAGE_TEST_CATALOG = stage_test_catalog
