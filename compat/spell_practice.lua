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

for _, practice_boss in ipairs(practice_bosses) do
    local class_name = practice_boss[1]
    local label = practice_boss[2]
    local boss_class = assert(_editor_class[class_name], "missing spell-practice boss class: " .. class_name)
    local attack_number = 0

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
        end
    end
end
