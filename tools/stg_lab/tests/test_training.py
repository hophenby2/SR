import numpy as np
import torch

from stg_lab.policy import HumanVisionPolicy, PolicyConfig
from stg_lab.training import (
    TEACHER_ACTION_COLLIDED_INDEX,
    TEACHER_ACTION_EVALUATION_FIELDS,
    TEACHER_ACTION_MINIMUM_MARGIN_INDEX,
    TEACHER_ACTION_SELECTED_INDEX,
    Demonstrations,
    TrainingConfig,
    TrainingMetrics,
    expand_checkpoint_with_previous_action_context,
    load_checkpoint,
    previous_actions_from_targets,
    save_checkpoint,
    teacher_action_acceptance_weights,
    teacher_action_collision_ranking_loss,
    teacher_set_valued_action_loss,
    to_recurrent_sequences,
    train_behavior_cloning,
)


def test_set_valued_teacher_loss_accepts_one_safe_route_without_mode_averaging() -> None:
    evaluations = torch.zeros((
        1, 1, 18, len(TEACHER_ACTION_EVALUATION_FIELDS),
    ))
    evaluations[..., TEACHER_ACTION_COLLIDED_INDEX] = 1.0
    evaluations[..., TEACHER_ACTION_MINIMUM_MARGIN_INDEX] = -1.0
    left = 3
    stationary = 4
    right = 5
    up = 7
    for action in (left, stationary, right, up):
        evaluations[..., action, TEACHER_ACTION_COLLIDED_INDEX] = 0.0
        evaluations[..., action, TEACHER_ACTION_MINIMUM_MARGIN_INDEX] = 20.0
    evaluations[..., right, TEACHER_ACTION_SELECTED_INDEX] = 1.0
    regrets = torch.zeros((1, 1, 18))
    regrets[..., up] = 4.0
    available = torch.ones((1, 1), dtype=torch.bool)
    actions = torch.full((1, 1), right, dtype=torch.long)

    weights, mask = teacher_action_acceptance_weights(
        evaluations,
        regrets,
        available,
        actions,
        temperature=4.0,
        safety_margin=12.0,
    )

    assert mask.item() is True
    assert weights[0, 0, left] == 1.0
    assert weights[0, 0, right] == 1.0
    assert weights[0, 0, up] == torch.exp(torch.tensor(-1.0))
    # The teacher is moving, so a geometrically safe stationary action cannot
    # absorb the probability mass needed for movement onset.
    assert weights[0, 0, stationary] == 0.0

    def route_loss(action: int) -> float:
        logits = torch.full((1, 1, 18), -8.0)
        logits[..., action] = 8.0
        losses, _ = teacher_set_valued_action_loss(
            logits,
            evaluations,
            regrets,
            available,
            actions,
            temperature=4.0,
            safety_margin=12.0,
        )
        return float(losses.item())

    assert route_loss(left) < 1e-3
    assert route_loss(right) < 1e-3
    assert route_loss(stationary) > 10.0


def test_collision_ranking_requires_a_safe_argmax_without_selecting_one_route() -> None:
    evaluations = torch.zeros((
        1, 1, 18, len(TEACHER_ACTION_EVALUATION_FIELDS),
    ))
    evaluations[..., TEACHER_ACTION_COLLIDED_INDEX] = 1.0
    evaluations[..., TEACHER_ACTION_MINIMUM_MARGIN_INDEX] = -1.0
    left, right = 3, 5
    for action in (left, right):
        evaluations[..., action, TEACHER_ACTION_COLLIDED_INDEX] = 0.0
        evaluations[..., action, TEACHER_ACTION_MINIMUM_MARGIN_INDEX] = 20.0
    evaluations[..., right, TEACHER_ACTION_SELECTED_INDEX] = 1.0
    regrets = torch.zeros((1, 1, 18))
    available = torch.ones((1, 1), dtype=torch.bool)
    actions = torch.full((1, 1), right, dtype=torch.long)
    logits = torch.zeros((1, 1, 18))
    logits[..., 0] = 2.0

    unsafe_loss, mask = teacher_action_collision_ranking_loss(
        logits,
        evaluations,
        regrets,
        available,
        actions,
        temperature=2.0,
        safety_margin=16.0,
        ranking_margin=1.0,
    )
    logits[..., left] = 4.0
    safe_loss, _ = teacher_action_collision_ranking_loss(
        logits,
        evaluations,
        regrets,
        available,
        actions,
        temperature=2.0,
        safety_margin=16.0,
        ranking_margin=1.0,
    )

    assert mask.item() is True
    assert unsafe_loss.item() == 3.0
    assert safe_loss.item() == 0.0


def test_tiny_behavior_cloning_and_checkpoint(tmp_path) -> None:
    rng = np.random.default_rng(8)
    samples, steps = 12, 2
    global_frames = rng.random((samples, steps, 6, 16, 16), dtype=np.float32)
    local_frames = rng.random((samples, steps, 6, 16, 16), dtype=np.float32)
    actions = np.zeros((samples, steps), dtype=np.int64)
    risks = np.zeros((samples, steps), dtype=np.float32)
    demonstrations = Demonstrations(global_frames, local_frames, actions, risks)
    checkpoint = tmp_path / "tiny.pt"
    model, history = train_behavior_cloning(
        demonstrations,
        policy_config=PolicyConfig(feature_size=12, recurrent_size=16),
        training_config=TrainingConfig(
            seed=3,
            epochs=3,
            batch_size=4,
            learning_rate=3e-3,
            device="cpu",
        ),
        output=checkpoint,
    )
    assert model is not None
    assert checkpoint.is_file()
    assert history[-1].train_loss < history[0].train_loss
    loaded, metadata = load_checkpoint(checkpoint)
    assert loaded is not None
    assert metadata["version"] == 3
    assert metadata["training_config"]["seed"] == 3
    assert metadata["training_data"] == {}


def test_demonstration_save_round_trips_semantic_frames_as_float32(
    tmp_path,
) -> None:
    shape = (2, 1, 6, 3, 2)
    global_frames = np.linspace(
        -0.9876543, 0.9876543, num=np.prod(shape), dtype=np.float32,
    ).reshape(shape)
    local_frames = global_frames * np.float32(0.7312345)
    assert not np.array_equal(
        global_frames,
        global_frames.astype(np.float16).astype(np.float32),
    )
    demonstrations = Demonstrations(
        global_frames=global_frames,
        local_frames=local_frames,
        actions=np.asarray(((3,), (5,)), dtype=np.int64),
        risks=np.asarray(((0.25,), (0.75,)), dtype=np.float32),
    )
    path = tmp_path / "float32.npz"

    demonstrations.save(path)

    with np.load(path) as archive:
        assert archive["global_frames"].dtype == np.float32
        assert archive["local_frames"].dtype == np.float32
        np.testing.assert_array_equal(archive["global_frames"], global_frames)
        np.testing.assert_array_equal(archive["local_frames"], local_frames)
    loaded = Demonstrations.load(path)
    assert loaded.global_frames.dtype == np.float32
    assert loaded.local_frames.dtype == np.float32
    np.testing.assert_array_equal(loaded.global_frames, global_frames)
    np.testing.assert_array_equal(loaded.local_frames, local_frames)


def test_demonstration_load_keeps_legacy_float16_frames_compatible(
    tmp_path,
) -> None:
    shape = (2, 1, 6, 3, 2)
    global_frames = np.linspace(-1.0, 1.0, num=np.prod(shape)).reshape(
        shape,
    ).astype(np.float16)
    local_frames = (global_frames * np.float16(0.5)).astype(np.float16)
    legacy_path = tmp_path / "legacy-float16.npz"
    np.savez_compressed(
        legacy_path,
        global_frames=global_frames,
        local_frames=local_frames,
        actions=np.asarray(((3,), (5,)), dtype=np.int64),
        risks=np.asarray(((0.25,), (0.75,)), dtype=np.float32),
    )

    loaded = Demonstrations.load(legacy_path)

    assert loaded.global_frames.dtype == np.float16
    assert loaded.local_frames.dtype == np.float16
    np.testing.assert_array_equal(loaded.global_frames, global_frames)
    np.testing.assert_array_equal(loaded.local_frames, local_frames)

    upgraded_path = tmp_path / "upgraded-float32.npz"
    loaded.save(upgraded_path)
    with np.load(upgraded_path) as upgraded:
        assert upgraded["global_frames"].dtype == np.float32
        assert upgraded["local_frames"].dtype == np.float32
        np.testing.assert_array_equal(
            upgraded["global_frames"], global_frames.astype(np.float32),
        )


def test_decision_windows_convert_to_episode_grouped_recurrent_sequences() -> None:
    samples, history = 12, 4
    values = np.arange(samples * history * 6 * 4 * 4, dtype=np.float32).reshape(
        samples, history, 6, 4, 4,
    )
    demonstrations = Demonstrations(
        global_frames=values,
        local_frames=values.copy(),
        actions=np.tile(np.arange(history), (samples, 1)),
        risks=np.zeros((samples, history), dtype=np.float32),
        episode_ids=np.repeat((10, 11), 6),
        correction_mask=(
            np.arange(samples * history).reshape(samples, history) % 5 == 0
        ),
    )
    sequences = to_recurrent_sequences(demonstrations, sequence_length=4)
    assert sequences.global_frames.shape == (4, 4, 6, 4, 4)
    assert sequences.global_frames.dtype == np.float32
    assert sequences.local_frames.dtype == np.float32
    assert set(sequences.episode_ids) == {10, 11}
    assert np.all(sequences.supervision_mask)
    np.testing.assert_array_equal(sequences.global_frames[0, 0], values[0, -1])
    selected = np.asarray((
        0, 1, 2, 3,
        2, 3, 4, 5,
        6, 7, 8, 9,
        8, 9, 10, 11,
    ))
    np.testing.assert_array_equal(
        sequences.correction_mask.reshape(-1),
        demonstrations.correction_mask[selected, -1],
    )


def test_legacy_archive_defaults_correction_mask_to_false_and_saves_it(
    tmp_path,
) -> None:
    path = tmp_path / "legacy.npz"
    global_frames = np.zeros((2, 1, 6, 4, 4), dtype=np.float32)
    local_frames = np.zeros_like(global_frames)
    actions = np.asarray(((3,), (5,)), dtype=np.int64)
    np.savez_compressed(
        path,
        global_frames=global_frames,
        local_frames=local_frames,
        actions=actions,
        risks=np.zeros((2, 1), dtype=np.float32),
    )

    demonstrations = Demonstrations.load(path)
    np.testing.assert_array_equal(
        demonstrations.correction_mask,
        np.zeros_like(actions, dtype=np.bool_),
    )
    demonstrations.correction_mask[1, 0] = True
    demonstrations.save(path)

    with np.load(path) as archive:
        assert "correction_mask" in archive.files
        np.testing.assert_array_equal(archive["correction_mask"], ((0,), (1,)))
    np.testing.assert_array_equal(
        Demonstrations.load(path).correction_mask,
        ((False,), (True,)),
    )


def test_previous_actions_cannot_be_guessed_from_multiframe_windows() -> None:
    demonstrations = Demonstrations(
        global_frames=np.zeros((2, 2, 6, 4, 4), dtype=np.float32),
        local_frames=np.zeros((2, 2, 6, 4, 4), dtype=np.float32),
        actions=np.zeros((2, 2), dtype=np.int64),
        risks=np.zeros((2, 2), dtype=np.float32),
        episode_ids=np.zeros(2, dtype=np.int64),
    )

    with np.testing.assert_raises_regex(ValueError, "multi-frame windows"):
        previous_actions_from_targets(demonstrations)


def test_legacy_checkpoint_loads_with_stream_inference(tmp_path) -> None:
    config = PolicyConfig(feature_size=12, recurrent_size=16)
    model = HumanVisionPolicy(config)
    legacy_config = {
        "channels": config.channels,
        "feature_size": config.feature_size,
        "recurrent_size": config.recurrent_size,
        "memory_size": config.memory_size,
        "action_count": config.action_count,
    }
    checkpoint = tmp_path / "legacy.pt"
    torch.save({
        "version": 1,
        "policy_config": legacy_config,
        "state_dict": model.state_dict(),
        "history": [],
    }, checkpoint)

    loaded, metadata = load_checkpoint(checkpoint)
    assert loaded.config.inference_mode == "stream"
    assert loaded.config.local_feature_grid_size == 4
    assert loaded.config.local_downsample_stages == 3
    assert metadata["policy_config"]["inference_mode"] == "stream"


def test_checkpoint_restores_scenario_vocabulary_for_live_inference(tmp_path) -> None:
    config = PolicyConfig(
        feature_size=12,
        recurrent_size=16,
        memory_size=2,
        proficiency_size=0,
        inference_mode="stream",
    )
    model = HumanVisionPolicy(config)
    checkpoint = tmp_path / "context.pt"
    vocabulary = ["<unknown>", "attack:okuu:Lunatic#3"]
    save_checkpoint(
        model,
        checkpoint,
        policy_config=config,
        training_data={"scenario_vocabulary": vocabulary},
    )

    loaded, metadata = load_checkpoint(checkpoint)

    assert loaded.scenario_vocabulary == tuple(vocabulary)
    assert metadata["training_data"]["scenario_vocabulary"] == vocabulary


def test_checkpoint_restores_previous_executed_action_context(tmp_path) -> None:
    vocabulary = ["<unknown>", "attack:okuu:Lunatic#3"]
    config = PolicyConfig(
        feature_size=12,
        recurrent_size=16,
        memory_size=len(vocabulary) + 18,
        proficiency_size=0,
        inference_mode="stream",
    )
    checkpoint = tmp_path / "motor-context.pt"
    save_checkpoint(
        HumanVisionPolicy(config),
        checkpoint,
        policy_config=config,
        training_data={
            "scenario_vocabulary": vocabulary,
            "previous_action_size": 18,
            "previous_action_offset": len(vocabulary),
        },
    )

    loaded, _ = load_checkpoint(checkpoint)

    assert loaded.scenario_vocabulary == tuple(vocabulary)
    assert loaded.previous_action_size == 18
    assert loaded.previous_action_offset == len(vocabulary)


def test_previous_action_checkpoint_expansion_is_epoch_zero_equivalent(
    tmp_path,
) -> None:
    torch.manual_seed(8105)
    vocabulary = ["<unknown>", "attack:okuu:Lunatic#3"]
    source_config = PolicyConfig(
        feature_size=12,
        recurrent_size=16,
        memory_size=2,
        proficiency_size=5,
        inference_mode="stream",
    )
    source = tmp_path / "source.pt"
    output = tmp_path / "expanded.pt"
    save_checkpoint(
        HumanVisionPolicy(source_config),
        source,
        policy_config=source_config,
        history=[TrainingMetrics(7, 0.4, 0.5, 0.8, 0.1)],
        training_data={
            "scenario_vocabulary": vocabulary,
            "previous_action_size": 0,
            "previous_action_offset": 2,
        },
    )

    report = expand_checkpoint_with_previous_action_context(source, output)
    source_model, source_checkpoint = load_checkpoint(source)
    expanded_model, expanded_checkpoint = load_checkpoint(output)

    assert expanded_model.config == PolicyConfig(
        feature_size=12,
        recurrent_size=16,
        memory_size=20,
        proficiency_size=5,
        inference_mode="stream",
    )
    assert expanded_model.scenario_vocabulary == tuple(vocabulary)
    assert expanded_model.previous_action_offset == 2
    assert expanded_model.previous_action_size == 18
    assert expanded_checkpoint["history"] == source_checkpoint["history"]
    source_state = source_model.state_dict()
    expanded_state = expanded_model.state_dict()
    recurrent_name = "recurrent.weight_ih_l0"
    for name, value in source_state.items():
        if name != recurrent_name:
            assert torch.equal(expanded_state[name], value), name
    insertion = source_config.feature_size * 2 + source_config.memory_size
    source_recurrent = source_state[recurrent_name]
    expanded_recurrent = expanded_state[recurrent_name]
    assert torch.equal(
        expanded_recurrent[:, :insertion],
        source_recurrent[:, :insertion],
    )
    assert torch.count_nonzero(
        expanded_recurrent[:, insertion:insertion + 18]
    ).item() == 0
    assert torch.equal(
        expanded_recurrent[:, insertion + 18:],
        source_recurrent[:, insertion:],
    )

    batch, steps = 18, 2
    global_frames = torch.randn(batch, steps, 6, 16, 16)
    local_frames = torch.randn(batch, steps, 6, 16, 16)
    scenario_memory = torch.randn(batch, steps, 2)
    proficiency = torch.randn(batch, steps, 5)
    initial_hidden = torch.randn(1, batch, source_config.recurrent_size)
    expanded_memory = torch.zeros(batch, steps, 20)
    expanded_memory[..., :2] = scenario_memory
    token_ids = torch.arange(18)
    arbitrary_tokens = torch.stack((token_ids, (token_ids * 7 + 3) % 18), dim=1)
    expanded_memory.scatter_(
        -1,
        (arbitrary_tokens + 2).unsqueeze(-1),
        1.0,
    )
    with torch.no_grad():
        source_logits, source_risk, source_hidden = source_model(
            global_frames,
            local_frames,
            scenario_memory,
            proficiency,
            initial_hidden,
        )
        expanded_logits, expanded_risk, expanded_hidden = expanded_model(
            global_frames,
            local_frames,
            expanded_memory,
            proficiency,
            initial_hidden,
        )
    torch.testing.assert_close(expanded_logits, source_logits, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(expanded_risk, source_risk, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(expanded_hidden, source_hidden, atol=1e-6, rtol=1e-6)

    training_data = expanded_checkpoint["training_data"]
    proof = training_data["previous_action_expansion"]
    assert training_data["parent_checkpoint"] == str(source)
    assert training_data["parent_checkpoint_sha256"] == report[
        "parent_checkpoint_sha256"
    ]
    assert training_data["initialization"] == (
        "complete_policy_state_with_zero_initialized_previous_action_input"
    )
    assert proof["weight_copy_proof"]["all_source_values_copied_exactly"] is True
    assert proof["zero_initialization_proof"]["nonzero_count"] == 0
    assert proof["zero_initialization_proof"]["verified_exact"] is True
    assert proof["epoch_zero_equivalence"]["guaranteed"] is True
    assert report["checkpoint_sha256"]


def test_previous_action_checkpoint_expansion_requires_declared_vocabulary(
    tmp_path,
) -> None:
    config = PolicyConfig(
        feature_size=12,
        recurrent_size=16,
        memory_size=2,
        proficiency_size=0,
        inference_mode="stream",
    )
    source = tmp_path / "no-vocabulary.pt"
    output = tmp_path / "invalid-expanded.pt"
    save_checkpoint(
        HumanVisionPolicy(config),
        source,
        policy_config=config,
    )

    with np.testing.assert_raises_regex(ValueError, "scenario_vocabulary"):
        expand_checkpoint_with_previous_action_context(source, output)
    assert not output.exists()
