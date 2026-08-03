import numpy as np
import torch

from stg_lab.policy import HumanVisionPolicy, PolicyConfig
from stg_lab.training import (
    Demonstrations,
    TrainingConfig,
    load_checkpoint,
    previous_actions_from_targets,
    save_checkpoint,
    to_recurrent_sequences,
    train_behavior_cloning,
)


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
    )
    sequences = to_recurrent_sequences(demonstrations, sequence_length=4)
    assert sequences.global_frames.shape == (4, 4, 6, 4, 4)
    assert set(sequences.episode_ids) == {10, 11}
    assert np.all(sequences.supervision_mask)
    np.testing.assert_array_equal(sequences.global_frames[0, 0], values[0, -1])


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
