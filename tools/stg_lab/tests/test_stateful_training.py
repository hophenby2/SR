from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from stg_lab.policy import HumanVisionPolicy, PolicyConfig
from stg_lab.protocol import Action
from stg_lab.stateful_training import (
    StatefulTrainingConfig,
    drop_previous_action_context,
    evaluate_stateful_policy,
    initialize_visual_encoders,
    ordered_episode_sequences,
    reflect_horizontal_action_context,
    reflect_horizontal_stream_batch,
    split_episode_ids,
    teacher_transition_sample_weights,
    train_stateful_behavior_cloning,
)
from stg_lab.training import Demonstrations


def demonstrations(episode_ids: tuple[int, ...], *, history: int = 3) -> Demonstrations:
    samples = len(episode_ids)
    shape = (samples, history, 6, 8, 8)
    global_frames = np.empty(shape, dtype=np.float32)
    local_frames = np.empty(shape, dtype=np.float32)
    for sample in range(samples):
        for step in range(history):
            global_frames[sample, step].fill(sample * 10 + step)
            local_frames[sample, step].fill(-(sample * 10 + step))
    return Demonstrations(
        global_frames=global_frames,
        local_frames=local_frames,
        actions=np.zeros((samples, history), dtype=np.int64),
        risks=np.linspace(0.0, 0.5, samples, dtype=np.float32)[:, None].repeat(
            history, axis=1,
        ),
        episode_ids=np.asarray(episode_ids, dtype=np.int64),
    )


class RecordingStreamPolicy(torch.nn.Module):
    def __init__(self, config: PolicyConfig) -> None:
        super().__init__()
        self.config = config
        self.anchor = torch.nn.Parameter(torch.tensor(0.1))
        self.calls: list[dict[str, object]] = []

    def forward(
        self,
        global_frames,
        local_frames,
        memory=None,
        proficiency=None,
        hidden=None,
    ):
        batch, steps = global_frames.shape[:2]
        self.calls.append({
            "training": self.training,
            "steps": steps,
            "hidden_none": hidden is None,
            "hidden_requires_grad": (
                None if hidden is None else bool(hidden.requires_grad)
            ),
            "latest": global_frames[0, :, 0, 0, 0].detach().cpu().tolist(),
            "memory": None if memory is None else memory.detach().cpu().clone(),
        })
        previous = (
            torch.zeros((batch, 1), device=global_frames.device)
            if hidden is None else
            hidden[0]
        )
        increments = torch.ones(
            (batch, steps), dtype=global_frames.dtype, device=global_frames.device,
        ) * (1.0 + self.anchor)
        state = previous + torch.cumsum(increments, dim=1)
        logits = torch.cat((
            state[..., None],
            -state[..., None],
            torch.zeros(
                (batch, steps, self.config.action_count - 2),
                dtype=global_frames.dtype,
                device=global_frames.device,
            ),
        ), dim=-1)
        risk = torch.sigmoid(state * 0.1)
        next_hidden = state[:, -1].reshape(1, batch, 1)
        return logits, risk, next_hidden


class FutureRecordingStreamPolicy(RecordingStreamPolicy):
    def __init__(self, config: PolicyConfig) -> None:
        super().__init__(config)
        self.target_calls: list[dict[str, object]] = []

    def encode_visual(self, global_frames, local_frames):
        self.target_calls.append({
            "grad_enabled": torch.is_grad_enabled(),
            "global_latest": global_frames[
                0, :, 0, 0, 0
            ].detach().cpu().tolist(),
            "local_latest": local_frames[
                0, :, 0, 0, 0
            ].detach().cpu().tolist(),
        })
        global_values = global_frames[:, :, 0, 0, 0, None].expand(
            -1, -1, self.config.feature_size,
        )
        local_values = local_frames[:, :, 0, 0, 0, None].expand(
            -1, -1, self.config.feature_size,
        )
        return torch.cat((global_values, local_values), dim=-1)

    def forward_with_recurrent(
        self,
        global_frames,
        local_frames,
        memory=None,
        proficiency=None,
        hidden=None,
    ):
        logits, risk, next_hidden = self.forward(
            global_frames,
            local_frames,
            memory,
            proficiency,
            hidden,
        )
        recurrent = next_hidden.new_ones((
            global_frames.shape[0],
            global_frames.shape[1],
            self.config.recurrent_size,
        )) * self.anchor
        return logits, risk, next_hidden, recurrent


class ZeroFuturePredictor(torch.nn.Module):
    def __init__(self, recurrent_size: int, visual_size: int) -> None:
        super().__init__()
        self.head = torch.nn.Linear(recurrent_size, visual_size, bias=False)
        torch.nn.init.zeros_(self.head.weight)
        self.calls: list[dict[str, object]] = []

    def forward(self, recurrent, horizon):
        self.calls.append({
            "horizon": horizon,
            "source_shape": tuple(recurrent.shape),
            "source_requires_grad": recurrent.requires_grad,
        })
        return self.head(recurrent)


class CountingSGD(torch.optim.SGD):
    def __init__(self, parameters, *, lr: float) -> None:
        super().__init__(parameters, lr=lr)
        self.step_count = 0

    def step(self, closure=None):
        self.step_count += 1
        return super().step(closure)


def stream_config() -> PolicyConfig:
    return PolicyConfig(
        feature_size=4,
        recurrent_size=8,
        memory_size=0,
        proficiency_size=0,
        inference_mode="stream",
    )


def test_factorized_action_loss_controls_are_validated() -> None:
    with pytest.raises(ValueError, match="supervised action-loss"):
        StatefulTrainingConfig(
            exact_action_loss_weight=0.0,
            direction_loss_weight=0.0,
            speed_loss_weight=0.0,
        )
    with pytest.raises(ValueError, match="nonnegative"):
        StatefulTrainingConfig(direction_consistency_weight=-0.1)
    with pytest.raises(ValueError, match="previous_action_dropout_probability"):
        StatefulTrainingConfig(previous_action_dropout_probability=1.01)
    with pytest.raises(ValueError, match="nonnegative"):
        StatefulTrainingConfig(future_visual_loss_weight=-0.1)
    with pytest.raises(ValueError, match="duplicates"):
        StatefulTrainingConfig(future_visual_horizons=(20, 20))
    with pytest.raises(ValueError, match="strictly increasing"):
        StatefulTrainingConfig(future_visual_horizons=(40, 20))


def test_episode_blocks_must_be_contiguous_and_split_without_leakage() -> None:
    values = demonstrations((10, 10, 20, 20, 30, 30, 40, 40))
    sequences = ordered_episode_sequences(values)
    assert [(item.episode_id, item.start, item.stop) for item in sequences] == [
        (10, 0, 2),
        (20, 2, 4),
        (30, 4, 6),
        (40, 6, 8),
    ]

    first = split_episode_ids(values, validation_fraction=0.25, seed=91)
    second = split_episode_ids(values, validation_fraction=0.25, seed=91)
    assert first == second
    assert not set(first.train_episode_ids) & set(first.validation_episode_ids)
    assert set(first.train_episode_ids) | set(first.validation_episode_ids) == {
        10, 20, 30, 40,
    }

    with pytest.raises(ValueError, match="contiguous block"):
        ordered_episode_sequences(demonstrations((10, 10, 20, 10)))


def test_explicit_validation_episodes_support_per_card_heldout_seeds() -> None:
    values = demonstrations((10, 10, 20, 20, 30, 30, 40, 40))

    split = split_episode_ids(
        values,
        validation_fraction=0.25,
        seed=91,
        validation_episode_ids=(20, 40),
    )

    assert split.train_episode_ids == (10, 30)
    assert split.validation_episode_ids == (20, 40)
    with pytest.raises(ValueError, match="unknown validation"):
        split_episode_ids(
            values,
            validation_fraction=0.25,
            seed=91,
            validation_episode_ids=(99,),
        )
    with pytest.raises(ValueError, match="complete dataset"):
        split_episode_ids(
            values,
            validation_fraction=0.25,
            seed=91,
            validation_episode_ids=(10, 20, 30, 40),
        )


def test_stream_evaluation_uses_latest_frames_and_preserves_hidden_per_episode() -> None:
    values = demonstrations((3, 3, 3, 9, 9, 9))
    model = RecordingStreamPolicy(stream_config())

    metrics = evaluate_stateful_policy(
        model,
        values,
        chunk_length=2,
        device="cpu",
    )

    assert [call["steps"] for call in model.calls] == [2, 1, 2, 1]
    assert [call["hidden_none"] for call in model.calls] == [True, False, True, False]
    assert [call["latest"] for call in model.calls] == [
        [2.0, 12.0],
        [22.0],
        [32.0, 42.0],
        [52.0],
    ]
    assert all(call["training"] is False for call in model.calls)
    assert metrics.decisions == 6
    assert metrics.labels == 6
    assert metrics.risk_labels == 6
    assert metrics.chunks == 4
    assert metrics.episodes == 2


def test_future_visual_loss_uses_detached_within_episode_semantic_targets() -> None:
    import stg_lab.stateful_training as stateful_training

    values = demonstrations((10, 10, 10, 20, 20))
    episodes = ordered_episode_sequences(values)
    config = stream_config()
    baseline = stateful_training._stateful_pass(
        RecordingStreamPolicy(config),
        values,
        episodes,
        chunk_length=2,
        risk_loss_weight=0.2,
        gradient_clip=5.0,
        device="cpu",
        optimizer=None,
    )
    model = FutureRecordingStreamPolicy(config)
    predictor = ZeroFuturePredictor(
        config.recurrent_size,
        config.feature_size * 2,
    )

    measured = stateful_training._stateful_pass(
        model,
        values,
        episodes,
        chunk_length=2,
        risk_loss_weight=0.2,
        gradient_clip=5.0,
        device="cpu",
        optimizer=None,
        future_visual_loss_weight=2.0,
        future_visual_horizons=(1, 2),
        future_visual_predictor=predictor,
    )

    assert measured.future_visual_labels == 4
    assert measured.future_visual_loss > 0.0
    assert measured.loss == pytest.approx(
        baseline.loss + 2.0 * measured.future_visual_loss
    )
    assert [call["global_latest"] for call in model.target_calls] == [
        [12.0, 22.0],
        [42.0],
    ]
    assert [call["local_latest"] for call in model.target_calls] == [
        [-12.0, -22.0],
        [-42.0],
    ]
    assert all(call["grad_enabled"] is False for call in model.target_calls)
    assert [call["horizon"] for call in predictor.calls] == [1, 2, 1]
    assert all(
        call["source_requires_grad"] is False for call in predictor.calls
    )

    training_model = FutureRecordingStreamPolicy(config)
    training_predictor = ZeroFuturePredictor(
        config.recurrent_size,
        config.feature_size * 2,
    )
    optimizer = CountingSGD(
        list(training_model.parameters()) + list(training_predictor.parameters()),
        lr=1e-4,
    )
    trained = stateful_training._stateful_pass(
        training_model,
        values,
        episodes,
        chunk_length=2,
        risk_loss_weight=0.2,
        gradient_clip=5.0,
        device="cpu",
        optimizer=optimizer,
        future_visual_loss_weight=1.0,
        future_visual_horizons=(1, 2),
        future_visual_predictor=training_predictor,
    )
    assert trained.future_visual_labels == 4
    assert any(
        call["source_requires_grad"] is True
        for call in training_predictor.calls
    )
    assert all(
        call["grad_enabled"] is False for call in training_model.target_calls
    )


def test_horizontal_reflection_preserves_episode_semantics() -> None:
    global_frames = torch.zeros((1, 2, 6, 2, 3))
    local_frames = torch.zeros((1, 2, 6, 2, 3))
    global_frames[:, :, 0, 0] = torch.tensor((1.0, 2.0, 3.0))
    local_frames[:, :, 0, 1] = torch.tensor((4.0, 5.0, 6.0))
    global_frames[:, :, 1].fill_(0.25)
    local_frames[:, :, 1].fill_(-0.5)
    actions = torch.tensor([[
        Action(move_x=-1, move_y=1, slow=False).discrete,
        Action(move_x=1, move_y=-1, slow=True).discrete,
    ]])

    reflected_global, reflected_local, reflected_actions = (
        reflect_horizontal_stream_batch(global_frames, local_frames, actions)
    )

    assert reflected_global[0, 0, 0, 0].tolist() == [3.0, 2.0, 1.0]
    assert reflected_local[0, 0, 0, 1].tolist() == [6.0, 5.0, 4.0]
    assert torch.all(reflected_global[:, :, 1] == -0.25)
    assert torch.all(reflected_local[:, :, 1] == 0.5)
    assert reflected_actions.tolist() == [[
        Action(move_x=1, move_y=1, slow=False).discrete,
        Action(move_x=-1, move_y=-1, slow=True).discrete,
    ]]


def test_horizontal_reflection_mirrors_previous_action_context_only() -> None:
    offset = 3
    memory = torch.zeros((1, 3, offset + 18))
    memory[:, :, :offset] = torch.tensor((1.0, 2.0, 3.0))
    left = Action(move_x=-1, move_y=1, slow=False).discrete
    right = Action(move_x=1, move_y=-1, slow=True).discrete
    memory[0, 0, offset + left] = 1.0
    memory[0, 1, offset + right] = 1.0
    previous_actions = torch.tensor([[left, right, -1]])

    reflected = reflect_horizontal_action_context(
        memory,
        previous_actions,
        previous_action_offset=offset,
    )

    mirrored_left = Action(move_x=1, move_y=1, slow=False).discrete
    mirrored_right = Action(move_x=-1, move_y=-1, slow=True).discrete
    assert torch.equal(reflected[:, :, :offset], memory[:, :, :offset])
    assert reflected[0, 0, offset + mirrored_left] == 1.0
    assert reflected[0, 1, offset + mirrored_right] == 1.0
    assert reflected[0, 0, offset + left] == 0.0
    assert reflected[0, 1, offset + right] == 0.0
    assert torch.count_nonzero(reflected[0, 2, offset:]) == 0

    invalid = memory.clone()
    invalid[0, 2, offset] = 1.0
    with pytest.raises(ValueError, match="does not match recorded"):
        reflect_horizontal_action_context(
            invalid,
            previous_actions,
            previous_action_offset=offset,
        )


def test_previous_action_dropout_preserves_identity_and_selected_context() -> None:
    offset = 2
    memory = torch.zeros((1, 3, offset + 18))
    memory[:, :, :offset] = torch.tensor((0.25, 0.75))
    memory[0, 0, offset + 3] = 1.0
    memory[0, 1, offset + 7] = 1.0
    memory[0, 2, offset + 11] = 1.0

    dropped = drop_previous_action_context(
        memory,
        torch.tensor([[False, True, True]]),
        previous_action_offset=offset,
    )

    assert torch.equal(dropped[:, :, :offset], memory[:, :, :offset])
    assert torch.equal(dropped[0, 0, offset:], memory[0, 0, offset:])
    assert torch.count_nonzero(dropped[0, 1:, offset:]) == 0
    assert torch.equal(memory[0, 1:, offset:], torch.stack((
        torch.nn.functional.one_hot(torch.tensor(7), 18),
        torch.nn.functional.one_hot(torch.tensor(11), 18),
    )).to(dtype=memory.dtype))


def test_training_drops_action_context_but_validation_keeps_recorded_actions() -> None:
    offset = 2
    values = demonstrations((10, 10, 20, 20), history=1)
    values.memory = np.zeros((4, 1, offset + 18), dtype=np.float32)
    values.memory[:, :, :offset] = (0.25, 0.75)
    values.previous_actions = np.asarray((3, 7, 11, 15), dtype=np.int64)[:, None]
    for sample, action in enumerate(values.previous_actions[:, 0]):
        values.memory[sample, 0, offset + action] = 1.0
    config = PolicyConfig(
        feature_size=4,
        recurrent_size=8,
        memory_size=offset + 18,
        proficiency_size=0,
        inference_mode="stream",
    )
    model = RecordingStreamPolicy(config)

    train_stateful_behavior_cloning(
        values,
        policy_config=config,
        training_config=StatefulTrainingConfig(
            seed=17,
            epochs=1,
            chunk_length=2,
            learning_rate=1e-3,
            validation_fraction=0.5,
            validation_episode_ids=(20,),
            class_balance=False,
            device="cpu",
            previous_action_dropout_probability=1.0,
        ),
        model=model,
        training_data={
            "previous_action_size": 18,
            "previous_action_offset": offset,
        },
    )

    training_memory = next(
        call["memory"] for call in model.calls if call["training"] is True
    )
    validation_memory = next(
        call["memory"] for call in model.calls if call["training"] is False
    )
    assert torch.all(training_memory[..., :offset] == torch.tensor((0.25, 0.75)))
    assert torch.count_nonzero(training_memory[..., offset:]) == 0
    assert torch.count_nonzero(validation_memory[..., offset:]) == 2


def test_teacher_transition_weights_ignore_slow_mode_and_episode_boundaries() -> None:
    values = demonstrations((10, 10, 10, 10, 10, 10, 20, 20, 20))
    labels = (
        Action().discrete,
        Action(slow=True).discrete,
        Action(move_x=1, slow=True).discrete,
        Action(move_x=1).discrete,
        Action(move_y=1).discrete,
        Action().discrete,
        Action(move_x=-1).discrete,
        Action(move_x=-1, slow=True).discrete,
        Action(move_x=-1, move_y=-1).discrete,
    )
    values.actions[:, -1] = labels

    weights = teacher_transition_sample_weights(
        values,
        movement_onset_weight=5.0,
        direction_change_weight=3.0,
    )

    np.testing.assert_array_equal(weights, (1, 1, 5, 1, 3, 1, 1, 1, 3))

    values.supervision_mask = np.ones_like(values.actions, dtype=np.bool_)
    values.supervision_mask[2, -1] = False
    masked = teacher_transition_sample_weights(
        values,
        movement_onset_weight=5.0,
        direction_change_weight=3.0,
    )
    assert masked[2] == 1.0
    assert masked[4] == 3.0


def test_episode_balancing_updates_once_per_episode_instead_of_per_chunk() -> None:
    import stg_lab.stateful_training as stateful_training

    values = demonstrations((10, 10, 10, 10, 10, 20, 20))
    episodes = ordered_episode_sequences(values)
    balanced_model = RecordingStreamPolicy(stream_config())
    balanced_optimizer = CountingSGD(balanced_model.parameters(), lr=1e-3)

    balanced = stateful_training._stateful_pass(
        balanced_model,
        values,
        episodes,
        chunk_length=2,
        risk_loss_weight=0.2,
        gradient_clip=5.0,
        device="cpu",
        optimizer=balanced_optimizer,
        episode_balanced=True,
    )

    chunk_model = RecordingStreamPolicy(stream_config())
    chunk_optimizer = CountingSGD(chunk_model.parameters(), lr=1e-3)
    chunk_weighted = stateful_training._stateful_pass(
        chunk_model,
        values,
        episodes,
        chunk_length=2,
        risk_loss_weight=0.2,
        gradient_clip=5.0,
        device="cpu",
        optimizer=chunk_optimizer,
        episode_balanced=False,
    )

    assert balanced.chunks == chunk_weighted.chunks == 4
    assert balanced.optimizer_steps == balanced_optimizer.step_count == 2
    assert chunk_weighted.optimizer_steps == chunk_optimizer.step_count == 4


def test_correction_only_keeps_full_gru_context_and_all_risk_targets() -> None:
    import stg_lab.stateful_training as stateful_training

    values = demonstrations((10, 10, 10, 10, 20, 20, 20, 20))
    values.supervision_mask = np.zeros_like(values.actions, dtype=np.bool_)
    values.supervision_mask[1, -1] = True
    values.supervision_mask[5, -1] = True
    episodes = ordered_episode_sequences(values)

    model = RecordingStreamPolicy(stream_config())
    optimizer = CountingSGD(model.parameters(), lr=1e-3)
    metrics = stateful_training._stateful_pass(
        model,
        values,
        episodes,
        chunk_length=2,
        risk_loss_weight=0.2,
        gradient_clip=5.0,
        device="cpu",
        optimizer=optimizer,
        risk_on_all_decisions=True,
    )

    assert [call["steps"] for call in model.calls] == [2, 2, 2, 2]
    assert [call["hidden_none"] for call in model.calls] == [
        True, False, True, False,
    ]
    assert metrics.decisions == 8
    assert metrics.labels == 2
    assert metrics.risk_labels == 8
    assert metrics.optimizer_steps == optimizer.step_count == 4

    masked_model = RecordingStreamPolicy(stream_config())
    masked_optimizer = CountingSGD(masked_model.parameters(), lr=1e-3)
    masked = stateful_training._stateful_pass(
        masked_model,
        values,
        episodes,
        chunk_length=2,
        risk_loss_weight=0.2,
        gradient_clip=5.0,
        device="cpu",
        optimizer=masked_optimizer,
    )
    assert masked.labels == masked.risk_labels == 2
    assert masked.optimizer_steps == masked_optimizer.step_count == 2


def test_correction_only_requires_an_explicit_nonempty_action_mask() -> None:
    values = demonstrations((10, 10, 20, 20))
    with pytest.raises(ValueError, match="requires a supervision_mask"):
        train_stateful_behavior_cloning(
            values,
            policy_config=stream_config(),
            training_config=StatefulTrainingConfig(
                epochs=1,
                validation_fraction=0.5,
                class_balance=False,
                device="cpu",
                correction_only=True,
            ),
        )

    values.supervision_mask = np.zeros_like(values.actions, dtype=np.bool_)
    with pytest.raises(ValueError, match="at least one supervised correction"):
        train_stateful_behavior_cloning(
            values,
            policy_config=stream_config(),
            training_config=StatefulTrainingConfig(
                epochs=1,
                validation_fraction=0.5,
                class_balance=False,
                device="cpu",
                correction_only=True,
            ),
        )


def test_correction_only_checkpoint_records_supervision_scopes(tmp_path) -> None:
    values = demonstrations((10, 10, 20, 20))
    values.supervision_mask = np.zeros_like(values.actions, dtype=np.bool_)
    values.supervision_mask[1, -1] = True
    values.supervision_mask[3, -1] = True
    output = tmp_path / "correction-only.pt"

    train_stateful_behavior_cloning(
        values,
        policy_config=stream_config(),
        training_config=StatefulTrainingConfig(
            seed=11,
            epochs=1,
            chunk_length=1,
            learning_rate=1e-3,
            validation_fraction=0.5,
            validation_episode_ids=(20,),
            class_balance=False,
            device="cpu",
            correction_only=True,
        ),
        model=RecordingStreamPolicy(stream_config()),
        output=output,
    )

    checkpoint = torch.load(output, map_location="cpu", weights_only=False)
    assert checkpoint["training_config"]["correction_only"] is True
    assert checkpoint["training_data"]["loss_weighting"]["action_supervision"] == (
        "supervision_mask"
    )
    assert checkpoint["training_data"]["loss_weighting"]["risk_supervision"] == (
        "all_decisions"
    )
    assert checkpoint["training_data"]["loss_weighting"]["recurrent_context"] == (
        "complete_episode"
    )
    assert checkpoint["training_data"]["correction_only_supervision"] == {
        "action_loss_mask": "supervision_mask",
        "risk_loss_mask": "all_latest_frame_decisions",
        "recurrent_context": "complete_episode",
        "train_action_labels": 1,
        "train_risk_labels": 2,
        "validation_action_labels": 1,
        "validation_risk_labels": 2,
    }


def test_training_detaches_chunks_without_resetting_episode_hidden() -> None:
    values = demonstrations(tuple(
        episode_id
        for episode_id in (10, 20, 30, 40)
        for _ in range(5)
    ))
    config = stream_config()
    model = RecordingStreamPolicy(config)

    trained, history = train_stateful_behavior_cloning(
        values,
        policy_config=config,
        training_config=StatefulTrainingConfig(
            seed=7,
            epochs=1,
            chunk_length=2,
            learning_rate=1e-3,
            validation_fraction=0.5,
            class_balance=False,
            device="cpu",
        ),
        model=model,
    )

    assert trained is model
    assert len(history) == 1
    assert math.isfinite(history[0].train_loss)
    assert math.isfinite(history[0].validation_loss)
    training_calls = [call for call in model.calls if call["training"] is True]
    validation_calls = [call for call in model.calls if call["training"] is False]
    assert [call["steps"] for call in training_calls] == [2, 2, 1, 2, 2, 1]
    assert [call["steps"] for call in validation_calls] == [2, 2, 1, 2, 2, 1]
    assert [call["hidden_none"] for call in training_calls] == [
        True, False, False, True, False, False,
    ]
    assert [call["hidden_none"] for call in validation_calls] == [
        True, False, False, True, False, False,
    ]
    assert all(
        call["hidden_requires_grad"] is False
        for call in training_calls
        if call["hidden_none"] is False
    )


def test_visual_pretraining_transfers_encoders_without_route_or_recurrent_state() -> None:
    source_config = PolicyConfig(
        feature_size=4,
        recurrent_size=7,
        memory_size=4,
        proficiency_size=0,
        inference_mode="window",
    )
    target_config = PolicyConfig(
        feature_size=4,
        recurrent_size=8,
        memory_size=0,
        proficiency_size=0,
        inference_mode="stream",
    )
    source = HumanVisionPolicy(source_config)
    target = HumanVisionPolicy(target_config)
    with torch.no_grad():
        for parameter in source.global_encoder.parameters():
            parameter.fill_(0.25)
        for parameter in source.local_encoder.parameters():
            parameter.fill_(-0.5)
    recurrent_before = target.recurrent.weight_hh_l0.detach().clone()

    initialize_visual_encoders(target, source)

    assert all(
        torch.equal(target_value, source_value)
        for target_value, source_value in zip(
            target.global_encoder.state_dict().values(),
            source.global_encoder.state_dict().values(),
        )
    )
    assert all(
        torch.equal(target_value, source_value)
        for target_value, source_value in zip(
            target.local_encoder.state_dict().values(),
            source.local_encoder.state_dict().values(),
        )
    )
    assert torch.equal(target.recurrent.weight_hh_l0, recurrent_before)

    mismatched = HumanVisionPolicy(PolicyConfig(
        feature_size=5,
        recurrent_size=8,
        memory_size=0,
        proficiency_size=0,
        inference_mode="stream",
    ))
    with pytest.raises(ValueError, match="feature_size"):
        initialize_visual_encoders(mismatched, source)


def test_real_stream_policy_trains_and_records_tbptt_checkpoint_metadata(tmp_path) -> None:
    rng = np.random.default_rng(4)
    samples, history = 8, 2
    values = Demonstrations(
        global_frames=rng.random((samples, history, 6, 8, 8), dtype=np.float32),
        local_frames=rng.random((samples, history, 6, 8, 8), dtype=np.float32),
        actions=np.asarray(([4, 0] * 4), dtype=np.int64)[:, None].repeat(
            history, axis=1,
        ),
        risks=np.zeros((samples, history), dtype=np.float32),
        episode_ids=np.repeat(np.arange(4), 2),
    )
    output = tmp_path / "stateful.pt"
    config = stream_config()

    model, history_values = train_stateful_behavior_cloning(
        values,
        policy_config=config,
        training_config=StatefulTrainingConfig(
            seed=11,
            epochs=1,
            chunk_length=1,
            learning_rate=1e-3,
            validation_fraction=0.5,
            class_balance=False,
            device="cpu",
            validation_episode_ids=(1, 3),
            movement_onset_weight=5.0,
            direction_change_weight=3.0,
            episode_balanced=True,
            exact_action_loss_weight=0.25,
            direction_loss_weight=1.0,
            speed_loss_weight=0.2,
            direction_consistency_weight=0.1,
            future_visual_loss_weight=0.5,
            future_visual_horizons=(1,),
        ),
        output=output,
    )

    checkpoint = torch.load(output, map_location="cpu", weights_only=False)
    assert model.config.inference_mode == "stream"
    assert len(history_values) == 1
    assert history_values[0].train_future_visual_loss > 0.0
    assert history_values[0].validation_future_visual_loss > 0.0
    assert history_values[0].train_future_visual_labels == 2
    assert history_values[0].validation_future_visual_labels == 2
    assert checkpoint["version"] == 3
    assert checkpoint["training_data"]["training_mode"] == "episode_stateful_tbptt"
    assert checkpoint["training_data"]["inference_semantics"] == (
        "latest_visible_frame_stream"
    )
    assert checkpoint["training_data"]["tbptt_chunk_length"] == 1
    assert not (
        set(checkpoint["training_data"]["train_episode_ids"])
        & set(checkpoint["training_data"]["validation_episode_ids"])
    )
    assert checkpoint["training_data"]["train_episode_ids"] == [0, 2]
    assert checkpoint["training_data"]["validation_episode_ids"] == [1, 3]
    assert checkpoint["training_data"]["checkpoint_selection"] == (
        "minimum_complete_episode_validation_loss"
    )
    assert checkpoint["training_data"]["selected_epoch"] == 1
    assert math.isfinite(checkpoint["training_data"]["selected_validation_loss"])
    assert checkpoint["training_data"]["loss_weighting"] == {
        "class_balance": False,
        "class_balance_power": 0.5,
        "movement_onset_weight": 5.0,
        "direction_change_weight": 3.0,
        "exact_action_loss_weight": 0.25,
        "direction_loss_weight": 1.0,
        "speed_loss_weight": 0.2,
        "direction_consistency_weight": 0.1,
        "previous_action_dropout_probability": 0.0,
        "future_visual_loss_weight": 0.5,
        "future_visual_horizons_decisions": [1],
        "transition_source": "adjacent_teacher_actions_within_episode",
        "direction_semantics": "move_xy_ignoring_slow_mode",
        "train_movement_onsets": 2,
        "train_direction_changes": 0,
        "validation_movement_onsets": 2,
        "validation_direction_changes": 0,
    }
    prediction = checkpoint["training_data"]["future_visual_prediction"]
    assert prediction["enabled"] is True
    assert prediction["source"] == "per_decision_gru_hidden"
    assert prediction["target"] == (
        "stop_gradient_concatenated_global_local_visual_encoding"
    )
    assert prediction["horizons_decisions"] == [1]
    assert prediction["train_labels"] == 2
    assert prediction["validation_labels"] == 2
    assert prediction["validation_loss_included"] is True
    assert prediction["predictor_checkpointed"] is False
    assert checkpoint["training_config"]["future_visual_loss_weight"] == 0.5
    assert checkpoint["training_config"]["future_visual_horizons"] == (1,)
    assert not any(
        key.startswith("future_visual") for key in checkpoint["state_dict"]
    )
    assert checkpoint["training_data"]["episode_balance"] == {
        "enabled": True,
        "optimizer_step_unit": "complete_episode",
        "episode_order": "deterministic_epoch_shuffle",
        "optimizer_steps_per_epoch": 2,
        "optimizer_loss_reduction": "mean_of_per_episode_weighted_means",
        "validation_loss_reduction": "mean_of_per_episode_weighted_means",
        "train_supervised_labels": {"0": 2, "2": 2},
    }
