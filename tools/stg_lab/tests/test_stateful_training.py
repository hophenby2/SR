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
    reflect_horizontal_teacher_action_evidence,
    split_episode_ids,
    teacher_transition_sample_weights,
    train_stateful_behavior_cloning,
)
from stg_lab.training import (
    TEACHER_ACTION_COLLIDED_INDEX,
    TEACHER_ACTION_EVALUATION_FIELDS,
    TEACHER_ACTION_MINIMUM_MARGIN_INDEX,
    TEACHER_ACTION_SELECTED_INDEX,
    Demonstrations,
)


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


def attach_teacher_evidence(
    values: Demonstrations,
    *,
    selected_action: int = 5,
) -> None:
    shape = (*values.actions.shape, 18, len(TEACHER_ACTION_EVALUATION_FIELDS))
    evaluations = np.zeros(shape, dtype=np.float32)
    evaluations[..., TEACHER_ACTION_COLLIDED_INDEX] = 1.0
    evaluations[..., TEACHER_ACTION_MINIMUM_MARGIN_INDEX] = -1.0
    for action in (3, selected_action):
        evaluations[..., action, TEACHER_ACTION_COLLIDED_INDEX] = 0.0
        evaluations[..., action, TEACHER_ACTION_MINIMUM_MARGIN_INDEX] = 20.0
    evaluations[..., selected_action, TEACHER_ACTION_SELECTED_INDEX] = 1.0
    values.actions.fill(selected_action)
    values.teacher_action_evaluations = evaluations
    values.teacher_action_regrets = np.zeros(
        (*values.actions.shape, 18), dtype=np.float32,
    )
    values.teacher_action_evaluation_mask = np.ones(
        values.actions.shape, dtype=bool,
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


class FixedActionStreamPolicy(torch.nn.Module):
    def __init__(self, config: PolicyConfig, action: int) -> None:
        super().__init__()
        self.config = config
        self.action = action

    def forward(
        self,
        global_frames,
        local_frames,
        memory=None,
        proficiency=None,
        hidden=None,
    ):
        del local_frames, memory, proficiency, hidden
        batch, steps = global_frames.shape[:2]
        logits = torch.zeros(
            (batch, steps, self.config.action_count),
            dtype=global_frames.dtype,
            device=global_frames.device,
        )
        logits[..., self.action] = 5.0
        risk = torch.zeros(
            (batch, steps),
            dtype=global_frames.dtype,
            device=global_frames.device,
        )
        return logits, risk, None


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
    with pytest.raises(ValueError, match="nonnegative"):
        StatefulTrainingConfig(action_consistency_weight=-0.1)
    with pytest.raises(ValueError, match="nonnegative"):
        StatefulTrainingConfig(initial_policy_kl_weight=-0.1)
    with pytest.raises(ValueError, match="positive"):
        StatefulTrainingConfig(movement_stop_weight=0.0)
    with pytest.raises(ValueError, match="positive"):
        StatefulTrainingConfig(movement_speed_change_weight=0.0)
    with pytest.raises(ValueError, match="nonnegative"):
        StatefulTrainingConfig(transition_action_rank_weight=-0.1)
    with pytest.raises(ValueError, match="nonnegative"):
        StatefulTrainingConfig(movement_onset_rank_weight=-0.1)
    with pytest.raises(ValueError, match="nonnegative"):
        StatefulTrainingConfig(movement_speed_change_rank_weight=-0.1)
    with pytest.raises(ValueError, match="nonnegative"):
        StatefulTrainingConfig(motion_boundary_rank_weight=-0.1)
    with pytest.raises(ValueError, match="transition_action_rank_margin"):
        StatefulTrainingConfig(transition_action_rank_margin=-0.1)
    with pytest.raises(ValueError, match="motion_boundary_rank_margin"):
        StatefulTrainingConfig(motion_boundary_rank_margin=-0.1)
    with pytest.raises(ValueError, match="motion_boundary_rank_lookback"):
        StatefulTrainingConfig(motion_boundary_rank_lookback=0)
    with pytest.raises(ValueError, match="motion_boundary_rank_lookback"):
        StatefulTrainingConfig(motion_boundary_rank_lookback=4)
    with pytest.raises(ValueError, match="episode-balanced"):
        StatefulTrainingConfig(motion_boundary_rank_weight=1.0)
    with pytest.raises(ValueError, match="nonnegative"):
        StatefulTrainingConfig(safety_correction_pairwise_rank_weight=-0.1)
    with pytest.raises(ValueError, match="pairwise_rank_margin"):
        StatefulTrainingConfig(safety_correction_pairwise_rank_margin=-0.1)
    with pytest.raises(ValueError, match="episode-balanced"):
        StatefulTrainingConfig(safety_correction_pairwise_rank_weight=1.0)
    pairwise_only = StatefulTrainingConfig(
        episode_balanced=True,
        exact_action_loss_weight=0.0,
        safety_correction_pairwise_rank_weight=1.0,
    )
    assert pairwise_only.safety_correction_pairwise_rank_margin == 0.25
    with pytest.raises(ValueError, match="nonnegative"):
        StatefulTrainingConfig(safety_correction_top1_rank_weight=-0.1)
    with pytest.raises(ValueError, match="top1_rank_margin"):
        StatefulTrainingConfig(safety_correction_top1_rank_margin=-0.1)
    with pytest.raises(ValueError, match="episode-balanced"):
        StatefulTrainingConfig(safety_correction_top1_rank_weight=1.0)
    top1_only = StatefulTrainingConfig(
        episode_balanced=True,
        exact_action_loss_weight=0.0,
        safety_correction_top1_rank_weight=1.0,
    )
    assert top1_only.safety_correction_top1_rank_margin == 0.25
    with pytest.raises(ValueError, match="nonnegative"):
        StatefulTrainingConfig(safety_correction_minimal_edit_weight=-0.1)
    with pytest.raises(ValueError, match="minimal_edit_margin"):
        StatefulTrainingConfig(safety_correction_minimal_edit_margin=-0.1)
    with pytest.raises(ValueError, match="episode-balanced"):
        StatefulTrainingConfig(safety_correction_minimal_edit_weight=1.0)
    minimal_edit_only = StatefulTrainingConfig(
        episode_balanced=True,
        exact_action_loss_weight=0.0,
        safety_correction_minimal_edit_weight=1.0,
    )
    assert minimal_edit_only.safety_correction_minimal_edit_margin == 0.25
    with pytest.raises(ValueError, match="previous_action_dropout_probability"):
        StatefulTrainingConfig(previous_action_dropout_probability=1.01)
    with pytest.raises(ValueError, match="nonnegative"):
        StatefulTrainingConfig(future_visual_loss_weight=-0.1)
    with pytest.raises(ValueError, match="must be a boolean"):
        StatefulTrainingConfig(policy_head_only=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="cannot be combined"):
        StatefulTrainingConfig(
            policy_head_only=True,
            future_visual_loss_weight=1.0,
        )
    with pytest.raises(ValueError, match="duplicates"):
        StatefulTrainingConfig(future_visual_horizons=(20, 20))
    with pytest.raises(ValueError, match="strictly increasing"):
        StatefulTrainingConfig(future_visual_horizons=(40, 20))
    assert StatefulTrainingConfig(
        exact_action_loss_weight=0.0,
        soft_action_loss_weight=1.0,
    ).soft_action_loss_weight == 1.0
    with pytest.raises(ValueError, match="soft_action_temperature"):
        StatefulTrainingConfig(soft_action_temperature=0.0)
    with pytest.raises(ValueError, match="soft_action_safety_margin"):
        StatefulTrainingConfig(soft_action_safety_margin=-1.0)
    with pytest.raises(ValueError, match="collision ranking requires"):
        StatefulTrainingConfig(soft_action_collision_rank_weight=1.0)
    with pytest.raises(ValueError, match="collision_rank_margin"):
        StatefulTrainingConfig(soft_action_collision_rank_margin=-1.0)


def test_hard_action_ranking_requires_the_label_to_lead_all_alternatives() -> None:
    import stg_lab.stateful_training as stateful_training

    logits = torch.tensor((((0.0, 2.0, 1.5), (3.0, 1.0, 0.0)),))
    actions = torch.tensor(((1, 1),))

    terms = stateful_training._hard_action_ranking_terms(
        logits,
        actions,
        margin=1.0,
    )

    np.testing.assert_allclose(terms.numpy(), ((0.5, 3.0),))


def test_motion_boundary_ranking_compares_only_old_and_new_actions() -> None:
    import stg_lab.stateful_training as stateful_training

    logits = torch.tensor(((0.0, 2.0, 9.0), (1.0, -4.0, 3.0)))
    preferred = torch.tensor((1, 0))
    rejected = torch.tensor((0, 2))

    terms = stateful_training._motion_boundary_ranking_terms(
        logits,
        preferred,
        rejected,
        margin=1.0,
    )

    np.testing.assert_allclose(terms.numpy(), (0.0, 3.0))


def test_safety_correction_pairwise_ranking_compares_only_parent_argmax() -> None:
    import stg_lab.stateful_training as stateful_training

    logits = torch.tensor(((0.0, 2.0, 9.0), (1.0, -4.0, 3.0)))
    preferred = torch.tensor((1, 0))
    rejected = torch.tensor((0, 2))

    terms = stateful_training._safety_correction_pairwise_ranking_terms(
        logits,
        preferred,
        rejected,
        margin=0.5,
    )

    # Action 2's very large first-row logit is irrelevant because the frozen
    # parent rejected action is explicitly action 0.
    np.testing.assert_allclose(terms.numpy(), (0.0, 2.5))


def test_safety_correction_top1_ranking_blocks_third_action_takeover() -> None:
    import stg_lab.stateful_training as stateful_training

    logits = torch.tensor(((0.0, 2.0, 9.0), (1.0, -4.0, 3.0)))
    preferred = torch.tensor((1, 0))

    terms = stateful_training._safety_correction_top1_ranking_terms(
        logits,
        preferred,
        margin=0.5,
    )

    np.testing.assert_allclose(terms.numpy(), (7.5, 2.5))


def test_safety_correction_minimal_edit_changes_only_preferred_target_logit() -> None:
    import stg_lab.stateful_training as stateful_training

    reference = torch.tensor(((2.0, 1.0, 9.0, -3.0),), requires_grad=True)
    preferred = torch.tensor((1,))

    target = stateful_training._safety_correction_minimal_edit_target_logits(
        reference,
        preferred,
        margin=0.5,
    )

    assert target.requires_grad is False
    assert target[0, 1].item() == pytest.approx(9.5)
    np.testing.assert_allclose(
        target[0, (0, 2, 3)].numpy(),
        reference.detach()[0, (0, 2, 3)].numpy(),
    )


def test_safety_correction_minimal_edit_penalizes_third_action_takeover() -> None:
    import stg_lab.stateful_training as stateful_training

    reference = torch.tensor(((2.0, 1.0, 9.0, -3.0),))
    preferred = torch.tensor((1,))
    target = stateful_training._safety_correction_minimal_edit_target_logits(
        reference,
        preferred,
        margin=0.5,
    )
    matched = stateful_training._safety_correction_minimal_edit_terms(
        target,
        reference,
        preferred,
        margin=0.5,
    )
    assert matched.item() == pytest.approx(0.0, abs=1e-6)

    takeover = target.clone().detach()
    takeover[0, 3] += 6.0
    takeover.requires_grad_(True)
    loss = stateful_training._safety_correction_minimal_edit_terms(
        takeover,
        reference,
        preferred,
        margin=0.5,
    ).sum()
    loss.backward()

    assert loss.item() > 0.0
    assert takeover.grad is not None
    assert takeover.grad[0, 3].item() > 0.0


def test_policy_head_only_freezes_parameters_and_limits_optimizer(
    monkeypatch,
    tmp_path,
) -> None:
    import stg_lab.stateful_training as stateful_training

    values = demonstrations((10, 10, 20, 20), history=1)
    config = stream_config()
    model = HumanVisionPolicy(config)
    output = tmp_path / "policy-head-only.pt"
    training_passes = 0

    def fake_pass(current_model, _demonstrations, episodes, **kwargs):
        nonlocal training_passes
        optimizer = kwargs["optimizer"]
        if optimizer is not None:
            training_passes += 1
            named_parameters = dict(current_model.named_parameters())
            trainable_names = {
                name
                for name, parameter in named_parameters.items()
                if parameter.requires_grad
            }
            assert trainable_names == {
                "policy_head.weight",
                "policy_head.bias",
            }
            optimized_ids = {
                id(parameter)
                for group in optimizer.param_groups
                for parameter in group["params"]
            }
            assert optimized_ids == {
                id(named_parameters[name]) for name in trainable_names
            }

            inputs = torch.ones((1, 1, config.channels, 8, 8))
            logits, _risk, _hidden = current_model(inputs, inputs)
            logits.sum().backward()
            assert all(
                named_parameters[name].grad is not None
                for name in trainable_names
            )
            assert all(
                parameter.grad is None
                for name, parameter in named_parameters.items()
                if name not in trainable_names
            )
            optimizer.zero_grad(set_to_none=True)

        reference = kwargs["reference_model"]
        assert reference is not None
        assert all(
            not parameter.requires_grad for parameter in reference.parameters()
        )
        decisions = sum(episode.decisions for episode in episodes)
        return stateful_training.StatefulPassMetrics(
            loss=0.0,
            action_accuracy=1.0,
            risk_mae=0.0,
            labels=decisions,
            risk_labels=decisions,
            decisions=decisions,
            chunks=len(episodes),
            episodes=len(episodes),
            optimizer_steps=0,
            movement_onsets=0,
            direction_changes=0,
            future_visual_loss=0.0,
            future_visual_labels=0,
        )

    monkeypatch.setattr(stateful_training, "_stateful_pass", fake_pass)
    trained, _history = train_stateful_behavior_cloning(
        values,
        policy_config=config,
        training_config=StatefulTrainingConfig(
            epochs=1,
            validation_fraction=0.5,
            validation_episode_ids=(20,),
            class_balance=False,
            device="cpu",
            initial_policy_kl_weight=1.0,
            policy_head_only=True,
        ),
        model=model,
        output=output,
    )

    assert trained is model
    assert training_passes == 1
    checkpoint = torch.load(output, map_location="cpu", weights_only=False)
    assert checkpoint["training_config"]["policy_head_only"] is True
    assert checkpoint["training_data"]["policy_head_only"] is True


@pytest.mark.parametrize(
    "weight_name",
    (
        "safety_correction_pairwise_rank_weight",
        "safety_correction_minimal_edit_weight",
    ),
)
def test_safety_correction_training_requires_explicit_marked_data(
    weight_name,
) -> None:
    import stg_lab.stateful_training as stateful_training

    values = demonstrations((10, 10))
    reference = FixedActionStreamPolicy(stream_config(), action=0)
    arguments = {
        "chunk_length": 2,
        "risk_loss_weight": 0.0,
        "gradient_clip": 5.0,
        "device": "cpu",
        "optimizer": None,
        "episode_balanced": True,
        "exact_action_loss_weight": 0.0,
        weight_name: 1.0,
    }

    with pytest.raises(ValueError, match="marked latest-frame correction"):
        stateful_training._stateful_pass(
            RecordingStreamPolicy(stream_config()),
            values,
            ordered_episode_sequences(values),
            reference_model=reference,
            **arguments,
        )

    assert values.correction_mask is not None
    values.correction_mask[0, -1] = True
    with pytest.raises(ValueError, match="frozen reference model"):
        stateful_training._stateful_pass(
            RecordingStreamPolicy(stream_config()),
            values,
            ordered_episode_sequences(values),
            **arguments,
        )


def test_safety_correction_top1_requires_marked_data_but_not_reference() -> None:
    import stg_lab.stateful_training as stateful_training

    values = demonstrations((10, 10))
    arguments = {
        "chunk_length": 2,
        "risk_loss_weight": 0.0,
        "gradient_clip": 5.0,
        "device": "cpu",
        "optimizer": None,
        "episode_balanced": True,
        "exact_action_loss_weight": 0.0,
        "safety_correction_top1_rank_weight": 1.0,
    }

    with pytest.raises(ValueError, match="marked latest-frame correction"):
        stateful_training._stateful_pass(
            RecordingStreamPolicy(stream_config()),
            values,
            ordered_episode_sequences(values),
            **arguments,
        )

    assert values.correction_mask is not None
    values.correction_mask[0, -1] = True
    metrics = stateful_training._stateful_pass(
        RecordingStreamPolicy(stream_config()),
        values,
        ordered_episode_sequences(values),
        **arguments,
    )

    assert metrics.safety_correction_top1_rank_labels == 1


def test_safety_correction_rows_are_pairwise_only(monkeypatch) -> None:
    import stg_lab.stateful_training as stateful_training

    stationary = Action().discrete
    correction = Action(move_x=1).discrete
    values = demonstrations((10, 10, 10))
    attach_teacher_evidence(values, selected_action=correction)
    values.actions[:, -1] = (stationary, correction, stationary)
    values.supervision_mask = np.zeros_like(values.actions, dtype=np.bool_)
    values.supervision_mask[1, -1] = True
    assert values.teacher_action_evaluation_mask is not None
    values.teacher_action_evaluation_mask.fill(False)
    values.teacher_action_evaluation_mask[1, -1] = True
    assert values.correction_mask is not None
    values.correction_mask[1, -1] = True

    def constant_correction_terms(logits, preferred, rejected, *, margin):
        del preferred, rejected, margin
        return logits[:, 0] * 0.0 + 3.0

    monkeypatch.setattr(
        stateful_training,
        "_safety_correction_pairwise_ranking_terms",
        constant_correction_terms,
    )
    common = {
        "chunk_length": 1,
        "risk_loss_weight": 0.0,
        "gradient_clip": 5.0,
        "device": "cpu",
        "optimizer": None,
        "episode_balanced": True,
        "safety_correction_pairwise_rank_weight": 1.0,
        "reference_model": FixedActionStreamPolicy(stream_config(), action=stationary),
    }
    pairwise_only = stateful_training._stateful_pass(
        RecordingStreamPolicy(stream_config()),
        values,
        ordered_episode_sequences(values),
        exact_action_loss_weight=0.0,
        **common,
    )
    all_other_losses = stateful_training._stateful_pass(
        RecordingStreamPolicy(stream_config()),
        values,
        ordered_episode_sequences(values),
        exact_action_loss_weight=7.0,
        direction_loss_weight=5.0,
        speed_loss_weight=4.0,
        direction_consistency_weight=3.0,
        action_consistency_weight=2.0,
        transition_action_rank_weight=6.0,
        movement_onset_rank_weight=6.0,
        movement_speed_change_rank_weight=6.0,
        motion_boundary_rank_weight=6.0,
        soft_action_loss_weight=8.0,
        soft_action_collision_rank_weight=9.0,
        **common,
    )

    assert pairwise_only.loss == pytest.approx(3.0)
    assert all_other_losses.loss == pytest.approx(pairwise_only.loss)
    assert all_other_losses.safety_correction_pairwise_rank_labels == 1
    assert all_other_losses.transition_action_rank_labels == 0
    assert all_other_losses.movement_onset_rank_labels == 0
    assert all_other_losses.movement_speed_change_rank_labels == 0
    assert all_other_losses.motion_boundary_rank_events == 0


def test_safety_correction_pairwise_rank_normalizes_each_episode(monkeypatch) -> None:
    import stg_lab.stateful_training as stateful_training

    first_action = Action(move_x=1).discrete
    second_action = Action(move_y=1).discrete
    values = demonstrations((10, 20, 20, 20))
    values.actions[:, -1] = (first_action,) + (second_action,) * 3
    values.supervision_mask = np.zeros_like(values.actions, dtype=np.bool_)
    assert values.correction_mask is not None
    values.correction_mask[:, -1] = True

    def episode_specific_terms(logits, preferred, rejected, *, margin):
        del rejected, margin
        zero = logits[:, 0] * 0.0
        return torch.where(preferred == first_action, zero + 2.0, zero + 6.0)

    monkeypatch.setattr(
        stateful_training,
        "_safety_correction_pairwise_ranking_terms",
        episode_specific_terms,
    )
    metrics = stateful_training._stateful_pass(
        RecordingStreamPolicy(stream_config()),
        values,
        ordered_episode_sequences(values),
        chunk_length=1,
        risk_loss_weight=0.0,
        gradient_clip=5.0,
        device="cpu",
        optimizer=None,
        episode_balanced=True,
        exact_action_loss_weight=0.0,
        safety_correction_pairwise_rank_weight=3.0,
        reference_model=FixedActionStreamPolicy(stream_config(), action=0),
    )

    # Episode 10 contributes 2, while episode 20 contributes mean(6, 6, 6).
    # Their equal episode mean is 4, independent of correction count and chunks.
    assert metrics.safety_correction_pairwise_rank_loss == pytest.approx(4.0)
    assert metrics.loss == pytest.approx(12.0)
    assert metrics.safety_correction_pairwise_rank_labels == 4
    assert stateful_training._optimizer_steps_per_epoch(
        values,
        ordered_episode_sequences(values),
        chunk_length=1,
        episode_balanced=True,
        risk_loss_weight=0.0,
        risk_on_all_decisions=False,
        future_visual_loss_weight=0.0,
        future_visual_horizons=(1,),
        hard_action_terms_enabled=False,
        safety_correction_pairwise_rank_weight=3.0,
    ) == 2


def test_safety_correction_top1_rank_normalizes_each_episode(monkeypatch) -> None:
    import stg_lab.stateful_training as stateful_training

    first_action = Action(move_x=1).discrete
    second_action = Action(move_y=1).discrete
    values = demonstrations((10, 20, 20, 20))
    values.actions[:, -1] = (first_action,) + (second_action,) * 3
    values.supervision_mask = np.zeros_like(values.actions, dtype=np.bool_)
    assert values.correction_mask is not None
    values.correction_mask[:, -1] = True

    def episode_specific_terms(logits, preferred, *, margin):
        del margin
        zero = logits[:, 0] * 0.0
        return torch.where(preferred == first_action, zero + 2.0, zero + 6.0)

    monkeypatch.setattr(
        stateful_training,
        "_safety_correction_top1_ranking_terms",
        episode_specific_terms,
    )
    metrics = stateful_training._stateful_pass(
        RecordingStreamPolicy(stream_config()),
        values,
        ordered_episode_sequences(values),
        chunk_length=1,
        risk_loss_weight=0.0,
        gradient_clip=5.0,
        device="cpu",
        optimizer=None,
        episode_balanced=True,
        exact_action_loss_weight=0.0,
        safety_correction_top1_rank_weight=3.0,
    )

    assert metrics.safety_correction_top1_rank_loss == pytest.approx(4.0)
    assert metrics.loss == pytest.approx(12.0)
    assert metrics.safety_correction_top1_rank_labels == 4
    assert stateful_training._optimizer_steps_per_epoch(
        values,
        ordered_episode_sequences(values),
        chunk_length=1,
        episode_balanced=True,
        risk_loss_weight=0.0,
        risk_on_all_decisions=False,
        future_visual_loss_weight=0.0,
        future_visual_horizons=(1,),
        hard_action_terms_enabled=False,
        safety_correction_top1_rank_weight=3.0,
    ) == 2


def test_safety_correction_minimal_edit_normalizes_each_episode(monkeypatch) -> None:
    import stg_lab.stateful_training as stateful_training

    first_action = Action(move_x=1).discrete
    second_action = Action(move_y=1).discrete
    values = demonstrations((10, 20, 20, 20))
    values.actions[:, -1] = (first_action,) + (second_action,) * 3
    values.supervision_mask = np.zeros_like(values.actions, dtype=np.bool_)
    assert values.correction_mask is not None
    values.correction_mask[:, -1] = True

    def episode_specific_terms(logits, reference, preferred, *, margin):
        del reference, margin
        zero = logits[:, 0] * 0.0
        return torch.where(preferred == first_action, zero + 2.0, zero + 6.0)

    monkeypatch.setattr(
        stateful_training,
        "_safety_correction_minimal_edit_terms",
        episode_specific_terms,
    )
    metrics = stateful_training._stateful_pass(
        RecordingStreamPolicy(stream_config()),
        values,
        ordered_episode_sequences(values),
        chunk_length=1,
        risk_loss_weight=0.0,
        gradient_clip=5.0,
        device="cpu",
        optimizer=None,
        episode_balanced=True,
        exact_action_loss_weight=0.0,
        safety_correction_minimal_edit_weight=3.0,
        reference_model=FixedActionStreamPolicy(stream_config(), action=0),
    )

    assert metrics.safety_correction_minimal_edit_loss == pytest.approx(4.0)
    assert metrics.loss == pytest.approx(12.0)
    assert metrics.safety_correction_minimal_edit_labels == 4
    assert stateful_training._optimizer_steps_per_epoch(
        values,
        ordered_episode_sequences(values),
        chunk_length=1,
        episode_balanced=True,
        risk_loss_weight=0.0,
        risk_on_all_decisions=False,
        future_visual_loss_weight=0.0,
        future_visual_horizons=(1,),
        hard_action_terms_enabled=False,
        safety_correction_minimal_edit_weight=3.0,
    ) == 2


def test_safety_correction_rank_and_minimal_edit_losses_are_additive(
    monkeypatch,
) -> None:
    import stg_lab.stateful_training as stateful_training

    values = demonstrations((10,))
    values.supervision_mask = np.zeros_like(values.actions, dtype=np.bool_)
    assert values.correction_mask is not None
    values.correction_mask[0, -1] = True

    monkeypatch.setattr(
        stateful_training,
        "_safety_correction_pairwise_ranking_terms",
        lambda logits, preferred, rejected, *, margin: logits[:, 0] * 0.0 + 2.0,
    )
    monkeypatch.setattr(
        stateful_training,
        "_safety_correction_minimal_edit_terms",
        lambda logits, reference, preferred, *, margin: logits[:, 0] * 0.0 + 3.0,
    )
    monkeypatch.setattr(
        stateful_training,
        "_safety_correction_top1_ranking_terms",
        lambda logits, preferred, *, margin: logits[:, 0] * 0.0 + 4.0,
    )
    metrics = stateful_training._stateful_pass(
        RecordingStreamPolicy(stream_config()),
        values,
        ordered_episode_sequences(values),
        chunk_length=1,
        risk_loss_weight=0.0,
        gradient_clip=5.0,
        device="cpu",
        optimizer=None,
        episode_balanced=True,
        exact_action_loss_weight=0.0,
        safety_correction_pairwise_rank_weight=0.5,
        safety_correction_top1_rank_weight=2.0,
        safety_correction_minimal_edit_weight=4.0,
        reference_model=FixedActionStreamPolicy(stream_config(), action=0),
    )

    assert metrics.safety_correction_pairwise_rank_loss == pytest.approx(2.0)
    assert metrics.safety_correction_top1_rank_loss == pytest.approx(4.0)
    assert metrics.safety_correction_minimal_edit_loss == pytest.approx(3.0)
    assert metrics.loss == pytest.approx(
        0.5 * 2.0 + 2.0 * 4.0 + 4.0 * 3.0
    )


def test_safety_correction_pairwise_rank_reflects_both_action_ids(
    monkeypatch,
) -> None:
    import stg_lab.stateful_training as stateful_training

    right = Action(move_x=1).discrete
    left = Action(move_x=-1).discrete
    values = demonstrations((10,))
    values.actions[0, -1] = right
    assert values.correction_mask is not None
    values.correction_mask[0, -1] = True
    captured: list[tuple[list[int], list[int]]] = []

    def capture_terms(logits, preferred, rejected, *, margin):
        del margin
        captured.append((preferred.tolist(), rejected.tolist()))
        return logits[:, 0] * 0.0

    monkeypatch.setattr(
        stateful_training,
        "_safety_correction_pairwise_ranking_terms",
        capture_terms,
    )
    model = RecordingStreamPolicy(stream_config())
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    stateful_training._stateful_pass(
        model,
        values,
        ordered_episode_sequences(values),
        chunk_length=1,
        risk_loss_weight=0.0,
        gradient_clip=5.0,
        device="cpu",
        optimizer=optimizer,
        horizontal_reflection_probability=1.0,
        episode_balanced=True,
        exact_action_loss_weight=0.0,
        safety_correction_pairwise_rank_weight=1.0,
        reference_model=FixedActionStreamPolicy(stream_config(), action=left),
    )

    assert captured == [([left], [right])]


def test_safety_correction_top1_rank_reflects_preferred_action(monkeypatch) -> None:
    import stg_lab.stateful_training as stateful_training

    right = Action(move_x=1).discrete
    left = Action(move_x=-1).discrete
    values = demonstrations((10,))
    values.actions[0, -1] = right
    assert values.correction_mask is not None
    values.correction_mask[0, -1] = True
    captured: list[list[int]] = []

    def capture_terms(logits, preferred, *, margin):
        del margin
        captured.append(preferred.tolist())
        return logits[:, 0] * 0.0

    monkeypatch.setattr(
        stateful_training,
        "_safety_correction_top1_ranking_terms",
        capture_terms,
    )
    model = RecordingStreamPolicy(stream_config())
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    stateful_training._stateful_pass(
        model,
        values,
        ordered_episode_sequences(values),
        chunk_length=1,
        risk_loss_weight=0.0,
        gradient_clip=5.0,
        device="cpu",
        optimizer=optimizer,
        horizontal_reflection_probability=1.0,
        episode_balanced=True,
        exact_action_loss_weight=0.0,
        safety_correction_top1_rank_weight=1.0,
    )

    assert captured == [[left]]


def test_safety_correction_minimal_edit_reflects_target_and_reference_axis(
    monkeypatch,
) -> None:
    import stg_lab.stateful_training as stateful_training

    right = Action(move_x=1).discrete
    left = Action(move_x=-1).discrete
    values = demonstrations((10,))
    values.actions[0, -1] = right
    assert values.correction_mask is not None
    values.correction_mask[0, -1] = True
    captured: list[tuple[list[int], list[int]]] = []

    def capture_terms(logits, reference, preferred, *, margin):
        del margin
        captured.append((preferred.tolist(), reference.argmax(dim=-1).tolist()))
        return logits[:, 0] * 0.0

    monkeypatch.setattr(
        stateful_training,
        "_safety_correction_minimal_edit_terms",
        capture_terms,
    )
    model = RecordingStreamPolicy(stream_config())
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    stateful_training._stateful_pass(
        model,
        values,
        ordered_episode_sequences(values),
        chunk_length=1,
        risk_loss_weight=0.0,
        gradient_clip=5.0,
        device="cpu",
        optimizer=optimizer,
        horizontal_reflection_probability=1.0,
        episode_balanced=True,
        exact_action_loss_weight=0.0,
        safety_correction_minimal_edit_weight=1.0,
        reference_model=FixedActionStreamPolicy(stream_config(), action=left),
    )

    assert captured == [([left], [right])]


def test_initial_policy_kl_excludes_corrections_and_allows_empty_correction_split(
    monkeypatch,
) -> None:
    import stg_lab.stateful_training as stateful_training

    values = demonstrations((10, 10, 20, 20))
    values.supervision_mask = np.zeros_like(values.actions, dtype=np.bool_)
    assert values.correction_mask is not None
    values.correction_mask[0, -1] = True

    def unit_kl(input, target, *, reduction):
        del target
        assert reduction == "none"
        return input * 0.0 + 1.0

    monkeypatch.setattr(stateful_training.F, "kl_div", unit_kl)
    monkeypatch.setattr(
        stateful_training,
        "_safety_correction_pairwise_ranking_terms",
        lambda logits, preferred, rejected, *, margin: logits[:, 0] * 0.0,
    )
    monkeypatch.setattr(
        stateful_training,
        "_safety_correction_minimal_edit_terms",
        lambda logits, reference, preferred, *, margin: logits[:, 0] * 0.0,
    )
    monkeypatch.setattr(
        stateful_training,
        "_safety_correction_top1_ranking_terms",
        lambda logits, preferred, *, margin: logits[:, 0] * 0.0,
    )
    episodes = ordered_episode_sequences(values)
    corrected = stateful_training._stateful_pass(
        RecordingStreamPolicy(stream_config()),
        values,
        episodes[:1],
        chunk_length=1,
        risk_loss_weight=0.0,
        gradient_clip=5.0,
        device="cpu",
        optimizer=None,
        episode_balanced=True,
        exact_action_loss_weight=0.0,
        safety_correction_pairwise_rank_weight=1.0,
        safety_correction_top1_rank_weight=1.0,
        safety_correction_minimal_edit_weight=1.0,
        initial_policy_kl_weight=2.0,
        reference_model=FixedActionStreamPolicy(stream_config(), action=0),
    )
    assert corrected.safety_correction_pairwise_rank_labels == 1
    assert corrected.safety_correction_top1_rank_labels == 1
    assert corrected.safety_correction_minimal_edit_labels == 1
    assert corrected.initial_policy_kl_labels == 1
    assert corrected.initial_policy_kl_loss == pytest.approx(18.0)
    assert corrected.loss == pytest.approx(36.0)

    metrics = stateful_training._stateful_pass(
        RecordingStreamPolicy(stream_config()),
        values,
        episodes[1:],
        chunk_length=1,
        risk_loss_weight=0.0,
        gradient_clip=5.0,
        device="cpu",
        optimizer=None,
        episode_balanced=True,
        exact_action_loss_weight=0.0,
        safety_correction_pairwise_rank_weight=1.0,
        safety_correction_top1_rank_weight=1.0,
        safety_correction_minimal_edit_weight=1.0,
        initial_policy_kl_weight=2.0,
        reference_model=FixedActionStreamPolicy(stream_config(), action=0),
    )

    # The selected validation episode has no correction labels, but its two
    # non-correction rows remain valid frozen-policy anchors.
    assert metrics.labels == 0
    assert metrics.safety_correction_pairwise_rank_labels == 0
    assert metrics.safety_correction_top1_rank_labels == 0
    assert metrics.safety_correction_minimal_edit_labels == 0
    assert metrics.initial_policy_kl_labels == 2
    assert metrics.initial_policy_kl_loss == pytest.approx(18.0)
    assert metrics.loss == pytest.approx(36.0)


def test_motion_boundary_constraints_cover_both_sides_and_all_event_kinds() -> None:
    import stg_lab.stateful_training as stateful_training

    stationary = Action().discrete
    right = Action(move_x=1).discrete
    up = Action(move_y=1).discrete
    slow_right = Action(move_x=1, slow=True).discrete
    actions = (
        stationary, stationary, stationary,
        right, right, right,
        up, up,
        stationary, stationary,
        right, right,
        slow_right,
    )
    values = demonstrations((10,) * len(actions))
    values.actions[:, -1] = actions

    constraints = stateful_training._motion_boundary_rank_constraints(
        values,
        ordered_episode_sequences(values),
        lookback=3,
    )

    np.testing.assert_array_equal(constraints.event_indices, (3, 6, 8, 10, 12))
    assert constraints.event_kinds == (
        "onset", "turn", "stop", "onset", "speed_change",
    )
    np.testing.assert_array_equal(
        np.bincount(constraints.event_ids),
        (4, 4, 3, 3, 3),
    )
    for event_id, event_index in enumerate(constraints.event_indices):
        selected = constraints.event_ids == event_id
        assert constraints.pair_weights[selected].sum() == pytest.approx(1.0)
        event_pair = selected & (constraints.state_indices == event_index)
        assert event_pair.sum() == 1
        assert constraints.pair_weights[event_pair].item() == pytest.approx(0.5)
        assert constraints.preferred_actions[event_pair].item() == actions[event_index]
        assert constraints.rejected_actions[event_pair].item() != actions[event_index]
        preceding = selected & (constraints.state_indices != event_index)
        assert constraints.pair_weights[preceding].sum() == pytest.approx(0.5)
        assert np.all(constraints.preferred_actions[preceding] != actions[event_index])
        assert np.all(constraints.rejected_actions[preceding] == actions[event_index])


def test_motion_boundary_rank_is_event_normalized_across_holds_and_chunks(
    monkeypatch,
) -> None:
    import stg_lab.stateful_training as stateful_training

    stationary = Action().discrete
    moving = Action(move_x=1).discrete

    def constant_terms(logits, preferred, rejected, *, margin):
        del preferred, rejected, margin
        return logits[:, 0] * 0.0 + 2.0

    monkeypatch.setattr(
        stateful_training,
        "_motion_boundary_ranking_terms",
        constant_terms,
    )

    def rank_contribution(actions, *, chunk_length):
        values = demonstrations((10,) * len(actions))
        values.actions[:, -1] = actions
        episodes = ordered_episode_sequences(values)
        arguments = {
            "chunk_length": chunk_length,
            "risk_loss_weight": 0.0,
            "gradient_clip": 5.0,
            "device": "cpu",
            "optimizer": None,
            "episode_balanced": True,
            "exact_action_loss_weight": 1.0,
        }
        baseline = stateful_training._stateful_pass(
            RecordingStreamPolicy(stream_config()), values, episodes, **arguments,
        )
        ranked = stateful_training._stateful_pass(
            RecordingStreamPolicy(stream_config()),
            values,
            episodes,
            motion_boundary_rank_weight=5.0,
            motion_boundary_rank_margin=1.0,
            motion_boundary_rank_lookback=3,
            **arguments,
        )
        return ranked.loss - baseline.loss, ranked

    results = (
        rank_contribution((stationary, moving), chunk_length=1),
        rank_contribution(
            (stationary, stationary, stationary, stationary, moving),
            chunk_length=2,
        ),
        rank_contribution(
            (stationary, stationary, stationary, stationary, moving),
            chunk_length=5,
        ),
    )
    expected_pairs = (2, 4, 4)
    for (contribution, metrics), pairs in zip(results, expected_pairs, strict=True):
        assert contribution == pytest.approx(10.0)
        assert metrics.motion_boundary_rank_loss == pytest.approx(2.0)
        assert metrics.motion_boundary_rank_events == 1
        assert metrics.motion_boundary_rank_pairs == pairs
        assert metrics.motion_boundary_rank_margin_satisfaction == 0.0


def test_motion_boundary_rank_gives_pre_event_and_event_sides_equal_loss_mass(
    monkeypatch,
) -> None:
    import stg_lab.stateful_training as stateful_training

    stationary = Action().discrete
    moving = Action(move_x=1).discrete

    def side_specific_terms(logits, preferred, rejected, *, margin):
        del rejected, margin
        zero = logits[:, 0] * 0.0
        return torch.where(preferred == moving, zero + 6.0, zero + 2.0)

    monkeypatch.setattr(
        stateful_training,
        "_motion_boundary_ranking_terms",
        side_specific_terms,
    )

    def rank_contribution(actions, *, chunk_length):
        values = demonstrations((10,) * len(actions))
        values.actions[:, -1] = actions
        episodes = ordered_episode_sequences(values)
        arguments = {
            "chunk_length": chunk_length,
            "risk_loss_weight": 0.0,
            "gradient_clip": 5.0,
            "device": "cpu",
            "optimizer": None,
            "episode_balanced": True,
            "exact_action_loss_weight": 1.0,
        }
        baseline = stateful_training._stateful_pass(
            RecordingStreamPolicy(stream_config()), values, episodes, **arguments,
        )
        ranked = stateful_training._stateful_pass(
            RecordingStreamPolicy(stream_config()),
            values,
            episodes,
            motion_boundary_rank_weight=5.0,
            motion_boundary_rank_lookback=3,
            **arguments,
        )
        return ranked.loss - baseline.loss, ranked

    results = (
        rank_contribution((stationary, moving), chunk_length=1),
        rank_contribution(
            (stationary, stationary, stationary, stationary, moving),
            chunk_length=2,
        ),
    )
    for contribution, metrics in results:
        # Pre-event states contribute 0.5 * 2 and the event contributes
        # 0.5 * 6, independent of how many reliable lookback states exist.
        assert contribution == pytest.approx(20.0)
        assert metrics.motion_boundary_rank_loss == pytest.approx(4.0)
        assert metrics.motion_boundary_rank_events == 1


def test_motion_boundary_rank_balances_each_episode_before_combining_them(
    monkeypatch,
) -> None:
    import stg_lab.stateful_training as stateful_training

    stationary = Action().discrete
    right = Action(move_x=1).discrete
    left = Action(move_x=-1).discrete
    values = demonstrations((10, 10, 20, 20, 20, 20))
    values.actions[:, -1] = (
        stationary, right,
        stationary, left, stationary, left,
    )

    # Identify episode 10 by its unique right action; all left/stationary pairs
    # in episode 20 contribute three regardless of its three event count.
    def pair_specific_terms(logits, preferred, rejected, *, margin):
        del margin
        zero = logits[:, 0] * 0.0
        right_pair = (preferred == right) | (rejected == right)
        return torch.where(right_pair, zero + 1.0, zero + 3.0)

    monkeypatch.setattr(
        stateful_training,
        "_motion_boundary_ranking_terms",
        pair_specific_terms,
    )
    episodes = ordered_episode_sequences(values)
    arguments = {
        "chunk_length": 1,
        "risk_loss_weight": 0.0,
        "gradient_clip": 5.0,
        "device": "cpu",
        "optimizer": None,
        "episode_balanced": True,
        "exact_action_loss_weight": 1.0,
    }
    baseline = stateful_training._stateful_pass(
        RecordingStreamPolicy(stream_config()), values, episodes, **arguments,
    )
    ranked = stateful_training._stateful_pass(
        RecordingStreamPolicy(stream_config()),
        values,
        episodes,
        motion_boundary_rank_weight=2.0,
        motion_boundary_rank_lookback=1,
        **arguments,
    )

    assert ranked.loss - baseline.loss == pytest.approx(4.0)
    assert ranked.motion_boundary_rank_loss == pytest.approx(2.0)
    assert ranked.motion_boundary_rank_events == 4
    assert ranked.motion_boundary_rank_pairs == 8


def test_teacher_evaluated_rows_never_form_motion_boundary_hard_pairs(
    monkeypatch,
) -> None:
    import stg_lab.stateful_training as stateful_training

    stationary = Action().discrete
    moving = Action(move_x=1).discrete
    values = demonstrations((10, 10, 10, 10, 10))
    attach_teacher_evidence(values, selected_action=moving)
    values.actions[:, :] = np.asarray(
        (stationary, stationary, moving, moving, moving),
        dtype=np.int64,
    )[:, None]
    assert values.teacher_action_evaluation_mask is not None
    values.teacher_action_evaluation_mask.fill(False)
    values.teacher_action_evaluation_mask[2:4, -1] = True

    constraints = stateful_training._motion_boundary_rank_constraints(
        values,
        ordered_episode_sequences(values),
        lookback=3,
    )
    assert constraints.events == 0
    assert constraints.pairs == 0

    def unexpected_terms(*args, **kwargs):
        raise AssertionError("teacher-evaluated rows must not form hard pairs")

    monkeypatch.setattr(
        stateful_training,
        "_motion_boundary_ranking_terms",
        unexpected_terms,
    )
    metrics = stateful_training._stateful_pass(
        RecordingStreamPolicy(stream_config()),
        values,
        ordered_episode_sequences(values),
        chunk_length=1,
        risk_loss_weight=0.0,
        gradient_clip=5.0,
        device="cpu",
        optimizer=None,
        exact_action_loss_weight=1.0,
        motion_boundary_rank_weight=2.0,
    )
    assert metrics.motion_boundary_rank_events == 0
    assert metrics.motion_boundary_rank_pairs == 0
    assert metrics.motion_boundary_rank_loss == 0.0


def test_motion_boundary_rank_uses_one_episode_optimizer_step() -> None:
    import stg_lab.stateful_training as stateful_training

    values = demonstrations((10, 10, 10))
    values.actions[:, -1] = (
        Action().discrete,
        Action().discrete,
        Action(move_x=1).discrete,
    )
    steps = stateful_training._optimizer_steps_per_epoch(
        values,
        ordered_episode_sequences(values),
        chunk_length=1,
        episode_balanced=True,
        risk_loss_weight=0.0,
        risk_on_all_decisions=False,
        future_visual_loss_weight=0.0,
        future_visual_horizons=(1,),
        hard_action_terms_enabled=False,
        motion_boundary_rank_weight=1.0,
        motion_boundary_rank_lookback=2,
    )

    assert steps == 1


def test_motion_boundary_rank_rejects_per_chunk_optimizer_updates() -> None:
    import stg_lab.stateful_training as stateful_training

    values = demonstrations((10, 10))
    values.actions[:, -1] = (
        Action().discrete,
        Action(move_x=1).discrete,
    )
    model = RecordingStreamPolicy(stream_config())
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    with pytest.raises(ValueError, match="episode-balanced"):
        stateful_training._stateful_pass(
            model,
            values,
            ordered_episode_sequences(values),
            chunk_length=1,
            risk_loss_weight=0.0,
            gradient_clip=5.0,
            device="cpu",
            optimizer=optimizer,
            episode_balanced=False,
            motion_boundary_rank_weight=1.0,
        )


def test_motion_boundary_rank_reflects_old_and_new_action_ids(monkeypatch) -> None:
    import stg_lab.stateful_training as stateful_training

    stationary = Action().discrete
    right = Action(move_x=1).discrete
    left = Action(move_x=-1).discrete
    values = demonstrations((10, 10))
    values.actions[:, -1] = (stationary, right)
    captured: list[tuple[list[int], list[int]]] = []

    def capture_terms(logits, preferred, rejected, *, margin):
        del margin
        captured.append((preferred.tolist(), rejected.tolist()))
        return logits[:, 0] * 0.0

    monkeypatch.setattr(
        stateful_training,
        "_motion_boundary_ranking_terms",
        capture_terms,
    )
    model = RecordingStreamPolicy(stream_config())
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    stateful_training._stateful_pass(
        model,
        values,
        ordered_episode_sequences(values),
        chunk_length=2,
        risk_loss_weight=0.0,
        gradient_clip=5.0,
        device="cpu",
        optimizer=optimizer,
        horizontal_reflection_probability=1.0,
        episode_balanced=True,
        exact_action_loss_weight=1.0,
        motion_boundary_rank_weight=1.0,
    )

    assert captured == [([stationary, left], [left, stationary])]


def test_transition_ranking_is_independent_of_holds_chunks_and_sample_weights(
    monkeypatch,
) -> None:
    import stg_lab.stateful_training as stateful_training

    stationary = Action().discrete
    moving = Action(move_y=-1).discrete

    def constant_ranking_terms(logits, actions, *, margin):
        del actions, margin
        return logits[..., 0] * 0.0 + 2.0

    monkeypatch.setattr(
        stateful_training,
        "_hard_action_ranking_terms",
        constant_ranking_terms,
    )

    def rank_contribution(actions, *, chunk_length, onset_weight, stop_weight):
        values = demonstrations((10,) * len(actions))
        values.actions[:, -1] = actions
        episodes = ordered_episode_sequences(values)
        arguments = {
            "chunk_length": chunk_length,
            "risk_loss_weight": 0.0,
            "gradient_clip": 5.0,
            "device": "cpu",
            "optimizer": None,
            "episode_balanced": True,
            "movement_onset_weight": onset_weight,
            "movement_stop_weight": stop_weight,
            "exact_action_loss_weight": 1.0,
        }
        baseline = stateful_training._stateful_pass(
            RecordingStreamPolicy(stream_config()),
            values,
            episodes,
            **arguments,
        )
        ranked = stateful_training._stateful_pass(
            RecordingStreamPolicy(stream_config()),
            values,
            episodes,
            transition_action_rank_weight=3.0,
            transition_action_rank_margin=1.0,
            **arguments,
        )
        return ranked.loss - baseline.loss, ranked

    compact = (stationary, moving, moving, stationary)
    padded = (
        stationary, stationary, stationary, stationary,
        moving, moving, moving, moving,
        stationary, stationary, stationary, stationary,
    )
    results = [
        rank_contribution(
            compact,
            chunk_length=chunk_length,
            onset_weight=onset_weight,
            stop_weight=stop_weight,
        )
        for chunk_length, onset_weight, stop_weight in (
            (1, 1.0, 1.0),
            (2, 10.0, 7.0),
            (len(compact), 100.0, 50.0),
        )
    ]
    results.append(rank_contribution(
        padded,
        chunk_length=3,
        onset_weight=100.0,
        stop_weight=50.0,
    ))

    for contribution, metrics in results:
        assert contribution == pytest.approx(6.0)
        assert metrics.transition_action_rank_loss == pytest.approx(2.0)
        assert metrics.transition_action_rank_labels == 2
        assert metrics.transition_action_rank_margin_satisfaction == 0.0


def test_transition_ranking_balances_each_episode_before_combining_them(
    monkeypatch,
) -> None:
    import stg_lab.stateful_training as stateful_training

    stationary = Action().discrete
    first_motion = Action(move_x=1).discrete
    second_motion = Action(move_x=-1).discrete
    values = demonstrations((10, 10, 20, 20, 20, 20))
    values.actions[:, -1] = (
        stationary,
        first_motion,
        stationary,
        second_motion,
        stationary,
        second_motion,
    )

    def episode_specific_terms(logits, actions, *, margin):
        del margin
        zero = logits[..., 0] * 0.0
        return torch.where(actions == first_motion, zero + 1.0, zero + 3.0)

    monkeypatch.setattr(
        stateful_training,
        "_hard_action_ranking_terms",
        episode_specific_terms,
    )
    episodes = ordered_episode_sequences(values)
    arguments = {
        "chunk_length": 1,
        "risk_loss_weight": 0.0,
        "gradient_clip": 5.0,
        "device": "cpu",
        "optimizer": None,
        "episode_balanced": True,
        "exact_action_loss_weight": 1.0,
    }
    baseline = stateful_training._stateful_pass(
        RecordingStreamPolicy(stream_config()), values, episodes, **arguments,
    )
    ranked = stateful_training._stateful_pass(
        RecordingStreamPolicy(stream_config()),
        values,
        episodes,
        transition_action_rank_weight=2.0,
        **arguments,
    )

    assert ranked.loss - baseline.loss == pytest.approx(4.0)
    assert ranked.transition_action_rank_loss == pytest.approx(2.0)
    assert ranked.transition_action_rank_labels == 4


def test_onset_ranking_is_independent_of_holds_chunks_and_sample_weights(
    monkeypatch,
) -> None:
    import stg_lab.stateful_training as stateful_training

    stationary = Action().discrete
    right = Action(move_x=1).discrete
    left = Action(move_x=-1).discrete

    def constant_ranking_terms(logits, actions, *, margin):
        del actions, margin
        return logits[..., 0] * 0.0 + 2.0

    monkeypatch.setattr(
        stateful_training,
        "_hard_action_ranking_terms",
        constant_ranking_terms,
    )

    def rank_contribution(actions, *, chunk_length, sample_weight):
        values = demonstrations((10,) * len(actions))
        values.actions[:, -1] = actions
        episodes = ordered_episode_sequences(values)
        arguments = {
            "chunk_length": chunk_length,
            "risk_loss_weight": 0.0,
            "gradient_clip": 5.0,
            "device": "cpu",
            "optimizer": None,
            "episode_balanced": True,
            "movement_onset_weight": sample_weight,
            "exact_action_loss_weight": 1.0,
        }
        baseline = stateful_training._stateful_pass(
            RecordingStreamPolicy(stream_config()),
            values,
            episodes,
            **arguments,
        )
        ranked = stateful_training._stateful_pass(
            RecordingStreamPolicy(stream_config()),
            values,
            episodes,
            movement_onset_rank_weight=5.0,
            **arguments,
        )
        return ranked.loss - baseline.loss, ranked

    compact = (stationary, right, stationary, left)
    padded = (
        stationary, stationary, stationary, stationary,
        right, right, right, right,
        stationary, stationary, stationary, stationary,
        left, left, left, left,
    )
    results = [
        rank_contribution(
            compact,
            chunk_length=chunk_length,
            sample_weight=sample_weight,
        )
        for chunk_length, sample_weight in (
            (1, 1.0),
            (2, 10.0),
            (len(compact), 100.0),
        )
    ]
    results.append(rank_contribution(
        padded,
        chunk_length=3,
        sample_weight=100.0,
    ))

    for contribution, metrics in results:
        assert contribution == pytest.approx(10.0)
        assert metrics.movement_onset_rank_loss == pytest.approx(2.0)
        assert metrics.movement_onset_rank_labels == 2
        assert metrics.movement_onset_rank_margin_satisfaction == 0.0


def test_onset_ranking_balances_each_episode_before_combining_them(
    monkeypatch,
) -> None:
    import stg_lab.stateful_training as stateful_training

    stationary = Action().discrete
    right = Action(move_x=1).discrete
    left = Action(move_x=-1).discrete
    values = demonstrations((10, 10, 20, 20, 20, 20, 20, 20))
    values.actions[:, -1] = (
        stationary,
        right,
        stationary,
        left,
        stationary,
        left,
        stationary,
        left,
    )

    def episode_specific_terms(logits, actions, *, margin):
        del margin
        zero = logits[..., 0] * 0.0
        return torch.where(actions == right, zero + 1.0, zero + 3.0)

    monkeypatch.setattr(
        stateful_training,
        "_hard_action_ranking_terms",
        episode_specific_terms,
    )
    episodes = ordered_episode_sequences(values)
    arguments = {
        "chunk_length": 1,
        "risk_loss_weight": 0.0,
        "gradient_clip": 5.0,
        "device": "cpu",
        "optimizer": None,
        "episode_balanced": True,
        "exact_action_loss_weight": 1.0,
        "movement_onset_weight": 100.0,
    }
    baseline = stateful_training._stateful_pass(
        RecordingStreamPolicy(stream_config()), values, episodes, **arguments,
    )
    ranked = stateful_training._stateful_pass(
        RecordingStreamPolicy(stream_config()),
        values,
        episodes,
        movement_onset_rank_weight=2.0,
        **arguments,
    )

    assert ranked.loss - baseline.loss == pytest.approx(4.0)
    assert ranked.movement_onset_rank_loss == pytest.approx(2.0)
    assert ranked.movement_onset_rank_labels == 4


def test_soft_evaluated_onset_is_excluded_from_hard_ranking(monkeypatch) -> None:
    import stg_lab.stateful_training as stateful_training

    values = demonstrations((10, 10))
    moving = Action(move_y=-1).discrete
    attach_teacher_evidence(values, selected_action=moving)
    values.actions[0, -1] = Action().discrete
    assert values.teacher_action_evaluation_mask is not None
    values.teacher_action_evaluation_mask.fill(False)
    values.teacher_action_evaluation_mask[1, -1] = True

    def unexpected_ranking(*args, **kwargs):
        raise AssertionError("soft-evaluated movement onsets must not be hard-ranked")

    monkeypatch.setattr(
        stateful_training,
        "_hard_action_ranking_terms",
        unexpected_ranking,
    )
    metrics = stateful_training._stateful_pass(
        RecordingStreamPolicy(stream_config()),
        values,
        ordered_episode_sequences(values),
        chunk_length=1,
        risk_loss_weight=0.0,
        gradient_clip=5.0,
        device="cpu",
        optimizer=None,
        episode_balanced=True,
        exact_action_loss_weight=0.0,
        soft_action_loss_weight=1.0,
        movement_onset_rank_weight=2.0,
    )

    assert math.isfinite(metrics.loss)
    assert metrics.movement_onset_rank_labels == 0
    assert metrics.movement_onset_rank_loss == 0.0


def test_onset_ranking_counts_rank_only_optimizer_steps() -> None:
    import stg_lab.stateful_training as stateful_training

    stationary = Action().discrete
    moving = Action(move_x=1).discrete
    values = demonstrations((10, 10, 20, 20))
    values.actions[:, -1] = (stationary, moving, stationary, moving)
    episodes = ordered_episode_sequences(values)

    steps = stateful_training._optimizer_steps_per_epoch(
        values,
        episodes,
        chunk_length=1,
        episode_balanced=False,
        risk_loss_weight=0.0,
        risk_on_all_decisions=False,
        future_visual_loss_weight=0.0,
        future_visual_horizons=(1,),
        hard_action_terms_enabled=False,
        movement_onset_rank_weight=1.0,
    )

    assert steps == 2


def test_onset_ranking_is_additive_with_generic_transition_ranking(
    monkeypatch,
) -> None:
    import stg_lab.stateful_training as stateful_training

    values = demonstrations((10, 10))
    values.actions[:, -1] = (
        Action().discrete,
        Action(move_x=1).discrete,
    )

    def constant_ranking_terms(logits, actions, *, margin):
        del actions, margin
        return logits[..., 0] * 0.0 + 2.0

    monkeypatch.setattr(
        stateful_training,
        "_hard_action_ranking_terms",
        constant_ranking_terms,
    )
    episodes = ordered_episode_sequences(values)
    arguments = {
        "chunk_length": 1,
        "risk_loss_weight": 0.0,
        "gradient_clip": 5.0,
        "device": "cpu",
        "optimizer": None,
        "episode_balanced": True,
        "exact_action_loss_weight": 1.0,
    }
    baseline = stateful_training._stateful_pass(
        RecordingStreamPolicy(stream_config()), values, episodes, **arguments,
    )
    ranked = stateful_training._stateful_pass(
        RecordingStreamPolicy(stream_config()),
        values,
        episodes,
        transition_action_rank_weight=3.0,
        movement_onset_rank_weight=5.0,
        **arguments,
    )

    assert ranked.loss - baseline.loss == pytest.approx(16.0)
    assert ranked.transition_action_rank_labels == 1
    assert ranked.movement_onset_rank_labels == 1


def test_speed_change_ranking_is_independent_of_holds_chunks_and_sample_weights(
    monkeypatch,
) -> None:
    import stg_lab.stateful_training as stateful_training

    fast = Action(move_y=-1, slow=False).discrete
    slow = Action(move_y=-1, slow=True).discrete

    def constant_ranking_terms(logits, actions, *, margin):
        del actions, margin
        return logits[..., 0] * 0.0 + 2.0

    monkeypatch.setattr(
        stateful_training,
        "_hard_action_ranking_terms",
        constant_ranking_terms,
    )

    def rank_contribution(actions, *, chunk_length, sample_weight):
        values = demonstrations((10,) * len(actions))
        values.actions[:, -1] = actions
        episodes = ordered_episode_sequences(values)
        arguments = {
            "chunk_length": chunk_length,
            "risk_loss_weight": 0.0,
            "gradient_clip": 5.0,
            "device": "cpu",
            "optimizer": None,
            "episode_balanced": True,
            "movement_speed_change_weight": sample_weight,
            "exact_action_loss_weight": 1.0,
        }
        baseline = stateful_training._stateful_pass(
            RecordingStreamPolicy(stream_config()),
            values,
            episodes,
            **arguments,
        )
        ranked = stateful_training._stateful_pass(
            RecordingStreamPolicy(stream_config()),
            values,
            episodes,
            movement_speed_change_rank_weight=5.0,
            **arguments,
        )
        return ranked.loss - baseline.loss, ranked

    compact = (fast, slow, slow, fast)
    padded = (
        fast, fast, fast, fast,
        slow, slow, slow, slow,
        fast, fast, fast, fast,
    )
    results = [
        rank_contribution(
            compact,
            chunk_length=chunk_length,
            sample_weight=sample_weight,
        )
        for chunk_length, sample_weight in (
            (1, 1.0),
            (2, 10.0),
            (len(compact), 100.0),
        )
    ]
    results.append(rank_contribution(
        padded,
        chunk_length=3,
        sample_weight=100.0,
    ))

    for contribution, metrics in results:
        assert contribution == pytest.approx(10.0)
        assert metrics.movement_speed_change_rank_loss == pytest.approx(2.0)
        assert metrics.movement_speed_change_rank_labels == 2
        assert metrics.movement_speed_change_rank_margin_satisfaction == 0.0


def test_speed_change_ranking_balances_each_episode_before_combining_them(
    monkeypatch,
) -> None:
    import stg_lab.stateful_training as stateful_training

    right_fast = Action(move_x=1, slow=False).discrete
    right_slow = Action(move_x=1, slow=True).discrete
    left_fast = Action(move_x=-1, slow=False).discrete
    left_slow = Action(move_x=-1, slow=True).discrete
    values = demonstrations((10, 10, 20, 20, 20, 20))
    values.actions[:, -1] = (
        right_fast,
        right_slow,
        left_slow,
        left_fast,
        left_slow,
        left_fast,
    )

    def episode_specific_terms(logits, actions, *, margin):
        del margin
        zero = logits[..., 0] * 0.0
        right_direction = right_fast % 9
        return torch.where(actions % 9 == right_direction, zero + 1.0, zero + 3.0)

    monkeypatch.setattr(
        stateful_training,
        "_hard_action_ranking_terms",
        episode_specific_terms,
    )
    episodes = ordered_episode_sequences(values)
    arguments = {
        "chunk_length": 1,
        "risk_loss_weight": 0.0,
        "gradient_clip": 5.0,
        "device": "cpu",
        "optimizer": None,
        "episode_balanced": True,
        "exact_action_loss_weight": 1.0,
        "movement_speed_change_weight": 100.0,
    }
    baseline = stateful_training._stateful_pass(
        RecordingStreamPolicy(stream_config()), values, episodes, **arguments,
    )
    ranked = stateful_training._stateful_pass(
        RecordingStreamPolicy(stream_config()),
        values,
        episodes,
        movement_speed_change_rank_weight=2.0,
        **arguments,
    )

    assert ranked.loss - baseline.loss == pytest.approx(4.0)
    assert ranked.movement_speed_change_rank_loss == pytest.approx(2.0)
    assert ranked.movement_speed_change_rank_labels == 4


def test_soft_evaluated_speed_change_is_excluded_from_hard_ranking(
    monkeypatch,
) -> None:
    import stg_lab.stateful_training as stateful_training

    values = demonstrations((10, 10))
    fast = Action(move_y=-1, slow=False).discrete
    slow = Action(move_y=-1, slow=True).discrete
    attach_teacher_evidence(values, selected_action=slow)
    values.actions[0, -1] = fast
    assert values.teacher_action_evaluation_mask is not None
    values.teacher_action_evaluation_mask.fill(False)
    values.teacher_action_evaluation_mask[1, -1] = True

    def unexpected_ranking(*args, **kwargs):
        raise AssertionError("soft-evaluated speed changes must not be hard-ranked")

    monkeypatch.setattr(
        stateful_training,
        "_hard_action_ranking_terms",
        unexpected_ranking,
    )
    metrics = stateful_training._stateful_pass(
        RecordingStreamPolicy(stream_config()),
        values,
        ordered_episode_sequences(values),
        chunk_length=1,
        risk_loss_weight=0.0,
        gradient_clip=5.0,
        device="cpu",
        optimizer=None,
        episode_balanced=True,
        exact_action_loss_weight=0.0,
        soft_action_loss_weight=1.0,
        movement_speed_change_rank_weight=2.0,
    )

    assert math.isfinite(metrics.loss)
    assert metrics.movement_speed_change_rank_labels == 0
    assert metrics.movement_speed_change_rank_loss == 0.0


def test_soft_evaluated_transition_is_excluded_from_hard_ranking(monkeypatch) -> None:
    import stg_lab.stateful_training as stateful_training

    values = demonstrations((10, 10))
    moving = Action(move_y=-1).discrete
    attach_teacher_evidence(values, selected_action=moving)
    values.actions[0, -1] = Action().discrete
    assert values.teacher_action_evaluation_mask is not None
    values.teacher_action_evaluation_mask.fill(False)
    values.teacher_action_evaluation_mask[1, -1] = True

    def unexpected_ranking(*args, **kwargs):
        raise AssertionError("soft-evaluated transitions must not be hard-ranked")

    monkeypatch.setattr(
        stateful_training,
        "_hard_action_ranking_terms",
        unexpected_ranking,
    )
    metrics = stateful_training._stateful_pass(
        RecordingStreamPolicy(stream_config()),
        values,
        ordered_episode_sequences(values),
        chunk_length=1,
        risk_loss_weight=0.0,
        gradient_clip=5.0,
        device="cpu",
        optimizer=None,
        episode_balanced=True,
        exact_action_loss_weight=0.0,
        soft_action_loss_weight=1.0,
        transition_action_rank_weight=2.0,
    )

    assert math.isfinite(metrics.loss)
    assert metrics.transition_action_rank_labels == 0
    assert metrics.transition_action_rank_loss == 0.0


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

    evaluations = torch.zeros((1, 2, 18, len(TEACHER_ACTION_EVALUATION_FIELDS)))
    regrets = torch.arange(18, dtype=torch.float32).view(1, 1, 18).expand(
        1, 2, 18,
    )
    left = Action(move_x=-1, move_y=1).discrete
    mirrored_left = Action(move_x=1, move_y=1).discrete
    evaluations[..., left, TEACHER_ACTION_SELECTED_INDEX] = 1.0
    reflected_evaluations, reflected_regrets = (
        reflect_horizontal_teacher_action_evidence(evaluations, regrets)
    )
    assert reflected_evaluations[
        0, 0, mirrored_left, TEACHER_ACTION_SELECTED_INDEX
    ] == 1.0
    assert reflected_regrets[0, 0, mirrored_left] == regrets[0, 0, left]


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
        movement_stop_weight=7.0,
        direction_change_weight=3.0,
    )

    np.testing.assert_array_equal(weights, (1, 1, 5, 1, 3, 7, 1, 1, 3))
    speed_weighted = teacher_transition_sample_weights(
        values,
        movement_speed_change_weight=11.0,
    )
    np.testing.assert_array_equal(
        np.flatnonzero(speed_weighted == 11.0),
        (3, 7),
    )

    values.supervision_mask = np.ones_like(values.actions, dtype=np.bool_)
    values.supervision_mask[2, -1] = False
    masked = teacher_transition_sample_weights(
        values,
        movement_onset_weight=5.0,
        movement_stop_weight=7.0,
        direction_change_weight=3.0,
    )
    assert masked[2] == 1.0
    # The exact-hold transition row is masked, so the next exact label bridges it.
    assert masked[3] == 5.0
    assert masked[4] == 3.0
    assert masked[5] == 7.0

    values.supervision_mask[3, -1] = False
    sparse = teacher_transition_sample_weights(
        values,
        movement_onset_weight=5.0,
        movement_stop_weight=7.0,
        direction_change_weight=3.0,
    )
    # Two missing decisions are a sparse intervention gap, not one transition.
    assert sparse[4] == 1.0

    import stg_lab.stateful_training as stateful_training

    _onsets, _stops, _changes, speed_changes = (
        stateful_training._teacher_motion_transition_masks(
            values,
            ordered_episode_sequences(values),
        )
    )
    np.testing.assert_array_equal(np.flatnonzero(speed_changes), (7,))


def test_action_consistency_penalizes_only_stable_teacher_actions() -> None:
    import stg_lab.stateful_training as stateful_training

    stable = demonstrations((10, 10, 10, 10))
    episodes = ordered_episode_sequences(stable)
    baseline = stateful_training._stateful_pass(
        RecordingStreamPolicy(stream_config()),
        stable,
        episodes,
        chunk_length=2,
        risk_loss_weight=0.0,
        gradient_clip=5.0,
        device="cpu",
        optimizer=None,
    )
    regularized = stateful_training._stateful_pass(
        RecordingStreamPolicy(stream_config()),
        stable,
        episodes,
        chunk_length=2,
        risk_loss_weight=0.0,
        gradient_clip=5.0,
        device="cpu",
        optimizer=None,
        action_consistency_weight=2.0,
    )
    assert regularized.loss > baseline.loss

    changing = demonstrations((10, 10, 10, 10))
    # A real focus toggle is an action change even when move_xy is unchanged.
    changing.actions[:, -1] = (0, 9, 0, 9)
    changing_episodes = ordered_episode_sequences(changing)
    changing_baseline = stateful_training._stateful_pass(
        RecordingStreamPolicy(stream_config()),
        changing,
        changing_episodes,
        chunk_length=2,
        risk_loss_weight=0.0,
        gradient_clip=5.0,
        device="cpu",
        optimizer=None,
    )
    changing_regularized = stateful_training._stateful_pass(
        RecordingStreamPolicy(stream_config()),
        changing,
        changing_episodes,
        chunk_length=2,
        risk_loss_weight=0.0,
        gradient_clip=5.0,
        device="cpu",
        optimizer=None,
        action_consistency_weight=2.0,
    )
    assert changing_regularized.loss == pytest.approx(changing_baseline.loss)


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


def test_initial_policy_kl_uses_one_frozen_copy_of_initial_policy(
    monkeypatch,
) -> None:
    import stg_lab.stateful_training as stateful_training

    values = demonstrations((10, 10, 20, 20))
    config = stream_config()
    model = RecordingStreamPolicy(config)
    references = []
    training_calls = 0

    def fake_pass(current_model, _demonstrations, episodes, **kwargs):
        nonlocal training_calls
        reference = kwargs["reference_model"]
        references.append(reference)
        assert reference is not current_model
        assert reference.training is False
        assert all(not parameter.requires_grad for parameter in reference.parameters())
        assert float(reference.anchor.detach()) == pytest.approx(0.1)
        if kwargs["optimizer"] is not None:
            training_calls += 1
            with torch.no_grad():
                current_model.anchor.fill_(0.5 + training_calls)
        return stateful_training.StatefulPassMetrics(
            loss=float(training_calls),
            action_accuracy=1.0,
            risk_mae=0.0,
            labels=sum(episode.decisions for episode in episodes),
            risk_labels=sum(episode.decisions for episode in episodes),
            decisions=sum(episode.decisions for episode in episodes),
            chunks=len(episodes),
            episodes=len(episodes),
            optimizer_steps=0,
            movement_onsets=0,
            direction_changes=0,
            future_visual_loss=0.0,
            future_visual_labels=0,
        )

    monkeypatch.setattr(stateful_training, "_stateful_pass", fake_pass)
    train_stateful_behavior_cloning(
        values,
        policy_config=config,
        training_config=StatefulTrainingConfig(
            epochs=2,
            validation_fraction=0.5,
            validation_episode_ids=(20,),
            class_balance=False,
            device="cpu",
            initial_policy_kl_weight=1.0,
        ),
        model=model,
    )

    assert len(references) == 4
    assert all(reference is references[0] for reference in references)
    assert references[0].anchor.item() == pytest.approx(0.1)


@pytest.mark.parametrize(
    ("weight_name", "margin_name"),
    (
        (
            "safety_correction_pairwise_rank_weight",
            "safety_correction_pairwise_rank_margin",
        ),
        (
            "safety_correction_minimal_edit_weight",
            "safety_correction_minimal_edit_margin",
        ),
    ),
)
def test_safety_correction_losses_use_one_frozen_copy_of_parent_policy(
    monkeypatch, weight_name, margin_name,
) -> None:
    import stg_lab.stateful_training as stateful_training

    values = demonstrations((10, 10, 20, 20))
    assert values.correction_mask is not None
    values.correction_mask[0, -1] = True
    model = RecordingStreamPolicy(stream_config())
    references = []

    def fake_pass(current_model, _demonstrations, episodes, **kwargs):
        reference = kwargs["reference_model"]
        references.append(reference)
        assert reference is not current_model
        assert reference.training is False
        assert all(not parameter.requires_grad for parameter in reference.parameters())
        assert float(reference.anchor.detach()) == pytest.approx(0.1)
        assert kwargs[weight_name] == 1.0
        assert kwargs[margin_name] == 0.25
        if kwargs["optimizer"] is not None:
            with torch.no_grad():
                current_model.anchor.fill_(0.9)
        decisions = sum(episode.decisions for episode in episodes)
        return stateful_training.StatefulPassMetrics(
            loss=0.0,
            action_accuracy=1.0,
            risk_mae=0.0,
            labels=decisions,
            risk_labels=decisions,
            decisions=decisions,
            chunks=len(episodes),
            episodes=len(episodes),
            optimizer_steps=0,
            movement_onsets=0,
            direction_changes=0,
            future_visual_loss=0.0,
            future_visual_labels=0,
        )

    monkeypatch.setattr(stateful_training, "_stateful_pass", fake_pass)
    train_stateful_behavior_cloning(
        values,
        policy_config=stream_config(),
        training_config=StatefulTrainingConfig(
            epochs=1,
            validation_fraction=0.5,
            validation_episode_ids=(20,),
            class_balance=False,
            device="cpu",
            risk_loss_weight=0.0,
            episode_balanced=True,
            exact_action_loss_weight=0.0,
            **{weight_name: 1.0},
        ),
        model=model,
    )

    assert len(references) == 2
    assert references[0] is references[1]
    assert references[0].anchor.item() == pytest.approx(0.1)


def test_epoch_history_and_checkpoint_expose_correction_and_kl_metrics(
    monkeypatch,
    tmp_path,
) -> None:
    import stg_lab.stateful_training as stateful_training

    values = demonstrations((10, 10, 20, 20), history=1)
    assert values.correction_mask is not None
    values.correction_mask[0, -1] = True
    output = tmp_path / "pairwise-metrics.pt"

    def fake_pass(_model, _demonstrations, episodes, **kwargs):
        training = kwargs["optimizer"] is not None
        decisions = sum(episode.decisions for episode in episodes)
        return stateful_training.StatefulPassMetrics(
            loss=0.75 if training else 0.5,
            action_accuracy=0.25,
            risk_mae=0.125,
            labels=1 if training else 0,
            risk_labels=decisions,
            decisions=decisions,
            chunks=decisions,
            episodes=len(episodes),
            optimizer_steps=1 if training else 0,
            movement_onsets=0,
            direction_changes=0,
            future_visual_loss=0.0,
            future_visual_labels=0,
            safety_correction_pairwise_rank_loss=(2.5 if training else 0.0),
            safety_correction_pairwise_rank_labels=(1 if training else 0),
            safety_correction_pairwise_rank_margin_satisfaction=(
                0.4 if training else 0.0
            ),
            safety_correction_top1_rank_loss=(3.25 if training else 0.0),
            safety_correction_top1_rank_labels=(1 if training else 0),
            safety_correction_top1_rank_margin_satisfaction=(
                0.8 if training else 0.0
            ),
            safety_correction_minimal_edit_loss=(1.75 if training else 0.0),
            safety_correction_minimal_edit_labels=(1 if training else 0),
            safety_correction_minimal_edit_margin_satisfaction=(
                0.6 if training else 0.0
            ),
            initial_policy_kl_loss=0.125 if training else 0.25,
            initial_policy_kl_labels=1 if training else 2,
        )

    monkeypatch.setattr(stateful_training, "_stateful_pass", fake_pass)
    _model, history = train_stateful_behavior_cloning(
        values,
        policy_config=stream_config(),
        training_config=StatefulTrainingConfig(
            epochs=1,
            validation_fraction=0.5,
            validation_episode_ids=(20,),
            class_balance=False,
            device="cpu",
            risk_loss_weight=0.0,
            episode_balanced=True,
            exact_action_loss_weight=0.0,
            safety_correction_pairwise_rank_weight=1.0,
            safety_correction_top1_rank_weight=0.75,
            safety_correction_minimal_edit_weight=1.5,
            initial_policy_kl_weight=2.0,
        ),
        model=RecordingStreamPolicy(stream_config()),
        output=output,
    )

    assert len(history) == 1
    metrics = history[0]
    assert metrics.train_safety_correction_pairwise_rank_loss == pytest.approx(2.5)
    assert metrics.validation_safety_correction_pairwise_rank_loss == 0.0
    assert metrics.train_safety_correction_pairwise_rank_labels == 1
    assert metrics.validation_safety_correction_pairwise_rank_labels == 0
    assert (
        metrics.train_safety_correction_pairwise_rank_margin_satisfaction
        == pytest.approx(0.4)
    )
    assert (
        metrics.validation_safety_correction_pairwise_rank_margin_satisfaction
        == 0.0
    )
    assert metrics.train_safety_correction_top1_rank_loss == pytest.approx(3.25)
    assert metrics.validation_safety_correction_top1_rank_loss == 0.0
    assert metrics.train_safety_correction_top1_rank_labels == 1
    assert metrics.validation_safety_correction_top1_rank_labels == 0
    assert (
        metrics.train_safety_correction_top1_rank_margin_satisfaction
        == pytest.approx(0.8)
    )
    assert metrics.train_safety_correction_minimal_edit_loss == pytest.approx(1.75)
    assert metrics.validation_safety_correction_minimal_edit_loss == 0.0
    assert metrics.train_safety_correction_minimal_edit_labels == 1
    assert metrics.validation_safety_correction_minimal_edit_labels == 0
    assert (
        metrics.train_safety_correction_minimal_edit_margin_satisfaction
        == pytest.approx(0.6)
    )
    assert metrics.train_initial_policy_kl_loss == pytest.approx(0.125)
    assert metrics.validation_initial_policy_kl_loss == pytest.approx(0.25)
    assert metrics.train_initial_policy_kl_labels == 1
    assert metrics.validation_initial_policy_kl_labels == 2

    checkpoint = torch.load(output, map_location="cpu", weights_only=False)
    stored = checkpoint["history"][0]
    assert stored["train_safety_correction_pairwise_rank_loss"] == pytest.approx(2.5)
    assert stored["validation_safety_correction_pairwise_rank_labels"] == 0
    assert stored[
        "train_safety_correction_pairwise_rank_margin_satisfaction"
    ] == pytest.approx(0.4)
    assert stored["train_safety_correction_top1_rank_loss"] == pytest.approx(3.25)
    assert stored["validation_safety_correction_top1_rank_labels"] == 0
    assert stored[
        "train_safety_correction_top1_rank_margin_satisfaction"
    ] == pytest.approx(0.8)
    assert stored["train_safety_correction_minimal_edit_loss"] == pytest.approx(1.75)
    assert stored["validation_safety_correction_minimal_edit_labels"] == 0
    assert stored[
        "train_safety_correction_minimal_edit_margin_satisfaction"
    ] == pytest.approx(0.6)
    assert stored["train_initial_policy_kl_loss"] == pytest.approx(0.125)
    assert stored["validation_initial_policy_kl_loss"] == pytest.approx(0.25)
    assert stored["train_initial_policy_kl_labels"] == 1
    assert stored["validation_initial_policy_kl_labels"] == 2
    weighting = checkpoint["training_data"]["loss_weighting"]
    assert weighting["safety_correction_minimal_edit_target"] == (
        "copy_reference_logits_then_set_only_preferred_to_"
        "reference_max_plus_margin"
    )
    assert weighting["safety_correction_minimal_edit_other_actions"] == (
        "retain_frozen_reference_logits"
    )
    assert weighting["safety_correction_top1_rank_rejected"] == (
        "strongest_current_nonpreferred_policy_logit"
    )
    assert weighting["safety_correction_top1_rank_loss"] == (
        "relu(margin+max_nonpreferred_logit-preferred_logit)"
    )


def test_zero_initial_policy_kl_does_not_copy_or_route_a_reference(
    monkeypatch,
) -> None:
    import stg_lab.stateful_training as stateful_training

    values = demonstrations((10, 10, 20, 20))
    original_pass = stateful_training._stateful_pass
    routed_references = []

    def recording_pass(*args, **kwargs):
        routed_references.append(kwargs.get("reference_model"))
        return original_pass(*args, **kwargs)

    def unexpected_copy(_model):
        raise AssertionError("KL=0 must not copy a reference policy")

    monkeypatch.setattr(stateful_training, "_stateful_pass", recording_pass)
    monkeypatch.setattr(stateful_training.copy, "deepcopy", unexpected_copy)
    train_stateful_behavior_cloning(
        values,
        policy_config=stream_config(),
        training_config=StatefulTrainingConfig(
            epochs=1,
            validation_fraction=0.5,
            validation_episode_ids=(20,),
            class_balance=False,
            device="cpu",
            initial_policy_kl_weight=0.0,
        ),
        model=RecordingStreamPolicy(stream_config()),
    )

    assert routed_references == [None, None]


def test_top1_only_does_not_copy_or_route_a_reference(monkeypatch) -> None:
    import stg_lab.stateful_training as stateful_training

    values = demonstrations((10, 10, 20, 20))
    assert values.correction_mask is not None
    values.correction_mask[0, -1] = True
    original_pass = stateful_training._stateful_pass
    routed_references = []

    def recording_pass(*args, **kwargs):
        routed_references.append(kwargs.get("reference_model"))
        return original_pass(*args, **kwargs)

    def unexpected_copy(_model):
        raise AssertionError("top-1 ranking must not copy a reference policy")

    monkeypatch.setattr(stateful_training, "_stateful_pass", recording_pass)
    monkeypatch.setattr(stateful_training.copy, "deepcopy", unexpected_copy)
    train_stateful_behavior_cloning(
        values,
        policy_config=stream_config(),
        training_config=StatefulTrainingConfig(
            epochs=1,
            validation_fraction=0.5,
            validation_episode_ids=(20,),
            class_balance=False,
            device="cpu",
            risk_loss_weight=0.0,
            episode_balanced=True,
            exact_action_loss_weight=0.0,
            safety_correction_top1_rank_weight=1.0,
        ),
        model=RecordingStreamPolicy(stream_config()),
    )

    assert routed_references == [None, None]


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

    mismatched_local = HumanVisionPolicy(PolicyConfig(
        feature_size=4,
        recurrent_size=8,
        memory_size=0,
        proficiency_size=0,
        inference_mode="stream",
        local_feature_grid_size=8,
        local_downsample_stages=2,
    ))
    with pytest.raises(ValueError, match="local_feature_grid_size"):
        initialize_visual_encoders(mismatched_local, source)


def test_real_stream_policy_trains_and_records_tbptt_checkpoint_metadata(tmp_path) -> None:
    rng = np.random.default_rng(4)
    samples, history = 12, 2
    stationary = Action().discrete
    right_fast = Action(move_x=1, slow=False).discrete
    right_slow = Action(move_x=1, slow=True).discrete
    values = Demonstrations(
        global_frames=rng.random((samples, history, 6, 8, 8), dtype=np.float32),
        local_frames=rng.random((samples, history, 6, 8, 8), dtype=np.float32),
        actions=np.asarray(
            ([stationary, right_fast, right_slow] * 4), dtype=np.int64,
        )[:, None].repeat(
            history, axis=1,
        ),
        risks=np.zeros((samples, history), dtype=np.float32),
        episode_ids=np.repeat(np.arange(4), 3),
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
            movement_stop_weight=2.0,
            movement_speed_change_weight=4.0,
            direction_change_weight=3.0,
            episode_balanced=True,
            exact_action_loss_weight=0.25,
            direction_loss_weight=1.0,
            speed_loss_weight=0.2,
            direction_consistency_weight=0.1,
            action_consistency_weight=0.15,
            transition_action_rank_weight=0.4,
            transition_action_rank_margin=1.5,
            movement_onset_rank_weight=0.7,
            movement_speed_change_rank_weight=0.6,
            motion_boundary_rank_weight=0.8,
            motion_boundary_rank_margin=1.25,
            motion_boundary_rank_lookback=2,
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
    assert history_values[0].train_future_visual_labels == 4
    assert history_values[0].validation_future_visual_labels == 4
    assert history_values[0].train_transition_action_rank_labels == 4
    assert history_values[0].validation_transition_action_rank_labels == 4
    assert history_values[0].train_transition_action_rank_loss >= 0.0
    assert history_values[0].validation_transition_action_rank_loss >= 0.0
    assert 0.0 <= (
        history_values[0].validation_transition_action_rank_margin_satisfaction
    ) <= 1.0
    assert history_values[0].train_movement_onset_rank_labels == 2
    assert history_values[0].validation_movement_onset_rank_labels == 2
    assert history_values[0].train_movement_onset_rank_loss >= 0.0
    assert history_values[0].validation_movement_onset_rank_loss >= 0.0
    assert 0.0 <= (
        history_values[0].validation_movement_onset_rank_margin_satisfaction
    ) <= 1.0
    assert history_values[0].train_movement_speed_change_rank_labels == 2
    assert history_values[0].validation_movement_speed_change_rank_labels == 2
    assert history_values[0].train_movement_speed_change_rank_loss >= 0.0
    assert history_values[0].validation_movement_speed_change_rank_loss >= 0.0
    assert 0.0 <= (
        history_values[
            0
        ].validation_movement_speed_change_rank_margin_satisfaction
    ) <= 1.0
    assert history_values[0].train_motion_boundary_rank_events == 4
    assert history_values[0].validation_motion_boundary_rank_events == 4
    assert history_values[0].train_motion_boundary_rank_pairs == 8
    assert history_values[0].validation_motion_boundary_rank_pairs == 8
    assert history_values[0].train_motion_boundary_rank_loss >= 0.0
    assert history_values[0].validation_motion_boundary_rank_loss >= 0.0
    assert 0.0 <= (
        history_values[0].validation_motion_boundary_rank_margin_satisfaction
    ) <= 1.0
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
        "movement_stop_weight": 2.0,
        "movement_speed_change_weight": 4.0,
        "direction_change_weight": 3.0,
        "exact_action_loss_weight": 0.25,
        "direction_loss_weight": 1.0,
        "speed_loss_weight": 0.2,
        "direction_consistency_weight": 0.1,
        "action_consistency_weight": 0.15,
        "transition_action_rank_weight": 0.4,
        "transition_action_rank_margin": 1.5,
        "transition_action_rank_scope": (
            "reliable_supervised_movement_action_transitions"
        ),
        "transition_action_rank_reduction": (
            "mean_per_episode_over_transition_labels"
        ),
        "transition_action_rank_sample_weighting": (
            "independent_of_transition_sample_weights"
        ),
        "transition_action_rank_soft_evaluation_policy": (
            "exclude_teacher_evaluated_rows"
        ),
        "movement_onset_rank_weight": 0.7,
        "movement_onset_rank_margin": 1.5,
        "movement_onset_rank_margin_source": (
            "shared_transition_action_rank_margin"
        ),
        "movement_onset_rank_scope": (
            "reliable_hard_supervised_stationary_to_moving_transitions"
        ),
        "movement_onset_rank_reduction": (
            "mean_per_episode_over_onset_labels"
        ),
        "movement_onset_rank_sample_weighting": (
            "independent_of_transition_sample_weights"
        ),
        "movement_onset_rank_soft_evaluation_policy": (
            "exclude_teacher_evaluated_rows"
        ),
        "movement_onset_rank_interaction": (
            "additive_with_transition_action_rank_when_enabled"
        ),
        "movement_speed_change_rank_weight": 0.6,
        "movement_speed_change_rank_margin": 1.5,
        "movement_speed_change_rank_margin_source": (
            "shared_transition_action_rank_margin"
        ),
        "movement_speed_change_rank_scope": (
            "reliable_supervised_focused_speed_changes_while_"
            "movement_direction_is_held"
        ),
        "movement_speed_change_rank_reduction": (
            "mean_per_episode_over_speed_change_labels"
        ),
        "movement_speed_change_rank_sample_weighting": (
            "independent_of_transition_sample_weights"
        ),
        "movement_speed_change_rank_soft_evaluation_policy": (
            "exclude_teacher_evaluated_rows"
        ),
        "movement_speed_change_rank_interaction": (
            "additive_with_transition_action_rank_when_enabled"
        ),
        "motion_boundary_rank_weight": 0.8,
        "motion_boundary_rank_margin": 1.25,
        "motion_boundary_rank_lookback": 2,
        "motion_boundary_rank_event_types": [
            "onset",
            "stop",
            "turn",
            "speed_change",
        ],
        "motion_boundary_rank_pairing": (
            "preceding_old_action_over_future_new_action_and_"
            "event_new_action_over_old_action"
        ),
        "motion_boundary_rank_side_weighting": (
            "0.5_pre_event_total_and_0.5_event"
        ),
        "motion_boundary_rank_reduction": (
            "equal_side_weighted_pairs_per_event_then_mean_events_per_episode_"
            "then_mean_episodes"
        ),
        "motion_boundary_rank_optimizer_step_unit": "complete_episode",
        "motion_boundary_rank_hard_state_policy": (
            "supervision_mask_and_not_teacher_evaluated_with_at_most_one_"
            "intervening_nonhard_row"
        ),
        "motion_boundary_rank_soft_evaluation_policy": (
            "exclude_teacher_evaluated_rows_from_events_and_lookback_even_when_"
            "soft_loss_is_disabled"
        ),
        "motion_boundary_rank_episode_admission": (
            "input_episode_blocks_must_be_strict_successes;_outcome_is_not_a_"
            "model_input_or_npz_field"
        ),
        "train_motion_boundary_rank_events": 4,
        "train_motion_boundary_rank_pairs": 8,
        "train_motion_boundary_rank_event_counts": {
            "onset": 2,
            "stop": 0,
            "turn": 0,
            "speed_change": 2,
        },
        "validation_motion_boundary_rank_events": 4,
        "validation_motion_boundary_rank_pairs": 8,
        "validation_motion_boundary_rank_event_counts": {
            "onset": 2,
            "stop": 0,
            "turn": 0,
            "speed_change": 2,
        },
        "previous_action_dropout_probability": 0.0,
        "future_visual_loss_weight": 0.5,
        "future_visual_horizons_decisions": [1],
        "transition_source": (
            "consecutive_supervised_teacher_actions_with_at_most_one_"
            "masked_mixed_window_within_episode"
        ),
        "direction_semantics": "move_xy_ignoring_slow_mode",
        "train_movement_onsets": 2,
        "train_movement_stops": 0,
        "train_direction_changes": 0,
        "train_movement_speed_changes": 2,
        "validation_movement_onsets": 2,
        "validation_movement_stops": 0,
        "validation_direction_changes": 0,
        "validation_movement_speed_changes": 2,
    }
    prediction = checkpoint["training_data"]["future_visual_prediction"]
    assert prediction["enabled"] is True
    assert prediction["source"] == "per_decision_gru_hidden"
    assert prediction["target"] == (
        "stop_gradient_concatenated_global_local_visual_encoding"
    )
    assert prediction["horizons_decisions"] == [1]
    assert prediction["train_labels"] == 4
    assert prediction["validation_labels"] == 4
    assert prediction["validation_loss_included"] is True
    assert prediction["predictor_checkpointed"] is False
    assert checkpoint["training_config"]["future_visual_loss_weight"] == 0.5
    assert checkpoint["training_config"]["future_visual_horizons"] == (1,)
    assert checkpoint["training_config"]["movement_onset_rank_weight"] == 0.7
    assert checkpoint["training_config"]["motion_boundary_rank_weight"] == 0.8
    assert checkpoint["training_config"]["motion_boundary_rank_margin"] == 1.25
    assert checkpoint["training_config"]["motion_boundary_rank_lookback"] == 2
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
        "train_supervised_labels": {"0": 3, "2": 3},
    }


def test_stateful_training_supports_soft_only_safe_action_sets(tmp_path) -> None:
    values = demonstrations((10, 10, 20, 20), history=1)
    attach_teacher_evidence(values)
    values.supervision_mask = np.asarray(
        ((False,), (True,), (False,), (True,)), dtype=np.bool_,
    )
    output = tmp_path / "soft-set.pt"

    model, history = train_stateful_behavior_cloning(
        values,
        policy_config=stream_config(),
        training_config=StatefulTrainingConfig(
            seed=19,
            epochs=1,
            chunk_length=2,
            learning_rate=1e-3,
            validation_fraction=0.5,
            validation_episode_ids=(20,),
            class_balance=False,
            device="cpu",
            exact_action_loss_weight=0.0,
            soft_action_loss_weight=1.5,
            soft_action_temperature=3.0,
            soft_action_safety_margin=14.0,
        ),
        output=output,
    )

    checkpoint = torch.load(output, map_location="cpu", weights_only=False)
    assert len(history) == 1
    weighting = checkpoint["training_data"]["loss_weighting"]
    assert weighting["soft_action_loss_weight"] == 1.5
    assert weighting["soft_action_temperature"] == 3.0
    assert weighting["soft_action_safety_margin"] == 14.0
    assert weighting["soft_action_objective"] == (
        "negative_log_regret_weighted_acceptable_probability_mass"
    )
    assert weighting["soft_action_mask"] == (
        "teacher_action_evaluation_mask_independent_of_hard_supervision_mask"
    )
    evaluated = evaluate_stateful_policy(
        model,
        values,
        chunk_length=2,
        risk_loss_weight=0.0,
        device="cpu",
        exact_action_loss_weight=0.0,
        soft_action_loss_weight=1.5,
        soft_action_temperature=3.0,
        soft_action_safety_margin=14.0,
    )
    assert evaluated.labels == 4


def test_soft_action_sets_replace_hard_labels_on_evaluated_rows() -> None:
    values = demonstrations((10, 10), history=1)
    attach_teacher_evidence(values, selected_action=5)
    values.teacher_action_evaluations[
        ..., 0, TEACHER_ACTION_COLLIDED_INDEX
    ] = 0.0
    values.teacher_action_evaluations[
        ..., 0, TEACHER_ACTION_MINIMUM_MARGIN_INDEX
    ] = 20.0
    model = RecordingStreamPolicy(stream_config())

    soft_only = evaluate_stateful_policy(
        model,
        values,
        chunk_length=2,
        risk_loss_weight=0.0,
        device="cpu",
        exact_action_loss_weight=0.0,
        soft_action_loss_weight=1.0,
    )
    with_hard_weight = evaluate_stateful_policy(
        model,
        values,
        chunk_length=2,
        risk_loss_weight=0.0,
        device="cpu",
        exact_action_loss_weight=100.0,
        soft_action_loss_weight=1.0,
    )

    assert with_hard_weight.loss == pytest.approx(soft_only.loss)
