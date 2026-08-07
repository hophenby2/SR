from dataclasses import asdict
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import experiments.train_temporal_residual_adapter as residual_training
import stg_lab.residual_adapter as residual_adapter_module
from stg_lab.policy import PolicyConfig
from stg_lab.residual_adapter import (
    ResidualAdapterConfig,
    ResidualCorrectionAdapter,
    ResidualPolicyWrapper,
    ResidualRuntimeConfig,
    decode_residual_action_logits,
    ensemble_action_summary,
    finite_action_probabilities,
    load_residual_adapter,
    residual_candidate_selection,
    residual_future_onset_mask,
    residual_override_masks,
    save_residual_adapter,
    semantic_player_position_features,
)
from experiments.train_temporal_residual_adapter import (
    FIT_CHECKPOINT_KIND,
    FIT_CHECKPOINT_VERSION,
    MEMBERSHIP_LOSS_MODES,
    _calibrate,
    _clip_member_gradients,
    _early_event_cluster_metrics,
    _executed_action_context,
    _fit_source_inventory,
    _future_onset_calibration_diagnostics,
    _future_onset_split_diagnostics,
    _gradient_clip_parameter_groups,
    _load_fit_checkpoint,
    _dense_safety_losses,
    _labels_from_evidence,
    _metrics,
    _membership_loss_metadata,
    _normalize,
    _offline_deployment_eligible,
    _override_masks as _offline_override_masks,
    _physical_danger_loss,
    _predict_episode,
    _preferred_action_membership_loss,
    _preferred_action_loss,
    _preferred_action_set_loss,
    _parent_copy_loss,
    _save_fit_checkpoint,
    _strict_success,
    _train_member,
    _validate_distinct_workflow_paths,
    _validate_fit_resume_metadata,
    _validate_action_training_semantics,
    _validate_restored_membership_loss_mode,
    _validate_training_loss_weights,
    _write_json_atomic,
)
from stg_lab.provenance import file_sha256
from stg_lab.training import (
    Demonstrations,
    TEACHER_ACTION_COLLIDED_INDEX,
    TEACHER_ACTION_EVALUATION_FIELDS,
    TEACHER_ACTION_MINIMUM_MARGIN_INDEX,
    TEACHER_ACTION_SELECTED_INDEX,
)


class StubParent(torch.nn.Module):
    def __init__(self, recurrent_size: int = 4) -> None:
        super().__init__()
        self.config = PolicyConfig(
            feature_size=8,
            recurrent_size=recurrent_size,
            memory_size=0,
            proficiency_size=0,
            inference_mode="stream",
            local_feature_grid_size=2,
            local_downsample_stages=2,
        )
        self.anchor = torch.nn.Parameter(torch.zeros(()))


class VisualStubParent(StubParent):
    def forward_with_visual_features(
        self,
        global_frames,
        local_frames,
        memory=None,
        proficiency=None,
        hidden=None,
    ):
        batch, steps = global_frames.shape[:2]
        logits = global_frames.new_zeros((batch, steps, 18))
        logits[..., 5] = 1.0
        risk = global_frames.new_zeros((batch, steps))
        recurrent = global_frames.new_zeros((batch, steps, self.config.recurrent_size))
        visual = global_frames.new_zeros((batch, steps, self.config.feature_size * 2))
        next_hidden = global_frames.new_zeros((1, batch, self.config.recurrent_size))
        return logits, risk, next_hidden, recurrent, visual


class StreamingStubParent(StubParent):
    def forward_with_recurrent(
        self,
        global_frames,
        local_frames,
        memory=None,
        proficiency=None,
        hidden=None,
    ):
        batch, steps = global_frames.shape[:2]
        logits = global_frames.new_zeros((batch, steps, 18))
        logits[..., 5] = 1.0
        risk = global_frames.new_zeros((batch, steps))
        recurrent = global_frames.new_zeros(
            (batch, steps, self.config.recurrent_size),
        )
        next_hidden = global_frames.new_zeros(
            (1, batch, self.config.recurrent_size),
        )
        return logits, risk, next_hidden, recurrent


def _constant_adapter(
    *,
    gate_bias: float,
    correction_action: int,
    ensemble_size: int = 2,
) -> ResidualCorrectionAdapter:
    adapter = ResidualCorrectionAdapter(ResidualAdapterConfig(
        recurrent_size=4,
        hidden_size=4,
        ensemble_size=ensemble_size,
    ))
    with torch.no_grad():
        for parameter in adapter.parameters():
            parameter.zero_()
        for member in adapter.members:
            member.gate_head.bias.fill_(gate_bias)
            member.action_head.bias[correction_action] = 10.0
    return adapter


def _future_onset_runtime(
    *,
    prefer_safe_previous_action: bool = False,
) -> ResidualRuntimeConfig:
    return ResidualRuntimeConfig(
        gate_probability_threshold=0.5,
        minimum_member_gate_probability=0.5,
        action_probability_threshold=0.5,
        ensemble_agreement_threshold=1.0,
        legacy_gate_enabled=False,
        critic_enabled=True,
        prefer_safe_previous_action=prefer_safe_previous_action,
        critic_signal="physical_danger",
        parent_physical_danger_probability_threshold=0.7,
        candidate_physical_danger_probability_threshold=0.2,
        future_onset_gate_enabled=True,
    )


def _visual_physical_adapter() -> ResidualCorrectionAdapter:
    return ResidualCorrectionAdapter(ResidualAdapterConfig(
        recurrent_size=4,
        hidden_size=4,
        ensemble_size=2,
        executed_action_context=True,
        per_action_safety_critic=True,
        visual_latent_size=16,
        per_action_physical_danger=True,
    ))


def _residual_position_adapter(
    mode: str = "parent_residual_joint",
) -> ResidualCorrectionAdapter:
    return ResidualCorrectionAdapter(ResidualAdapterConfig(
        recurrent_size=4,
        hidden_size=4,
        ensemble_size=2,
        executed_action_context=True,
        per_action_safety_critic=True,
        visual_latent_size=16,
        per_action_physical_danger=True,
        action_logit_mode=mode,
        semantic_player_position=True,
    ))


def test_semantic_player_position_decodes_corners_and_weighted_centroid() -> None:
    corners = torch.zeros((4, 6, 3, 5), dtype=torch.float32)
    corners[0, 4, 0, 0] = 1.0
    corners[1, 4, 0, -1] = 1.0
    corners[2, 4, -1, 0] = 1.0
    corners[3, 4, -1, -1] = 1.0

    decoded = semantic_player_position_features(corners)

    assert torch.equal(decoded, torch.tensor([
        [-1.0, -1.0],
        [1.0, -1.0],
        [-1.0, 1.0],
        [1.0, 1.0],
    ]))

    weighted = torch.zeros((1, 2, 6, 3, 5), dtype=torch.float32)
    weighted[0, 0, 4, 0, 0] = 1.0
    weighted[0, 0, 4, -1, -1] = 3.0
    weighted[0, 1, 4, 1, 1] = 1.0
    weighted[0, 1, 4, 1, 3] = 1.0

    assert torch.allclose(
        semantic_player_position_features(weighted),
        torch.tensor([[[0.5, 0.5], [0.0, 0.0]]]),
    )


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_semantic_player_position_rejects_missing_and_nonfinite(
    bad_value: float,
) -> None:
    missing = torch.zeros((1, 6, 3, 5), dtype=torch.float32)
    with pytest.raises(ValueError, match="marker is missing"):
        semantic_player_position_features(missing)

    invalid = missing.clone()
    invalid[0, 4, 1, 2] = bad_value
    with pytest.raises(ValueError, match="must be finite"):
        semantic_player_position_features(invalid)


def test_semantic_player_position_config_and_raw_feature_contract() -> None:
    with pytest.raises(ValueError, match="requires a visual latent"):
        ResidualAdapterConfig(
            recurrent_size=4,
            semantic_player_position=True,
        )
    with pytest.raises(ValueError, match="must be a Boolean"):
        ResidualAdapterConfig(
            recurrent_size=4,
            visual_latent_size=16,
            semantic_player_position=1,  # type: ignore[arg-type]
        )

    legacy_config = ResidualAdapterConfig(
        recurrent_size=4,
        visual_latent_size=16,
    )
    conditioned_config = ResidualAdapterConfig(
        recurrent_size=4,
        visual_latent_size=16,
        semantic_player_position=True,
    )
    assert conditioned_config.feature_size == legacy_config.feature_size + 2

    adapter = ResidualCorrectionAdapter(conditioned_config)
    recurrent = torch.zeros((1, 2, 4))
    logits = torch.zeros((1, 2, 18))
    visual = torch.zeros((1, 2, 16))
    position = torch.tensor([[[-0.5, 0.25], [0.75, -1.0]]])

    with pytest.raises(ValueError, match="semantic player position is required"):
        adapter.raw_features(
            recurrent,
            logits,
            visual_features=visual,
        )
    with pytest.raises(ValueError, match="does not align"):
        adapter.raw_features(
            recurrent,
            logits,
            visual_features=visual,
            player_position_features=position[:, :1],
        )

    raw = adapter.raw_features(
        recurrent,
        logits,
        visual_features=visual,
        player_position_features=position,
    )
    assert raw.shape == (1, 2, conditioned_config.feature_size)
    assert torch.equal(raw[..., -2:], position)


def test_live_wrapper_passes_semantic_player_position_to_adapter() -> None:
    parent = VisualStubParent()
    adapter = ResidualCorrectionAdapter(ResidualAdapterConfig(
        recurrent_size=4,
        hidden_size=4,
        ensemble_size=2,
        visual_latent_size=16,
        semantic_player_position=True,
    ))
    wrapper = ResidualPolicyWrapper(
        parent,
        adapter,
        ResidualRuntimeConfig(
            gate_probability_threshold=1.0,
            minimum_member_gate_probability=1.0,
        ),
    )
    global_frames = torch.zeros((1, 2, 6, 3, 5), dtype=torch.float32)
    global_frames[0, 0, 4, 0, 0] = 1.0
    global_frames[0, 0, 4, -1, -1] = 3.0
    global_frames[0, 1, 4, 1, 1] = 1.0
    global_frames[0, 1, 4, 1, 3] = 1.0
    local_frames = torch.zeros((1, 2, 6, 3, 5), dtype=torch.float32)
    captured: list[torch.Tensor] = []

    def capture_features(_module, inputs) -> None:
        captured.append(inputs[0].detach().clone())

    handle = adapter.members[0].input_projection.register_forward_pre_hook(
        capture_features,
    )
    try:
        wrapper.forward_with_recurrent(global_frames, local_frames)
    finally:
        handle.remove()

    expected = semantic_player_position_features(global_frames)
    assert len(captured) == 1
    assert torch.equal(captured[0][..., -2:], expected)

    missing = global_frames.clone()
    missing[0, 1, 4].zero_()
    with pytest.raises(ValueError, match="marker is missing"):
        wrapper.forward_with_recurrent(missing, local_frames)


@pytest.mark.parametrize(
    "mode",
    ("parent_residual_joint", "parent_residual_factorized"),
)
def test_parent_residual_zero_delta_is_exactly_parent(mode: str) -> None:
    parent = torch.randn((2, 3, 18))
    delta = torch.zeros((4, 2, 3, 18))

    decoded = decode_residual_action_logits(delta, parent, mode)

    assert torch.equal(decoded, parent.unsqueeze(0).expand_as(decoded))


@pytest.mark.parametrize(
    "mode",
    ("parent_residual_joint", "parent_residual_factorized"),
)
@pytest.mark.parametrize(("parent_action", "target_action"), ((5, 14), (14, 5)))
def test_parent_residual_can_express_speed_switches(
    mode: str,
    parent_action: int,
    target_action: int,
) -> None:
    parent = torch.zeros(18)
    parent[parent_action] = 2.0
    delta = torch.zeros(18)
    delta[target_action] = 8.0

    decoded = decode_residual_action_logits(delta, parent, mode)

    assert parent.argmax().item() == parent_action
    assert decoded.argmax().item() == target_action


@pytest.mark.parametrize(
    "mode",
    ("parent_residual_joint", "parent_residual_factorized"),
)
@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_nonfinite_parent_residual_is_vetoed_after_decoding(
    mode: str,
    bad_value: float,
) -> None:
    parent = torch.randn((1, 1, 18))
    delta = torch.zeros((2, 1, 1, 18))
    delta[0, 0, 0, 14] = bad_value

    decoded = decode_residual_action_logits(delta, parent, mode)
    probabilities, member_finite = finite_action_probabilities(decoded)

    assert member_finite[:, 0, 0].tolist() == [False, True]
    assert probabilities[0, 0, 0].count_nonzero().item() == 0
    assert torch.isfinite(probabilities[1, 0, 0]).all()
    assert probabilities[1, 0, 0].sum().item() == pytest.approx(1.0)


def test_certified_membership_decodes_raw_logits_without_parent_dependence() -> None:
    config = ResidualAdapterConfig(
        recurrent_size=4,
        action_logit_mode="certified_membership",
    )
    assert config.action_logit_mode == "certified_membership"
    values = torch.randn((2, 3, 18))
    first_parent = torch.randn((3, 18))
    second_parent = torch.randn((3, 18)) * 10.0

    first = decode_residual_action_logits(
        values,
        first_parent,
        "certified_membership",
    )
    second = decode_residual_action_logits(
        values,
        second_parent,
        "certified_membership",
    )

    assert first is values
    assert second is values


@pytest.mark.parametrize(
    "mode",
    ("absolute", "parent_residual_joint", "parent_residual_factorized"),
)
def test_legacy_action_probabilities_remain_exact_softmax_and_shift_invariant(
    mode: str,
) -> None:
    logits = torch.randn((3, 4, 18))

    probabilities, finite = finite_action_probabilities(logits, mode)
    shifted, shifted_finite = finite_action_probabilities(logits + 7.25, mode)

    assert torch.equal(probabilities, torch.softmax(logits, dim=-1))
    assert torch.equal(shifted, torch.softmax(logits + 7.25, dim=-1))
    assert torch.allclose(probabilities, shifted, atol=1e-7, rtol=1e-6)
    assert finite.all()
    assert torch.equal(finite, shifted_finite)


def test_membership_probabilities_are_independent_sigmoids_not_shift_invariant(
) -> None:
    logits = torch.linspace(-3.0, 3.0, 18).reshape(1, 1, 18)

    probabilities, finite = finite_action_probabilities(
        logits,
        "certified_membership",
    )
    shifted, shifted_finite = finite_action_probabilities(
        logits + 2.0,
        "certified_membership",
    )

    assert torch.equal(probabilities, torch.sigmoid(logits))
    assert torch.equal(shifted, torch.sigmoid(logits + 2.0))
    assert not torch.allclose(probabilities, shifted)
    assert finite.tolist() == [[True]]
    assert torch.equal(finite, shifted_finite)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_membership_nonfinite_action_vetoes_the_complete_ensemble_decision(
    bad_value: float,
) -> None:
    logits = torch.zeros((3, 1, 18))
    logits[1, 0, 7] = bad_value

    probabilities, member_finite = finite_action_probabilities(
        logits,
        "certified_membership",
    )
    summary = ensemble_action_summary(probabilities, member_finite)

    assert member_finite[:, 0].tolist() == [True, False, True]
    assert probabilities[1, 0, 7].item() == 0.0
    assert torch.equal(
        probabilities[1, 0, :7],
        torch.full((7,), 0.5),
    )
    assert summary["action_all_members_finite"].tolist() == [False]
    assert torch.isfinite(summary["mean_action_probabilities"]).all()


def test_membership_offline_and_live_use_identical_ensemble_action_summary(
    monkeypatch,
) -> None:
    parent = StreamingStubParent()
    adapter = ResidualCorrectionAdapter(ResidualAdapterConfig(
        recurrent_size=4,
        hidden_size=4,
        ensemble_size=2,
        action_logit_mode="certified_membership",
    ))
    with torch.no_grad():
        for parameter in adapter.parameters():
            parameter.zero_()
        adapter.members[0].gate_head.bias.fill_(-20.0)
        adapter.members[1].gate_head.bias.fill_(-20.0)
        adapter.members[0].action_head.bias[6] = 2.0
        adapter.members[0].action_head.bias[7] = 1.0
        adapter.members[1].action_head.bias[6] = 3.0
        adapter.members[1].action_head.bias[7] = 0.5
    decisions = 3
    parent_logits = torch.zeros((1, decisions, 18))
    parent_logits[..., 5] = 1.0
    episode = SimpleNamespace(
        decisions=decisions,
        features=torch.zeros((1, decisions, adapter.config.feature_size)),
        parent_logits=parent_logits,
    )
    summaries: list[dict[str, torch.Tensor]] = []
    probability_inputs: list[torch.Tensor] = []
    original_summary = ensemble_action_summary

    def capture_summary(probabilities, member_finite):
        result = original_summary(probabilities, member_finite)
        probability_inputs.append(probabilities.detach().cpu().clone())
        summaries.append({
            name: value.detach().cpu().clone()
            for name, value in result.items()
        })
        return result

    monkeypatch.setattr(
        residual_training,
        "ensemble_action_summary",
        capture_summary,
    )
    monkeypatch.setattr(
        residual_adapter_module,
        "ensemble_action_summary",
        capture_summary,
    )
    offline = _predict_episode(adapter, episode, device="cpu")
    wrapper = ResidualPolicyWrapper(
        parent,
        adapter,
        ResidualRuntimeConfig(
            gate_probability_threshold=1.0,
            minimum_member_gate_probability=1.0,
        ),
    )
    frames = torch.zeros((1, decisions, 1))
    wrapper.forward_with_recurrent(frames, frames)

    assert len(summaries) == 2
    assert offline["action_probabilities"].shape == (2, decisions, 18)
    assert offline["action_member_finite"].shape == (2, decisions)
    assert offline["action_member_finite"].all()
    assert offline["mean_action_probabilities"].shape == (decisions, 18)
    assert torch.equal(offline["action_probabilities"], probability_inputs[0])
    assert torch.equal(probability_inputs[0], probability_inputs[1][:, 0])
    for name in (
        "action_all_members_finite",
        "mean_action_probabilities",
        "candidates",
        "action_confidence",
        "agreement",
    ):
        assert torch.equal(summaries[0][name], summaries[1][name][0])
    assert torch.equal(
        offline["mean_action_probabilities"],
        summaries[0]["mean_action_probabilities"],
    )
    assert torch.equal(offline["candidates"], summaries[0]["candidates"])
    assert torch.equal(
        offline["action_confidence"],
        summaries[0]["action_confidence"],
    )
    assert torch.equal(offline["agreement"], summaries[0]["agreement"])
    assert offline["candidates"].tolist() == [6, 6, 6]


def test_runtime_config_rejects_invalid_thresholds() -> None:
    with pytest.raises(ValueError, match="thresholds"):
        ResidualRuntimeConfig(gate_probability_threshold=1.01)
    with pytest.raises(ValueError, match="override_logit_margin"):
        ResidualRuntimeConfig(override_logit_margin=0.0)
    with pytest.raises(ValueError, match="agreement thresholds"):
        ResidualRuntimeConfig(candidate_safety_agreement_threshold=0.0)


@pytest.mark.parametrize("legacy_request", [False, True])
@pytest.mark.parametrize("parent_danger", [False, True])
@pytest.mark.parametrize("candidate_safe", [False, True])
def test_safety_critic_truth_table_vetoes_every_unverified_candidate(
    legacy_request: bool,
    parent_danger: bool,
    candidate_safe: bool,
) -> None:
    ensemble = 3
    parent_actions = torch.tensor([2])
    candidates = torch.tensor([5])
    collision = torch.full((ensemble, 1, 18), 0.1)
    margins = torch.full((ensemble, 1, 18), 20.0)
    if parent_danger:
        collision[:, 0, 2] = 0.9
    if not candidate_safe:
        margins[:, 0, 5] = 0.0
    runtime = ResidualRuntimeConfig(
        gate_probability_threshold=0.5,
        minimum_member_gate_probability=0.5,
        action_probability_threshold=0.5,
        ensemble_agreement_threshold=1.0,
        critic_enabled=True,
        parent_collision_probability_threshold=0.8,
        candidate_collision_probability_threshold=0.2,
        parent_minimum_margin_threshold=8.0,
        candidate_minimum_margin_threshold=8.0,
        parent_danger_agreement_threshold=2.0 / 3.0,
        candidate_safety_agreement_threshold=1.0,
    )

    masks = residual_override_masks(
        mean_gate=torch.tensor([0.9 if legacy_request else 0.1]),
        minimum_gate=torch.tensor([0.9 if legacy_request else 0.1]),
        action_all_members_finite=torch.ones(1, dtype=torch.bool),
        correction_confidence=torch.tensor([0.9]),
        agreement=torch.tensor([1.0]),
        correction_actions=candidates,
        parent_actions=parent_actions,
        runtime_config=runtime,
        collision_probabilities=collision,
        minimum_margins=margins,
    )

    assert bool(masks["override"].item()) is (
        candidate_safe and (legacy_request or parent_danger)
    )
    assert bool(masks["candidate_safety_veto"].item()) is (
        not candidate_safe and (legacy_request or parent_danger)
    )


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_safety_critic_nonfinite_candidate_is_fail_closed(bad_value: float) -> None:
    collision = torch.zeros((3, 1, 18))
    margins = torch.full((3, 1, 18), 20.0)
    margins[:, 0, 5] = bad_value
    masks = residual_override_masks(
        mean_gate=torch.tensor([1.0]),
        minimum_gate=torch.tensor([1.0]),
        action_all_members_finite=torch.ones(1, dtype=torch.bool),
        correction_confidence=torch.tensor([1.0]),
        agreement=torch.tensor([1.0]),
        correction_actions=torch.tensor([5]),
        parent_actions=torch.tensor([2]),
        runtime_config=ResidualRuntimeConfig(
            critic_enabled=True,
            gate_probability_threshold=0.5,
            minimum_member_gate_probability=0.5,
        ),
        collision_probabilities=collision,
        minimum_margins=margins,
    )

    assert not bool(masks["candidate_safe"].item())
    assert not bool(masks["override"].item())


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_physical_danger_critic_is_direct_and_fail_closed(bad_value: float) -> None:
    danger = torch.full((3, 1, 18), 0.1)
    danger[:, 0, 2] = 0.9
    runtime = ResidualRuntimeConfig(
        critic_enabled=True,
        critic_signal="physical_danger",
        legacy_gate_enabled=False,
        parent_physical_danger_probability_threshold=0.7,
        candidate_physical_danger_probability_threshold=0.2,
        action_probability_threshold=0.5,
    )
    values = dict(
        mean_gate=torch.zeros(1),
        minimum_gate=torch.zeros(1),
        action_all_members_finite=torch.ones(1, dtype=torch.bool),
        correction_confidence=torch.ones(1),
        agreement=torch.ones(1),
        correction_actions=torch.tensor([5]),
        parent_actions=torch.tensor([2]),
        runtime_config=runtime,
    )

    accepted = residual_override_masks(
        **values,
        physical_danger_probabilities=danger,
    )
    assert accepted["parent_danger"].tolist() == [True]
    assert accepted["candidate_safe"].tolist() == [True]
    assert accepted["override"].tolist() == [True]

    danger[:, 0, 5] = bad_value
    rejected = residual_override_masks(
        **values,
        physical_danger_probabilities=danger,
    )
    assert rejected["candidate_safe"].tolist() == [False]
    assert rejected["override"].tolist() == [False]


@pytest.mark.parametrize("candidate_safe", [False, True])
def test_future_onset_can_only_override_with_a_currently_safe_candidate(
    candidate_safe: bool,
) -> None:
    danger = torch.full((3, 1, 18), 0.1)
    if not candidate_safe:
        danger[:, 0, 5] = 0.9

    masks = residual_override_masks(
        mean_gate=torch.tensor([0.9]),
        minimum_gate=torch.tensor([0.9]),
        action_all_members_finite=torch.ones(1, dtype=torch.bool),
        correction_confidence=torch.tensor([0.9]),
        agreement=torch.tensor([1.0]),
        correction_actions=torch.tensor([5]),
        parent_actions=torch.tensor([2]),
        runtime_config=_future_onset_runtime(),
        physical_danger_probabilities=danger,
    )

    assert masks["parent_danger"].tolist() == [False]
    assert masks["critic_request"].tolist() == [False]
    assert masks["future_onset_request"].tolist() == [True]
    assert masks["candidate_safe"].tolist() == [candidate_safe]
    assert masks["future_onset_accepted"].tolist() == [candidate_safe]
    assert masks["candidate_safety_veto"].tolist() == [not candidate_safe]
    assert masks["override"].tolist() == [candidate_safe]


@pytest.mark.parametrize(
    ("mean_gate", "minimum_gate"),
    [
        (float("nan"), 0.9),
        (float("inf"), 0.9),
        (0.9, float("nan")),
        (0.9, float("inf")),
    ],
)
def test_future_onset_nonfinite_gate_is_fail_closed(
    mean_gate: float,
    minimum_gate: float,
) -> None:
    runtime = _future_onset_runtime()
    mean = torch.tensor([mean_gate])
    minimum = torch.tensor([minimum_gate])

    onset = residual_future_onset_mask(mean, minimum, runtime)
    masks = residual_override_masks(
        mean_gate=mean,
        minimum_gate=minimum,
        action_all_members_finite=torch.ones(1, dtype=torch.bool),
        correction_confidence=torch.tensor([0.9]),
        agreement=torch.tensor([1.0]),
        correction_actions=torch.tensor([5]),
        parent_actions=torch.tensor([2]),
        runtime_config=runtime,
        physical_danger_probabilities=torch.full((3, 1, 18), 0.1),
    )

    assert onset.tolist() == [False]
    assert masks["future_onset_request"].tolist() == [False]
    assert masks["override"].tolist() == [False]


def test_future_onset_rejects_unproven_previous_action_inertia() -> None:
    with pytest.raises(ValueError, match="continuation must be learned"):
        _future_onset_runtime(prefer_safe_previous_action=True)


def test_offline_future_onset_masks_match_runtime_primitives() -> None:
    runtime = _future_onset_runtime()
    episode = SimpleNamespace(
        parent_actions=torch.tensor([2, 2, 2]),
        previous_actions=torch.tensor([2, 5, 2]),
    )
    danger = torch.full((3, 3, 18), 0.1)
    danger[:, 1, 6] = 0.9
    values = {
        "mean_gate": torch.tensor([0.9, 0.9, 0.1]),
        "minimum_gate": torch.tensor([0.9, 0.9, 0.1]),
        "action_all_members_finite": torch.ones(3, dtype=torch.bool),
        "action_confidence": torch.tensor([0.9, 0.9, 0.9]),
        "agreement": torch.ones(3),
        "candidates": torch.tensor([5, 6, 7]),
        "physical_danger_probabilities": danger,
    }

    offline = _offline_override_masks(values, episode, runtime)
    onset = residual_future_onset_mask(
        values["mean_gate"],
        values["minimum_gate"],
        runtime,
    )
    direct = residual_candidate_selection(
        correction_actions=values["candidates"],
        correction_confidence=values["action_confidence"],
        agreement=values["agreement"],
        previous_actions=episode.previous_actions,
        runtime_config=runtime,
        physical_danger_probabilities=danger,
        parent_actions=episode.parent_actions,
        future_onset=onset,
    )
    masks = residual_override_masks(
        mean_gate=values["mean_gate"],
        minimum_gate=values["minimum_gate"],
        action_all_members_finite=values["action_all_members_finite"],
        correction_confidence=direct["correction_confidence"],
        agreement=direct["agreement"],
        correction_actions=direct["correction_actions"],
        parent_actions=episode.parent_actions,
        runtime_config=runtime,
        physical_danger_probabilities=danger,
    )
    direct.update(masks)

    assert offline.keys() == direct.keys()
    for name in offline:
        assert torch.equal(offline[name], direct[name]), name


def test_safety_critic_prefers_only_a_fully_certified_previous_action() -> None:
    collision = torch.zeros((3, 1, 18))
    margins = torch.full((3, 1, 18), 20.0)
    runtime = ResidualRuntimeConfig(
        critic_enabled=True,
        prefer_safe_previous_action=True,
        candidate_collision_probability_threshold=0.2,
        candidate_minimum_margin_threshold=8.0,
        candidate_safety_agreement_threshold=1.0,
    )

    selected = residual_candidate_selection(
        correction_actions=torch.tensor([2]),
        correction_confidence=torch.tensor([0.4]),
        agreement=torch.tensor([2.0 / 3.0]),
        previous_actions=torch.tensor([5]),
        runtime_config=runtime,
        collision_probabilities=collision,
        minimum_margins=margins,
    )

    assert selected["correction_actions"].tolist() == [5]
    assert selected["used_previous"].tolist() == [True]
    assert selected["correction_confidence"].tolist() == [1.0]
    margins[0, 0, 5] = 0.0
    rejected = residual_candidate_selection(
        correction_actions=torch.tensor([2]),
        correction_confidence=torch.tensor([0.4]),
        agreement=torch.tensor([2.0 / 3.0]),
        previous_actions=torch.tensor([5]),
        runtime_config=runtime,
        collision_probabilities=collision,
        minimum_margins=margins,
    )
    assert rejected["correction_actions"].tolist() == [2]
    assert rejected["used_previous"].tolist() == [False]


def test_action_nonfinite_veto_survives_safe_previous_selection() -> None:
    runtime = ResidualRuntimeConfig(
        critic_enabled=True,
        prefer_safe_previous_action=True,
        critic_signal="physical_danger",
    )
    episode = SimpleNamespace(
        parent_actions=torch.tensor([2]),
        previous_actions=torch.tensor([5]),
    )
    values = {
        "mean_gate": torch.tensor([0.9]),
        "minimum_gate": torch.tensor([0.9]),
        "action_all_members_finite": torch.tensor([False]),
        "action_confidence": torch.tensor([0.9]),
        "agreement": torch.tensor([1.0]),
        "candidates": torch.tensor([6]),
        "physical_danger_probabilities": torch.full((3, 1, 18), 0.1),
    }

    masks = _offline_override_masks(values, episode, runtime)

    assert masks["used_previous"].tolist() == [True]
    assert masks["correction_actions"].tolist() == [5]
    assert masks["correction_confidence"].tolist() == [1.0]
    assert masks["agreement"].tolist() == [1.0]
    assert masks["action_all_members_finite"].tolist() == [False]
    assert masks["quality"].tolist() == [False]
    assert masks["future_onset_request"].tolist() == [False]
    assert masks["override"].tolist() == [False]


def test_dense_safety_loss_masks_nonfinite_margins_and_has_finite_gradients() -> None:
    collision_logits = torch.zeros((2, 18), requires_grad=True)
    margin_predictions = torch.zeros((2, 18), requires_grad=True)
    collided = torch.zeros((2, 18), dtype=torch.bool)
    collided[0, 3] = True
    collided[1] = True
    margins = torch.full((2, 18), 12.0)
    margins[0, 0] = float("inf")
    margins[0, 1] = float("-inf")
    margins[0, 2] = float("nan")
    finite = torch.isfinite(margins)

    collision_loss, margin_loss = _dense_safety_losses(
        collision_logits,
        margin_predictions,
        collided,
        margins,
        finite,
        collision_positive_weights=torch.full((18,), 8.0),
        all_collision_row_weight=0.25,
    )
    (collision_loss + margin_loss).backward()

    assert torch.isfinite(collision_loss)
    assert torch.isfinite(margin_loss)
    assert torch.isfinite(collision_logits.grad).all()
    assert torch.isfinite(margin_predictions.grad).all()
    assert margin_predictions.grad[0, :3].count_nonzero() == 0


def test_physical_danger_loss_directly_supervises_clearance_class() -> None:
    logits = torch.zeros((2, 18), requires_grad=True)
    safe = torch.ones((2, 18), dtype=torch.bool)
    safe[0, 3] = False
    safe[1, 8] = False

    loss = _physical_danger_loss(
        logits,
        safe,
        positive_weights=torch.full((18,), 8.0),
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0, 3] < 0.0
    assert logits.grad[0, 4] > 0.0


def test_training_normalization_applies_train_statistics_to_validation() -> None:
    adapter = ResidualCorrectionAdapter(ResidualAdapterConfig(
        recurrent_size=4,
        hidden_size=4,
        ensemble_size=2,
    ))
    width = adapter.config.feature_size
    training = SimpleNamespace(features=torch.stack((
        torch.zeros(width),
        torch.full((width,), 2.0),
    )).unsqueeze(0))
    validation = SimpleNamespace(
        features=torch.full((1, 1, width), 5.0),
    )

    _normalize(adapter, [training], [training, validation])

    expected_scale = torch.full((width,), 2.0 ** 0.5)
    assert torch.allclose(adapter.feature_mean, torch.ones(width))
    assert torch.allclose(adapter.feature_scale, expected_scale)
    assert torch.allclose(
        validation.features,
        torch.full((1, 1, width), 4.0 / (2.0 ** 0.5)),
    )


def test_training_recreates_executed_action_and_hold_context_without_leakage() -> None:
    demonstrations = Demonstrations(
        global_frames=np.zeros((4, 1, 6, 2, 2), dtype=np.float32),
        local_frames=np.zeros((4, 1, 6, 2, 2), dtype=np.float32),
        actions=np.asarray([[5], [5], [6], [6]], dtype=np.int64),
        previous_actions=np.asarray([[-1], [5], [5], [6]], dtype=np.int64),
        risks=np.zeros((4, 1), dtype=np.float32),
    )

    context = _executed_action_context(demonstrations)

    assert context.shape == (1, 4, 20)
    assert context[0, 0].count_nonzero() == 0
    assert context[0, 1, 5] == 1.0
    assert context[0, 2, 5] == 1.0
    assert context[0, 3, 6] == 1.0
    assert context[0, :, -2].tolist() == [0.0, 1.0, 1.0, 1.0]
    assert context[0, :, -1].tolist() == pytest.approx([
        0.0,
        math.log1p(1),
        math.log1p(2),
        math.log1p(1),
    ])


def test_training_rejects_inconsistent_recorded_previous_action() -> None:
    demonstrations = Demonstrations(
        global_frames=np.zeros((2, 1, 6, 2, 2), dtype=np.float32),
        local_frames=np.zeros((2, 1, 6, 2, 2), dtype=np.float32),
        actions=np.asarray([[5], [6]], dtype=np.int64),
        previous_actions=np.asarray([[-1], [7]], dtype=np.int64),
        risks=np.zeros((2, 1), dtype=np.float32),
    )

    with pytest.raises(ValueError, match="do not match"):
        _executed_action_context(demonstrations)


def test_offline_metrics_reject_unlabelled_and_unsafe_overrides() -> None:
    safe_actions = torch.zeros((3, 18), dtype=torch.bool)
    safe_actions[0, 1] = True
    safe_actions[1, 1] = True
    safe_actions[2, 2] = True
    episode = SimpleNamespace(
        seed=7,
        decisions=3,
        parent_actions=torch.zeros(3, dtype=torch.int64),
        gate_valid=torch.tensor([False, True, True]),
        gate_targets=torch.tensor([0.0, 0.0, 1.0]),
        hard_positive=torch.tensor([False, False, True]),
        anticipatory=torch.zeros(3, dtype=torch.bool),
        preferred_actions=torch.tensor([-1, -1, 2]),
        safe_actions=safe_actions,
        teacher_selected_collision=torch.tensor([True, False, False]),
    )
    predictions = {7: {
        "mean_gate": torch.full((3,), 0.9),
        "minimum_gate": torch.full((3,), 0.9),
        "action_all_members_finite": torch.ones(3, dtype=torch.bool),
        "action_confidence": torch.full((3,), 0.9),
        "candidates": torch.ones(3, dtype=torch.int64),
        "agreement": torch.ones(3),
    }}

    result = _metrics(
        predictions,
        [episode],
        ResidualRuntimeConfig(
            gate_probability_threshold=0.5,
            minimum_member_gate_probability=0.5,
            action_probability_threshold=0.5,
            ensemble_agreement_threshold=1.0,
        ),
    )["total"]

    assert result["overrides"] == 3
    assert result["false_overrides"] == 2
    assert result["positive_overrides"] == 1
    assert result["safe_positive_overrides"] == 0
    assert result["unsafe_overrides"] == 1
    assert result["preferred_action_overrides"] == 0
    assert result["nonpreferred_overrides"] == 3


def test_offline_metrics_report_preferred_and_anticipatory_coverage() -> None:
    safe_actions = torch.zeros((3, 18), dtype=torch.bool)
    safe_actions[0, 2] = True
    safe_actions[1, 3] = True
    safe_actions[2, 4:6] = True
    episode = SimpleNamespace(
        seed=11,
        decisions=3,
        parent_actions=torch.zeros(3, dtype=torch.int64),
        gate_valid=torch.ones(3, dtype=torch.bool),
        gate_targets=torch.ones(3),
        hard_positive=torch.tensor([True, False, False]),
        anticipatory=torch.tensor([False, True, True]),
        preferred_actions=torch.tensor([2, 3, 4]),
        safe_actions=safe_actions,
        teacher_selected_collision=torch.zeros(3, dtype=torch.bool),
    )
    predictions = {11: {
        "mean_gate": torch.full((3,), 0.9),
        "minimum_gate": torch.full((3,), 0.9),
        "action_all_members_finite": torch.ones(3, dtype=torch.bool),
        "action_confidence": torch.full((3,), 0.9),
        "candidates": torch.tensor([2, 3, 4]),
        "agreement": torch.ones(3),
    }}

    metrics = _metrics(
        predictions,
        [episode],
        ResidualRuntimeConfig(
            gate_probability_threshold=0.5,
            minimum_member_gate_probability=0.5,
            action_probability_threshold=0.5,
            ensemble_agreement_threshold=1.0,
        ),
    )
    result = metrics["total"]

    assert result["anticipatory_overrides"] == 2
    assert result["safe_anticipatory_overrides"] == 2
    assert result["anticipatory_recall"] == pytest.approx(1.0)
    assert result["preferred_action_overrides"] == 3
    assert result["nonpreferred_overrides"] == 0
    assert result["preferred_action_override_rate"] == pytest.approx(1.0)
    assert _offline_deployment_eligible(metrics) is True

    result["false_overrides"] = 1
    assert _offline_deployment_eligible(metrics) is False


def test_future_onset_metrics_report_lead_buckets_and_early_clusters() -> None:
    def episode_and_predictions(
        seed: int,
        leads: list[int],
        override_indices: list[int],
    ) -> tuple[SimpleNamespace, dict[str, torch.Tensor]]:
        lead = torch.tensor(leads, dtype=torch.int64)
        anticipatory = lead > 0
        decisions = len(leads)
        safe_actions = torch.zeros((decisions, 18), dtype=torch.bool)
        safe_actions[:, 5] = True
        mean_gate = torch.full((decisions,), 0.1)
        mean_gate[override_indices] = 0.9
        preferred_actions = torch.full((decisions,), -1, dtype=torch.int64)
        preferred_actions[anticipatory] = 5
        episode = SimpleNamespace(
            seed=seed,
            decisions=decisions,
            parent_actions=torch.zeros(decisions, dtype=torch.int64),
            previous_actions=torch.full((decisions,), -1, dtype=torch.int64),
            gate_valid=anticipatory.clone(),
            gate_targets=anticipatory.to(torch.float32),
            hard_positive=torch.zeros(decisions, dtype=torch.bool),
            anticipatory=anticipatory,
            anticipatory_lead_decisions=lead,
            preferred_actions=preferred_actions,
            safe_actions=safe_actions,
            evaluation_safe_actions=safe_actions.clone(),
            parent_evaluation_danger=torch.zeros(decisions, dtype=torch.bool),
            teacher_selected_collision=torch.zeros(decisions, dtype=torch.bool),
        )
        predictions = {
            "mean_gate": mean_gate,
            "minimum_gate": mean_gate.clone(),
            "action_all_members_finite": torch.ones(
                decisions,
                dtype=torch.bool,
            ),
            "action_confidence": torch.full((decisions,), 0.9),
            "candidates": torch.full((decisions,), 5, dtype=torch.int64),
            "agreement": torch.ones(decisions),
            "physical_danger_probabilities": torch.full(
                (3, decisions, 18),
                0.1,
            ),
        }
        return episode, predictions

    first, first_predictions = episode_and_predictions(
        101,
        [10, 9, 8, 0, 3, 2, 0, 6, 4, 0],
        [1, 2, 4, 8],
    )
    second, second_predictions = episode_and_predictions(
        102,
        [6, 5, 0],
        [0],
    )

    metrics = _metrics(
        {101: first_predictions, 102: second_predictions},
        [first, second],
        _future_onset_runtime(),
    )
    total = metrics["total"]

    assert _early_event_cluster_metrics(
        first.anticipatory & (first.anticipatory_lead_decisions >= 4),
        first_predictions["mean_gate"] >= 0.5,
        first.anticipatory_lead_decisions,
    ) == (2, 2, 13, 9)
    assert metrics["101"]["early_danger_event_clusters"] == 2
    assert metrics["101"]["early_covered_danger_event_clusters"] == 2
    assert total["early_danger_event_clusters"] == 3
    assert total["early_covered_danger_event_clusters"] == 3
    assert total["covered_early_lead_decisions_sum"] == 19
    assert total["covered_early_lead_decisions_maximum"] == 9
    assert total["mean_covered_early_lead_decisions"] == pytest.approx(19 / 3)
    assert total["anticipatory_targets_lead_1_3"] == 2
    assert total["anticipatory_overrides_lead_1_3"] == 1
    assert total["anticipatory_recall_lead_1_3"] == pytest.approx(0.5)
    assert total["anticipatory_targets_lead_4_6"] == 4
    assert total["anticipatory_overrides_lead_4_6"] == 2
    assert total["anticipatory_recall_lead_4_6"] == pytest.approx(0.5)
    assert total["anticipatory_targets_lead_7_10"] == 3
    assert total["anticipatory_overrides_lead_7_10"] == 2
    assert total["anticipatory_recall_lead_7_10"] == pytest.approx(2 / 3)
    assert total["early_beneficial_overrides"] == 4


def _future_diagnostic_episode(
    seed: int,
    *,
    staged_failures: bool,
) -> tuple[SimpleNamespace, dict[str, torch.Tensor]]:
    leads = torch.arange(4, 11, dtype=torch.int64)
    decisions = len(leads)
    evaluation_safe = torch.ones((decisions, 18), dtype=torch.bool)
    mean_gate = torch.full((decisions,), 0.9)
    minimum_gate = torch.full((decisions,), 0.9)
    confidence = torch.full((decisions,), 0.9)
    agreement = torch.ones(decisions)
    candidates = torch.full((decisions,), 5, dtype=torch.int64)
    physical = torch.full((3, decisions, 18), 0.1)
    if staged_failures:
        mean_gate[1] = 0.4
        candidates[2] = 4
        evaluation_safe[3, 5] = False
        physical[:, 4, 5] = float("nan")
        physical[:, 5, 5] = 0.9
        confidence[6] = 0.1
    episode = SimpleNamespace(
        seed=seed,
        parent_actions=torch.zeros(decisions, dtype=torch.int64),
        previous_actions=torch.full((decisions,), -1, dtype=torch.int64),
        anticipatory=torch.ones(decisions, dtype=torch.bool),
        anticipatory_lead_decisions=leads,
        preferred_actions=torch.full((decisions,), 5, dtype=torch.int64),
        evaluation_safe_actions=evaluation_safe,
    )
    return episode, {
        "mean_gate": mean_gate,
        "minimum_gate": minimum_gate,
        "action_all_members_finite": torch.ones(decisions, dtype=torch.bool),
        "action_confidence": confidence,
        "agreement": agreement,
        "candidates": candidates,
        "physical_danger_probabilities": physical,
    }


def test_future_onset_diagnostics_report_each_early_funnel_stage_and_blocker(
    tmp_path,
) -> None:
    training, training_predictions = _future_diagnostic_episode(
        111,
        staged_failures=True,
    )
    calibration, calibration_predictions = _future_diagnostic_episode(
        112,
        staged_failures=False,
    )
    predictions = {
        111: training_predictions,
        112: calibration_predictions,
    }

    split = _future_onset_split_diagnostics(
        predictions,
        [training],
        ensemble_size=3,
    )
    stage = split["stage_counts"]
    assert stage == {
        "early_4_10_targets": 7,
        "early_correction_required_targets": 7,
        "gate_outputs_finite": 7,
        "raw_gate_positive_at_0_5": 6,
        "raw_action_all_members_finite": 7,
        "exact_preferred_candidate_possible": 6,
        "certified_equivalent_candidate_possible": 6,
        "physical_all_members_finite": 5,
        "physical_predicted_safe_at_0_5": 4,
        "prethreshold_intersection_upper_bound": 4,
        "minimum_search_threshold_intersection_upper_bound": 2,
    }
    assert list(split["by_lead_decisions"]) == [
        str(lead) for lead in range(4, 11)
    ]
    assert split["by_lead_decisions"]["4"][
        "minimum_search_threshold_intersection_upper_bound"
    ] == 1
    assert split["by_lead_decisions"]["5"][
        "raw_gate_positive_at_0_5"
    ] == 0
    assert split["by_lead_decisions"]["5"][
        "prethreshold_intersection_upper_bound"
    ] == 1
    assert split["by_lead_decisions"]["6"][
        "exact_preferred_candidate_possible"
    ] == 0
    assert split["by_lead_decisions"]["6"][
        "certified_equivalent_candidate_possible"
    ] == 1
    assert split["by_lead_decisions"]["7"][
        "certified_equivalent_candidate_possible"
    ] == 0
    assert split["by_lead_decisions"]["8"][
        "physical_all_members_finite"
    ] == 0
    assert split["by_lead_decisions"]["9"][
        "physical_predicted_safe_at_0_5"
    ] == 0
    assert split["blocking_reasons"] == {
        "gate_output_nonfinite": 0,
        "mean_gate_below_0_5": 1,
        "minimum_member_gate_below_0_25": 0,
        "raw_action_member_nonfinite": 0,
        "action_summary_nonfinite": 0,
        "action_confidence_below_0_2": 1,
        "action_agreement_below_minimum": 0,
        "candidate_not_certified_equivalent": 1,
        "candidate_physical_nonfinite": 1,
        "candidate_predicted_unsafe": 1,
    }

    diagnostics = _future_onset_calibration_diagnostics(
        predictions,
        [training],
        [calibration],
        ensemble_size=3,
    )
    assert diagnostics["read_only_audit"] is True
    assert diagnostics["schema_version"] == 2
    assert diagnostics["threshold_selection_uses_validation"] is False
    assert diagnostics["early_lead_decisions"] == [4, 10]
    assert diagnostics["splits"]["training"] == split
    assert diagnostics["splits"]["calibration"]["stage_counts"][
        "minimum_search_threshold_intersection_upper_bound"
    ] == 7

    diagnostic_path = tmp_path / "calibration-failure.json"
    _write_json_atomic(diagnostic_path, diagnostics)
    assert json.loads(diagnostic_path.read_text(encoding="utf-8")) == diagnostics


def _future_calibration_episode(
    seed: int,
    *,
    include_early_event: bool,
) -> tuple[SimpleNamespace, dict[str, torch.Tensor]]:
    decisions = 6
    leads = torch.tensor(
        [6, 5, 4, 0, 0, 0] if include_early_event else [0] * decisions,
        dtype=torch.int64,
    )
    anticipatory = leads > 0
    safe_actions = torch.zeros((decisions, 18), dtype=torch.bool)
    safe_actions[:, 5] = True
    preferred_actions = torch.full((decisions,), -1, dtype=torch.int64)
    preferred_actions[anticipatory] = 5
    episode = SimpleNamespace(
        seed=seed,
        decisions=decisions,
        parent_actions=torch.zeros(decisions, dtype=torch.int64),
        previous_actions=torch.full((decisions,), -1, dtype=torch.int64),
        gate_valid=torch.ones(decisions, dtype=torch.bool),
        gate_targets=anticipatory.to(torch.float32),
        hard_positive=torch.zeros(decisions, dtype=torch.bool),
        anticipatory=anticipatory,
        anticipatory_lead_decisions=leads,
        preferred_actions=preferred_actions,
        safe_actions=safe_actions,
        evaluation_safe_actions=safe_actions.clone(),
        parent_evaluation_danger=torch.zeros(decisions, dtype=torch.bool),
        teacher_selected_collision=torch.zeros(decisions, dtype=torch.bool),
    )
    mean_gate = torch.where(
        anticipatory,
        torch.full((decisions,), 0.9),
        torch.full((decisions,), 0.1),
    )
    predictions = {
        "mean_gate": mean_gate,
        "minimum_gate": mean_gate.clone(),
        "action_all_members_finite": torch.ones(decisions, dtype=torch.bool),
        "action_confidence": torch.full((decisions,), 0.9),
        "candidates": torch.full((decisions,), 5, dtype=torch.int64),
        "agreement": torch.ones(decisions),
        "physical_danger_probabilities": torch.full(
            (2, decisions, 18),
            0.05,
        ),
    }
    return episode, predictions


def test_future_onset_calibration_requires_early_train_and_calibration_coverage(
) -> None:
    train, train_predictions = _future_calibration_episode(
        201,
        include_early_event=True,
    )
    calibration, calibration_predictions = _future_calibration_episode(
        202,
        include_early_event=True,
    )
    predictions = {201: train_predictions, 202: calibration_predictions}

    runtime = _calibrate(
        predictions,
        [train],
        [calibration],
        ensemble_size=2,
        per_action_safety_critic=True,
        per_action_physical_danger=True,
        future_onset_gate=True,
    )

    assert runtime.future_onset_gate_enabled is True
    assert runtime.legacy_gate_enabled is False
    assert _metrics(predictions, [train], runtime)["total"][
        "early_beneficial_overrides"
    ] > 0
    assert _metrics(predictions, [calibration], runtime)["total"][
        "early_beneficial_overrides"
    ] > 0

    calibration_without_early, no_early_predictions = _future_calibration_episode(
        203,
        include_early_event=False,
    )
    with pytest.raises(ValueError, match="both training and calibration"):
        _calibrate(
            {201: train_predictions, 203: no_early_predictions},
            [train],
            [calibration_without_early],
            ensemble_size=2,
            per_action_safety_critic=True,
            per_action_physical_danger=True,
            future_onset_gate=True,
        )


def test_future_onset_calibration_accepts_certified_nonexact_action() -> None:
    train, train_predictions = _future_calibration_episode(
        211,
        include_early_event=True,
    )
    calibration, calibration_predictions = _future_calibration_episode(
        212,
        include_early_event=True,
    )
    for episode, predictions in (
        (train, train_predictions),
        (calibration, calibration_predictions),
    ):
        episode.safe_actions[:, 4] = True
        episode.evaluation_safe_actions[:, 4] = True
        predictions["candidates"].fill_(4)

    predictions = {211: train_predictions, 212: calibration_predictions}
    runtime = _calibrate(
        predictions,
        [train],
        [calibration],
        ensemble_size=2,
        per_action_safety_critic=True,
        per_action_physical_danger=True,
        future_onset_gate=True,
    )
    metrics = _metrics(predictions, [calibration], runtime)

    assert metrics["total"]["preferred_action_overrides"] == 0
    assert metrics["total"]["equivalent_action_overrides"] > 0
    assert metrics["total"]["nonpreferred_overrides"] > 0
    assert metrics["total"]["non_equivalent_overrides"] == 0
    assert _offline_deployment_eligible(metrics) is True


def test_preferred_action_loss_ignores_unlabelled_rows() -> None:
    logits = torch.tensor([
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
        [0.0, 0.0, 10.0],
    ])
    preferred = torch.tensor([-1, 1, 2])
    positive = torch.tensor([True, False, True])

    loss = _preferred_action_loss(logits, preferred, positive)

    assert loss == pytest.approx(
        torch.nn.functional.cross_entropy(logits[2:3], preferred[2:3]).item()
    )


def test_preferred_action_set_loss_accepts_any_certified_equivalent() -> None:
    logits = torch.tensor([
        [0.0, 8.0, -4.0],
        [0.0, -4.0, 8.0],
    ])
    accepted = torch.tensor([
        [False, True, True],
        [True, True, False],
    ])

    loss = _preferred_action_set_loss(
        logits,
        accepted,
        torch.ones(2, dtype=torch.bool),
    )

    assert loss.item() > 3.0
    logits[1, 0] = 9.0
    improved = _preferred_action_set_loss(
        logits,
        accepted,
        torch.ones(2, dtype=torch.bool),
    )
    assert improved < loss


def test_preferred_action_membership_bce_balances_each_rows_positive_and_negative(
) -> None:
    logits = torch.zeros((2, 6), requires_grad=True)
    accepted = torch.tensor([
        [True, False, False, False, False, False],
        [True, True, True, False, False, False],
    ])

    loss = _preferred_action_membership_loss(
        logits,
        accepted,
        torch.ones(2, dtype=torch.bool),
    )
    loss.backward()

    assert loss.item() == pytest.approx(math.log(2.0))
    assert logits.grad is not None
    for row, target in zip(logits.grad, accepted, strict=True):
        positive_total = -row[target].sum().item()
        negative_total = row[~target].sum().item()
        assert positive_total == pytest.approx(negative_total)
        assert positive_total > 0.0


def test_balanced_membership_bce_remains_the_exact_default() -> None:
    default_logits = torch.tensor(
        [[2.0, -1.0, 0.5, -3.0], [-2.0, 1.5, 0.25, 3.0]],
        requires_grad=True,
    )
    explicit_logits = default_logits.detach().clone().requires_grad_(True)
    accepted = torch.tensor([
        [True, False, False, False],
        [True, True, True, False],
    ])
    mask = torch.ones(2, dtype=torch.bool)

    default_loss = _preferred_action_membership_loss(
        default_logits,
        accepted,
        mask,
    )
    explicit_loss = _preferred_action_membership_loss(
        explicit_logits,
        accepted,
        mask,
        mode="balanced",
    )
    default_loss.backward()
    explicit_loss.backward()

    assert torch.equal(default_loss, explicit_loss)
    assert default_logits.grad is not None
    assert explicit_logits.grad is not None
    assert torch.equal(default_logits.grad, explicit_logits.grad)


def test_unweighted_membership_bce_means_cells_then_labelled_rows() -> None:
    logits = torch.tensor([
        [2.0, -1.0, 0.5, -3.0],
        [-2.0, 1.5, 0.25, 3.0],
        [4.0, -4.0, 2.0, -2.0],
    ], requires_grad=True)
    reference_logits = logits.detach().clone().requires_grad_(True)
    accepted = torch.tensor([
        [True, False, False, False],
        [True, True, True, True],
        [False, False, False, False],
    ])
    mask = torch.ones(3, dtype=torch.bool)

    loss = _preferred_action_membership_loss(
        logits,
        accepted,
        mask,
        mode="unweighted",
    )
    reference_terms = torch.nn.functional.binary_cross_entropy_with_logits(
        reference_logits,
        accepted.to(reference_logits.dtype),
        reduction="none",
    )
    expected = reference_terms.mean(dim=-1)[:2].mean()
    loss.backward()
    expected.backward()

    assert torch.equal(loss, expected)
    assert logits.grad is not None
    assert reference_logits.grad is not None
    assert torch.equal(logits.grad, reference_logits.grad)
    assert logits.grad[2].count_nonzero().item() == 0


def test_membership_loss_mode_and_metadata_are_fail_closed_and_exact() -> None:
    values = {
        "action_logit_mode": "absolute",
        "action_loss_weight": 2.0,
        "parent_copy_weight": 0.05,
        "preferred_action_uniform_loss_weight": 0.0,
        "preferred_action_tiebreak_loss_weight": 0.0,
        "preferred_action_rank_loss_weight": 0.0,
        "safety_candidate_loss_weight": 2.0,
        "membership_loss_mode": "unweighted",
    }
    with pytest.raises(ValueError, match="only applicable"):
        _validate_action_training_semantics(**values)

    values.update({
        "action_logit_mode": "certified_membership",
        "action_loss_weight": 0.0,
        "parent_copy_weight": 0.0,
        "safety_candidate_loss_weight": 0.0,
        "membership_loss_mode": "unknown",
    })
    with pytest.raises(ValueError, match="membership loss mode"):
        _validate_action_training_semantics(**values)

    assert _membership_loss_metadata(
        action_logit_mode="certified_membership",
        membership_loss_mode="unweighted",
    ) == {
        "membership_loss_mode": "unweighted",
        "preferred_action_loss_semantics": (
            "unweighted independent membership BCE, mean over action cells "
            "within each labelled row then mean over labelled rows"
        ),
    }
    assert _membership_loss_metadata(
        action_logit_mode="parent_residual_joint",
        membership_loss_mode="balanced",
    ) == {
        "membership_loss_mode": None,
        "preferred_action_loss_semantics": (
            "negative log softmax mass on the certified action set"
        ),
    }


def test_restored_membership_weights_require_their_recorded_loss_mode() -> None:
    direct = {"training_controls": {"membership_loss_mode": "unweighted"}}
    assert _validate_restored_membership_loss_mode(
        direct,
        action_logit_mode="certified_membership",
        requested_mode="unweighted",
    ) == "unweighted"

    nested = {
        "training_controls": {"recalibration_only": True},
        "frozen_adapter_weight_source": {"training_metadata": direct},
    }
    assert _validate_restored_membership_loss_mode(
        nested,
        action_logit_mode="certified_membership",
        requested_mode="unweighted",
    ) == "unweighted"

    with pytest.raises(ValueError, match="must match the restored adapter"):
        _validate_restored_membership_loss_mode(
            direct,
            action_logit_mode="certified_membership",
            requested_mode="balanced",
        )
    with pytest.raises(ValueError, match="invalid membership loss mode"):
        _validate_restored_membership_loss_mode(
            {"training_controls": {"membership_loss_mode": "unknown"}},
            action_logit_mode="certified_membership",
            requested_mode="balanced",
        )


def test_restored_membership_mode_traverses_consistent_nested_provenance() -> None:
    leaf = {"training_controls": {"membership_loss_mode": "unweighted"}}
    middle = {
        "training_controls": {"membership_loss_mode": "unweighted"},
        "fit_checkpoint_weight_source": {"training_metadata": leaf},
    }
    outer = {
        "training_controls": {"membership_loss_mode": "unweighted"},
        "frozen_adapter_weight_source": {"training_metadata": middle},
    }

    assert _validate_restored_membership_loss_mode(
        outer,
        action_logit_mode="certified_membership",
        requested_mode="unweighted",
    ) == "unweighted"


@pytest.mark.parametrize(
    ("outer_mode", "inner_mode"),
    (("unweighted", "balanced"), ("balanced", "unweighted")),
)
def test_restored_membership_mode_rejects_outer_inner_conflicts(
    outer_mode: str,
    inner_mode: str,
) -> None:
    metadata = {
        "training_controls": {"membership_loss_mode": outer_mode},
        "frozen_adapter_weight_source": {
            "training_metadata": {
                "training_controls": {"membership_loss_mode": inner_mode},
            },
        },
    }

    with pytest.raises(ValueError, match="inconsistent membership loss mode"):
        _validate_restored_membership_loss_mode(
            metadata,
            action_logit_mode="certified_membership",
            requested_mode=outer_mode,
        )


def test_restored_membership_mode_rejects_invalid_inner_provenance() -> None:
    metadata = {
        "training_controls": {"membership_loss_mode": "balanced"},
        "fit_checkpoint_weight_source": {
            "training_metadata": {
                "training_controls": {"membership_loss_mode": "invalid"},
            },
        },
    }

    with pytest.raises(ValueError, match="invalid membership loss mode"):
        _validate_restored_membership_loss_mode(
            metadata,
            action_logit_mode="certified_membership",
            requested_mode="balanced",
        )


@pytest.mark.parametrize(
    "metadata",
    (
        {"training_controls": None},
        {"frozen_adapter_weight_source": None},
        {
            "fit_checkpoint_weight_source": {
                "training_metadata": None,
            },
        },
    ),
)
def test_restored_membership_mode_rejects_malformed_provenance(
    metadata: dict[str, object],
) -> None:
    with pytest.raises(TypeError, match="must be a mapping|invalid training metadata"):
        _validate_restored_membership_loss_mode(
            metadata,
            action_logit_mode="certified_membership",
            requested_mode="balanced",
        )


def test_legacy_restored_membership_weights_are_explicitly_balanced() -> None:
    legacy = {"training_controls": {"epochs": 6}}

    assert _validate_restored_membership_loss_mode(
        legacy,
        action_logit_mode="certified_membership",
        requested_mode="balanced",
    ) == "balanced"
    with pytest.raises(ValueError, match="restored legacy adapter"):
        _validate_restored_membership_loss_mode(
            legacy,
            action_logit_mode="certified_membership",
            requested_mode="unweighted",
        )


def test_train_member_accepts_unweighted_membership_mode_before_fitting() -> None:
    adapter = _residual_position_adapter("certified_membership")

    history = _train_member(
        adapter,
        0,
        [],
        seed=7,
        epochs=1,
        learning_rate=3e-4,
        weight_decay=1e-4,
        chunk_length=128,
        gate_positive_weight=4.0,
        action_loss_weight=0.0,
        parent_copy_weight=0.0,
        device="cpu",
        membership_loss_mode="unweighted",
        preferred_action_uniform_loss_weight=0.0,
        preferred_action_tiebreak_loss_weight=0.0,
        preferred_action_rank_loss_weight=0.0,
        safety_candidate_loss_weight=0.0,
    )

    assert history == [{"epoch": 1.0, "mean_chunk_loss": 0.0}]


def test_membership_loss_cli_exposes_only_supported_modes(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr("sys.argv", ["train_temporal_residual_adapter.py", "--help"])

    with pytest.raises(SystemExit, match="0"):
        residual_training.main()

    help_text = capsys.readouterr().out
    assert MEMBERSHIP_LOSS_MODES == ("balanced", "unweighted")
    assert "--membership-loss-mode {balanced,unweighted}" in help_text


def test_membership_bce_breaks_the_set_softmax_translation_invariance() -> None:
    logits = torch.tensor([[2.0, 0.0, -1.0, -2.0]])
    accepted = torch.tensor([[True, True, False, False]])
    mask = torch.ones(1, dtype=torch.bool)

    set_loss = _preferred_action_set_loss(logits, accepted, mask)
    shifted_set_loss = _preferred_action_set_loss(logits + 3.0, accepted, mask)
    membership_loss = _preferred_action_membership_loss(logits, accepted, mask)
    shifted_membership_loss = _preferred_action_membership_loss(
        logits + 3.0,
        accepted,
        mask,
    )

    assert torch.equal(set_loss, shifted_set_loss)
    assert membership_loss != shifted_membership_loss


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("action_loss_weight", 0.1),
        ("parent_copy_weight", 0.1),
        ("preferred_action_uniform_loss_weight", 0.1),
        ("preferred_action_tiebreak_loss_weight", 0.1),
        ("preferred_action_rank_loss_weight", 0.1),
        ("safety_candidate_loss_weight", 0.1),
    ),
)
def test_membership_training_rejects_inapplicable_softmax_objectives(
    field: str,
    value: float,
) -> None:
    values = {
        "action_logit_mode": "certified_membership",
        "action_loss_weight": 0.0,
        "parent_copy_weight": 0.0,
        "preferred_action_uniform_loss_weight": 0.0,
        "preferred_action_tiebreak_loss_weight": 0.0,
        "preferred_action_rank_loss_weight": 0.0,
        "safety_candidate_loss_weight": 0.0,
    }
    values[field] = value

    with pytest.raises(ValueError, match="zero inapplicable loss weights"):
        _validate_action_training_semantics(**values)


@pytest.mark.parametrize("bad_value", [-0.1, float("nan"), float("inf")])
def test_training_loss_weights_must_be_finite_and_nonnegative(
    bad_value: float,
) -> None:
    with pytest.raises(ValueError, match="finite and nonnegative"):
        _validate_training_loss_weights(action=bad_value, gate_positive=1.0)


def test_membership_parent_copy_is_strictly_zero_and_inapplicable() -> None:
    logits = torch.randn((2, 18), requires_grad=True)
    parent = torch.randn((2, 18))
    loss = _parent_copy_loss(
        logits,
        parent,
        torch.ones(2, dtype=torch.bool),
        action_logit_mode="certified_membership",
    )

    loss.backward()

    assert loss.item() == 0.0
    assert logits.grad is not None
    assert logits.grad.count_nonzero().item() == 0


def test_temporal_labels_propagate_a_future_safe_move_to_predecessors() -> None:
    samples = 4
    field_count = len(TEACHER_ACTION_EVALUATION_FIELDS)
    evaluations = torch.zeros((samples, 18, field_count))
    evaluations[..., TEACHER_ACTION_COLLIDED_INDEX] = 1.0
    evaluations[..., TEACHER_ACTION_MINIMUM_MARGIN_INDEX] = -1.0
    regrets = torch.full((samples, 18), 10.0)
    for index in range(samples - 1):
        evaluations[index, 13, TEACHER_ACTION_COLLIDED_INDEX] = 0.0
        evaluations[index, 13, TEACHER_ACTION_MINIMUM_MARGIN_INDEX] = 20.0
        evaluations[index, 13, TEACHER_ACTION_SELECTED_INDEX] = 1.0
        regrets[index, 13] = 0.0
        evaluations[index, 14, TEACHER_ACTION_COLLIDED_INDEX] = 0.0
        evaluations[index, 14, TEACHER_ACTION_MINIMUM_MARGIN_INDEX] = 18.0
        regrets[index, 14] = 2.0
    evaluations[-1, 14, TEACHER_ACTION_COLLIDED_INDEX] = 0.0
    evaluations[-1, 14, TEACHER_ACTION_MINIMUM_MARGIN_INDEX] = 20.0
    evaluations[-1, 14, TEACHER_ACTION_SELECTED_INDEX] = 1.0
    regrets[-1, 14] = 0.0
    demonstrations = Demonstrations(
        global_frames=torch.zeros((samples, 1, 6, 2, 2)).numpy(),
        local_frames=torch.zeros((samples, 1, 6, 2, 2)).numpy(),
        actions=torch.zeros((samples, 1), dtype=torch.int64).numpy(),
        previous_actions=torch.tensor([[-1], [0], [0], [0]]).numpy(),
        risks=torch.zeros((samples, 1)).numpy(),
        teacher_action_evaluations=evaluations.unsqueeze(1).numpy(),
        teacher_action_regrets=regrets.unsqueeze(1).numpy(),
        teacher_action_evaluation_mask=torch.ones(
            (samples, 1), dtype=torch.bool,
        ).numpy(),
    )
    parent_logits = torch.zeros((1, samples, 18))
    parent_logits[..., 13] = 1.0

    labels = _labels_from_evidence(
        demonstrations,
        parent_logits,
        safe_regret=1.0,
        minimum_parent_margin=8.0,
        minimum_margin_gain=1.0,
        predecessor_decisions=3,
    )

    assert labels["hard_positive"].tolist() == [False, False, False, True]
    assert labels["anticipatory"].tolist() == [True, True, True, False]
    assert labels["preferred_actions"].tolist() == [14, 14, 14, 14]
    assert labels["safe_actions"][:, 14].tolist() == [True, True, True, True]
    assert labels["safe_actions"][:3, 13].tolist() == [False, False, False]


def _future_onset_label_inputs(
    *,
    unsafe_predecessor: int | None = None,
    onset_parent_collision: bool = True,
) -> tuple[Demonstrations, torch.Tensor]:
    samples = 16
    onset = 12
    field_count = len(TEACHER_ACTION_EVALUATION_FIELDS)
    evaluations = torch.zeros((samples, 18, field_count))
    evaluations[..., TEACHER_ACTION_COLLIDED_INDEX] = 1.0
    evaluations[..., TEACHER_ACTION_MINIMUM_MARGIN_INDEX] = -1.0
    regrets = torch.full((samples, 18), 10.0)
    for index in range(samples):
        evaluations[index, 14, TEACHER_ACTION_COLLIDED_INDEX] = 0.0
        evaluations[index, 14, TEACHER_ACTION_MINIMUM_MARGIN_INDEX] = 20.0
        evaluations[index, 14, TEACHER_ACTION_SELECTED_INDEX] = 1.0
        regrets[index, 14] = 0.0
        evaluations[index, 13, TEACHER_ACTION_COLLIDED_INDEX] = 0.0
        evaluations[index, 13, TEACHER_ACTION_MINIMUM_MARGIN_INDEX] = 20.0
        regrets[index, 13] = 0.0
    evaluations[onset, 13, TEACHER_ACTION_COLLIDED_INDEX] = float(
        onset_parent_collision
    )
    evaluations[onset, 13, TEACHER_ACTION_MINIMUM_MARGIN_INDEX] = (
        0.0 if onset_parent_collision else 7.0
    )
    if unsafe_predecessor is not None:
        evaluations[
            unsafe_predecessor,
            14,
            TEACHER_ACTION_MINIMUM_MARGIN_INDEX,
        ] = 7.0
    demonstrations = Demonstrations(
        global_frames=np.zeros((samples, 1, 6, 2, 2), dtype=np.float32),
        local_frames=np.zeros((samples, 1, 6, 2, 2), dtype=np.float32),
        actions=np.full((samples, 1), 13, dtype=np.int64),
        previous_actions=np.asarray(
            [[-1]] + [[13]] * (samples - 1),
            dtype=np.int64,
        ),
        risks=np.zeros((samples, 1), dtype=np.float32),
        teacher_action_evaluations=evaluations.unsqueeze(1).numpy(),
        teacher_action_regrets=regrets.unsqueeze(1).numpy(),
        teacher_action_evaluation_mask=np.ones((samples, 1), dtype=bool),
    )
    parent_logits = torch.zeros((1, samples, 18))
    parent_logits[..., 13] = 1.0
    return demonstrations, parent_logits


def test_future_onset_labels_use_ten_binary_decisions_and_tail_censoring() -> None:
    demonstrations, parent_logits = _future_onset_label_inputs()

    labels = _labels_from_evidence(
        demonstrations,
        parent_logits,
        safe_regret=1.0,
        minimum_parent_margin=8.0,
        minimum_margin_gain=1.0,
        predecessor_decisions=1,
        future_onset_gate=True,
    )

    assert labels["hard_positive"].nonzero().flatten().tolist() == [12]
    assert labels["anticipatory"].nonzero().flatten().tolist() == list(
        range(2, 12)
    )
    assert labels["anticipatory_lead_decisions"][2:12].tolist() == list(
        range(10, 0, -1)
    )
    assert labels["gate_targets"][2:13].tolist() == [1.0] * 11
    assert set(labels["gate_targets"].tolist()) == {0.0, 1.0}
    assert labels["gate_valid"][:13].tolist() == [True] * 13
    assert labels["gate_valid"][13:].tolist() == [False, False, False]
    assert torch.equal(labels["future_onset_valid"], labels["gate_valid"])


def test_future_onset_labels_preserve_multiple_certified_equivalents() -> None:
    demonstrations, parent_logits = _future_onset_label_inputs()
    assert demonstrations.teacher_action_evaluations is not None
    assert demonstrations.teacher_action_regrets is not None
    demonstrations.teacher_action_evaluations[
        :, 0, 15, TEACHER_ACTION_COLLIDED_INDEX
    ] = 0.0
    demonstrations.teacher_action_evaluations[
        :, 0, 15, TEACHER_ACTION_MINIMUM_MARGIN_INDEX
    ] = 18.0
    demonstrations.teacher_action_regrets[:, 0, 15] = 1.0

    labels = _labels_from_evidence(
        demonstrations,
        parent_logits,
        safe_regret=1.0,
        minimum_parent_margin=8.0,
        minimum_margin_gain=1.0,
        predecessor_decisions=10,
        future_onset_gate=True,
    )

    for index in range(2, 13):
        assert bool(labels["preferred_correction_required"][index])
        assert labels["preferred_equivalent_actions"][index].nonzero().flatten().tolist() == [
            14,
            15,
        ]
        assert torch.equal(
            labels["preferred_action_set"][index],
            labels["preferred_equivalent_actions"][index],
        )


def test_future_onset_label_requires_current_physical_candidate_clearance() -> None:
    demonstrations, parent_logits = _future_onset_label_inputs(
        unsafe_predecessor=5,
    )

    labels = _labels_from_evidence(
        demonstrations,
        parent_logits,
        safe_regret=1.0,
        minimum_parent_margin=8.0,
        minimum_margin_gain=1.0,
        predecessor_decisions=10,
        future_onset_gate=True,
    )

    assert bool(labels["safe_actions"][5, 14])
    assert not bool(labels["evaluation_safe_actions"][5, 14])
    assert labels["anticipatory"].nonzero().flatten().tolist() == list(
        range(6, 12)
    )


def test_future_onset_falls_back_from_low_margin_previous_to_safe_teacher() -> None:
    demonstrations, parent_logits = _future_onset_label_inputs(
        onset_parent_collision=False,
    )

    labels = _labels_from_evidence(
        demonstrations,
        parent_logits,
        safe_regret=1.0,
        minimum_parent_margin=8.0,
        minimum_margin_gain=1.0,
        predecessor_decisions=3,
        future_onset_gate=True,
    )

    assert labels["hard_positive"].nonzero().flatten().tolist() == [12]
    assert labels["preferred_actions"][12].item() == 14
    assert labels["anticipatory"].nonzero().flatten().tolist() == list(
        range(2, 12)
    )


def test_temporal_labels_stop_when_preferred_move_becomes_unsafe() -> None:
    samples = 5
    field_count = len(TEACHER_ACTION_EVALUATION_FIELDS)
    evaluations = torch.zeros((samples, 18, field_count))
    evaluations[..., TEACHER_ACTION_COLLIDED_INDEX] = 1.0
    evaluations[..., TEACHER_ACTION_MINIMUM_MARGIN_INDEX] = -1.0
    regrets = torch.full((samples, 18), 10.0)
    for index in range(samples - 1):
        evaluations[index, 12, TEACHER_ACTION_COLLIDED_INDEX] = 0.0
        evaluations[index, 12, TEACHER_ACTION_MINIMUM_MARGIN_INDEX] = 20.0
        evaluations[index, 12, TEACHER_ACTION_SELECTED_INDEX] = 1.0
        regrets[index, 12] = 0.0
    evaluations[-1, 14, TEACHER_ACTION_COLLIDED_INDEX] = 0.0
    evaluations[-1, 14, TEACHER_ACTION_MINIMUM_MARGIN_INDEX] = 20.0
    evaluations[-1, 14, TEACHER_ACTION_SELECTED_INDEX] = 1.0
    regrets[-1, 14] = 0.0
    evaluations[-1, 15, TEACHER_ACTION_COLLIDED_INDEX] = 0.0
    evaluations[-1, 15, TEACHER_ACTION_MINIMUM_MARGIN_INDEX] = 19.0
    regrets[-1, 15] = 1.0
    evaluations[-2, 14, TEACHER_ACTION_COLLIDED_INDEX] = 0.0
    evaluations[-2, 15, TEACHER_ACTION_COLLIDED_INDEX] = 0.0
    demonstrations = Demonstrations(
        global_frames=torch.zeros((samples, 1, 6, 2, 2)).numpy(),
        local_frames=torch.zeros((samples, 1, 6, 2, 2)).numpy(),
        actions=torch.zeros((samples, 1), dtype=torch.int64).numpy(),
        previous_actions=torch.tensor([[-1], [0], [0], [0], [0]]).numpy(),
        risks=torch.zeros((samples, 1)).numpy(),
        teacher_action_evaluations=evaluations.unsqueeze(1).numpy(),
        teacher_action_regrets=regrets.unsqueeze(1).numpy(),
        teacher_action_evaluation_mask=torch.ones(
            (samples, 1), dtype=torch.bool,
        ).numpy(),
    )
    parent_logits = torch.zeros((1, samples, 18))
    parent_logits[..., 12] = 1.0

    labels = _labels_from_evidence(
        demonstrations,
        parent_logits,
        safe_regret=1.0,
        minimum_parent_margin=8.0,
        minimum_margin_gain=1.0,
        predecessor_decisions=4,
    )

    assert labels["hard_positive"].tolist() == [False, False, False, False, True]
    assert labels["anticipatory"].tolist() == [False, False, False, True, False]
    assert labels["preferred_actions"].tolist() == [-1, -1, -1, 14, 14]


def test_preferred_action_keeps_previous_execution_only_while_currently_safe() -> None:
    samples = 2
    field_count = len(TEACHER_ACTION_EVALUATION_FIELDS)
    evaluations = torch.zeros((samples, 18, field_count))
    evaluations[..., TEACHER_ACTION_COLLIDED_INDEX] = 1.0
    evaluations[..., TEACHER_ACTION_MINIMUM_MARGIN_INDEX] = -1.0
    regrets = torch.full((samples, 18), 10.0)
    evaluations[0, 12, TEACHER_ACTION_COLLIDED_INDEX] = 0.0
    evaluations[0, 12, TEACHER_ACTION_MINIMUM_MARGIN_INDEX] = 20.0
    evaluations[0, 12, TEACHER_ACTION_SELECTED_INDEX] = 1.0
    regrets[0, 12] = 0.0
    for action in (12, 14):
        evaluations[1, action, TEACHER_ACTION_COLLIDED_INDEX] = 0.0
        evaluations[1, action, TEACHER_ACTION_MINIMUM_MARGIN_INDEX] = 20.0
        regrets[1, action] = 0.0
    evaluations[1, 14, TEACHER_ACTION_SELECTED_INDEX] = 1.0
    demonstrations = Demonstrations(
        global_frames=np.zeros((samples, 1, 6, 2, 2), dtype=np.float32),
        local_frames=np.zeros((samples, 1, 6, 2, 2), dtype=np.float32),
        actions=np.asarray([[12], [14]], dtype=np.int64),
        previous_actions=np.asarray([[-1], [12]], dtype=np.int64),
        risks=np.zeros((samples, 1), dtype=np.float32),
        teacher_action_evaluations=evaluations.unsqueeze(1).numpy(),
        teacher_action_regrets=regrets.unsqueeze(1).numpy(),
        teacher_action_evaluation_mask=np.ones((samples, 1), dtype=bool),
    )
    parent_logits = torch.zeros((1, samples, 18))
    parent_logits[..., 0] = 1.0

    labels = _labels_from_evidence(
        demonstrations,
        parent_logits,
        safe_regret=1.0,
        minimum_parent_margin=8.0,
        minimum_margin_gain=1.0,
        predecessor_decisions=0,
    )

    assert labels["hard_positive"].tolist() == [True, True]
    assert labels["preferred_actions"].tolist() == [12, 12]
    assert labels["safety_candidate_valid"].tolist() == [True, True]
    assert labels["safety_candidate_actions"].tolist() == [12, 12]


def _strict_source_probe(
    tmp_path,
    *,
    report_seed: int = 7,
    manifest_seed: int = 7,
):
    dataset = tmp_path / "episode.npz"
    manifest = tmp_path / "episode.manifest.json"
    dataset.write_bytes(b"strict-source-probe")
    digest = file_sha256(dataset)
    manifest.write_text(
        json.dumps({
            "dataset_sha256": digest,
            "accepted_episodes": [{
                "strict_success": True,
                "seed": manifest_seed,
            }],
        }),
        encoding="utf-8",
    )
    report = {
        "success": True,
        "termination_reason": "attack_complete",
        "continuous_fire": True,
        "shoot_command_rate": 1.0,
        "outcome_evidence": {"final_player": {"death": 0}},
        "config": {
            "record_teacher_evaluations": True,
            "supervision_mode": "corrective",
            "spell_forced_off": True,
            "decision_interval": 3,
            "observation_delay": 5,
            "vision": {
                "global_width": 48,
                "global_height": 56,
                "local_width": 40,
                "local_height": 40,
                "local_extent_x": 72.0,
                "local_extent_y": 72.0,
                "history": 1,
                "observation_delay": 5,
                "channels": 6,
            },
        },
        "controller": {"student": {
            "action_selection": "joint",
            "action_selection_uses_safety_state": False,
        }},
        "demonstrations": {"dataset_sha256": digest},
        "seed": report_seed,
    }
    return report, dataset, manifest


def test_strict_source_accepts_required_dagger_protocol(tmp_path) -> None:
    report, dataset, manifest = _strict_source_probe(tmp_path)

    assert _strict_success(report, dataset, manifest) == 7


@pytest.mark.parametrize(
    ("path", "invalid_value"),
    (
        (("config", "decision_interval"), 3.0),
        (("config", "decision_interval"), True),
        (("config", "observation_delay"), 4),
        (("config", "vision", "global_width"), 48.0),
        (("config", "vision", "global_height"), 55),
        (("config", "vision", "local_width"), 40.0),
        (("config", "vision", "local_height"), False),
        (("config", "vision", "local_extent_x"), True),
        (("config", "vision", "local_extent_x"), math.inf),
        (("config", "vision", "local_extent_y"), math.nan),
        (("config", "vision", "local_extent_y"), 71.0),
        (("config", "vision", "history"), 1.0),
        (("config", "vision", "observation_delay"), 4),
        (("config", "vision", "channels"), 6.0),
        (("controller", "student", "action_selection"), "factorized"),
        (("controller", "student", "action_selection_uses_safety_state"), True),
    ),
)
def test_strict_source_rejects_incompatible_dagger_protocol(
    tmp_path,
    path,
    invalid_value,
) -> None:
    report, dataset, manifest = _strict_source_probe(tmp_path)
    target = report
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = invalid_value

    with pytest.raises(ValueError, match="protocol field"):
        _strict_success(report, dataset, manifest)


def test_strict_source_rejects_boolean_death_evidence(tmp_path) -> None:
    report, dataset, manifest = _strict_source_probe(tmp_path)
    report["outcome_evidence"]["final_player"]["death"] = False

    with pytest.raises(ValueError, match="death evidence"):
        _strict_success(report, dataset, manifest)


def test_strict_source_rejects_manifest_report_seed_mismatch(tmp_path) -> None:
    report, dataset, manifest = _strict_source_probe(tmp_path, manifest_seed=8)

    with pytest.raises(ValueError, match="seed does not match"):
        _strict_success(report, dataset, manifest)


def test_residual_wrapper_preserves_parent_logits_exactly_without_gate() -> None:
    wrapper = ResidualPolicyWrapper(
        StubParent(),
        _constant_adapter(gate_bias=-20.0, correction_action=5),
        ResidualRuntimeConfig(
            gate_probability_threshold=0.9,
            minimum_member_gate_probability=0.9,
            action_probability_threshold=0.9,
        ),
    )
    parent_logits = torch.tensor([[[0.0, 0.5, 2.0, -1.0] + [0.0] * 14]])
    recurrent = torch.zeros((1, 1, 4))

    effective, hidden = wrapper._apply_residual(parent_logits, recurrent)

    assert torch.equal(effective, parent_logits)
    assert len(hidden) == 2
    assert wrapper.residual_runtime_stats()["overrides"] == 0


def test_residual_wrapper_overrides_only_the_consensus_action() -> None:
    wrapper = ResidualPolicyWrapper(
        StubParent(),
        _constant_adapter(gate_bias=20.0, correction_action=5),
        ResidualRuntimeConfig(
            gate_probability_threshold=0.9,
            minimum_member_gate_probability=0.9,
            action_probability_threshold=0.9,
            ensemble_agreement_threshold=1.0,
            override_logit_margin=0.25,
        ),
    )
    parent_logits = torch.tensor([[[0.0, 0.5, 2.0, -1.0] + [0.0] * 14]])
    recurrent = torch.zeros((1, 1, 4))

    effective, hidden = wrapper._apply_residual(parent_logits, recurrent)

    assert int(effective.argmax(dim=-1).item()) == 5
    assert effective[0, 0, 5].item() == pytest.approx(2.25)
    unchanged = torch.ones(18, dtype=torch.bool)
    unchanged[5] = False
    assert torch.equal(effective[0, 0, unchanged], parent_logits[0, 0, unchanged])
    stats = wrapper.residual_runtime_stats()
    assert stats["decisions"] == 1
    assert stats["overrides"] == 1
    assert stats["override_action_counts"] == {"5": 1}
    assert len(hidden) == 2


def test_residual_wrapper_vetoes_one_nonfinite_member_action_logit() -> None:
    adapter = _constant_adapter(
        gate_bias=20.0,
        correction_action=5,
        ensemble_size=3,
    )
    with torch.no_grad():
        adapter.members[0].action_head.bias[17] = float("-inf")
    wrapper = ResidualPolicyWrapper(
        StubParent(),
        adapter,
        ResidualRuntimeConfig(
            gate_probability_threshold=0.9,
            minimum_member_gate_probability=0.9,
            action_probability_threshold=0.5,
            ensemble_agreement_threshold=2.0 / 3.0,
        ),
    )
    parent_logits = torch.tensor([[[0.0, 0.5, 2.0, -1.0] + [0.0] * 14]])

    effective, hidden = wrapper._apply_residual(
        parent_logits,
        torch.zeros((1, 1, 4)),
    )

    assert torch.isfinite(effective).all()
    assert torch.equal(effective, parent_logits)
    assert wrapper.residual_runtime_stats()["overrides"] == 0
    assert len(hidden) == 3


@pytest.mark.parametrize(
    "bad_value",
    [float("nan"), float("inf"), float("-inf")],
)
def test_residual_wrapper_vetoes_entire_nonfinite_action_member(
    bad_value: float,
) -> None:
    adapter = _constant_adapter(
        gate_bias=20.0,
        correction_action=5,
        ensemble_size=3,
    )
    with torch.no_grad():
        adapter.members[0].action_head.bias.fill_(bad_value)
    wrapper = ResidualPolicyWrapper(
        StubParent(),
        adapter,
        ResidualRuntimeConfig(
            gate_probability_threshold=0.9,
            minimum_member_gate_probability=0.9,
            action_probability_threshold=0.5,
            ensemble_agreement_threshold=2.0 / 3.0,
        ),
    )
    parent_logits = torch.tensor([[[0.0, 0.5, 2.0, -1.0] + [0.0] * 14]])

    effective, hidden = wrapper._apply_residual(
        parent_logits,
        torch.zeros((1, 1, 4)),
    )

    assert torch.isfinite(effective).all()
    assert torch.equal(effective, parent_logits)
    assert wrapper.residual_runtime_stats()["overrides"] == 0
    assert len(hidden) == 3


def test_action_conditioned_residual_uses_actual_execution_and_hold_count() -> None:
    adapter = ResidualCorrectionAdapter(ResidualAdapterConfig(
        recurrent_size=4,
        hidden_size=4,
        ensemble_size=2,
        executed_action_context=True,
    ))
    wrapper = ResidualPolicyWrapper(
        StubParent(),
        adapter,
        ResidualRuntimeConfig(),
    )
    reference = torch.zeros((1, 1, 18))

    initial = wrapper._executed_action_context(reference)
    assert initial is not None
    assert initial.count_nonzero() == 0

    action = SimpleNamespace(discrete=7)
    wrapper.commit_executed_action(action, frames=3)
    first = wrapper._executed_action_context(reference)
    wrapper.commit_executed_action(action, frames=3)
    second = wrapper._executed_action_context(reference)

    assert first is not None and second is not None
    assert first[0, 0, 7] == 1.0
    assert first[0, 0, -2] == 1.0
    assert first[0, 0, -1] == pytest.approx(math.log1p(1))
    assert second[0, 0, -1] == pytest.approx(math.log1p(2))
    wrapper.reset_runtime_state()
    reset = wrapper._executed_action_context(reference)
    assert reset is not None
    assert reset.count_nonzero() == 0


def test_action_conditioned_residual_requires_context_but_legacy_does_not() -> None:
    recurrent = torch.zeros((1, 1, 4))
    logits = torch.zeros((1, 1, 18))
    conditioned = ResidualCorrectionAdapter(ResidualAdapterConfig(
        recurrent_size=4,
        hidden_size=4,
        ensemble_size=2,
        executed_action_context=True,
    ))
    legacy = ResidualCorrectionAdapter(ResidualAdapterConfig(
        recurrent_size=4,
        hidden_size=4,
        ensemble_size=2,
    ))

    with pytest.raises(ValueError, match="context is required"):
        conditioned.raw_features(recurrent, logits)
    assert legacy.raw_features(recurrent, logits).shape[-1] == 40
    assert conditioned.raw_features(
        recurrent,
        logits,
        torch.zeros((1, 1, 20)),
    ).shape[-1] == 60


def test_safety_heads_preserve_forward_contract_and_chunk_hidden_carry() -> None:
    torch.manual_seed(9)
    adapter = ResidualCorrectionAdapter(ResidualAdapterConfig(
        recurrent_size=4,
        hidden_size=6,
        ensemble_size=3,
        per_action_safety_critic=True,
    )).eval()
    recurrent = torch.randn((2, 7, 4))
    logits = torch.randn((2, 7, 18))

    gates, actions, hidden = adapter(recurrent, logits)
    safe_gates, safe_actions, collision, margins, safe_hidden = (
        adapter.forward_with_safety(recurrent, logits)
    )

    assert torch.equal(gates, safe_gates)
    assert torch.equal(actions, safe_actions)
    assert collision is not None and collision.shape == (3, 2, 7, 18)
    assert margins is not None and margins.shape == (3, 2, 7, 18)
    assert all(torch.equal(left, right) for left, right in zip(
        hidden,
        safe_hidden,
        strict=True,
    ))

    first = adapter.forward_with_safety(recurrent[:, :3], logits[:, :3])
    second = adapter.forward_with_safety(
        recurrent[:, 3:],
        logits[:, 3:],
        first[-1],
    )
    for full, left, right in zip(
        (safe_gates, safe_actions, collision, margins),
        first[:4],
        second[:4],
        strict=True,
    ):
        assert full is not None and left is not None and right is not None
        assert torch.allclose(full, torch.cat((left, right), dim=2), atol=1e-6)
    assert all(torch.allclose(left, right, atol=1e-6) for left, right in zip(
        safe_hidden,
        second[-1],
        strict=True,
    ))


def test_separate_action_recurrent_requires_residual_mode_and_defaults_off() -> None:
    with pytest.raises(ValueError, match="requires parent-logit residuals"):
        ResidualAdapterConfig(
            recurrent_size=4,
            separate_action_recurrent=True,
        )

    legacy = ResidualCorrectionAdapter(ResidualAdapterConfig(
        recurrent_size=4,
        hidden_size=5,
        ensemble_size=2,
    ))
    _gates, _actions, hidden = legacy(
        torch.zeros((1, 2, 4)),
        torch.zeros((1, 2, 18)),
    )

    assert legacy.config.separate_action_recurrent is False
    assert all(member.action_recurrent is None for member in legacy.members)
    assert all(value.shape == (1, 1, 5) for value in hidden)


def test_separate_action_recurrent_carries_two_hidden_layers_across_chunks() -> None:
    torch.manual_seed(71)
    adapter = ResidualCorrectionAdapter(ResidualAdapterConfig(
        recurrent_size=4,
        hidden_size=6,
        ensemble_size=3,
        action_logit_mode="parent_residual_joint",
        separate_action_recurrent=True,
    )).eval()
    with torch.no_grad():
        for member in adapter.members:
            member.action_head.weight.normal_(std=0.15)
    recurrent = torch.randn((2, 7, 4))
    parent_logits = torch.randn((2, 7, 18))

    full_gates, full_actions, full_hidden = adapter(recurrent, parent_logits)
    first = adapter(recurrent[:, :3], parent_logits[:, :3])
    second = adapter(
        recurrent[:, 3:],
        parent_logits[:, 3:],
        first[-1],
    )

    assert all(value.shape == (2, 2, 6) for value in full_hidden)
    assert torch.allclose(
        full_gates,
        torch.cat((first[0], second[0]), dim=2),
        atol=1e-6,
    )
    assert torch.allclose(
        full_actions,
        torch.cat((first[1], second[1]), dim=2),
        atol=1e-6,
    )
    assert all(torch.allclose(left, right, atol=1e-6) for left, right in zip(
        full_hidden,
        second[-1],
        strict=True,
    ))


def test_separate_action_loss_does_not_backpropagate_into_shared_trunk() -> None:
    torch.manual_seed(77)
    adapter = ResidualCorrectionAdapter(ResidualAdapterConfig(
        recurrent_size=4,
        hidden_size=6,
        ensemble_size=1,
        per_action_safety_critic=True,
        visual_latent_size=16,
        per_action_physical_danger=True,
        action_logit_mode="parent_residual_joint",
        separate_action_recurrent=True,
    ))
    member = adapter.members[0]
    assert member.action_input_projection is not None
    assert member.action_recurrent is not None
    assert member.collision_head is not None
    assert member.minimum_margin_head is not None
    assert member.physical_danger_head is not None
    with torch.no_grad():
        member.action_head.weight.normal_(std=0.15)

    outputs = adapter.forward_with_all_safety(
        torch.randn((1, 5, 4)),
        torch.randn((1, 5, 18)),
        visual_features=torch.randn((1, 5, 16)),
    )
    preferred = torch.tensor([[14, 14, 5, 5, 14]])
    loss = torch.nn.functional.cross_entropy(
        outputs[1][0].reshape(-1, 18),
        preferred.reshape(-1),
    )
    loss.backward()

    shared_modules = (
        member.input_projection,
        member.recurrent,
        member.gate_head,
        member.collision_head,
        member.minimum_margin_head,
        member.physical_danger_head,
    )
    assert all(
        parameter.grad is None
        for module in shared_modules
        for parameter in module.parameters()
    )
    action_modules = (
        member.action_input_projection,
        member.action_recurrent,
        member.action_head,
    )
    assert all(
        parameter.grad is not None
        for module in action_modules
        for parameter in module.parameters()
    )
    assert all(
        torch.isfinite(parameter.grad).all()
        and bool(parameter.grad.abs().sum() > 0.0)
        for module in action_modules
        for parameter in module.parameters()
        if parameter.grad is not None
    )


def test_separate_action_gradient_clipping_uses_two_complete_groups() -> None:
    adapter = ResidualCorrectionAdapter(ResidualAdapterConfig(
        recurrent_size=4,
        hidden_size=6,
        ensemble_size=1,
        per_action_safety_critic=True,
        visual_latent_size=16,
        per_action_physical_danger=True,
        action_logit_mode="parent_residual_joint",
        separate_action_recurrent=True,
    ))
    member = adapter.members[0]
    groups = _gradient_clip_parameter_groups(member)
    assert tuple(groups) == ("shared_safety", "action")
    expected_action = {
        id(parameter)
        for module in (
            member.action_input_projection,
            member.action_recurrent,
            member.action_head,
        )
        for parameter in module.parameters()
        if parameter.requires_grad
    }
    grouped = [parameter for values in groups.values() for parameter in values]
    assert {id(parameter) for parameter in groups["action"]} == expected_action
    assert len({id(parameter) for parameter in grouped}) == len(grouped)
    assert {id(parameter) for parameter in grouped} == {
        id(parameter) for parameter in member.parameters() if parameter.requires_grad
    }

    for parameter in grouped:
        parameter.grad = torch.ones_like(parameter) * 10.0
    pre_clip_norms = _clip_member_gradients(member, max_norm=5.0)
    assert set(pre_clip_norms) == {"shared_safety", "action"}
    assert all(float(norm) > 5.0 for norm in pre_clip_norms.values())
    for parameters in groups.values():
        norm = torch.linalg.vector_norm(torch.cat([
            parameter.grad.flatten() for parameter in parameters
        ]))
        assert norm.item() == pytest.approx(5.0, rel=1e-5)


def test_nonseparate_action_gradient_clipping_preserves_global_semantics() -> None:
    adapter = ResidualCorrectionAdapter(ResidualAdapterConfig(
        recurrent_size=4,
        hidden_size=6,
        ensemble_size=1,
    ))
    member = adapter.members[0]
    groups = _gradient_clip_parameter_groups(member)
    assert tuple(groups) == ("global",)
    for parameter in groups["global"]:
        parameter.grad = torch.ones_like(parameter) * 10.0
    norms = _clip_member_gradients(member, max_norm=5.0)
    norm = torch.linalg.vector_norm(torch.cat([
        parameter.grad.flatten() for parameter in groups["global"]
    ]))
    assert set(norms) == {"global"}
    assert float(norms["global"]) > 5.0
    assert norm.item() == pytest.approx(5.0, rel=1e-5)


def test_separate_action_gradient_groups_fail_on_missing_or_overlap() -> None:
    config = ResidualAdapterConfig(
        recurrent_size=4,
        hidden_size=6,
        ensemble_size=1,
        action_logit_mode="parent_residual_joint",
        separate_action_recurrent=True,
    )
    missing = ResidualCorrectionAdapter(config).members[0]
    missing.action_input_projection = None
    with pytest.raises(RuntimeError, match="missing module action_input_projection"):
        _gradient_clip_parameter_groups(missing)

    overlapping = ResidualCorrectionAdapter(config).members[0]
    overlapping.action_head = overlapping.gate_head
    with pytest.raises(RuntimeError, match="gradient groups overlap"):
        _gradient_clip_parameter_groups(overlapping)


def _sample_fit_source_inventory() -> list[dict[str, object]]:
    return [
        {
            "seed": 701,
            "role": "training",
            "dataset": "train.npz",
            "dataset_sha256": "1" * 64,
            "report": "train.json",
            "report_sha256": "2" * 64,
            "manifest": "train.manifest.json",
            "manifest_sha256": "3" * 64,
        },
        {
            "seed": 702,
            "role": "calibration",
            "dataset": "calibration.npz",
            "dataset_sha256": "4" * 64,
            "report": "calibration.json",
            "report_sha256": "5" * 64,
            "manifest": "calibration.manifest.json",
            "manifest_sha256": "6" * 64,
        },
        {
            "seed": 703,
            "role": "validation",
            "dataset": "validation.npz",
            "dataset_sha256": "7" * 64,
            "report": "validation.json",
            "report_sha256": "8" * 64,
            "manifest": "validation.manifest.json",
            "manifest_sha256": "9" * 64,
        },
    ]


def _fit_label_metadata() -> dict[str, object]:
    return {
        "future_onset_gate": True,
        "future_onset_horizon_decisions": 10,
        "safe_regret": 1.0,
        "minimum_parent_margin": 8.0,
        "minimum_margin_gain": 1.0,
    }


def test_fit_checkpoint_round_trip_is_exact_and_not_a_deployment_artifact(
    tmp_path,
    monkeypatch,
) -> None:
    parent = StubParent()
    adapter = _visual_physical_adapter()
    feature_count = adapter.feature_mean.numel()
    feature_mean = torch.linspace(-2.0, 2.0, feature_count)
    feature_scale = torch.linspace(0.25, 2.25, feature_count)
    adapter.set_feature_normalization(feature_mean, feature_scale)
    parent_checkpoint = tmp_path / "parent.pt"
    fit_checkpoint = tmp_path / "adapter-fit.pt"
    torch.save({"identity": "parent"}, parent_checkpoint)
    training_metadata = {
        "member_histories": [[{"epoch": 1.0, "mean_chunk_loss": 0.25}]],
        "source_inventory": _sample_fit_source_inventory(),
        "label_metadata": _fit_label_metadata(),
    }
    expected_state = {
        name: value.detach().cpu().clone()
        for name, value in adapter.state_dict().items()
    }

    descriptor = _save_fit_checkpoint(
        adapter,
        fit_checkpoint,
        parent_checkpoint=parent_checkpoint,
        parent_policy_config=asdict(parent.config),
        training_metadata=training_metadata,
    )

    def fail_normalization(*_args, **_kwargs) -> None:
        raise AssertionError("resume must not recompute feature normalization")

    monkeypatch.setattr(residual_training, "_normalize", fail_normalization)
    restored, metadata = _load_fit_checkpoint(
        fit_checkpoint,
        parent_checkpoint=parent_checkpoint,
        parent_policy_config=asdict(parent.config),
        expected_adapter_config=adapter.config,
    )

    assert descriptor["kind"] == FIT_CHECKPOINT_KIND
    assert descriptor["deployment_artifact"] is False
    assert metadata["version"] == FIT_CHECKPOINT_VERSION
    assert metadata["kind"] == FIT_CHECKPOINT_KIND
    assert metadata["deployment_artifact"] is False
    assert metadata["deployment_eligible"] is False
    assert metadata["calibration_complete"] is False
    assert metadata["acceptance_claim"] is False
    assert metadata["parent_checkpoint_sha256"] == file_sha256(parent_checkpoint)
    assert metadata["parent_policy_config"] == asdict(parent.config)
    assert metadata["adapter_config"] == asdict(adapter.config)
    assert metadata["training_metadata"] == training_metadata
    assert torch.equal(restored.feature_mean.cpu(), feature_mean)
    assert torch.equal(restored.feature_scale.cpu(), feature_scale)
    assert all(
        torch.equal(value, restored.state_dict()[name].cpu())
        for name, value in expected_state.items()
    )
    with pytest.raises(ValueError, match="unsupported residual adapter artifact"):
        load_residual_adapter(
            StubParent(),
            fit_checkpoint,
            parent_checkpoint=parent_checkpoint,
        )


@pytest.mark.parametrize(
    "mode",
    ("parent_residual_joint", "parent_residual_factorized"),
)
def test_fit_checkpoint_v2_round_trip_preserves_new_input_semantics(
    tmp_path,
    mode: str,
) -> None:
    parent = StubParent()
    adapter = _residual_position_adapter(mode)
    parent_checkpoint = tmp_path / "parent.pt"
    fit_checkpoint = tmp_path / f"adapter-{mode}-fit.pt"
    torch.save({"identity": "parent"}, parent_checkpoint)

    _save_fit_checkpoint(
        adapter,
        fit_checkpoint,
        parent_checkpoint=parent_checkpoint,
        parent_policy_config=asdict(parent.config),
        training_metadata={"input_contract": "semantic-player-position"},
    )
    restored, metadata = _load_fit_checkpoint(
        fit_checkpoint,
        parent_checkpoint=parent_checkpoint,
        parent_policy_config=asdict(parent.config),
        expected_adapter_config=adapter.config,
    )

    assert metadata["version"] == 2
    assert metadata["adapter_config"]["action_logit_mode"] == mode
    assert metadata["adapter_config"]["semantic_player_position"] is True
    assert restored.config == adapter.config
    assert all(
        torch.equal(value, restored.state_dict()[name])
        for name, value in adapter.state_dict().items()
    )


def test_legacy_fit_checkpoint_without_new_fields_loads_absolute_unconditioned(
    tmp_path,
) -> None:
    parent = StubParent()
    adapter = _visual_physical_adapter()
    parent_checkpoint = tmp_path / "parent.pt"
    fit_checkpoint = tmp_path / "legacy-fit.pt"
    torch.save({"identity": "parent"}, parent_checkpoint)
    _save_fit_checkpoint(
        adapter,
        fit_checkpoint,
        parent_checkpoint=parent_checkpoint,
        parent_policy_config=asdict(parent.config),
        training_metadata={},
    )
    payload = torch.load(fit_checkpoint, map_location="cpu", weights_only=False)
    payload["version"] = 1
    payload["adapter_config"].pop("action_logit_mode")
    payload["adapter_config"].pop("semantic_player_position")
    torch.save(payload, fit_checkpoint)

    restored, metadata = _load_fit_checkpoint(
        fit_checkpoint,
        parent_checkpoint=parent_checkpoint,
        parent_policy_config=asdict(parent.config),
        expected_adapter_config=adapter.config,
    )

    assert metadata["version"] == 1
    assert metadata["adapter_config"]["action_logit_mode"] == "absolute"
    assert metadata["adapter_config"]["semantic_player_position"] is False
    assert restored.config.action_logit_mode == "absolute"
    assert restored.config.semantic_player_position is False


def test_legacy_fit_checkpoint_cannot_disguise_parent_residual_logits(
    tmp_path,
) -> None:
    parent = StubParent()
    adapter = _visual_physical_adapter()
    parent_checkpoint = tmp_path / "parent.pt"
    fit_checkpoint = tmp_path / "disguised-fit.pt"
    torch.save({"identity": "parent"}, parent_checkpoint)
    _save_fit_checkpoint(
        adapter,
        fit_checkpoint,
        parent_checkpoint=parent_checkpoint,
        parent_policy_config=asdict(parent.config),
        training_metadata={},
    )
    payload = torch.load(fit_checkpoint, map_location="cpu", weights_only=False)
    payload["version"] = 1
    payload["adapter_config"]["action_logit_mode"] = "parent_residual_joint"
    torch.save(payload, fit_checkpoint)

    with pytest.raises(ValueError, match="cannot contain residual logits"):
        _load_fit_checkpoint(
            fit_checkpoint,
            parent_checkpoint=parent_checkpoint,
            parent_policy_config=asdict(parent.config),
            expected_adapter_config=adapter.config,
        )


def test_fit_checkpoint_rejects_parent_and_adapter_config_mismatches(
    tmp_path,
) -> None:
    parent = StubParent()
    adapter = _visual_physical_adapter()
    parent_checkpoint = tmp_path / "parent.pt"
    wrong_parent_checkpoint = tmp_path / "wrong-parent.pt"
    fit_checkpoint = tmp_path / "adapter-fit.pt"
    torch.save({"identity": "parent"}, parent_checkpoint)
    torch.save({"identity": "different parent"}, wrong_parent_checkpoint)
    _save_fit_checkpoint(
        adapter,
        fit_checkpoint,
        parent_checkpoint=parent_checkpoint,
        parent_policy_config=asdict(parent.config),
        training_metadata={
            "source_inventory": _sample_fit_source_inventory(),
            "label_metadata": _fit_label_metadata(),
        },
    )

    with pytest.raises(ValueError, match="parent hash does not match"):
        _load_fit_checkpoint(
            fit_checkpoint,
            parent_checkpoint=wrong_parent_checkpoint,
            parent_policy_config=asdict(parent.config),
            expected_adapter_config=adapter.config,
        )
    wrong_policy_config = asdict(parent.config)
    wrong_policy_config["recurrent_size"] += 1
    with pytest.raises(ValueError, match="parent policy config does not match"):
        _load_fit_checkpoint(
            fit_checkpoint,
            parent_checkpoint=parent_checkpoint,
            parent_policy_config=wrong_policy_config,
            expected_adapter_config=adapter.config,
        )
    wrong_adapter_config = asdict(adapter.config)
    wrong_adapter_config["hidden_size"] += 1
    with pytest.raises(ValueError, match="adapter config does not match"):
        _load_fit_checkpoint(
            fit_checkpoint,
            parent_checkpoint=parent_checkpoint,
            parent_policy_config=asdict(parent.config),
            expected_adapter_config=wrong_adapter_config,
        )


@pytest.mark.parametrize("corruption", ["nonpositive", "state_mismatch"])
def test_fit_checkpoint_rejects_invalid_or_inconsistent_normalization(
    tmp_path,
    corruption: str,
) -> None:
    parent = StubParent()
    adapter = _visual_physical_adapter()
    parent_checkpoint = tmp_path / "parent.pt"
    fit_checkpoint = tmp_path / "adapter-fit.pt"
    torch.save({"identity": "parent"}, parent_checkpoint)
    _save_fit_checkpoint(
        adapter,
        fit_checkpoint,
        parent_checkpoint=parent_checkpoint,
        parent_policy_config=asdict(parent.config),
        training_metadata={},
    )
    payload = torch.load(fit_checkpoint, map_location="cpu", weights_only=False)
    if corruption == "nonpositive":
        payload["feature_normalization"]["scale"][0] = 0.0
    else:
        payload["feature_normalization"]["mean"][0] += 1.0
    torch.save(payload, fit_checkpoint)

    message = "positive" if corruption == "nonpositive" else "does not match"
    with pytest.raises(ValueError, match=message):
        _load_fit_checkpoint(
            fit_checkpoint,
            parent_checkpoint=parent_checkpoint,
            parent_policy_config=asdict(parent.config),
            expected_adapter_config=adapter.config,
        )


def test_fit_source_inventory_hashes_all_three_sources_and_records_split(
    tmp_path,
) -> None:
    episodes = []
    for seed, role in ((701, "training"), (702, "calibration"), (703, "validation")):
        dataset = tmp_path / f"{seed}.npz"
        report = tmp_path / f"{seed}.json"
        manifest = tmp_path / f"{seed}.manifest.json"
        dataset.write_bytes(f"dataset-{seed}".encode("ascii"))
        report.write_text(f"report-{seed}", encoding="ascii")
        manifest.write_text(f"manifest-{seed}", encoding="ascii")
        episodes.append(SimpleNamespace(
            seed=seed,
            dataset=str(dataset),
            report=str(report),
            manifest=str(manifest),
            expected_role=role,
        ))

    inventory = _fit_source_inventory(
        episodes,
        calibration_seeds={702},
        validation_seeds={703},
    )

    for episode, source in zip(episodes, inventory, strict=True):
        assert source["seed"] == episode.seed
        assert source["role"] == episode.expected_role
        assert source["dataset_sha256"] == file_sha256(episode.dataset)
        assert source["report_sha256"] == file_sha256(episode.report)
        assert source["manifest_sha256"] == file_sha256(episode.manifest)


def test_fit_resume_metadata_binds_source_hashes_roles_and_label_contract() -> None:
    inventory = _sample_fit_source_inventory()
    labels = _fit_label_metadata()
    metadata = {
        "training_metadata": {
            "source_inventory": inventory,
            "label_metadata": labels,
        },
    }
    _validate_fit_resume_metadata(
        metadata,
        source_inventory=inventory,
        label_metadata=labels,
    )

    changed_dataset = [dict(source) for source in inventory]
    changed_dataset[0]["dataset_sha256"] = "a" * 64
    with pytest.raises(ValueError, match="source inventory"):
        _validate_fit_resume_metadata(
            metadata,
            source_inventory=changed_dataset,
            label_metadata=labels,
        )

    swapped_roles = [dict(source) for source in inventory]
    swapped_roles[1]["role"], swapped_roles[2]["role"] = (
        swapped_roles[2]["role"],
        swapped_roles[1]["role"],
    )
    with pytest.raises(ValueError, match="split roles"):
        _validate_fit_resume_metadata(
            metadata,
            source_inventory=swapped_roles,
            label_metadata=labels,
        )

    changed_labels = dict(labels)
    changed_labels["future_onset_horizon_decisions"] = 9
    with pytest.raises(ValueError, match="label configuration"):
        _validate_fit_resume_metadata(
            metadata,
            source_inventory=inventory,
            label_metadata=changed_labels,
        )


def test_fit_workflow_paths_keep_inputs_outputs_and_sources_distinct(
    tmp_path,
) -> None:
    parent = tmp_path / "parent.pt"
    resume = tmp_path / "resume-fit.pt"
    source = tmp_path / "source.npz"
    fit_output = tmp_path / "new-fit.pt"
    deployment = tmp_path / "deployment.pt"
    report = tmp_path / "report.json"
    diagnostics = tmp_path / "diagnostics.json"
    _validate_distinct_workflow_paths(
        outputs={
            "fit checkpoint output": fit_output,
            "deployment artifact": deployment,
            "report": report,
            "calibration diagnostics": diagnostics,
        },
        protected_inputs={
            "parent checkpoint": parent,
            "resume fit checkpoint": resume,
        },
        source_paths=[source],
    )

    with pytest.raises(ValueError, match="must use different paths"):
        _validate_distinct_workflow_paths(
            outputs={
                "fit checkpoint output": fit_output,
                "deployment artifact": fit_output.parent / "." / fit_output.name,
            },
            protected_inputs={"parent checkpoint": parent},
            source_paths=[source],
        )
    with pytest.raises(ValueError, match="cannot overwrite resume fit checkpoint"):
        _validate_distinct_workflow_paths(
            outputs={"fit checkpoint output": resume},
            protected_inputs={"resume fit checkpoint": resume},
            source_paths=[source],
        )
    with pytest.raises(ValueError, match="cannot overwrite a DAgger source"):
        _validate_distinct_workflow_paths(
            outputs={"fit checkpoint output": source},
            protected_inputs={"parent checkpoint": parent},
            source_paths=[source],
        )
    with pytest.raises(ValueError, match="cannot also be a DAgger source"):
        _validate_distinct_workflow_paths(
            outputs={"deployment artifact": deployment},
            protected_inputs={"resume fit checkpoint": source},
            source_paths=[source],
        )


def test_residual_artifact_requires_exact_parent_hash(tmp_path) -> None:
    parent = StubParent()
    adapter = _constant_adapter(gate_bias=-20.0, correction_action=5)
    parent_checkpoint = tmp_path / "parent.pt"
    wrong_checkpoint = tmp_path / "wrong.pt"
    artifact = tmp_path / "adapter.pt"
    torch.save({"identity": "parent"}, parent_checkpoint)
    torch.save({"identity": "wrong"}, wrong_checkpoint)
    save_residual_adapter(
        adapter,
        artifact,
        parent_checkpoint=parent_checkpoint,
        parent_policy_config=asdict(parent.config),
        runtime_config=ResidualRuntimeConfig(),
        training_metadata={"strict_success_sources_only": True},
    )

    wrapper, metadata = load_residual_adapter(
        parent,
        artifact,
        parent_checkpoint=parent_checkpoint,
    )

    assert isinstance(wrapper, ResidualPolicyWrapper)
    assert metadata["version"] == 1
    assert metadata["training_metadata"]["strict_success_sources_only"] is True
    with pytest.raises(ValueError, match="hash does not match"):
        load_residual_adapter(
            StubParent(),
            artifact,
            parent_checkpoint=wrong_checkpoint,
        )


def test_version_two_action_context_artifact_round_trip(tmp_path) -> None:
    parent = StubParent()
    adapter = ResidualCorrectionAdapter(ResidualAdapterConfig(
        recurrent_size=4,
        hidden_size=4,
        ensemble_size=2,
        executed_action_context=True,
    ))
    parent_checkpoint = tmp_path / "parent.pt"
    artifact = tmp_path / "adapter-v2.pt"
    torch.save({"identity": "parent"}, parent_checkpoint)

    save_residual_adapter(
        adapter,
        artifact,
        parent_checkpoint=parent_checkpoint,
        parent_policy_config=asdict(parent.config),
        runtime_config=ResidualRuntimeConfig(),
        training_metadata={"source_version": 2},
    )
    wrapper, metadata = load_residual_adapter(
        parent,
        artifact,
        parent_checkpoint=parent_checkpoint,
    )

    assert metadata["version"] == 2
    assert wrapper.adapter.config.executed_action_context is True
    assert wrapper.runtime_config.future_onset_gate_enabled is False
    assert all(
        torch.equal(value, wrapper.adapter.state_dict()[name])
        for name, value in adapter.state_dict().items()
    )


def test_version_three_safety_critic_artifact_round_trip(tmp_path) -> None:
    parent = StubParent()
    adapter = ResidualCorrectionAdapter(ResidualAdapterConfig(
        recurrent_size=4,
        hidden_size=4,
        ensemble_size=2,
        per_action_safety_critic=True,
    ))
    parent_checkpoint = tmp_path / "parent.pt"
    artifact = tmp_path / "adapter-v3.pt"
    torch.save({"identity": "parent"}, parent_checkpoint)
    runtime = ResidualRuntimeConfig(
        critic_enabled=True,
        parent_collision_probability_threshold=0.8,
        candidate_collision_probability_threshold=0.2,
    )

    save_residual_adapter(
        adapter,
        artifact,
        parent_checkpoint=parent_checkpoint,
        parent_policy_config=asdict(parent.config),
        runtime_config=runtime,
        training_metadata={"strict_success_sources_only": True},
    )
    wrapper, metadata = load_residual_adapter(
        parent,
        artifact,
        parent_checkpoint=parent_checkpoint,
    )

    assert metadata["version"] == 3
    assert wrapper.adapter.config.per_action_safety_critic is True
    assert wrapper.runtime_config == runtime
    assert all(torch.equal(value, wrapper.adapter.state_dict()[name]) for name, value in (
        adapter.state_dict().items()
    ))


def test_version_four_visual_physical_critic_round_trip(tmp_path) -> None:
    parent = StubParent()
    adapter = ResidualCorrectionAdapter(ResidualAdapterConfig(
        recurrent_size=4,
        hidden_size=4,
        ensemble_size=2,
        executed_action_context=True,
        per_action_safety_critic=True,
        visual_latent_size=16,
        per_action_physical_danger=True,
    ))
    recurrent = torch.zeros((1, 2, 4))
    logits = torch.zeros((1, 2, 18))
    context = torch.zeros((1, 2, 20))
    visual = torch.zeros((1, 2, 16))
    with pytest.raises(ValueError, match="visual features are required"):
        adapter.raw_features(recurrent, logits, context)
    assert adapter.raw_features(
        recurrent,
        logits,
        context,
        visual,
    ).shape == (1, 2, adapter.config.feature_size)

    parent_checkpoint = tmp_path / "parent.pt"
    artifact = tmp_path / "adapter-v4.pt"
    torch.save({"identity": "parent"}, parent_checkpoint)
    runtime = ResidualRuntimeConfig(
        critic_enabled=True,
        critic_signal="physical_danger",
        prefer_safe_previous_action=True,
        legacy_gate_enabled=False,
        parent_physical_danger_probability_threshold=0.7,
        candidate_physical_danger_probability_threshold=0.2,
    )
    save_residual_adapter(
        adapter,
        artifact,
        parent_checkpoint=parent_checkpoint,
        parent_policy_config=asdict(parent.config),
        runtime_config=runtime,
        training_metadata={"strict_success_sources_only": True},
    )
    wrapper, metadata = load_residual_adapter(
        parent,
        artifact,
        parent_checkpoint=parent_checkpoint,
    )

    assert metadata["version"] == 4
    assert wrapper.adapter.config.visual_latent_size == 16
    assert wrapper.adapter.config.per_action_physical_danger is True
    assert wrapper.runtime_config == runtime


def test_version_five_future_onset_artifact_round_trip(tmp_path) -> None:
    parent = StubParent()
    adapter = _visual_physical_adapter()
    parent_checkpoint = tmp_path / "parent.pt"
    artifact = tmp_path / "adapter-v5.pt"
    torch.save({"identity": "parent"}, parent_checkpoint)
    runtime = _future_onset_runtime()

    save_residual_adapter(
        adapter,
        artifact,
        parent_checkpoint=parent_checkpoint,
        parent_policy_config=asdict(parent.config),
        runtime_config=runtime,
        training_metadata={
            "future_onset_horizon_decisions": 10,
            "tail_censoring": True,
        },
    )
    wrapper, metadata = load_residual_adapter(
        parent,
        artifact,
        parent_checkpoint=parent_checkpoint,
    )

    assert metadata["version"] == 5
    assert metadata["runtime_config"]["future_onset_gate_enabled"] is True
    assert metadata["training_metadata"]["future_onset_horizon_decisions"] == 10
    assert wrapper.runtime_config == runtime
    assert wrapper.adapter.config.per_action_physical_danger is True
    assert all(
        torch.equal(value, wrapper.adapter.state_dict()[name])
        for name, value in adapter.state_dict().items()
    )


@pytest.mark.parametrize(
    "mode",
    ("parent_residual_joint", "parent_residual_factorized"),
)
def test_version_six_parent_residual_artifact_round_trip_and_tamper_rejection(
    tmp_path,
    mode: str,
) -> None:
    parent = StubParent()
    adapter = _residual_position_adapter(mode)
    parent_checkpoint = tmp_path / "parent.pt"
    artifact = tmp_path / f"adapter-v6-{mode}.pt"
    torch.save({"identity": "parent"}, parent_checkpoint)
    runtime = _future_onset_runtime()

    save_residual_adapter(
        adapter,
        artifact,
        parent_checkpoint=parent_checkpoint,
        parent_policy_config=asdict(parent.config),
        runtime_config=runtime,
        training_metadata={"action_training": "parent-logit-residual"},
    )
    wrapper, metadata = load_residual_adapter(
        parent,
        artifact,
        parent_checkpoint=parent_checkpoint,
    )

    assert metadata["version"] == 6
    assert metadata["action_logit_semantics"] == {
        "name": "learned_parent_logit_residual",
        "version": 1,
        "mode": mode,
        "zero_delta": "parent_logits",
    }
    assert wrapper.adapter.config.action_logit_mode == mode
    assert wrapper.adapter.config.semantic_player_position is True
    assert wrapper.runtime_config == runtime
    assert all(
        torch.equal(value, wrapper.adapter.state_dict()[name])
        for name, value in adapter.state_dict().items()
    )

    payload = torch.load(artifact, map_location="cpu", weights_only=False)
    payload["action_logit_semantics"]["zero_delta"] = "tampered"
    torch.save(payload, artifact)
    with pytest.raises(ValueError, match="action semantics do not match"):
        load_residual_adapter(
            StubParent(),
            artifact,
            parent_checkpoint=parent_checkpoint,
        )


def test_version_six_membership_artifact_records_exact_probability_semantics(
    tmp_path,
) -> None:
    parent = StubParent()
    adapter = _residual_position_adapter("certified_membership")
    parent_checkpoint = tmp_path / "parent.pt"
    artifact = tmp_path / "adapter-v6-membership.pt"
    torch.save({"identity": "parent"}, parent_checkpoint)
    runtime = _future_onset_runtime()
    save_residual_adapter(
        adapter,
        artifact,
        parent_checkpoint=parent_checkpoint,
        parent_policy_config=asdict(parent.config),
        runtime_config=runtime,
        training_metadata={"action_training": "balanced-membership-bce"},
    )

    wrapper, metadata = load_residual_adapter(
        parent,
        artifact,
        parent_checkpoint=parent_checkpoint,
    )

    expected_semantics = {
        "name": "independent_certified_action_membership",
        "version": 1,
        "mode": "certified_membership",
        "decode": "raw_membership_logits",
        "probability": "finite_per_action_sigmoid",
        "ensemble_candidate": "argmax_mean_membership_probability",
        "parent_logits_added_during_decode": False,
        "parent_context_remains_model_input": True,
    }
    assert metadata["version"] == 6
    assert metadata["action_logit_semantics"] == expected_semantics
    assert wrapper.adapter.config.action_logit_mode == "certified_membership"
    assert all(
        torch.equal(value, wrapper.adapter.state_dict()[name])
        for name, value in adapter.state_dict().items()
    )

    original = torch.load(artifact, map_location="cpu", weights_only=False)
    missing = dict(original)
    missing.pop("action_logit_semantics")
    torch.save(missing, artifact)
    with pytest.raises(ValueError, match="action semantics do not match"):
        load_residual_adapter(
            StubParent(),
            artifact,
            parent_checkpoint=parent_checkpoint,
        )

    tampered = dict(original)
    tampered["action_logit_semantics"] = dict(
        original["action_logit_semantics"],
    )
    tampered["action_logit_semantics"]["probability"] = "softmax"
    torch.save(tampered, artifact)
    with pytest.raises(ValueError, match="action semantics do not match"):
        load_residual_adapter(
            StubParent(),
            artifact,
            parent_checkpoint=parent_checkpoint,
        )


def test_version_four_mapping_without_future_onset_field_still_loads(
    tmp_path,
) -> None:
    parent = StubParent()
    adapter = _visual_physical_adapter()
    parent_checkpoint = tmp_path / "parent.pt"
    artifact = tmp_path / "adapter-old-v4.pt"
    torch.save({"identity": "parent"}, parent_checkpoint)
    runtime = ResidualRuntimeConfig(
        critic_enabled=True,
        critic_signal="physical_danger",
        prefer_safe_previous_action=True,
        legacy_gate_enabled=False,
        parent_physical_danger_probability_threshold=0.7,
        candidate_physical_danger_probability_threshold=0.2,
    )
    save_residual_adapter(
        adapter,
        artifact,
        parent_checkpoint=parent_checkpoint,
        parent_policy_config=asdict(parent.config),
        runtime_config=runtime,
        training_metadata={"source_version": "pre-v5"},
    )
    payload = torch.load(artifact, map_location="cpu", weights_only=False)
    assert payload["version"] == 4
    payload["runtime_config"].pop("future_onset_gate_enabled")
    torch.save(payload, artifact)

    wrapper, metadata = load_residual_adapter(
        parent,
        artifact,
        parent_checkpoint=parent_checkpoint,
    )

    assert metadata["version"] == 4
    assert "future_onset_gate_enabled" not in metadata["runtime_config"]
    assert wrapper.runtime_config.future_onset_gate_enabled is False
    assert wrapper.runtime_config == runtime


def test_artifact_save_rejects_enabled_critic_without_heads(tmp_path) -> None:
    parent_checkpoint = tmp_path / "parent.pt"
    torch.save({"identity": "parent"}, parent_checkpoint)

    with pytest.raises(ValueError, match="requires per-action safety heads"):
        save_residual_adapter(
            _constant_adapter(gate_bias=0.0, correction_action=5),
            tmp_path / "invalid.pt",
            parent_checkpoint=parent_checkpoint,
            parent_policy_config=asdict(StubParent().config),
            runtime_config=ResidualRuntimeConfig(critic_enabled=True),
            training_metadata={},
        )
