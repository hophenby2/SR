import numpy as np
import pytest
import torch

from stg_lab.policy import (
    HumanVisionPolicy,
    PlayerProficiencyProfile,
    PolicyConfig,
    ProficiencyRuntime,
    available_policy_action_selections,
    available_proficiencies,
    policy_action_scores,
    proficiency_vector,
    resolve_proficiency,
)
from stg_lab.protocol import Action


def _logits(best: int, second: int | None = None) -> np.ndarray:
    values = np.full(18, -10.0, dtype=np.float32)
    values[best] = 10.0
    if second is not None:
        values[second] = 9.0
    return values


def _profile(**overrides) -> PlayerProficiencyProfile:
    values = {
        "name": "test",
        "reaction_delay_frames": 0,
        "direction_hold_frames": 0,
        "prediction_horizon_frames": 12,
        "shield_probability": 1.0,
        "suboptimal_action_probability": 0.0,
    }
    values.update(overrides)
    return PlayerProficiencyProfile(**values)


def test_proficiency_profiles_are_ordered_and_have_stable_vectors() -> None:
    assert available_proficiencies() == ("novice", "intermediate", "expert")
    novice = resolve_proficiency("NOVICE")
    intermediate = resolve_proficiency("intermediate")
    expert = resolve_proficiency("expert")

    assert novice.reaction_delay_frames > intermediate.reaction_delay_frames
    assert intermediate.reaction_delay_frames > expert.reaction_delay_frames
    assert novice.direction_hold_frames > intermediate.direction_hold_frames
    assert intermediate.direction_hold_frames > expert.direction_hold_frames
    assert novice.prediction_horizon_frames < intermediate.prediction_horizon_frames
    assert intermediate.prediction_horizon_frames < expert.prediction_horizon_frames
    assert novice.shield_probability < intermediate.shield_probability
    assert intermediate.shield_probability < expert.shield_probability
    assert novice.suboptimal_action_probability > intermediate.suboptimal_action_probability
    assert intermediate.suboptimal_action_probability > expert.suboptimal_action_probability
    np.testing.assert_array_equal(proficiency_vector(expert), expert.vector())


def test_factorized_action_scores_use_model_direction_and_speed_marginals() -> None:
    logits = np.full((2, 9), -10.0, dtype=np.float64)
    logits[0, 0] = 10.0
    logits[:, 1] = 9.5
    flattened = logits.reshape(18)

    joint = policy_action_scores(flattened, "joint")
    factorized = policy_action_scores(flattened, "factorized")
    direction_scores = np.logaddexp.reduce(logits, axis=0)
    speed_scores = np.logaddexp.reduce(logits, axis=1)

    assert available_policy_action_selections() == ("joint", "factorized")
    np.testing.assert_array_equal(joint, flattened)
    np.testing.assert_allclose(
        factorized,
        (speed_scores[:, None] + direction_scores[None, :]).reshape(18),
    )
    assert int(np.argmax(joint)) == 0
    assert int(np.argmax(factorized)) == 1


def test_factorized_action_scores_reject_invalid_mode_or_shape() -> None:
    with pytest.raises(ValueError, match="unknown policy action selection"):
        policy_action_scores(np.zeros(18), "independent")
    with pytest.raises(ValueError, match="exactly 18 logits"):
        policy_action_scores(np.zeros(17), "factorized")


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("name", ""),
        ("reaction_delay_frames", -1),
        ("direction_hold_frames", 1.5),
        ("prediction_horizon_frames", True),
        ("shield_probability", 1.1),
        ("suboptimal_action_probability", float("nan")),
    ),
)
def test_proficiency_profile_rejects_invalid_values(field, value) -> None:
    with pytest.raises(ValueError):
        _profile(**{field: value})

    with pytest.raises(ValueError, match="unknown proficiency"):
        resolve_proficiency("impossible")


def test_proficiency_runtime_applies_reaction_delay_and_direction_hold() -> None:
    delayed = ProficiencyRuntime(
        _profile(reaction_delay_frames=6), seed=7,
    )
    assert delayed.preferred_action(_logits(5), decision_interval=3) == Action()
    delayed.commit(Action(), decision_interval=3)
    assert delayed.preferred_action(_logits(3), decision_interval=3) == Action()
    delayed.commit(Action(), decision_interval=3)
    assert delayed.preferred_action(_logits(7), decision_interval=3) == Action.from_discrete(5)

    held = ProficiencyRuntime(
        _profile(direction_hold_frames=6), seed=7,
    )
    right = held.preferred_action(_logits(5), decision_interval=3)
    assert right == Action.from_discrete(5)
    held.commit(right, decision_interval=3)
    assert held.preferred_action(_logits(3), decision_interval=3) == right
    held.commit(right, decision_interval=3)
    assert held.preferred_action(_logits(3), decision_interval=3) == Action.from_discrete(3)


def test_proficiency_runtime_suboptimal_choice_and_shield_are_seeded() -> None:
    runtime = ProficiencyRuntime(
        _profile(
            shield_probability=0.0,
            suboptimal_action_probability=1.0,
        ),
        seed=91,
    )
    assert runtime.preferred_action(_logits(5, 7), decision_interval=3) == (
        Action.from_discrete(7)
    )
    assert runtime.should_apply_shield() is False
    runtime.reset(91)
    assert runtime.preferred_action(_logits(5, 7), decision_interval=3) == (
        Action.from_discrete(7)
    )


def test_policy_forward_and_gradient() -> None:
    config = PolicyConfig(feature_size=24, recurrent_size=32)
    model = HumanVisionPolicy(config)
    global_frames = torch.rand(2, 4, 6, 56, 48)
    local_frames = torch.rand(2, 4, 6, 40, 40)
    memory = torch.rand(2, 4)
    logits, risk, hidden = model(global_frames, local_frames, memory)
    visual = model.encode_visual(global_frames, local_frames)
    recurrent_logits, recurrent_risk, recurrent_hidden, recurrent = (
        model.forward_with_recurrent(global_frames, local_frames, memory)
    )
    assert logits.shape == (2, 4, 18)
    assert risk.shape == (2, 4)
    assert hidden.shape == (1, 2, 32)
    assert visual.shape == (2, 4, 48)
    assert recurrent_logits.shape == logits.shape
    assert recurrent_risk.shape == risk.shape
    assert recurrent_hidden.shape == hidden.shape
    assert recurrent.shape == (2, 4, 32)
    loss = logits.square().mean() + risk.square().mean()
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_policy_forward_with_visual_features_preserves_legacy_outputs() -> None:
    torch.manual_seed(81)
    config = PolicyConfig(feature_size=16, recurrent_size=24)
    model = HumanVisionPolicy(config)
    global_frames = torch.rand(2, 3, 6, 56, 48)
    local_frames = torch.rand(2, 3, 6, 40, 40)
    memory = torch.rand(2, 3, config.memory_size)
    proficiency = torch.rand(2, config.proficiency_size)
    initial_hidden = torch.rand(1, 2, config.recurrent_size)

    encoder_calls = {"global": 0, "local": 0}

    def count_global(_module, _inputs, _output) -> None:
        encoder_calls["global"] += 1

    def count_local(_module, _inputs, _output) -> None:
        encoder_calls["local"] += 1

    handles = (
        model.global_encoder.register_forward_hook(count_global),
        model.local_encoder.register_forward_hook(count_local),
    )
    outputs = model.forward_with_visual_features(
        global_frames,
        local_frames,
        memory,
        proficiency,
        initial_hidden,
    )
    for handle in handles:
        handle.remove()
    legacy = model.forward_with_recurrent(
        global_frames,
        local_frames,
        memory,
        proficiency,
        initial_hidden,
    )
    logits, risk, hidden, recurrent, visual_features = outputs

    assert encoder_calls == {"global": 1, "local": 1}
    assert logits.shape == (2, 3, config.action_count)
    assert risk.shape == (2, 3)
    assert hidden.shape == (1, 2, config.recurrent_size)
    assert recurrent.shape == (2, 3, config.recurrent_size)
    assert visual_features.shape == (2, 3, config.feature_size * 2)
    for legacy_output, feature_output in zip(legacy, outputs[:4], strict=True):
        assert torch.equal(legacy_output, feature_output)


def test_policy_forward_with_visual_features_carries_hidden_across_chunks() -> None:
    torch.manual_seed(29)
    config = PolicyConfig(feature_size=16, recurrent_size=24)
    model = HumanVisionPolicy(config)
    global_frames = torch.rand(2, 5, 6, 56, 48)
    local_frames = torch.rand(2, 5, 6, 40, 40)
    memory = torch.rand(2, 5, config.memory_size)
    proficiency = torch.rand(2, 5, config.proficiency_size)

    whole = model.forward_with_visual_features(
        global_frames,
        local_frames,
        memory,
        proficiency,
    )
    first = model.forward_with_visual_features(
        global_frames[:, :2],
        local_frames[:, :2],
        memory[:, :2],
        proficiency[:, :2],
    )
    second = model.forward_with_visual_features(
        global_frames[:, 2:],
        local_frames[:, 2:],
        memory[:, 2:],
        proficiency[:, 2:],
        first[2],
    )

    for output_index in (0, 1, 3, 4):
        chunked = torch.cat((first[output_index], second[output_index]), dim=1)
        torch.testing.assert_close(chunked, whole[output_index])
    torch.testing.assert_close(second[2], whole[2])


def test_policy_high_resolution_local_encoder_preserves_output_contract() -> None:
    config = PolicyConfig(
        feature_size=16,
        recurrent_size=24,
        local_feature_grid_size=8,
        local_downsample_stages=2,
    )
    model = HumanVisionPolicy(config)
    global_frames = torch.rand(1, 3, 6, 56, 48)
    local_frames = torch.rand(1, 3, 6, 40, 40)

    logits, risk, hidden = model(global_frames, local_frames)

    assert logits.shape == (1, 3, 18)
    assert risk.shape == (1, 3)
    assert hidden.shape == (1, 1, 24)
    assert model.local_encoder.network[4].stride == (1, 1)
    assert model.local_encoder.network[8].in_features == 64 * 8 * 8


@pytest.mark.parametrize(
    "overrides",
    (
        {"local_feature_grid_size": 0},
        {"local_downsample_stages": 1},
    ),
)
def test_policy_rejects_invalid_local_encoder_configuration(overrides) -> None:
    with pytest.raises(ValueError):
        PolicyConfig(**overrides)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable")
def test_policy_forward_supports_real_vision_shapes_on_mps() -> None:
    model = HumanVisionPolicy(PolicyConfig(feature_size=16, recurrent_size=24)).to("mps")
    global_frames = torch.rand(1, 4, 6, 56, 48, device="mps")
    local_frames = torch.rand(1, 4, 6, 40, 40, device="mps")
    logits, risk, _ = model(global_frames, local_frames)
    assert logits.shape == (1, 4, 18)
    assert risk.shape == (1, 4)
