import numpy as np
import pytest
import torch

from stg_lab.policy import (
    HumanVisionPolicy,
    PlayerProficiencyProfile,
    PolicyConfig,
    ProficiencyRuntime,
    available_proficiencies,
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


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable")
def test_policy_forward_supports_real_vision_shapes_on_mps() -> None:
    model = HumanVisionPolicy(PolicyConfig(feature_size=16, recurrent_size=24)).to("mps")
    global_frames = torch.rand(1, 4, 6, 56, 48, device="mps")
    local_frames = torch.rand(1, 4, 6, 40, 40, device="mps")
    logits, risk, _ = model(global_frames, local_frames)
    assert logits.shape == (1, 4, 18)
    assert risk.shape == (1, 4)
