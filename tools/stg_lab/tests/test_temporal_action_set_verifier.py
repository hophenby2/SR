from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from experiments import temporal_action_set_verifier as module
from experiments.temporal_action_set_verifier import (
    INFERENCE_INPUT_SEMANTICS,
    TemporalActionSetEpisode,
    TemporalActionSetTrainingConfig,
    TemporalActionSetVerifier,
    TemporalActionSetVerifierConfig,
    action_set_verifier_state_sha256,
    build_temporal_action_set_inputs,
    predict_temporal_action_set_verifier,
    temporal_action_set_targets,
    train_temporal_action_set_verifier,
)
from experiments.temporal_verifier_calibration import _beneficial_candidate_mask


def _config() -> TemporalActionSetVerifierConfig:
    return TemporalActionSetVerifierConfig(
        latent_size=6,
        hidden_size=8,
        bottleneck_size=4,
        ensemble_size=3,
    )


def _inputs(config: TemporalActionSetVerifierConfig, decisions: int = 8) -> torch.Tensor:
    generator = torch.Generator().manual_seed(31)
    selector = torch.rand((decisions, 18), generator=generator)
    selector /= selector.sum(dim=-1, keepdim=True)
    return build_temporal_action_set_inputs(
        policy_latents=torch.randn(
            (1, decisions, config.latent_size), generator=generator
        ),
        mean_action_probabilities=selector,
        mean_gate=torch.rand(decisions, generator=generator),
        minimum_gate=torch.rand(decisions, generator=generator),
        physical_danger_probabilities=torch.rand(
            (1, decisions, 18), generator=generator
        ),
        parent_actions=torch.arange(decisions).remainder(18),
        config=config,
    )


def _episodes(config: TemporalActionSetVerifierConfig) -> list[TemporalActionSetEpisode]:
    result = []
    for seed in (1, 2):
        inputs = _inputs(config) + seed / 100.0
        labels = torch.zeros((8, 18), dtype=torch.bool)
        labels[0, 2] = True
        labels[2, 4] = True
        labels[3, 6] = True
        labels[5, 8] = True
        candidates = torch.tensor([2, 1, 4, 6, 1, 8, 1, 1])
        result.append(TemporalActionSetEpisode(
            seed,
            inputs,
            torch.ones(8, dtype=torch.bool),
            labels,
            candidates,
            torch.ones(8, dtype=torch.bool),
            labels.gather(-1, candidates.unsqueeze(-1)).squeeze(-1),
        ))
    return result


def test_action_set_input_contains_only_frozen_live_fields() -> None:
    config = _config()
    inputs = _inputs(config)
    assert inputs.shape == (1, 8, config.input_size)
    assert torch.isfinite(inputs).all()
    parameters = inspect.signature(build_temporal_action_set_inputs).parameters
    for forbidden in INFERENCE_INPUT_SEMANTICS["forbidden"]:
        assert forbidden not in parameters
    assert INFERENCE_INPUT_SEMANTICS["candidate_role"].startswith("gather_only")


def _label_episode() -> SimpleNamespace:
    labels = torch.zeros((6, 18), dtype=torch.bool)
    labels[0, 2] = True
    labels[2, 4] = True
    labels[3, 5] = True
    evaluation_safe = torch.ones((6, 18), dtype=torch.bool)
    return SimpleNamespace(
        parent_actions=torch.ones(6, dtype=torch.int64),
        preferred_equivalent_actions=labels,
        preferred_correction_required=torch.tensor(
            [True, False, True, True, False, False], dtype=torch.bool
        ),
        evaluation_safe_actions=evaluation_safe,
        gate_valid=torch.tensor([True, True, True, True, True, False]),
        gate_targets=torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 0.0]),
        anticipatory=torch.tensor([True, True, True, True, False, False]),
        anticipatory_lead_decisions=torch.tensor([4, 6, 10, 3, 0, 0]),
    )


def test_action_set_targets_keep_full_set_and_no_correction_negative() -> None:
    episode = _label_episode()
    candidates = torch.tensor([2, 1, 4, 5, 2, 2])
    dense_mask, labels, selected_mask, selected_labels = (
        temporal_action_set_targets(episode, candidates)
    )
    assert dense_mask.tolist() == [True, True, True, False, False, False]
    assert labels[0, 2]
    assert not labels[1].any()
    assert labels[2, 4]
    assert selected_mask.tolist() == [True, False, True, True, True, True]
    assert selected_labels.tolist() == [True, False, True, True, False, False]


def test_selected_targets_equal_runtime_beneficial_candidate_definition() -> None:
    episode = _label_episode()
    candidates = torch.tensor([2, 1, 4, 5, 2, 2])
    _dense, _labels, selected_mask, selected_labels = (
        temporal_action_set_targets(episode, candidates)
    )

    assert torch.equal(
        selected_labels,
        _beneficial_candidate_mask(episode, candidates),
    )
    assert selected_mask[3]
    assert int(episode.anticipatory_lead_decisions[3]) < 4
    assert selected_mask[4] and not selected_labels[4]
    assert not selected_mask[1]


def test_gate_invalid_changed_candidate_is_supported_negative() -> None:
    episode = _label_episode()
    candidates = torch.tensor([2, 1, 4, 5, 2, 2])
    _dense, _labels, selected_mask, selected_labels = (
        temporal_action_set_targets(episode, candidates)
    )

    assert not episode.gate_valid[5]
    assert candidates[5] != episode.parent_actions[5]
    assert selected_mask[5]
    assert not selected_labels[5]
    assert not selected_mask[1]


def test_action_set_targets_reject_unsafe_equivalent_label() -> None:
    episode = _label_episode()
    episode.evaluation_safe_actions[2, 4] = False

    with pytest.raises(ValueError, match="must be evaluation-safe"):
        temporal_action_set_targets(
            episode,
            torch.tensor([2, 1, 4, 5, 2, 2]),
        )


def test_runtime_support_negative_has_downward_gradient() -> None:
    logits = torch.tensor([2.0, 1.0, -1.0], requires_grad=True)
    labels = torch.tensor([False, True, False])
    selected_bce = torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        labels.to(logits.dtype),
    )
    pairwise = module.hard_negative_pairwise_loss(
        logits,
        labels,
        margin=0.25,
        temperature=0.5,
        hard_negative_fraction=0.25,
    )
    (selected_bce + 2.0 * pairwise).backward()

    assert logits.grad is not None
    assert logits.grad[0] > 0.0


def test_temporal_verifier_is_prefix_causal_and_resets_between_episodes() -> None:
    config = _config()
    torch.manual_seed(83)
    verifier = TemporalActionSetVerifier(config).eval()
    first = _inputs(config)
    second = first.flip(1).contiguous()

    with torch.no_grad():
        full = verifier.forward_logits(first)
        prefix = verifier.forward_logits(first[:, :5])
        second_before = verifier.forward_logits(second)
        verifier.forward_logits(first)
        second_after = verifier.forward_logits(second)

    assert torch.allclose(full[:, :5], prefix, atol=1e-6, rtol=1e-6)
    assert torch.equal(second_before, second_after)


def test_action_set_training_is_deterministic_and_restores_state() -> None:
    config = _config()
    torch.manual_seed(55)
    first = TemporalActionSetVerifier(config)
    torch.manual_seed(55)
    second = TemporalActionSetVerifier(config)
    before = action_set_verifier_state_sha256(first)
    first.eval()
    second.eval()
    first_parameter = next(first.parameters())
    second_parameter = next(second.parameters())
    first_parameter.requires_grad_(False)
    second_parameter.requires_grad_(False)
    first_parameter.grad = torch.ones_like(first_parameter)
    second_parameter.grad = torch.ones_like(second_parameter)
    training = TemporalActionSetTrainingConfig(epochs=2)
    first_history = train_temporal_action_set_verifier(
        first, _episodes(config), seed=71, config=training
    )
    second_history = train_temporal_action_set_verifier(
        second, _episodes(config), seed=71, config=training
    )
    assert first_history == second_history
    assert action_set_verifier_state_sha256(first) == (
        action_set_verifier_state_sha256(second)
    )
    assert action_set_verifier_state_sha256(first) != before
    assert not first.training and not second.training
    assert not first_parameter.requires_grad and not second_parameter.requires_grad
    assert torch.equal(first_parameter.grad, torch.ones_like(first_parameter))
    assert torch.equal(second_parameter.grad, torch.ones_like(second_parameter))


def test_action_set_training_failure_rolls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config()
    verifier = TemporalActionSetVerifier(config)
    before = action_set_verifier_state_sha256(verifier)

    def fail(*_args: object, **_kwargs: object) -> torch.Tensor:
        raise RuntimeError("forced pairwise failure")

    monkeypatch.setattr(module, "hard_negative_pairwise_loss", fail)
    with pytest.raises(RuntimeError, match="forced pairwise"):
        train_temporal_action_set_verifier(
            verifier,
            _episodes(config),
            seed=3,
            config=TemporalActionSetTrainingConfig(epochs=1),
        )
    assert action_set_verifier_state_sha256(verifier) == before


def test_prediction_gathers_frozen_candidate_without_changing_it() -> None:
    config = _config()
    verifier = TemporalActionSetVerifier(config)
    candidates = torch.arange(8).remainder(18)
    prediction = predict_temporal_action_set_verifier(
        verifier, _inputs(config), candidates
    )
    assert prediction["confidence"].shape == candidates.shape
    assert prediction["all_selected_members_finite"].all()
    expected = prediction["mean_membership_probabilities"].gather(
        -1, candidates.unsqueeze(-1)
    ).squeeze(-1)
    assert torch.allclose(prediction["confidence"], expected)
