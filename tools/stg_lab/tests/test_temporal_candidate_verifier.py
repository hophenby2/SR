from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from experiments import temporal_candidate_verifier as verifier_module
from experiments.temporal_candidate_verifier import (
    INFERENCE_INPUT_SEMANTICS,
    TemporalCandidateVerifier,
    TemporalCandidateVerifierConfig,
    TemporalVerifierEpisode,
    TemporalVerifierTrainingConfig,
    build_temporal_verifier_inputs,
    hard_negative_pairwise_loss,
    predict_temporal_candidate_verifier,
    selected_candidate_targets,
    train_temporal_candidate_verifier,
    verifier_state_sha256,
)


def _config() -> TemporalCandidateVerifierConfig:
    return TemporalCandidateVerifierConfig(
        latent_size=4,
        hidden_size=8,
        bottleneck_size=4,
        ensemble_size=3,
    )


def _frozen_inputs(
    config: TemporalCandidateVerifierConfig,
    decisions: int = 8,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(17)
    candidates = torch.arange(decisions, dtype=torch.int64).remainder(18)
    parent = (candidates + 1).remainder(18)
    action_probabilities = torch.rand(
        (decisions, 18), generator=generator, dtype=torch.float32
    )
    action_probabilities /= action_probabilities.sum(dim=-1, keepdim=True)
    return {
        "action_latents": torch.randn(
            (1, decisions, config.latent_size),
            generator=generator,
        ),
        "candidates": candidates,
        "mean_action_probabilities": action_probabilities,
        "mean_gate": torch.rand(decisions, generator=generator),
        "minimum_gate": torch.rand(decisions, generator=generator),
        "physical_danger_probabilities": torch.rand(
            (1, decisions, 18), generator=generator
        ),
        "parent_actions": parent,
        "config": config,
    }


def _training_episodes(
    config: TemporalCandidateVerifierConfig,
) -> list[TemporalVerifierEpisode]:
    episodes = []
    for seed in (11, 12):
        values = _frozen_inputs(config)
        values["action_latents"] = values["action_latents"] + seed / 100.0
        inputs = build_temporal_verifier_inputs(**values)
        mask = torch.ones(8, dtype=torch.bool)
        labels = torch.tensor(
            [True, False, False, True, False, False, True, False],
            dtype=torch.bool,
        )
        episodes.append(TemporalVerifierEpisode(seed, inputs, mask, labels))
    return episodes


def test_inference_builder_accepts_one_base_member_and_has_no_teacher_argument() -> None:
    config = _config()
    result = build_temporal_verifier_inputs(**_frozen_inputs(config))
    assert result.shape == (1, 8, config.input_size)
    assert torch.isfinite(result).all()

    parameters = inspect.signature(build_temporal_verifier_inputs).parameters
    for forbidden in INFERENCE_INPUT_SEMANTICS["forbidden"]:
        assert forbidden not in parameters
    assert "preferred_equivalent_actions" not in parameters
    assert INFERENCE_INPUT_SEMANTICS["source"].endswith("outputs_only")


def test_input_builder_rejects_nonfinite_and_bad_candidates() -> None:
    values = _frozen_inputs(_config())
    values["action_latents"][0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        build_temporal_verifier_inputs(**values)

    values = _frozen_inputs(_config())
    values["candidates"][0] = 18
    with pytest.raises(ValueError, match="outside"):
        build_temporal_verifier_inputs(**values)


def _label_episode() -> SimpleNamespace:
    decisions = 4
    equivalent = torch.zeros((decisions, 18), dtype=torch.bool)
    equivalent[0, 2] = True
    equivalent[2, 4] = True
    equivalent[3, 5] = True
    return SimpleNamespace(
        parent_actions=torch.tensor([1, 1, 1, 1]),
        preferred_equivalent_actions=equivalent,
        preferred_correction_required=torch.tensor(
            [True, False, True, True], dtype=torch.bool
        ),
        gate_valid=torch.ones(decisions, dtype=torch.bool),
        gate_targets=torch.ones(decisions),
        anticipatory=torch.ones(decisions, dtype=torch.bool),
        anticipatory_lead_decisions=torch.tensor([4, 6, 10, 3]),
    )


def test_selected_labels_include_no_correction_negative_and_only_lead_four_to_ten() -> None:
    candidates = torch.tensor([2, 2, 4, 5])
    mask, labels = selected_candidate_targets(_label_episode(), candidates)
    assert mask.tolist() == [True, True, True, False]
    assert labels.tolist() == [True, False, True, True]


def test_selected_labels_reject_invalid_equivalence_contracts() -> None:
    episode = _label_episode()
    episode.preferred_equivalent_actions[0].zero_()
    with pytest.raises(ValueError, match="must contain"):
        selected_candidate_targets(episode, torch.tensor([2, 2, 4, 5]))

    episode = _label_episode()
    episode.preferred_equivalent_actions[0, 1] = True
    with pytest.raises(ValueError, match="parent action"):
        selected_candidate_targets(episode, torch.tensor([2, 2, 4, 5]))


def test_hard_negative_pairwise_loss_rewards_positive_tail_separation() -> None:
    labels = torch.tensor([True, True, False, False, False, False])
    separated = torch.tensor([3.0, 2.0, -2.0, -3.0, -4.0, -5.0])
    reversed_scores = -separated
    kwargs = {
        "margin": 0.25,
        "temperature": 0.5,
        "hard_negative_fraction": 0.25,
    }
    assert hard_negative_pairwise_loss(
        separated, labels, **kwargs
    ) < hard_negative_pairwise_loss(reversed_scores, labels, **kwargs)


def test_training_is_deterministic_updates_weights_and_restores_runtime_state() -> None:
    config = _config()
    torch.manual_seed(101)
    first = TemporalCandidateVerifier(config)
    torch.manual_seed(101)
    second = TemporalCandidateVerifier(config)
    before = verifier_state_sha256(first)
    first.train(False)
    second.train(False)
    first_parameter = next(first.parameters())
    second_parameter = next(second.parameters())
    first_parameter.requires_grad_(False)
    second_parameter.requires_grad_(False)
    first_parameter.grad = torch.ones_like(first_parameter)
    second_parameter.grad = torch.ones_like(second_parameter)
    training = TemporalVerifierTrainingConfig(epochs=2)
    first_history = train_temporal_candidate_verifier(
        first, _training_episodes(config), seed=77, config=training
    )
    second_history = train_temporal_candidate_verifier(
        second, _training_episodes(config), seed=77, config=training
    )
    assert first_history == second_history
    assert verifier_state_sha256(first) == verifier_state_sha256(second)
    assert verifier_state_sha256(first) != before
    assert first.training is False and second.training is False
    assert first_parameter.requires_grad is False
    assert second_parameter.requires_grad is False
    assert torch.equal(first_parameter.grad, torch.ones_like(first_parameter))
    assert torch.equal(second_parameter.grad, torch.ones_like(second_parameter))


def test_training_failure_rolls_back_weights_and_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config()
    torch.manual_seed(9)
    verifier = TemporalCandidateVerifier(config)
    verifier.train(False)
    parameter = next(verifier.parameters())
    parameter.requires_grad_(False)
    parameter.grad = torch.full_like(parameter, 2.0)
    before = verifier_state_sha256(verifier)

    def fail(*_args: object, **_kwargs: object) -> torch.Tensor:
        raise RuntimeError("forced failure")

    monkeypatch.setattr(verifier_module, "hard_negative_pairwise_loss", fail)
    with pytest.raises(RuntimeError, match="forced failure"):
        train_temporal_candidate_verifier(
            verifier,
            _training_episodes(config),
            seed=5,
            config=TemporalVerifierTrainingConfig(epochs=1),
        )
    assert verifier_state_sha256(verifier) == before
    assert verifier.training is False
    assert parameter.requires_grad is False
    assert torch.equal(parameter.grad, torch.full_like(parameter, 2.0))


def test_prediction_reports_nonfinite_member_fail_closed() -> None:
    config = _config()
    verifier = TemporalCandidateVerifier(config)
    inputs = build_temporal_verifier_inputs(**_frozen_inputs(config))
    clean = predict_temporal_candidate_verifier(verifier, inputs)
    assert clean["all_members_finite"].all()
    assert torch.isfinite(clean["confidence"]).all()

    with torch.no_grad():
        next(verifier.members[0].parameters()).fill_(float("nan"))
    failed = predict_temporal_candidate_verifier(verifier, inputs)
    assert not failed["all_members_finite"].any()
    assert torch.isnan(failed["confidence"]).all()
