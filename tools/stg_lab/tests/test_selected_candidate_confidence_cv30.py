from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from experiments import compare_selected_candidate_confidence_cv30 as screen
from experiments import train_temporal_residual_adapter as trainer
from experiments.compare_selected_candidate_confidence_cv30 import (
    ARM_NAME,
    BASE_SEED,
    DUAL_REFERENCE_SHA256,
    FIXED_CALIBRATION_FUNCTION_SHA256,
    HEAD_LOSS_WEIGHT,
    HEAD_TRAINING_EPOCHS,
    SCREENING_EPOCHS,
    SELECTED_CONFIDENCE_CONFIG,
    _adaptive_gate,
    _argument_parser,
    _calibration_function_sha256,
    _candidate_digest,
    _copy_plain_state_into_dual,
    _file_digest_snapshot,
    _reserve_formal_campaign,
    _reverify_file_digest_snapshot,
    _selected_candidate_confidence_diagnostics,
    _validate_dual_reference,
    _verified_source_digest_map,
    main,
)
from experiments.train_temporal_residual_adapter import (
    _early_selected_candidate_confidence_mask,
    _frozen_ensemble_selected_candidates,
    _selected_candidate_confidence_targets,
    _train_frozen_ensemble_selected_confidence_heads,
)
from stg_lab.residual_adapter import (
    ResidualAdapterConfig,
    ResidualCorrectionAdapter,
    ensemble_action_summary,
    finite_action_probabilities,
)


def _adapter() -> ResidualCorrectionAdapter:
    return ResidualCorrectionAdapter(ResidualAdapterConfig(
        recurrent_size=4,
        hidden_size=6,
        ensemble_size=3,
        action_logit_mode="parent_residual_joint",
        separate_action_recurrent=True,
        per_action_membership_confidence=True,
    ))


def _episode(adapter: ResidualCorrectionAdapter) -> SimpleNamespace:
    decisions = 4
    equivalent = torch.zeros((decisions, 18), dtype=torch.bool)
    equivalent[[0, 2, 3], 0] = True
    return SimpleNamespace(
        seed=17,
        decisions=decisions,
        features=torch.zeros((1, decisions, adapter.config.feature_size)),
        parent_logits=torch.zeros((1, decisions, 18)),
        parent_actions=torch.ones(decisions, dtype=torch.int64),
        gate_valid=torch.ones(decisions, dtype=torch.bool),
        gate_targets=torch.ones(decisions),
        anticipatory=torch.ones(decisions, dtype=torch.bool),
        anticipatory_lead_decisions=torch.tensor([4, 6, 10, 3]),
        preferred_equivalent_actions=equivalent,
        preferred_correction_required=torch.tensor(
            [True, False, True, True],
            dtype=torch.bool,
        ),
    )


def _base_state(adapter: ResidualCorrectionAdapter) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().clone()
        for name, value in adapter.state_dict().items()
        if ".membership_head." not in name
    }


def _parameter_runtime_state(
    adapter: ResidualCorrectionAdapter,
) -> list[tuple[bool, torch.Tensor | None]]:
    return [
        (
            parameter.requires_grad,
            None if parameter.grad is None else parameter.grad.detach().clone(),
        )
        for parameter in adapter.parameters()
    ]


def _assert_parameter_runtime_state(
    adapter: ResidualCorrectionAdapter,
    expected: list[tuple[bool, torch.Tensor | None]],
) -> None:
    for parameter, (requires_grad, gradient) in zip(
        adapter.parameters(),
        expected,
        strict=True,
    ):
        assert parameter.requires_grad is requires_grad
        if gradient is None:
            assert parameter.grad is None
        else:
            assert parameter.grad is not None
            assert torch.equal(parameter.grad, gradient)


def _selector_candidates(
    adapter: ResidualCorrectionAdapter,
    episode: SimpleNamespace,
) -> tuple[torch.Tensor, torch.Tensor]:
    outputs = [
        member.forward_with_all_safety_and_membership(episode.features)
        for member in adapter.members
    ]
    logits = torch.stack([
        adapter.decode_action_logits(output[1][0], episode.parent_logits[0])
        for output in outputs
    ])
    probabilities, finite = finite_action_probabilities(
        logits,
        adapter.config.action_logit_mode,
    )
    summary = ensemble_action_summary(probabilities, finite)
    return summary["candidates"], probabilities.argmax(dim=-1)


def _passing_summary(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "calibration_successful_folds": 2,
        "audit_runtime_eligible_folds": 2,
        "calibrated_audit_runtime_ineligible_folds": [],
        "outer_audit_micro": {
            "targets": 690,
            "finite_top1": 690,
            "equivalent_top1": 138,
            "direction_correct": 138,
            "speed_correct": 437,
        },
    }
    values.update(overrides)
    return values


def _dual_reference() -> dict[str, object]:
    return {
        "kind": "dual_head_membership_third_adaptive_development_screen_cv30",
        "training_only": True,
        "deployment_artifact_written": False,
        "formal_deployment_artifact_written": False,
        "deployment_eligible": False,
        "acceptance_claim": False,
        "adaptive_development_screen": True,
        "adaptive_development_screen_sequence": 3,
        "independent_statistical_validation": False,
        "objective_arms": [screen.dual.DUAL_HEAD_ARM_NAME],
        "experiment_config": {
            "epochs": 6,
            "base_seed": 20260901,
            "device": "cpu",
            "deterministic_algorithms": True,
            "ensemble_size_screening_override": 1,
        },
        "adaptive_development_gate": {
            "passed": False,
            "eligible_for_fixed_followup": False,
        },
        "dual_head_summary": {
            "outer_audit_micro": {"targets": 690},
        },
        "folds": [
            {
                "fold": fold.index,
                "fit_seeds": list(fold.fit_seeds),
                "calibration_seeds": list(fold.calibration_seeds),
                "audit_seeds": list(fold.audit_seeds),
                "fold_seed": BASE_SEED + fold.index * 100_003,
                "member_seed": BASE_SEED + fold.index * 100_003 + 1_009,
                "initial_membership_head_state_sha256": (
                    f"{fold.index + 1:064x}"
                ),
                "dual_head": {
                    "implementation_invariants": {"all_passed": True},
                },
            }
            for fold in screen.balanced._fixed_cv30_folds()
        ],
    }


def test_early_mask_includes_no_correction_rows_and_excludes_lead_three() -> None:
    adapter = _adapter()
    episode = _episode(adapter)
    mask = _early_selected_candidate_confidence_mask(episode)
    assert mask.tolist() == [True, True, True, False]

    candidates = torch.zeros(episode.decisions, dtype=torch.int64)
    target_mask, targets = _selected_candidate_confidence_targets(
        episode,
        candidates,
    )
    assert torch.equal(target_mask, mask)
    assert targets.tolist() == [True, False, True, True]


def test_selected_confidence_target_rejects_required_row_without_equivalent() -> None:
    episode = _episode(_adapter())
    episode.preferred_equivalent_actions[0].zero_()
    candidates = torch.zeros(episode.decisions, dtype=torch.int64)

    with pytest.raises(ValueError, match="must contain an equivalent action"):
        _selected_candidate_confidence_targets(episode, candidates)


def test_selected_confidence_target_rejects_parent_as_equivalent() -> None:
    episode = _episode(_adapter())
    episode.preferred_equivalent_actions[0, 1] = True
    candidates = torch.zeros(episode.decisions, dtype=torch.int64)

    with pytest.raises(ValueError, match="parent action cannot be"):
        _selected_candidate_confidence_targets(episode, candidates)


def test_head_only_training_uses_one_global_candidate_for_divergent_members() -> None:
    torch.manual_seed(11)
    adapter = _adapter()
    episode = _episode(adapter)
    episode.preferred_correction_required[1] = True
    episode.preferred_equivalent_actions[1, 0] = True
    for index, member in enumerate(adapter.members):
        member.action_head.weight.data.zero_()
        member.action_head.bias.data.zero_()
        member.action_head.bias.data[index] = 8.0
        assert member.membership_head is not None
        member.membership_head.weight.data.zero_()
        member.membership_head.bias.data.zero_()

    global_candidates, local_candidates = _selector_candidates(adapter, episode)
    assert global_candidates.tolist() == [0] * episode.decisions
    assert local_candidates[:, 0].tolist() == [0, 1, 2]
    base_before = _base_state(adapter)
    frozen = _frozen_ensemble_selected_candidates(
        adapter,
        [episode],
        chunk_length=4,
    )
    candidate_digest_before = _candidate_digest([episode], frozen)

    history = _train_frozen_ensemble_selected_confidence_heads(
        adapter,
        [episode],
        frozen_candidates=frozen,
        seed=20261910,
        epochs=1,
        learning_rate=3e-4,
        weight_decay=1e-3,
        chunk_length=4,
        loss_weight=12.0,
        device="cpu",
    )

    assert history[0]["labelled_rows"] == 3.0
    assert history[0]["positive_rows"] == 3.0
    for member in adapter.members:
        assert member.membership_head is not None
        assert member.membership_head.bias[0].item() > 0.0
    assert all(
        torch.equal(value, _base_state(adapter)[name])
        for name, value in base_before.items()
    )
    repeated = _frozen_ensemble_selected_candidates(
        adapter,
        [episode],
        chunk_length=4,
    )
    assert _candidate_digest([episode], repeated) == candidate_digest_before


def test_no_correction_row_trains_selected_candidate_as_a_negative() -> None:
    adapter = _adapter()
    episode = _episode(adapter)
    episode.anticipatory_lead_decisions[:] = torch.tensor([4, 3, 3, 3])
    episode.preferred_correction_required[0] = False
    episode.preferred_equivalent_actions[0].zero_()
    for member in adapter.members:
        member.action_head.weight.data.zero_()
        member.action_head.bias.data.zero_()
        assert member.membership_head is not None
        member.membership_head.weight.data.zero_()
        member.membership_head.bias.data.zero_()
    frozen = _frozen_ensemble_selected_candidates(
        adapter,
        [episode],
        chunk_length=4,
    )

    history = _train_frozen_ensemble_selected_confidence_heads(
        adapter,
        [episode],
        frozen_candidates=frozen,
        seed=20261910,
        epochs=1,
        learning_rate=3e-4,
        weight_decay=1e-3,
        chunk_length=4,
        loss_weight=12.0,
        device="cpu",
    )

    assert history[0]["labelled_rows"] == 1.0
    assert history[0]["positive_rows"] == 0.0
    for member in adapter.members:
        assert member.membership_head is not None
        assert member.membership_head.bias[0].item() < 0.0


def test_head_fit_restores_grad_flags_gradients_and_module_modes() -> None:
    adapter = _adapter()
    episode = _episode(adapter)
    for index, parameter in enumerate(adapter.parameters()):
        parameter.requires_grad_(index % 2 == 0)
        parameter.grad = (
            torch.full_like(parameter, float(index + 1))
            if index % 3 == 0 else
            None
        )
    adapter.train()
    adapter.members[0].action_head.eval()
    adapter.members[1].membership_head.eval()  # type: ignore[union-attr]
    parameter_state = _parameter_runtime_state(adapter)
    module_modes = [module.training for module in adapter.modules()]
    frozen = _frozen_ensemble_selected_candidates(
        adapter,
        [episode],
        chunk_length=4,
    )

    _train_frozen_ensemble_selected_confidence_heads(
        adapter,
        [episode],
        frozen_candidates=frozen,
        seed=20261910,
        epochs=1,
        learning_rate=3e-4,
        weight_decay=1e-3,
        chunk_length=4,
        loss_weight=12.0,
        device="cpu",
    )

    _assert_parameter_runtime_state(adapter, parameter_state)
    assert [module.training for module in adapter.modules()] == module_modes


def test_head_fit_rolls_back_all_state_after_post_backward_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _adapter()
    episode = _episode(adapter)
    for index, parameter in enumerate(adapter.parameters()):
        parameter.grad = (
            torch.full_like(parameter, float(index + 1))
            if index % 4 == 0 else
            None
        )
    state_before = {
        name: value.detach().clone()
        for name, value in adapter.state_dict().items()
    }
    parameter_state = _parameter_runtime_state(adapter)
    module_modes = [module.training for module in adapter.modules()]
    frozen = _frozen_ensemble_selected_candidates(
        adapter,
        [episode],
        chunk_length=4,
    )

    def fail_after_backward(*_args: object, **_kwargs: object) -> torch.Tensor:
        raise RuntimeError("injected post-backward failure")

    monkeypatch.setattr(
        trainer,
        "_clip_membership_confidence_gradients",
        fail_after_backward,
    )
    with pytest.raises(RuntimeError, match="injected post-backward failure"):
        _train_frozen_ensemble_selected_confidence_heads(
            adapter,
            [episode],
            frozen_candidates=frozen,
            seed=20261910,
            epochs=1,
            learning_rate=3e-4,
            weight_decay=1e-3,
            chunk_length=4,
            loss_weight=12.0,
            device="cpu",
        )

    assert all(
        torch.equal(value, adapter.state_dict()[name])
        for name, value in state_before.items()
    )
    _assert_parameter_runtime_state(adapter, parameter_state)
    assert [module.training for module in adapter.modules()] == module_modes


def test_head_only_training_fails_closed_on_nonfinite_global_selector() -> None:
    adapter = _adapter()
    episode = _episode(adapter)
    adapter.members[1].action_head.bias.data[0] = float("nan")
    frozen = {episode.seed: torch.zeros(episode.decisions, dtype=torch.int64)}
    with pytest.raises(ValueError, match="selector is nonfinite"):
        _train_frozen_ensemble_selected_confidence_heads(
            adapter,
            [episode],
            frozen_candidates=frozen,
            seed=20261910,
            epochs=1,
            learning_rate=3e-4,
            weight_decay=1e-3,
            chunk_length=4,
            loss_weight=12.0,
            device="cpu",
        )


def test_head_only_training_rejects_nonfinite_unselected_membership_cell() -> None:
    adapter = _adapter()
    episode = _episode(adapter)
    membership_head = adapter.members[2].membership_head
    assert membership_head is not None
    membership_head.bias.data[17] = float("nan")
    frozen = _frozen_ensemble_selected_candidates(
        adapter,
        [episode],
        chunk_length=4,
    )
    with pytest.raises(ValueError, match="membership confidence is nonfinite"):
        _train_frozen_ensemble_selected_confidence_heads(
            adapter,
            [episode],
            frozen_candidates=frozen,
            seed=20261910,
            epochs=1,
            learning_rate=3e-4,
            weight_decay=1e-3,
            chunk_length=4,
            loss_weight=12.0,
            device="cpu",
        )


def test_fourth_screen_cli_exposes_no_training_or_selection_controls() -> None:
    destinations = {action.dest for action in _argument_parser()._actions}
    assert destinations == {
        "help",
        "failure",
        "expansion_inventory",
        "parent",
        "plain_reference",
        "balanced_reference",
        "unweighted_reference",
        "dual_reference",
        "output",
        "cpu_threads",
    }
    for prohibited in (
        "epochs",
        "seed",
        "retry",
        "loss_weight",
        "loss_mode",
        "row_mask",
        "candidate_mode",
        "threshold",
    ):
        assert prohibited not in destinations


def test_fourth_screen_objective_is_fully_frozen() -> None:
    assert ARM_NAME == "frozen_plain_selector_early_selected_candidate_confidence"
    assert SCREENING_EPOCHS == HEAD_TRAINING_EPOCHS == 6
    assert BASE_SEED == 20260901
    assert HEAD_LOSS_WEIGHT == 12.0
    assert DUAL_REFERENCE_SHA256 == (
        "88adfb80f8aa09a7551be3bef363abb994e4580581c31de64ca3dfac1c179a4a"
    )
    assert SELECTED_CONFIDENCE_CONFIG == {
        "schema": "frozen_ensemble_selected_candidate_early_unweighted_bce_v1",
        "selector_phase_epochs": 6,
        "confidence_phase_epochs": 6,
        "target_rows": (
            "fit_only_gate_valid_positive_anticipatory_lead_4_through_10"
        ),
        "no_correction_rows": "included_as_negative",
        "candidate": "argmax_mean_frozen_selector_softmax",
        "candidate_device": "deterministic_cpu_float32",
        "target": "preferred_equivalent_actions_at_frozen_candidate",
        "loss": "selected_scalar_unweighted_binary_cross_entropy",
        "loss_weight": 12.0,
        "head_input": "cached_detached_frozen_action_recurrent",
        "selector_gradient_from_confidence_loss": False,
        "optimizer": "independent_adamw_per_membership_head",
        "learning_rate": 3e-4,
        "weight_decay": 1e-3,
        "fixed_label_batch_size": 128,
        "gradient_clip": "independent_membership_head_group",
        "gradient_clip_max_norm": 5.0,
        "retry": False,
    }


def test_fixed_calibration_function_digest_matches_preregistered_source() -> None:
    assert _calibration_function_sha256() == FIXED_CALIBRATION_FUNCTION_SHA256


def test_plain_state_copy_preserves_head_and_copies_every_base_tensor() -> None:
    dual_config = _adapter().config
    plain_config = replace(
        dual_config,
        per_action_membership_confidence=False,
    )
    torch.manual_seed(991)
    plain_adapter = ResidualCorrectionAdapter(plain_config)
    torch.manual_seed(991)
    dual_adapter = ResidualCorrectionAdapter(dual_config)
    with torch.no_grad():
        for parameter in plain_adapter.parameters():
            parameter.add_(0.25)
    head_before = {
        name: value.detach().clone()
        for name, value in dual_adapter.state_dict().items()
        if ".membership_head." in name
    }

    _copy_plain_state_into_dual(plain_adapter, dual_adapter)

    dual_state = dual_adapter.state_dict()
    assert all(
        torch.equal(value, dual_state[name])
        for name, value in plain_adapter.state_dict().items()
    )
    assert all(
        torch.equal(value, dual_state[name])
        for name, value in head_before.items()
    )


def test_dual_reference_requires_exact_sequence_three_protocol() -> None:
    prior, folds = _validate_dual_reference(_dual_reference())
    assert prior == {
        "adaptive_development_screen_sequence": 3,
        "gate_passed": False,
        "outer_audit_micro": {"targets": 690},
    }
    assert set(folds) == {0, 1, 2}
    assert [
        folds[index]["initial_membership_head_state_sha256"]
        for index in sorted(folds)
    ] == [f"{index + 1:064x}" for index in range(3)]


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda value: value.update({
                "adaptive_development_screen_sequence": 2,
            }),
            "sequence differs",
        ),
        (
            lambda value: value.update({"acceptance_claim": True}),
            "flag differs",
        ),
        (
            lambda value: value["adaptive_development_gate"].update({
                "passed": True,
            }),
            "unexpectedly passed",
        ),
        (
            lambda value: value["folds"][1]["dual_head"][
                "implementation_invariants"
            ].update({"all_passed": False}),
            "invariant did not pass",
        ),
        (
            lambda value: value["folds"][1].update({
                "initial_membership_head_state_sha256": "too-short",
            }),
            "initial membership state hash",
        ),
    ),
)
def test_dual_reference_rejects_forged_prior(
    mutation: object,
    message: str,
) -> None:
    reference = _dual_reference()
    mutation(reference)  # type: ignore[operator]
    with pytest.raises(ValueError, match=message):
        _validate_dual_reference(reference)


def test_fourth_adaptive_gate_reuses_fixed_gate_and_declares_sequence_four() -> None:
    base = screen.balanced._adaptive_development_gate(
        _passing_summary(),
        epochs=SCREENING_EPOCHS,
    )
    gate = _adaptive_gate(_passing_summary())
    assert all(
        gate["criteria"][name] == value
        for name, value in base["criteria"].items()
    )
    assert all(
        gate["checks"][name] == value
        for name, value in base["checks"].items()
    )
    assert gate["criteria"]["required_selector_training_epochs"] == 6
    assert gate["criteria"]["required_confidence_training_epochs"] == 6
    assert gate["checks"]["selector_training_is_exactly_six_epochs"] is True
    assert gate["checks"]["confidence_training_is_exactly_six_epochs"] is True
    assert gate["checks"]["confidence_loss_weight_is_exactly_twelve"] is True
    assert gate["criteria"]["required_confidence_loss_weight"] == 12.0
    assert gate["passed"] is True
    assert gate["eligible_for_fixed_followup"] is True
    assert gate["fourth_adaptive_development_screen"] is True
    assert gate["adaptive_development_screen_sequence"] == 4
    assert gate["independent_statistical_validation"] is False
    assert gate["specified_after_observing_dual_head_membership_result"] is True
    assert gate["preregistered_before_selected_candidate_confidence_audit"] is True
    assert "preregistered_before_membership_cv30_audit" not in gate


def test_fourth_gate_fails_closed_when_head_epochs_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(screen, "HEAD_TRAINING_EPOCHS", 7)
    gate = _adaptive_gate(_passing_summary())
    assert gate["checks"]["confidence_training_is_exactly_six_epochs"] is False
    assert gate["applicable"] is False
    assert gate["passed"] is False
    assert gate["eligible_for_fixed_followup"] is False
    assert gate["inapplicable_reason"] is not None
    assert "preregistered_before_membership_cv30_audit" not in gate


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("finite_top1", 689),
        ("equivalent_top1", 137),
        ("direction_correct", 137),
        ("speed_correct", 436),
    ),
)
def test_fourth_gate_keeps_each_plain_selector_floor(
    field: str,
    value: int,
) -> None:
    summary = _passing_summary()
    audit = dict(summary["outer_audit_micro"])  # type: ignore[arg-type]
    audit[field] = value
    summary["outer_audit_micro"] = audit
    assert _adaptive_gate(summary)["passed"] is False


def test_candidate_digest_is_ordered_and_candidate_sensitive() -> None:
    adapter = _adapter()
    first = _episode(adapter)
    second = _episode(adapter)
    second.seed = 18
    candidates = {
        17: torch.tensor([0, 1, 2, 3]),
        18: torch.tensor([4, 5, 6, 7]),
    }
    digest = _candidate_digest([first, second], candidates)
    assert digest == _candidate_digest([first, second], candidates)
    assert digest != _candidate_digest([second, first], candidates)
    changed = dict(candidates)
    changed[18] = torch.tensor([4, 5, 6, 8])
    assert digest != _candidate_digest([first, second], changed)
    with pytest.raises(ValueError, match="lacks an episode tensor"):
        _candidate_digest([first, second], {17: candidates[17]})


def test_verified_source_digest_map_binds_declared_hashes_and_detects_drift(
    tmp_path: Path,
) -> None:
    triplet = tuple(
        tmp_path / name for name in ("episode.npz", "report.json", "manifest.json")
    )
    for index, path in enumerate(triplet):
        path.write_bytes(f"source-{index}".encode("ascii"))
    roles = ("dataset", "report", "manifest")
    record: dict[str, object] = {"declared_hashes_verified": True}
    for role, path in zip(roles, triplet, strict=True):
        record[role] = str(path)
        record[f"{role}_sha256"] = screen.file_sha256(path)

    paths, expected = _verified_source_digest_map([triplet], [record])

    assert _file_digest_snapshot(paths) == expected
    triplet[0].write_bytes(b"changed-after-verification")
    with pytest.raises(ValueError, match="changed during run"):
        _reverify_file_digest_snapshot(paths, expected)


def test_confidence_diagnostics_use_exact_candidate_and_no_correction_negative(
) -> None:
    episode = _episode(_adapter())
    episode.preferred_equivalent_actions.zero_()
    episode.preferred_equivalent_actions[0, 5] = True
    episode.preferred_equivalent_actions[2, 4] = True
    episode.preferred_equivalent_actions[3, 4] = True
    candidates = torch.tensor([5, 7, 3, 4], dtype=torch.int64)
    scores = torch.tensor([0.9, 0.8, 0.3, 0.7])
    membership = torch.zeros((1, episode.decisions, 18))
    membership[0, torch.arange(episode.decisions), candidates] = scores
    mean_membership = membership.mean(dim=0)
    member_finite = torch.ones((1, episode.decisions), dtype=torch.bool)
    row_finite = torch.ones(episode.decisions, dtype=torch.bool)
    prediction = {
        "membership_probabilities": membership,
        "mean_membership_probabilities": mean_membership,
        "membership_member_finite": member_finite,
        "candidates": candidates,
        "action_confidence": mean_membership.gather(
            -1,
            candidates.unsqueeze(-1),
        ).squeeze(-1),
        "selector_all_members_finite": row_finite,
        "action_all_members_finite": row_finite.clone(),
    }

    diagnostics = _selected_candidate_confidence_diagnostics(
        [episode],
        {episode.seed: prediction},
    )

    assert diagnostics["rows"] == diagnostics["finite_rows"] == 3
    assert diagnostics["correction_required_rows"] == 2
    assert diagnostics["no_correction_rows"] == 1
    assert diagnostics["positive_rows"] == 1
    assert diagnostics["positive_rate"] == pytest.approx(1 / 3)
    assert diagnostics["positive_confidence"]["mean"] == pytest.approx(0.9)
    assert diagnostics["negative_confidence"]["mean"] == pytest.approx(0.55)
    assert diagnostics[
        "positive_minus_negative_mean_confidence"
    ] == pytest.approx(0.35)
    assert diagnostics["brier_score"] == pytest.approx(
        ((0.9 - 1.0) ** 2 + 0.8**2 + 0.3**2) / 3
    )


def test_head_fit_rejects_any_candidate_seed_outside_exact_fit_split() -> None:
    adapter = _adapter()
    episode = _episode(adapter)
    frozen = _frozen_ensemble_selected_candidates(
        adapter,
        [episode],
        chunk_length=4,
    )
    frozen[99999] = frozen[episode.seed].clone()
    with pytest.raises(ValueError, match="exactly match fitting seeds"):
        _train_frozen_ensemble_selected_confidence_heads(
            adapter,
            [episode],
            frozen_candidates=frozen,
            seed=20261910,
            epochs=1,
            learning_rate=3e-4,
            weight_decay=1e-3,
            chunk_length=4,
            loss_weight=12.0,
            device="cpu",
        )


def _redirect_formal_campaign_to_tmp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path]:
    output = tmp_path / "formal" / "selected-confidence.json"
    ledger = tmp_path / "formal" / ".selected-confidence.started.json"
    monkeypatch.setattr(screen, "FORMAL_OUTPUT", output)
    monkeypatch.setattr(screen, "FORMAL_CAMPAIGN_LEDGER", ledger)
    return output, ledger


def test_formal_campaign_reservation_is_atomic_and_never_retryable(
    tmp_path: Path,
) -> None:
    output = tmp_path / "formal" / "result.json"
    ledger = tmp_path / "formal" / ".campaign.started.json"
    startup_sha256 = {
        "failure": "a" * 64,
        "experiment_script": "b" * 64,
    }

    _reserve_formal_campaign(
        output=output,
        ledger=ledger,
        startup_sha256=startup_sha256,
        calibration_function_sha256=FIXED_CALIBRATION_FUNCTION_SHA256,
    )

    ledger_bytes = ledger.read_bytes()
    ledger_payload = json.loads(ledger_bytes)
    output_payload = json.loads(output.read_text(encoding="utf-8"))
    assert ledger_payload["status"] == "started_and_consumed_no_retry"
    assert ledger_payload["startup_sha256"] == startup_sha256
    assert output_payload["kind"] == (
        "selected_candidate_confidence_cv30_incomplete_tombstone"
    )
    assert output_payload["campaign_ledger"] == str(ledger)

    output.unlink()
    with pytest.raises(ValueError, match="already started; refusing to retry"):
        _reserve_formal_campaign(
            output=output,
            ledger=ledger,
            startup_sha256=startup_sha256,
            calibration_function_sha256=FIXED_CALIBRATION_FUNCTION_SHA256,
        )
    assert ledger.read_bytes() == ledger_bytes
    assert not output.exists()


def test_main_refuses_existing_output_before_any_input_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    output, ledger = _redirect_formal_campaign_to_tmp(tmp_path, monkeypatch)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("preserve-me\n", encoding="utf-8")
    arguments = ["compare_selected_candidate_confidence_cv30.py"]
    monkeypatch.setattr(sys, "argv", arguments)

    def unexpected_read(*_args: object, **_kwargs: object) -> object:
        pytest.fail("formal input loading must not start for an existing output")

    monkeypatch.setattr(screen, "_file_digest_snapshot", unexpected_read)
    monkeypatch.setattr(screen.balanced, "_read_json", unexpected_read)
    with pytest.raises(ValueError, match="already exists; refusing to overwrite"):
        main()
    assert output.read_text(encoding="utf-8") == "preserve-me\n"
    assert not ledger.exists()


def test_main_rejects_noncanonical_output_before_any_input_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    formal_output, ledger = _redirect_formal_campaign_to_tmp(
        tmp_path,
        monkeypatch,
    )
    output = tmp_path / "different.json"
    monkeypatch.setattr(sys, "argv", [
        "compare_selected_candidate_confidence_cv30.py",
        "--output",
        str(output),
    ])

    def unexpected_read(*_args: object, **_kwargs: object) -> object:
        pytest.fail("noncanonical output must be rejected before input loading")

    monkeypatch.setattr(screen, "_file_digest_snapshot", unexpected_read)
    monkeypatch.setattr(screen.balanced, "_read_json", unexpected_read)
    with pytest.raises(ValueError, match="output is fixed"):
        main()
    assert not output.exists()
    assert not formal_output.exists()
    assert not ledger.exists()


def test_main_requires_exactly_one_cpu_thread_before_any_input_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    output, ledger = _redirect_formal_campaign_to_tmp(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, "argv", [
        "compare_selected_candidate_confidence_cv30.py",
        "--cpu-threads",
        "2",
    ])

    def unexpected_read(*_args: object, **_kwargs: object) -> object:
        pytest.fail("thread-count drift must be rejected before input loading")

    monkeypatch.setattr(screen, "_file_digest_snapshot", unexpected_read)
    monkeypatch.setattr(screen.balanced, "_read_json", unexpected_read)
    with pytest.raises(ValueError, match="exactly 1"):
        main()
    assert not output.exists()
    assert not ledger.exists()


def test_main_atomically_reserves_output_before_first_input_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    output, ledger = _redirect_formal_campaign_to_tmp(tmp_path, monkeypatch)
    monkeypatch.setattr(
        sys,
        "argv",
        ["compare_selected_candidate_confidence_cv30.py"],
    )

    class ExpectedStop(RuntimeError):
        pass

    startup_sha256: dict[str, str] = {}

    def snapshot(paths: object) -> dict[str, str]:
        assert isinstance(paths, dict)
        startup_sha256.update({
            name: f"{index + 1:064x}"
            for index, name in enumerate(paths)
        })
        return dict(startup_sha256)

    def stop_after_reservation(*_args: object, **_kwargs: object) -> object:
        assert output.is_file()
        assert ledger.is_file()
        ledger_payload = json.loads(ledger.read_text(encoding="utf-8"))
        output_payload = json.loads(output.read_text(encoding="utf-8"))
        assert ledger_payload["startup_sha256"] == startup_sha256
        assert ledger_payload["calibration_function_sha256"] == (
            FIXED_CALIBRATION_FUNCTION_SHA256
        )
        assert output_payload["kind"] == (
            "selected_candidate_confidence_cv30_incomplete_tombstone"
        )
        assert output_payload["campaign_ledger"] == str(ledger)
        raise ExpectedStop

    monkeypatch.setattr(screen, "_file_digest_snapshot", snapshot)
    monkeypatch.setattr(screen.balanced, "_read_json", stop_after_reservation)
    with pytest.raises(ExpectedStop):
        main()
    assert output.is_file()
    assert ledger.is_file()
