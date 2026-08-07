from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from experiments import compare_certified_membership_cv30 as balanced
from experiments import compare_dual_head_membership_cv30 as dual_head
from experiments import compare_plain_certified_set_cv30 as plain
from experiments import compare_unbalanced_membership_cv30 as unweighted
from experiments.compare_dual_head_membership_cv30 import (
    AUXILIARY_MEMBERSHIP_CONFIG,
    BALANCED_REFERENCE_SHA256,
    BASE_SEED,
    DEFAULT_OUTPUT,
    DUAL_HEAD_ARM_NAME,
    MEMBERSHIP_CONFIDENCE_LOSS_MODE,
    MEMBERSHIP_CONFIDENCE_LOSS_WEIGHT,
    PLAIN_REFERENCE_SHA256,
    SCREENING_EPOCHS,
    SELECTOR_OBJECTIVE_CONFIG,
    SELECTOR_TRAINING_CONFIG,
    THIRD_ADAPTIVE_SCREEN_CONTEXT,
    UNWEIGHTED_REFERENCE_SHA256,
    _argument_parser,
    _dual_head_adapter_config,
    _dual_head_adaptive_development_gate,
    _implementation_invariants,
    _partition_membership_state,
    _reference_fold_map,
    _read_frozen_reference,
    _reverify_frozen_reference_hashes,
    _reserve_new_output_path,
    _selected_membership_confidence_diagnostics,
    _selector_prediction_digest,
    _selector_training_control_differences,
    _snapshot_state,
    _train_dual_head_member,
    _validate_new_output_path,
    _validate_prior_adaptive_reference,
    _validated_dual_head_prediction,
    main,
)
from experiments.compare_preferred_objectives_cv import EXPECTED_V81_CONFIG


def _summary(**overrides: object) -> dict[str, object]:
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


def _raw_metrics(value: int = 1) -> dict[str, dict[str, int]]:
    return {
        "fit": {"value": value},
        "calibration": {"value": value + 1},
        "audit": {"value": value + 2},
    }


def _plain_reference(*, membership_field: bool | None = None) -> dict[str, object]:
    adapter_config = {
        **EXPECTED_V81_CONFIG,
        "ensemble_size": 1,
    }
    if membership_field is not None:
        adapter_config["per_action_membership_confidence"] = membership_field
    folds = []
    for fold in balanced._fixed_cv30_folds():
        folds.append(
            {
                "fold": fold.index,
                "fit_seeds": list(fold.fit_seeds),
                "calibration_seeds": list(fold.calibration_seeds),
                "audit_seeds": list(fold.audit_seeds),
                "initial_state_sha256": f"initial-{fold.index}",
                "normalization_sha256": f"normalization-{fold.index}",
                "plain": {
                    "trained_state_sha256": f"trained-{fold.index}",
                    "action_branch_state_sha256": f"action-{fold.index}",
                    "non_action_branch_state_sha256": f"shared-{fold.index}",
                    "raw_action_metrics": _raw_metrics(fold.index),
                },
            }
        )
    return {
        "kind": "plain_certified_set_training_only_cv30",
        "training_only": True,
        "deployment_artifact_written": False,
        "formal_deployment_artifact_written": False,
        "deployment_eligible": False,
        "acceptance_claim": False,
        "plain_objective": dict(plain.PLAIN_OBJECTIVE_CONFIG),
        "experiment_config": {
            "epochs": SCREENING_EPOCHS,
            "base_seed": BASE_SEED,
            "device": "cpu",
            "deterministic_algorithms": True,
            "ensemble_size_screening_override": 1,
            "adapter_config": adapter_config,
            "training_config": dict(plain.TRAINING_CONFIG),
        },
        "folds": folds,
    }


def _prior_reference(*, sequence: int, include_sequence: bool = True) -> dict[str, object]:
    balanced_reference = sequence == 1
    result: dict[str, object] = {
        "kind": (
            "certified_membership_adaptive_development_screen_cv30"
            if balanced_reference else
            "unweighted_membership_second_adaptive_development_screen_cv30"
        ),
        "training_only": True,
        "deployment_artifact_written": False,
        "formal_deployment_artifact_written": False,
        "deployment_eligible": False,
        "acceptance_claim": False,
        "adaptive_development_screen": True,
        "independent_statistical_validation": False,
        "specified_after_observing_plain_cv30_negative_result": True,
        "objective_arms": [
            balanced.MEMBERSHIP_ARM_NAME
            if balanced_reference else
            unweighted.UNBALANCED_ARM_NAME
        ],
        "experiment_config": {
            "epochs": SCREENING_EPOCHS,
            "base_seed": BASE_SEED,
        },
        "adaptive_development_gate": {"passed": False},
        "membership_summary": {"outer_audit_micro": {"targets": 690}},
    }
    if include_sequence:
        result["adaptive_development_screen_sequence"] = sequence
    if sequence == 2:
        result[
            "specified_after_observing_balanced_membership_negative_result"
        ] = True
    return result


def _episode_and_prediction() -> tuple[SimpleNamespace, dict[str, torch.Tensor]]:
    decisions = 2
    selector = torch.full((1, decisions, 18), 0.01)
    selector[0, 0, 5] = 0.9
    selector[0, 1, 2] = 0.8
    mean_selector = selector.mean(dim=0)
    candidates = mean_selector.argmax(dim=-1)
    membership = torch.full((1, decisions, 18), 0.1)
    membership[0, 0, 5] = 0.8
    membership[0, 1, 2] = 0.3
    mean_membership = membership.mean(dim=0)
    confidence = mean_membership.gather(
        -1,
        candidates.unsqueeze(-1),
    ).squeeze(-1)
    preferred_set = torch.zeros(decisions, 18, dtype=torch.bool)
    preferred_set[0, 5] = True
    preferred_set[1, 7] = True
    episode = SimpleNamespace(
        seed=7,
        decisions=decisions,
        gate_valid=torch.ones(decisions, dtype=torch.bool),
        gate_targets=torch.ones(decisions),
        preferred_correction_required=torch.ones(decisions, dtype=torch.bool),
        anticipatory_lead_decisions=torch.tensor([4, 6]),
        preferred_action_set=preferred_set,
    )
    member_finite = torch.ones((1, decisions), dtype=torch.bool)
    row_finite = torch.ones(decisions, dtype=torch.bool)
    prediction = {
        "mean_gate": torch.tensor([0.8, 0.9]),
        "minimum_gate": torch.tensor([0.8, 0.9]),
        "action_member_finite": member_finite.clone(),
        "action_probabilities": selector,
        "mean_action_probabilities": mean_selector,
        "selector_all_members_finite": row_finite.clone(),
        "action_all_members_finite": row_finite.clone(),
        "candidates": candidates,
        "agreement": torch.ones(decisions),
        "collision_probabilities": torch.zeros((1, decisions, 18)),
        "minimum_margins": torch.full((1, decisions, 18), 20.0),
        "physical_danger_probabilities": torch.full((1, decisions, 18), 0.1),
        "membership_probabilities": membership,
        "mean_membership_probabilities": mean_membership,
        "membership_member_finite": member_finite.clone(),
        "action_confidence": confidence,
    }
    return episode, prediction


def test_dual_head_config_changes_only_membership_confidence_flag() -> None:
    failure = {"fit_checkpoint": {"adapter_config": EXPECTED_V81_CONFIG}}
    config = _dual_head_adapter_config(failure)
    expected = {
        **EXPECTED_V81_CONFIG,
        "ensemble_size": 1,
        "per_action_membership_confidence": True,
    }
    assert asdict(config) == expected
    assert config.action_logit_mode == "parent_residual_joint"


def test_selector_and_auxiliary_training_controls_are_frozen() -> None:
    assert _selector_training_control_differences() == {}
    assert SELECTOR_OBJECTIVE_CONFIG["schema"] == (
        "preferred_certified_equivalence_set_nll"
    )
    assert SELECTOR_OBJECTIVE_CONFIG["action_logit_mode"] == (
        "parent_residual_joint"
    )
    assert SELECTOR_OBJECTIVE_CONFIG["parent_copy_weight"] == 0.1
    assert SELECTOR_TRAINING_CONFIG["preferred_action_loss_weight"] == 12.0
    assert SELECTOR_TRAINING_CONFIG["parent_copy_weight"] == 0.1
    assert MEMBERSHIP_CONFIDENCE_LOSS_WEIGHT == 12.0
    assert MEMBERSHIP_CONFIDENCE_LOSS_MODE == "unweighted"
    assert AUXILIARY_MEMBERSHIP_CONFIG["head_input"] == (
        "detached_selector_recurrent"
    )
    assert AUXILIARY_MEMBERSHIP_CONFIG[
        "selector_gradient_from_membership_loss"
    ] is False
    assert AUXILIARY_MEMBERSHIP_CONFIG["optimizer"] == "independent_adamw"
    assert AUXILIARY_MEMBERSHIP_CONFIG["learning_rate"] == 3e-4
    assert AUXILIARY_MEMBERSHIP_CONFIG["weight_decay"] == 1e-3
    assert AUXILIARY_MEMBERSHIP_CONFIG["gradient_clip_max_norm"] == 5.0


def test_protocol_freezes_sources_folds_seed_epochs_and_ensemble() -> None:
    assert BASE_SEED == 20260901
    assert SCREENING_EPOCHS == 6
    assert DUAL_HEAD_ARM_NAME == "plain_selector_auxiliary_membership_confidence"
    folds = balanced._fixed_cv30_folds()
    assert [
        (len(fold.fit_seeds), len(fold.calibration_seeds), len(fold.audit_seeds))
        for fold in folds
    ] == [(16, 4, 10)] * 3
    assert sorted(seed for fold in folds for seed in fold.audit_seeds) == sorted(
        balanced.ALL_TRAINING_SEEDS
    )
    assert THIRD_ADAPTIVE_SCREEN_CONTEXT["sequence"] == 3
    assert THIRD_ADAPTIVE_SCREEN_CONTEXT[
        "specified_after_observing_unweighted_membership_result"
    ] is True


def test_cli_exposes_no_epoch_seed_retry_mode_or_weight_controls() -> None:
    destinations = {action.dest for action in _argument_parser()._actions}
    assert destinations == {
        "help",
        "failure",
        "expansion_inventory",
        "parent",
        "plain_reference",
        "balanced_reference",
        "unweighted_reference",
        "output",
        "cpu_threads",
    }
    for prohibited in (
        "epochs",
        "seed",
        "retry",
        "mode",
        "membership_confidence_loss_mode",
        "membership_confidence_loss_weight",
    ):
        assert prohibited not in destinations


def test_train_wrapper_forwards_exact_plain_and_auxiliary_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_train(*args: object, **kwargs: object) -> list[dict[str, float]]:
        captured["args"] = args
        captured.update(kwargs)
        return [{"epoch": 6.0, "mean_chunk_loss": 1.0}]

    monkeypatch.setattr(balanced, "_train_member", fake_train)
    adapter = object()
    fit: list[object] = []
    collision = torch.ones(18)
    physical = torch.ones(18)
    result = _train_dual_head_member(
        adapter,  # type: ignore[arg-type]
        fit,
        member_seed=20261910,
        collision_weights=collision,
        physical_weights=physical,
    )

    assert captured["args"] == (adapter, 0, fit)
    assert captured["seed"] == 20261910
    assert captured["epochs"] == 6
    assert captured["preferred_action_loss_weight"] == 12.0
    assert captured["preferred_action_uniform_loss_weight"] == 0.0
    assert captured["preferred_action_tiebreak_loss_weight"] == 0.0
    assert captured["preferred_action_rank_loss_weight"] == 0.0
    assert captured["parent_copy_weight"] == 0.1
    assert captured["membership_confidence_loss_weight"] == 12.0
    assert captured["membership_confidence_loss_mode"] == "unweighted"
    assert captured["collision_positive_weights"] is collision
    assert captured["physical_danger_positive_weights"] is physical
    assert captured["device"] == "cpu"
    assert result == [{"epoch": 6.0, "mean_chunk_loss": 1.0}]


def test_plain_reference_accepts_legacy_missing_false_config_field() -> None:
    legacy = _reference_fold_map(_plain_reference(membership_field=None))
    explicit = _reference_fold_map(_plain_reference(membership_field=False))
    assert set(legacy) == set(explicit) == {0, 1, 2}
    with pytest.raises(ValueError, match="unexpectedly has a membership head"):
        _reference_fold_map(_plain_reference(membership_field=True))


def test_balanced_legacy_reference_infers_only_exact_sequence_one() -> None:
    legacy = _prior_reference(sequence=1, include_sequence=False)
    validated = _validate_prior_adaptive_reference(
        legacy,
        kind="certified_membership_adaptive_development_screen_cv30",
        sequence=1,
    )
    assert validated["adaptive_development_screen_sequence"] == 1
    assert validated["source_sequence_field"] is None
    assert validated[
        "legacy_sequence_inferred_from_exact_kind_and_provenance"
    ] is True

    forged = dict(legacy)
    forged["objective_arms"] = ["different"]
    with pytest.raises(ValueError, match="objective arm differs"):
        _validate_prior_adaptive_reference(
            forged,
            kind="certified_membership_adaptive_development_screen_cv30",
            sequence=1,
        )


def test_unweighted_reference_requires_explicit_sequence_two() -> None:
    reference = _prior_reference(sequence=2)
    validated = _validate_prior_adaptive_reference(
        reference,
        kind="unweighted_membership_second_adaptive_development_screen_cv30",
        sequence=2,
    )
    assert validated["source_sequence_field"] == 2
    assert validated[
        "legacy_sequence_inferred_from_exact_kind_and_provenance"
    ] is False
    reference.pop("adaptive_development_screen_sequence")
    with pytest.raises(ValueError, match="sequence is missing"):
        _validate_prior_adaptive_reference(
            reference,
            kind="unweighted_membership_second_adaptive_development_screen_cv30",
            sequence=2,
        )


def test_unweighted_reference_requires_balanced_result_provenance() -> None:
    reference = _prior_reference(sequence=2)
    reference.pop("specified_after_observing_balanced_membership_negative_result")
    with pytest.raises(ValueError, match="lacks balanced-result provenance"):
        _validate_prior_adaptive_reference(
            reference,
            kind="unweighted_membership_second_adaptive_development_screen_cv30",
            sequence=2,
        )


@pytest.mark.parametrize("sequence", (1, 2))
@pytest.mark.parametrize(
    "field",
    (
        "deployment_artifact_written",
        "formal_deployment_artifact_written",
        "deployment_eligible",
        "acceptance_claim",
    ),
)
def test_prior_references_reject_deployment_or_acceptance_claims(
    sequence: int,
    field: str,
) -> None:
    reference = _prior_reference(sequence=sequence)
    reference[field] = True
    with pytest.raises(ValueError, match="deployment or acceptance claim"):
        _validate_prior_adaptive_reference(
            reference,
            kind=str(reference["kind"]),
            sequence=sequence,
        )


@pytest.mark.parametrize("sequence", (1, 2))
@pytest.mark.parametrize(
    "field",
    (
        "deployment_artifact_written",
        "formal_deployment_artifact_written",
        "deployment_eligible",
        "acceptance_claim",
    ),
)
def test_prior_references_require_explicit_false_deployment_fields(
    sequence: int,
    field: str,
) -> None:
    reference = _prior_reference(sequence=sequence)
    reference.pop(field)
    with pytest.raises(ValueError, match="deployment or acceptance claim"):
        _validate_prior_adaptive_reference(
            reference,
            kind=str(reference["kind"]),
            sequence=sequence,
        )


@pytest.mark.parametrize("sequence", (1, 2))
@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("adaptive_development_screen", False, "not an adaptive development screen"),
        ("independent_statistical_validation", True, "claims independent validation"),
        ("objective_arms", ["different"], "objective arm differs"),
        (
            "specified_after_observing_plain_cv30_negative_result",
            False,
            "lacks observed-result provenance",
        ),
    ),
)
def test_prior_references_reject_semantic_drift(
    sequence: int,
    field: str,
    value: object,
    message: str,
) -> None:
    reference = _prior_reference(sequence=sequence)
    reference[field] = value
    with pytest.raises(ValueError, match=message):
        _validate_prior_adaptive_reference(
            reference,
            kind=str(reference["kind"]),
            sequence=sequence,
        )


def test_frozen_reference_hashes_match_the_committed_e6_artifacts() -> None:
    assert PLAIN_REFERENCE_SHA256 == (
        "8f9a0106e26b612657345be14e9e1c0c56f137f4a02d579ecbf6ad10451ed206"
    )
    assert BALANCED_REFERENCE_SHA256 == (
        "ddff204e80ce0b646bf1dbe3285b7c7b774a858b253c11dbff666e0754ae1184"
    )
    assert UNWEIGHTED_REFERENCE_SHA256 == (
        "da079dc5a0a234e5c545f27570bd459fc8c457031abe186d553fa9c568d223ec"
    )


def test_frozen_reference_hash_is_checked_before_content_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(dual_head, "file_sha256", lambda _path: "wrong")

    def unexpected_read(_path: Path) -> object:
        pytest.fail("reference content must not be read after a pre-read hash failure")

    monkeypatch.setattr(balanced, "_read_json", unexpected_read)
    with pytest.raises(ValueError, match="hash differs from the frozen e6 artifact"):
        _read_frozen_reference(
            Path("reference.json"),
            expected_sha256=PLAIN_REFERENCE_SHA256,
            label="plain",
        )


def test_frozen_reference_hash_is_rechecked_after_content_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hashes = iter((PLAIN_REFERENCE_SHA256, "changed"))
    monkeypatch.setattr(dual_head, "file_sha256", lambda _path: next(hashes))
    monkeypatch.setattr(balanced, "_read_json", lambda _path: {"kind": "plain"})
    with pytest.raises(ValueError, match="changed while being read"):
        _read_frozen_reference(
            Path("reference.json"),
            expected_sha256=PLAIN_REFERENCE_SHA256,
            label="plain",
        )


def test_frozen_reference_hashes_are_reverified_after_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = {
        Path("plain.json"): PLAIN_REFERENCE_SHA256,
        Path("balanced.json"): BALANCED_REFERENCE_SHA256,
        Path("unweighted.json"): UNWEIGHTED_REFERENCE_SHA256,
    }
    monkeypatch.setattr(
        dual_head,
        "file_sha256",
        lambda path: expected[path],
    )
    assert _reverify_frozen_reference_hashes(
        plain_path=Path("plain.json"),
        balanced_path=Path("balanced.json"),
        unweighted_path=Path("unweighted.json"),
    ) == {
        "plain": PLAIN_REFERENCE_SHA256,
        "balanced": BALANCED_REFERENCE_SHA256,
        "unweighted": UNWEIGHTED_REFERENCE_SHA256,
    }


@pytest.mark.parametrize(
    ("changed_path", "label"),
    (
        (Path("plain.json"), "plain"),
        (Path("balanced.json"), "balanced membership"),
        (Path("unweighted.json"), "unweighted membership"),
    ),
)
def test_post_training_reference_drift_is_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    changed_path: Path,
    label: str,
) -> None:
    expected = {
        Path("plain.json"): PLAIN_REFERENCE_SHA256,
        Path("balanced.json"): BALANCED_REFERENCE_SHA256,
        Path("unweighted.json"): UNWEIGHTED_REFERENCE_SHA256,
    }
    monkeypatch.setattr(
        dual_head,
        "file_sha256",
        lambda path: "changed" if path == changed_path else expected[path],
    )
    with pytest.raises(ValueError, match=f"{label} reference changed during"):
        _reverify_frozen_reference_hashes(
            plain_path=Path("plain.json"),
            balanced_path=Path("balanced.json"),
            unweighted_path=Path("unweighted.json"),
        )


def test_state_snapshot_and_membership_partition_do_not_alias_training_state() -> None:
    state = {
        "members.0.action_head.weight": torch.tensor([1.0]),
        "members.0.membership_head.weight": torch.tensor([2.0]),
        "feature_mean": torch.tensor([3.0]),
    }
    snapshot = _snapshot_state(state)
    membership, non_membership = _partition_membership_state(snapshot)
    state["members.0.action_head.weight"].add_(10.0)
    state["members.0.membership_head.weight"].add_(10.0)
    assert snapshot["members.0.action_head.weight"].tolist() == [1.0]
    assert snapshot["members.0.membership_head.weight"].tolist() == [2.0]
    assert set(membership) == {"members.0.membership_head.weight"}
    assert set(non_membership) == {
        "members.0.action_head.weight",
        "feature_mean",
    }


def test_implementation_invariants_require_all_plain_hashes_and_raw_metrics() -> None:
    fold = balanced._fixed_cv30_folds()[0]
    reference = _plain_reference()["folds"][0]
    result = _implementation_invariants(
        fold,
        reference,  # type: ignore[arg-type]
        initial_non_membership_state_sha256="initial-0",
        normalization_sha256="normalization-0",
        trained_non_membership_state_sha256="trained-0",
        selector_action_branch_state_sha256="action-0",
        shared_non_action_branch_state_sha256="shared-0",
        raw_action_metrics=_raw_metrics(0),
        selector_prediction_sha256="prediction-digest",
    )
    assert result["all_passed"] is True
    assert all(result["checks"].values())
    assert result["selector_prediction_sha256"] == "prediction-digest"

    with pytest.raises(AssertionError, match="audit_raw_selector_metrics_equal"):
        _implementation_invariants(
            fold,
            reference,  # type: ignore[arg-type]
            initial_non_membership_state_sha256="initial-0",
            normalization_sha256="normalization-0",
            trained_non_membership_state_sha256="trained-0",
            selector_action_branch_state_sha256="action-0",
            shared_non_action_branch_state_sha256="shared-0",
            raw_action_metrics={
                **_raw_metrics(0),
                "audit": {"value": 999},
            },
            selector_prediction_sha256="prediction-digest",
        )


def test_selector_prediction_digest_excludes_auxiliary_membership_values() -> None:
    episode, prediction = _episode_and_prediction()
    baseline = _selector_prediction_digest([episode], {episode.seed: prediction})
    changed_membership = dict(prediction)
    changed_membership["membership_probabilities"] = torch.zeros_like(
        prediction["membership_probabilities"]
    )
    changed_membership["mean_membership_probabilities"] = torch.zeros_like(
        prediction["mean_membership_probabilities"]
    )
    changed_membership["action_confidence"] = torch.zeros_like(
        prediction["action_confidence"]
    )
    assert _selector_prediction_digest(
        [episode],
        {episode.seed: changed_membership},
    ) == baseline

    changed_selector = dict(prediction)
    changed_selector["action_probabilities"] = (
        prediction["action_probabilities"] + 0.01
    )
    assert _selector_prediction_digest(
        [episode],
        {episode.seed: changed_selector},
    ) != baseline


def test_selected_membership_confidence_uses_selector_candidate() -> None:
    episode, prediction = _episode_and_prediction()
    selected, candidates, finite = _validated_dual_head_prediction(
        episode,
        prediction,
    )
    assert candidates.tolist() == [5, 2]
    assert selected.tolist() == pytest.approx([0.8, 0.3])
    assert finite.tolist() == [True, True]

    diagnostics = _selected_membership_confidence_diagnostics(
        [episode],
        {episode.seed: prediction},
    )
    early = diagnostics["early_correction_required_4_10"]
    assert early["rows"] == early["finite_rows"] == 2
    assert early["selected_certified_rows"] == 1
    assert early["selected_certified_rate"] == pytest.approx(0.5)
    assert early["selected_certified_confidence"]["mean"] == pytest.approx(0.8)
    assert early["selected_rejected_confidence"]["mean"] == pytest.approx(0.3)
    assert early["certified_minus_rejected_mean_confidence"] == pytest.approx(0.5)


def test_dual_head_prediction_rejects_confidence_or_finite_contract_drift() -> None:
    episode, prediction = _episode_and_prediction()
    bad_confidence = dict(prediction)
    bad_confidence["action_confidence"] = torch.tensor([0.7, 0.3])
    with pytest.raises(ValueError, match="selected membership probability"):
        _validated_dual_head_prediction(episode, bad_confidence)

    bad_finite = dict(prediction)
    bad_finite["action_all_members_finite"] = torch.tensor([False, True])
    with pytest.raises(ValueError, match="selector AND membership"):
        _validated_dual_head_prediction(episode, bad_finite)


def test_third_adaptive_gate_reuses_every_frozen_boundary() -> None:
    base = balanced._adaptive_development_gate(_summary())
    gate = _dual_head_adaptive_development_gate(_summary())
    assert gate["criteria"] == base["criteria"]
    assert gate["checks"] == base["checks"]
    assert gate["passed"] is True
    assert gate["eligible_for_fixed_followup"] is True
    assert gate["third_adaptive_development_screen"] is True
    assert gate["adaptive_development_screen_sequence"] == 3
    assert gate["independent_statistical_validation"] is False
    assert gate["preregistered_before_dual_head_membership_cv30_audit"] is True


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("finite_top1", 689),
        ("equivalent_top1", 137),
        ("direction_correct", 137),
        ("speed_correct", 436),
    ),
)
def test_third_adaptive_gate_keeps_each_fixed_floor(
    field: str,
    value: int,
) -> None:
    audit = dict(_summary()["outer_audit_micro"])
    audit[field] = value
    gate = _dual_head_adaptive_development_gate(
        _summary(outer_audit_micro=audit)
    )
    assert gate["passed"] is False


def test_third_adaptive_gate_keeps_calibration_runtime_and_e6_requirements() -> None:
    assert _dual_head_adaptive_development_gate(
        _summary(calibration_successful_folds=1)
    )["passed"] is False
    assert _dual_head_adaptive_development_gate(
        _summary(audit_runtime_eligible_folds=1)
    )["passed"] is False
    assert _dual_head_adaptive_development_gate(
        _summary(
            calibration_successful_folds=3,
            audit_runtime_eligible_folds=2,
            calibrated_audit_runtime_ineligible_folds=[2],
        )
    )["passed"] is False
    assert _dual_head_adaptive_development_gate(
        _summary(),
        epochs=7,
    )["applicable"] is False


@pytest.mark.parametrize("use_default", (True, False))
def test_main_refuses_existing_output_before_loading_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    use_default: bool,
) -> None:
    monkeypatch.chdir(tmp_path)
    output = tmp_path / (DEFAULT_OUTPUT if use_default else "explicit.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("preserve-me\n", encoding="utf-8")
    arguments = ["compare_dual_head_membership_cv30.py"]
    if not use_default:
        arguments.extend(("--output", str(output)))
    monkeypatch.setattr(sys, "argv", arguments)

    def unexpected_read(*_args: object, **_kwargs: object) -> object:
        pytest.fail("input loading must not begin when output already exists")

    monkeypatch.setattr(balanced, "_read_json", unexpected_read)
    with pytest.raises(ValueError, match="already exists; refusing to overwrite"):
        main()
    assert output.read_text(encoding="utf-8") == "preserve-me\n"


def test_new_output_validation_and_reservation_are_fail_closed(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.json"
    source.write_text("protected\n", encoding="utf-8")
    with pytest.raises(ValueError, match="output must not overwrite protected input"):
        _validate_new_output_path(source, [source])
    assert source.read_text(encoding="utf-8") == "protected\n"

    output = tmp_path / "nested" / "new.json"
    _reserve_new_output_path(output)
    assert output.is_file()
    with pytest.raises(ValueError, match="already exists; refusing to overwrite"):
        _reserve_new_output_path(output)
