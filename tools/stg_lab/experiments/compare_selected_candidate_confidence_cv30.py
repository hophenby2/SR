"""Fourth adaptive CV30 screen for frozen selected-candidate confidence.

The plain selector is trained first and then frozen.  A separate confidence
head learns only whether the selector's immutable ensemble-global candidate is
certified on the early future-onset rows consumed by runtime calibration.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch

from stg_lab import native_dataset as native_dataset_module
from stg_lab import policy as policy_module
from stg_lab import provenance as provenance_module
from stg_lab import residual_adapter as residual_adapter_module
from stg_lab import rollout as rollout_module
from stg_lab import training as training_module
from stg_lab.provenance import file_sha256
from stg_lab.residual_adapter import ResidualCorrectionAdapter
from stg_lab.training import load_checkpoint

if __package__:
    from . import compare_dual_head_membership_cv30 as dual
    from .train_temporal_residual_adapter import (
        _frozen_ensemble_selected_candidates,
        _selected_candidate_confidence_targets,
        _train_frozen_ensemble_selected_confidence_heads,
    )
else:  # pragma: no cover - exercised by the real script invocation
    import compare_dual_head_membership_cv30 as dual
    from train_temporal_residual_adapter import (
        _frozen_ensemble_selected_candidates,
        _selected_candidate_confidence_targets,
        _train_frozen_ensemble_selected_confidence_heads,
    )


balanced = dual.balanced
plain = dual.plain
unweighted = dual.unweighted

ARM_NAME = "frozen_plain_selector_early_selected_candidate_confidence"
ARM_REPORT_KEY = "selected_candidate_confidence"
SCREENING_EPOCHS = 6
HEAD_TRAINING_EPOCHS = 6
BASE_SEED = 20260901
HEAD_LOSS_WEIGHT = 12.0

PLAIN_REFERENCE_SHA256 = dual.PLAIN_REFERENCE_SHA256
BALANCED_REFERENCE_SHA256 = dual.BALANCED_REFERENCE_SHA256
UNWEIGHTED_REFERENCE_SHA256 = dual.UNWEIGHTED_REFERENCE_SHA256
DUAL_REFERENCE_SHA256 = (
    "88adfb80f8aa09a7551be3bef363abb994e4580581c31de64ca3dfac1c179a4a"
)

DEFAULT_PLAIN_REFERENCE = dual.DEFAULT_PLAIN_REFERENCE
DEFAULT_BALANCED_REFERENCE = dual.DEFAULT_BALANCED_REFERENCE
DEFAULT_UNWEIGHTED_REFERENCE = dual.DEFAULT_UNWEIGHTED_REFERENCE
DEFAULT_DUAL_REFERENCE = dual.DEFAULT_OUTPUT
DEFAULT_OUTPUT = Path(
    "artifacts/policy-humanlike-highres-okuu3-selected-candidate-confidence-"
    "cv30-e6.json"
)
STG_LAB_ROOT = Path(__file__).resolve().parent.parent
FORMAL_OUTPUT = (STG_LAB_ROOT / DEFAULT_OUTPUT).resolve()
FORMAL_CAMPAIGN_LEDGER = (
    STG_LAB_ROOT
    / "artifacts/.policy-humanlike-highres-okuu3-selected-candidate-"
    "confidence-cv30-e6.started.json"
).resolve()
FIXED_CALIBRATION_FUNCTION_SHA256 = (
    "540ce805226884e73c44d785740b96e16a4531a60b3a949e2934dcd36d44caa2"
)

SELECTOR_TRAINING_CONFIG = dict(dual.SELECTOR_TRAINING_CONFIG)
SELECTED_CONFIDENCE_CONFIG = {
    "schema": "frozen_ensemble_selected_candidate_early_unweighted_bce_v1",
    "selector_phase_epochs": SCREENING_EPOCHS,
    "confidence_phase_epochs": HEAD_TRAINING_EPOCHS,
    "target_rows": (
        "fit_only_gate_valid_positive_anticipatory_lead_4_through_10"
    ),
    "no_correction_rows": "included_as_negative",
    "candidate": "argmax_mean_frozen_selector_softmax",
    "candidate_device": "deterministic_cpu_float32",
    "target": "preferred_equivalent_actions_at_frozen_candidate",
    "loss": "selected_scalar_unweighted_binary_cross_entropy",
    "loss_weight": HEAD_LOSS_WEIGHT,
    "head_input": "cached_detached_frozen_action_recurrent",
    "selector_gradient_from_confidence_loss": False,
    "optimizer": "independent_adamw_per_membership_head",
    "learning_rate": SELECTOR_TRAINING_CONFIG["learning_rate"],
    "weight_decay": SELECTOR_TRAINING_CONFIG["weight_decay"],
    "fixed_label_batch_size": SELECTOR_TRAINING_CONFIG["chunk_length"],
    "gradient_clip": "independent_membership_head_group",
    "gradient_clip_max_norm": balanced.GRADIENT_CLIP_MAX_NORM,
    "retry": False,
}


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fourth adaptive CV30 screen for post-fit selected-candidate "
            "confidence."
        )
    )
    parser.add_argument("--failure", type=Path, default=balanced.DEFAULT_FAILURE)
    parser.add_argument(
        "--expansion-inventory",
        type=Path,
        default=balanced.DEFAULT_EXPANSION_INVENTORY,
    )
    parser.add_argument("--parent", type=Path, default=balanced.DEFAULT_PARENT)
    parser.add_argument(
        "--plain-reference", type=Path, default=DEFAULT_PLAIN_REFERENCE
    )
    parser.add_argument(
        "--balanced-reference", type=Path, default=DEFAULT_BALANCED_REFERENCE
    )
    parser.add_argument(
        "--unweighted-reference", type=Path, default=DEFAULT_UNWEIGHTED_REFERENCE
    )
    parser.add_argument(
        "--dual-reference", type=Path, default=DEFAULT_DUAL_REFERENCE
    )
    parser.add_argument("--output", type=Path, default=FORMAL_OUTPUT)
    parser.add_argument("--cpu-threads", type=int, default=1)
    return parser


def _calibration_function_sha256() -> str:
    source = inspect.getsource(balanced._calibrate).encode("utf-8")
    return hashlib.sha256(source).hexdigest()


def _required_source_path(value: Any, *, label: str) -> Path:
    source = inspect.getsourcefile(value)
    if not isinstance(source, str):
        raise ValueError(f"cannot locate implementation source: {label}")
    return Path(source).resolve()


def _file_digest_snapshot(paths: Mapping[str, Path]) -> dict[str, str]:
    return {name: file_sha256(path) for name, path in paths.items()}


def _reverify_file_digest_snapshot(
    paths: Mapping[str, Path],
    expected: Mapping[str, str],
) -> None:
    if set(paths) != set(expected):
        raise AssertionError("formal integrity inventory changed")
    for name, path in paths.items():
        if file_sha256(path) != expected[name]:
            raise ValueError(f"formal input or code changed during run: {name}")


def _verified_source_digest_map(
    triplets: Sequence[tuple[Path, Path, Path]],
    provenance: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Path], dict[str, str]]:
    if len(triplets) != len(provenance):
        raise ValueError("verified source provenance does not align with triplets")
    paths: dict[str, Path] = {}
    digests: dict[str, str] = {}
    for index, (triplet, record) in enumerate(
        zip(triplets, provenance, strict=True)
    ):
        if record.get("declared_hashes_verified") is not True:
            raise ValueError("source provenance lacks declared-hash verification")
        for role, path in zip(
            ("dataset", "report", "manifest"),
            triplet,
            strict=True,
        ):
            if Path(str(record.get(role))).resolve() != path.resolve():
                raise ValueError("source provenance path differs from triplet")
            digest = record.get(f"{role}_sha256")
            if not isinstance(digest, str) or len(digest) != 64:
                raise ValueError("source provenance lacks a SHA-256 digest")
            name = f"training_source_{index:02d}_{role}"
            paths[name] = path
            digests[name] = digest
    return paths, digests


def _reserve_formal_campaign(
    *,
    output: Path,
    ledger: Path,
    startup_sha256: Mapping[str, str],
    calibration_function_sha256: str,
) -> None:
    if output.resolve() == ledger.resolve():
        raise ValueError("formal output and campaign ledger must differ")
    ledger.parent.mkdir(parents=True, exist_ok=True)
    tombstone = {
        "schema_version": 1,
        "kind": "selected_candidate_confidence_cv30_campaign_started",
        "status": "started_and_consumed_no_retry",
        "adaptive_development_screen_sequence": 4,
        "selector_epochs": SCREENING_EPOCHS,
        "confidence_epochs": HEAD_TRAINING_EPOCHS,
        "base_seed": BASE_SEED,
        "cpu_threads": 1,
        "output": str(output),
        "startup_sha256": dict(startup_sha256),
        "calibration_function_sha256": calibration_function_sha256,
    }
    try:
        with ledger.open("x", encoding="utf-8") as handle:
            json.dump(tombstone, handle, indent=2, allow_nan=False)
            handle.write("\n")
    except FileExistsError as error:
        raise ValueError(
            "selected-confidence CV30 campaign was already started; refusing "
            f"to retry: {ledger}"
        ) from error

    dual._reserve_new_output_path(output)
    balanced._write_json_atomic(output, {
        **tombstone,
        "kind": "selected_candidate_confidence_cv30_incomplete_tombstone",
        "campaign_ledger": str(ledger),
        "campaign_ledger_sha256": file_sha256(ledger),
    })


def _candidate_digest(
    episodes: Sequence[Any],
    candidates: Mapping[int, torch.Tensor],
) -> str:
    digest = hashlib.sha256()
    for episode in episodes:
        values = candidates.get(episode.seed)
        if not isinstance(values, torch.Tensor):
            raise ValueError("frozen candidate map lacks an episode tensor")
        if values.shape != (episode.decisions,) or values.dtype != torch.int64:
            raise ValueError("frozen candidate tensor is invalid")
        digest.update(str(int(episode.seed)).encode("ascii"))
        dual._update_tensor_digest(digest, "candidates", values)
    return digest.hexdigest()


def _validate_dual_reference(
    reference: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[int, Mapping[str, Any]]]:
    if reference.get("kind") != (
        "dual_head_membership_third_adaptive_development_screen_cv30"
    ):
        raise ValueError("dual reference has the wrong artifact kind")
    expected_boolean_flags = {
        "training_only": True,
        "deployment_artifact_written": False,
        "formal_deployment_artifact_written": False,
        "deployment_eligible": False,
        "acceptance_claim": False,
        "adaptive_development_screen": True,
        "independent_statistical_validation": False,
    }
    for name, expected in expected_boolean_flags.items():
        if reference.get(name) is not expected:
            raise ValueError(f"dual reference flag differs: {name}")
    if reference.get("adaptive_development_screen_sequence") != 3:
        raise ValueError("dual reference adaptive screen sequence differs")
    if reference.get("objective_arms") != [dual.DUAL_HEAD_ARM_NAME]:
        raise ValueError("dual reference objective arm differs")
    experiment = reference.get("experiment_config")
    if not isinstance(experiment, Mapping):
        raise ValueError("dual reference lacks experiment configuration")
    if (
        experiment.get("epochs") != SCREENING_EPOCHS
        or experiment.get("base_seed") != BASE_SEED
        or experiment.get("device") != "cpu"
        or experiment.get("deterministic_algorithms") is not True
        or experiment.get("ensemble_size_screening_override") != 1
    ):
        raise ValueError("dual reference protocol differs")
    gate = reference.get("adaptive_development_gate")
    summary = reference.get("dual_head_summary")
    if not isinstance(gate, Mapping) or not isinstance(summary, Mapping):
        raise ValueError("dual reference lacks its result or gate")
    if gate.get("passed") is not False or gate.get(
        "eligible_for_fixed_followup"
    ) is not False:
        raise ValueError("dual reference unexpectedly passed its fixed gate")
    folds = reference.get("folds")
    if not isinstance(folds, list) or len(folds) != 3:
        raise ValueError("dual reference must contain three folds")
    fold_map: dict[int, Mapping[str, Any]] = {}
    for expected, raw in zip(balanced._fixed_cv30_folds(), folds, strict=True):
        if not isinstance(raw, Mapping) or raw.get("fold") != expected.index:
            raise ValueError("dual reference fold order differs")
        for name, seeds in (
            ("fit_seeds", expected.fit_seeds),
            ("calibration_seeds", expected.calibration_seeds),
            ("audit_seeds", expected.audit_seeds),
        ):
            if raw.get(name) != list(seeds):
                raise ValueError(
                    f"dual reference {name} differs in fold {expected.index}"
                )
        expected_fold_seed = BASE_SEED + expected.index * 100_003
        if (
            raw.get("fold_seed") != expected_fold_seed
            or raw.get("member_seed") != expected_fold_seed + 1_009
        ):
            raise ValueError("dual reference fold or member seed differs")
        arm = raw.get("dual_head")
        if not isinstance(arm, Mapping):
            raise ValueError("dual reference fold lacks its arm")
        invariants = arm.get("implementation_invariants")
        if not isinstance(invariants, Mapping) or invariants.get(
            "all_passed"
        ) is not True:
            raise ValueError("dual reference selector invariant did not pass")
        initial_membership_sha256 = raw.get(
            "initial_membership_head_state_sha256"
        )
        if (
            not isinstance(initial_membership_sha256, str)
            or len(initial_membership_sha256) != 64
        ):
            raise ValueError("dual reference lacks initial membership state hash")
        fold_map[expected.index] = raw
    return ({
        "adaptive_development_screen_sequence": 3,
        "gate_passed": False,
        "outer_audit_micro": summary.get("outer_audit_micro"),
    }, fold_map)


def _train_plain_selector(
    adapter: ResidualCorrectionAdapter,
    fit: list[Any],
    *,
    member_seed: int,
    collision_weights: torch.Tensor,
    physical_weights: torch.Tensor,
) -> list[dict[str, float]]:
    config = SELECTOR_TRAINING_CONFIG
    torch.manual_seed(member_seed)
    return balanced._train_member(
        adapter,
        0,
        fit,
        seed=member_seed,
        epochs=SCREENING_EPOCHS,
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        chunk_length=config["chunk_length"],
        gate_positive_weight=config["gate_positive_weight"],
        action_loss_weight=config["action_loss_weight"],
        preferred_action_loss_weight=config["preferred_action_loss_weight"],
        preferred_action_uniform_loss_weight=config[
            "preferred_action_uniform_loss_weight"
        ],
        preferred_action_tiebreak_loss_weight=config[
            "preferred_action_tiebreak_loss_weight"
        ],
        preferred_action_rank_loss_weight=config[
            "preferred_action_rank_loss_weight"
        ],
        preferred_action_rank_margin=config["preferred_action_rank_margin"],
        safety_candidate_loss_weight=config["safety_candidate_loss_weight"],
        parent_copy_weight=config["parent_copy_weight"],
        collision_loss_weight=config["collision_loss_weight"],
        minimum_margin_loss_weight=config["minimum_margin_loss_weight"],
        physical_danger_loss_weight=config["physical_danger_loss_weight"],
        collision_positive_weights=collision_weights,
        physical_danger_positive_weights=physical_weights,
        all_collision_row_weight=config["all_collision_row_weight"],
        episode_bootstrap=config["episode_bootstrap"],
        device="cpu",
    )


def _copy_plain_state_into_dual(
    plain_adapter: ResidualCorrectionAdapter,
    adapter: ResidualCorrectionAdapter,
) -> None:
    expected_missing = {
        name for name in adapter.state_dict()
        if ".membership_head." in name
    }
    membership_before, _base_before = dual._partition_membership_state(
        dual._snapshot_state(adapter.state_dict())
    )
    restored = adapter.load_state_dict(plain_adapter.state_dict(), strict=False)
    if set(restored.missing_keys) != expected_missing or restored.unexpected_keys:
        raise AssertionError("plain-to-dual state transfer has unexpected keys")
    membership_after, _base_after = dual._partition_membership_state(
        adapter.state_dict()
    )
    if set(membership_before) != set(membership_after) or any(
        not torch.equal(value, membership_after[name])
        for name, value in membership_before.items()
    ):
        raise AssertionError("plain state transfer changed membership initialization")


def _selected_candidate_confidence_diagnostics(
    episodes: Sequence[Any],
    predictions: Mapping[int, Mapping[str, torch.Tensor]],
) -> dict[str, Any]:
    """Describe confidence using the exact labels used by the second phase."""

    scores: list[float] = []
    labels: list[bool] = []
    rows = 0
    finite_rows = 0
    correction_required_rows = 0
    no_correction_rows = 0
    for episode in episodes:
        selected, candidates, finite = dual._validated_dual_head_prediction(
            episode,
            predictions[episode.seed],
        )
        mask, target = _selected_candidate_confidence_targets(
            episode,
            candidates,
        )
        rows += int(mask.sum())
        correction_required_rows += int(
            (mask & episode.preferred_correction_required).sum()
        )
        no_correction_rows += int(
            (mask & ~episode.preferred_correction_required).sum()
        )
        valid = mask & finite
        finite_rows += int(valid.sum())
        scores.extend(float(value) for value in selected[valid].tolist())
        labels.extend(bool(value) for value in target[valid].tolist())

    positive = [
        score for score, label in zip(scores, labels, strict=True) if label
    ]
    negative = [
        score for score, label in zip(scores, labels, strict=True) if not label
    ]
    brier = (
        sum(
            (score - float(label)) ** 2
            for score, label in zip(scores, labels, strict=True)
        ) / len(scores)
        if scores else
        None
    )
    positive_mean = sum(positive) / len(positive) if positive else None
    negative_mean = sum(negative) / len(negative) if negative else None
    return {
        "semantics": (
            "mean auxiliary probability at the immutable ensemble-global plain "
            "selector candidate on gate-valid positive anticipatory lead-4-through-"
            "10 rows; the label is preferred-equivalent membership when correction "
            "is required, and false on no-correction rows"
        ),
        "rows": rows,
        "finite_rows": finite_rows,
        "nonfinite_rows": rows - finite_rows,
        "correction_required_rows": correction_required_rows,
        "no_correction_rows": no_correction_rows,
        "positive_rows": sum(labels),
        "positive_rate": sum(labels) / finite_rows if finite_rows else None,
        "selected_confidence": balanced._score_summary(scores),
        "positive_confidence": balanced._score_summary(positive),
        "negative_confidence": balanced._score_summary(negative),
        "positive_minus_negative_mean_confidence": (
            positive_mean - negative_mean
            if positive_mean is not None and negative_mean is not None else
            None
        ),
        "brier_score": brier,
        "reliability": balanced._reliability(scores, labels),
    }


def _evaluate_frozen_head(
    adapter: ResidualCorrectionAdapter,
    episodes: list[Any],
    fold: Any,
    reference_fold: Mapping[str, Any],
    dual_reference_fold: Mapping[str, Any],
    *,
    member_seed: int,
    selector_history: list[dict[str, float]],
    confidence_history: list[dict[str, float]],
    initial_membership_state_sha256: str,
    initial_non_membership_state_sha256: str,
    normalization_sha256: str,
    frozen_base_state_sha256: str,
    frozen_candidate_sha256: str,
    pre_audit_integrity_check: Callable[[], None],
) -> dict[str, Any]:
    fit = balanced._episodes_by_seed(episodes, fold.fit_seeds)
    calibration = balanced._episodes_by_seed(episodes, fold.calibration_seeds)
    audit = balanced._episodes_by_seed(episodes, fold.audit_seeds)

    fit_cal_predictions = balanced._prediction_map(adapter, [*fit, *calibration])
    runtime = None
    calibration_error = None
    calibration_failure_diagnostics = None
    try:
        runtime = balanced._calibrate(
            fit_cal_predictions,
            fit,
            calibration,
            ensemble_size=adapter.config.ensemble_size,
            per_action_safety_critic=adapter.config.per_action_safety_critic,
            per_action_physical_danger=adapter.config.per_action_physical_danger,
            future_onset_gate=True,
        )
    except ValueError as error:
        expected = "no fail-closed future-onset calibration covers early events"
        if expected not in str(error):
            raise
        calibration_error = str(error)
        calibration_failure_diagnostics = (
            balanced._future_onset_calibration_diagnostics(
                fit_cal_predictions,
                fit,
                calibration,
                ensemble_size=adapter.config.ensemble_size,
            )
        )

    # Audit is evaluated once, only after both training phases and calibration.
    pre_audit_integrity_check()
    audit_predictions = balanced._prediction_map(adapter, audit)
    predictions = {**fit_cal_predictions, **audit_predictions}
    for episode in episodes:
        dual._validated_dual_head_prediction(episode, predictions[episode.seed])

    raw = {
        "fit": balanced._raw_action_metrics(fit, predictions),
        "calibration": balanced._raw_action_metrics(calibration, predictions),
        "audit": balanced._raw_action_metrics(audit, predictions),
    }
    runtime_metrics = None
    if runtime is not None:
        runtime_metrics = {
            "fit": balanced._runtime_metrics(predictions, fit, runtime),
            "calibration": balanced._runtime_metrics(
                predictions, calibration, runtime
            ),
            "audit": balanced._runtime_metrics(predictions, audit, runtime),
        }
    confidence_diagnostics = {
        "fit": _selected_candidate_confidence_diagnostics(fit, predictions),
        "calibration": _selected_candidate_confidence_diagnostics(
            calibration, predictions
        ),
        "audit": _selected_candidate_confidence_diagnostics(audit, predictions),
    }

    trained_state = adapter.state_dict()
    membership_state, non_membership_state = dual._partition_membership_state(
        trained_state
    )
    action_state, shared_state = balanced._partition_action_branch_state(
        non_membership_state
    )
    trained_non_membership_sha256 = balanced._state_digest(non_membership_state)
    selector_prediction_sha256 = dual._selector_prediction_digest(
        [*fit, *calibration, *audit], predictions
    )
    invariants = dual._implementation_invariants(
        fold,
        reference_fold,
        initial_non_membership_state_sha256=(
            initial_non_membership_state_sha256
        ),
        normalization_sha256=normalization_sha256,
        trained_non_membership_state_sha256=trained_non_membership_sha256,
        selector_action_branch_state_sha256=balanced._state_digest(action_state),
        shared_non_action_branch_state_sha256=balanced._state_digest(shared_state),
        raw_action_metrics=raw,
        selector_prediction_sha256=selector_prediction_sha256,
    )
    reference_arm = reference_fold["plain"]
    dual_reference_arm = dual_reference_fold["dual_head"]
    extra_checks = {
        "selector_history_equal": selector_history == reference_arm.get("history"),
        "base_state_unchanged_by_head_fit": (
            trained_non_membership_sha256 == frozen_base_state_sha256
        ),
        "selector_prediction_equal_to_prior_dual": (
            selector_prediction_sha256
            == dual_reference_arm.get("implementation_invariants", {}).get(
                "selector_prediction_sha256"
            )
        ),
        "membership_initialization_equal_to_prior_dual": (
            initial_membership_state_sha256
            == dual_reference_fold.get(
                "initial_membership_head_state_sha256"
            )
        ),
    }
    failed = [name for name, passed in extra_checks.items() if not passed]
    if failed:
        raise AssertionError(
            "selected-confidence two-phase invariant failed: "
            + ", ".join(failed)
        )
    invariants["checks"].update(extra_checks)
    invariants["frozen_candidate_sha256"] = frozen_candidate_sha256
    return {
        "name": ARM_NAME,
        "member_seed": member_seed,
        "selector_history": selector_history,
        "confidence_history": confidence_history,
        "frozen_candidate_sha256": frozen_candidate_sha256,
        "trained_state_sha256": balanced._state_digest(trained_state),
        "membership_head_state_sha256": balanced._state_digest(membership_state),
        "trained_non_membership_state_sha256": trained_non_membership_sha256,
        "implementation_invariants": invariants,
        "calibration": {
            "success": runtime is not None,
            "error": calibration_error,
            "runtime_config": None if runtime is None else asdict(runtime),
            "calibration_grid_changed_from_plain": False,
            "failure_diagnostics": calibration_failure_diagnostics,
        },
        "raw_action_metrics": raw,
        "runtime_metrics": runtime_metrics,
        "selected_candidate_confidence_diagnostics": confidence_diagnostics,
    }


def _summary(folds: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    audit_rows = [
        fold[ARM_REPORT_KEY]["raw_action_metrics"]["audit"] for fold in folds
    ]
    calibrated = [
        bool(fold[ARM_REPORT_KEY]["calibration"]["success"]) for fold in folds
    ]
    audit_eligible: list[bool] = []
    for fold, success in zip(folds, calibrated, strict=True):
        runtime_metrics = fold[ARM_REPORT_KEY].get("runtime_metrics")
        audit = (
            runtime_metrics.get("audit")
            if isinstance(runtime_metrics, Mapping) else None
        )
        audit_eligible.append(
            success
            and isinstance(audit, Mapping)
            and audit.get("offline_deployment_eligible") is True
        )
    return {
        "outer_audit_micro": balanced._sum_raw(audit_rows),
        "calibration_successful_folds": sum(calibrated),
        "calibration_failed_folds": [
            int(fold["fold"])
            for fold, success in zip(folds, calibrated, strict=True)
            if not success
        ],
        "audit_runtime_eligible_folds": sum(audit_eligible),
        "audit_runtime_eligible_fold_indices": [
            int(fold["fold"])
            for fold, eligible in zip(folds, audit_eligible, strict=True)
            if eligible
        ],
        "calibrated_audit_runtime_ineligible_folds": [
            int(fold["fold"])
            for fold, success, eligible in zip(
                folds, calibrated, audit_eligible, strict=True
            )
            if success and not eligible
        ],
    }


def _adaptive_gate(summary: Mapping[str, Any]) -> dict[str, Any]:
    result = balanced._adaptive_development_gate(
        summary, epochs=SCREENING_EPOCHS
    )
    result.pop("preregistered_before_membership_cv30_audit", None)
    result["criteria"].update({
        "required_selector_training_epochs": 6,
        "required_confidence_training_epochs": 6,
        "required_confidence_loss_weight": 12.0,
    })
    result["checks"].update({
        "selector_training_is_exactly_six_epochs": SCREENING_EPOCHS == 6,
        "confidence_training_is_exactly_six_epochs": (
            HEAD_TRAINING_EPOCHS == 6
        ),
        "confidence_loss_weight_is_exactly_twelve": HEAD_LOSS_WEIGHT == 12.0,
    })
    result["applicable"] = bool(
        result["applicable"]
        and SCREENING_EPOCHS == 6
        and HEAD_TRAINING_EPOCHS == 6
        and HEAD_LOSS_WEIGHT == 12.0
    )
    result["inapplicable_reason"] = (
        None
        if result["applicable"] else
        "screen requires 6 selector epochs, 6 confidence epochs, and loss weight 12"
    )
    passed = result["applicable"] and all(result["checks"].values())
    result["passed"] = passed
    result["eligible_for_fixed_followup"] = passed
    result.update({
        "fourth_adaptive_development_screen": True,
        "adaptive_development_screen_sequence": 4,
        "independent_statistical_validation": False,
        "specified_after_observing_balanced_membership_result": True,
        "specified_after_observing_unweighted_membership_result": True,
        "specified_after_observing_dual_head_membership_result": True,
        "preregistered_before_selected_candidate_confidence_audit": True,
    })
    return result


def _reverify_references(args: argparse.Namespace) -> dict[str, str]:
    expected = {
        "plain": (args.plain_reference, PLAIN_REFERENCE_SHA256),
        "balanced": (args.balanced_reference, BALANCED_REFERENCE_SHA256),
        "unweighted": (
            args.unweighted_reference, UNWEIGHTED_REFERENCE_SHA256
        ),
        "dual": (args.dual_reference, DUAL_REFERENCE_SHA256),
    }
    result: dict[str, str] = {}
    for name, (path, digest) in expected.items():
        if file_sha256(path) != digest:
            raise ValueError(f"{name} reference changed during the CV30 run")
        result[name] = digest
    return result


def main() -> None:
    args = _argument_parser().parse_args()
    output = args.output.resolve()
    if output != FORMAL_OUTPUT.resolve():
        raise ValueError(
            "selected-confidence CV30 output is fixed; refusing an alternate path"
        )
    if args.cpu_threads != 1:
        raise ValueError("selected-confidence CV30 requires exactly 1 CPU thread")
    if SCREENING_EPOCHS != 6 or HEAD_TRAINING_EPOCHS != 6:
        raise AssertionError(
            "selected-confidence screen requires fixed e6 selector and head phases"
        )
    if HEAD_LOSS_WEIGHT != 12.0:
        raise AssertionError("selected-confidence fixed loss weight changed")
    if dual._selector_training_control_differences():
        raise AssertionError("selected-confidence selector differs from plain")
    if (
        SELECTED_CONFIDENCE_CONFIG.get("selector_phase_epochs")
        != SCREENING_EPOCHS
        or SELECTED_CONFIDENCE_CONFIG.get("confidence_phase_epochs")
        != HEAD_TRAINING_EPOCHS
        or SELECTED_CONFIDENCE_CONFIG.get("loss_weight") != HEAD_LOSS_WEIGHT
    ):
        raise AssertionError("selected-confidence protocol metadata differs")

    script_path = Path(__file__).resolve()
    helper_path = script_path.with_name("train_temporal_residual_adapter.py")
    dual_script_path = script_path.with_name(
        "compare_dual_head_membership_cv30.py"
    )
    protected = (
        args.failure,
        args.expansion_inventory,
        args.parent,
        args.plain_reference,
        args.balanced_reference,
        args.unweighted_reference,
        args.dual_reference,
        script_path,
        helper_path,
        dual_script_path,
    )
    dual._validate_new_output_path(output, protected)
    calibration_function_sha256 = _calibration_function_sha256()
    if calibration_function_sha256 != FIXED_CALIBRATION_FUNCTION_SHA256:
        raise AssertionError(
            "fixed future-onset calibration implementation changed"
        )
    startup_integrity_paths = {
        "failure": args.failure,
        "expansion_inventory": args.expansion_inventory,
        "parent": args.parent,
        "plain_reference": args.plain_reference,
        "balanced_reference": args.balanced_reference,
        "unweighted_reference": args.unweighted_reference,
        "dual_reference": args.dual_reference,
        "experiment_script": script_path,
        "training_helper": helper_path,
        "dual_experiment_script": dual_script_path,
        "balanced_experiment_script": _required_source_path(
            balanced,
            label="balanced experiment",
        ),
        "plain_experiment_script": _required_source_path(
            plain,
            label="plain experiment",
        ),
        "unweighted_experiment_script": _required_source_path(
            unweighted,
            label="unweighted experiment",
        ),
        "legacy_cv_script": _required_source_path(
            balanced._read_json,
            label="legacy CV helpers",
        ),
        "native_dataset_module": _required_source_path(
            native_dataset_module,
            label="native dataset module",
        ),
        "policy_module": _required_source_path(
            policy_module,
            label="policy module",
        ),
        "provenance_module": _required_source_path(
            provenance_module,
            label="provenance module",
        ),
        "residual_adapter_module": _required_source_path(
            residual_adapter_module,
            label="residual adapter module",
        ),
        "rollout_module": _required_source_path(
            rollout_module,
            label="rollout module",
        ),
        "training_module": _required_source_path(
            training_module,
            label="training module",
        ),
    }
    startup_sha256 = _file_digest_snapshot(startup_integrity_paths)
    _reserve_formal_campaign(
        output=output,
        ledger=FORMAL_CAMPAIGN_LEDGER,
        startup_sha256=startup_sha256,
        calibration_function_sha256=calibration_function_sha256,
    )
    campaign_ledger_sha256 = file_sha256(FORMAL_CAMPAIGN_LEDGER)

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    failure = balanced._read_json(args.failure)
    expansion = balanced._read_json(args.expansion_inventory)
    plain_reference = dual._read_frozen_reference(
        args.plain_reference,
        expected_sha256=PLAIN_REFERENCE_SHA256,
        label="plain",
    )
    balanced_reference = dual._read_frozen_reference(
        args.balanced_reference,
        expected_sha256=BALANCED_REFERENCE_SHA256,
        label="balanced membership",
    )
    unweighted_reference = dual._read_frozen_reference(
        args.unweighted_reference,
        expected_sha256=UNWEIGHTED_REFERENCE_SHA256,
        label="unweighted membership",
    )
    dual_reference = dual._read_frozen_reference(
        args.dual_reference,
        expected_sha256=DUAL_REFERENCE_SHA256,
        label="dual-head membership",
    )
    _reverify_file_digest_snapshot(startup_integrity_paths, startup_sha256)
    reference_folds = dual._reference_fold_map(plain_reference)
    dual_prior, dual_reference_folds = _validate_dual_reference(dual_reference)
    prior_results = {
        "plain": {
            "adaptive_development_screen_sequence": 0,
            "gate_passed": plain_reference.get(
                "preregistered_e6_promotion_gate", {}
            ).get("passed") is True,
            "outer_audit_micro": plain_reference.get("plain_summary", {}).get(
                "outer_audit_micro"
            ),
        },
        "balanced_membership": dual._validate_prior_adaptive_reference(
            balanced_reference,
            kind="certified_membership_adaptive_development_screen_cv30",
            sequence=1,
        ),
        "unweighted_membership": dual._validate_prior_adaptive_reference(
            unweighted_reference,
            kind="unweighted_membership_second_adaptive_development_screen_cv30",
            sequence=2,
        ),
        "dual_head_membership": dual_prior,
    }

    parent_sha256 = file_sha256(args.parent)
    if parent_sha256 != failure.get("parent_checkpoint_sha256"):
        raise ValueError("parent checkpoint hash does not match diagnostics")
    expected_provenance_sha256 = {
        "failure_sha256": startup_sha256["failure"],
        "expansion_inventory_sha256": startup_sha256["expansion_inventory"],
        "parent_sha256": parent_sha256,
    }
    for name, reference in (
        ("plain", plain_reference),
        ("balanced", balanced_reference),
        ("unweighted", unweighted_reference),
        ("dual", dual_reference),
    ):
        provenance = reference.get("input_provenance")
        if not isinstance(provenance, Mapping):
            raise ValueError(f"{name} reference lacks input provenance")
        for field, expected_sha256 in expected_provenance_sha256.items():
            if provenance.get(field) != expected_sha256:
                raise ValueError(f"{name} reference {field} differs")

    legacy_selected = balanced._select_training_inventory(failure)
    expansion_selected = balanced._select_expansion_inventory(
        expansion, checkpoint_sha256=parent_sha256
    )
    legacy_triplets, legacy_provenance = balanced._verify_training_sources(
        legacy_selected
    )
    expansion_triplets, expansion_provenance = (
        balanced._verify_expansion_sources(
            expansion_selected, checkpoint=args.parent
        )
    )
    triplets, source_provenance = balanced._merge_verified_sources(
        legacy_triplets,
        legacy_provenance,
        expansion_triplets,
        expansion_provenance,
    )
    source_paths, verified_source_sha256 = _verified_source_digest_map(
        triplets,
        source_provenance,
    )
    balanced._validate_output_path(
        output, list(source_paths.values())
    )
    observed_source_sha256 = _file_digest_snapshot(source_paths)
    if observed_source_sha256 != verified_source_sha256:
        raise ValueError("training source changed after declared-hash verification")
    integrity_paths = {
        **startup_integrity_paths,
        **source_paths,
        "campaign_ledger": FORMAL_CAMPAIGN_LEDGER,
    }
    integrity_sha256 = {
        **startup_sha256,
        **verified_source_sha256,
        "campaign_ledger": campaign_ledger_sha256,
    }
    integrity_reverification_count = 0

    def pre_audit_integrity_check() -> None:
        nonlocal integrity_reverification_count
        _reverify_file_digest_snapshot(integrity_paths, integrity_sha256)
        if _calibration_function_sha256() != calibration_function_sha256:
            raise ValueError("calibration implementation changed during run")
        integrity_reverification_count += 1

    dual_config = dual._dual_head_adapter_config(failure)
    plain_config = replace(
        dual_config, per_action_membership_confidence=False
    )
    parent, _metadata = load_checkpoint(args.parent, device="cpu")
    parent.cpu().eval()
    for parameter in parent.parameters():
        parameter.requires_grad_(False)
    if parent.config.recurrent_size != dual_config.recurrent_size:
        raise ValueError("parent recurrent size does not match adapter")
    feature_adapter = ResidualCorrectionAdapter(dual_config)
    raw_episodes = [
        balanced._load_episode(
            parent,
            feature_adapter,
            dataset,
            report,
            manifest,
            parent_checkpoint_sha256=parent_sha256,
            device="cpu",
            chunk_length=256,
            **balanced.LABEL_CONFIG,
        )
        for dataset, report, manifest in triplets
    ]
    _reverify_file_digest_snapshot(integrity_paths, integrity_sha256)
    if tuple(episode.seed for episode in raw_episodes) != (
        balanced.ALL_TRAINING_SEEDS
    ):
        raise ValueError("loaded episodes do not match CV30 whitelist")

    fold_reports: list[dict[str, Any]] = []
    for fold in balanced._fixed_cv30_folds():
        fold_seed = BASE_SEED + fold.index * 100_003
        normalized = [balanced._clone_episode(item) for item in raw_episodes]
        fit = balanced._episodes_by_seed(normalized, fold.fit_seeds)

        torch.manual_seed(fold_seed)
        plain_adapter = ResidualCorrectionAdapter(plain_config)
        balanced._normalize(plain_adapter, fit, normalized)
        torch.manual_seed(fold_seed)
        adapter = ResidualCorrectionAdapter(dual_config)
        with torch.no_grad():
            adapter.feature_mean.copy_(plain_adapter.feature_mean)
            adapter.feature_scale.copy_(plain_adapter.feature_scale)

        initial_state = dual._snapshot_state(adapter.state_dict())
        initial_membership, initial_non_membership = (
            dual._partition_membership_state(initial_state)
        )
        initial_membership_sha256 = balanced._state_digest(initial_membership)
        if initial_membership_sha256 != dual_reference_folds[fold.index].get(
            "initial_membership_head_state_sha256"
        ):
            raise AssertionError(
                "membership head initialization differs from frozen dual run"
            )
        initial_non_membership_sha256 = balanced._state_digest(
            initial_non_membership
        )
        normalization_sha256 = balanced._state_digest({
            "feature_mean": adapter.feature_mean,
            "feature_scale": adapter.feature_scale,
        })
        collision_weights = balanced._collision_positive_weights(
            fit,
            maximum_weight=SELECTOR_TRAINING_CONFIG[
                "maximum_collision_positive_weight"
            ],
        )
        physical_weights = balanced._physical_danger_positive_weights(
            fit,
            maximum_weight=SELECTOR_TRAINING_CONFIG[
                "maximum_physical_danger_positive_weight"
            ],
        )
        member_seed = fold_seed + 1_009
        selector_history = _train_plain_selector(
            plain_adapter,
            fit,
            member_seed=member_seed,
            collision_weights=collision_weights,
            physical_weights=physical_weights,
        )
        if selector_history != reference_folds[fold.index]["plain"]["history"]:
            raise AssertionError("selector history differs from frozen plain")
        _copy_plain_state_into_dual(plain_adapter, adapter)
        _membership, frozen_base = dual._partition_membership_state(
            adapter.state_dict()
        )
        frozen_base_sha256 = balanced._state_digest(frozen_base)

        frozen_candidates = _frozen_ensemble_selected_candidates(
            adapter,
            fit,
            chunk_length=SELECTOR_TRAINING_CONFIG["chunk_length"],
        )
        candidate_sha256 = _candidate_digest(fit, frozen_candidates)
        confidence_history = (
            _train_frozen_ensemble_selected_confidence_heads(
                adapter,
                fit,
                frozen_candidates=frozen_candidates,
                seed=member_seed,
                epochs=HEAD_TRAINING_EPOCHS,
                learning_rate=SELECTED_CONFIDENCE_CONFIG["learning_rate"],
                weight_decay=SELECTED_CONFIDENCE_CONFIG["weight_decay"],
                chunk_length=SELECTED_CONFIDENCE_CONFIG[
                    "fixed_label_batch_size"
                ],
                loss_weight=HEAD_LOSS_WEIGHT,
                max_norm=SELECTED_CONFIDENCE_CONFIG[
                    "gradient_clip_max_norm"
                ],
                device="cpu",
            )
        )
        repeated_candidates = _frozen_ensemble_selected_candidates(
            adapter,
            fit,
            chunk_length=SELECTOR_TRAINING_CONFIG["chunk_length"],
        )
        if _candidate_digest(fit, repeated_candidates) != candidate_sha256:
            raise AssertionError("frozen candidate digest changed after head fit")

        arm = _evaluate_frozen_head(
            adapter,
            normalized,
            fold,
            reference_folds[fold.index],
            dual_reference_folds[fold.index],
            member_seed=member_seed,
            selector_history=selector_history,
            confidence_history=confidence_history,
            initial_membership_state_sha256=initial_membership_sha256,
            initial_non_membership_state_sha256=(
                initial_non_membership_sha256
            ),
            normalization_sha256=normalization_sha256,
            frozen_base_state_sha256=frozen_base_sha256,
            frozen_candidate_sha256=candidate_sha256,
            pre_audit_integrity_check=pre_audit_integrity_check,
        )
        fold_reports.append({
            "fold": fold.index,
            "fit_seeds": list(fold.fit_seeds),
            "calibration_seeds": list(fold.calibration_seeds),
            "audit_seeds": list(fold.audit_seeds),
            "normalization_fit_seeds": list(fold.fit_seeds),
            "confidence_fit_seeds": list(fold.fit_seeds),
            "split_acquisition": {
                name: balanced._split_acquisition_audit(seeds)
                for name, seeds in (
                    ("fit", fold.fit_seeds),
                    ("calibration", fold.calibration_seeds),
                    ("audit", fold.audit_seeds),
                )
            },
            "fold_seed": fold_seed,
            "member_seed": member_seed,
            "initial_membership_head_state_sha256": initial_membership_sha256,
            "initial_non_membership_state_sha256": (
                initial_non_membership_sha256
            ),
            "normalization_sha256": normalization_sha256,
            "collision_positive_weights": collision_weights.tolist(),
            "physical_danger_positive_weights": physical_weights.tolist(),
            ARM_REPORT_KEY: arm,
        })

    if integrity_reverification_count != 3:
        raise AssertionError("each fold must reverify integrity before audit")
    pre_audit_integrity_check()
    if integrity_reverification_count != 4:
        raise AssertionError("final integrity reverification did not run")
    summary = _summary(fold_reports)
    verified_reference_sha256 = _reverify_references(args)
    report = {
        "schema_version": 1,
        "kind": (
            "selected_candidate_confidence_fourth_adaptive_development_"
            "screen_cv30"
        ),
        "training_only": True,
        "deployment_artifact_written": False,
        "formal_deployment_artifact_written": False,
        "deployment_eligible": False,
        "acceptance_claim": False,
        "adaptive_development_screen": True,
        "adaptive_development_screen_sequence": 4,
        "independent_statistical_validation": False,
        "specified_after_observing_plain_cv30_result": True,
        "specified_after_observing_balanced_membership_result": True,
        "specified_after_observing_unweighted_membership_result": True,
        "specified_after_observing_dual_head_membership_result": True,
        "objective_arms": [ARM_NAME],
        "variant_objectives_evaluated": [],
        "objective_scope": (
            "one fixed two-phase selected-candidate confidence objective; no "
            "epoch, seed, loss, mask, threshold, aggregation, or retry variants"
        ),
        "one_shot_campaign": {
            "output_path_fixed": True,
            "cpu_threads_fixed_to_one": True,
            "campaign_ledger": str(FORMAL_CAMPAIGN_LEDGER),
            "campaign_ledger_sha256": campaign_ledger_sha256,
            "ledger_reserved_before_training_or_audit": True,
            "crash_or_failure_consumes_campaign": True,
            "retry_allowed": False,
        },
        "selector_objective": dict(dual.SELECTOR_OBJECTIVE_CONFIG),
        "selected_candidate_confidence_objective": dict(
            SELECTED_CONFIDENCE_CONFIG
        ),
        "prior_observed_results": prior_results,
        "audit_used_during_fit_or_calibration": False,
        "audit_used_for_threshold_epoch_seed_mode_weight_or_retry_selection": False,
        "audit_used_for_after_freeze_adaptive_screen": True,
        "audit_prediction_policy": (
            "each fold predicts audit exactly once after selector fit, frozen "
            "candidate extraction, confidence fit, and calibration freeze"
        ),
        "data_isolation": {
            "legacy_training_seeds": list(balanced.LEGACY_TRAINING_SEEDS),
            "expansion_training_seeds": list(balanced.EXPANSION_TRAINING_SEEDS),
            "ordered_interleaved_training_seeds": list(
                balanced.ALL_TRAINING_SEEDS
            ),
            "prohibited_source_seeds": sorted(
                balanced.PROHIBITED_SOURCE_SEEDS
            ),
            "selection_before_path_access": True,
            "confidence_training_uses_fit_only": True,
            "source_inventory": source_provenance,
        },
        "fold_protocol": {
            "folds": 3,
            "fit_episodes_per_fold": 16,
            "calibration_episodes_per_fold": 4,
            "audit_episodes_per_fold": 10,
            "legacy_expansion_counts_per_fold": {
                "fit": [8, 8],
                "calibration": [2, 2],
                "audit": [5, 5],
            },
            "outer_audit_covers_each_training_episode_once": True,
            "audit_predictions_per_fold": 1,
        },
        "input_provenance": {
            "failure": str(args.failure),
            "failure_sha256": startup_sha256["failure"],
            "expansion_inventory": str(args.expansion_inventory),
            "expansion_inventory_sha256": startup_sha256[
                "expansion_inventory"
            ],
            "parent": str(args.parent),
            "parent_sha256": parent_sha256,
            "plain_reference": str(args.plain_reference),
            "balanced_reference": str(args.balanced_reference),
            "unweighted_reference": str(args.unweighted_reference),
            "dual_reference": str(args.dual_reference),
            "frozen_reference_hash_verification": {
                "verified_before_content_read": True,
                "reverified_after_training": True,
                "sha256": verified_reference_sha256,
            },
            "experiment_script": str(script_path),
            "experiment_script_sha256": startup_sha256["experiment_script"],
            "dual_experiment_script": str(dual_script_path),
            "dual_experiment_script_sha256": startup_sha256[
                "dual_experiment_script"
            ],
            "training_helper": str(helper_path),
            "training_helper_sha256": startup_sha256["training_helper"],
            "implementation_file_sha256": {
                name: startup_sha256[name]
                for name in (
                    "experiment_script",
                    "training_helper",
                    "dual_experiment_script",
                    "balanced_experiment_script",
                    "plain_experiment_script",
                    "unweighted_experiment_script",
                    "legacy_cv_script",
                    "native_dataset_module",
                    "policy_module",
                    "provenance_module",
                    "residual_adapter_module",
                    "rollout_module",
                    "training_module",
                )
            },
            "source_file_sha256": {
                name: integrity_sha256[name] for name in source_paths
            },
            "integrity_reverification": {
                "after_episode_loading": True,
                "before_each_fold_audit": True,
                "pre_audit_reverification_count": 3,
                "after_all_training_and_audit": True,
                "total_reverification_count": integrity_reverification_count,
            },
        },
        "experiment_config": {
            "selector_epochs": SCREENING_EPOCHS,
            "confidence_epochs": HEAD_TRAINING_EPOCHS,
            "base_seed": BASE_SEED,
            "device": "cpu",
            "cpu_threads": 1,
            "torch_version": str(torch.__version__),
            "deterministic_algorithms": True,
            "adapter_config": asdict(dual_config),
            "label_config": balanced.LABEL_CONFIG,
            "selector_training_config": SELECTOR_TRAINING_CONFIG,
            "selected_confidence_config": SELECTED_CONFIDENCE_CONFIG,
            "ensemble_size_screening_override": 1,
            "calibration_grid_changed_from_plain": False,
            "fixed_calibration_function_sha256": (
                calibration_function_sha256
            ),
        },
        "folds": fold_reports,
        "selected_candidate_confidence_summary": summary,
        "adaptive_development_gate": _adaptive_gate(summary),
    }
    balanced._write_json_atomic(output, report)
    print(json.dumps({
        "artifact": str(output),
        "sha256": file_sha256(output),
        "training_only": True,
        "adaptive_development_screen_sequence": 4,
        "independent_statistical_validation": False,
        "eligible_for_fixed_followup": report[
            "adaptive_development_gate"
        ]["eligible_for_fixed_followup"],
    }))


if __name__ == "__main__":
    main()
