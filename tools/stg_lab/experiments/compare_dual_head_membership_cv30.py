"""Third adaptive CV30 screen for a detached membership-confidence head.

The selector is the frozen plain certified-set NLL arm.  The only model change
is an auxiliary per-action membership head whose input is detached from the
selector/shared path.  This screen was specified after the plain, balanced
membership, and unweighted membership results were inspected, so it is a
training-only adaptive development result, not independent validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch

from stg_lab.provenance import file_sha256
from stg_lab.residual_adapter import ResidualCorrectionAdapter
from stg_lab.training import load_checkpoint

if __package__:
    from . import compare_certified_membership_cv30 as balanced
    from . import compare_plain_certified_set_cv30 as plain
    from . import compare_unbalanced_membership_cv30 as unweighted
    from .compare_preferred_objectives_cv import EXPECTED_V81_CONFIG
else:  # pragma: no cover - exercised by the real script invocation
    import compare_certified_membership_cv30 as balanced
    import compare_plain_certified_set_cv30 as plain
    import compare_unbalanced_membership_cv30 as unweighted
    from compare_preferred_objectives_cv import EXPECTED_V81_CONFIG


DUAL_HEAD_ARM_NAME = "plain_selector_auxiliary_membership_confidence"
SCREENING_EPOCHS = 6
BASE_SEED = 20260901
MEMBERSHIP_CONFIDENCE_LOSS_WEIGHT = 12.0
MEMBERSHIP_CONFIDENCE_LOSS_MODE = "unweighted"

DEFAULT_PLAIN_REFERENCE = plain.DEFAULT_OUTPUT
DEFAULT_BALANCED_REFERENCE = balanced.DEFAULT_OUTPUT
DEFAULT_UNWEIGHTED_REFERENCE = unweighted.DEFAULT_OUTPUT
PLAIN_REFERENCE_SHA256 = (
    "8f9a0106e26b612657345be14e9e1c0c56f137f4a02d579ecbf6ad10451ed206"
)
BALANCED_REFERENCE_SHA256 = (
    "ddff204e80ce0b646bf1dbe3285b7c7b774a858b253c11dbff666e0754ae1184"
)
UNWEIGHTED_REFERENCE_SHA256 = (
    "da079dc5a0a234e5c545f27570bd459fc8c457031abe186d553fa9c568d223ec"
)
DEFAULT_OUTPUT = Path(
    "artifacts/policy-humanlike-highres-okuu3-dual-head-membership-cv30-e6.json"
)

SELECTOR_OBJECTIVE_CONFIG = {
    **plain.PLAIN_OBJECTIVE_CONFIG,
    "action_logit_mode": "parent_residual_joint",
    "parent_copy_weight": plain.TRAINING_CONFIG["parent_copy_weight"],
}
SELECTOR_TRAINING_CONFIG = {
    **plain.TRAINING_CONFIG,
    "preferred_action_uniform_loss_weight": 0.0,
    "preferred_action_tiebreak_loss_weight": 0.0,
    "preferred_action_rank_loss_weight": 0.0,
    "preferred_action_rank_margin": 1.0,
}
AUXILIARY_MEMBERSHIP_CONFIG = {
    "schema": "detached_certified_action_membership_unweighted_bce",
    "per_action_membership_confidence": True,
    "membership_confidence_loss_weight": MEMBERSHIP_CONFIDENCE_LOSS_WEIGHT,
    "membership_confidence_loss_mode": MEMBERSHIP_CONFIDENCE_LOSS_MODE,
    "target_rows": "gate_valid_and_positive_only",
    "target_actions": "preferred_certified_action_set",
    "row_balance": "none_equal_weight_per_action_cell",
    "head_input": "detached_selector_recurrent",
    "selector_gradient_from_membership_loss": False,
    "optimizer": "independent_adamw",
    "learning_rate": plain.TRAINING_CONFIG["learning_rate"],
    "weight_decay": plain.TRAINING_CONFIG["weight_decay"],
    "gradient_clip": "independent_membership_head_group",
    "gradient_clip_max_norm": balanced.GRADIENT_CLIP_MAX_NORM,
    "runtime_confidence": (
        "mean_membership_probability_at_selector_selected_action"
    ),
}

THIRD_ADAPTIVE_SCREEN_CONTEXT = {
    "sequence": 3,
    "adaptive_development_screen": True,
    "independent_statistical_validation": False,
    "specified_after_observing_plain_cv30_result": True,
    "specified_after_observing_balanced_membership_result": True,
    "specified_after_observing_unweighted_membership_result": True,
    "selector_training_change_from_plain": "none",
    "sole_model_change_from_plain": "per_action_membership_confidence: false -> true",
    "auxiliary_loss": "detached unweighted membership BCE weight 12",
    "epoch_seed_mode_weight_or_retry_selected_from_this_result": False,
}

SELECTOR_PREDICTION_FIELDS = (
    "mean_gate",
    "minimum_gate",
    "action_member_finite",
    "action_probabilities",
    "mean_action_probabilities",
    "selector_all_members_finite",
    "candidates",
    "agreement",
    "collision_probabilities",
    "minimum_margins",
    "physical_danger_probabilities",
)


def _dual_head_adapter_config(failure: Mapping[str, Any]) -> Any:
    base = plain._adapter_config(failure)
    if base.action_logit_mode != "parent_residual_joint":
        raise ValueError("fixed plain selector must use parent_residual_joint logits")
    if base.per_action_membership_confidence:
        raise ValueError("plain selector unexpectedly already has a membership head")
    result = replace(base, per_action_membership_confidence=True)
    before = asdict(base)
    after = asdict(result)
    changed = {
        name: (before[name], after[name])
        for name in before
        if before[name] != after[name]
    }
    if changed != {"per_action_membership_confidence": (False, True)}:
        raise AssertionError("dual-head config changed more than membership confidence")
    return result


def _selector_training_control_differences() -> dict[str, tuple[Any, Any]]:
    expected = {
        **plain.TRAINING_CONFIG,
        "preferred_action_uniform_loss_weight": 0.0,
        "preferred_action_tiebreak_loss_weight": 0.0,
        "preferred_action_rank_loss_weight": 0.0,
        "preferred_action_rank_margin": 1.0,
    }
    return {
        name: (expected.get(name), SELECTOR_TRAINING_CONFIG.get(name))
        for name in expected.keys() | SELECTOR_TRAINING_CONFIG.keys()
        if expected.get(name) != SELECTOR_TRAINING_CONFIG.get(name)
    }


def _partition_membership_state(
    state: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    membership: dict[str, torch.Tensor] = {}
    non_membership: dict[str, torch.Tensor] = {}
    for name, tensor in state.items():
        target = membership if "membership_head" in name.split(".") else non_membership
        target[name] = tensor
    if not membership:
        raise AssertionError("dual-head state has no membership_head tensors")
    if not non_membership:
        raise AssertionError("dual-head state has no non-membership tensors")
    if set(membership) & set(non_membership):
        raise AssertionError("membership state partition overlaps")
    if set(membership) | set(non_membership) != set(state):
        raise AssertionError("membership state partition omits tensors")
    return membership, non_membership


def _snapshot_state(
    state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in state.items()
    }


def _update_tensor_digest(
    digest: Any,
    name: str,
    tensor: torch.Tensor,
) -> None:
    value = tensor.detach().cpu().contiguous()
    digest.update(name.encode("utf-8"))
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(value.numpy().tobytes())


def _selector_prediction_digest(
    episodes: Sequence[Any],
    predictions: Mapping[int, Mapping[str, torch.Tensor]],
) -> str:
    digest = hashlib.sha256()
    for episode in episodes:
        digest.update(str(int(episode.seed)).encode("ascii"))
        values = predictions[episode.seed]
        for name in SELECTOR_PREDICTION_FIELDS:
            tensor = values.get(name)
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f"selector prediction lacks tensor field {name}")
            _update_tensor_digest(digest, name, tensor)
    return digest.hexdigest()


def _validated_dual_head_prediction(
    episode: Any,
    values: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    member_probabilities = values.get("membership_probabilities")
    mean_probabilities = values.get("mean_membership_probabilities")
    member_finite = values.get("membership_member_finite")
    candidates = values.get("candidates")
    confidence = values.get("action_confidence")
    selector_finite = values.get("selector_all_members_finite")
    combined_finite = values.get("action_all_members_finite")
    required = (
        member_probabilities,
        mean_probabilities,
        member_finite,
        candidates,
        confidence,
        selector_finite,
        combined_finite,
    )
    if any(value is None for value in required):
        raise ValueError("dual-head prediction lacks membership confidence fields")
    assert member_probabilities is not None
    assert mean_probabilities is not None
    assert member_finite is not None
    assert candidates is not None
    assert confidence is not None
    assert selector_finite is not None
    assert combined_finite is not None
    decisions = int(episode.decisions)
    if member_probabilities.ndim != 3 or member_probabilities.shape[1:] != (
        decisions,
        18,
    ):
        raise ValueError("membership member probabilities do not align with episode")
    if mean_probabilities.shape != (decisions, 18):
        raise ValueError("mean membership probabilities do not align with episode")
    if member_finite.shape != member_probabilities.shape[:-1]:
        raise ValueError("membership member finite mask does not align")
    for name, value in (
        ("membership member finite", member_finite),
        ("selector finite", selector_finite),
        ("combined finite", combined_finite),
    ):
        if value.shape[-1:] != (decisions,) or value.dtype != torch.bool:
            raise ValueError(f"{name} mask is invalid")
    if selector_finite.shape != (decisions,) or combined_finite.shape != (decisions,):
        raise ValueError("selector/combined finite masks do not align with episode")
    if candidates.shape != (decisions,) or confidence.shape != (decisions,):
        raise ValueError("selector candidate/confidence does not align with episode")
    finite_cells = torch.isfinite(member_probabilities).all(dim=-1)
    effective_member_finite = member_finite & finite_cells
    safe_membership = torch.where(
        effective_member_finite.unsqueeze(-1),
        member_probabilities,
        torch.zeros_like(member_probabilities),
    )
    expected_mean = safe_membership.mean(dim=0)
    if not torch.equal(mean_probabilities, expected_mean):
        raise ValueError("mean membership probabilities do not match members")
    membership_finite = effective_member_finite.all(dim=0)
    expected_combined = selector_finite & membership_finite
    if not torch.equal(combined_finite, expected_combined):
        raise ValueError("combined finite mask is not selector AND membership finite")
    if bool(((candidates < 0) | (candidates >= 18)).any()):
        raise ValueError("selector candidate is outside the action vocabulary")
    selected = mean_probabilities.gather(
        -1,
        candidates.unsqueeze(-1),
    ).squeeze(-1)
    if not torch.equal(confidence, selected):
        raise ValueError("action confidence is not selected membership probability")
    if bool((combined_finite & ~torch.isfinite(confidence)).any()):
        raise ValueError("finite dual-head rows contain nonfinite confidence")
    return selected, candidates, combined_finite


def _selected_confidence_scope(
    episodes: Sequence[Any],
    predictions: Mapping[int, Mapping[str, torch.Tensor]],
    mask_for_episode: Callable[[Any], torch.Tensor],
) -> dict[str, Any]:
    scores: list[float] = []
    labels: list[bool] = []
    total_rows = 0
    finite_rows = 0
    for episode in episodes:
        selected, candidates, finite = _validated_dual_head_prediction(
            episode,
            predictions[episode.seed],
        )
        mask = mask_for_episode(episode)
        if mask.shape != (episode.decisions,) or mask.dtype != torch.bool:
            raise ValueError("membership confidence diagnostic mask is invalid")
        total_rows += int(mask.sum())
        valid = mask & finite
        finite_rows += int(valid.sum())
        target = episode.preferred_action_set.gather(
            -1,
            candidates.unsqueeze(-1),
        ).squeeze(-1)
        scores.extend(float(value) for value in selected[valid].tolist())
        labels.extend(bool(value) for value in target[valid].tolist())
    certified = [score for score, label in zip(scores, labels, strict=True) if label]
    rejected = [score for score, label in zip(scores, labels, strict=True) if not label]
    brier = (
        sum((score - float(label)) ** 2 for score, label in zip(scores, labels, strict=True))
        / len(scores)
        if scores else
        None
    )
    certified_mean = sum(certified) / len(certified) if certified else None
    rejected_mean = sum(rejected) / len(rejected) if rejected else None
    return {
        "rows": total_rows,
        "finite_rows": finite_rows,
        "nonfinite_rows": total_rows - finite_rows,
        "selected_certified_rows": sum(labels),
        "selected_certified_rate": (
            sum(labels) / finite_rows if finite_rows else None
        ),
        "selected_confidence": balanced._score_summary(scores),
        "selected_certified_confidence": balanced._score_summary(certified),
        "selected_rejected_confidence": balanced._score_summary(rejected),
        "certified_minus_rejected_mean_confidence": (
            certified_mean - rejected_mean
            if certified_mean is not None and rejected_mean is not None else
            None
        ),
        "selected_confidence_brier_score": brier,
        "selected_confidence_reliability": balanced._reliability(scores, labels),
    }


def _selected_membership_confidence_diagnostics(
    episodes: Sequence[Any],
    predictions: Mapping[int, Mapping[str, torch.Tensor]],
) -> dict[str, Any]:
    return {
        "semantics": (
            "auxiliary mean membership probability gathered at the frozen plain "
            "selector candidate; descriptive after training/calibration freeze"
        ),
        "all_positive_rows": _selected_confidence_scope(
            episodes,
            predictions,
            balanced._positive_rows,
        ),
        "early_correction_required_4_10": _selected_confidence_scope(
            episodes,
            predictions,
            balanced._early_correction_rows,
        ),
    }


def _reference_fold_map(reference: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    if reference.get("kind") != "plain_certified_set_training_only_cv30":
        raise ValueError("plain reference has the wrong artifact kind")
    if reference.get("training_only") is not True:
        raise ValueError("plain reference is not training-only")
    if any(
        reference.get(name) is not False
        for name in (
            "deployment_artifact_written",
            "formal_deployment_artifact_written",
            "deployment_eligible",
            "acceptance_claim",
        )
    ):
        raise ValueError("plain reference makes a deployment or acceptance claim")
    experiment = reference.get("experiment_config")
    if not isinstance(experiment, Mapping):
        raise ValueError("plain reference lacks experiment configuration")
    if experiment.get("epochs") != SCREENING_EPOCHS:
        raise ValueError("plain reference is not the frozen e6 run")
    if experiment.get("base_seed") != BASE_SEED:
        raise ValueError("plain reference base seed differs")
    if experiment.get("device") != "cpu" or experiment.get(
        "deterministic_algorithms"
    ) is not True:
        raise ValueError("plain reference is not deterministic CPU")
    if experiment.get("ensemble_size_screening_override") != 1:
        raise ValueError("plain reference is not the ensemble-one screen")
    adapter_metadata = experiment.get("adapter_config")
    if not isinstance(adapter_metadata, Mapping):
        raise ValueError("plain reference lacks adapter configuration")
    normalized_adapter_metadata = dict(adapter_metadata)
    normalized_adapter_metadata.setdefault(
        "per_action_membership_confidence",
        False,
    )
    expected_adapter_metadata = {
        **EXPECTED_V81_CONFIG,
        "ensemble_size": 1,
        "per_action_membership_confidence": False,
    }
    if normalized_adapter_metadata != expected_adapter_metadata:
        if normalized_adapter_metadata.get(
            "per_action_membership_confidence"
        ) is not False:
            raise ValueError("plain reference unexpectedly has a membership head")
        raise ValueError("plain reference adapter configuration differs")
    if reference.get("plain_objective") != plain.PLAIN_OBJECTIVE_CONFIG:
        raise ValueError("plain reference objective differs from the frozen set NLL")
    if experiment.get("training_config") != plain.TRAINING_CONFIG:
        raise ValueError("plain reference training controls differ")
    folds = reference.get("folds")
    if not isinstance(folds, list) or len(folds) != 3:
        raise ValueError("plain reference must contain exactly three folds")
    result: dict[int, Mapping[str, Any]] = {}
    for expected, raw in zip(balanced._fixed_cv30_folds(), folds, strict=True):
        if not isinstance(raw, Mapping) or raw.get("fold") != expected.index:
            raise ValueError("plain reference fold order differs")
        for name, seeds in (
            ("fit_seeds", expected.fit_seeds),
            ("calibration_seeds", expected.calibration_seeds),
            ("audit_seeds", expected.audit_seeds),
        ):
            if raw.get(name) != list(seeds):
                raise ValueError(f"plain reference {name} differs in fold {expected.index}")
        arm = raw.get("plain")
        if not isinstance(arm, Mapping):
            raise ValueError("plain reference fold lacks its plain arm")
        result[expected.index] = raw
    return result


def _read_frozen_reference(
    path: Path,
    *,
    expected_sha256: str,
    label: str,
) -> Mapping[str, Any]:
    before_sha256 = file_sha256(path)
    if before_sha256 != expected_sha256:
        raise ValueError(f"{label} reference hash differs from the frozen e6 artifact")
    reference = balanced._read_json(path)
    after_sha256 = file_sha256(path)
    if after_sha256 != expected_sha256:
        raise ValueError(
            f"{label} reference changed while being read or differs from the "
            "frozen e6 artifact"
        )
    return reference


def _reverify_frozen_reference_hashes(
    *,
    plain_path: Path,
    balanced_path: Path,
    unweighted_path: Path,
) -> dict[str, str]:
    verified: dict[str, str] = {}
    for name, label, path, expected_sha256 in (
        ("plain", "plain", plain_path, PLAIN_REFERENCE_SHA256),
        (
            "balanced",
            "balanced membership",
            balanced_path,
            BALANCED_REFERENCE_SHA256,
        ),
        (
            "unweighted",
            "unweighted membership",
            unweighted_path,
            UNWEIGHTED_REFERENCE_SHA256,
        ),
    ):
        if file_sha256(path) != expected_sha256:
            raise ValueError(
                f"{label} reference changed during the dual-head CV30 run"
            )
        verified[name] = expected_sha256
    return verified


def _validate_prior_adaptive_reference(
    reference: Mapping[str, Any],
    *,
    kind: str,
    sequence: int,
) -> dict[str, Any]:
    if reference.get("kind") != kind:
        raise ValueError("prior adaptive reference has the wrong artifact kind")
    if reference.get("training_only") is not True:
        raise ValueError("prior adaptive reference is not training-only")
    if any(
        reference.get(name) is not False
        for name in (
            "deployment_artifact_written",
            "formal_deployment_artifact_written",
            "deployment_eligible",
            "acceptance_claim",
        )
    ):
        raise ValueError(
            "prior adaptive reference makes a deployment or acceptance claim"
        )
    if reference.get("adaptive_development_screen") is not True:
        raise ValueError(
            "prior adaptive reference is not an adaptive development screen"
        )
    if reference.get("independent_statistical_validation") is not False:
        raise ValueError("prior adaptive reference claims independent validation")
    expected_objective_arms = {
        (
            "certified_membership_adaptive_development_screen_cv30",
            1,
        ): [balanced.MEMBERSHIP_ARM_NAME],
        (
            "unweighted_membership_second_adaptive_development_screen_cv30",
            2,
        ): [unweighted.UNBALANCED_ARM_NAME],
    }.get((kind, sequence))
    if expected_objective_arms is None:
        raise ValueError("prior adaptive reference kind and sequence are unsupported")
    if reference.get("objective_arms") != expected_objective_arms:
        raise ValueError("prior adaptive reference objective arm differs")
    if reference.get(
        "specified_after_observing_plain_cv30_negative_result"
    ) is not True:
        raise ValueError("prior adaptive reference lacks observed-result provenance")
    if sequence == 2 and reference.get(
        "specified_after_observing_balanced_membership_negative_result"
    ) is not True:
        raise ValueError(
            "second prior adaptive reference lacks balanced-result provenance"
        )
    declared_sequence = reference.get("adaptive_development_screen_sequence")
    legacy_sequence_inferred = False
    if declared_sequence is None:
        legacy_sequence_inferred = (
            sequence == 1
            and kind == "certified_membership_adaptive_development_screen_cv30"
            and reference.get("adaptive_development_screen") is True
            and reference.get("independent_statistical_validation") is False
            and reference.get(
                "specified_after_observing_plain_cv30_negative_result"
            ) is True
            and reference.get("objective_arms") == [balanced.MEMBERSHIP_ARM_NAME]
        )
        if not legacy_sequence_inferred:
            raise ValueError("prior adaptive reference sequence is missing")
    elif declared_sequence != sequence:
        raise ValueError("prior adaptive reference sequence differs")
    experiment = reference.get("experiment_config")
    if not isinstance(experiment, Mapping):
        raise ValueError("prior adaptive reference lacks experiment config")
    if experiment.get("epochs") != SCREENING_EPOCHS:
        raise ValueError("prior adaptive reference is not e6")
    if experiment.get("base_seed") != BASE_SEED:
        raise ValueError("prior adaptive reference base seed differs")
    gate = reference.get("adaptive_development_gate")
    summary = reference.get("membership_summary")
    if not isinstance(gate, Mapping) or not isinstance(summary, Mapping):
        raise ValueError("prior adaptive reference lacks result summary")
    return {
        "adaptive_development_screen_sequence": sequence,
        "source_sequence_field": declared_sequence,
        "legacy_sequence_inferred_from_exact_kind_and_provenance": (
            legacy_sequence_inferred
        ),
        "gate_passed": gate.get("passed") is True,
        "outer_audit_micro": summary.get("outer_audit_micro"),
    }


def _implementation_invariants(
    fold: Any,
    reference_fold: Mapping[str, Any],
    *,
    initial_non_membership_state_sha256: str,
    normalization_sha256: str,
    trained_non_membership_state_sha256: str,
    selector_action_branch_state_sha256: str,
    shared_non_action_branch_state_sha256: str,
    raw_action_metrics: Mapping[str, Any],
    selector_prediction_sha256: str,
) -> dict[str, Any]:
    reference_arm = reference_fold.get("plain")
    if not isinstance(reference_arm, Mapping):
        raise ValueError("plain reference fold lacks arm metrics")
    checks = {
        "fold_index_equal": reference_fold.get("fold") == fold.index,
        "fit_seeds_equal": reference_fold.get("fit_seeds") == list(fold.fit_seeds),
        "calibration_seeds_equal": (
            reference_fold.get("calibration_seeds") == list(fold.calibration_seeds)
        ),
        "audit_seeds_equal": (
            reference_fold.get("audit_seeds") == list(fold.audit_seeds)
        ),
        "initial_non_membership_state_equal": (
            initial_non_membership_state_sha256
            == reference_fold.get("initial_state_sha256")
        ),
        "normalization_state_equal": (
            normalization_sha256 == reference_fold.get("normalization_sha256")
        ),
        "trained_non_membership_state_equal": (
            trained_non_membership_state_sha256
            == reference_arm.get("trained_state_sha256")
        ),
        "selector_action_branch_state_equal": (
            selector_action_branch_state_sha256
            == reference_arm.get("action_branch_state_sha256")
        ),
        "shared_non_action_branch_state_equal": (
            shared_non_action_branch_state_sha256
            == reference_arm.get("non_action_branch_state_sha256")
        ),
        "fit_raw_selector_metrics_equal": (
            raw_action_metrics.get("fit")
            == reference_arm.get("raw_action_metrics", {}).get("fit")
        ),
        "calibration_raw_selector_metrics_equal": (
            raw_action_metrics.get("calibration")
            == reference_arm.get("raw_action_metrics", {}).get("calibration")
        ),
        "audit_raw_selector_metrics_equal": (
            raw_action_metrics.get("audit")
            == reference_arm.get("raw_action_metrics", {}).get("audit")
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AssertionError(
            "dual-head implementation changed the frozen plain selector/shared "
            "path: " + ", ".join(failed)
        )
    return {
        "all_passed": True,
        "checks": checks,
        "state_hash_semantics": (
            "all membership_head.* tensors removed before comparison with the "
            "immutable plain e6 artifact"
        ),
        "prediction_invariant_semantics": (
            "stripped trained state is bit-identical to plain; the single dual-head "
            "post-freeze selector prediction pass is hashed below and its fit/cal/"
            "audit raw metrics are exactly equal to plain"
        ),
        "selector_prediction_sha256": selector_prediction_sha256,
    }


def _train_dual_head_member(
    adapter: ResidualCorrectionAdapter,
    fit: list[Any],
    *,
    member_seed: int,
    collision_weights: torch.Tensor,
    physical_weights: torch.Tensor,
) -> list[dict[str, float]]:
    config = SELECTOR_TRAINING_CONFIG
    expected_auxiliary_optimizer = {
        "optimizer": "independent_adamw",
        "learning_rate": config["learning_rate"],
        "weight_decay": config["weight_decay"],
        "gradient_clip": "independent_membership_head_group",
        "gradient_clip_max_norm": balanced.GRADIENT_CLIP_MAX_NORM,
    }
    actual_auxiliary_optimizer = {
        name: AUXILIARY_MEMBERSHIP_CONFIG[name]
        for name in expected_auxiliary_optimizer
    }
    if actual_auxiliary_optimizer != expected_auxiliary_optimizer:
        raise AssertionError("auxiliary membership optimizer controls changed")
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
        preferred_action_rank_loss_weight=config["preferred_action_rank_loss_weight"],
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
        membership_confidence_loss_weight=MEMBERSHIP_CONFIDENCE_LOSS_WEIGHT,
        membership_confidence_loss_mode=MEMBERSHIP_CONFIDENCE_LOSS_MODE,
        device="cpu",
    )


def _run_dual_head_arm(
    adapter: ResidualCorrectionAdapter,
    episodes: list[Any],
    fold: Any,
    reference_fold: Mapping[str, Any],
    *,
    member_seed: int,
    collision_weights: torch.Tensor,
    physical_weights: torch.Tensor,
    initial_non_membership_state_sha256: str,
    normalization_sha256: str,
) -> dict[str, Any]:
    fit = balanced._episodes_by_seed(episodes, fold.fit_seeds)
    calibration = balanced._episodes_by_seed(episodes, fold.calibration_seeds)
    audit = balanced._episodes_by_seed(episodes, fold.audit_seeds)
    history = _train_dual_head_member(
        adapter,
        fit,
        member_seed=member_seed,
        collision_weights=collision_weights,
        physical_weights=physical_weights,
    )
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

    # Exactly one audit prediction, after training and calibration have frozen.
    audit_predictions = balanced._prediction_map(adapter, audit)
    predictions = {**fit_cal_predictions, **audit_predictions}
    for episode in episodes:
        _validated_dual_head_prediction(episode, predictions[episode.seed])
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
                predictions,
                calibration,
                runtime,
            ),
            "audit": balanced._runtime_metrics(predictions, audit, runtime),
        }
    membership_diagnostics = {
        "fit": _selected_membership_confidence_diagnostics(fit, predictions),
        "calibration": _selected_membership_confidence_diagnostics(
            calibration,
            predictions,
        ),
        "audit": _selected_membership_confidence_diagnostics(audit, predictions),
    }
    trained_state = adapter.state_dict()
    membership_state, non_membership_state = _partition_membership_state(trained_state)
    action_state, shared_state = balanced._partition_action_branch_state(
        non_membership_state
    )
    trained_non_membership_sha256 = balanced._state_digest(non_membership_state)
    action_sha256 = balanced._state_digest(action_state)
    shared_sha256 = balanced._state_digest(shared_state)
    selector_prediction_sha256 = _selector_prediction_digest(
        [*fit, *calibration, *audit],
        predictions,
    )
    invariants = _implementation_invariants(
        fold,
        reference_fold,
        initial_non_membership_state_sha256=initial_non_membership_state_sha256,
        normalization_sha256=normalization_sha256,
        trained_non_membership_state_sha256=trained_non_membership_sha256,
        selector_action_branch_state_sha256=action_sha256,
        shared_non_action_branch_state_sha256=shared_sha256,
        raw_action_metrics=raw,
        selector_prediction_sha256=selector_prediction_sha256,
    )
    return {
        "name": DUAL_HEAD_ARM_NAME,
        "selector_objective_controls": dict(SELECTOR_OBJECTIVE_CONFIG),
        "auxiliary_membership_controls": dict(AUXILIARY_MEMBERSHIP_CONFIG),
        "member_seed": member_seed,
        "history": history,
        "trained_state_sha256": balanced._state_digest(trained_state),
        "membership_head_state_sha256": balanced._state_digest(membership_state),
        "trained_non_membership_state_sha256": trained_non_membership_sha256,
        "selector_action_branch_state_sha256": action_sha256,
        "shared_non_action_branch_state_sha256": shared_sha256,
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
        "selected_membership_confidence_diagnostics": membership_diagnostics,
    }


def _dual_head_summary(folds: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    audit_rows = [
        fold["dual_head"]["raw_action_metrics"]["audit"] for fold in folds
    ]
    calibrated = [
        bool(fold["dual_head"]["calibration"]["success"]) for fold in folds
    ]
    audit_runtime_eligible: list[bool] = []
    for fold, calibration_success in zip(folds, calibrated, strict=True):
        runtime_metrics = fold["dual_head"].get("runtime_metrics")
        audit_metrics = (
            runtime_metrics.get("audit")
            if isinstance(runtime_metrics, Mapping) else
            None
        )
        audit_runtime_eligible.append(
            calibration_success
            and isinstance(audit_metrics, Mapping)
            and audit_metrics.get("offline_deployment_eligible") is True
        )
    return {
        "outer_audit_micro": balanced._sum_raw(audit_rows),
        "calibration_successful_folds": sum(calibrated),
        "calibration_failed_folds": [
            int(fold["fold"])
            for fold, success in zip(folds, calibrated, strict=True)
            if not success
        ],
        "audit_runtime_eligible_folds": sum(audit_runtime_eligible),
        "audit_runtime_eligible_fold_indices": [
            int(fold["fold"])
            for fold, eligible in zip(folds, audit_runtime_eligible, strict=True)
            if eligible
        ],
        "calibrated_audit_runtime_ineligible_folds": [
            int(fold["fold"])
            for fold, success, eligible in zip(
                folds,
                calibrated,
                audit_runtime_eligible,
                strict=True,
            )
            if success and not eligible
        ],
    }


def _dual_head_adaptive_development_gate(
    summary: Mapping[str, Any],
    *,
    epochs: int = SCREENING_EPOCHS,
) -> dict[str, Any]:
    result = balanced._adaptive_development_gate(summary, epochs=epochs)
    result.update(
        {
            "third_adaptive_development_screen": True,
            "adaptive_development_screen_sequence": 3,
            "independent_statistical_validation": False,
            "specified_after_observing_balanced_membership_result": True,
            "specified_after_observing_unweighted_membership_result": True,
            "preregistered_before_membership_cv30_audit": False,
            "preregistered_before_unweighted_membership_cv30_audit": False,
            "preregistered_before_dual_head_membership_cv30_audit": True,
        }
    )
    return result


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Third adaptive CV30 screen for a frozen plain selector with a "
            "detached auxiliary membership-confidence head."
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
        "--plain-reference",
        type=Path,
        default=DEFAULT_PLAIN_REFERENCE,
    )
    parser.add_argument(
        "--balanced-reference",
        type=Path,
        default=DEFAULT_BALANCED_REFERENCE,
    )
    parser.add_argument(
        "--unweighted-reference",
        type=Path,
        default=DEFAULT_UNWEIGHTED_REFERENCE,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cpu-threads", type=int, default=1)
    return parser


def _validate_new_output_path(
    output: Path,
    protected_inputs: Sequence[Path],
) -> None:
    balanced._validate_output_path(output, protected_inputs)
    if output.exists() or output.is_symlink():
        raise ValueError(
            f"dual-head CV30 output already exists; refusing to overwrite: {output}"
        )


def _reserve_new_output_path(output: Path) -> None:
    _validate_new_output_path(output, ())
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        output.touch(exist_ok=False)
    except FileExistsError as error:
        raise ValueError(
            f"dual-head CV30 output already exists; refusing to overwrite: {output}"
        ) from error


def main() -> None:
    args = _argument_parser().parse_args()
    if args.cpu_threads <= 0:
        raise ValueError("cpu threads must be positive")
    if _selector_training_control_differences():
        raise AssertionError("dual-head selector differs from frozen plain training")
    if SELECTOR_TRAINING_CONFIG["parent_copy_weight"] != 0.1:
        raise AssertionError("dual-head selector must retain plain parent_copy=0.1")
    if (
        MEMBERSHIP_CONFIDENCE_LOSS_WEIGHT != 12.0
        or MEMBERSHIP_CONFIDENCE_LOSS_MODE != "unweighted"
    ):
        raise AssertionError("dual-head auxiliary membership controls changed")

    script_path = Path(__file__)
    plain_script_path = script_path.with_name("compare_plain_certified_set_cv30.py")
    balanced_script_path = script_path.with_name(
        "compare_certified_membership_cv30.py"
    )
    unweighted_script_path = script_path.with_name(
        "compare_unbalanced_membership_cv30.py"
    )
    legacy_script_path = script_path.with_name("compare_preferred_objectives_cv.py")
    helper_path = script_path.with_name("train_temporal_residual_adapter.py")
    protected = (
        args.failure,
        args.expansion_inventory,
        args.parent,
        args.plain_reference,
        args.balanced_reference,
        args.unweighted_reference,
        script_path,
        plain_script_path,
        balanced_script_path,
        unweighted_script_path,
        legacy_script_path,
        helper_path,
    )
    _validate_new_output_path(args.output, protected)

    torch.set_num_threads(args.cpu_threads)
    torch.use_deterministic_algorithms(True)
    failure = balanced._read_json(args.failure)
    expansion = balanced._read_json(args.expansion_inventory)
    plain_reference = _read_frozen_reference(
        args.plain_reference,
        expected_sha256=PLAIN_REFERENCE_SHA256,
        label="plain",
    )
    balanced_reference = _read_frozen_reference(
        args.balanced_reference,
        expected_sha256=BALANCED_REFERENCE_SHA256,
        label="balanced membership",
    )
    unweighted_reference = _read_frozen_reference(
        args.unweighted_reference,
        expected_sha256=UNWEIGHTED_REFERENCE_SHA256,
        label="unweighted membership",
    )
    reference_folds = _reference_fold_map(plain_reference)
    prior_results = {
        "balanced_membership": _validate_prior_adaptive_reference(
            balanced_reference,
            kind="certified_membership_adaptive_development_screen_cv30",
            sequence=1,
        ),
        "unweighted_membership": _validate_prior_adaptive_reference(
            unweighted_reference,
            kind="unweighted_membership_second_adaptive_development_screen_cv30",
            sequence=2,
        ),
    }
    prior_results["plain"] = {
        "adaptive_development_screen_sequence": 0,
        "gate_passed": plain_reference.get(
            "preregistered_e6_promotion_gate",
            {},
        ).get("passed") is True,
        "outer_audit_micro": plain_reference.get("plain_summary", {}).get(
            "outer_audit_micro"
        ),
    }

    parent_sha256 = file_sha256(args.parent)
    if parent_sha256 != failure.get("parent_checkpoint_sha256"):
        raise ValueError("parent checkpoint hash does not match failure diagnostics")
    for name, reference in (
        ("plain", plain_reference),
        ("balanced", balanced_reference),
        ("unweighted", unweighted_reference),
    ):
        provenance = reference.get("input_provenance")
        if not isinstance(provenance, Mapping) or provenance.get(
            "parent_sha256"
        ) != parent_sha256:
            raise ValueError(f"{name} reference parent hash differs")

    legacy_selected = balanced._select_training_inventory(failure)
    expansion_selected = balanced._select_expansion_inventory(
        expansion,
        checkpoint_sha256=parent_sha256,
    )
    legacy_triplets, legacy_provenance = balanced._verify_training_sources(
        legacy_selected
    )
    expansion_triplets, expansion_provenance = balanced._verify_expansion_sources(
        expansion_selected,
        checkpoint=args.parent,
    )
    triplets, source_provenance = balanced._merge_verified_sources(
        legacy_triplets,
        legacy_provenance,
        expansion_triplets,
        expansion_provenance,
    )
    _validate_new_output_path(
        args.output,
        [path for triplet in triplets for path in triplet],
    )

    config = _dual_head_adapter_config(failure)
    parent, _metadata = load_checkpoint(args.parent, device="cpu")
    parent.cpu().eval()
    for parameter in parent.parameters():
        parameter.requires_grad_(False)
    if parent.config.recurrent_size != config.recurrent_size:
        raise ValueError("parent recurrent size does not match the fixed adapter")
    feature_adapter = ResidualCorrectionAdapter(config)
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
    if tuple(episode.seed for episode in raw_episodes) != balanced.ALL_TRAINING_SEEDS:
        raise ValueError("loaded episodes do not match the ordered CV30 whitelist")

    fold_reports: list[dict[str, Any]] = []
    for fold in balanced._fixed_cv30_folds():
        fold_seed = BASE_SEED + fold.index * 100_003
        normalized = [balanced._clone_episode(episode) for episode in raw_episodes]
        torch.manual_seed(fold_seed)
        adapter = ResidualCorrectionAdapter(config)
        fit = balanced._episodes_by_seed(normalized, fold.fit_seeds)
        balanced._normalize(adapter, fit, normalized)
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
        initial_state = _snapshot_state(adapter.state_dict())
        initial_membership_state, initial_non_membership_state = (
            _partition_membership_state(initial_state)
        )
        initial_state_sha256 = balanced._state_digest(initial_state)
        initial_membership_state_sha256 = balanced._state_digest(
            initial_membership_state
        )
        initial_non_membership_sha256 = balanced._state_digest(
            initial_non_membership_state
        )
        normalization_sha256 = balanced._state_digest(
            {
                "feature_mean": adapter.feature_mean,
                "feature_scale": adapter.feature_scale,
            }
        )
        member_seed = fold_seed + 1_009
        dual_head = _run_dual_head_arm(
            adapter,
            normalized,
            fold,
            reference_folds[fold.index],
            member_seed=member_seed,
            collision_weights=collision_weights,
            physical_weights=physical_weights,
            initial_non_membership_state_sha256=initial_non_membership_sha256,
            normalization_sha256=normalization_sha256,
        )
        split_acquisition = {
            name: balanced._split_acquisition_audit(seeds)
            for name, seeds in (
                ("fit", fold.fit_seeds),
                ("calibration", fold.calibration_seeds),
                ("audit", fold.audit_seeds),
            )
        }
        fold_reports.append(
            {
                "fold": fold.index,
                "fit_seeds": list(fold.fit_seeds),
                "calibration_seeds": list(fold.calibration_seeds),
                "audit_seeds": list(fold.audit_seeds),
                "normalization_fit_seeds": list(fold.fit_seeds),
                "positive_weight_fit_seeds": list(fold.fit_seeds),
                "split_acquisition": split_acquisition,
                "fold_seed": fold_seed,
                "member_seed": member_seed,
                "initial_state_sha256": initial_state_sha256,
                "initial_membership_head_state_sha256": (
                    initial_membership_state_sha256
                ),
                "initial_non_membership_state_sha256": (
                    initial_non_membership_sha256
                ),
                "normalization_sha256": normalization_sha256,
                "collision_positive_weights": collision_weights.tolist(),
                "physical_danger_positive_weights": physical_weights.tolist(),
                "dual_head": dual_head,
            }
        )

    summary = _dual_head_summary(fold_reports)
    verified_reference_sha256 = _reverify_frozen_reference_hashes(
        plain_path=args.plain_reference,
        balanced_path=args.balanced_reference,
        unweighted_path=args.unweighted_reference,
    )
    report = {
        "schema_version": 1,
        "kind": "dual_head_membership_third_adaptive_development_screen_cv30",
        "training_only": True,
        "deployment_artifact_written": False,
        "formal_deployment_artifact_written": False,
        "deployment_eligible": False,
        "acceptance_claim": False,
        "adaptive_development_screen": True,
        "adaptive_development_screen_sequence": 3,
        "independent_statistical_validation": False,
        "specified_after_observing_plain_cv30_result": True,
        "specified_after_observing_balanced_membership_result": True,
        "specified_after_observing_unweighted_membership_result": True,
        "objective_arms": [DUAL_HEAD_ARM_NAME],
        "variant_objectives_evaluated": [],
        "objective_scope": (
            "one fixed plain selector plus one detached auxiliary unweighted "
            "membership-confidence head; no epoch, seed, retry, mode, threshold, "
            "or weight alternatives"
        ),
        "single_change_audit": {
            "selector_training_control_differences": (
                _selector_training_control_differences()
            ),
            **THIRD_ADAPTIVE_SCREEN_CONTEXT,
        },
        "selector_objective": dict(SELECTOR_OBJECTIVE_CONFIG),
        "auxiliary_membership_objective": dict(AUXILIARY_MEMBERSHIP_CONFIG),
        "prior_observed_results": prior_results,
        "audit_used_during_fit_or_calibration": False,
        "audit_used_for_threshold_epoch_seed_fold_mode_weight_or_retry_selection": False,
        "audit_used_for_after_freeze_adaptive_screen": True,
        "audit_prediction_policy": (
            "each dual-head fold predicts its audit split exactly once after "
            "training and calibration freeze; the immutable plain reference is "
            "not retrained and audit is not re-predicted for invariants"
        ),
        "runtime_score_semantics": (
            "unchanged mean-onset gate AND auxiliary selected-membership confidence; "
            "candidate and agreement remain the frozen plain selector"
        ),
        "data_isolation": {
            "legacy_training_seeds": list(balanced.LEGACY_TRAINING_SEEDS),
            "expansion_training_seeds": list(balanced.EXPANSION_TRAINING_SEEDS),
            "ordered_interleaved_training_seeds": list(
                balanced.ALL_TRAINING_SEEDS
            ),
            "prohibited_source_seeds": sorted(balanced.PROHIBITED_SOURCE_SEEDS),
            "selection_before_path_access": True,
            "nontraining_path_fields_accessed": False,
            "acquisition_cohorts": {
                balanced.LEGACY_ACQUISITION_COHORT: len(
                    balanced.LEGACY_TRAINING_SEEDS
                ),
                balanced.EXPANSION_ACQUISITION_COHORT: len(
                    balanced.EXPANSION_TRAINING_SEEDS
                ),
            },
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
            "acquisition_cohorts_strictly_interleaved_within_each_split": True,
            "legacy_role_assignment_matches_fixed_cv15": True,
            "outer_audit_covers_each_training_episode_once": True,
            "audit_predictions_per_dual_head_fold": 1,
        },
        "input_provenance": {
            "failure": str(args.failure),
            "failure_sha256": file_sha256(args.failure),
            "expansion_inventory": str(args.expansion_inventory),
            "expansion_inventory_sha256": file_sha256(args.expansion_inventory),
            "parent": str(args.parent),
            "parent_sha256": parent_sha256,
            "plain_reference": str(args.plain_reference),
            "plain_reference_sha256": verified_reference_sha256["plain"],
            "balanced_reference": str(args.balanced_reference),
            "balanced_reference_sha256": verified_reference_sha256["balanced"],
            "unweighted_reference": str(args.unweighted_reference),
            "unweighted_reference_sha256": verified_reference_sha256[
                "unweighted"
            ],
            "frozen_reference_hash_verification": {
                "verified_before_content_read": True,
                "reverified_after_training": True,
                "sha256": verified_reference_sha256,
            },
            "experiment_script": str(script_path),
            "experiment_script_sha256": file_sha256(script_path),
            "plain_cv30_script": str(plain_script_path),
            "plain_cv30_script_sha256": file_sha256(plain_script_path),
            "balanced_membership_cv30_script": str(balanced_script_path),
            "balanced_membership_cv30_script_sha256": file_sha256(
                balanced_script_path
            ),
            "unweighted_membership_cv30_script": str(unweighted_script_path),
            "unweighted_membership_cv30_script_sha256": file_sha256(
                unweighted_script_path
            ),
            "legacy_cv_script": str(legacy_script_path),
            "legacy_cv_script_sha256": file_sha256(legacy_script_path),
            "training_helper": str(helper_path),
            "training_helper_sha256": file_sha256(helper_path),
        },
        "experiment_config": {
            "epochs": SCREENING_EPOCHS,
            "base_seed": BASE_SEED,
            "device": "cpu",
            "cpu_threads": args.cpu_threads,
            "deterministic_algorithms": True,
            "adapter_config": asdict(config),
            "label_config": balanced.LABEL_CONFIG,
            "selector_training_config": SELECTOR_TRAINING_CONFIG,
            "auxiliary_membership_config": AUXILIARY_MEMBERSHIP_CONFIG,
            "ensemble_size_screening_override": 1,
            "calibration_grid_changed_from_plain": False,
            "gradient_clipping": {
                "max_norm": balanced.GRADIENT_CLIP_MAX_NORM,
                "separate_action_recurrent_semantics": (
                    balanced.SEPARATE_ACTION_GRADIENT_CLIP_SEMANTICS
                ),
                "action_group_modules": list(balanced.ACTION_BRANCH_MODULE_NAMES),
                "membership_head_gradient_group": "independent_auxiliary_group",
                "shared_safety_group": "all_other_trainable_member_parameters",
                "non_separate_architecture_semantics": (
                    balanced.GLOBAL_GRADIENT_CLIP_SEMANTICS
                ),
            },
        },
        "folds": fold_reports,
        "dual_head_summary": summary,
        "adaptive_development_gate": _dual_head_adaptive_development_gate(summary),
    }
    _reserve_new_output_path(args.output)
    balanced._write_json_atomic(args.output, report)
    print(
        json.dumps(
            {
                "artifact": str(args.output),
                "sha256": file_sha256(args.output),
                "training_only": True,
                "adaptive_development_screen": True,
                "adaptive_development_screen_sequence": 3,
                "independent_statistical_validation": False,
                "eligible_for_fixed_followup": report[
                    "adaptive_development_gate"
                ]["eligible_for_fixed_followup"],
            }
        )
    )


if __name__ == "__main__":
    main()
