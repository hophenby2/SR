"""One-shot adaptive CV30 screen for certified-action membership logits.

This protocol was specified after the negative plain certified-set CV30 result.
It is therefore an adaptive development screen, not an independent statistical
validation.  The three outer audit folds are still predicted exactly once after
training and calibration freeze.  Fixed membership diagnostics are descriptive
only and cannot select a threshold, epoch, seed, retry, or objective variant.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch

from stg_lab.provenance import file_sha256
from stg_lab.residual_adapter import ResidualCorrectionAdapter
from stg_lab.training import load_checkpoint

if __package__:
    from .compare_plain_certified_set_cv30 import (
        ALL_TRAINING_SEEDS,
        DEFAULT_EXPANSION_INVENTORY,
        EXPANSION_ACQUISITION_COHORT,
        EXPANSION_TRAINING_SEEDS,
        LEGACY_ACQUISITION_COHORT,
        LEGACY_TRAINING_SEEDS,
        _acquisition_cohort,
        _fixed_cv30_folds,
        _merge_verified_sources,
        _select_expansion_inventory,
        _split_acquisition_audit,
        _verify_expansion_sources,
    )
    from .compare_preferred_objectives_cv import (
        DEFAULT_FAILURE,
        DEFAULT_PARENT,
        LABEL_CONFIG,
        PROHIBITED_SOURCE_SEEDS,
        Fold,
        _adapter_config,
        _clone_episode,
        _episodes_by_seed,
        _partition_action_branch_state,
        _prediction_map,
        _raw_action_metrics,
        _read_json,
        _runtime_metrics,
        _select_training_inventory,
        _state_digest,
        _sum_raw,
        _validate_output_path,
        _verify_training_sources,
        _write_json_atomic,
    )
    from .train_temporal_residual_adapter import (
        ACTION_BRANCH_MODULE_NAMES,
        EARLY_ONSET_MINIMUM_LEAD_DECISIONS,
        FUTURE_ONSET_HORIZON_DECISIONS,
        GLOBAL_GRADIENT_CLIP_SEMANTICS,
        GRADIENT_CLIP_MAX_NORM,
        SEPARATE_ACTION_GRADIENT_CLIP_SEMANTICS,
        EpisodeFeatures,
        _calibrate,
        _collision_positive_weights,
        _future_onset_calibration_diagnostics,
        _load_episode,
        _normalize,
        _physical_danger_positive_weights,
        _train_member,
    )
else:  # pragma: no cover - exercised by the real script invocation
    from compare_plain_certified_set_cv30 import (
        ALL_TRAINING_SEEDS,
        DEFAULT_EXPANSION_INVENTORY,
        EXPANSION_ACQUISITION_COHORT,
        EXPANSION_TRAINING_SEEDS,
        LEGACY_ACQUISITION_COHORT,
        LEGACY_TRAINING_SEEDS,
        _acquisition_cohort,
        _fixed_cv30_folds,
        _merge_verified_sources,
        _select_expansion_inventory,
        _split_acquisition_audit,
        _verify_expansion_sources,
    )
    from compare_preferred_objectives_cv import (
        DEFAULT_FAILURE,
        DEFAULT_PARENT,
        LABEL_CONFIG,
        PROHIBITED_SOURCE_SEEDS,
        Fold,
        _adapter_config,
        _clone_episode,
        _episodes_by_seed,
        _partition_action_branch_state,
        _prediction_map,
        _raw_action_metrics,
        _read_json,
        _runtime_metrics,
        _select_training_inventory,
        _state_digest,
        _sum_raw,
        _validate_output_path,
        _verify_training_sources,
        _write_json_atomic,
    )
    from train_temporal_residual_adapter import (
        ACTION_BRANCH_MODULE_NAMES,
        EARLY_ONSET_MINIMUM_LEAD_DECISIONS,
        FUTURE_ONSET_HORIZON_DECISIONS,
        GLOBAL_GRADIENT_CLIP_SEMANTICS,
        GRADIENT_CLIP_MAX_NORM,
        SEPARATE_ACTION_GRADIENT_CLIP_SEMANTICS,
        EpisodeFeatures,
        _calibrate,
        _collision_positive_weights,
        _future_onset_calibration_diagnostics,
        _load_episode,
        _normalize,
        _physical_danger_positive_weights,
        _train_member,
    )


MEMBERSHIP_ARM_NAME = "certified_membership"
SCREENING_EPOCHS = 6
BASE_SEED = 20260901
DEFAULT_OUTPUT = Path(
    "artifacts/policy-humanlike-highres-okuu3-certified-membership-cv30-e6.json"
)

MEMBERSHIP_TRAINING_CONFIG = {
    "learning_rate": 3e-4,
    "weight_decay": 1e-3,
    "chunk_length": 128,
    "gate_positive_weight": 8.0,
    "action_loss_weight": 0.0,
    "preferred_action_loss_weight": 12.0,
    "preferred_action_uniform_loss_weight": 0.0,
    "preferred_action_tiebreak_loss_weight": 0.0,
    "preferred_action_rank_loss_weight": 0.0,
    "preferred_action_rank_margin": 1.0,
    "safety_candidate_loss_weight": 0.0,
    "parent_copy_weight": 0.0,
    "collision_loss_weight": 0.0,
    "minimum_margin_loss_weight": 0.0,
    "physical_danger_loss_weight": 8.0,
    "maximum_collision_positive_weight": 24.0,
    "maximum_physical_danger_positive_weight": 24.0,
    "all_collision_row_weight": 0.25,
    "episode_bootstrap": False,
}

MEMBERSHIP_OBJECTIVE_CONFIG = {
    "schema": "independent_certified_action_membership_balanced_bce",
    "action_logit_mode": "certified_membership",
    "target_rows": "gate_valid_and_positive_only",
    "row_balance": "positive_actions_0.5_negative_actions_0.5",
    "ensemble_candidate": "argmax_mean_membership_probability",
    "action_confidence": "selected_mean_membership_probability",
    "preferred_action_loss_weight": 12.0,
    "parent_copy_weight": 0.0,
}

ADAPTIVE_SCREEN_GATE = {
    "screening_epochs": SCREENING_EPOCHS,
    "expected_outer_audit_targets": 690,
    "minimum_calibration_successful_folds": 2,
    "minimum_audit_runtime_eligible_folds": 2,
    "all_calibration_successful_folds_must_be_audit_runtime_eligible": True,
    "reference_plain_cv30": {
        "targets": 690,
        "equivalent_top1": 138,
        "direction_correct": 138,
        "speed_correct": 437,
    },
}

MEMBERSHIP_DIAGNOSTIC_POLICY = {
    "computed_after_training_and_calibration_freeze": True,
    "audit_predicted_once": True,
    "membership_threshold": 0.5,
    "ece_equal_width_bins": 10,
    "auprc_tie_break": "original_flattened_cell_order",
    "cardinality_buckets": ["1", "2", "3-4", "5-8", "9+"],
    "product_score": "mean_onset_probability_times_selected_membership_probability",
    "product_score_used_for_calibration": False,
    "product_score_used_by_runtime": False,
    "product_score_used_for_gate": False,
    "diagnostics_used_for_threshold_selection": False,
    "diagnostics_used_for_epoch_selection": False,
    "diagnostics_used_for_seed_selection": False,
    "diagnostics_used_for_retry": False,
}


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _quantile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _score_summary(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "minimum": None,
            "q05": None,
            "q25": None,
            "median": None,
            "q75": None,
            "q95": None,
            "maximum": None,
            "mean": None,
        }
    return {
        "count": len(values),
        "minimum": min(values),
        "q05": _quantile(values, 0.05),
        "q25": _quantile(values, 0.25),
        "median": _quantile(values, 0.5),
        "q75": _quantile(values, 0.75),
        "q95": _quantile(values, 0.95),
        "maximum": max(values),
        "mean": sum(values) / len(values),
    }


def _average_precision(scores: Sequence[float], labels: Sequence[bool]) -> float | None:
    if len(scores) != len(labels):
        raise ValueError("AUCPR scores and labels do not align")
    positives = sum(labels)
    if positives == 0:
        return None
    order = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
    true_positives = 0
    precision_sum = 0.0
    for rank, index in enumerate(order, start=1):
        if labels[index]:
            true_positives += 1
            precision_sum += true_positives / rank
    return precision_sum / positives


def _reliability(
    scores: Sequence[float],
    labels: Sequence[bool],
    *,
    bins: int = 10,
) -> dict[str, Any]:
    if len(scores) != len(labels):
        raise ValueError("reliability scores and labels do not align")
    if bins <= 0:
        raise ValueError("reliability bin count must be positive")
    records: list[dict[str, Any]] = []
    weighted_gap = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        members = [
            position
            for position, score in enumerate(scores)
            if score >= lower and (score < upper or index == bins - 1)
        ]
        count = len(members)
        mean_score = (
            sum(scores[position] for position in members) / count if count else None
        )
        observed = (
            sum(labels[position] for position in members) / count if count else None
        )
        gap = (
            abs(mean_score - observed)
            if mean_score is not None and observed is not None
            else None
        )
        if gap is not None:
            weighted_gap += count * gap
        records.append(
            {
                "lower_inclusive": lower,
                "upper_inclusive_only_for_last_bin": upper,
                "count": count,
                "mean_probability": mean_score,
                "observed_rate": observed,
                "absolute_gap": gap,
            }
        )
    return {
        "bins": bins,
        "expected_calibration_error": (weighted_gap / len(scores) if scores else None),
        "reliability_bins": records,
    }


def _cardinality_bucket(cardinality: torch.Tensor, name: str) -> torch.Tensor:
    if name == "1":
        return cardinality == 1
    if name == "2":
        return cardinality == 2
    if name == "3-4":
        return (cardinality >= 3) & (cardinality <= 4)
    if name == "5-8":
        return (cardinality >= 5) & (cardinality <= 8)
    if name == "9+":
        return cardinality >= 9
    raise ValueError(f"unknown cardinality bucket: {name}")


def _positive_rows(episode: EpisodeFeatures) -> torch.Tensor:
    return episode.gate_valid & (episode.gate_targets > 0.0)


def _early_correction_rows(episode: EpisodeFeatures) -> torch.Tensor:
    return (
        _positive_rows(episode)
        & episode.preferred_correction_required
        & (episode.anticipatory_lead_decisions >= EARLY_ONSET_MINIMUM_LEAD_DECISIONS)
        & (episode.anticipatory_lead_decisions <= FUTURE_ONSET_HORIZON_DECISIONS)
    )


def _validated_membership_prediction(
    episode: EpisodeFeatures,
    values: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    probabilities = values.get("mean_action_probabilities")
    candidates = values.get("candidates")
    finite = values.get("action_all_members_finite")
    if probabilities is None or candidates is None or finite is None:
        raise ValueError("membership prediction lacks probability/top1/finite fields")
    expected = (episode.decisions, 18)
    if probabilities.shape != expected:
        raise ValueError("mean membership probabilities do not align with episode")
    if candidates.shape != (episode.decisions,) or finite.shape != (episode.decisions,):
        raise ValueError("membership top1 or finite mask does not align with episode")
    if finite.dtype != torch.bool:
        raise ValueError("membership finite mask must be Boolean")
    finite_probabilities = torch.isfinite(probabilities).all(dim=-1)
    if bool((finite & ~finite_probabilities).any()):
        raise ValueError("finite membership rows contain nonfinite probabilities")
    if bool(finite.any()):
        selected = probabilities.argmax(dim=-1)
        if not torch.equal(candidates[finite], selected[finite]):
            raise ValueError("membership candidates are not mean-probability argmax")
        confidence = values.get("action_confidence")
        if confidence is None or confidence.shape != (episode.decisions,):
            raise ValueError("membership prediction lacks aligned confidence")
        if not torch.allclose(
            confidence[finite],
            probabilities.amax(dim=-1)[finite],
            rtol=0.0,
            atol=1e-7,
        ):
            raise ValueError("membership confidence is not selected probability")
    return probabilities, candidates, finite


def _membership_metrics(
    episodes: Sequence[EpisodeFeatures],
    predictions: Mapping[int, Mapping[str, torch.Tensor]],
    mask_for_episode: Callable[[EpisodeFeatures], torch.Tensor],
) -> dict[str, Any]:
    probability_rows: list[torch.Tensor] = []
    target_rows: list[torch.Tensor] = []
    candidate_rows: list[torch.Tensor] = []
    preferred_rows: list[torch.Tensor] = []
    equivalent_rows: list[torch.Tensor] = []
    finite_rows: list[torch.Tensor] = []
    for episode in episodes:
        probabilities, candidates, finite = _validated_membership_prediction(
            episode,
            predictions[episode.seed],
        )
        mask = mask_for_episode(episode)
        if mask.shape != (episode.decisions,) or mask.dtype != torch.bool:
            raise ValueError("membership diagnostic row mask is invalid")
        probability_rows.append(probabilities[mask])
        target_rows.append(episode.preferred_action_set[mask])
        candidate_rows.append(candidates[mask])
        preferred_rows.append(episode.preferred_actions[mask])
        equivalent_rows.append(episode.preferred_equivalent_actions[mask])
        finite_rows.append(finite[mask])

    probabilities = torch.cat(probability_rows, dim=0)
    targets = torch.cat(target_rows, dim=0)
    candidates = torch.cat(candidate_rows, dim=0)
    preferred = torch.cat(preferred_rows, dim=0)
    equivalent = torch.cat(equivalent_rows, dim=0)
    finite = torch.cat(finite_rows, dim=0)
    rows = int(targets.shape[0])
    finite_count = int(finite.sum())
    if rows and bool((targets.sum(dim=-1) == 0).any()):
        raise ValueError("diagnosed positive row has an empty membership target")

    finite_probabilities = probabilities[finite]
    finite_targets = targets[finite]
    scores = [float(value) for value in finite_probabilities.flatten().tolist()]
    labels = [bool(value) for value in finite_targets.flatten().tolist()]
    positive_cells = sum(labels)
    negative_cells = len(labels) - positive_cells
    threshold_predictions = [score >= 0.5 for score in scores]
    true_positive = sum(
        predicted and target
        for predicted, target in zip(threshold_predictions, labels, strict=True)
    )
    false_positive = sum(
        predicted and not target
        for predicted, target in zip(threshold_predictions, labels, strict=True)
    )
    positive_brier_values = [
        (score - 1.0) ** 2
        for score, target in zip(scores, labels, strict=True)
        if target
    ]
    negative_brier_values = [
        score**2 for score, target in zip(scores, labels, strict=True) if not target
    ]
    positive_brier = (
        sum(positive_brier_values) / len(positive_brier_values)
        if positive_brier_values
        else None
    )
    negative_brier = (
        sum(negative_brier_values) / len(negative_brier_values)
        if negative_brier_values
        else None
    )

    true_cardinality = finite_targets.sum(dim=-1).to(torch.int64)
    predicted_cardinality = (finite_probabilities >= 0.5).sum(dim=-1).to(torch.int64)
    true_distribution = Counter(int(value) for value in true_cardinality.tolist())
    predicted_distribution = Counter(
        int(value) for value in predicted_cardinality.tolist()
    )

    finite_candidates = candidates[finite]
    negative_infinity = torch.full_like(finite_probabilities, -torch.inf)
    best_certified_scores = torch.where(
        finite_targets,
        finite_probabilities,
        negative_infinity,
    ).amax(dim=-1)
    best_rejected_scores = torch.where(
        ~finite_targets,
        finite_probabilities,
        negative_infinity,
    ).amax(dim=-1)
    certified_rejected_margins = (
        best_certified_scores - best_rejected_scores
    ).tolist()
    selected_scores = (
        finite_probabilities.gather(-1, finite_candidates.unsqueeze(-1))
        .squeeze(-1)
        .tolist()
    )
    selected_target = (
        finite_targets.gather(-1, finite_candidates.unsqueeze(-1)).squeeze(-1).tolist()
    )
    selected_equivalent = (
        equivalent[finite].gather(-1, finite_candidates.unsqueeze(-1)).squeeze(-1)
    )
    equivalent_count = int(selected_equivalent.sum())
    exact = finite_candidates == preferred[finite]
    direction = (finite_candidates % 9) == (preferred[finite] % 9)
    speed = (finite_candidates >= 9) == (preferred[finite] >= 9)
    represented_directions = finite_targets.reshape(-1, 2, 9).any(dim=1).sum(dim=-1)
    multidirection = represented_directions > 1

    reliability = _reliability(selected_scores, selected_target)
    brier = (
        sum(
            (score - float(target)) ** 2
            for score, target in zip(
                scores,
                labels,
                strict=True,
            )
        )
        / len(scores)
        if scores
        else None
    )
    cardinality_absolute_error = (
        (predicted_cardinality - true_cardinality).abs().to(torch.float32)
        if finite_count
        else torch.empty(0)
    )
    return {
        "rows": rows,
        "finite_rows": finite_count,
        "nonfinite_rows": rows - finite_count,
        "action_cells": len(scores),
        "positive_cells": positive_cells,
        "negative_cells": negative_cells,
        "brier_score": brier,
        "balanced_brier_score": (
            (positive_brier + negative_brier) / 2.0
            if positive_brier is not None and negative_brier is not None
            else None
        ),
        "positive_brier_score": positive_brier,
        "negative_brier_score": negative_brier,
        "ece": _reliability(scores, labels),
        "auprc_average_precision": _average_precision(scores, labels),
        "threshold_0_5": {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "positive_recall": _ratio(true_positive, positive_cells),
            "negative_false_positive_rate": _ratio(
                false_positive,
                negative_cells,
            ),
        },
        "cardinality": {
            "true_distribution": {
                str(key): value for key, value in sorted(true_distribution.items())
            },
            "predicted_distribution_at_0_5": {
                str(key): value for key, value in sorted(predicted_distribution.items())
            },
            "true_mean": (
                float(true_cardinality.to(torch.float32).mean())
                if finite_count
                else None
            ),
            "predicted_mean_at_0_5": (
                float(predicted_cardinality.to(torch.float32).mean())
                if finite_count
                else None
            ),
            "exact_match_rows_at_0_5": (
                int((predicted_cardinality == true_cardinality).sum())
                if finite_count
                else 0
            ),
            "mean_absolute_error_at_0_5": (
                float(cardinality_absolute_error.mean()) if finite_count else None
            ),
        },
        "selected_certified_reliability": {
            "rows": finite_count,
            "selected_target_member": sum(selected_target),
            "selected_target_member_rate": _ratio(sum(selected_target), finite_count),
            "mean_selected_membership_probability": (
                sum(selected_scores) / finite_count if finite_count else None
            ),
            "brier_score": (
                sum(
                    (score - float(target)) ** 2
                    for score, target in zip(
                        selected_scores,
                        selected_target,
                        strict=True,
                    )
                )
                / finite_count
                if finite_count
                else None
            ),
            "ece": reliability,
        },
        "certified_vs_rejected_score_margin": {
            "semantics": (
                "per-row maximum certified membership probability minus maximum "
                "rejected membership probability"
            ),
            "summary": _score_summary(certified_rejected_margins),
            "certified_top1_rows": sum(selected_target),
            "certified_top1_rate": _ratio(
                sum(selected_target),
                finite_count,
            ),
        },
        "selected_action_quality": {
            "certified_equivalent": equivalent_count,
            "certified_equivalent_rate": _ratio(equivalent_count, finite_count),
            "exact_given_equivalent": int((exact & selected_equivalent).sum()),
            "exact_rate_given_equivalent": _ratio(
                int((exact & selected_equivalent).sum()),
                equivalent_count,
            ),
            "direction_given_equivalent": int((direction & selected_equivalent).sum()),
            "direction_rate_given_equivalent": _ratio(
                int((direction & selected_equivalent).sum()),
                equivalent_count,
            ),
            "speed_given_equivalent": int((speed & selected_equivalent).sum()),
            "speed_rate_given_equivalent": _ratio(
                int((speed & selected_equivalent).sum()),
                equivalent_count,
            ),
            "multidirection_target_rows": int(multidirection.sum()),
            "canonical_direction_on_multidirection_rows": int(
                (direction & multidirection).sum()
            ),
            "canonical_direction_rate_on_multidirection_rows": _ratio(
                int((direction & multidirection).sum()),
                int(multidirection.sum()),
            ),
        },
    }


def _product_score_diagnostics(
    episodes: Sequence[EpisodeFeatures],
    predictions: Mapping[int, Mapping[str, torch.Tensor]],
) -> dict[str, Any]:
    scopes: dict[str, dict[str, list[float] | Counter[str]]] = {
        "changed_candidate_before_physical_veto": {
            "beneficial": [],
            "disallowed": [],
            "beneficial_onset": [],
            "disallowed_onset": [],
            "beneficial_membership": [],
            "disallowed_membership": [],
            "reasons": Counter(),
        },
        "changed_candidate_after_fixed_physical_veto_0_5": {
            "beneficial": [],
            "disallowed": [],
            "beneficial_onset": [],
            "disallowed_onset": [],
            "beneficial_membership": [],
            "disallowed_membership": [],
            "reasons": Counter(),
        },
    }
    for episode in episodes:
        values = predictions[episode.seed]
        probabilities, candidates, action_finite = _validated_membership_prediction(
            episode,
            values,
        )
        mean_gate = values["mean_gate"]
        minimum_gate = values["minimum_gate"]
        if mean_gate.shape != (episode.decisions,) or minimum_gate.shape != (
            episode.decisions,
        ):
            raise ValueError("onset probabilities do not align with episode")
        candidate_membership = probabilities.gather(
            -1,
            candidates.unsqueeze(-1),
        ).squeeze(-1)
        product_score = mean_gate * candidate_membership
        finite = (
            action_finite
            & torch.isfinite(mean_gate)
            & torch.isfinite(minimum_gate)
            & torch.isfinite(candidate_membership)
            & (candidates != episode.parent_actions)
        )
        positive = _positive_rows(episode)
        correction_required = positive & episode.preferred_correction_required
        equivalent = episode.preferred_equivalent_actions.gather(
            -1,
            candidates.unsqueeze(-1),
        ).squeeze(-1)
        actually_safe = episode.evaluation_safe_actions.gather(
            -1,
            candidates.unsqueeze(-1),
        ).squeeze(-1)
        beneficial = correction_required & equivalent & actually_safe
        physical = values.get("physical_danger_probabilities")
        if (
            physical is None
            or physical.ndim != 3
            or physical.shape[1:]
            != (
                episode.decisions,
                18,
            )
        ):
            raise ValueError("membership product diagnostics require physical heads")
        candidate_indices = (
            candidates.unsqueeze(0)
            .unsqueeze(-1)
            .expand(
                physical.shape[0],
                episode.decisions,
                1,
            )
        )
        physical_scores = physical.gather(-1, candidate_indices).squeeze(-1)
        fixed_physical_safe = torch.isfinite(physical_scores).all(dim=0) & (
            physical_scores <= 0.5
        ).all(dim=0)

        for scope_name, scope_mask in (
            ("changed_candidate_before_physical_veto", finite),
            (
                "changed_candidate_after_fixed_physical_veto_0_5",
                finite & fixed_physical_safe,
            ),
        ):
            record = scopes[scope_name]
            allowed = scope_mask & beneficial
            disallowed = scope_mask & ~beneficial
            for key, tensor in (
                ("beneficial", product_score[allowed]),
                ("disallowed", product_score[disallowed]),
                ("beneficial_onset", mean_gate[allowed]),
                ("disallowed_onset", mean_gate[disallowed]),
                ("beneficial_membership", candidate_membership[allowed]),
                ("disallowed_membership", candidate_membership[disallowed]),
            ):
                values_list = record[key]
                assert isinstance(values_list, list)
                values_list.extend(float(value) for value in tensor.tolist())
            reasons = record["reasons"]
            assert isinstance(reasons, Counter)
            reasons["not_correction_required"] += int(
                (disallowed & ~correction_required).sum()
            )
            reasons["not_certified_equivalent"] += int((disallowed & ~equivalent).sum())
            reasons["not_evaluation_safe"] += int((disallowed & ~actually_safe).sum())

    result: dict[str, Any] = {}
    for name, raw in scopes.items():
        beneficial = raw["beneficial"]
        disallowed = raw["disallowed"]
        assert isinstance(beneficial, list) and isinstance(disallowed, list)
        maximum_disallowed = max(disallowed) if disallowed else None
        minimum_beneficial = min(beneficial) if beneficial else None
        maximum_beneficial = max(beneficial) if beneficial else None
        reasons = raw["reasons"]
        assert isinstance(reasons, Counter)
        result[name] = {
            "candidate_rows": len(beneficial) + len(disallowed),
            "beneficial_product_score": _score_summary(beneficial),
            "disallowed_product_score": _score_summary(disallowed),
            "beneficial_onset_probability": _score_summary(
                raw["beneficial_onset"]  # type: ignore[arg-type]
            ),
            "disallowed_onset_probability": _score_summary(
                raw["disallowed_onset"]  # type: ignore[arg-type]
            ),
            "beneficial_selected_membership_probability": _score_summary(
                raw["beneficial_membership"]  # type: ignore[arg-type]
            ),
            "disallowed_selected_membership_probability": _score_summary(
                raw["disallowed_membership"]  # type: ignore[arg-type]
            ),
            "best_beneficial_minus_max_disallowed_product_gap": (
                maximum_beneficial - maximum_disallowed
                if maximum_beneficial is not None and maximum_disallowed is not None
                else None
            ),
            "min_beneficial_minus_max_disallowed_product_gap": (
                minimum_beneficial - maximum_disallowed
                if minimum_beneficial is not None and maximum_disallowed is not None
                else None
            ),
            "overlapping_disallowed_reason_counts": dict(reasons),
        }
    return {
        "score_semantics": MEMBERSHIP_DIAGNOSTIC_POLICY["product_score"],
        "descriptive_only": True,
        "used_for_calibration_runtime_gate_or_selection": False,
        "fixed_physical_veto_diagnostic_threshold": 0.5,
        "scopes": result,
    }


def _membership_split_diagnostics(
    episodes: Sequence[EpisodeFeatures],
    predictions: Mapping[int, Mapping[str, torch.Tensor]],
) -> dict[str, Any]:
    early = _early_correction_rows
    by_cohort = {
        cohort: _membership_metrics(
            [
                episode
                for episode in episodes
                if _acquisition_cohort(episode.seed) == cohort
            ],
            predictions,
            early,
        )
        for cohort in (
            LEGACY_ACQUISITION_COHORT,
            EXPANSION_ACQUISITION_COHORT,
        )
    }
    by_cardinality = {
        bucket: _membership_metrics(
            episodes,
            predictions,
            lambda episode, bucket=bucket: (
                early(episode)
                & _cardinality_bucket(
                    episode.preferred_action_set.sum(dim=-1),
                    bucket,
                )
            ),
        )
        for bucket in MEMBERSHIP_DIAGNOSTIC_POLICY["cardinality_buckets"]
    }
    by_lead = {
        str(lead): _membership_metrics(
            episodes,
            predictions,
            lambda episode, lead=lead: (
                early(episode) & (episode.anticipatory_lead_decisions == lead)
            ),
        )
        for lead in range(
            EARLY_ONSET_MINIMUM_LEAD_DECISIONS,
            FUTURE_ONSET_HORIZON_DECISIONS + 1,
        )
    }
    return {
        "supervised_positive_rows": _membership_metrics(
            episodes,
            predictions,
            _positive_rows,
        ),
        "early_correction_required_4_10": {
            "overall": _membership_metrics(episodes, predictions, early),
            "by_acquisition_cohort": by_cohort,
            "by_true_cardinality": by_cardinality,
            "by_lead_decisions": by_lead,
        },
        "onset_times_selected_membership_score_gap": _product_score_diagnostics(
            episodes,
            predictions,
        ),
    }


def _membership_adapter_config(failure: Mapping[str, Any]) -> Any:
    base = _adapter_config(failure)
    if base.action_logit_mode != "parent_residual_joint":
        raise ValueError("fixed v81 adapter must use parent_residual_joint logits")
    result = replace(base, action_logit_mode="certified_membership")
    before = asdict(base)
    after = asdict(result)
    changed = {
        key: (before[key], after[key]) for key in before if before[key] != after[key]
    }
    if changed != {
        "action_logit_mode": (
            "parent_residual_joint",
            "certified_membership",
        )
    }:
        raise AssertionError("membership adapter changed more than action-logit mode")
    return result


def _run_membership_arm(
    adapter: ResidualCorrectionAdapter,
    episodes: list[EpisodeFeatures],
    fold: Fold,
    *,
    member_seed: int,
    collision_weights: torch.Tensor,
    physical_weights: torch.Tensor,
    membership_loss_mode: str = "balanced",
    training_config: Mapping[str, Any] = MEMBERSHIP_TRAINING_CONFIG,
    objective_config: Mapping[str, Any] = MEMBERSHIP_OBJECTIVE_CONFIG,
    arm_name: str = MEMBERSHIP_ARM_NAME,
) -> dict[str, Any]:
    fit = _episodes_by_seed(episodes, fold.fit_seeds)
    calibration = _episodes_by_seed(episodes, fold.calibration_seeds)
    audit = _episodes_by_seed(episodes, fold.audit_seeds)
    config = training_config
    torch.manual_seed(member_seed)
    history = _train_member(
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
        membership_loss_mode=membership_loss_mode,
        device="cpu",
    )

    fit_cal_predictions = _prediction_map(adapter, [*fit, *calibration])
    runtime = None
    calibration_error = None
    calibration_failure_diagnostics = None
    try:
        runtime = _calibrate(
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
        calibration_failure_diagnostics = _future_onset_calibration_diagnostics(
            fit_cal_predictions,
            fit,
            calibration,
            ensemble_size=adapter.config.ensemble_size,
        )

    # Audit prediction and all descriptive diagnostics happen only after the
    # calibration attempt has terminated.  No value below can affect a choice.
    audit_predictions = _prediction_map(adapter, audit)
    predictions = {**fit_cal_predictions, **audit_predictions}
    raw = {
        "fit": _raw_action_metrics(fit, predictions),
        "calibration": _raw_action_metrics(calibration, predictions),
        "audit": _raw_action_metrics(audit, predictions),
    }
    runtime_metrics = None
    if runtime is not None:
        runtime_metrics = {
            "fit": _runtime_metrics(predictions, fit, runtime),
            "calibration": _runtime_metrics(predictions, calibration, runtime),
            "audit": _runtime_metrics(predictions, audit, runtime),
        }
    membership_diagnostics = {
        "fit": _membership_split_diagnostics(fit, predictions),
        "calibration": _membership_split_diagnostics(calibration, predictions),
        "audit": _membership_split_diagnostics(audit, predictions),
    }
    trained_state = adapter.state_dict()
    action_state, non_action_state = _partition_action_branch_state(trained_state)
    return {
        "name": arm_name,
        "objective_controls": dict(objective_config),
        "member_seed": member_seed,
        "history": history,
        "trained_state_sha256": _state_digest(trained_state),
        "action_branch_state_sha256": _state_digest(action_state),
        "non_action_branch_state_sha256": _state_digest(non_action_state),
        "calibration": {
            "success": runtime is not None,
            "error": calibration_error,
            "runtime_config": None if runtime is None else asdict(runtime),
            "runtime_threshold_semantics": (
                "existing mean-onset AND selected-membership-confidence thresholds; "
                "the product diagnostic is not used"
            ),
            "failure_diagnostics": calibration_failure_diagnostics,
        },
        "raw_action_metrics": raw,
        "runtime_metrics": runtime_metrics,
        "membership_diagnostics": membership_diagnostics,
    }


def _membership_summary(folds: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    audit_rows = [fold["membership"]["raw_action_metrics"]["audit"] for fold in folds]
    calibration_successful = [
        bool(fold["membership"]["calibration"]["success"]) for fold in folds
    ]
    audit_runtime_eligible: list[bool] = []
    for fold, calibrated in zip(folds, calibration_successful, strict=True):
        runtime_metrics = fold["membership"].get("runtime_metrics")
        audit_metrics = (
            runtime_metrics.get("audit")
            if isinstance(runtime_metrics, Mapping)
            else None
        )
        audit_runtime_eligible.append(
            calibrated
            and isinstance(audit_metrics, Mapping)
            and audit_metrics.get("offline_deployment_eligible") is True
        )
    return {
        "outer_audit_micro": _sum_raw(audit_rows),
        "calibration_successful_folds": sum(calibration_successful),
        "calibration_failed_folds": [
            int(fold["fold"])
            for fold, calibrated in zip(folds, calibration_successful, strict=True)
            if not calibrated
        ],
        "audit_runtime_eligible_folds": sum(audit_runtime_eligible),
        "audit_runtime_eligible_fold_indices": [
            int(fold["fold"])
            for fold, eligible in zip(folds, audit_runtime_eligible, strict=True)
            if eligible
        ],
        "calibrated_audit_runtime_ineligible_folds": [
            int(fold["fold"])
            for fold, calibrated, eligible in zip(
                folds,
                calibration_successful,
                audit_runtime_eligible,
                strict=True,
            )
            if calibrated and not eligible
        ],
    }


def _meets_reference_rate(
    metrics: Mapping[str, Any],
    field: str,
    *,
    reference_numerator: int,
    reference_denominator: int,
) -> bool:
    targets = int(metrics["targets"])
    count = int(metrics[field])
    return targets > 0 and count * reference_denominator >= (
        targets * reference_numerator
    )


def _adaptive_development_gate(
    summary: Mapping[str, Any],
    *,
    epochs: int = SCREENING_EPOCHS,
) -> dict[str, Any]:
    audit = summary["outer_audit_micro"]
    reference = ADAPTIVE_SCREEN_GATE["reference_plain_cv30"]
    applicable = epochs == SCREENING_EPOCHS
    checks = {
        "calibration_succeeds_on_at_least_two_of_three_folds": (
            int(summary["calibration_successful_folds"])
            >= ADAPTIVE_SCREEN_GATE["minimum_calibration_successful_folds"]
        ),
        "audit_runtime_is_eligible_on_at_least_two_of_three_folds": (
            int(summary["audit_runtime_eligible_folds"])
            >= ADAPTIVE_SCREEN_GATE["minimum_audit_runtime_eligible_folds"]
        ),
        "every_calibrated_fold_has_eligible_audit_runtime": (
            not summary["calibrated_audit_runtime_ineligible_folds"]
            and int(summary["audit_runtime_eligible_folds"])
            == int(summary["calibration_successful_folds"])
        ),
        "outer_audit_target_inventory_is_exactly_690": (
            int(audit["targets"])
            == ADAPTIVE_SCREEN_GATE["expected_outer_audit_targets"]
        ),
        "outer_audit_all_action_top1_outputs_finite": (
            int(audit["finite_top1"]) == int(audit["targets"])
        ),
        "outer_audit_equivalent_top1_rate_at_least_plain_cv30": (
            _meets_reference_rate(
                audit,
                "equivalent_top1",
                reference_numerator=reference["equivalent_top1"],
                reference_denominator=reference["targets"],
            )
        ),
        "outer_audit_direction_rate_at_least_plain_cv30": (
            _meets_reference_rate(
                audit,
                "direction_correct",
                reference_numerator=reference["direction_correct"],
                reference_denominator=reference["targets"],
            )
        ),
        "outer_audit_speed_rate_at_least_plain_cv30": _meets_reference_rate(
            audit,
            "speed_correct",
            reference_numerator=reference["speed_correct"],
            reference_denominator=reference["targets"],
        ),
    }
    return {
        "specified_after_observing_plain_cv30_negative_result": True,
        "adaptive_development_screen": True,
        "independent_statistical_validation": False,
        "preregistered_before_membership_cv30_audit": True,
        "applicable": applicable,
        "inapplicable_reason": (
            None if applicable else "screen applies only to exactly 6 training epochs"
        ),
        "criteria": {
            **ADAPTIVE_SCREEN_GATE,
            "reference_rates": {
                field: reference[field] / reference["targets"]
                for field in (
                    "equivalent_top1",
                    "direction_correct",
                    "speed_correct",
                )
            },
            "rate_comparison_semantics": (
                "exact integer cross multiplication against fixed plain CV30 e6 "
                "ratios; no result-dependent rounding"
            ),
        },
        "checks": checks,
        "passed": applicable and all(checks.values()),
        "eligible_for_fixed_followup": applicable and all(checks.values()),
        "deployment_eligible": False,
        "acceptance_claim": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "One-shot adaptive CV30 screen for independent certified-action "
            "membership logits."
        )
    )
    parser.add_argument("--failure", type=Path, default=DEFAULT_FAILURE)
    parser.add_argument(
        "--expansion-inventory",
        type=Path,
        default=DEFAULT_EXPANSION_INVENTORY,
    )
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cpu-threads", type=int, default=1)
    args = parser.parse_args()
    if args.cpu_threads <= 0:
        raise ValueError("cpu threads must be positive")

    script_path = Path(__file__)
    plain_script_path = script_path.with_name("compare_plain_certified_set_cv30.py")
    legacy_script_path = script_path.with_name("compare_preferred_objectives_cv.py")
    helper_path = script_path.with_name("train_temporal_residual_adapter.py")
    _validate_output_path(
        args.output,
        (
            args.failure,
            args.expansion_inventory,
            args.parent,
            script_path,
            plain_script_path,
            legacy_script_path,
            helper_path,
        ),
    )

    torch.set_num_threads(args.cpu_threads)
    torch.use_deterministic_algorithms(True)
    failure = _read_json(args.failure)
    expansion = _read_json(args.expansion_inventory)
    parent_sha256 = file_sha256(args.parent)
    if parent_sha256 != failure.get("parent_checkpoint_sha256"):
        raise ValueError("parent checkpoint hash does not match failure diagnostics")

    # Reuse the CV30 selectors so every seed/role field is checked before any
    # path-bearing field, then rebuild every native triplet from source evidence.
    legacy_selected = _select_training_inventory(failure)
    expansion_selected = _select_expansion_inventory(
        expansion,
        checkpoint_sha256=parent_sha256,
    )
    legacy_triplets, legacy_provenance = _verify_training_sources(legacy_selected)
    expansion_triplets, expansion_provenance = _verify_expansion_sources(
        expansion_selected,
        checkpoint=args.parent,
    )
    triplets, source_provenance = _merge_verified_sources(
        legacy_triplets,
        legacy_provenance,
        expansion_triplets,
        expansion_provenance,
    )
    _validate_output_path(
        args.output,
        [path for triplet in triplets for path in triplet],
    )

    config = _membership_adapter_config(failure)
    parent, _metadata = load_checkpoint(args.parent, device="cpu")
    parent.cpu().eval()
    for parameter in parent.parameters():
        parameter.requires_grad_(False)
    if parent.config.recurrent_size != config.recurrent_size:
        raise ValueError("parent recurrent size does not match the v81 adapter")

    feature_adapter = ResidualCorrectionAdapter(config)
    raw_episodes = [
        _load_episode(
            parent,
            feature_adapter,
            dataset,
            report,
            manifest,
            parent_checkpoint_sha256=parent_sha256,
            device="cpu",
            chunk_length=256,
            **LABEL_CONFIG,
        )
        for dataset, report, manifest in triplets
    ]
    if tuple(episode.seed for episode in raw_episodes) != ALL_TRAINING_SEEDS:
        raise ValueError("loaded episodes do not match the ordered CV30 whitelist")

    fold_reports: list[dict[str, Any]] = []
    for fold in _fixed_cv30_folds():
        fold_seed = BASE_SEED + fold.index * 100_003
        normalized = [_clone_episode(episode) for episode in raw_episodes]
        torch.manual_seed(fold_seed)
        adapter = ResidualCorrectionAdapter(config)
        fit = _episodes_by_seed(normalized, fold.fit_seeds)
        _normalize(adapter, fit, normalized)
        collision_weights = _collision_positive_weights(
            fit,
            maximum_weight=MEMBERSHIP_TRAINING_CONFIG[
                "maximum_collision_positive_weight"
            ],
        )
        physical_weights = _physical_danger_positive_weights(
            fit,
            maximum_weight=MEMBERSHIP_TRAINING_CONFIG[
                "maximum_physical_danger_positive_weight"
            ],
        )
        initial_sha256 = _state_digest(adapter.state_dict())
        member_seed = fold_seed + 1_009
        membership = _run_membership_arm(
            adapter,
            normalized,
            fold,
            member_seed=member_seed,
            collision_weights=collision_weights,
            physical_weights=physical_weights,
        )
        split_acquisition = {
            name: _split_acquisition_audit(seeds)
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
                "initial_state_sha256": initial_sha256,
                "normalization_sha256": _state_digest(
                    {
                        "feature_mean": adapter.feature_mean,
                        "feature_scale": adapter.feature_scale,
                    }
                ),
                "collision_positive_weights": collision_weights.tolist(),
                "physical_danger_positive_weights": physical_weights.tolist(),
                "membership": membership,
            }
        )

    summary = _membership_summary(fold_reports)
    report = {
        "schema_version": 1,
        "kind": "certified_membership_adaptive_development_screen_cv30",
        "training_only": True,
        "deployment_artifact_written": False,
        "formal_deployment_artifact_written": False,
        "deployment_eligible": False,
        "acceptance_claim": False,
        "adaptive_development_screen": True,
        "independent_statistical_validation": False,
        "specified_after_observing_plain_cv30_negative_result": True,
        "objective_arms": [MEMBERSHIP_ARM_NAME],
        "variant_objectives_evaluated": [],
        "objective_scope": (
            "one independent certified-membership arm only; no exact, set-NLL, "
            "rank, tiebreak, uniform, parent-copy, or retry variant"
        ),
        "membership_objective": dict(MEMBERSHIP_OBJECTIVE_CONFIG),
        "audit_used_during_fit_or_calibration": False,
        "audit_used_for_threshold_epoch_seed_fold_or_retry_selection": False,
        "audit_used_for_after_freeze_adaptive_screen": True,
        "audit_prediction_policy": (
            "predicted exactly once after each fold calibration attempt; never "
            "used for training, normalization, calibration thresholds, retries, "
            "epoch choice, seed choice, or fold construction"
        ),
        "membership_diagnostic_policy": dict(MEMBERSHIP_DIAGNOSTIC_POLICY),
        "runtime_score_semantics": (
            "existing mean-onset AND selected-membership-confidence thresholds; "
            "onset-times-membership is descriptive only in this one-change screen"
        ),
        "data_isolation": {
            "legacy_training_seeds": list(LEGACY_TRAINING_SEEDS),
            "expansion_training_seeds": list(EXPANSION_TRAINING_SEEDS),
            "ordered_interleaved_training_seeds": list(ALL_TRAINING_SEEDS),
            "prohibited_source_seeds": sorted(PROHIBITED_SOURCE_SEEDS),
            "selection_before_path_access": True,
            "nontraining_path_fields_accessed": False,
            "acquisition_cohorts": {
                LEGACY_ACQUISITION_COHORT: len(LEGACY_TRAINING_SEEDS),
                EXPANSION_ACQUISITION_COHORT: len(EXPANSION_TRAINING_SEEDS),
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
        },
        "input_provenance": {
            "failure": str(args.failure),
            "failure_sha256": file_sha256(args.failure),
            "expansion_inventory": str(args.expansion_inventory),
            "expansion_inventory_sha256": file_sha256(args.expansion_inventory),
            "parent": str(args.parent),
            "parent_sha256": parent_sha256,
            "experiment_script": str(script_path),
            "experiment_script_sha256": file_sha256(script_path),
            "plain_cv30_script": str(plain_script_path),
            "plain_cv30_script_sha256": file_sha256(plain_script_path),
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
            "label_config": LABEL_CONFIG,
            "training_config": MEMBERSHIP_TRAINING_CONFIG,
            "ensemble_size_screening_override": 1,
            "gradient_clipping": {
                "max_norm": GRADIENT_CLIP_MAX_NORM,
                "separate_action_recurrent_semantics": (
                    SEPARATE_ACTION_GRADIENT_CLIP_SEMANTICS
                ),
                "action_group_modules": list(ACTION_BRANCH_MODULE_NAMES),
                "shared_safety_group": "all_other_trainable_member_parameters",
                "non_separate_architecture_semantics": GLOBAL_GRADIENT_CLIP_SEMANTICS,
            },
        },
        "folds": fold_reports,
        "membership_summary": summary,
        "adaptive_development_gate": _adaptive_development_gate(summary),
    }
    _write_json_atomic(args.output, report)
    print(
        json.dumps(
            {
                "artifact": str(args.output),
                "sha256": file_sha256(args.output),
                "training_only": True,
                "adaptive_development_screen": True,
                "independent_statistical_validation": False,
                "eligible_for_fixed_followup": report["adaptive_development_gate"][
                    "eligible_for_fixed_followup"
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
