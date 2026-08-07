"""Fixed 30-episode CV for the plain certified-set NLL objective.

This is intentionally separate from ``compare_preferred_objectives_cv.py``.
That experiment and its fixed 15-episode, five-arm protocol remain unchanged.
The expanded protocol combines its legacy training whitelist with the fixed v83
prefix-zero cohort and trains exactly one objective arm.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from itertools import pairwise
from pathlib import Path
from typing import Any, TypeVar

import torch

from stg_lab.provenance import file_sha256
from stg_lab.residual_adapter import ResidualCorrectionAdapter
from stg_lab.training import load_checkpoint

if __package__:
    from .collect_strict_prefix0_dagger import (
        _validate_triplet as _validate_strict_prefix0_triplet,
    )
    from .compare_preferred_objectives_cv import (
        DEFAULT_FAILURE,
        DEFAULT_PARENT,
        LABEL_CONFIG,
        PROHIBITED_SOURCE_SEEDS,
        TRAINING_CONFIG,
        Fold,
        _adapter_config,
        _clone_episode,
        _episodes_by_seed,
        _read_json,
        _required_int,
        _required_string,
        _run_arm,
        _select_training_inventory,
        _state_digest,
        _sum_raw,
        _validate_output_path,
        _verify_training_sources,
        _write_json_atomic,
    )
    from .compare_preferred_objectives_cv import (
        TRAINING_SEEDS as LEGACY_TRAINING_SEEDS,
    )
    from .train_temporal_residual_adapter import (
        ACTION_BRANCH_MODULE_NAMES,
        GLOBAL_GRADIENT_CLIP_SEMANTICS,
        GRADIENT_CLIP_MAX_NORM,
        SEPARATE_ACTION_GRADIENT_CLIP_SEMANTICS,
        _collision_positive_weights,
        _load_episode,
        _normalize,
        _physical_danger_positive_weights,
    )
else:  # pragma: no cover - exercised by the real script invocation
    from collect_strict_prefix0_dagger import (
        _validate_triplet as _validate_strict_prefix0_triplet,
    )
    from compare_preferred_objectives_cv import (
        DEFAULT_FAILURE,
        DEFAULT_PARENT,
        LABEL_CONFIG,
        PROHIBITED_SOURCE_SEEDS,
        TRAINING_CONFIG,
        Fold,
        _adapter_config,
        _clone_episode,
        _episodes_by_seed,
        _read_json,
        _required_int,
        _required_string,
        _run_arm,
        _select_training_inventory,
        _state_digest,
        _sum_raw,
        _validate_output_path,
        _verify_training_sources,
        _write_json_atomic,
    )
    from compare_preferred_objectives_cv import (
        TRAINING_SEEDS as LEGACY_TRAINING_SEEDS,
    )
    from train_temporal_residual_adapter import (
        ACTION_BRANCH_MODULE_NAMES,
        GLOBAL_GRADIENT_CLIP_SEMANTICS,
        GRADIENT_CLIP_MAX_NORM,
        SEPARATE_ACTION_GRADIENT_CLIP_SEMANTICS,
        _collision_positive_weights,
        _load_episode,
        _normalize,
        _physical_danger_positive_weights,
    )


EXPANSION_TRAINING_SEEDS = tuple(range(10311, 10326))
LEGACY_TRAINING_SEED_SET = frozenset(LEGACY_TRAINING_SEEDS)
EXPANSION_TRAINING_SEED_SET = frozenset(EXPANSION_TRAINING_SEEDS)
ALL_TRAINING_SEED_SET = LEGACY_TRAINING_SEED_SET | EXPANSION_TRAINING_SEED_SET

LEGACY_ACQUISITION_COHORT = "legacy_v81_training_15"
EXPANSION_ACQUISITION_COHORT = "v83_prefix0_disagreement_intervention_15"
PLAIN_ARM_NAME = "equivalence"
PLAIN_OBJECTIVE_CONFIG = {
    "schema": "preferred_certified_equivalence_set_nll",
    "preferred_action_loss_weight": TRAINING_CONFIG["preferred_action_loss_weight"],
    "preferred_action_uniform_loss_weight": 0.0,
    "preferred_action_tiebreak_loss_weight": 0.0,
    "preferred_action_rank_loss_weight": 0.0,
    "preferred_action_rank_margin": 1.0,
}

DEFAULT_EXPANSION_INVENTORY = Path("artifacts/prefix0-expansion-v83-inventory.json")
DEFAULT_OUTPUT = Path(
    "artifacts/policy-humanlike-highres-okuu3-plain-certified-set-cv30-e6.json"
)

EXPANSION_INVENTORY_CONTRACT = {
    "schema_version": 1,
    "kind": "strict_prefix_zero_dagger_training_inventory",
    "training_only": True,
    "acceptance_claim": False,
    "scenario": "okuu:Lunatic",
    "attack": 3,
    "intervene_on_disagreement": True,
}

PROMOTION_GATE = {
    "screening_epochs": 6,
    "minimum_calibration_successful_folds": 2,
    "minimum_audit_runtime_eligible_folds": 2,
    "all_calibration_successful_folds_must_be_audit_runtime_eligible": True,
    "reference_baseline": {
        "targets": 359,
        "equivalent_top1": 59,
        "direction_correct": 81,
        "speed_correct": 194,
    },
}

T = TypeVar("T")


def _interleave_equal(left: Sequence[T], right: Sequence[T]) -> tuple[T, ...]:
    if len(left) != len(right):
        raise ValueError("acquisition cohorts must have equal lengths to interleave")
    return tuple(value for pair in zip(left, right, strict=True) for value in pair)


ALL_TRAINING_SEEDS = _interleave_equal(
    LEGACY_TRAINING_SEEDS,
    EXPANSION_TRAINING_SEEDS,
)


def _acquisition_cohort(seed: int) -> str:
    if seed in LEGACY_TRAINING_SEED_SET:
        return LEGACY_ACQUISITION_COHORT
    if seed in EXPANSION_TRAINING_SEED_SET:
        return EXPANSION_ACQUISITION_COHORT
    raise ValueError(f"seed {seed} is outside the fixed 30-episode whitelist")


def _split_acquisition_audit(seeds: Sequence[int]) -> dict[str, Any]:
    ordered = [
        {"seed": seed, "acquisition_cohort": _acquisition_cohort(seed)}
        for seed in seeds
    ]
    counts = Counter(item["acquisition_cohort"] for item in ordered)
    cohort_order = [item["acquisition_cohort"] for item in ordered]
    strictly_interleaved = all(left != right for left, right in pairwise(cohort_order))
    return {
        "ordered_episodes": ordered,
        "counts": {
            LEGACY_ACQUISITION_COHORT: counts[LEGACY_ACQUISITION_COHORT],
            EXPANSION_ACQUISITION_COHORT: counts[EXPANSION_ACQUISITION_COHORT],
        },
        "strictly_interleaved": strictly_interleaved,
    }


def _fixed_cv30_folds() -> tuple[Fold, ...]:
    legacy_groups = tuple(
        LEGACY_TRAINING_SEEDS[index : index + 5] for index in range(0, 15, 5)
    )
    expansion_groups = tuple(
        EXPANSION_TRAINING_SEEDS[index : index + 5] for index in range(0, 15, 5)
    )
    folds: list[Fold] = []
    for index in range(3):
        calibration_group = (index + 1) % 3
        fit_group = (index + 2) % 3
        folds.append(
            Fold(
                index=index,
                audit_seeds=_interleave_equal(
                    legacy_groups[index], expansion_groups[index]
                ),
                calibration_seeds=_interleave_equal(
                    legacy_groups[calibration_group][:2],
                    expansion_groups[calibration_group][:2],
                ),
                fit_seeds=(
                    *_interleave_equal(
                        legacy_groups[calibration_group][2:],
                        expansion_groups[calibration_group][2:],
                    ),
                    *_interleave_equal(
                        legacy_groups[fit_group], expansion_groups[fit_group]
                    ),
                ),
            )
        )

    result = tuple(folds)
    expected_counts = {
        "fit": (8, 8),
        "calibration": (2, 2),
        "audit": (5, 5),
    }
    for fold in result:
        split_values = {
            "fit": fold.fit_seeds,
            "calibration": fold.calibration_seeds,
            "audit": fold.audit_seeds,
        }
        split_sets = {name: set(values) for name, values in split_values.items()}
        if tuple(len(split_values[name]) for name in expected_counts) != (16, 4, 10):
            raise AssertionError("CV30 fold role sizes are invalid")
        if (
            split_sets["fit"] & split_sets["calibration"]
            or split_sets["fit"] & split_sets["audit"]
            or split_sets["calibration"] & split_sets["audit"]
        ):
            raise AssertionError("CV30 fold roles overlap")
        if set.union(*split_sets.values()) != ALL_TRAINING_SEED_SET:
            raise AssertionError("CV30 fold roles do not cover the fixed whitelist")
        for name, (legacy_count, expansion_count) in expected_counts.items():
            audit = _split_acquisition_audit(split_values[name])
            if audit["counts"] != {
                LEGACY_ACQUISITION_COHORT: legacy_count,
                EXPANSION_ACQUISITION_COHORT: expansion_count,
            }:
                raise AssertionError(f"CV30 {name} acquisition cohorts are imbalanced")
            if not audit["strictly_interleaved"]:
                raise AssertionError(
                    f"CV30 {name} acquisition cohorts are not interleaved"
                )
    if Counter(seed for fold in result for seed in fold.audit_seeds) != Counter(
        {seed: 1 for seed in ALL_TRAINING_SEEDS}
    ):
        raise AssertionError("CV30 outer audit must cover every episode exactly once")
    return result


def _validate_expansion_header(
    inventory: Mapping[str, Any],
    *,
    checkpoint_sha256: str,
) -> None:
    for field, expected in EXPANSION_INVENTORY_CONTRACT.items():
        actual = inventory.get(field)
        if isinstance(expected, bool):
            valid = actual is expected
        elif isinstance(expected, int):
            valid = (
                not isinstance(actual, bool)
                and isinstance(actual, int)
                and actual == expected
            )
        else:
            valid = actual == expected
        if not valid:
            raise ValueError(
                f"expansion inventory {field} must be {expected!r}, got {actual!r}"
            )
    if inventory.get("seeds") != list(EXPANSION_TRAINING_SEEDS):
        raise ValueError(
            "expansion inventory seeds must match the fixed ordered cohort"
        )
    if inventory.get("reserved_seeds_excluded") != sorted(PROHIBITED_SOURCE_SEEDS):
        raise ValueError("expansion inventory reserved seed declaration is invalid")
    declared_checkpoint = _required_string(
        inventory.get("checkpoint_sha256"),
        field="expansion inventory checkpoint_sha256",
    )
    if declared_checkpoint != checkpoint_sha256:
        raise ValueError(
            "expansion inventory checkpoint SHA-256 does not match the parent"
        )


def _select_expansion_inventory(
    inventory: Mapping[str, Any],
    *,
    checkpoint_sha256: str,
) -> list[Mapping[str, Any]]:
    """Validate all seed/role fields before any path-bearing field is accessed."""

    _validate_expansion_header(inventory, checkpoint_sha256=checkpoint_sha256)
    raw_inventory = inventory.get("source_inventory")
    if not isinstance(raw_inventory, list):
        raise TypeError("expansion inventory has no source inventory")

    selected: dict[int, Mapping[str, Any]] = {}
    observed_order: list[int] = []
    for index, raw in enumerate(raw_inventory):
        if not isinstance(raw, Mapping):
            raise TypeError(f"source_inventory[{index}] must be an object")
        seed = _required_int(raw.get("seed"), field=f"source_inventory[{index}].seed")
        role = raw.get("role")
        if not isinstance(role, str):
            raise TypeError(f"source_inventory[{index}].role must be a string")
        if seed in PROHIBITED_SOURCE_SEEDS and role == "training":
            raise ValueError(f"prohibited source seed {seed} is marked training")
        if seed in EXPANSION_TRAINING_SEED_SET:
            if role != "training":
                raise ValueError(f"expansion seed {seed} must have role='training'")
            if seed in selected:
                raise ValueError(f"duplicate expansion source seed: {seed}")
            selected[seed] = raw
            observed_order.append(seed)
        else:
            raise ValueError(f"unexpected expansion source seed: {seed}")

    missing = EXPANSION_TRAINING_SEED_SET - selected.keys()
    if missing:
        raise ValueError(
            f"expansion training whitelist is incomplete: {sorted(missing)}"
        )
    if tuple(observed_order) != EXPANSION_TRAINING_SEEDS:
        raise ValueError("expansion source inventory seed order is not fixed")

    # Only protocol fields are touched after the complete seed/role whitelist is known.
    for seed in EXPANSION_TRAINING_SEEDS:
        raw = selected[seed]
        for field in ("strict_prefix_zero", "strict_success"):
            if raw.get(field) is not True:
                raise ValueError(
                    f"expansion seed {seed} {field} must be true, "
                    f"got {raw.get(field)!r}"
                )
        shoot_rate = raw.get("shoot_command_rate")
        if not isinstance(shoot_rate, float) or shoot_rate != 1.0:
            raise ValueError(
                f"expansion seed {seed} shoot_command_rate must be float 1.0, "
                f"got {shoot_rate!r}"
            )
        if raw.get("external_region_memory") is not None:
            raise ValueError(
                f"expansion seed {seed} external_region_memory must be None, "
                f"got {raw.get('external_region_memory')!r}"
            )
    return [selected[seed] for seed in EXPANSION_TRAINING_SEEDS]


def _verify_expansion_sources(
    inventory: Sequence[Mapping[str, Any]],
    *,
    checkpoint: Path,
) -> tuple[list[tuple[Path, Path, Path]], list[dict[str, Any]]]:
    if tuple(raw.get("seed") for raw in inventory) != EXPANSION_TRAINING_SEEDS:
        raise AssertionError("path verification received an unordered expansion cohort")
    triplets: list[tuple[Path, Path, Path]] = []
    provenance: list[dict[str, Any]] = []
    for raw in inventory:
        seed = _required_int(raw.get("seed"), field="expansion source seed")
        paths: dict[str, Path] = {}
        for kind in ("dataset", "report", "manifest"):
            path = Path(_required_string(raw.get(kind), field=f"seed {seed} {kind}"))
            paths[kind] = path
        rebuilt = _validate_strict_prefix0_triplet(
            paths,
            seed=seed,
            checkpoint=checkpoint,
            intervene_on_disagreement=True,
        )
        for field in (
            "seed",
            "role",
            "strict_prefix_zero",
            "strict_success",
            "dataset",
            "dataset_sha256",
            "report",
            "report_sha256",
            "manifest",
            "manifest_sha256",
            "decisions",
            "teacher_interventions",
            "shoot_command_rate",
            "external_region_memory",
        ):
            if raw.get(field) != rebuilt.get(field):
                raise ValueError(
                    f"expansion seed {seed} inventory field {field} differs from "
                    "the strict record rebuilt from report/manifest evidence"
                )
        triplets.append((paths["dataset"], paths["report"], paths["manifest"]))
        provenance.append(
            {
                **rebuilt,
                "acquisition_cohort": EXPANSION_ACQUISITION_COHORT,
                "declared_hashes_verified": True,
                "strict_native_triplet_revalidated": True,
            }
        )
    return triplets, provenance


def _merge_verified_sources(
    legacy_triplets: Sequence[tuple[Path, Path, Path]],
    legacy_provenance: Sequence[Mapping[str, Any]],
    expansion_triplets: Sequence[tuple[Path, Path, Path]],
    expansion_provenance: Sequence[Mapping[str, Any]],
) -> tuple[list[tuple[Path, Path, Path]], list[dict[str, Any]]]:
    if len(legacy_triplets) != len(legacy_provenance) or len(expansion_triplets) != len(
        expansion_provenance
    ):
        raise ValueError("verified source triplets and provenance lengths differ")
    by_seed: dict[int, tuple[tuple[Path, Path, Path], dict[str, Any]]] = {}
    for expected, triplet, raw in (
        *zip(LEGACY_TRAINING_SEEDS, legacy_triplets, legacy_provenance, strict=True),
        *zip(
            EXPANSION_TRAINING_SEEDS,
            expansion_triplets,
            expansion_provenance,
            strict=True,
        ),
    ):
        record = dict(raw)
        if record.get("seed") != expected:
            raise ValueError("verified source provenance order differs from its cohort")
        cohort = _acquisition_cohort(expected)
        if expected in LEGACY_TRAINING_SEED_SET:
            record["acquisition_cohort"] = LEGACY_ACQUISITION_COHORT
        elif record.get("acquisition_cohort") != cohort:
            raise ValueError("expansion source acquisition cohort is invalid")
        if expected in by_seed:
            raise ValueError(f"duplicate verified source seed: {expected}")
        by_seed[expected] = (triplet, record)
    if set(by_seed) != ALL_TRAINING_SEED_SET:
        raise ValueError("verified sources do not cover the fixed 30-episode whitelist")
    return (
        [by_seed[seed][0] for seed in ALL_TRAINING_SEEDS],
        [by_seed[seed][1] for seed in ALL_TRAINING_SEEDS],
    )


def _plain_summary(folds: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    audit_rows = [fold["plain"]["raw_action_metrics"]["audit"] for fold in folds]
    calibration_successful = [
        bool(fold["plain"]["calibration"]["success"]) for fold in folds
    ]
    audit_runtime_eligible: list[bool] = []
    for fold, calibrated in zip(folds, calibration_successful, strict=True):
        runtime_metrics = fold["plain"].get("runtime_metrics")
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


def _plain_promotion_gate(
    summary: Mapping[str, Any],
    *,
    epochs: int,
) -> dict[str, Any]:
    audit = summary["outer_audit_micro"]
    reference = PROMOTION_GATE["reference_baseline"]
    applicable = epochs == PROMOTION_GATE["screening_epochs"]
    checks = {
        "plain_calibration_succeeds_on_at_least_two_of_three_folds": (
            int(summary["calibration_successful_folds"])
            >= PROMOTION_GATE["minimum_calibration_successful_folds"]
        ),
        "audit_runtime_is_eligible_on_at_least_two_of_three_folds": (
            int(summary["audit_runtime_eligible_folds"])
            >= PROMOTION_GATE["minimum_audit_runtime_eligible_folds"]
        ),
        "every_calibrated_fold_has_eligible_audit_runtime": (
            not summary["calibrated_audit_runtime_ineligible_folds"]
            and int(summary["audit_runtime_eligible_folds"])
            == int(summary["calibration_successful_folds"])
        ),
        "outer_audit_has_targets": int(audit["targets"]) > 0,
        "outer_audit_all_action_top1_outputs_finite": (
            int(audit["finite_top1"]) == int(audit["targets"])
        ),
        "outer_audit_equivalent_top1_rate_at_least_legacy_plain": (
            _meets_reference_rate(
                audit,
                "equivalent_top1",
                reference_numerator=reference["equivalent_top1"],
                reference_denominator=reference["targets"],
            )
        ),
        "outer_audit_direction_rate_at_least_legacy_plain": (
            _meets_reference_rate(
                audit,
                "direction_correct",
                reference_numerator=reference["direction_correct"],
                reference_denominator=reference["targets"],
            )
        ),
        "outer_audit_speed_rate_at_least_legacy_plain": _meets_reference_rate(
            audit,
            "speed_correct",
            reference_numerator=reference["speed_correct"],
            reference_denominator=reference["targets"],
        ),
    }
    criteria = {
        **PROMOTION_GATE,
        "reference_rates": {
            name: reference[name] / reference["targets"]
            for name in ("equivalent_top1", "direction_correct", "speed_correct")
        },
        "rate_comparison_semantics": (
            "exact integer cross multiplication against the fixed legacy e6 "
            "plain-baseline ratios; no result-dependent rounding"
        ),
    }
    return {
        "preregistered_before_cv30_audit": True,
        "applicable": applicable,
        "inapplicable_reason": (
            None if applicable else "promotion is defined only for exactly 6 epochs"
        ),
        "criteria": criteria,
        "checks": checks,
        "passed": applicable and all(checks.values()),
        "eligible_for_fixed_e20_followup": applicable and all(checks.values()),
        "deployment_eligible": False,
        "acceptance_claim": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fixed CV30 for plain certified preferred-action set NLL."
    )
    parser.add_argument("--failure", type=Path, default=DEFAULT_FAILURE)
    parser.add_argument(
        "--expansion-inventory",
        type=Path,
        default=DEFAULT_EXPANSION_INVENTORY,
    )
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--cpu-threads", type=int, default=1)
    args = parser.parse_args()
    if args.epochs <= 0 or args.cpu_threads <= 0:
        raise ValueError("epochs and cpu threads must be positive")

    script_path = Path(__file__)
    legacy_script_path = script_path.with_name("compare_preferred_objectives_cv.py")
    helper_path = script_path.with_name("train_temporal_residual_adapter.py")
    _validate_output_path(
        args.output,
        (
            args.failure,
            args.expansion_inventory,
            args.parent,
            script_path,
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

    # Both selectors finish seed/role validation before source paths are resolved.
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

    config = _adapter_config(failure)
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
        fold_seed = args.seed + fold.index * 100_003
        normalized = [_clone_episode(episode) for episode in raw_episodes]
        torch.manual_seed(fold_seed)
        adapter = ResidualCorrectionAdapter(config)
        fit = _episodes_by_seed(normalized, fold.fit_seeds)
        _normalize(adapter, fit, normalized)
        collision_weights = _collision_positive_weights(
            fit,
            maximum_weight=TRAINING_CONFIG["maximum_collision_positive_weight"],
        )
        physical_weights = _physical_danger_positive_weights(
            fit,
            maximum_weight=TRAINING_CONFIG["maximum_physical_danger_positive_weight"],
        )
        initial_sha256 = _state_digest(adapter.state_dict())
        member_seed = fold_seed + 1_009
        plain = _run_arm(
            "preferred_certified_equivalence_set_plain",
            adapter,
            normalized,
            fold,
            epochs=args.epochs,
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
                "plain": plain,
            }
        )

    summary = _plain_summary(fold_reports)
    report = {
        "schema_version": 1,
        "kind": "plain_certified_set_training_only_cv30",
        "training_only": True,
        "deployment_artifact_written": False,
        "formal_deployment_artifact_written": False,
        "deployment_eligible": False,
        "acceptance_claim": False,
        "objective_arms": [PLAIN_ARM_NAME],
        "variant_objectives_evaluated": [],
        "objective_scope": (
            "plain certified preferred-action set NLL only; rank, conditional "
            "tiebreak, uniform soft-target, and exact one-hot variants are excluded"
        ),
        "plain_objective": dict(PLAIN_OBJECTIVE_CONFIG),
        "audit_used_during_fit_or_calibration": False,
        "audit_used_for_threshold_epoch_fold_or_retry_selection": False,
        "audit_used_for_after_freeze_promotion_decision": True,
        "audit_prediction_policy": (
            "predicted exactly once after each fold's calibration attempt; never "
            "used for training, normalization, calibration thresholds, retries, "
            "epoch choice, or fold construction; used only by the preregistered "
            "after-freeze e6 promotion gate"
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
            "legacy_cv_script": str(legacy_script_path),
            "legacy_cv_script_sha256": file_sha256(legacy_script_path),
            "training_helper": str(helper_path),
            "training_helper_sha256": file_sha256(helper_path),
        },
        "experiment_config": {
            "epochs": args.epochs,
            "base_seed": args.seed,
            "device": "cpu",
            "cpu_threads": args.cpu_threads,
            "deterministic_algorithms": True,
            "adapter_config": asdict(config),
            "label_config": LABEL_CONFIG,
            "training_config": TRAINING_CONFIG,
            "ensemble_size_screening_override": 1,
            "gradient_clipping": {
                "max_norm": GRADIENT_CLIP_MAX_NORM,
                "separate_action_recurrent_semantics": (
                    SEPARATE_ACTION_GRADIENT_CLIP_SEMANTICS
                ),
                "action_group_modules": list(ACTION_BRANCH_MODULE_NAMES),
                "shared_safety_group": "all_other_trainable_member_parameters",
                "non_separate_architecture_semantics": (GLOBAL_GRADIENT_CLIP_SEMANTICS),
            },
        },
        "folds": fold_reports,
        "plain_summary": summary,
        "preregistered_e6_promotion_gate": _plain_promotion_gate(
            summary,
            epochs=args.epochs,
        ),
    }
    _write_json_atomic(args.output, report)
    print(
        json.dumps(
            {
                "artifact": str(args.output),
                "sha256": file_sha256(args.output),
                "training_only": True,
                "deployment_artifact_written": False,
                "eligible_for_fixed_e20_followup": report[
                    "preregistered_e6_promotion_gate"
                ]["eligible_for_fixed_e20_followup"],
            }
        )
    )


if __name__ == "__main__":
    main()
