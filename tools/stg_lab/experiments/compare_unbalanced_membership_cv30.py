"""Second adaptive CV30 screen for unweighted membership BCE.

The balanced certified-membership screen was already inspected before this
single change was specified.  This is consequently a second adaptive
development screen, not an independent validation.  It reuses the exact CV30
sources, folds, model, seeds, weights, diagnostics, runtime calibration, and
fixed gate from the balanced screen.  The only training change is replacing
per-row class-balanced membership BCE with unweighted per-action-cell BCE.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from stg_lab.provenance import file_sha256
from stg_lab.residual_adapter import ResidualCorrectionAdapter
from stg_lab.training import load_checkpoint

if __package__:
    from . import compare_certified_membership_cv30 as balanced
else:  # pragma: no cover - exercised by the real script invocation
    import compare_certified_membership_cv30 as balanced


UNBALANCED_ARM_NAME = "certified_membership_unweighted"
MEMBERSHIP_LOSS_MODE = "unweighted"
BALANCED_MEMBERSHIP_LOSS_MODE = "balanced"
SCREENING_EPOCHS = balanced.SCREENING_EPOCHS
BASE_SEED = balanced.BASE_SEED
DEFAULT_OUTPUT = Path(
    "artifacts/policy-humanlike-highres-okuu3-certified-membership-"
    "unweighted-cv30-e6.json"
)

# Keep the full balanced training mapping intact and record the sole objective
# switch alongside it.  _run_membership_arm consumes the shared numeric keys.
UNBALANCED_MEMBERSHIP_TRAINING_CONFIG = {
    **balanced.MEMBERSHIP_TRAINING_CONFIG,
    "membership_loss_mode": MEMBERSHIP_LOSS_MODE,
}
UNBALANCED_MEMBERSHIP_OBJECTIVE_CONFIG = {
    **balanced.MEMBERSHIP_OBJECTIVE_CONFIG,
    "schema": "independent_certified_action_membership_unweighted_bce",
    "row_balance": "none_equal_weight_per_action_cell",
    "membership_loss_mode": MEMBERSHIP_LOSS_MODE,
}

SECOND_ADAPTIVE_SCREEN_CONTEXT = {
    "sequence": 2,
    "adaptive_development_screen": True,
    "independent_statistical_validation": False,
    "specified_after_observing_plain_cv30_negative_result": True,
    "specified_after_observing_balanced_membership_negative_result": True,
    "sole_training_change": "membership_loss_mode: balanced -> unweighted",
    "epoch_seed_or_retry_selected_from_prior_result": False,
}


def _training_control_differences() -> dict[str, tuple[Any, Any]]:
    before = {
        **balanced.MEMBERSHIP_TRAINING_CONFIG,
        "membership_loss_mode": BALANCED_MEMBERSHIP_LOSS_MODE,
    }
    after = dict(UNBALANCED_MEMBERSHIP_TRAINING_CONFIG)
    return {
        key: (before.get(key), after.get(key))
        for key in before.keys() | after.keys()
        if before.get(key) != after.get(key)
    }


def _unbalanced_adaptive_development_gate(
    summary: Mapping[str, Any],
    *,
    epochs: int = SCREENING_EPOCHS,
) -> dict[str, Any]:
    """Apply the frozen balanced-screen gate with honest adaptive provenance."""

    result = balanced._adaptive_development_gate(summary, epochs=epochs)
    result.update(
        {
            "specified_after_observing_balanced_membership_negative_result": True,
            "second_adaptive_development_screen": True,
            "preregistered_before_membership_cv30_audit": False,
            "preregistered_before_unweighted_membership_cv30_audit": True,
        }
    )
    return result


def _run_unbalanced_membership_arm(
    adapter: ResidualCorrectionAdapter,
    episodes: list[balanced.EpisodeFeatures],
    fold: balanced.Fold,
    *,
    member_seed: int,
    collision_weights: torch.Tensor,
    physical_weights: torch.Tensor,
) -> dict[str, Any]:
    return balanced._run_membership_arm(
        adapter,
        episodes,
        fold,
        member_seed=member_seed,
        collision_weights=collision_weights,
        physical_weights=physical_weights,
        membership_loss_mode=MEMBERSHIP_LOSS_MODE,
        training_config=UNBALANCED_MEMBERSHIP_TRAINING_CONFIG,
        objective_config=UNBALANCED_MEMBERSHIP_OBJECTIVE_CONFIG,
        arm_name=UNBALANCED_ARM_NAME,
    )


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Second adaptive CV30 screen for unweighted independent "
            "certified-action membership BCE."
        )
    )
    parser.add_argument("--failure", type=Path, default=balanced.DEFAULT_FAILURE)
    parser.add_argument(
        "--expansion-inventory",
        type=Path,
        default=balanced.DEFAULT_EXPANSION_INVENTORY,
    )
    parser.add_argument("--parent", type=Path, default=balanced.DEFAULT_PARENT)
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
            f"unweighted CV30 output already exists; refusing to overwrite: {output}"
        )


def _reserve_new_output_path(output: Path) -> None:
    _validate_new_output_path(output, ())
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        output.touch(exist_ok=False)
    except FileExistsError as error:
        raise ValueError(
            f"unweighted CV30 output already exists; refusing to overwrite: {output}"
        ) from error


def main() -> None:
    args = _argument_parser().parse_args()
    if args.cpu_threads <= 0:
        raise ValueError("cpu threads must be positive")
    if _training_control_differences() != {
        "membership_loss_mode": (
            BALANCED_MEMBERSHIP_LOSS_MODE,
            MEMBERSHIP_LOSS_MODE,
        )
    }:
        raise AssertionError("unweighted screen changed more than membership BCE mode")

    script_path = Path(__file__)
    balanced_script_path = script_path.with_name(
        "compare_certified_membership_cv30.py"
    )
    plain_script_path = script_path.with_name("compare_plain_certified_set_cv30.py")
    legacy_script_path = script_path.with_name("compare_preferred_objectives_cv.py")
    helper_path = script_path.with_name("train_temporal_residual_adapter.py")
    _validate_new_output_path(
        args.output,
        (
            args.failure,
            args.expansion_inventory,
            args.parent,
            script_path,
            balanced_script_path,
            plain_script_path,
            legacy_script_path,
            helper_path,
        ),
    )

    torch.set_num_threads(args.cpu_threads)
    torch.use_deterministic_algorithms(True)
    failure = balanced._read_json(args.failure)
    expansion = balanced._read_json(args.expansion_inventory)
    parent_sha256 = file_sha256(args.parent)
    if parent_sha256 != failure.get("parent_checkpoint_sha256"):
        raise ValueError("parent checkpoint hash does not match failure diagnostics")

    # These are the same strict selectors and native triplet reconstruction used
    # by the first membership screen; role fields are checked before path access.
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

    config = balanced._membership_adapter_config(failure)
    parent, _metadata = load_checkpoint(args.parent, device="cpu")
    parent.cpu().eval()
    for parameter in parent.parameters():
        parameter.requires_grad_(False)
    if parent.config.recurrent_size != config.recurrent_size:
        raise ValueError("parent recurrent size does not match the v81 adapter")

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
            maximum_weight=UNBALANCED_MEMBERSHIP_TRAINING_CONFIG[
                "maximum_collision_positive_weight"
            ],
        )
        physical_weights = balanced._physical_danger_positive_weights(
            fit,
            maximum_weight=UNBALANCED_MEMBERSHIP_TRAINING_CONFIG[
                "maximum_physical_danger_positive_weight"
            ],
        )
        initial_sha256 = balanced._state_digest(adapter.state_dict())
        member_seed = fold_seed + 1_009
        membership = _run_unbalanced_membership_arm(
            adapter,
            normalized,
            fold,
            member_seed=member_seed,
            collision_weights=collision_weights,
            physical_weights=physical_weights,
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
                "initial_state_sha256": initial_sha256,
                "normalization_sha256": balanced._state_digest(
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

    summary = balanced._membership_summary(fold_reports)
    report = {
        "schema_version": 1,
        "kind": "unweighted_membership_second_adaptive_development_screen_cv30",
        "training_only": True,
        "deployment_artifact_written": False,
        "formal_deployment_artifact_written": False,
        "deployment_eligible": False,
        "acceptance_claim": False,
        "adaptive_development_screen": True,
        "adaptive_development_screen_sequence": 2,
        "independent_statistical_validation": False,
        "specified_after_observing_plain_cv30_negative_result": True,
        "specified_after_observing_balanced_membership_negative_result": True,
        "objective_arms": [UNBALANCED_ARM_NAME],
        "variant_objectives_evaluated": [],
        "objective_scope": (
            "one unweighted independent certified-membership arm only; network, "
            "folds, seeds, epochs, ensemble size, numeric weights, diagnostics, "
            "calibration, and gate are frozen from the balanced screen"
        ),
        "single_change_audit": {
            "training_control_differences": _training_control_differences(),
            **SECOND_ADAPTIVE_SCREEN_CONTEXT,
        },
        "membership_objective": dict(UNBALANCED_MEMBERSHIP_OBJECTIVE_CONFIG),
        "audit_used_during_fit_or_calibration": False,
        "audit_used_for_threshold_epoch_seed_fold_or_retry_selection": False,
        "audit_used_for_after_freeze_adaptive_screen": True,
        "audit_prediction_policy": (
            "predicted exactly once after each fold calibration attempt; never "
            "used for training, normalization, calibration thresholds, retries, "
            "epoch choice, seed choice, or fold construction"
        ),
        "membership_diagnostic_policy": dict(
            balanced.MEMBERSHIP_DIAGNOSTIC_POLICY
        ),
        "runtime_score_semantics": (
            "existing mean-onset AND selected-membership-confidence thresholds; "
            "onset-times-membership remains descriptive only"
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
            "balanced_membership_cv30_script": str(balanced_script_path),
            "balanced_membership_cv30_script_sha256": file_sha256(
                balanced_script_path
            ),
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
            "label_config": balanced.LABEL_CONFIG,
            "training_config": UNBALANCED_MEMBERSHIP_TRAINING_CONFIG,
            "ensemble_size_screening_override": 1,
            "gradient_clipping": {
                "max_norm": balanced.GRADIENT_CLIP_MAX_NORM,
                "separate_action_recurrent_semantics": (
                    balanced.SEPARATE_ACTION_GRADIENT_CLIP_SEMANTICS
                ),
                "action_group_modules": list(balanced.ACTION_BRANCH_MODULE_NAMES),
                "shared_safety_group": "all_other_trainable_member_parameters",
                "non_separate_architecture_semantics": (
                    balanced.GLOBAL_GRADIENT_CLIP_SEMANTICS
                ),
            },
        },
        "folds": fold_reports,
        "membership_summary": summary,
        "adaptive_development_gate": _unbalanced_adaptive_development_gate(
            summary
        ),
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
                "adaptive_development_screen_sequence": 2,
                "independent_statistical_validation": False,
                "eligible_for_fixed_followup": report[
                    "adaptive_development_gate"
                ]["eligible_for_fixed_followup"],
            }
        )
    )


if __name__ == "__main__":
    main()
