"""Fit/calibration-only probe for the fifth temporal-verifier candidate.

This development probe never predicts an outer-audit split.  Its output is
descriptive only and cannot be used as deployment or acceptance evidence.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch

try:
    from . import compare_selected_candidate_confidence_cv30 as previous
    from . import train_temporal_residual_adapter as trainer
    from .formal_campaign import file_sha256, write_json_exclusive
    from .temporal_action_set_verifier import (
        INFERENCE_INPUT_SEMANTICS,
        SUPERVISION_SEMANTICS,
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
    from .temporal_verifier_calibration import (
        calibrate_temporal_verifier,
        temporal_confidence_thresholds,
    )
except ImportError:
    import compare_selected_candidate_confidence_cv30 as previous
    import train_temporal_residual_adapter as trainer
    from formal_campaign import file_sha256, write_json_exclusive
    from temporal_action_set_verifier import (
        INFERENCE_INPUT_SEMANTICS,
        SUPERVISION_SEMANTICS,
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
    from temporal_verifier_calibration import (
        calibrate_temporal_verifier,
        temporal_confidence_thresholds,
    )


balanced = previous.balanced
dual = previous.dual
BASE_SEED = 20260901
VERIFIER_SEED_OFFSET = 2_017
DEFAULT_OUTPUT = Path(
    "artifacts/policy-humanlike-highres-okuu3-temporal-action-set-verifier-"
    "corrected-runtime-support-fit-cal-probe-v6.json"
)


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit/calibration-only probe for temporal candidate verification."
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
        default=previous.DEFAULT_PLAIN_REFERENCE,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cpu-threads", type=int, default=1)
    return parser


def _prediction_with_action_latents(
    adapter: Any,
    episode: Any,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Capture frozen action-GRU outputs during the predictor's single forward."""

    shared_captures: list[list[torch.Tensor]] = [
        [] for _ in range(adapter.config.ensemble_size)
    ]
    action_captures: list[list[torch.Tensor]] = [
        [] for _ in range(adapter.config.ensemble_size)
    ]
    handles = []
    for member_index, member in enumerate(adapter.members):
        if member.action_recurrent is None:
            raise ValueError("temporal verification requires a separate action recurrent")

        def capture_shared(
            _module: Any,
            _inputs: Any,
            output: Any,
            *,
            index: int = member_index,
        ) -> None:
            recurrent, _hidden = output
            shared_captures[index].append(recurrent.detach().cpu())

        def capture_action(
            _module: Any,
            _inputs: Any,
            output: Any,
            *,
            index: int = member_index,
        ) -> None:
            recurrent, _hidden = output
            action_captures[index].append(recurrent.detach().cpu())

        handles.append(member.recurrent.register_forward_hook(capture_shared))
        handles.append(member.action_recurrent.register_forward_hook(capture_action))
    try:
        values = trainer._predict_episode(adapter, episode, device="cpu")
    finally:
        for handle in handles:
            handle.remove()
    member_latents = []
    for shared_capture, action_capture in zip(
        shared_captures, action_captures, strict=True
    ):
        if not shared_capture or not action_capture:
            raise RuntimeError("frozen action recurrent hook captured no output")
        shared = torch.cat(shared_capture, dim=1)
        action = torch.cat(action_capture, dim=1)
        if shared.shape[:2] != (1, episode.decisions) or action.shape != shared.shape:
            raise RuntimeError("captured action recurrent latent is misaligned")
        member_latents.append(torch.cat((shared[0], action[0]), dim=-1))
    latents = torch.stack(member_latents, dim=0)
    if not bool(torch.isfinite(latents).all()):
        raise ValueError("captured action recurrent latent is nonfinite")
    return values, latents


def _verifier_episode(
    episode: Any,
    values: Mapping[str, torch.Tensor],
    action_latents: torch.Tensor,
    config: TemporalActionSetVerifierConfig,
) -> TemporalActionSetEpisode:
    candidates = values["candidates"]
    (
        dense_mask,
        labels,
        selected_mask,
        selected_labels,
    ) = temporal_action_set_targets(episode, candidates)
    inputs = build_temporal_action_set_inputs(
        policy_latents=action_latents,
        mean_action_probabilities=values["mean_action_probabilities"],
        mean_gate=values["mean_gate"],
        minimum_gate=values["minimum_gate"],
        physical_danger_probabilities=values[
            "physical_danger_probabilities"
        ],
        parent_actions=episode.parent_actions,
        config=config,
    )
    return TemporalActionSetEpisode(
        episode.seed,
        inputs,
        dense_mask,
        labels,
        candidates,
        selected_mask,
        selected_labels,
    )


def _inject_verifier_confidence(
    values: Mapping[str, torch.Tensor],
    prediction: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    result = dict(values)
    confidence = prediction["confidence"]
    finite = prediction["all_selected_members_finite"]
    if confidence.shape != values["action_confidence"].shape:
        raise ValueError("temporal confidence does not align with selector")
    if finite.shape != values["action_all_members_finite"].shape:
        raise ValueError("temporal finite mask does not align with selector")
    result["selector_action_confidence"] = values["action_confidence"]
    result["action_confidence"] = confidence
    result["action_all_members_finite"] = (
        values["action_all_members_finite"] & finite
    )
    result["temporal_verifier_member_probabilities"] = prediction[
        "selected_member_probabilities"
    ]
    result["temporal_verifier_all_members_finite"] = finite
    return result


def _confidence_diagnostics(
    episodes: Sequence[Any],
    verifier_episodes: Mapping[int, TemporalActionSetEpisode],
    predictions: Mapping[int, Mapping[str, torch.Tensor]],
) -> dict[str, Any]:
    scores: list[float] = []
    labels: list[bool] = []
    finite_rows = 0
    total_rows = 0
    dense_early_rows = 0
    selected_non_early_rows = 0
    early_no_correction_changed_rows = 0
    early_no_correction_parent_rows = 0
    selected_gate_invalid_rows = 0
    for episode in episodes:
        labelled = verifier_episodes[episode.seed]
        values = predictions[episode.seed]
        mask = labelled.selected_label_mask
        dense_mask = labelled.label_mask
        changed = labelled.selected_candidates != episode.parent_actions
        no_correction = ~episode.preferred_correction_required
        finite = values["temporal_verifier_all_members_finite"]
        total_rows += int(mask.sum())
        dense_early_rows += int(dense_mask.sum())
        selected_non_early_rows += int((mask & ~dense_mask).sum())
        early_no_correction_changed_rows += int(
            (dense_mask & no_correction & changed).sum()
        )
        early_no_correction_parent_rows += int(
            (dense_mask & no_correction & ~changed).sum()
        )
        selected_gate_invalid_rows += int((mask & ~episode.gate_valid).sum())
        valid = mask & finite
        finite_rows += int(valid.sum())
        scores.extend(float(value) for value in values["action_confidence"][valid])
        labels.extend(bool(value) for value in labelled.selected_labels[valid])
    positive = [score for score, label in zip(scores, labels, strict=True) if label]
    negative = [score for score, label in zip(scores, labels, strict=True) if not label]
    positive_mean = sum(positive) / len(positive) if positive else None
    negative_mean = sum(negative) / len(negative) if negative else None
    return {
        "rows": total_rows,
        "dense_early_rows": dense_early_rows,
        "selected_runtime_support_rows": total_rows,
        "selected_non_early_rows": selected_non_early_rows,
        "early_no_correction_changed_rows": (
            early_no_correction_changed_rows
        ),
        "early_no_correction_parent_rows_excluded": (
            early_no_correction_parent_rows
        ),
        "selected_gate_invalid_negative_rows": selected_gate_invalid_rows,
        "finite_rows": finite_rows,
        "positive_rows": sum(labels),
        "negative_rows": finite_rows - sum(labels),
        "positive_rate": sum(labels) / finite_rows if finite_rows else None,
        "confidence": balanced._score_summary(scores),
        "positive_confidence": balanced._score_summary(positive),
        "negative_confidence": balanced._score_summary(negative),
        "positive_minus_negative_mean": (
            positive_mean - negative_mean
            if positive_mean is not None and negative_mean is not None
            else None
        ),
        "brier_score": (
            sum(
                (score - float(label)) ** 2
                for score, label in zip(scores, labels, strict=True)
            ) / len(scores)
            if scores else None
        ),
        "reliability": balanced._reliability(scores, labels),
    }


def _load_inputs(args: argparse.Namespace) -> tuple[Any, list[Any], str, dict[str, str]]:
    failure = balanced._read_json(args.failure)
    expansion = balanced._read_json(args.expansion_inventory)
    parent_sha256 = file_sha256(args.parent)
    if parent_sha256 != failure.get("parent_checkpoint_sha256"):
        raise ValueError("parent checkpoint hash does not match diagnostics")
    plain_reference = dual._read_frozen_reference(
        args.plain_reference,
        expected_sha256=previous.PLAIN_REFERENCE_SHA256,
        label="plain",
    )
    reference_folds = dual._reference_fold_map(plain_reference)
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
    triplets, _source_provenance = balanced._merge_verified_sources(
        legacy_triplets,
        legacy_provenance,
        expansion_triplets,
        expansion_provenance,
    )
    dual_config = dual._dual_head_adapter_config(failure)
    plain_config = replace(dual_config, per_action_membership_confidence=False)
    parent, _metadata = balanced.load_checkpoint(args.parent, device="cpu")
    parent.cpu().eval()
    for parameter in parent.parameters():
        parameter.requires_grad_(False)
    feature_adapter = balanced.ResidualCorrectionAdapter(dual_config)
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
        raise ValueError("loaded episodes do not match the fixed CV30 whitelist")
    return plain_config, raw_episodes, parent_sha256, reference_folds


def main() -> None:
    args = _argument_parser().parse_args()
    if args.cpu_threads != 1:
        raise ValueError("the temporal verifier probe requires exactly one CPU thread")
    if args.output.exists():
        raise FileExistsError(f"probe output already exists: {args.output}")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    plain_config, raw_episodes, parent_sha256, reference_folds = _load_inputs(args)
    training_config = TemporalActionSetTrainingConfig()
    verifier_config = TemporalActionSetVerifierConfig(
        latent_size=plain_config.hidden_size * 2,
    )
    fold_reports = []
    for fold in balanced._fixed_cv30_folds():
        fold_seed = BASE_SEED + fold.index * 100_003
        normalized = [balanced._clone_episode(item) for item in raw_episodes]
        fit = balanced._episodes_by_seed(normalized, fold.fit_seeds)
        calibration = balanced._episodes_by_seed(
            normalized, fold.calibration_seeds
        )
        torch.manual_seed(fold_seed)
        adapter = balanced.ResidualCorrectionAdapter(plain_config)
        balanced._normalize(adapter, fit, normalized)
        collision_weights = balanced._collision_positive_weights(
            fit,
            maximum_weight=previous.SELECTOR_TRAINING_CONFIG[
                "maximum_collision_positive_weight"
            ],
        )
        physical_weights = balanced._physical_danger_positive_weights(
            fit,
            maximum_weight=previous.SELECTOR_TRAINING_CONFIG[
                "maximum_physical_danger_positive_weight"
            ],
        )
        member_seed = fold_seed + 1_009
        selector_history = previous._train_plain_selector(
            adapter,
            fit,
            member_seed=member_seed,
            collision_weights=collision_weights,
            physical_weights=physical_weights,
        )
        reference_history = reference_folds[fold.index]["plain"]["history"]
        if selector_history != reference_history:
            raise AssertionError("probe selector differs from frozen plain reference")

        base_predictions: dict[int, dict[str, torch.Tensor]] = {}
        verifier_episodes: dict[int, TemporalActionSetEpisode] = {}
        for episode in (*fit, *calibration):
            values, latents = _prediction_with_action_latents(adapter, episode)
            base_predictions[episode.seed] = values
            verifier_episodes[episode.seed] = _verifier_episode(
                episode, values, latents, verifier_config
            )

        verifier_seed = fold_seed + VERIFIER_SEED_OFFSET
        torch.manual_seed(verifier_seed)
        verifier = TemporalActionSetVerifier(verifier_config)
        initial_verifier_sha256 = action_set_verifier_state_sha256(verifier)
        history = train_temporal_action_set_verifier(
            verifier,
            [verifier_episodes[episode.seed] for episode in fit],
            seed=verifier_seed,
            config=training_config,
        )
        predictions: dict[int, dict[str, torch.Tensor]] = {}
        for episode in (*fit, *calibration):
            verifier_prediction = predict_temporal_action_set_verifier(
                verifier,
                verifier_episodes[episode.seed].inputs,
                base_predictions[episode.seed]["candidates"],
            )
            predictions[episode.seed] = _inject_verifier_confidence(
                base_predictions[episode.seed], verifier_prediction
            )
        runtime = None
        calibration_error = None
        try:
            runtime = calibrate_temporal_verifier(
                predictions,
                fit,
                calibration,
                ensemble_size=adapter.config.ensemble_size,
            )
        except ValueError as error:
            expected = "no fail-closed temporal-verifier calibration covers early events"
            if expected not in str(error):
                raise
            calibration_error = str(error)
        fold_reports.append({
            "fold": fold.index,
            "fit_seeds": list(fold.fit_seeds),
            "calibration_seeds": list(fold.calibration_seeds),
            "audit_seeds_not_predicted": list(fold.audit_seeds),
            "selector_history_equal_to_plain": True,
            "initial_verifier_state_sha256": initial_verifier_sha256,
            "trained_verifier_state_sha256": (
                action_set_verifier_state_sha256(verifier)
            ),
            "verifier_history": history,
            "confidence_diagnostics": {
                "fit": _confidence_diagnostics(
                    fit, verifier_episodes, predictions
                ),
                "calibration": _confidence_diagnostics(
                    calibration, verifier_episodes, predictions
                ),
            },
            "calibration": {
                "success": runtime is not None,
                "error": calibration_error,
                "runtime_config": None if runtime is None else asdict(runtime),
                "candidate_confidence_thresholds": list(
                    temporal_confidence_thresholds(
                        predictions, [*fit, *calibration]
                    )
                ),
                "fit_runtime_metrics": (
                    None
                    if runtime is None else
                    balanced._runtime_metrics(predictions, fit, runtime)
                ),
                "calibration_runtime_metrics": (
                    None
                    if runtime is None else
                    balanced._runtime_metrics(predictions, calibration, runtime)
                ),
            },
        })
    payload = {
        "schema_version": 1,
        "kind": (
            "temporal_action_set_verifier_corrected_runtime_support_"
            "fit_calibration_only_probe_v6"
        ),
        "training_only": True,
        "acceptance_claim": False,
        "deployment_eligible": False,
        "outer_audit_predictions": 0,
        "adaptive_development_screen_sequence_if_preregistered": 5,
        "parent_sha256": parent_sha256,
        "selector_epochs": previous.SCREENING_EPOCHS,
        "verifier_config": asdict(verifier_config),
        "verifier_training_config": asdict(training_config),
        "inference_input_semantics": dict(INFERENCE_INPUT_SEMANTICS),
        "supervision_semantics": dict(SUPERVISION_SEMANTICS),
        "folds": fold_reports,
        "calibration_successful_folds": sum(
            bool(fold["calibration"]["success"]) for fold in fold_reports
        ),
    }
    digest = write_json_exclusive(args.output.resolve(), payload)
    print(json.dumps({
        "output": str(args.output),
        "sha256": digest,
        "calibration_successful_folds": payload[
            "calibration_successful_folds"
        ],
        "outer_audit_predictions": 0,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
