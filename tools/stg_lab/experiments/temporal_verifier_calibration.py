"""Fail-closed calibration for narrow temporal-verifier confidence scores."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import Any

import torch

from stg_lab.residual_adapter import (
    ResidualRuntimeConfig,
    residual_candidate_selection,
)

try:
    from . import train_temporal_residual_adapter as trainer
except ImportError:
    import train_temporal_residual_adapter as trainer


MAXIMUM_TAIL_THRESHOLDS = 32
MATERIAL_GATE_EPSILON = 1e-4


def _thresholds_above_tail(
    score_tensors: Sequence[torch.Tensor],
    *,
    maximum_tail_thresholds: int,
) -> tuple[float, ...]:
    supported_dtypes = (torch.float32, torch.float64)
    typed_scores: set[tuple[float, torch.dtype]] = set()
    for scores in score_tensors:
        if not isinstance(scores, torch.Tensor):
            raise TypeError("temporal confidence scores must be tensors")
        if scores.dtype not in supported_dtypes:
            raise TypeError(
                "temporal confidence scores must use float32 or float64"
            )
        for raw_score in scores.detach().reshape(-1):
            score = float(raw_score)
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ValueError("temporal confidence is outside [0, 1]")
            typed_scores.add((score, scores.dtype))

    # For equal numeric scores, the coarser dtype has the larger cut point and
    # therefore conservatively covers both representations at a capped tail.
    dtype_priority = {torch.float32: 1, torch.float64: 0}
    highest = sorted(
        typed_scores,
        key=lambda item: (item[0], dtype_priority[item[1]]),
        reverse=True,
    )[:maximum_tail_thresholds]
    thresholds = {0.2, 0.4, 0.6, 0.8}
    for score, dtype in highest:
        source = torch.tensor(score, dtype=dtype)
        cutpoint = torch.nextafter(source, torch.full_like(source, math.inf))
        threshold = float(cutpoint)
        if threshold > 1.0:
            # No legal probability threshold can sit above an exact score of
            # one. Keep the saturated boundary so the row remains active in
            # the gate inventory and final fail-closed metrics.
            thresholds.add(1.0)
            continue
        if bool(source >= threshold):
            raise AssertionError(
                "temporal confidence cut point does not exclude its source score"
            )
        thresholds.add(threshold)
    return tuple(sorted(thresholds))


def _beneficial_candidate_mask(
    episode: Any,
    candidates: torch.Tensor,
) -> torch.Tensor:
    positive = episode.gate_valid & (episode.gate_targets > 0.0)
    _preferred, correction_required, equivalent = trainer._preferred_equivalence(
        episode, positive
    )
    candidate_equivalent = equivalent.gather(
        -1, candidates.unsqueeze(-1)
    ).squeeze(-1)
    candidate_safe = episode.evaluation_safe_actions.gather(
        -1, candidates.unsqueeze(-1)
    ).squeeze(-1)
    return correction_required & candidate_equivalent & candidate_safe


def temporal_confidence_thresholds(
    predictions: Mapping[int, Mapping[str, torch.Tensor]],
    episodes: Sequence[Any],
    *,
    maximum_tail_thresholds: int = MAXIMUM_TAIL_THRESHOLDS,
) -> tuple[float, ...]:
    """Return fixed-grid floors plus cut points above high-scoring mistakes."""

    if (
        isinstance(maximum_tail_thresholds, bool)
        or not isinstance(maximum_tail_thresholds, int)
        or maximum_tail_thresholds <= 0
    ):
        raise ValueError("maximum tail thresholds must be a positive integer")
    disallowed_scores: list[torch.Tensor] = []
    for episode in episodes:
        values = predictions[episode.seed]
        confidence = values["action_confidence"]
        candidates = values["candidates"]
        if confidence.shape != candidates.shape:
            raise ValueError("temporal confidence and candidates do not align")
        finite = torch.isfinite(confidence) & values["action_all_members_finite"]
        changed = candidates != episode.parent_actions
        beneficial = _beneficial_candidate_mask(episode, candidates)
        disallowed_scores.append(
            confidence[finite & changed & ~beneficial]
        )
    return _thresholds_above_tail(
        disallowed_scores,
        maximum_tail_thresholds=maximum_tail_thresholds,
    )


def calibrate_temporal_verifier(
    predictions: dict[int, dict[str, torch.Tensor]],
    training_episodes: list[Any],
    calibration_episodes: list[Any],
    *,
    ensemble_size: int,
) -> ResidualRuntimeConfig:
    """Calibrate conservative future-onset requests without outer-audit access."""

    if ensemble_size <= 0:
        raise ValueError("ensemble size must be positive")
    pool = [*training_episodes, *calibration_episodes]
    if not training_episodes or not calibration_episodes:
        raise ValueError("temporal calibration requires both split roles")
    if len({episode.seed for episode in pool}) != len(pool):
        raise ValueError("temporal calibration split roles overlap")
    agreement_values = sorted({
        math.ceil(ensemble_size * 2 / 3) / ensemble_size,
        1.0,
    })
    candidates: list[tuple[tuple[Any, ...], ResidualRuntimeConfig]] = []
    for minimum_gate in (0.25, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        for action_agreement in agreement_values:
            for candidate_danger in (
                0.1,
                0.15,
                0.2,
                0.25,
                0.3,
                0.35,
                0.4,
                0.5,
            ):
                for candidate_agreement in agreement_values:
                    # Candidate confidence is deliberately disabled while the
                    # executable tail for this safety setting is inventoried.
                    inventory_runtime = ResidualRuntimeConfig(
                        gate_probability_threshold=0.5,
                        minimum_member_gate_probability=minimum_gate,
                        action_probability_threshold=0.0,
                        ensemble_agreement_threshold=action_agreement,
                        override_logit_margin=1.0,
                        legacy_gate_enabled=False,
                        critic_enabled=True,
                        current_critic_request_enabled=False,
                        prefer_safe_previous_action=False,
                        critic_signal="physical_danger",
                        candidate_physical_danger_probability_threshold=(
                            candidate_danger
                        ),
                        candidate_safety_agreement_threshold=candidate_agreement,
                        future_onset_gate_enabled=True,
                    )
                    setting_rows: list[
                        tuple[
                            Mapping[str, torch.Tensor],
                            torch.Tensor,
                            torch.Tensor,
                            torch.Tensor,
                        ]
                    ] = []
                    setting_disallowed_confidence: list[torch.Tensor] = []
                    for episode in pool:
                        values = predictions[episode.seed]
                        physical = values.get("physical_danger_probabilities")
                        if physical is None:
                            raise ValueError(
                                "temporal calibration lacks physical predictions"
                            )
                        selection = residual_candidate_selection(
                            correction_actions=values["candidates"],
                            correction_confidence=values["action_confidence"],
                            agreement=values["agreement"],
                            previous_actions=episode.previous_actions,
                            runtime_config=inventory_runtime,
                            physical_danger_probabilities=physical,
                            parent_actions=episode.parent_actions,
                            future_onset=torch.ones_like(
                                values["mean_gate"], dtype=torch.bool
                            ),
                        )
                        selected = selection["correction_actions"]
                        indices = (
                            selected.unsqueeze(0)
                            .unsqueeze(-1)
                            .expand(physical.shape[0], *selected.shape, 1)
                        )
                        physical_scores = physical.gather(-1, indices).squeeze(-1)
                        predicted_safe = (
                            torch.isfinite(physical_scores).all(dim=0)
                            & (
                                (
                                    torch.isfinite(physical_scores)
                                    & (physical_scores <= candidate_danger)
                                )
                                .to(values["mean_gate"].dtype)
                                .mean(dim=0)
                                >= candidate_agreement
                            )
                        )
                        positive = episode.gate_valid & (episode.gate_targets > 0.0)
                        _preferred, correction_required, equivalent = (
                            trainer._preferred_equivalence(episode, positive)
                        )
                        selected_equivalent = equivalent.gather(
                            -1, selected.unsqueeze(-1)
                        ).squeeze(-1)
                        actual_safe = episode.evaluation_safe_actions.gather(
                            -1, selected.unsqueeze(-1)
                        ).squeeze(-1)
                        eligible_without_confidence = (
                            torch.isfinite(values["mean_gate"])
                            & torch.isfinite(values["minimum_gate"])
                            & torch.isfinite(selection["correction_confidence"])
                            & values["action_all_members_finite"]
                            & (values["minimum_gate"] >= minimum_gate)
                            & (selection["agreement"] >= action_agreement)
                            & (selected != episode.parent_actions)
                            & predicted_safe
                        )
                        disallowed = (
                            ~correction_required
                            | ~actual_safe
                            | ~selected_equivalent
                        )
                        setting_disallowed_confidence.append(
                            selection["correction_confidence"][
                                eligible_without_confidence & disallowed
                            ]
                        )
                        setting_rows.append((
                            values,
                            selection["correction_confidence"],
                            eligible_without_confidence,
                            disallowed,
                        ))
                    action_thresholds = _thresholds_above_tail(
                        setting_disallowed_confidence,
                        maximum_tail_thresholds=MAXIMUM_TAIL_THRESHOLDS,
                    )
                    for action_confidence in action_thresholds:
                        selection_runtime = ResidualRuntimeConfig(
                            gate_probability_threshold=0.5,
                            minimum_member_gate_probability=minimum_gate,
                            action_probability_threshold=action_confidence,
                            ensemble_agreement_threshold=action_agreement,
                            override_logit_margin=1.0,
                            legacy_gate_enabled=False,
                            critic_enabled=True,
                            current_critic_request_enabled=False,
                            prefer_safe_previous_action=False,
                            critic_signal="physical_danger",
                            candidate_physical_danger_probability_threshold=(
                                candidate_danger
                            ),
                            candidate_safety_agreement_threshold=(
                                candidate_agreement
                            ),
                            future_onset_gate_enabled=True,
                        )
                        disallowed_scores = [
                            values["mean_gate"][
                                eligible_without_confidence
                                & (confidence >= action_confidence)
                                & disallowed
                            ]
                            for (
                                values,
                                confidence,
                                eligible_without_confidence,
                                disallowed,
                            ) in setting_rows
                        ]
                        available = [
                            value for value in disallowed_scores if value.numel()
                        ]
                        maximum_disallowed = (
                            max(float(value.max()) for value in available)
                            if available else
                            0.4999
                        )
                        runtime_values = asdict(selection_runtime)
                        runtime_values["gate_probability_threshold"] = min(
                            1.0,
                            max(
                                0.5,
                                maximum_disallowed + MATERIAL_GATE_EPSILON,
                            ),
                        )
                        runtime = ResidualRuntimeConfig(**runtime_values)
                        train = trainer._metrics(
                            predictions, training_episodes, runtime
                        )["total"]
                        calibration = trainer._metrics(
                            predictions, calibration_episodes, runtime
                        )["total"]
                        fail_closed = all(
                            metrics[name] == 0
                            for metrics in (train, calibration)
                            for name in (
                                "unbeneficial_overrides",
                                "false_overrides",
                                "unsafe_overrides",
                                "non_equivalent_overrides",
                            )
                        )
                        early_covered = (
                            train["early_beneficial_overrides"] > 0
                            and calibration["early_beneficial_overrides"] > 0
                        )
                        if not (fail_closed and early_covered):
                            continue
                        score = (
                            min(
                                train["early_anticipatory_opportunity_recall"],
                                calibration[
                                    "early_anticipatory_opportunity_recall"
                                ],
                            ),
                            min(
                                train["early_danger_event_cluster_recall"],
                                calibration["early_danger_event_cluster_recall"],
                            ),
                            train["early_beneficial_overrides"]
                            + calibration["early_beneficial_overrides"],
                            train["beneficial_overrides"]
                            + calibration["beneficial_overrides"],
                            -train["candidate_safety_vetoes"]
                            - calibration["candidate_safety_vetoes"],
                        )
                        candidates.append((score, runtime))
    if not candidates:
        raise ValueError(
            "no fail-closed temporal-verifier calibration covers early events "
            "in both training and calibration episodes"
        )
    return max(candidates, key=lambda item: item[0])[1]
