from __future__ import annotations

import inspect
import math
from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from experiments import temporal_verifier_calibration as calibration_module
from experiments.temporal_verifier_calibration import (
    MAXIMUM_TAIL_THRESHOLDS,
    _thresholds_above_tail,
    calibrate_temporal_verifier,
    temporal_confidence_thresholds,
)


def _episode(
    seed: int,
    confidences: list[float],
    *,
    physical_scores: list[float] | None = None,
    beneficial_rows: tuple[int, ...] = (0,),
    dtype: torch.dtype = torch.float64,
) -> tuple[SimpleNamespace, dict[str, torch.Tensor]]:
    decisions = len(confidences)
    candidate_action = 2
    parent_actions = torch.ones(decisions, dtype=torch.int64)
    candidates = torch.full(
        (decisions,), candidate_action, dtype=torch.int64
    )
    correction_required = torch.zeros(decisions, dtype=torch.bool)
    equivalent = torch.zeros((decisions, 18), dtype=torch.bool)
    gate_targets = torch.zeros(decisions, dtype=dtype)
    for row in beneficial_rows:
        correction_required[row] = True
        equivalent[row, candidate_action] = True
        gate_targets[row] = 1.0
    evaluation_safe = torch.ones((decisions, 18), dtype=torch.bool)
    episode = SimpleNamespace(
        seed=seed,
        decisions=decisions,
        parent_actions=parent_actions,
        previous_actions=torch.full((decisions,), -1, dtype=torch.int64),
        gate_valid=torch.ones(decisions, dtype=torch.bool),
        gate_targets=gate_targets,
        preferred_actions=candidates.clone(),
        preferred_correction_required=correction_required,
        preferred_equivalent_actions=equivalent,
        evaluation_safe_actions=evaluation_safe,
    )
    physical = torch.ones((1, decisions, 18), dtype=dtype)
    selected_physical = (
        physical_scores
        if physical_scores is not None else
        [0.1] * decisions
    )
    physical[0, torch.arange(decisions), candidates] = torch.tensor(
        selected_physical, dtype=dtype
    )
    prediction = {
        "action_confidence": torch.tensor(confidences, dtype=dtype),
        "candidates": candidates,
        "agreement": torch.ones(decisions, dtype=dtype),
        "mean_gate": torch.full((decisions,), 0.9, dtype=dtype),
        "minimum_gate": torch.full((decisions,), 0.95, dtype=dtype),
        "action_all_members_finite": torch.ones(decisions, dtype=torch.bool),
        "physical_danger_probabilities": physical,
    }
    return episode, prediction


def _metric_totals(*, early_covered: bool) -> dict[str, Any]:
    covered = int(early_covered)
    return {
        "unbeneficial_overrides": 0,
        "false_overrides": 0,
        "unsafe_overrides": 0,
        "non_equivalent_overrides": 0,
        "early_beneficial_overrides": covered,
        "early_anticipatory_opportunity_recall": float(covered),
        "early_danger_event_cluster_recall": float(covered),
        "beneficial_overrides": covered,
        "candidate_safety_vetoes": 0,
    }


def test_setting_irrelevant_high_errors_do_not_evict_relevant_tail_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relevant_error = 0.7
    source = torch.tensor(relevant_error, dtype=torch.float64)
    expected_threshold = float(
        torch.nextafter(source, torch.full_like(source, math.inf))
    )
    unrelated_high_errors = [0.99 - index * 0.005 for index in range(33)]
    assert len(unrelated_high_errors) > MAXIMUM_TAIL_THRESHOLDS
    assert min(unrelated_high_errors) > relevant_error
    confidences = [0.8, relevant_error, *unrelated_high_errors]
    physical_scores = [0.1, 0.1, *([0.45] * len(unrelated_high_errors))]
    training, training_prediction = _episode(
        101,
        confidences,
        physical_scores=physical_scores,
    )
    calibration, calibration_prediction = _episode(
        102,
        confidences,
        physical_scores=physical_scores,
    )

    def metrics(
        _predictions: object,
        _episodes: object,
        runtime: object,
    ) -> dict[str, Any]:
        covered = (
            runtime.minimum_member_gate_probability == 0.25
            and runtime.candidate_physical_danger_probability_threshold == 0.4
            and runtime.action_probability_threshold == expected_threshold
        )
        return {"total": _metric_totals(early_covered=covered)}

    monkeypatch.setattr(calibration_module.trainer, "_metrics", metrics)
    runtime = calibrate_temporal_verifier(
        {101: training_prediction, 102: calibration_prediction},
        [training],
        [calibration],
        ensemble_size=1,
    )

    assert runtime.candidate_physical_danger_probability_threshold == 0.4
    assert runtime.action_probability_threshold == expected_threshold


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_tail_threshold_uses_source_dtype_with_inclusive_runtime_comparison(
    dtype: torch.dtype,
) -> None:
    confidence = torch.tensor([0.75], dtype=dtype)
    threshold = float(
        torch.nextafter(confidence[0], torch.full_like(confidence[0], math.inf))
    )

    assert threshold in _thresholds_above_tail(
        [confidence], maximum_tail_thresholds=1
    )
    assert not bool(confidence[0] >= threshold)
    assert bool(torch.tensor(threshold, dtype=dtype) >= threshold)


def test_tail_thresholds_preserve_each_source_dtype_in_mixed_inventory() -> None:
    float32_scores = torch.tensor([0.75], dtype=torch.float32)
    float64_scores = torch.tensor([0.75], dtype=torch.float64)
    float32_threshold = float(
        torch.nextafter(
            float32_scores[0], torch.full_like(float32_scores[0], math.inf)
        )
    )
    float64_threshold = float(
        torch.nextafter(
            float64_scores[0], torch.full_like(float64_scores[0], math.inf)
        )
    )

    thresholds = _thresholds_above_tail(
        [float32_scores, float64_scores], maximum_tail_thresholds=2
    )

    assert float32_threshold in thresholds
    assert float64_threshold in thresholds
    assert not bool(float32_scores[0] >= float32_threshold)
    assert not bool(float64_scores[0] >= float64_threshold)


@pytest.mark.parametrize(
    "scores",
    (
        torch.tensor([0.75], dtype=torch.float16),
        torch.tensor([0.75], dtype=torch.bfloat16),
        torch.tensor([0], dtype=torch.int64),
        torch.tensor([False], dtype=torch.bool),
    ),
)
def test_tail_thresholds_reject_unsupported_source_dtype(
    scores: torch.Tensor,
) -> None:
    with pytest.raises(TypeError, match="must use float32 or float64"):
        _thresholds_above_tail([scores], maximum_tail_thresholds=1)


def test_exact_one_score_keeps_saturated_inclusive_boundary() -> None:
    thresholds = _thresholds_above_tail(
        [torch.tensor([1.0], dtype=torch.float32)],
        maximum_tail_thresholds=1,
    )

    assert thresholds == (0.2, 0.4, 0.6, 0.8, 1.0)
    assert bool(torch.tensor(1.0, dtype=torch.float32) >= thresholds[-1])


def test_exact_one_error_remains_in_gate_inventory_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    training, training_prediction = _episode(
        31, [0.95, 1.0], dtype=torch.float32
    )
    calibration, calibration_prediction = _episode(
        32, [0.95, 1.0], dtype=torch.float32
    )
    for prediction in (training_prediction, calibration_prediction):
        prediction["mean_gate"] = torch.tensor(
            [0.95, 0.9], dtype=torch.float32
        )

    def metrics(
        predictions: dict[int, dict[str, torch.Tensor]],
        episodes: list[Any],
        runtime: object,
    ) -> dict[str, Any]:
        beneficial_active = 0
        disallowed_active = 0
        for episode in episodes:
            prediction = predictions[episode.seed]
            active = (
                (prediction["action_confidence"] >= runtime.action_probability_threshold)
                & (prediction["mean_gate"] >= runtime.gate_probability_threshold)
                & (
                    prediction["minimum_gate"]
                    >= runtime.minimum_member_gate_probability
                )
            )
            beneficial_active += int(active[0])
            disallowed_active += int(active[1])
        totals = _metric_totals(early_covered=beneficial_active > 0)
        totals["unbeneficial_overrides"] = disallowed_active
        return {"total": totals}

    monkeypatch.setattr(calibration_module.trainer, "_metrics", metrics)
    runtime = calibrate_temporal_verifier(
        {31: training_prediction, 32: calibration_prediction},
        [training],
        [calibration],
        ensemble_size=1,
    )

    assert runtime.gate_probability_threshold > 0.9
    assert bool(
        training_prediction["action_confidence"][1]
        >= runtime.action_probability_threshold
    )
    assert not bool(
        training_prediction["mean_gate"][1]
        >= runtime.gate_probability_threshold
    )


def test_temporal_calibration_rejects_overlapping_split_seeds() -> None:
    shared = SimpleNamespace(seed=41)

    with pytest.raises(ValueError, match="split roles overlap"):
        calibrate_temporal_verifier(
            {}, [shared], [shared], ensemble_size=1
        )


def test_nonfinite_confidence_is_excluded_fail_closed_from_tail_inventory() -> None:
    episode, prediction = _episode(
        51,
        [float("nan"), float("inf"), -float("inf")],
        beneficial_rows=(),
    )

    assert temporal_confidence_thresholds(
        {episode.seed: prediction}, [episode]
    ) == (0.2, 0.4, 0.6, 0.8)
    with pytest.raises(ValueError, match=r"outside \[0, 1\]"):
        _thresholds_above_tail(
            [torch.tensor([float("nan")], dtype=torch.float32)],
            maximum_tail_thresholds=1,
        )


def test_temporal_calibration_has_no_permissive_fallback_when_no_solution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    training, training_prediction = _episode(61, [0.8])
    calibration, calibration_prediction = _episode(62, [0.8])

    def no_coverage(*_args: object, **_kwargs: object) -> dict[str, Any]:
        return {"total": _metric_totals(early_covered=False)}

    monkeypatch.setattr(calibration_module.trainer, "_metrics", no_coverage)
    with pytest.raises(ValueError, match="no fail-closed temporal-verifier"):
        calibrate_temporal_verifier(
            {61: training_prediction, 62: calibration_prediction},
            [training],
            [calibration],
            ensemble_size=1,
        )


def test_temporal_calibration_never_accepts_or_reads_an_audit_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    training, training_prediction = _episode(71, [0.8])
    calibration, calibration_prediction = _episode(72, [0.8])
    _audit, audit_prediction = _episode(73, [0.99])
    accesses: list[int] = []
    metric_episode_seeds: list[tuple[int, ...]] = []

    class TrackingPredictions(dict[int, dict[str, torch.Tensor]]):
        def __getitem__(self, seed: int) -> dict[str, torch.Tensor]:
            accesses.append(seed)
            if seed == 73:
                raise AssertionError("audit prediction was accessed")
            return super().__getitem__(seed)

    predictions = TrackingPredictions({
        71: training_prediction,
        72: calibration_prediction,
        73: audit_prediction,
    })

    def covered(
        _predictions: object,
        episodes: list[Any],
        _runtime: object,
    ) -> dict[str, Any]:
        metric_episode_seeds.append(tuple(episode.seed for episode in episodes))
        return {"total": _metric_totals(early_covered=True)}

    monkeypatch.setattr(calibration_module.trainer, "_metrics", covered)
    calibrate_temporal_verifier(
        predictions, [training], [calibration], ensemble_size=1
    )

    assert "audit_episodes" not in inspect.signature(
        calibrate_temporal_verifier
    ).parameters
    assert accesses
    assert set(accesses) == {71, 72}
    assert metric_episode_seeds
    assert set(metric_episode_seeds) == {(71,), (72,)}
