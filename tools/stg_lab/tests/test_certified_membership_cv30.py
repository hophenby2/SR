from dataclasses import asdict

import pytest

torch = pytest.importorskip("torch")

from experiments.compare_certified_membership_cv30 import (
    ADAPTIVE_SCREEN_GATE,
    MEMBERSHIP_ARM_NAME,
    MEMBERSHIP_DIAGNOSTIC_POLICY,
    MEMBERSHIP_OBJECTIVE_CONFIG,
    MEMBERSHIP_TRAINING_CONFIG,
    SCREENING_EPOCHS,
    _adaptive_development_gate,
    _average_precision,
    _membership_adapter_config,
    _membership_metrics,
    _membership_split_diagnostics,
    _product_score_diagnostics,
    _reliability,
)
from experiments.compare_plain_certified_set_cv30 import (
    EXPANSION_TRAINING_SEEDS,
    LEGACY_TRAINING_SEEDS,
    _fixed_cv30_folds,
)
from experiments.compare_preferred_objectives_cv import EXPECTED_V81_CONFIG
from experiments.train_temporal_residual_adapter import EpisodeFeatures


def make_episode(seed: int) -> EpisodeFeatures:
    decisions = 4
    action_count = 18
    parent_actions = torch.tensor([0, 0, 4, 3])
    preferred_actions = torch.tensor([1, 2, 4, 3])
    preferred_set = torch.zeros(decisions, action_count, dtype=torch.bool)
    preferred_set[0, [1, 10]] = True
    preferred_set[1, 2] = True
    preferred_set[2, 4] = True
    equivalent = torch.zeros_like(preferred_set)
    equivalent[0, [1, 10]] = True
    equivalent[1, 2] = True
    safe = torch.ones_like(preferred_set)
    return EpisodeFeatures(
        seed=seed,
        dataset=f"dataset-{seed}",
        report=f"report-{seed}",
        manifest=f"manifest-{seed}",
        features=torch.zeros(1, decisions, 3),
        parent_logits=torch.zeros(1, decisions, action_count),
        parent_actions=parent_actions,
        previous_actions=torch.tensor([1, 2, 4, 3]),
        gate_targets=torch.tensor([1.0, 1.0, 1.0, 0.0]),
        gate_valid=torch.ones(decisions, dtype=torch.bool),
        hard_positive=torch.tensor([False, False, True, False]),
        correctable_hard_positive=torch.tensor([False, False, True, False]),
        anticipatory=torch.tensor([True, True, True, False]),
        future_onset_valid=torch.ones(decisions, dtype=torch.bool),
        anticipatory_lead_decisions=torch.tensor([4, 6, 8, 0]),
        preferred_actions=preferred_actions,
        preferred_action_set=preferred_set,
        preferred_equivalent_actions=equivalent,
        preferred_correction_required=torch.tensor([True, True, False, False]),
        safety_candidate_actions=preferred_actions.clone(),
        safety_candidate_valid=torch.ones(decisions, dtype=torch.bool),
        safe_actions=safe,
        evaluation_safe_actions=safe.clone(),
        parent_evaluation_danger=torch.tensor([True, True, False, False]),
        collided_actions=torch.zeros_like(safe),
        minimum_margins=torch.full((decisions, action_count), 32.0),
        minimum_margin_mask=torch.ones_like(safe),
        teacher_selected_collision=torch.zeros(decisions, dtype=torch.bool),
    )


def make_prediction(episode: EpisodeFeatures) -> dict[str, torch.Tensor]:
    probabilities = torch.full((episode.decisions, 18), 0.1)
    probabilities[0, 1] = 0.9
    probabilities[0, 10] = 0.8
    probabilities[1, 2] = 0.6
    probabilities[1, 3] = 0.7
    probabilities[2, 4] = 0.8
    probabilities[3, 3] = 0.9
    candidates = probabilities.argmax(dim=-1)
    return {
        "mean_gate": torch.tensor([0.8, 0.9, 0.1, 0.95]),
        "minimum_gate": torch.tensor([0.8, 0.9, 0.1, 0.95]),
        "action_probabilities": probabilities.unsqueeze(0),
        "mean_action_probabilities": probabilities,
        "action_all_members_finite": torch.ones(
            episode.decisions,
            dtype=torch.bool,
        ),
        "action_confidence": probabilities.amax(dim=-1),
        "candidates": candidates,
        "agreement": torch.ones(episode.decisions),
        "physical_danger_probabilities": torch.full(
            (1, episode.decisions, 18),
            0.1,
        ),
    }


def summary(**overrides: object) -> dict[str, object]:
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


def test_membership_adapter_changes_only_joint_logit_mode() -> None:
    failure = {"fit_checkpoint": {"adapter_config": EXPECTED_V81_CONFIG}}
    config = _membership_adapter_config(failure)
    expected = {**EXPECTED_V81_CONFIG, "ensemble_size": 1}
    expected["action_logit_mode"] = "certified_membership"
    expected["per_action_membership_confidence"] = False
    assert asdict(config) == expected


def test_membership_protocol_is_one_fixed_arm_with_zero_parent_copy() -> None:
    assert MEMBERSHIP_ARM_NAME == "certified_membership"
    assert SCREENING_EPOCHS == 6
    assert MEMBERSHIP_TRAINING_CONFIG["learning_rate"] == 3e-4
    assert MEMBERSHIP_TRAINING_CONFIG["weight_decay"] == 1e-3
    assert MEMBERSHIP_TRAINING_CONFIG["gate_positive_weight"] == 8.0
    assert MEMBERSHIP_TRAINING_CONFIG["preferred_action_loss_weight"] == 12.0
    assert MEMBERSHIP_TRAINING_CONFIG["physical_danger_loss_weight"] == 8.0
    for name in (
        "action_loss_weight",
        "preferred_action_uniform_loss_weight",
        "preferred_action_tiebreak_loss_weight",
        "preferred_action_rank_loss_weight",
        "safety_candidate_loss_weight",
        "parent_copy_weight",
        "collision_loss_weight",
        "minimum_margin_loss_weight",
    ):
        assert MEMBERSHIP_TRAINING_CONFIG[name] == 0.0
    assert MEMBERSHIP_OBJECTIVE_CONFIG["parent_copy_weight"] == 0.0
    assert [
        (len(fold.fit_seeds), len(fold.calibration_seeds), len(fold.audit_seeds))
        for fold in _fixed_cv30_folds()
    ] == [(16, 4, 10)] * 3


def test_adaptive_gate_accepts_exact_plain_cv30_boundaries() -> None:
    gate = _adaptive_development_gate(summary())
    assert gate["passed"] is True
    assert gate["independent_statistical_validation"] is False
    assert gate["specified_after_observing_plain_cv30_negative_result"] is True
    assert all(gate["checks"].values())
    assert ADAPTIVE_SCREEN_GATE["reference_plain_cv30"] == {
        "targets": 690,
        "equivalent_top1": 138,
        "direction_correct": 138,
        "speed_correct": 437,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("finite_top1", 689),
        ("equivalent_top1", 137),
        ("direction_correct", 137),
        ("speed_correct", 436),
    ),
)
def test_adaptive_gate_fails_each_outer_audit_floor(field: str, value: int) -> None:
    audit = dict(summary()["outer_audit_micro"])
    audit[field] = value
    assert (
        _adaptive_development_gate(summary(outer_audit_micro=audit))["passed"] is False
    )


def test_adaptive_gate_fails_calibration_runtime_and_epoch_constraints() -> None:
    assert (
        _adaptive_development_gate(summary(calibration_successful_folds=1))["passed"]
        is False
    )
    assert (
        _adaptive_development_gate(summary(audit_runtime_eligible_folds=1))["passed"]
        is False
    )
    assert (
        _adaptive_development_gate(
            summary(
                calibration_successful_folds=3,
                audit_runtime_eligible_folds=2,
                calibrated_audit_runtime_ineligible_folds=[2],
            )
        )["passed"]
        is False
    )
    assert _adaptive_development_gate(summary(), epochs=7)["applicable"] is False


def test_fixed_membership_metrics_cover_calibration_and_cardinality() -> None:
    episode = make_episode(LEGACY_TRAINING_SEEDS[0])
    predictions = {episode.seed: make_prediction(episode)}
    metrics = _membership_metrics(
        [episode],
        predictions,
        lambda value: value.gate_valid & (value.gate_targets > 0.0),
    )
    assert metrics["rows"] == metrics["finite_rows"] == 3
    assert metrics["action_cells"] == 54
    assert metrics["positive_cells"] == 4
    assert metrics["negative_cells"] == 50
    assert 0.0 <= metrics["brier_score"] <= 1.0
    assert 0.0 <= metrics["balanced_brier_score"] <= 1.0
    assert 0.0 <= metrics["ece"]["expected_calibration_error"] <= 1.0
    assert 0.0 <= metrics["auprc_average_precision"] <= 1.0
    assert metrics["threshold_0_5"]["positive_recall"] == pytest.approx(1.0)
    assert metrics["threshold_0_5"]["false_positive"] == 1
    assert metrics["cardinality"]["true_distribution"] == {"1": 2, "2": 1}
    assert metrics["cardinality"]["predicted_distribution_at_0_5"] == {
        "1": 1,
        "2": 2,
    }
    selected = metrics["selected_certified_reliability"]
    assert selected["selected_target_member"] == 2
    assert len(selected["ece"]["reliability_bins"]) == 10
    margin = metrics["certified_vs_rejected_score_margin"]
    assert margin["certified_top1_rows"] == 2
    assert margin["certified_top1_rate"] == pytest.approx(2 / 3)
    assert margin["summary"]["minimum"] == pytest.approx(-0.1)
    assert margin["summary"]["maximum"] == pytest.approx(0.8)


def test_split_diagnostics_are_fixed_and_stratified() -> None:
    legacy = make_episode(LEGACY_TRAINING_SEEDS[0])
    expansion = make_episode(EXPANSION_TRAINING_SEEDS[0])
    predictions = {
        legacy.seed: make_prediction(legacy),
        expansion.seed: make_prediction(expansion),
    }
    diagnostics = _membership_split_diagnostics(
        [legacy, expansion],
        predictions,
    )
    early = diagnostics["early_correction_required_4_10"]
    assert early["overall"]["rows"] == 4
    assert (
        min(
            early["by_acquisition_cohort"].values(),
            key=lambda row: row["rows"],
        )["rows"]
        == 2
    )
    assert set(early["by_true_cardinality"]) == {"1", "2", "3-4", "5-8", "9+"}
    assert set(early["by_lead_decisions"]) == {
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
    }


def test_product_gap_is_descriptive_and_never_selects_runtime() -> None:
    episode = make_episode(LEGACY_TRAINING_SEEDS[0])
    predictions = {episode.seed: make_prediction(episode)}
    diagnostics = _product_score_diagnostics([episode], predictions)
    assert diagnostics["used_for_calibration_runtime_gate_or_selection"] is False
    scope = diagnostics["scopes"]["changed_candidate_before_physical_veto"]
    assert scope["beneficial_product_score"]["count"] == 1
    assert scope["disallowed_product_score"]["count"] == 1
    assert scope["best_beneficial_minus_max_disallowed_product_gap"] == pytest.approx(
        0.09
    )
    for key in (
        "product_score_used_for_calibration",
        "product_score_used_by_runtime",
        "product_score_used_for_gate",
    ):
        assert MEMBERSHIP_DIAGNOSTIC_POLICY[key] is False


def test_auprc_and_ece_helpers_are_deterministic() -> None:
    assert _average_precision([0.9, 0.8, 0.1], [True, False, True]) == pytest.approx(
        (1.0 + 2 / 3) / 2
    )
    reliability = _reliability([0.9, 0.1], [True, False])
    assert reliability["expected_calibration_error"] == pytest.approx(0.1)
    assert len(reliability["reliability_bins"]) == 10
