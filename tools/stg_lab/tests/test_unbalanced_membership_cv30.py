import inspect
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from experiments import compare_certified_membership_cv30 as balanced
from experiments.compare_preferred_objectives_cv import EXPECTED_V81_CONFIG
from experiments.compare_unbalanced_membership_cv30 import (
    BALANCED_MEMBERSHIP_LOSS_MODE,
    BASE_SEED,
    DEFAULT_OUTPUT,
    MEMBERSHIP_LOSS_MODE,
    SCREENING_EPOCHS,
    SECOND_ADAPTIVE_SCREEN_CONTEXT,
    UNBALANCED_ARM_NAME,
    UNBALANCED_MEMBERSHIP_OBJECTIVE_CONFIG,
    UNBALANCED_MEMBERSHIP_TRAINING_CONFIG,
    _argument_parser,
    _reserve_new_output_path,
    _run_unbalanced_membership_arm,
    _training_control_differences,
    _unbalanced_adaptive_development_gate,
    _validate_new_output_path,
    main,
)


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


def test_unbalanced_protocol_changes_only_membership_loss_mode() -> None:
    assert MEMBERSHIP_LOSS_MODE == "unweighted"
    assert BALANCED_MEMBERSHIP_LOSS_MODE == "balanced"
    assert _training_control_differences() == {
        "membership_loss_mode": ("balanced", "unweighted")
    }
    for key, value in balanced.MEMBERSHIP_TRAINING_CONFIG.items():
        assert UNBALANCED_MEMBERSHIP_TRAINING_CONFIG[key] == value
    assert (
        UNBALANCED_MEMBERSHIP_OBJECTIVE_CONFIG["action_logit_mode"]
        == "certified_membership"
    )
    assert (
        UNBALANCED_MEMBERSHIP_OBJECTIVE_CONFIG["parent_copy_weight"]
        == balanced.MEMBERSHIP_OBJECTIVE_CONFIG["parent_copy_weight"]
        == 0.0
    )


def test_unbalanced_protocol_freezes_seed_epochs_model_and_folds() -> None:
    assert BASE_SEED == balanced.BASE_SEED == 20260901
    assert SCREENING_EPOCHS == balanced.SCREENING_EPOCHS == 6
    assert UNBALANCED_ARM_NAME == "certified_membership_unweighted"
    folds = balanced._fixed_cv30_folds()
    assert [
        (len(fold.fit_seeds), len(fold.calibration_seeds), len(fold.audit_seeds))
        for fold in folds
    ] == [(16, 4, 10)] * 3
    assert sorted(seed for fold in folds for seed in fold.audit_seeds) == sorted(
        balanced.ALL_TRAINING_SEEDS
    )
    for fold in folds:
        for seeds, expected_counts in (
            (fold.fit_seeds, [8, 8]),
            (fold.calibration_seeds, [2, 2]),
            (fold.audit_seeds, [5, 5]),
        ):
            audit = balanced._split_acquisition_audit(seeds)
            assert list(audit["counts"].values()) == expected_counts
            assert audit["strictly_interleaved"] is True
    failure = {"fit_checkpoint": {"adapter_config": EXPECTED_V81_CONFIG}}
    config = balanced._membership_adapter_config(failure)
    expected = {**EXPECTED_V81_CONFIG, "ensemble_size": 1}
    expected["action_logit_mode"] = "certified_membership"
    expected["per_action_membership_confidence"] = False
    assert asdict(config) == expected


def test_cli_does_not_expose_epoch_seed_retry_or_objective_controls() -> None:
    parser = _argument_parser()
    destinations = {action.dest for action in parser._actions}
    assert destinations == {
        "help",
        "failure",
        "expansion_inventory",
        "parent",
        "output",
        "cpu_threads",
    }
    for prohibited in ("epochs", "seed", "retry", "membership_loss_mode"):
        assert prohibited not in destinations


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
    arguments = ["compare_unbalanced_membership_cv30.py"]
    if not use_default:
        arguments.extend(("--output", str(output)))
    monkeypatch.setattr(sys, "argv", arguments)

    def unexpected_read(*_args: object, **_kwargs: object) -> object:
        pytest.fail("input loading must not begin when output already exists")

    monkeypatch.setattr(balanced, "_read_json", unexpected_read)
    with pytest.raises(ValueError, match="already exists; refusing to overwrite"):
        main()
    assert output.read_text(encoding="utf-8") == "preserve-me\n"


def test_new_output_validation_preserves_input_collision_error(tmp_path) -> None:
    source = tmp_path / "source.json"
    source.write_text("protected\n", encoding="utf-8")
    with pytest.raises(ValueError, match="output must not overwrite protected input"):
        _validate_new_output_path(source, [source])
    assert source.read_text(encoding="utf-8") == "protected\n"


def test_output_reservation_is_exclusive(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "new.json"
    _reserve_new_output_path(output)
    assert output.is_file()
    with pytest.raises(ValueError, match="already exists; refusing to overwrite"):
        _reserve_new_output_path(output)


def test_arm_wrapper_forwards_only_fixed_unweighted_variant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_run(*args: object, **kwargs: object) -> dict[str, object]:
        captured["args"] = args
        captured.update(kwargs)
        return {"name": kwargs["arm_name"]}

    monkeypatch.setattr(balanced, "_run_membership_arm", fake_run)
    adapter = object()
    episodes: list[object] = []
    fold = object()
    collision = torch.ones(18)
    physical = torch.ones(18)
    result = _run_unbalanced_membership_arm(
        adapter,  # type: ignore[arg-type]
        episodes,  # type: ignore[arg-type]
        fold,  # type: ignore[arg-type]
        member_seed=20261910,
        collision_weights=collision,
        physical_weights=physical,
    )

    assert captured["args"] == (adapter, episodes, fold)
    assert captured["member_seed"] == 20261910
    assert captured["collision_weights"] is collision
    assert captured["physical_weights"] is physical
    assert captured["membership_loss_mode"] == "unweighted"
    assert captured["training_config"] is UNBALANCED_MEMBERSHIP_TRAINING_CONFIG
    assert captured["objective_config"] is UNBALANCED_MEMBERSHIP_OBJECTIVE_CONFIG
    assert captured["arm_name"] == UNBALANCED_ARM_NAME
    assert result == {"name": UNBALANCED_ARM_NAME}


def test_balanced_helper_default_remains_balanced() -> None:
    signature = inspect.signature(balanced._run_membership_arm)
    assert signature.parameters["membership_loss_mode"].default == "balanced"
    assert (
        signature.parameters["training_config"].default
        is balanced.MEMBERSHIP_TRAINING_CONFIG
    )
    assert (
        signature.parameters["objective_config"].default
        is balanced.MEMBERSHIP_OBJECTIVE_CONFIG
    )
    assert signature.parameters["arm_name"].default == balanced.MEMBERSHIP_ARM_NAME


def test_second_adaptive_gate_reuses_fixed_thresholds_with_honest_provenance() -> None:
    base = balanced._adaptive_development_gate(summary())
    gate = _unbalanced_adaptive_development_gate(summary())
    assert gate["criteria"] == base["criteria"]
    assert gate["checks"] == base["checks"]
    assert gate["passed"] is True
    assert gate["eligible_for_fixed_followup"] is True
    assert gate["second_adaptive_development_screen"] is True
    assert gate["independent_statistical_validation"] is False
    assert gate["specified_after_observing_balanced_membership_negative_result"]
    assert gate["preregistered_before_membership_cv30_audit"] is False
    assert gate["preregistered_before_unweighted_membership_cv30_audit"] is True
    assert SECOND_ADAPTIVE_SCREEN_CONTEXT["sequence"] == 2


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("finite_top1", 689),
        ("equivalent_top1", 137),
        ("direction_correct", 137),
        ("speed_correct", 436),
    ),
)
def test_second_adaptive_gate_keeps_each_fixed_floor(
    field: str,
    value: int,
) -> None:
    audit = dict(summary()["outer_audit_micro"])
    audit[field] = value
    gate = _unbalanced_adaptive_development_gate(
        summary(outer_audit_micro=audit)
    )
    assert gate["passed"] is False


def test_second_adaptive_gate_is_fixed_to_six_epochs() -> None:
    gate = _unbalanced_adaptive_development_gate(summary(), epochs=7)
    assert gate["applicable"] is False
    assert gate["passed"] is False
