from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest

from stg_lab import cli
from stg_lab.engine_mpc import load_region_dynamics_memory
from stg_lab.provenance import file_sha256
from stg_lab.region_dynamics_training import (
    train_region_dynamics,
    write_region_dynamics_training,
)


def _radius(relative_frame: int) -> float:
    phase = relative_frame % 180
    if phase < 30:
        return 7.0 + 0.7 * phase
    if phase < 60:
        # The real engine median alternates on the upper plateau.
        return 28.0 if ((phase - 30) // 3) % 2 == 0 else 27.3
    if phase < 90:
        return 28.0 - 0.7 * (phase - 60)
    return 7.0


def _visible_observation(relative_frame: int) -> dict[str, object]:
    offset = 40.0 * math.sin(2.0 * math.pi * relative_frame / 360.0)
    records = [
        {
            "id": index,
            "x": x + offset,
            "y": 200.0,
            "collidable": True,
            # Raw motion fields are deliberately false and must be ignored.
            "dx": 900.0 + index,
            "vx": -900.0 - index,
        }
        for index, x in enumerate((-48.0, 0.0, 48.0), start=1)
    ]
    records.extend(
        {
            "id": 100 + index,
            "x": x,
            "y": 100.0,
            "collidable": True,
            "dx": -700.0,
            "vx": 700.0,
        }
        for index, x in enumerate((-72.0, -24.0, 24.0, 72.0), start=1)
    )
    return {"indestructibles": records}


def _artifact(*, shift: int = 0, scenario: str = "okuu:Lunatic", attack: int = 3):
    decisions = [
        {
            "source_frame": shift + relative_frame,
            "region_observed_radius": _radius(relative_frame),
            "control_source": "live_mpc",
            # Route-like output is present in the source report but is never fitted.
            "action": {"move_x": relative_frame % 2, "shoot": True},
            "reporting_only_authority_player": {"x": 123.0, "y": -45.0},
            "recorded_controller_input_observation": _visible_observation(
                relative_frame
            ),
        }
        for relative_frame in range(0, 3 * 180, 3)
    ]
    decisions[-1]["region_observed_radius"] = 6.3
    return {
        "schema_version": 1,
        "run_kind": "live_luastg_delayed_visible_mpc_teacher",
        "scenario": scenario,
        "attack": attack,
        "policy_validation_eligible": True,
        "config": {
            "authority_state_shield": False,
            "spell_forced_off": True,
        },
        "decisions": decisions,
    }


def _write(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _assert_expected_model(model: dict[str, object]) -> None:
    assert model == {
        "phase_order": [
            "expanding",
            "maximum_hold",
            "contracting",
            "minimum_hold",
        ],
        "minimum_radius": 7.0,
        "maximum_radius": 28.0,
        "growth_rate": 0.7,
        "contraction_rate": 0.7,
        "phase_durations": {
            "expanding": 30.0,
            "maximum_hold": 30.0,
            "contracting": 30.0,
            "minimum_hold": 90.0,
        },
        "cycle_frames": 180.0,
        "lateral_flow": {
            "cycle_frames": 360.0,
            "safe_side_rule": "opposite_incoming_lateral_flow",
        },
    }


def test_training_writes_strict_memory_and_separate_provenance(tmp_path: Path) -> None:
    source = tmp_path / "engine.json"
    memory_path = tmp_path / "memory.json"
    report_path = tmp_path / "training-report.json"
    _write(source, _artifact())

    result = train_region_dynamics([source])
    write_region_dynamics_training(
        result,
        memory_output=memory_path,
        report_output=report_path,
    )

    memory = json.loads(memory_path.read_text())
    assert memory["schema_version"] == 2
    assert set(memory) == {"schema_version", "kind", "scenario", "attack", "model"}
    _assert_expected_model(memory["model"])
    loaded = load_region_dynamics_memory(
        memory_path,
        scenario="okuu:Lunatic",
        attack=3,
    )
    assert loaded.cycle_frames == 180.0
    assert loaded.minimum_hold_frames == 90.0

    report = json.loads(report_path.read_text())
    assert report["inputs"] == [{
        "path": str(source),
        "sha256": file_sha256(source),
        "radius_sample_count": 180,
        "lateral_flow_sample_count": 179,
    }]
    assert report["schema_version"] == 2
    assert report["aggregate_samples"]["growth_rate"]["count"] == 3
    assert report["aggregate_samples"]["growth_rate"]["values"] == [0.7] * 3
    assert report["aggregate_samples"]["cycle_interval"]["values"] == [180.0] * 2
    assert report["aggregate_samples"]["minimum_hold_interval"]["values"] == [90.0] * 2
    assert report["aggregate_samples"]["lateral_flow_fit"][
        "observation_contract"
    ] == {
        "row_selection": "highest_visible_indestructible_collision_row",
        "velocity_estimator": "consecutive_visible_position_displacement",
        "raw_velocity_fields_used": False,
        "class_or_script_timer_fields_used": False,
        "minimum_matched_row_objects": 3,
    }
    assert report["aggregate_samples"]["lateral_flow_fit"]["repeat"] == {
        "lag_frames": 360.0,
        "pair_count": 59,
        "normalized_rmse": 0.0,
        "correlation": 1.0,
    }
    assert report["aggregate_samples"]["lateral_flow_fit"][
        "half_cycle_sign_inversion"
    ] == {
        "lag_frames": 180.0,
        "pair_count": 119,
        "normalized_rmse": 0.0,
        "correlation": 1.0,
    }

    def keys(value: object) -> set[str]:
        if isinstance(value, dict):
            return set(value) | {
                nested
                for item in value.values()
                for nested in keys(item)
            }
        if isinstance(value, list):
            return {nested for item in value for nested in keys(item)}
        return set()

    assert keys(memory).isdisjoint({
        "source_frame",
        "action",
        "player",
        "coordinate",
        "transition_frame",
        "x",
        "y",
        "dx",
        "vx",
        "waypoint",
        "recorded_controller_input_observation",
    })


def test_fit_is_invariant_to_an_absolute_timeline_shift(tmp_path: Path) -> None:
    original = tmp_path / "original.json"
    shifted = tmp_path / "shifted.json"
    _write(original, _artifact(shift=0))
    _write(shifted, _artifact(shift=50_000))

    first = train_region_dynamics([original])
    second = train_region_dynamics([shifted])

    assert first.memory == second.memory
    assert first.report["aggregate_samples"] == second.report["aggregate_samples"]
    assert "50000" not in json.dumps(second.report["aggregate_samples"])


def test_fit_uses_visible_position_displacement_not_raw_velocity(
    tmp_path: Path,
) -> None:
    original = tmp_path / "original.json"
    no_raw_motion = tmp_path / "no-raw-motion.json"
    tampered_raw_motion = tmp_path / "tampered-raw-motion.json"
    payload = _artifact()
    _write(original, payload)

    stripped = copy.deepcopy(payload)
    tampered = copy.deepcopy(payload)
    for decision in stripped["decisions"]:
        for record in decision["recorded_controller_input_observation"][
            "indestructibles"
        ]:
            record.pop("dx", None)
            record.pop("vx", None)
    for decision_index, decision in enumerate(tampered["decisions"]):
        for record_index, record in enumerate(
            decision["recorded_controller_input_observation"]["indestructibles"]
        ):
            record["dx"] = 1_000_000.0 + decision_index + record_index
            record["vx"] = -1_000_000.0 - decision_index - record_index
    _write(no_raw_motion, stripped)
    _write(tampered_raw_motion, tampered)

    expected = train_region_dynamics([original])
    without_raw = train_region_dynamics([no_raw_motion])
    with_tampering = train_region_dynamics([tampered_raw_motion])

    assert without_raw.memory == expected.memory == with_tampering.memory
    assert (
        without_raw.report["aggregate_samples"]
        == expected.report["aggregate_samples"]
        == with_tampering.report["aggregate_samples"]
    )


def test_training_requires_recorded_visible_positions(tmp_path: Path) -> None:
    source = tmp_path / "no-visible-observations.json"
    artifact = _artifact()
    for decision in artifact["decisions"]:
        decision["recorded_controller_input_observation"] = None
    _write(source, artifact)

    with pytest.raises(ValueError, match="recorded controller input observations"):
        train_region_dynamics([source])


def test_inconsistent_object_displacements_are_not_fitted(tmp_path: Path) -> None:
    source = tmp_path / "inconsistent-visible-positions.json"
    artifact = _artifact()
    for decision_index, decision in enumerate(artifact["decisions"]):
        records = decision["recorded_controller_input_observation"][
            "indestructibles"
        ]
        records[0]["x"] += 10_000.0 if decision_index % 2 else -10_000.0
    _write(source, artifact)

    with pytest.raises(ValueError, match="recorded controller input observations"):
        train_region_dynamics([source])


def test_training_aggregates_runs_and_rejects_mixed_attack_identity(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    wrong_attack = tmp_path / "wrong.json"
    _write(first, _artifact())
    _write(second, _artifact(shift=10_000))
    _write(wrong_attack, _artifact(attack=4))

    result = train_region_dynamics([first, second])
    _assert_expected_model(result.memory["model"])
    assert len(result.report["inputs"]) == 2
    assert result.report["aggregate_samples"]["cycle_interval"]["count"] == 4

    with pytest.raises(ValueError, match="same scenario and attack"):
        train_region_dynamics([first, wrong_attack])


def test_cli_emits_report_and_keeps_memory_pure(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "engine.json"
    memory = tmp_path / "out" / "memory.json"
    report = tmp_path / "out" / "report.json"
    _write(source, _artifact())

    assert cli.main([
        "train-region-dynamics",
        "--input", str(source),
        "--memory-output", str(memory),
        "--report-output", str(report),
    ]) == 0

    emitted = json.loads(capsys.readouterr().out)
    assert emitted == json.loads(report.read_text())
    _assert_expected_model(json.loads(memory.read_text())["model"])


def test_training_requires_repeated_complete_phase_transitions(tmp_path: Path) -> None:
    source = tmp_path / "short.json"
    artifact = _artifact()
    artifact["decisions"] = artifact["decisions"][:40]
    _write(source, artifact)

    with pytest.raises(ValueError, match="transitions"):
        train_region_dynamics([source])


def test_training_rejects_action_assisted_sources(tmp_path: Path) -> None:
    source = tmp_path / "action-assisted.json"
    artifact = _artifact()
    artifact["decisions"][0]["control_source"] = "recorded_prefix"
    _write(source, artifact)

    with pytest.raises(ValueError, match="non-live actions"):
        train_region_dynamics([source])

    artifact = _artifact()
    artifact["recorded_prefix"] = {"enabled": True}
    _write(source, artifact)
    with pytest.raises(ValueError, match="action-assisted"):
        train_region_dynamics([source])
