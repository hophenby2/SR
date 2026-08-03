from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from stg_lab.native_dataset import (
    NativeDemonstrationBuilder,
    NativeEpisodeIdentity,
    UNKNOWN_SCENARIO_CONTEXT,
    contextualize_demonstrations,
    episode_context_key,
    relabel_dagger_demonstration_archive,
    risk_from_clearance,
)
from stg_lab.protocol import Action
from stg_lab.provenance import file_sha256
from stg_lab.training import Demonstrations
from stg_lab.vision import VisionObservation


def visible(value: float) -> VisionObservation:
    return VisionObservation(
        global_frames=np.full((1, 6, 8, 7), value, dtype=np.float32),
        local_frames=np.full((1, 6, 6, 5), value, dtype=np.float32),
        source_frame=int(value),
    )


def _action_payload(discrete: int) -> dict[str, object]:
    payload = Action.from_discrete(discrete).to_dict()
    payload["discrete"] = discrete
    return payload


def _dagger_report(
    teacher: tuple[int, ...],
    student: tuple[int, ...],
    intervened: tuple[bool, ...],
) -> dict[str, object]:
    executed = tuple(
        teacher_action if use_teacher else student_action
        for teacher_action, student_action, use_teacher in zip(
            teacher, student, intervened, strict=True,
        )
    )
    decisions = [
        {
            "decision": index,
            "teacher_action": _action_payload(teacher_action),
            "student_action": _action_payload(student_action),
            "executed_action": _action_payload(executed_action),
            "teacher_intervened": use_teacher,
            "student_teacher_agreement": teacher_action == student_action,
        }
        for index, (
            teacher_action, student_action, executed_action, use_teacher,
        ) in enumerate(zip(teacher, student, executed, intervened, strict=True))
    ]
    return {
        "schema_version": 1,
        "run_kind": "live_luastg_native_dagger",
        "implementation_sha256": "a" * 64,
        "success": True,
        "passed": True,
        "episode_kind": "attack",
        "scenario": "okuu:Lunatic",
        "attack": 3,
        "seed": 20260808,
        "profile": "expert",
        "terminated": True,
        "termination_reason": "attack_complete",
        "engine_termination_reason": "attack_complete",
        "decision_count": len(decisions),
        "teacher_interventions": sum(intervened),
        "student_teacher_agreements": sum(
            left == right for left, right in zip(teacher, student, strict=True)
        ),
        "outcome_evidence": {"final_player": {"death": 0}},
        "decisions": decisions,
    }


def _write_dagger_archive(path: Path, teacher: tuple[int, ...]) -> None:
    samples = len(teacher)
    Demonstrations(
        global_frames=np.arange(
            samples * 6 * 3 * 2, dtype=np.float32,
        ).reshape(samples, 1, 6, 3, 2),
        local_frames=np.arange(
            samples * 6 * 2 * 2, dtype=np.float32,
        ).reshape(samples, 1, 6, 2, 2),
        actions=np.asarray(teacher, dtype=np.int64).reshape(samples, 1),
        risks=np.linspace(0.1, 0.9, samples, dtype=np.float32).reshape(samples, 1),
        previous_actions=np.asarray((-1, *teacher[:-1]), dtype=np.int64).reshape(
            samples, 1,
        ),
        memory=np.arange(samples * 3, dtype=np.float32).reshape(samples, 1, 3),
        episode_ids=np.zeros(samples, dtype=np.int64),
        supervision_mask=np.ones((samples, 1), dtype=bool),
    ).save(path)
    with np.load(path) as source:
        arrays = {name: source[name] for name in source.files}
    arrays["future_optional_field"] = np.arange(samples, dtype=np.int16)
    np.savez_compressed(path, **arrays)


def test_builder_keeps_only_strict_successes_and_preserves_episode_blocks(
    tmp_path: Path,
) -> None:
    builder = NativeDemonstrationBuilder()
    first = builder.begin(NativeEpisodeIdentity("attack", "a:Lunatic", 1, 10))
    first.record(visible(1), Action(move_x=1), 0.25)
    first.record(visible(2), Action(move_y=1, slow=True), 0.5)
    builder.finish(first, strict_success=True, termination_reason="attack_complete")
    failed = builder.begin(NativeEpisodeIdentity("attack", "b:Lunatic", 2, 11))
    failed.record(visible(3), Action(move_x=-1), 1.0)
    builder.finish(failed, strict_success=False, termination_reason="player_hit")
    second = builder.begin(NativeEpisodeIdentity("stage", "Stage 1@Lunatic", None, 12))
    second.record(visible(4), Action(), 0.1)
    builder.finish(second, strict_success=True, termination_reason="stage_complete")

    output = tmp_path / "native.npz"
    manifest = builder.save(output, manifest_path=tmp_path / "native.json")
    demonstrations = Demonstrations.load(output)

    np.testing.assert_array_equal(demonstrations.episode_ids, (0, 0, 1))
    assert demonstrations.actions.shape == (3, 1)
    np.testing.assert_array_equal(demonstrations.previous_actions, ((-1,), (5,), (-1,)))
    assert demonstrations.memory is None
    assert demonstrations.proficiency is None
    assert "previous_executed_motor_action" not in manifest["model_inputs"]
    assert "previous_executed_motor_action" in manifest["recorded_fields"]
    assert manifest["samples"] == 3
    assert len(manifest["accepted_episodes"]) == 2
    assert len(manifest["rejected_episodes"]) == 1


def test_corrective_dagger_relabel_preserves_arrays_and_uses_executed_actions(
    tmp_path: Path,
) -> None:
    source = tmp_path / "teacher.npz"
    report_path = tmp_path / "dagger.json"
    output = tmp_path / "corrective.npz"
    manifest_path = tmp_path / "corrective.manifest.json"
    teacher = (5, 13, 6)
    student = (3, 13, 8)
    intervened = (False, True, False)
    _write_dagger_archive(source, teacher)
    report_path.write_text(
        json.dumps(_dagger_report(teacher, student, intervened)),
        encoding="utf-8",
    )
    source_sha256 = file_sha256(source)

    manifest = relabel_dagger_demonstration_archive(
        source, report_path, output, manifest_path,
    )

    assert file_sha256(source) == source_sha256
    with np.load(source) as before, np.load(output) as after:
        assert before.files == after.files
        for name in before.files:
            if name == "actions":
                np.testing.assert_array_equal(after[name], ((3,), (13,), (8,)))
            else:
                np.testing.assert_array_equal(after[name], before[name])
                assert after[name].dtype == before[name].dtype
    assert manifest["source_dataset_sha256"] == source_sha256
    assert manifest["source_dagger_report_sha256"] == file_sha256(report_path)
    assert manifest["source_implementation_sha256"] == "a" * 64
    assert manifest["dataset_sha256"] == file_sha256(output)
    assert manifest["teacher_interventions"] == 1
    assert manifest["student_executions"] == 2
    assert manifest["replaced_labels"] == 2
    assert manifest["unchanged_labels"] == 1
    assert "future_optional_field" in manifest["preserved_arrays"]
    assert manifest["accepted_episodes"][0]["strict_success"] is True
    assert "executed_action.discrete" in manifest["label_semantics"]["output"]
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest


def test_corrective_dagger_relabel_can_supervise_interventions_only(
    tmp_path: Path,
) -> None:
    source = tmp_path / "teacher.npz"
    report_path = tmp_path / "dagger.json"
    output = tmp_path / "corrections.npz"
    manifest_path = tmp_path / "corrections.manifest.json"
    teacher = (5, 13, 6)
    student = (3, 13, 8)
    intervened = (False, True, False)
    _write_dagger_archive(source, teacher)
    report_path.write_text(
        json.dumps(_dagger_report(teacher, student, intervened)),
        encoding="utf-8",
    )

    manifest = relabel_dagger_demonstration_archive(
        source,
        report_path,
        output,
        manifest_path,
        interventions_only=True,
    )
    demonstrations = Demonstrations.load(output)

    np.testing.assert_array_equal(demonstrations.actions, ((3,), (13,), (8,)))
    np.testing.assert_array_equal(
        demonstrations.supervision_mask,
        ((False,), (True,), (False,)),
    )
    assert demonstrations.global_frames.shape[0] == 3
    assert manifest["action_supervision"] == {
        "mode": "teacher_interventions_only",
        "mask": "supervision_mask",
        "supervised_decisions": 1,
        "unsupervised_context_decisions": 2,
        "recurrent_context_decisions": 3,
        "risk_targets_available": 3,
    }
    assert "supervision_mask" not in manifest["preserved_arrays"]
    assert manifest["label_semantics"]["supervision_mask"] == (
        "true only where teacher_intervened=true"
    )


def test_corrective_dagger_relabel_accepts_direct_corrective_archive(
    tmp_path: Path,
) -> None:
    source = tmp_path / "direct-corrective.npz"
    report_path = tmp_path / "dagger.json"
    output = tmp_path / "corrections.npz"
    manifest_path = tmp_path / "corrections.manifest.json"
    teacher = (5, 13, 6)
    student = (3, 13, 8)
    intervened = (False, True, False)
    executed = (3, 13, 8)
    _write_dagger_archive(source, executed)
    report = _dagger_report(teacher, student, intervened)
    report["demonstration_supervision"] = {"mode": "corrective"}
    for decision in report["decisions"]:
        decision["supervised_action"] = decision["executed_action"]
    report_path.write_text(json.dumps(report), encoding="utf-8")

    manifest = relabel_dagger_demonstration_archive(
        source,
        report_path,
        output,
        manifest_path,
        interventions_only=True,
    )
    demonstrations = Demonstrations.load(output)

    np.testing.assert_array_equal(demonstrations.actions, ((3,), (13,), (8,)))
    np.testing.assert_array_equal(
        demonstrations.supervision_mask,
        ((False,), (True,), (False,)),
    )
    assert manifest["source_supervision_mode"] == "corrective"
    assert manifest["replaced_labels"] == 0
    assert manifest["label_semantics"]["source"] == (
        "executed_action.discrete from corrective DAgger collection"
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("success", False, "does not claim strict success"),
        ("terminated", False, "was not terminated"),
        ("termination_reason", "max_frames", "did not reach attack_complete"),
    ),
)
def test_corrective_dagger_relabel_rejects_non_strict_report(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    source = tmp_path / "teacher.npz"
    report_path = tmp_path / "dagger.json"
    teacher = (5,)
    _write_dagger_archive(source, teacher)
    report = _dagger_report(teacher, (5,), (False,))
    report[field] = value
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        relabel_dagger_demonstration_archive(
            source,
            report_path,
            tmp_path / "output.npz",
            tmp_path / "manifest.json",
        )


def test_corrective_dagger_relabel_rejects_death_and_misaligned_sources(
    tmp_path: Path,
) -> None:
    source = tmp_path / "teacher.npz"
    report_path = tmp_path / "dagger.json"
    teacher = (5, 6)
    report = _dagger_report(teacher, (3, 7), (False, False))
    _write_dagger_archive(source, teacher)
    report["outcome_evidence"] = {"final_player": {"death": 1}}
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="death is not zero"):
        relabel_dagger_demonstration_archive(
            source, report_path, tmp_path / "a.npz", tmp_path / "a.json",
        )

    report = _dagger_report(teacher, (3, 7), (False, False))
    report["decisions"] = report["decisions"][:-1]
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="decision count does not match"):
        relabel_dagger_demonstration_archive(
            source, report_path, tmp_path / "b.npz", tmp_path / "b.json",
        )

    report = _dagger_report((4, 6), (3, 7), (False, False))
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="labels do not match teacher_action"):
        relabel_dagger_demonstration_archive(
            source, report_path, tmp_path / "c.npz", tmp_path / "c.json",
        )

    report = _dagger_report(teacher, (3, 7), (True, False))
    report["teacher_interventions"] = True
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="teacher_interventions count"):
        relabel_dagger_demonstration_archive(
            source, report_path, tmp_path / "d.npz", tmp_path / "d.json",
        )


@pytest.mark.parametrize(
    "implementation_sha256",
    (None, "", "a" * 63, "A" * 64, "z" * 64),
)
def test_corrective_dagger_relabel_requires_valid_historical_source_fingerprint(
    tmp_path: Path,
    implementation_sha256: object,
) -> None:
    source = tmp_path / "teacher.npz"
    report_path = tmp_path / "dagger.json"
    teacher = (5,)
    _write_dagger_archive(source, teacher)
    report = _dagger_report(teacher, (5,), (False,))
    if implementation_sha256 is None:
        del report["implementation_sha256"]
    else:
        report["implementation_sha256"] = implementation_sha256
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="implementation_sha256"):
        relabel_dagger_demonstration_archive(
            source,
            report_path,
            tmp_path / "output.npz",
            tmp_path / "manifest.json",
        )


def test_clearance_risk_is_bounded_and_decreases_with_margin() -> None:
    assert risk_from_clearance(float("-inf")) == 1.0
    assert risk_from_clearance(float("inf")) == 0.0
    assert risk_from_clearance(0.0) == 1.0
    assert 0.0 < risk_from_clearance(20.0) < risk_from_clearance(4.0) < 1.0


def test_contextualize_demonstrations_adds_identity_only_one_hot_tokens() -> None:
    builder = NativeDemonstrationBuilder()
    first = builder.begin(NativeEpisodeIdentity("attack", "a:Lunatic", 2, 10))
    first.record(visible(1), Action(move_x=1), 0.25)
    first.record(visible(2), Action(move_y=1), 0.5)
    builder.finish(first, strict_success=True, termination_reason="attack_complete")
    second = builder.begin(NativeEpisodeIdentity("stage", "Stage 1@Normal", None, 11))
    second.record(visible(3), Action(), 0.1)
    builder.finish(second, strict_success=True, termination_reason="stage_complete")
    demonstrations = builder.build()
    identities = (
        NativeEpisodeIdentity("attack", "a:Lunatic", 2, 10),
        NativeEpisodeIdentity("stage", "Stage 1@Normal", None, 11),
    )

    conditioned, vocabulary, contexts = contextualize_demonstrations(
        demonstrations,
        identities,
    )

    assert vocabulary == (
        UNKNOWN_SCENARIO_CONTEXT,
        "attack:a:Lunatic#2",
        "stage:Stage 1@Normal",
    )
    assert contexts == ("attack:a:Lunatic#2", "stage:Stage 1@Normal")
    np.testing.assert_array_equal(conditioned.memory[0], ((0, 1, 0),))
    np.testing.assert_array_equal(conditioned.memory[1], ((0, 1, 0),))
    np.testing.assert_array_equal(conditioned.memory[2], ((0, 0, 1),))

    motor_conditioned, motor_vocabulary, _ = contextualize_demonstrations(
        demonstrations,
        identities,
        include_previous_action=True,
    )
    assert motor_conditioned.memory.shape == (3, 1, len(motor_vocabulary) + 18)
    assert motor_conditioned.memory[0, 0, len(motor_vocabulary):].sum() == 0
    assert motor_conditioned.memory[1, 0, len(motor_vocabulary) + Action(move_x=1).discrete] == 1
    assert motor_conditioned.memory[2, 0, len(motor_vocabulary):].sum() == 0


def test_episode_context_key_rejects_route_like_invalid_identity() -> None:
    assert episode_context_key("attack", "okuu:Lunatic", 3) == (
        "attack:okuu:Lunatic#3"
    )
    assert episode_context_key("stage", "Stage 1@Normal", None) == (
        "stage:Stage 1@Normal"
    )
    with np.testing.assert_raises_regex(ValueError, "positive attack"):
        episode_context_key("attack", "okuu:Lunatic", None)
