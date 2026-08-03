from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np

from stg_lab import cli
from stg_lab.metrics import EpisodeMetrics
from stg_lab.training import Demonstrations, TrainingMetrics


def demonstrations(episode_ids=(0, 1), *, steps: int = 2) -> Demonstrations:
    samples = len(episode_ids)
    return Demonstrations(
        global_frames=np.zeros((samples, steps, 6, 8, 8), dtype=np.float32),
        local_frames=np.zeros((samples, steps, 6, 8, 8), dtype=np.float32),
        actions=np.zeros((samples, steps), dtype=np.int64),
        risks=np.zeros((samples, steps), dtype=np.float32),
        episode_ids=np.asarray(episode_ids, dtype=np.int64),
    )


def episode(scenario: str, seed: int, survived: bool = True) -> EpisodeMetrics:
    return EpisodeMetrics(
        scenario=f"{scenario}:lunatic",
        seed=seed,
        survived=survived,
        frames=12,
        peak_risk=0.2,
        total_risk=0.5,
        state_hash=f"{scenario}-{seed}",
    )


def test_test_command_runs_pytest_and_forwards_arguments(monkeypatch) -> None:
    called = {}

    def fake_run(command, *, cwd, check):
        called.update(command=command, cwd=cwd, check=check)
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    assert cli.main(["test", "--", "-k", "vision"]) == 7
    assert called["command"] == [sys.executable, "-m", "pytest", "-k", "vision"]
    assert called["cwd"] == cli._PROJECT_ROOT
    assert called["check"] is False


def test_explicit_planner_benchmark_uses_public_configs(monkeypatch, tmp_path, capsys) -> None:
    from stg_lab import benchmark

    called = {}

    def fake_benchmark(*args, **kwargs):
        called.update(positional=args, **kwargs)
        return {
            "run_kind": "smoke",
            "acceptance_claim": False,
            "elapsed_seconds": np.float32(0.25),
        }

    monkeypatch.setattr(benchmark, "run_planner_benchmark", fake_benchmark)
    output = tmp_path / "planner.json"

    assert cli.main([
        "test",
        "--planner",
        "--scenario",
        "all",
        "--episodes",
        "3",
        "--duration-frames",
        "90",
        "--motor-delay-frames",
        "0",
        "--action-hold-frames",
        "3",
        "--shield-strategy",
        "toward",
        "--planner-horizon",
        "36",
        "--planner-sample-every",
        "6",
        "--workers",
        "2",
        "--output",
        str(output),
    ]) == 0
    assert called["positional"] == (("stage5_boss3", "stage5_boss4"), "lunatic", (
        20260729,
        20260730,
        20260731,
    ))
    assert called["planner_config"].risk.horizon_frames == 36
    assert called["vision_config"].observation_delay == 5
    assert called["rollout_config"].max_frames == 90
    assert called["rollout_config"].shield_strategy == "toward"
    assert called["simulation_config"].reaction_frames == 0
    assert called["simulation_config"].action_hold_frames == 3
    assert called["workers"] == 2
    payload = json.loads(output.read_text())
    assert payload["elapsed_seconds"] == 0.25
    assert json.loads(capsys.readouterr().out) == payload


def test_multi_scenario_collection_renumbers_episode_ids(monkeypatch) -> None:
    import stg_lab.rollout as rollout

    calls = []

    def fake_collect(factory, seeds, **kwargs):
        calls.append((factory.scenario, tuple(seeds), kwargs))
        data = demonstrations((4, 9))
        return data, tuple(episode(factory.scenario, seed) for seed in seeds)

    monkeypatch.setattr(rollout, "collect_demonstrations", fake_collect)
    args = cli.build_parser().parse_args([
        "train",
        "--scenario",
        "all",
        "--episodes",
        "2",
        "--duration-frames",
        "15",
    ])

    merged, reports = cli._collect_demonstrations(args)
    np.testing.assert_array_equal(merged.episode_ids, (0, 1, 2, 3))
    assert merged.actions.shape[0] == 4
    assert tuple(reports) == ("stage5_boss3", "stage5_boss4")
    assert [call[0] for call in calls] == ["stage5_boss3", "stage5_boss4"]
    assert all(call[1] == (20260729, 20260730) for call in calls)
    assert all(call[2]["config"].max_frames == 15 for call in calls)
    assert all(call[2]["shield"] is True for call in calls)


def test_train_consumes_demonstration_archive(monkeypatch, tmp_path, capsys) -> None:
    demos_path = tmp_path / "demos.npz"
    checkpoint = tmp_path / "policy.pt"
    metrics = tmp_path / "metrics.json"
    demonstrations((0, 0, 1)).save(demos_path)
    called = {}

    def fake_train(loaded, *, policy_config, training_config, output, training_data):
        called.update(
            loaded=loaded,
            policy_config=policy_config,
            training_config=training_config,
            output=output,
            training_data=training_data,
        )
        return object(), [TrainingMetrics(1, 0.5, 0.6, 0.75, 0.1)]

    def fake_write(path, history):
        called.update(metrics_path=path, history=list(history))

    import stg_lab.training as training

    monkeypatch.setattr(training, "train_behavior_cloning", fake_train)
    monkeypatch.setattr(training, "write_metrics", fake_write)

    assert cli.main([
        "train",
        "--demos",
        str(demos_path),
        "--checkpoint",
        str(checkpoint),
        "--metrics",
        str(metrics),
        "--epochs",
        "1",
        "--feature-size",
        "12",
        "--recurrent-size",
        "16",
        "--inference-mode",
        "stream",
        "--device",
        "cpu",
    ]) == 0
    assert called["output"] == checkpoint
    assert called["policy_config"].channels == 6
    assert called["policy_config"].feature_size == 12
    assert called["policy_config"].memory_size == 4
    assert called["policy_config"].inference_mode == "stream"
    assert called["training_config"].epochs == 1
    assert called["training_config"].class_balance is True
    assert called["metrics_path"] == metrics
    summary = json.loads(capsys.readouterr().out)
    assert summary["samples"] == 3
    assert summary["episode_groups"] == 2
    assert summary["run"]["run_kind"] == "dataset_training"
    assert summary["run"]["acceptance_claim"] is False
    assert summary["final_metrics"]["action_accuracy"] == 0.75


def test_stateful_train_disables_handwritten_inputs(monkeypatch, tmp_path, capsys) -> None:
    import stg_lab.stateful_training as stateful_training

    demos_path = tmp_path / "native.npz"
    demonstrations((0, 0, 1, 1, 2, 2)).save(demos_path)
    called = {}

    def fake_train(loaded, *, policy_config, training_config, output, training_data):
        called.update(
            loaded=loaded,
            policy_config=policy_config,
            training_config=training_config,
            output=output,
            training_data=training_data,
        )
        return object(), [TrainingMetrics(1, 0.4, 0.5, 0.8, 0.1)]

    monkeypatch.setattr(
        stateful_training,
        "train_stateful_behavior_cloning",
        fake_train,
    )
    assert cli.main([
        "train",
        "--demos", str(demos_path),
        "--checkpoint", str(tmp_path / "stream.pt"),
        "--metrics", str(tmp_path / "metrics.json"),
        "--stateful-tbptt",
        "--tbptt-chunk-length", "17",
        "--validation-episode-id", "1",
        "--validation-episode-id", "2",
        "--no-scenario-memory-conditioning",
        "--no-proficiency-conditioning",
        "--no-restore-best-validation",
        "--movement-onset-weight", "5",
        "--direction-change-weight", "2.5",
        "--exact-action-loss-weight", "0.25",
        "--direction-loss-weight", "1.0",
        "--speed-loss-weight", "0.2",
        "--direction-consistency-weight", "0.1",
        "--future-visual-loss-weight", "0.35",
        "--future-visual-horizons", "2", "4", "8",
        "--episode-balanced",
        "--epochs", "1",
        "--device", "cpu",
    ]) == 0

    assert called["policy_config"].inference_mode == "stream"
    assert called["policy_config"].memory_size == 0
    assert called["policy_config"].proficiency_size == 0
    assert called["training_config"].chunk_length == 17
    assert called["training_config"].validation_episode_ids == (1, 2)
    assert called["training_config"].restore_best_validation is False
    assert called["training_config"].movement_onset_weight == 5.0
    assert called["training_config"].direction_change_weight == 2.5
    assert called["training_config"].exact_action_loss_weight == 0.25
    assert called["training_config"].direction_loss_weight == 1.0
    assert called["training_config"].speed_loss_weight == 0.2
    assert called["training_config"].direction_consistency_weight == 0.1
    assert called["training_config"].future_visual_loss_weight == 0.35
    assert called["training_config"].future_visual_horizons == (2, 4, 8)
    assert called["training_config"].episode_balanced is True
    assert called["loaded"].memory is None
    assert called["loaded"].proficiency is None
    assert called["training_data"]["scenario_memory_input"] is False
    assert called["training_data"]["proficiency_conditioning_input"] is False
    summary = json.loads(capsys.readouterr().out)
    assert summary["training_mode"] == "episode_stateful_tbptt"
    assert summary["stateful_loss_controls"] == {
        "movement_onset_weight": 5.0,
        "direction_change_weight": 2.5,
        "episode_balanced": True,
        "exact_action_loss_weight": 0.25,
        "direction_loss_weight": 1.0,
        "speed_loss_weight": 0.2,
        "direction_consistency_weight": 0.1,
        "previous_action_dropout_probability": 0.0,
        "future_visual_loss_weight": 0.35,
        "future_visual_horizons": [2, 4, 8],
    }


def test_correction_only_train_routes_action_and_risk_supervision(
    monkeypatch, tmp_path, capsys,
) -> None:
    import stg_lab.stateful_training as stateful_training

    demos_path = tmp_path / "corrections.npz"
    values = demonstrations((0, 0, 1, 1))
    values.supervision_mask = np.zeros_like(values.actions, dtype=np.bool_)
    values.supervision_mask[1, -1] = True
    values.supervision_mask[3, -1] = True
    values.save(demos_path)
    called = {}

    def fake_train(loaded, *, policy_config, training_config, output, training_data):
        called.update(
            loaded=loaded,
            training_config=training_config,
        )
        return object(), [TrainingMetrics(1, 0.4, 0.5, 0.8, 0.1)]

    monkeypatch.setattr(
        stateful_training,
        "train_stateful_behavior_cloning",
        fake_train,
    )

    assert cli.main([
        "train",
        "--demos", str(demos_path),
        "--checkpoint", str(tmp_path / "corrections.pt"),
        "--stateful-tbptt",
        "--correction-only",
        "--epochs", "1",
        "--device", "cpu",
    ]) == 0

    assert called["training_config"].correction_only is True
    np.testing.assert_array_equal(
        called["loaded"].supervision_mask,
        values.supervision_mask,
    )
    controls = json.loads(capsys.readouterr().out)["stateful_loss_controls"]
    assert controls["correction_only"] is True
    assert controls["action_supervision"] == "supervision_mask"
    assert controls["risk_supervision"] == "all_decisions"


def test_correction_only_train_requires_stateful_tbptt(tmp_path, capsys) -> None:
    demos_path = tmp_path / "corrections.npz"
    values = demonstrations((0, 0, 1, 1))
    values.supervision_mask = np.ones_like(values.actions, dtype=np.bool_)
    values.save(demos_path)

    assert cli.main([
        "train",
        "--demos", str(demos_path),
        "--correction-only",
        "--epochs", "1",
        "--device", "cpu",
    ]) == 2
    assert "require --stateful-tbptt" in capsys.readouterr().err


def test_train_rejects_nonzero_memory_with_scenario_conditioning_disabled(
    tmp_path, capsys,
) -> None:
    demos_path = tmp_path / "native.npz"
    demonstrations((0, 0, 1, 1)).save(demos_path)

    assert cli.main([
        "train",
        "--demos", str(demos_path),
        "--no-scenario-memory-conditioning",
        "--memory-size", "4",
        "--epochs", "1",
        "--device", "cpu",
    ]) == 2
    assert "requires --memory-size 0" in capsys.readouterr().err


def test_train_rejects_memory_width_that_disagrees_with_archive(
    tmp_path, capsys,
) -> None:
    demos_path = tmp_path / "conditioned.npz"
    conditioned = demonstrations((0, 0, 1, 1))
    conditioned.memory = np.zeros((*conditioned.actions.shape, 4), dtype=np.float32)
    conditioned.save(demos_path)

    assert cli.main([
        "train",
        "--demos", str(demos_path),
        "--memory-size", "3",
        "--epochs", "1",
        "--device", "cpu",
    ]) == 2
    assert "does not match the demonstration memory width" in capsys.readouterr().err


def test_stateful_train_can_continue_the_complete_recurrent_policy(
    monkeypatch, tmp_path, capsys,
) -> None:
    from stg_lab.policy import PolicyConfig
    import stg_lab.stateful_training as stateful_training

    demos_path = tmp_path / "native.npz"
    parent_path = tmp_path / "parent.pt"
    parent_path.write_bytes(b"parent")
    demonstrations((0, 0, 1, 1)).save(demos_path)
    parent = type("Parent", (), {})()
    parent.config = PolicyConfig(
        channels=6,
        feature_size=12,
        recurrent_size=16,
        memory_size=0,
        proficiency_size=0,
        inference_mode="stream",
    )
    called = {}

    monkeypatch.setattr(
        cli,
        "_load_checkpoint",
        lambda path, device: (
            parent,
            {"version": 3, "policy_config": asdict(parent.config), "history": []},
        ),
    )

    def fake_train(
        loaded, *, model, policy_config, training_config, output, training_data,
    ):
        called.update(
            model=model,
            policy_config=policy_config,
            training_data=training_data,
        )
        return model, [TrainingMetrics(1, 0.4, 0.5, 0.8, 0.1)]

    monkeypatch.setattr(
        stateful_training,
        "train_stateful_behavior_cloning",
        fake_train,
    )
    assert cli.main([
        "train",
        "--demos", str(demos_path),
        "--checkpoint", str(tmp_path / "continued.pt"),
        "--init-checkpoint", str(parent_path),
        "--stateful-tbptt",
        "--validation-episode-id", "1",
        "--feature-size", "12",
        "--recurrent-size", "16",
        "--memory-size", "0",
        "--no-scenario-memory-conditioning",
        "--no-proficiency-conditioning",
        "--epochs", "1",
        "--device", "cpu",
    ]) == 0

    assert called["model"] is parent
    assert called["training_data"]["initialization"] == "complete_policy_state"
    assert called["training_data"]["parent_checkpoint"] == str(parent_path)
    assert len(called["training_data"]["parent_checkpoint_sha256"]) == 64
    assert json.loads(capsys.readouterr().out)["epochs"] == 1


def test_full_checkpoint_continuation_rejects_permuted_context_metadata() -> None:
    parent = SimpleNamespace(
        scenario_vocabulary=("<unknown>", "attack:a#1"),
        previous_action_size=18,
        previous_action_offset=2,
    )

    cli._validate_initial_policy_context(
        parent,
        ("<unknown>", "attack:a#1"),
        18,
        2,
    )
    with np.testing.assert_raises_regex(cli.CLIError, "context metadata"):
        cli._validate_initial_policy_context(
            parent,
            ("<unknown>", "attack:b#1"),
            18,
            2,
        )


def test_merge_demos_renumbers_episode_groups(tmp_path, capsys) -> None:
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    output = tmp_path / "merged.npz"
    first_values = demonstrations((4, 4, 9))
    first_values.supervision_mask = np.asarray(
        ((True, False), (False, True), (True, False)),
        dtype=np.bool_,
    )
    first_values.save(first)
    demonstrations((12, 12)).save(second)

    assert cli.main([
        "merge-demos",
        str(first),
        str(second),
        "--output", str(output),
    ]) == 0

    merged = Demonstrations.load(output)
    np.testing.assert_array_equal(merged.episode_ids, (0, 0, 1, 2, 2))
    np.testing.assert_array_equal(
        merged.supervision_mask,
        (
            (True, False),
            (False, True),
            (True, False),
            (True, True),
            (True, True),
        ),
    )
    report = json.loads(capsys.readouterr().out)
    assert report["samples"] == 5
    assert report["episode_groups"] == 3
    assert report["action_supervision"] == "supervision_mask"
    assert report["supervised_labels"] == 7


def test_contextualize_demos_uses_strict_manifest_episode_order(
    tmp_path, capsys,
) -> None:
    source = tmp_path / "source.npz"
    source_manifest = tmp_path / "source.manifest.json"
    output = tmp_path / "context.npz"
    output_manifest = tmp_path / "context.manifest.json"
    demonstrations((7, 7, 9), steps=1).save(source)
    source_manifest.write_text(json.dumps({
        "accepted_episodes": [
            {
                "episode_kind": "attack",
                "scenario": "okuu:Lunatic",
                "attack": 3,
                "seed": 10,
                "profile": "teacher",
            },
            {
                "episode_kind": "stage",
                "scenario": "Stage 1@Normal",
                "attack": None,
                "seed": 11,
                "profile": "teacher",
            },
        ],
    }), encoding="utf-8")

    assert cli.main([
        "contextualize-demos",
        "--demos", str(source),
        "--source-manifest", str(source_manifest),
        "--output", str(output),
        "--manifest", str(output_manifest),
        "--previous-action-conditioning",
    ]) == 0

    conditioned = Demonstrations.load(output)
    assert conditioned.memory.shape == (3, 1, 21)
    np.testing.assert_array_equal(conditioned.memory[0, 0, :3], (0, 1, 0))
    np.testing.assert_array_equal(conditioned.memory[2, 0, :3], (0, 0, 1))
    report = json.loads(output_manifest.read_text(encoding="utf-8"))
    assert report["scenario_vocabulary"] == [
        "<unknown>",
        "attack:okuu:Lunatic#3",
        "stage:Stage 1@Normal",
    ]
    assert report["previous_action_offset"] == 3
    assert report["previous_action_size"] == 18
    assert json.loads(capsys.readouterr().out)["episode_groups"] == 2


def test_relabel_dagger_command_writes_corrective_archive_and_manifest(
    tmp_path, capsys,
) -> None:
    source = tmp_path / "teacher.npz"
    report_path = tmp_path / "dagger.json"
    output = tmp_path / "corrective.npz"
    manifest_path = tmp_path / "corrective.manifest.json"
    demonstrations((0,), steps=1).save(source)
    report_path.write_text(json.dumps({
        "run_kind": "live_luastg_native_dagger",
        "implementation_sha256": "b" * 64,
        "success": True,
        "passed": True,
        "episode_kind": "stage",
        "scenario": "Stage 1@Normal",
        "attack": None,
        "seed": 42,
        "terminated": True,
        "termination_reason": "stage_complete",
        "engine_termination_reason": "stage_complete",
        "decision_count": 1,
        "teacher_interventions": 0,
        "student_teacher_agreements": 0,
        "outcome_evidence": {"final_player": {"death": 0}},
        "decisions": [{
            "decision": 0,
            "teacher_action": {"discrete": 0},
            "student_action": {"discrete": 8},
            "executed_action": {"discrete": 8},
            "teacher_intervened": False,
            "student_teacher_agreement": False,
        }],
    }), encoding="utf-8")

    assert cli.main([
        "relabel-dagger",
        "--demos", str(source),
        "--dagger-report", str(report_path),
        "--output", str(output),
        "--manifest", str(manifest_path),
        "--interventions-only",
    ]) == 0

    np.testing.assert_array_equal(Demonstrations.load(output).actions, ((8,),))
    np.testing.assert_array_equal(
        Demonstrations.load(output).supervision_mask,
        ((False,),),
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["run_kind"] == "corrective_dagger_relabel"
    assert manifest["replaced_labels"] == 1
    assert manifest["action_supervision"]["mode"] == (
        "teacher_interventions_only"
    )
    assert json.loads(capsys.readouterr().out) == manifest


def test_relabel_dagger_command_reports_strict_validation_error(
    tmp_path, capsys,
) -> None:
    source = tmp_path / "teacher.npz"
    report_path = tmp_path / "failed.json"
    demonstrations((0,), steps=1).save(source)
    report_path.write_text(json.dumps({
        "run_kind": "live_luastg_native_dagger",
        "success": False,
        "passed": False,
    }), encoding="utf-8")

    assert cli.main([
        "relabel-dagger",
        "--demos", str(source),
        "--dagger-report", str(report_path),
        "--output", str(tmp_path / "output.npz"),
        "--manifest", str(tmp_path / "output.json"),
    ]) == 1
    assert "does not claim strict success" in capsys.readouterr().err


def test_evaluate_loads_model_and_reports_each_scenario(monkeypatch, tmp_path, capsys) -> None:
    import stg_lab.rollout as rollout

    checkpoint = tmp_path / "policy.pt"
    checkpoint.touch()
    model = object()
    monkeypatch.setattr(
        cli,
        "_load_checkpoint",
        lambda path, device: (model, {"version": 1, "policy_config": {}, "history": []}),
    )
    calls = []

    def fake_evaluate(selected_model, factory, seeds, **kwargs):
        calls.append((selected_model, factory.scenario, tuple(seeds), kwargs))
        return tuple(
            episode(factory.scenario, seed, survived=index % 2 == 0)
            for index, seed in enumerate(seeds)
        )

    monkeypatch.setattr(rollout, "evaluate_policy", fake_evaluate)

    assert cli.main([
        "evaluate",
        "--checkpoint",
        str(checkpoint),
        "--scenario",
        "all",
        "--episodes",
        "2",
        "--duration-frames",
        "12",
        "--no-shield",
    ]) == 0
    assert [call[1] for call in calls] == ["stage5_boss3", "stage5_boss4"]
    assert all(call[0] is model for call in calls)
    assert all(call[2] == (20260729, 20260730) for call in calls)
    assert all(call[3]["planner"] is None for call in calls)
    assert all(call[3]["shield"] is False for call in calls)
    payload = json.loads(capsys.readouterr().out)
    assert payload["overall"]["episode_count"] == 4
    assert payload["overall"]["survival_rate"] == 0.5
    assert payload["scenarios"]["stage5_boss3"]["unique_state_hashes"] == 2


def test_parallel_evaluate_uses_cpu_chunks_and_preserves_report_schema(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    from stg_lab import policy_benchmark
    from stg_lab.benchmark import summarize_episodes

    checkpoint = tmp_path / "policy.pt"
    checkpoint.touch()
    monkeypatch.setattr(
        cli,
        "_load_checkpoint",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("parallel evaluation must load inside workers")
        ),
    )
    calls = []

    def fake_benchmark(selected_checkpoint, factory, seeds, **kwargs):
        seeds = tuple(seeds)
        calls.append((selected_checkpoint, factory, seeds, kwargs))
        episodes = tuple(episode(factory.scenario, seed) for seed in seeds)
        return {
            "checkpoint_metadata": {
                "checkpoint": str(checkpoint),
                "sha256": "abc",
                "size_bytes": 0,
                "version": 1,
                "policy_config": {"inference_mode": "window"},
                "epochs": 2,
                "final_training_metrics": {"epoch": 2},
            },
            "elapsed_seconds": 0.25,
            "workers": 2,
            "scenarios": {factory.scenario: summarize_episodes(episodes)},
        }

    monkeypatch.setattr(policy_benchmark, "run_policy_benchmark", fake_benchmark)
    assert cli.main([
        "evaluate",
        "--checkpoint", str(checkpoint),
        "--scenario", "all",
        "--episodes", "3",
        "--workers", "2",
        "--device", "cpu",
        "--duration-frames", "12",
    ]) == 0

    assert [call[1].scenario for call in calls] == ["stage5_boss3", "stage5_boss4"]
    assert all(call[2] == (20260729, 20260730, 20260731) for call in calls)
    assert all(call[3]["workers"] == 2 for call in calls)
    assert all(call[3]["shield"] is False for call in calls)
    assert all(call[3]["simulation_config"].action_hold_frames == 1 for call in calls)
    payload = json.loads(capsys.readouterr().out)
    assert payload["run"]["device"] == "cpu"
    assert payload["run"]["workers"] == 2
    assert payload["run"]["requested_workers"] == 2
    assert payload["checkpoint_metadata"]["checkpoint"] == str(checkpoint)
    assert payload["teacher_metrics"] is False
    assert payload["planner_config"] is None
    assert payload["elapsed_seconds"] == 0.5
    assert payload["overall"]["episode_count"] == 6
    assert tuple(payload["scenarios"]) == ("stage5_boss3", "stage5_boss4")


def test_parallel_evaluate_rejects_non_cpu_and_teacher_metrics(
    tmp_path,
    capsys,
) -> None:
    checkpoint = tmp_path / "policy.pt"
    checkpoint.touch()
    assert cli.main([
        "evaluate", "--checkpoint", str(checkpoint),
        "--workers", "2", "--device", "mps",
    ]) == 2
    assert "requires --device cpu" in capsys.readouterr().err

    assert cli.main([
        "evaluate", "--checkpoint", str(checkpoint),
        "--workers", "2", "--teacher-metrics",
    ]) == 2
    assert "cannot use --teacher-metrics" in capsys.readouterr().err


def test_evaluate_metadata_only_does_not_run_rollout(monkeypatch, tmp_path, capsys) -> None:
    checkpoint = tmp_path / "policy.pt"
    checkpoint.touch()
    monkeypatch.setattr(
        cli,
        "_load_checkpoint",
        lambda path, device: (object(), {"version": 1, "policy_config": {}, "history": []}),
    )

    assert cli.main([
        "evaluate",
        "--checkpoint",
        str(checkpoint),
        "--metadata-only",
    ]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["version"] == 1
    assert payload["checkpoint"] == str(checkpoint)


def test_evaluate_reports_missing_checkpoint(capsys, tmp_path) -> None:
    missing = tmp_path / "missing.pt"
    assert cli.main(["evaluate", "--checkpoint", str(missing)]) == 2
    assert f"checkpoint does not exist: {missing}" in capsys.readouterr().err


def test_engine_test_runs_live_catalog_benchmark(monkeypatch, tmp_path, capsys) -> None:
    from stg_lab import engine, engine_benchmark

    connected = {}

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            connected["closed"] = True

    def fake_connect(host, port, *, timeout):
        connected.update(host=host, port=port, timeout=timeout)
        return FakeClient()

    benchmark_args = {}

    def fake_benchmark(client, **kwargs):
        benchmark_args.update(client=client, **kwargs)
        return {"schema_version": 1, "passed": True, "attacks": []}

    monkeypatch.setattr(engine.EngineClient, "connect", fake_connect)
    monkeypatch.setattr(engine_benchmark, "run_engine_benchmark", fake_benchmark)
    output = tmp_path / "engine.json"
    assert cli.main([
        "engine-test",
        "--host", "127.0.0.2",
        "--port", "25000",
        "--timeout", "4.5",
        "--seed", "90",
        "--frames-per-attack", "5",
        "--step-batch", "2",
        "--expected-attacks", "53",
        "--no-shoot",
        "--output", str(output),
    ]) == 0
    assert connected == {
        "host": "127.0.0.2",
        "port": 25000,
        "timeout": 4.5,
        "closed": True,
    }
    assert benchmark_args["seed"] == 90
    assert benchmark_args["frames_per_attack"] == 5
    assert benchmark_args["step_batch"] == 2
    assert benchmark_args["action"].shoot is False
    assert json.loads(output.read_text())["passed"] is True
    assert json.loads(capsys.readouterr().out)["passed"] is True


def test_engine_mpc_matrix_resolves_catalog_and_profiles(
    monkeypatch, tmp_path, capsys,
) -> None:
    from stg_lab import engine, engine_matrix

    connected = {}
    calls = {}

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            connected["closed"] = True

        def catalog(self):
            return {"catalog": {"attacks": [], "stages": []}}

    def fake_connect(host, port, *, timeout):
        connected.update(host=host, port=port, timeout=timeout)
        return FakeClient()

    target = engine_matrix.EngineEpisodeTarget("attack", "okuu:Lunatic", 3)

    def fake_select(response, **kwargs):
        calls["catalog"] = response
        calls["selection"] = kwargs
        return (target,)

    def fake_matrix(client, **kwargs):
        calls["client"] = client
        calls["matrix"] = kwargs
        return {"schema_version": 1, "passed": True, "overall": {}}

    monkeypatch.setattr(engine.EngineClient, "connect", fake_connect)
    monkeypatch.setattr(engine_matrix, "select_catalog_targets", fake_select)
    monkeypatch.setattr(engine_matrix, "run_engine_matrix", fake_matrix)
    output = tmp_path / "matrix.json"
    assert cli.main([
        "engine-mpc-matrix",
        "--host", "127.0.0.2",
        "--port", "25000",
        "--timeout", "4.5",
        "--scenario", "okuu:Lunatic",
        "--attack", "3",
        "--stage", "Stage 5@Lunatic",
        "--seed", "11",
        "--seed", "12",
        "--profile", "current",
        "--profile", "general",
        "--profile", "legacy-clearance-12-1",
        "--max-frames", "9000",
        "--output", str(output),
    ]) == 0
    assert connected == {
        "host": "127.0.0.2",
        "port": 25000,
        "timeout": 4.5,
        "closed": True,
    }
    assert calls["selection"] == {
        "scenarios": ("okuu:Lunatic",),
        "attacks": (3,),
        "stages": ("Stage 5@Lunatic",),
        "all_attacks": False,
        "all_stages": False,
    }
    assert calls["matrix"]["targets"] == (target,)
    assert calls["matrix"]["seeds"] == (11, 12)
    assert calls["matrix"]["profiles"] == (
        "current", "general", "legacy-clearance-12-1",
    )
    assert calls["matrix"]["config"].max_frames == 9000
    assert json.loads(output.read_text())["passed"] is True
    assert json.loads(capsys.readouterr().out)["passed"] is True


def test_engine_policy_matrix_resolves_targets_and_proficiencies(
    monkeypatch, tmp_path, capsys,
) -> None:
    from stg_lab import engine, engine_matrix

    connected = {}
    calls = {}

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            connected["closed"] = True

        def catalog(self):
            return {"catalog": {"attacks": [], "stages": []}}

    def fake_connect(host, port, *, timeout):
        connected.update(host=host, port=port, timeout=timeout)
        return FakeClient()

    target = engine_matrix.EngineEpisodeTarget("attack", "okuu:Lunatic", 3)

    def fake_select(response, **kwargs):
        calls["catalog"] = response
        calls["selection"] = kwargs
        return (target,)

    def fake_matrix(client, **kwargs):
        calls["client"] = client
        calls["matrix"] = kwargs
        return {"schema_version": 1, "passed": True, "overall": {}}

    checkpoint = tmp_path / "policy.pt"
    checkpoint.write_bytes(b"checkpoint")
    model = SimpleNamespace(
        config=SimpleNamespace(inference_mode="stream"),
        scenario_vocabulary=None,
    )
    monkeypatch.setattr(cli, "_load_checkpoint", lambda *_args: (model, {}))
    monkeypatch.setattr(cli, "_checkpoint_metadata", lambda *_args: {"version": 3})
    monkeypatch.setattr(engine.EngineClient, "connect", fake_connect)
    monkeypatch.setattr(engine_matrix, "select_catalog_targets", fake_select)
    monkeypatch.setattr(engine_matrix, "run_engine_policy_matrix", fake_matrix)
    output = tmp_path / "policy-matrix.json"

    assert cli.main([
        "engine-policy-matrix",
        "--host", "127.0.0.2",
        "--port", "25001",
        "--timeout", "5.5",
        "--checkpoint", str(checkpoint),
        "--scenario", "okuu:Lunatic",
        "--attack", "3",
        "--stage", "Stage 1@Normal",
        "--seed", "21",
        "--seed", "22",
        "--proficiency", "expert",
        "--proficiency", "intermediate",
        "--max-frames", "8400",
        "--vision-history", "1",
        "--no-visible-safety-shield",
        "--output", str(output),
    ]) == 0

    assert connected == {
        "host": "127.0.0.2",
        "port": 25001,
        "timeout": 5.5,
        "closed": True,
    }
    assert calls["selection"] == {
        "scenarios": ("okuu:Lunatic",),
        "attacks": (3,),
        "stages": ("Stage 1@Normal",),
        "all_attacks": False,
        "all_stages": False,
    }
    assert calls["matrix"]["targets"] == (target,)
    assert calls["matrix"]["seeds"] == (21, 22)
    assert calls["matrix"]["proficiencies"] == ("expert", "intermediate")
    assert calls["matrix"]["config"].max_frames == 8400
    assert calls["matrix"]["config"].visible_safety_shield is False
    assert calls["matrix"]["controller_metadata"]["kind"] == (
        "streaming_visual_policy"
    )
    assert json.loads(output.read_text())["passed"] is True
    assert json.loads(capsys.readouterr().out)["passed"] is True


def test_engine_mpc_play_selects_general_controller_profile(
    monkeypatch, tmp_path, capsys,
) -> None:
    from stg_lab import engine, engine_mpc_play

    connected = {}
    calls = {}

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            connected["closed"] = True

    def fake_connect(host, port, *, timeout):
        connected.update(host=host, port=port, timeout=timeout)
        return FakeClient()

    def fake_play(client, **kwargs):
        calls.update(client=client, **kwargs)
        return {
            "passed": True,
            "success": True,
            "termination_reason": "attack_complete",
        }

    monkeypatch.setattr(engine.EngineClient, "connect", fake_connect)
    monkeypatch.setattr(engine_mpc_play, "run_engine_mpc_play", fake_play)
    output = tmp_path / "general.json"

    assert cli.main([
        "engine-mpc-play",
        "--host", "127.0.0.2",
        "--port", "25000",
        "--timeout", "4.5",
        "--scenario", "orin:Lunatic",
        "--attack", "4",
        "--seed", "20260731",
        "--profile", "general",
        "--horizon-frames", "60",
        "--no-gap-prediction",
        "--output", str(output),
    ]) == 0

    controller_config = calls["controller"].config
    assert controller_config.minimum_direction_hold_frames == 9
    assert controller_config.clearance_reward_cap == 36.0
    assert controller_config.switch_margin_gain == 6.0
    assert controller_config.safe_margin_target == 20.0
    assert controller_config.region_safe_margin_target == 8.0
    assert controller_config.gap_prediction_enabled is False
    assert connected == {
        "host": "127.0.0.2",
        "port": 25000,
        "timeout": 4.5,
        "closed": True,
    }
    assert json.loads(output.read_text())["profile"] == "general"
    assert json.loads(capsys.readouterr().out)["passed"] is True


def test_engine_mpc_play_can_target_a_complete_stage(monkeypatch, tmp_path, capsys) -> None:
    from stg_lab import engine, engine_mpc_play

    calls = {}

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    monkeypatch.setattr(
        engine.EngineClient,
        "connect",
        lambda *_args, **_kwargs: FakeClient(),
    )

    def fake_play(client, **kwargs):
        calls.update(client=client, **kwargs)
        return {
            "passed": True,
            "success": True,
            "termination_reason": "stage_complete",
        }

    monkeypatch.setattr(engine_mpc_play, "run_engine_mpc_play", fake_play)
    output = tmp_path / "stage.json"

    assert cli.main([
        "engine-mpc-play",
        "--port", "25000",
        "--stage", "Stage 1@Normal",
        "--seed", "20260731",
        "--profile", "general",
        "--output", str(output),
    ]) == 0

    assert calls["scenario"] == "Stage 1@Normal"
    assert calls["attack"] is None
    assert calls["stage"] == "Stage 1@Normal"
    assert json.loads(capsys.readouterr().out)["termination_reason"] == "stage_complete"


def test_engine_play_loads_route_and_runs_strict_live_demo(monkeypatch, tmp_path, capsys) -> None:
    from stg_lab import engine, engine_play

    connected = {}
    selected_controller = object()
    calls = {}

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            connected["closed"] = True

    def fake_connect(host, port, *, timeout):
        connected.update(host=host, port=port, timeout=timeout)
        return FakeClient()

    def fake_load(route_artifact, memory_database, memory_id, *, exhaustion):
        calls.update(
            route_artifact=route_artifact,
            memory_database=memory_database,
            memory_id=memory_id,
            exhaustion=exhaustion,
        )
        return selected_controller, {"kind": "external_route"}

    def fake_play(client, **kwargs):
        calls.update(client=client, play=kwargs)
        return {
            "schema_version": 1,
            "success": True,
            "termination_reason": "attack_complete",
        }

    monkeypatch.setattr(engine.EngineClient, "connect", fake_connect)
    monkeypatch.setattr(engine_play, "load_route_controller", fake_load)
    monkeypatch.setattr(engine_play, "run_engine_play", fake_play)
    output = tmp_path / "play.json"
    assert cli.main([
        "engine-play",
        "--host", "127.0.0.2",
        "--port", "25001",
        "--timeout", "4.0",
        "--scenario", "okuu:Lunatic",
        "--attack", "3",
        "--seed", "91",
        "--route-artifact", "route.json",
        "--memory-database", "memory.sqlite",
        "--memory-id", "7",
        "--max-frames", "900",
        "--observation-delay", "5",
        "--shoot-risk-threshold", "0.2",
        "--output", str(output),
    ]) == 0
    assert connected == {
        "host": "127.0.0.2",
        "port": 25001,
        "timeout": 4.0,
        "closed": True,
    }
    assert calls["route_artifact"] == Path("route.json")
    assert calls["memory_database"] == Path("memory.sqlite")
    assert calls["memory_id"] == 7
    assert calls["play"]["controller"] is selected_controller
    assert calls["play"]["scenario"] == "okuu:Lunatic"
    assert calls["play"]["attack"] == 3
    assert calls["play"]["config"].decision_interval == 3
    assert calls["play"]["config"].vision.observation_delay == 5
    assert calls["play"]["config"].shoot_risk_threshold == 0.2
    assert json.loads(output.read_text())["success"] is True
    assert json.loads(capsys.readouterr().out)["termination_reason"] == "attack_complete"


def test_engine_play_route_requires_sqlite_memory(capsys) -> None:
    assert cli.main([
        "engine-play",
        "--route-artifact", "route.json",
    ]) == 2
    assert "requires --memory-database and --memory-id" in capsys.readouterr().err


def test_engine_play_stage_routes_checkpoint_to_full_stage_runner(
    monkeypatch, tmp_path, capsys,
) -> None:
    from stg_lab import engine, engine_play

    checkpoint = tmp_path / "policy.pt"
    checkpoint.write_bytes(b"policy")
    selected_controller = object()
    calls = {}

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    monkeypatch.setattr(
        engine.EngineClient,
        "connect",
        lambda *_args, **_kwargs: FakeClient(),
    )
    monkeypatch.setattr(
        cli,
        "_load_checkpoint",
        lambda *_args, **_kwargs: (object(), {"version": 3}),
    )
    controller_args = {}

    def fake_controller(*args, **kwargs):
        controller_args.update(args=args, kwargs=kwargs)
        return selected_controller

    monkeypatch.setattr(engine_play, "VisualPolicyController", fake_controller)

    def fake_play(client, **kwargs):
        calls.update(client=client, **kwargs)
        return {
            "schema_version": 1,
            "success": True,
            "termination_reason": "stage_complete",
        }

    monkeypatch.setattr(engine_play, "run_engine_play", fake_play)
    assert cli.main([
        "engine-play",
        "--stage", "Stage 1@Normal",
        "--checkpoint", str(checkpoint),
        "--seed", "92",
        "--proficiency", "intermediate",
        "--visible-safety-shield",
        "--visible-safety-horizon", "5",
    ]) == 0

    assert calls["controller"] is selected_controller
    assert controller_args["kwargs"]["proficiency"] == "intermediate"
    assert calls["config"].visible_safety_shield is True
    assert calls["config"].visible_safety_horizon == 5
    assert calls["scenario"] == "Stage 1@Normal"
    assert calls["attack"] is None
    assert calls["stage"] == "Stage 1@Normal"
    assert json.loads(capsys.readouterr().out)["termination_reason"] == "stage_complete"


def test_engine_train_uses_one_connection_and_explicit_seed_split(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    from stg_lab import engine, engine_play, engine_training

    connected = {}
    calls = {}
    template = object()
    factory = object()

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            connected["closed"] = True

    def fake_connect(host, port, *, timeout):
        connected.update(host=host, port=port, timeout=timeout)
        return FakeClient()

    def fake_load(route_artifact, memory_database, memory_id, *, exhaustion):
        calls.update(
            route_artifact=route_artifact,
            memory_database=memory_database,
            memory_id=memory_id,
            exhaustion=exhaustion,
        )
        return template, {"kind": "external_route"}

    def fake_train(client, **kwargs):
        calls.update(client=client, training=kwargs)
        return {
            "schema_version": 1,
            "run_kind": "live_luastg_closed_loop_candidate_training",
            "passed": True,
            "selected_strategy": {"candidate_id": "candidate-test"},
        }

    def fake_write(path, report):
        calls.update(strategy_path=path, strategy_report=report)
        path.write_text('{"kind":"live_luastg_candidate_strategy"}')
        return "a" * 64

    monkeypatch.setattr(engine.EngineClient, "connect", fake_connect)
    monkeypatch.setattr(engine_play, "load_route_controller", fake_load)
    monkeypatch.setattr(engine_training, "controller_factory_from_template", lambda value: (
        factory if value is template else None
    ))
    monkeypatch.setattr(engine_training, "run_engine_training", fake_train)
    monkeypatch.setattr(engine_training, "write_strategy_artifact", fake_write)
    output = tmp_path / "training.json"
    strategy = tmp_path / "strategy.json"
    assert cli.main([
        "engine-train",
        "--host", "127.0.0.3",
        "--port", "25002",
        "--timeout", "5",
        "--scenario", "okuu:Lunatic",
        "--attack", "3",
        "--route-artifact", "route.json",
        "--memory-database", "memory.sqlite",
        "--memory-id", "1",
        "--train-seeds", "10", "11",
        "--heldout-seeds", "90", "91",
        "--candidate-count", "4",
        "--search-seed", "123",
        "--max-frames", "600",
        "--strategy-output", str(strategy),
        "--output", str(output),
    ]) == 0
    assert connected == {
        "host": "127.0.0.3",
        "port": 25002,
        "timeout": 5.0,
        "closed": True,
    }
    assert calls["training"]["controller_factory"] is factory
    training_config = calls["training"]["training_config"]
    assert training_config.train_seeds == (10, 11)
    assert training_config.heldout_seeds == (90, 91)
    assert training_config.candidate_count == 4
    assert training_config.search_seed == 123
    assert calls["training"]["trace_directory"] == tmp_path / "training_traces"
    assert calls["strategy_path"] == strategy
    payload = json.loads(output.read_text())
    assert payload["passed"] is True
    assert payload["strategy_artifact"]["sha256"] == "a" * 64
    assert json.loads(capsys.readouterr().out) == payload


def test_engine_accept_compares_two_process_reports(monkeypatch, tmp_path, capsys) -> None:
    from stg_lab import engine_acceptance

    first, second = tmp_path / "first.json", tmp_path / "second.json"
    first.write_text('{"engine_session_id":"a"}')
    second.write_text('{"engine_session_id":"b"}')
    called = {}

    def fake_compare(left, right, *, expected_attacks):
        called.update(left=left, right=right, expected_attacks=expected_attacks)
        return {"schema_version": 1, "passed": True, "engine_verified": True}

    monkeypatch.setattr(engine_acceptance, "compare_engine_reports", fake_compare)
    output = tmp_path / "accepted.json"
    assert cli.main([
        "engine-accept",
        "--first", str(first),
        "--second", str(second),
        "--expected-attacks", "53",
        "--output", str(output),
    ]) == 0
    assert called == {
        "left": {"engine_session_id": "a"},
        "right": {"engine_session_id": "b"},
        "expected_attacks": 53,
    }
    assert json.loads(output.read_text())["engine_verified"] is True
    assert json.loads(capsys.readouterr().out)["passed"] is True


def test_accept_compiles_all_artifacts_and_uses_passed_for_exit_status(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    from stg_lab import acceptance

    called = {}

    def fake_compile(**kwargs):
        called.update(kwargs)
        return {"schema_version": 1, "passed": True, "checks": {}}

    monkeypatch.setattr(acceptance, "compile_acceptance_report", fake_compile)
    output = tmp_path / "reports" / "acceptance.json"
    arguments = [
        "accept",
        "--planner-artifact", "planner-boss3.json",
        "--planner-artifact", "planner-boss4.json",
        "--visual-artifact", "visual-boss3.json",
        "--visual-artifact", "visual-boss4.json",
        "--agreement-artifact", "agreement.json",
        "--memory-artifact", "memory.json",
        "--determinism-artifact", "determinism.json",
        "--output", str(output),
    ]

    assert cli.main(arguments) == 0
    assert called["planner_artifacts"] == [
        Path("planner-boss3.json"),
        Path("planner-boss4.json"),
    ]
    assert called["visual_artifacts"] == [
        Path("visual-boss3.json"),
        Path("visual-boss4.json"),
    ]
    assert called["agreement_artifact"] == Path("agreement.json")
    assert json.loads(output.read_text())["passed"] is True
    assert json.loads(capsys.readouterr().out)["passed"] is True

    monkeypatch.setattr(
        acceptance,
        "compile_acceptance_report",
        lambda **_kwargs: {"schema_version": 1, "passed": False, "checks": {}},
    )
    assert cli.main(arguments) == 1
    assert json.loads(output.read_text())["passed"] is False
