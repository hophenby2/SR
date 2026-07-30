from __future__ import annotations

from copy import deepcopy
import hashlib

from stg_lab.acceptance import compile_acceptance_report


SCENARIO_FRAMES = {"stage5_boss3": 600, "stage5_boss4": 700}
CHECKPOINT = "policy.pt"
CHECKPOINT_SHA256 = "a" * 64


def _episodes(scenario: str, survived: int = 100, *, seed_start: int = 3001):
    frames = SCENARIO_FRAMES[scenario]
    return [
        {
            "scenario": f"{scenario}:lunatic",
            "seed": seed,
            "survived": index < survived,
            "frames": frames if index < survived else frames - 1,
            "state_hash": f"{seed:032x}",
        }
        for index, seed in enumerate(range(seed_start, seed_start + 100))
    ]


def _artifact(scenario: str, *, kind: str, survived: int = 100):
    frames = SCENARIO_FRAMES[scenario]
    run = {
        "rollout_config": {"max_frames": frames, "decision_interval": 3},
        "vision_config": {
            "observation_delay": 5,
            "motion_estimation": "visible_displacement",
        },
        "simulation_config": {"action_hold_frames": 3, "reaction_frames": 0},
    }
    result = {
        "scenarios": {scenario: {"episodes": _episodes(scenario, survived)}},
    }
    if kind == "planner":
        result.update(run)
    else:
        result.update({
            "run": run,
            "shield": False,
            "authority_state_used": False,
            "online_visible_cue": True,
            "checkpoint_metadata": {
                "checkpoint": CHECKPOINT,
                "sha256": CHECKPOINT_SHA256,
            },
        })
    return result


def _agreement(boss3: float = 0.90, boss4: float = 0.90):
    boss3_samples = 400
    boss4_samples = 500
    return {
        "split": "heldout",
        "checkpoint": CHECKPOINT,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "heldout": {
            "stage5_boss3": {
                "samples": boss3_samples,
                "correct": round(boss3_samples * boss3),
                "agreement": boss3,
            },
            "stage5_boss4": {
                "samples": boss4_samples,
                "correct": round(boss4_samples * boss4),
                "agreement": boss4,
            },
        },
    }


def _memory(*, second_survived: bool = True, first_risk: float = 0.0, second_risk: float = 0.0):
    return {
        "memory": {
            "first": {
                "scenario": "stage5_boss4:lunatic",
                "seed": 3001,
                "survived": False,
                "frames": 349,
                "peak_risk": first_risk,
                "state_hash": "1" * 32,
            },
            "second": {
                "scenario": "stage5_boss4:lunatic",
                "seed": 3001,
                "survived": second_survived,
                "frames": 700 if second_survived else 400,
                "peak_risk": second_risk,
                "state_hash": "2" * 32,
            },
            # Deliberately ignored by the compiler.
            "passed": True,
        },
    }


def _determinism(*, boss4_hash: str = "b4"):
    def comparison(scenario: str, seed: int, *, mismatch: bool = False):
        frames = SCENARIO_FRAMES[scenario]
        offset = 1000 if scenario == "stage5_boss3" else 2000
        first_hashes = [f"{offset + frame:032x}" for frame in range(frames + 1)]
        second_hashes = list(first_hashes)
        if mismatch:
            second_hashes[-1] = "f" * 32

        def trajectory(values):
            digest = hashlib.blake2s(digest_size=16)
            digest.update(len(values).to_bytes(8, "big"))
            for value in values:
                digest.update(bytes.fromhex(value))
            return digest.hexdigest()

        first_trajectory = trajectory(first_hashes)
        second_trajectory = trajectory(second_hashes)
        return {
            "scenario": f"{scenario}:lunatic",
            "seed": seed,
            "hash_scope": "per_frame",
            "initial_frame_included": True,
            "actions": [{} for _ in range(frames)],
            "first_hashes": first_hashes,
            "second_hashes": second_hashes,
            "first_trajectory_hash": first_trajectory,
            "second_trajectory_hash": second_trajectory,
            "first": {
                "frames": frames,
                "actions_consumed": frames,
                "terminated": True,
                "outcome": "clear",
                "frame_hashes": first_hashes,
                "trajectory_hash": first_trajectory,
            },
            "second": {
                "frames": frames,
                "actions_consumed": frames,
                "terminated": True,
                "outcome": "clear",
                "frame_hashes": second_hashes,
                "trajectory_hash": second_trajectory,
            },
            "matched": not mismatch,
            "survived": True,
            "passed": not mismatch,
        }

    return {
        "comparisons": [
            comparison("stage5_boss3", 3001),
            comparison("stage5_boss4", 3001, mismatch=boss4_hash != "b4"),
        ],
    }


def _compile(**overrides):
    values = {
        "planner_artifacts": [
            _artifact("stage5_boss3", kind="planner"),
            _artifact("stage5_boss4", kind="planner"),
        ],
        "visual_artifacts": [
            _artifact("stage5_boss3", kind="visual"),
            _artifact("stage5_boss4", kind="visual"),
        ],
        "agreement_artifact": _agreement(),
        "memory_artifact": _memory(),
        "determinism_artifact": _determinism(),
    }
    values.update(overrides)
    return compile_acceptance_report(**values)


def test_strict_acceptance_report_passes_complete_canonical_evidence() -> None:
    report = _compile()
    assert report["passed"]
    assert report["checks"]["planner"]["scenarios"]["stage5_boss3"]["distinct_seeds"] == 100
    assert report["checks"]["visual"]["scenarios"]["stage5_boss4"]["expected_frames"] == 700
    assert report["checks"]["memory"]["second_survived"]
    assert report["checks"]["determinism"]["passed"]


def test_survival_is_enforced_per_scenario_instead_of_pooled() -> None:
    report = _compile(planner_artifacts=[
        _artifact("stage5_boss3", kind="planner", survived=100),
        _artifact("stage5_boss4", kind="planner", survived=90),
    ])
    assert not report["passed"]
    assert report["checks"]["planner"]["scenarios"]["stage5_boss3"]["passed"]
    assert not report["checks"]["planner"]["scenarios"]["stage5_boss4"]["passed"]


def test_duplicate_seeds_and_wrong_visual_timing_are_rejected() -> None:
    visual_boss4 = _artifact("stage5_boss4", kind="visual")
    visual_boss4["run"]["simulation_config"]["action_hold_frames"] = 1
    visual_boss4["scenarios"]["stage5_boss4"]["episodes"].append(
        deepcopy(visual_boss4["scenarios"]["stage5_boss4"]["episodes"][0])
    )
    report = _compile(visual_artifacts=[
        _artifact("stage5_boss3", kind="visual"),
        visual_boss4,
    ])
    scenario = report["checks"]["visual"]["scenarios"]["stage5_boss4"]
    assert not report["passed"]
    assert scenario["duplicate_seed_count"] == 1
    assert any("action_hold_frames" in error for error in scenario["errors"])


def test_agreement_is_required_for_each_heldout_scenario() -> None:
    report = _compile(agreement_artifact=_agreement(boss3=0.90, boss4=0.84))
    assert not report["passed"]
    assert report["checks"]["agreement"]["scenarios"]["stage5_boss3"]["passed"]
    assert not report["checks"]["agreement"]["scenarios"]["stage5_boss4"]["passed"]


def test_memory_zero_risk_does_not_make_two_failed_attempts_pass() -> None:
    report = _compile(memory_artifact=_memory(second_survived=False, first_risk=0.0, second_risk=0.0))
    memory = report["checks"]["memory"]
    assert not report["passed"]
    assert not memory["passed"]
    assert not memory["risks_meaningful"]
    assert memory["risk_improvement"] is None


def test_memory_meaningful_risk_reduction_is_a_valid_alternative() -> None:
    report = _compile(memory_artifact=_memory(
        second_survived=False,
        first_risk=10.0,
        second_risk=6.0,
    ))
    assert report["passed"]
    assert report["checks"]["memory"]["risk_improvement"] == 0.4


def test_determinism_requires_matching_trajectory_hashes() -> None:
    report = _compile(determinism_artifact=_determinism(boss4_hash="changed"))
    assert not report["passed"]
    boss4 = report["checks"]["determinism"]["scenarios"]["stage5_boss4"]
    assert not boss4["passed"]
    assert not boss4["comparisons"][0]["matched"]


def test_determinism_rejects_matching_but_incomplete_early_termination() -> None:
    determinism = _determinism()
    comparison = determinism["comparisons"][0]
    for run_name in ("first", "second"):
        comparison[run_name]["frames"] = 1
        comparison[run_name]["actions_consumed"] = 1
        comparison[run_name]["outcome"] = "hit"
    report = _compile(determinism_artifact=determinism)
    assert not report["passed"]
    errors = report["checks"]["determinism"]["scenarios"]["stage5_boss3"]["errors"]
    assert any("canonical survival window" in error for error in errors)


def test_visual_and_agreement_checkpoint_hashes_must_match() -> None:
    agreement = _agreement()
    agreement["checkpoint_sha256"] = "b" * 64
    report = _compile(agreement_artifact=agreement)
    assert not report["passed"]
    assert not report["checks"]["checkpoint_consistency"]["passed"]
