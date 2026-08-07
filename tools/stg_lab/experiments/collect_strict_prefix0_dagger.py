"""Collect and verify strict prefix-zero Boss 3 DAgger episodes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Mapping


SCENARIO = "okuu:Lunatic"
ATTACK = 3
POLICY_SCENARIO_KEY = "attack:okuu:Lunatic#3"
RESERVED_SEEDS = frozenset((10306, 10307, 10308, 10309, 10310))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _finite_zero(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) == 0.0
    )


def _paths(
    artifacts: Path,
    *,
    seed: int,
    tag: str,
) -> dict[str, Path]:
    stem = f"humanlike-v54-seed{seed}-prefix0-margin30-{tag}"
    return {
        "dataset": artifacts / f"native-{stem}.npz",
        "manifest": artifacts / f"native-{stem}.manifest.json",
        "report": artifacts / f"dagger-{stem}-report.json",
        "log": artifacts / f"dagger-{stem}-cli.log",
    }


def _require_equal(actual: Any, expected: Any, field: str) -> None:
    if actual != expected:
        raise ValueError(f"{field} must be {expected!r}, got {actual!r}")


def _validate_triplet(
    paths: Mapping[str, Path],
    *,
    seed: int,
    checkpoint: Path,
    intervene_on_disagreement: bool,
) -> dict[str, Any]:
    dataset = paths["dataset"]
    manifest_path = paths["manifest"]
    report_path = paths["report"]
    for path in (dataset, manifest_path, report_path):
        if not path.is_file() or path.stat().st_size <= 0:
            raise ValueError(f"required nonempty artifact is missing: {path}")

    manifest = _read_object(manifest_path)
    report = _read_object(report_path)
    config = report.get("config")
    controller = report.get("controller")
    outcome = report.get("outcome_evidence")
    demonstrations = report.get("demonstrations")
    if not all(
        isinstance(value, Mapping)
        for value in (config, controller, outcome, demonstrations)
    ):
        raise ValueError("native report is missing strict structured evidence")

    _require_equal(report.get("passed"), True, "report.passed")
    _require_equal(report.get("terminated"), True, "report.terminated")
    _require_equal(
        report.get("termination_reason"),
        "attack_complete",
        "report.termination_reason",
    )
    _require_equal(report.get("scenario"), SCENARIO, "report.scenario")
    _require_equal(report.get("attack"), ATTACK, "report.attack")
    _require_equal(report.get("seed"), seed, "report.seed")
    _require_equal(report.get("continuous_fire"), True, "report.continuous_fire")
    _require_equal(report.get("shoot_command_rate"), 1.0, "shoot_command_rate")
    _require_equal(report.get("training_only"), True, "report.training_only")
    _require_equal(report.get("acceptance_claim"), False, "acceptance_claim")
    if not _finite_zero(outcome.get("boss_hp_last_observed")):
        raise ValueError("strict report must observe Boss HP equal to zero")
    final_player = outcome.get("final_player")
    if not isinstance(final_player, Mapping) or not _finite_zero(
        final_player.get("death")
    ):
        raise ValueError("strict report must contain finite numeric death=0")

    expected_config = {
        "continuous_fire": True,
        "decision_interval": 3,
        "failed_episode_labels_must_be_discarded": True,
        "intervene_on_disagreement": intervene_on_disagreement,
        "intervention_margin": 30.0,
        "intervention_regret": 6.0,
        "max_frames": 4200,
        "minimum_safety_margin_gain": None,
        "observation_delay": 5,
        "record_teacher_evaluations": True,
        "render": False,
        "spell_forced_off": True,
        "student_only_prefix_frames": 0,
        "supervision_mode": "corrective",
        "teacher_data_is_training_only": True,
        "teacher_probability": 0.0,
    }
    for name, expected in expected_config.items():
        _require_equal(config.get(name), expected, f"report.config.{name}")
    vision = config.get("vision")
    if not isinstance(vision, Mapping):
        raise ValueError("native report has no vision contract")
    for name, expected in {
        "channels": 6,
        "global_width": 48,
        "global_height": 56,
        "local_width": 40,
        "local_height": 40,
        "local_extent_x": 72.0,
        "local_extent_y": 72.0,
        "history": 1,
        "observation_delay": 5,
        "motion_estimation": "visible_displacement",
    }.items():
        _require_equal(vision.get(name), expected, f"report.config.vision.{name}")

    student = controller.get("student")
    teacher = controller.get("teacher")
    if not isinstance(student, Mapping) or not isinstance(teacher, Mapping):
        raise ValueError("native report has no student/teacher provenance")
    _require_equal(student.get("action_selection"), "joint", "student action mode")
    _require_equal(
        student.get("scenario_key"),
        POLICY_SCENARIO_KEY,
        "student scenario key",
    )
    _require_equal(
        student.get("checkpoint_sha256"),
        _sha256(checkpoint),
        "student checkpoint SHA-256",
    )
    teacher_config = teacher.get("config")
    if not isinstance(teacher_config, Mapping):
        raise ValueError("native report has no teacher configuration")
    _require_equal(
        teacher_config.get("region_dynamics_memory"),
        None,
        "teacher region dynamics memory",
    )
    _require_equal(
        teacher_config.get("horizon_frames"),
        60,
        "teacher horizon frames",
    )

    dataset_sha256 = _sha256(dataset)
    _require_equal(
        manifest.get("dataset_sha256"),
        dataset_sha256,
        "manifest dataset SHA-256",
    )
    _require_equal(
        demonstrations.get("dataset_sha256"),
        dataset_sha256,
        "report dataset SHA-256",
    )
    _require_equal(manifest.get("rejected_episodes"), [], "rejected episodes")
    accepted = manifest.get("accepted_episodes")
    if not isinstance(accepted, list) or len(accepted) != 1:
        raise ValueError("manifest must contain exactly one accepted episode")
    episode = accepted[0]
    if not isinstance(episode, Mapping):
        raise ValueError("accepted episode record must be an object")
    for name, expected in {
        "scenario": SCENARIO,
        "attack": ATTACK,
        "seed": seed,
        "strict_success": True,
    }.items():
        _require_equal(episode.get(name), expected, f"accepted episode {name}")
    decisions = report.get("decision_count")
    if isinstance(decisions, bool) or not isinstance(decisions, int) or decisions <= 0:
        raise ValueError("native report decision_count must be a positive integer")
    _require_equal(manifest.get("samples"), decisions, "manifest sample count")
    _require_equal(episode.get("decisions"), decisions, "accepted decision count")

    return {
        "seed": seed,
        "role": "training",
        "strict_prefix_zero": True,
        "strict_success": True,
        "dataset": str(dataset),
        "dataset_sha256": dataset_sha256,
        "report": str(report_path),
        "report_sha256": _sha256(report_path),
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "decisions": decisions,
        "teacher_interventions": report.get("teacher_interventions"),
        "shoot_command_rate": report.get("shoot_command_rate"),
        "external_region_memory": None,
    }


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            mode="w",
            encoding="utf-8",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(dict(payload), handle, indent=2, allow_nan=False)
            handle.write("\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _command(
    args: argparse.Namespace,
    paths: Mapping[str, Path],
    *,
    seed: int,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "stg_lab.cli",
        "engine-dagger-play",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--timeout",
        str(args.timeout),
        "--scenario",
        SCENARIO,
        "--attack",
        str(ATTACK),
        "--seed",
        str(seed),
        "--player",
        "reimu_player",
        "--checkpoint",
        str(args.checkpoint),
        "--device",
        args.device,
        "--policy-scenario-key",
        POLICY_SCENARIO_KEY,
        "--policy-action-selection",
        "joint",
        "--proficiency",
        "expert",
        "--profile",
        "humanlike",
        "--max-frames",
        "4200",
        "--horizon-frames",
        "60",
        "--observation-delay",
        "5",
        "--vision-history",
        "1",
        "--global-size",
        "48",
        "56",
        "--local-size",
        "40",
        "40",
        "--local-extent",
        "72",
        "72",
        "--teacher-probability",
        "0",
        "--intervention-margin",
        "30",
        "--intervention-regret",
        "6",
        "--student-only-prefix-frames",
        "0",
        (
            "--intervene-on-disagreement"
            if args.intervene_on_disagreement else
            "--no-intervene-on-disagreement"
        ),
        "--supervision-mode",
        "corrective",
        "--record-teacher-evaluations",
        "--no-render",
        "--save-demos",
        str(paths["dataset"]),
        "--demos-manifest",
        str(paths["manifest"]),
        "--output",
        str(paths["report"]),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect strict, memory-free, prefix-zero Boss 3 DAgger data."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=24816)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--first-seed", type=int, default=10311)
    parser.add_argument("--last-seed", type=int, default=10325)
    parser.add_argument("--tag", default="v83")
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "artifacts/policy-humanlike-highres-okuu3-v54-v37-onpolicy-"
            "gain025-minedit20-top1w2-kl20-ft60.pt"
        ),
    )
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument(
        "--intervene-on-disagreement",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "execute the teacher whenever its discrete movement differs from "
            "the student; enabled for the fixed v83 cohort"
        ),
    )
    parser.add_argument(
        "--inventory-output",
        type=Path,
        default=Path("artifacts/prefix0-expansion-v83-inventory.json"),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.first_seed > args.last_seed:
        raise ValueError("first seed must not exceed last seed")
    seeds = tuple(range(args.first_seed, args.last_seed + 1))
    overlap = RESERVED_SEEDS.intersection(seeds)
    if overlap:
        raise ValueError(f"collection range contains reserved seeds: {sorted(overlap)}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {args.checkpoint}")
    args.artifacts.mkdir(parents=True, exist_ok=True)

    inventory = []
    for seed in seeds:
        paths = _paths(args.artifacts, seed=seed, tag=args.tag)
        required = (paths["dataset"], paths["manifest"], paths["report"])
        if not args.overwrite and all(path.is_file() for path in required):
            record = _validate_triplet(
                paths,
                seed=seed,
                checkpoint=args.checkpoint,
                intervene_on_disagreement=args.intervene_on_disagreement,
            )
            inventory.append(record)
            print(json.dumps({"seed": seed, "status": "verified_existing"}))
            continue
        if not args.overwrite and any(path.exists() for path in required):
            raise FileExistsError(
                f"seed {seed} has a partial artifact set; inspect it or use --overwrite"
            )
        command = _command(args, paths, seed=seed)
        with paths["log"].open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
                text=True,
            )
        if completed.returncode != 0:
            raise RuntimeError(
                f"seed {seed} collection failed with exit {completed.returncode}; "
                f"see {paths['log']}"
            )
        record = _validate_triplet(
            paths,
            seed=seed,
            checkpoint=args.checkpoint,
            intervene_on_disagreement=args.intervene_on_disagreement,
        )
        inventory.append(record)
        print(json.dumps({
            "seed": seed,
            "status": "collected_and_verified",
            "decisions": record["decisions"],
            "teacher_interventions": record["teacher_interventions"],
        }))

    _write_json_atomic(args.inventory_output, {
        "schema_version": 1,
        "kind": "strict_prefix_zero_dagger_training_inventory",
        "training_only": True,
        "acceptance_claim": False,
        "scenario": SCENARIO,
        "attack": ATTACK,
        "seeds": list(seeds),
        "reserved_seeds_excluded": sorted(RESERVED_SEEDS),
        "intervene_on_disagreement": args.intervene_on_disagreement,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "source_inventory": inventory,
    })
    print(json.dumps({
        "inventory": str(args.inventory_output),
        "inventory_sha256": _sha256(args.inventory_output),
        "episodes": len(inventory),
    }))


if __name__ == "__main__":
    main()
