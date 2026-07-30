"""Compile independently produced artifacts into a strict acceptance report.

The compiler deliberately recomputes every decision from raw evidence.  Input
``passed`` fields are informational only and cannot make the final report pass.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

from .provenance import file_sha256, source_tree_sha256


Artifact = str | Path | Mapping[str, Any]
_STATE_HASH = re.compile(r"[0-9a-f]{32}")


@dataclass(frozen=True, slots=True)
class AcceptanceProfile:
    minimum_distinct_seeds: int = 100
    planner_survival: float = 0.95
    visual_survival: float = 0.90
    action_agreement: float = 0.85
    memory_risk_improvement: float = 0.30
    observation_delay: int = 5
    decision_interval: int = 3
    action_hold_frames: int = 3
    motor_delay_frames: int = 0
    boss3_frames: int = 600
    boss4_frames: int = 700

    @property
    def scenario_frames(self) -> dict[str, int]:
        return {
            "stage5_boss3": self.boss3_frames,
            "stage5_boss4": self.boss4_frames,
        }


DEFAULT_PROFILE = AcceptanceProfile()


@dataclass(frozen=True, slots=True)
class _LoadedArtifact:
    name: str
    digest: str
    value: Mapping[str, Any]

    @property
    def provenance(self) -> dict[str, str]:
        return {"source": self.name, "sha256": self.digest}


@dataclass(frozen=True, slots=True)
class _Episode:
    scenario: str
    seed: int
    survived: bool
    frames: int
    state_hash: str | None
    source: str


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _load_artifact(source: Artifact, *, label: str) -> _LoadedArtifact:
    if isinstance(source, Mapping):
        value = dict(source)
        raw = _canonical_json(value)
        return _LoadedArtifact(label, hashlib.sha256(raw).hexdigest(), value)
    path = Path(source)
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, Mapping):
        raise ValueError(f"artifact must contain a JSON object: {path}")
    return _LoadedArtifact(str(path), hashlib.sha256(raw).hexdigest(), value)


def _artifact_sequence(value: Artifact | Iterable[Artifact]) -> tuple[Artifact, ...]:
    if isinstance(value, (str, Path, Mapping)):
        return (value,)
    return tuple(value)


def _scenario_name(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    name = value.split(":", 1)[0]
    return name if name in DEFAULT_PROFILE.scenario_frames else None


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _integer(value: Any) -> int | None:
    number = _number(value)
    if number is None or not number.is_integer():
        return None
    return int(number)


def _nested(mapping: Mapping[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _file_reference_error(
    mapping: Mapping[str, Any],
    path_key: str,
    checksum_key: str,
    *,
    label: str,
) -> str | None:
    path_value = mapping.get(path_key)
    checksum = mapping.get(checksum_key)
    if not isinstance(path_value, str) or not Path(path_value).is_file():
        return f"{label} file does not exist"
    if not isinstance(checksum, str) or file_sha256(path_value) != checksum:
        return f"{label} checksum does not match the file"
    return None


def _checkpoint_metadata_errors(
    artifact: _LoadedArtifact,
    metadata: Mapping[str, Any],
) -> list[str]:
    errors: list[str] = []
    checkpoint_error = _file_reference_error(
        metadata,
        "checkpoint",
        "sha256",
        label=f"{artifact.name}: checkpoint",
    )
    if checkpoint_error:
        errors.append(checkpoint_error)
    if metadata.get("version") != 2:
        errors.append(f"{artifact.name}: checkpoint is not schema version 2")
    training_data = metadata.get("training_data")
    if not isinstance(training_data, Mapping):
        errors.append(f"{artifact.name}: checkpoint has no training-data provenance")
    else:
        training_error = _file_reference_error(
            training_data,
            "path",
            "sha256",
            label=f"{artifact.name}: checkpoint training dataset",
        )
        if training_error:
            errors.append(training_error)
    if not isinstance(metadata.get("training_config"), Mapping):
        errors.append(f"{artifact.name}: checkpoint has no training configuration")
    policy_config = metadata.get("policy_config")
    if not isinstance(policy_config, Mapping) or policy_config.get("inference_mode") != "window":
        errors.append(f"{artifact.name}: checkpoint does not use delayed-window inference")
    return errors


def _config_errors(
    artifact: _LoadedArtifact,
    scenario: str,
    *,
    kind: str,
    profile: AcceptanceProfile,
) -> list[str]:
    root = artifact.value if kind == "planner" else artifact.value.get("run")
    if not isinstance(root, Mapping):
        return [f"{artifact.name}: missing {kind} run configuration"]
    expected_frames = profile.scenario_frames[scenario]
    expected = (
        (("rollout_config", "max_frames"), expected_frames),
        (("rollout_config", "decision_interval"), profile.decision_interval),
        (("vision_config", "observation_delay"), profile.observation_delay),
        (("simulation_config", "action_hold_frames"), profile.action_hold_frames),
        (("simulation_config", "reaction_frames"), profile.motor_delay_frames),
    )
    errors = []
    if not artifact.name.startswith("<"):
        if artifact.value.get("implementation_sha256") != source_tree_sha256():
            errors.append(f"{artifact.name}: implementation fingerprint is stale or missing")
    for keys, wanted in expected:
        actual = _nested(root, *keys)
        if actual != wanted:
            errors.append(
                f"{artifact.name}: {scenario} {'.'.join(keys)}={actual!r}; expected {wanted}"
            )
    if _nested(root, "vision_config", "motion_estimation") != "visible_displacement":
        errors.append(f"{artifact.name}: visual motion must come from visible displacement")
    if kind == "visual" and artifact.value.get("shield") is not False:
        errors.append(f"{artifact.name}: deployable controller cannot use an authority shield")
    if kind == "visual" and artifact.value.get("authority_state_used") is not False:
        errors.append(f"{artifact.name}: deployable controller used authority state")
    if kind == "visual" and artifact.value.get("online_visible_cue") is not True:
        errors.append(f"{artifact.name}: controller memory was not selected from an online visible cue")
    if kind == "visual" and not artifact.name.startswith("<"):
        metadata = artifact.value.get("checkpoint_metadata")
        if not isinstance(metadata, Mapping):
            errors.append(f"{artifact.name}: missing checkpoint metadata")
        else:
            errors.extend(_checkpoint_metadata_errors(artifact, metadata))
            if metadata.get("role") != "system_checkpoint_reference":
                errors.append(f"{artifact.name}: external route has an invalid checkpoint role")
            if metadata.get("policy_actions_used") is not False:
                errors.append(f"{artifact.name}: route report unexpectedly used policy actions")
        controller_kind = artifact.value.get("controller_kind")
        if controller_kind not in {
            "external_route_memory",
            "external_route_library_memory",
        }:
            errors.append(f"{artifact.name}: unsupported deployable controller kind")
        route = artifact.value.get("route_memory")
        if not isinstance(route, Mapping):
            errors.append(f"{artifact.name}: missing external route provenance")
        else:
            for path_key, checksum_key, label in (
                ("artifact", "artifact_sha256", "route artifact"),
                ("database", "database_sha256", "episodic-memory database"),
            ):
                reference_error = _file_reference_error(
                    route,
                    path_key,
                    checksum_key,
                    label=f"{artifact.name}: {label}",
                )
                if reference_error:
                    errors.append(reference_error)
            route_artifact_path = route.get("artifact")
            if isinstance(route_artifact_path, str) and Path(route_artifact_path).is_file():
                try:
                    route_artifact_value = json.loads(
                        Path(route_artifact_path).read_text(encoding="utf-8")
                    )
                    if not isinstance(route_artifact_value, Mapping):
                        raise ValueError("route artifact is not an object")
                    generator_error = _file_reference_error(
                        route_artifact_value,
                        "generator",
                        "generator_sha256",
                        label=f"{artifact.name}: route-memory generator",
                    )
                    if generator_error:
                        errors.append(generator_error)
                    if _nested(route_artifact_value, "source", "implementation_sha256") != source_tree_sha256():
                        errors.append(f"{artifact.name}: route artifact implementation is stale")
                except (OSError, ValueError, json.JSONDecodeError) as error:
                    errors.append(
                        f"{artifact.name}: route artifact validation failed: {type(error).__name__}: {error}"
                    )
            if route.get("database_read_only") is not True:
                errors.append(f"{artifact.name}: route evaluation did not open memory read-only")
            route_config = route.get("config")
            if not isinstance(route_config, Mapping):
                errors.append(f"{artifact.name}: route controller configuration is missing")
            elif route_config.get("shield") is not False or route_config.get("route_origin") != "episode":
                errors.append(f"{artifact.name}: route controller used a non-deployable configuration")
    return errors


def _visual_route_episode_errors(
    artifact: _LoadedArtifact,
    raw_episodes: Sequence[Any],
    *,
    profile: AcceptanceProfile,
) -> list[str]:
    route = artifact.value.get("route_memory")
    if not isinstance(route, Mapping):
        return []
    kind = artifact.value.get("controller_kind")
    record_key = "triggers" if kind == "external_route_memory" else "selections"
    records = route.get(record_key)
    errors: list[str] = []
    if not isinstance(records, (list, tuple)) or len(records) != len(raw_episodes):
        return [
            f"{artifact.name}: route {record_key} do not cover every evaluated episode"
        ]
    expected_seeds = {
        value.get("seed")
        for value in raw_episodes
        if isinstance(value, Mapping) and _integer(value.get("seed")) is not None
    }
    observed_seeds: set[int] = set()
    selected_ids: list[int] = []
    for index, record in enumerate(records):
        prefix = f"{artifact.name}: route {record_key} record {index}"
        if not isinstance(record, Mapping):
            errors.append(prefix + " is not an object")
            continue
        seed = _integer(record.get("seed"))
        decision = _integer(record.get("decision"))
        source_frame = _integer(record.get("source_frame"))
        if seed is None:
            errors.append(prefix + " has no valid seed")
        else:
            observed_seeds.add(seed)
        if decision is None or decision <= 0 or source_frame is None:
            errors.append(prefix + " was not selected by an online visible cue")
        elif decision * profile.decision_interval - source_frame != profile.observation_delay:
            errors.append(prefix + " does not preserve the configured observation delay")
        if kind == "external_route_memory":
            if record.get("triggered") is not True:
                errors.append(prefix + " did not trigger its route")
        else:
            memory_id = _integer(record.get("memory_id"))
            if memory_id is None or memory_id <= 0:
                errors.append(prefix + " did not select a route memory")
            else:
                selected_ids.append(memory_id)
    if observed_seeds != expected_seeds:
        errors.append(f"{artifact.name}: route cue records use a different seed set")

    if kind == "external_route_memory":
        if route.get("untriggered_episodes") != 0:
            errors.append(f"{artifact.name}: one or more route episodes never triggered")
        memory_id = _integer(route.get("memory_id"))
        if memory_id is None or memory_id <= 0 or (_integer(route.get("route_actions")) or 0) <= 0:
            errors.append(f"{artifact.name}: single-route memory identity is invalid")
    elif kind == "external_route_library_memory":
        if route.get("unselected_episodes") != 0:
            errors.append(f"{artifact.name}: one or more route-library episodes were unselected")
        memory_ids = route.get("memory_ids")
        route_actions = route.get("route_actions")
        if (
            not isinstance(memory_ids, list)
            or not memory_ids
            or len(set(memory_ids)) != len(memory_ids)
            or not all((_integer(value) or 0) > 0 for value in memory_ids)
            or not isinstance(route_actions, list)
            or len(route_actions) != len(memory_ids)
            or not all((_integer(value) or 0) > 0 for value in route_actions)
        ):
            errors.append(f"{artifact.name}: route-library identities are invalid")
        elif not set(selected_ids).issubset(set(memory_ids)):
            errors.append(f"{artifact.name}: selected memory is outside the route library")
        claimed_counts = route.get("selection_counts")
        actual_counts = {
            str(memory_id): selected_ids.count(memory_id)
            for memory_id in sorted(set(selected_ids))
        }
        if claimed_counts != actual_counts:
            errors.append(f"{artifact.name}: route-library selection counts are inconsistent")
    return errors


def _episodes_from_artifact(
    artifact: _LoadedArtifact,
    *,
    kind: str,
    profile: AcceptanceProfile,
) -> tuple[list[_Episode], dict[str, list[str]]]:
    scenario_nodes = artifact.value.get("scenarios")
    if not isinstance(scenario_nodes, Mapping):
        return [], {
            scenario: [f"{artifact.name}: missing scenarios object"]
            for scenario in profile.scenario_frames
        }

    episodes: list[_Episode] = []
    errors = {scenario: [] for scenario in profile.scenario_frames}
    recognized = False
    for section_name, section in scenario_nodes.items():
        scenario = _scenario_name(section_name)
        if scenario is None:
            continue
        recognized = True
        errors[scenario].extend(_config_errors(
            artifact,
            scenario,
            kind=kind,
            profile=profile,
        ))
        raw_episodes = section.get("episodes") if isinstance(section, Mapping) else None
        if not isinstance(raw_episodes, list):
            errors[scenario].append(f"{artifact.name}: {scenario} missing episodes array")
            continue
        if kind == "visual":
            errors[scenario].extend(_visual_route_episode_errors(
                artifact,
                raw_episodes,
                profile=profile,
            ))
        for index, raw_episode in enumerate(raw_episodes):
            prefix = f"{artifact.name}: {scenario} episode {index}"
            if not isinstance(raw_episode, Mapping):
                errors[scenario].append(prefix + " is not an object")
                continue
            embedded = _scenario_name(raw_episode.get("scenario"))
            if embedded is not None and embedded != scenario:
                errors[scenario].append(prefix + " scenario does not match its section")
                continue
            seed = _integer(raw_episode.get("seed"))
            frames = _integer(raw_episode.get("frames"))
            survived = raw_episode.get("survived")
            if seed is None or frames is None or not isinstance(survived, bool):
                errors[scenario].append(prefix + " has invalid seed, frames, or survived fields")
                continue
            state_hash = raw_episode.get("state_hash")
            if not isinstance(state_hash, str) or _STATE_HASH.fullmatch(state_hash) is None:
                errors[scenario].append(prefix + " has no valid canonical state_hash")
                continue
            episodes.append(_Episode(
                scenario=scenario,
                seed=seed,
                survived=survived,
                frames=frames,
                state_hash=state_hash,
                source=artifact.name,
            ))
    if not recognized:
        for scenario in errors:
            errors[scenario].append(f"{artifact.name}: contains no canonical scenario")
    return episodes, errors


def _compile_episode_component(
    sources: Artifact | Iterable[Artifact],
    *,
    kind: str,
    profile: AcceptanceProfile,
) -> tuple[dict[str, Any], dict[str, set[int]], list[_LoadedArtifact], set[str]]:
    loaded = tuple(
        _load_artifact(source, label=f"<{kind}-{index}>")
        for index, source in enumerate(_artifact_sequence(sources))
    )
    records: list[_Episode] = []
    errors = {scenario: [] for scenario in profile.scenario_frames}
    checkpoints: set[str] = set()
    for artifact in loaded:
        artifact_records, artifact_errors = _episodes_from_artifact(
            artifact,
            kind=kind,
            profile=profile,
        )
        records.extend(artifact_records)
        for scenario, values in artifact_errors.items():
            errors[scenario].extend(values)
        if kind == "visual":
            checkpoint = _nested(artifact.value, "checkpoint_metadata", "checkpoint")
            if isinstance(checkpoint, str) and checkpoint:
                checksum = _nested(artifact.value, "checkpoint_metadata", "sha256")
                identity = Path(checkpoint).name
                if isinstance(checksum, str) and checksum:
                    identity += "@" + checksum
                checkpoints.add(identity)
            else:
                for scenario in errors:
                    errors[scenario].append(f"{artifact.name}: missing visual checkpoint identity")

    threshold = profile.planner_survival if kind == "planner" else profile.visual_survival
    scenarios: dict[str, Any] = {}
    seed_sets: dict[str, set[int]] = {}
    for scenario, expected_frames in profile.scenario_frames.items():
        values = [record for record in records if record.scenario == scenario]
        by_seed: dict[int, list[_Episode]] = {}
        for record in values:
            by_seed.setdefault(record.seed, []).append(record)
        duplicate_seeds = sorted(seed for seed, items in by_seed.items() if len(items) != 1)
        if duplicate_seeds:
            errors[scenario].append(
                f"duplicate evidence for {len(duplicate_seeds)} seed(s): {duplicate_seeds[:8]}"
            )
        unique_values = [items[0] for items in by_seed.values() if len(items) == 1]
        seed_sets[scenario] = set(by_seed)
        survived = sum(record.survived for record in unique_values)
        survival = survived / len(unique_values) if unique_values else 0.0
        invalid_frames = sorted(
            record.seed
            for record in unique_values
            if record.frames <= 0
            or record.frames > expected_frames
            or (record.survived and record.frames != expected_frames)
        )
        if invalid_frames:
            errors[scenario].append(
                f"invalid canonical frame count for {len(invalid_frames)} seed(s): {invalid_frames[:8]}"
            )
        if len(by_seed) < profile.minimum_distinct_seeds:
            errors[scenario].append(
                f"only {len(by_seed)} distinct seeds; expected at least {profile.minimum_distinct_seeds}"
            )
        if survival < threshold:
            errors[scenario].append(
                f"survival {survival:.6f} is below threshold {threshold:.6f}"
            )
        hashes = {record.state_hash for record in unique_values if record.state_hash}
        if len(hashes) != len(by_seed):
            errors[scenario].append(
                f"only {len(hashes)} unique terminal hashes for {len(by_seed)} distinct seeds"
            )
        scenarios[scenario] = {
            "passed": not errors[scenario],
            "expected_frames": expected_frames,
            "distinct_seeds": len(by_seed),
            "duplicate_seed_count": len(duplicate_seeds),
            "survived": survived,
            "survival_rate": survival,
            "threshold": threshold,
            "unique_terminal_state_hashes": len(hashes),
            "errors": errors[scenario],
        }
    result = {
        "passed": all(item["passed"] for item in scenarios.values()),
        "scenarios": scenarios,
        "artifacts": [artifact.provenance for artifact in loaded],
    }
    return result, seed_sets, list(loaded), checkpoints


def _compile_agreement(
    source: Artifact,
    *,
    profile: AcceptanceProfile,
) -> tuple[dict[str, Any], _LoadedArtifact, set[str]]:
    artifact = _load_artifact(source, label="<agreement>")
    value = artifact.value
    artifact_errors: list[str] = []
    if value.get("split") != "heldout":
        artifact_errors.append(f"{artifact.name}: agreement split must be heldout")
    replayed_counts: dict[str, tuple[int, int]] = {}
    if not artifact.name.startswith("<"):
        if value.get("implementation_sha256") != source_tree_sha256():
            artifact_errors.append(
                f"{artifact.name}: implementation fingerprint is stale or missing"
            )
        if value.get("schema_version") != 2 or value.get("run_kind") != "external_heldout_action_agreement":
            artifact_errors.append(f"{artifact.name}: agreement artifact has an invalid schema or run kind")
        generator_error = _file_reference_error(
            value,
            "generator",
            "generator_sha256",
            label=f"{artifact.name}: agreement generator",
        )
        if generator_error:
            artifact_errors.append(generator_error)
        for path_key, checksum_key, label in (
            ("checkpoint", "checkpoint_sha256", "checkpoint"),
            ("dataset", "dataset_sha256", "held-out dataset"),
        ):
            reference_error = _file_reference_error(
                value,
                path_key,
                checksum_key,
                label=f"{artifact.name}: {label}",
            )
            if reference_error:
                artifact_errors.append(reference_error)
        manifests = value.get("evidence_manifests")
        if not isinstance(manifests, Mapping) or not manifests:
            artifact_errors.append(f"{artifact.name}: agreement has no dataset manifests")
        else:
            for path_value, checksum in manifests.items():
                if not isinstance(path_value, str) or not Path(path_value).is_file():
                    artifact_errors.append(f"{artifact.name}: evidence manifest does not exist: {path_value}")
                elif not isinstance(checksum, str) or file_sha256(path_value) != checksum:
                    artifact_errors.append(f"{artifact.name}: evidence manifest checksum changed: {path_value}")
        binding = value.get("split_binding")
        if not isinstance(binding, Mapping) or binding.get("verified") is not True:
            artifact_errors.append(f"{artifact.name}: held-out split binding is missing or unverified")
        else:
            training_path = binding.get("checkpoint_training_dataset")
            training_sha = binding.get("checkpoint_training_dataset_sha256")
            if (
                not isinstance(training_path, str)
                or not Path(training_path).is_file()
                or not isinstance(training_sha, str)
                or file_sha256(training_path) != training_sha
            ):
                artifact_errors.append(f"{artifact.name}: checkpoint training dataset binding is invalid")
            if training_path == value.get("dataset") or training_sha == value.get("dataset_sha256"):
                artifact_errors.append(f"{artifact.name}: training and held-out datasets are not distinct")
            overlaps = binding.get("seed_overlap")
            if not isinstance(overlaps, Mapping) or any(overlaps.get(name) != [] for name in profile.scenario_frames):
                artifact_errors.append(f"{artifact.name}: training and held-out seed sets overlap")

        checkpoint_path = value.get("checkpoint")
        dataset_path = value.get("dataset")
        metadata = value.get("checkpoint_metadata")
        if isinstance(checkpoint_path, str) and isinstance(dataset_path, str) and isinstance(metadata, Mapping):
            try:
                import numpy as np

                from .rollout import teacher_action_agreement
                from .training import Demonstrations, load_checkpoint

                model, checkpoint_value = load_checkpoint(checkpoint_path, device="cpu")
                for key in ("version", "policy_config", "training_config", "training_data"):
                    if metadata.get(key) != checkpoint_value.get(key):
                        artifact_errors.append(
                            f"{artifact.name}: reported checkpoint {key} differs from the checkpoint file"
                        )
                training_data = checkpoint_value.get("training_data")
                if not isinstance(training_data, Mapping):
                    artifact_errors.append(f"{artifact.name}: checkpoint file has no training-data provenance")
                else:
                    training_error = _file_reference_error(
                        training_data,
                        "path",
                        "sha256",
                        label=f"{artifact.name}: checkpoint training dataset",
                    )
                    if training_error:
                        artifact_errors.append(training_error)
                    if isinstance(binding, Mapping) and (
                        training_data.get("path") != binding.get("checkpoint_training_dataset")
                        or training_data.get("sha256")
                        != binding.get("checkpoint_training_dataset_sha256")
                    ):
                        artifact_errors.append(
                            f"{artifact.name}: checkpoint and split binding identify different training data"
                        )
                demonstrations = Demonstrations.load(dataset_path)
                if demonstrations.episode_ids is None:
                    raise ValueError("held-out archive has no episode ids")
                heldout_nodes = value.get("heldout")
                if not isinstance(heldout_nodes, Mapping):
                    raise ValueError("agreement artifact has no held-out scenario records")
                required_manifests = {
                    "canonical_dataset_manifest.json",
                    "canonical_dataset_expanded_manifest.json",
                    "visible_dataset_v2_manifest.json",
                }
                manifest_paths = {
                    Path(path_value).name: Path(path_value)
                    for path_value in manifests
                    if isinstance(path_value, str)
                }
                if set(manifest_paths) != required_manifests:
                    raise ValueError("agreement does not bind the canonical three dataset manifests")
                visible_manifest = json.loads(
                    manifest_paths["visible_dataset_v2_manifest.json"].read_text(encoding="utf-8")
                )
                if (
                    not isinstance(visible_manifest, Mapping)
                    or visible_manifest.get("implementation_sha256") != source_tree_sha256()
                ):
                    raise ValueError("visible dataset manifest has a stale implementation fingerprint")
                outputs = visible_manifest.get("outputs")
                if not isinstance(outputs, list):
                    raise ValueError("visible dataset manifest has no outputs")
                heldout_output = next(
                    (
                        output
                        for output in outputs
                        if isinstance(output, Mapping)
                        and output.get("output") == dataset_path
                        and output.get("output_sha256") == value.get("dataset_sha256")
                    ),
                    None,
                )
                training_output = next(
                    (
                        output
                        for output in outputs
                        if isinstance(output, Mapping)
                        and isinstance(training_data, Mapping)
                        and output.get("output") == training_data.get("path")
                        and output.get("output_sha256") == training_data.get("sha256")
                    ),
                    None,
                )
                if not isinstance(heldout_output, Mapping) or not isinstance(training_output, Mapping):
                    raise ValueError("visible dataset manifest does not bind checkpoint and held-out archives")
                expected_episodes = {scenario: [] for scenario in profile.scenario_frames}
                expected_seeds = {scenario: [] for scenario in profile.scenario_frames}
                heldout_episode_records = heldout_output.get("episodes")
                training_episode_records = training_output.get("episodes")
                if not isinstance(heldout_episode_records, list) or not isinstance(
                    training_episode_records, list
                ):
                    raise ValueError("visible dataset manifest omits episode records")
                for episode in heldout_episode_records:
                    if not isinstance(episode, Mapping):
                        raise ValueError("visible dataset manifest has a malformed held-out episode")
                    scenario = _scenario_name(episode.get("scenario"))
                    episode_id = _integer(episode.get("episode_id"))
                    episode_seed = _integer(episode.get("seed"))
                    if scenario is None or episode_id is None or episode_seed is None:
                        raise ValueError("visible dataset manifest has an invalid held-out episode")
                    expected_episodes[scenario].append(episode_id)
                    expected_seeds[scenario].append(episode_seed)
                archive_ids = sorted(int(item) for item in np.unique(demonstrations.episode_ids))
                if sorted(value for values in expected_episodes.values() for value in values) != archive_ids:
                    raise ValueError("held-out manifest does not cover every archive episode exactly once")
                training_seeds = {scenario: set() for scenario in profile.scenario_frames}
                for episode in training_episode_records:
                    if not isinstance(episode, Mapping):
                        raise ValueError("visible dataset manifest has a malformed training episode")
                    scenario = _scenario_name(episode.get("scenario"))
                    episode_seed = _integer(episode.get("seed"))
                    if scenario is None or episode_seed is None:
                        raise ValueError("visible dataset manifest has an invalid training episode")
                    training_seeds[scenario].add(episode_seed)
                for scenario in profile.scenario_frames:
                    scenario_node = heldout_nodes.get(scenario)
                    if not isinstance(scenario_node, Mapping):
                        continue
                    if scenario_node.get("episode_ids") != expected_episodes[scenario]:
                        artifact_errors.append(
                            f"{artifact.name}: {scenario} episode split differs from the visible manifest"
                        )
                    if scenario_node.get("seeds") != expected_seeds[scenario]:
                        artifact_errors.append(
                            f"{artifact.name}: {scenario} seeds differ from the visible manifest"
                        )
                    if training_seeds[scenario] & set(expected_seeds[scenario]):
                        artifact_errors.append(
                            f"{artifact.name}: {scenario} manifest training/held-out seeds overlap"
                        )
                for scenario in profile.scenario_frames:
                    scenario_node = heldout_nodes.get(scenario)
                    if not isinstance(scenario_node, Mapping):
                        continue
                    episode_ids = scenario_node.get("episode_ids")
                    if not isinstance(episode_ids, list) or not episode_ids:
                        artifact_errors.append(
                            f"{artifact.name}: {scenario} has no held-out episode identities"
                        )
                        continue
                    mask = np.isin(demonstrations.episode_ids, episode_ids)
                    subset = Demonstrations(
                        global_frames=demonstrations.global_frames[mask],
                        local_frames=demonstrations.local_frames[mask],
                        actions=demonstrations.actions[mask],
                        risks=demonstrations.risks[mask],
                        memory=(None if demonstrations.memory is None else demonstrations.memory[mask]),
                        episode_ids=demonstrations.episode_ids[mask],
                        supervision_mask=(
                            None
                            if demonstrations.supervision_mask is None
                            else demonstrations.supervision_mask[mask]
                        ),
                    )
                    subset.validate()
                    samples = int(
                        len(subset.actions)
                        if subset.supervision_mask is None
                        else subset.supervision_mask.sum()
                    )
                    rate = teacher_action_agreement(model, subset, device="cpu")
                    replayed_counts[scenario] = (round(rate * samples), samples)
            except Exception as error:
                artifact_errors.append(
                    f"{artifact.name}: held-out agreement replay failed: {type(error).__name__}: {error}"
                )
        else:
            artifact_errors.append(f"{artifact.name}: checkpoint metadata is missing")
    heldout = value.get("heldout")
    if not isinstance(heldout, Mapping) and value.get("split") == "heldout":
        heldout = value.get("scenarios")
    if not isinstance(heldout, Mapping):
        heldout = {}

    scenarios: dict[str, Any] = {}
    for scenario in profile.scenario_frames:
        node = heldout.get(scenario)
        errors: list[str] = list(artifact_errors)
        agreement = None
        samples = 0
        correct = 0
        if not isinstance(node, Mapping):
            errors.append(f"{artifact.name}: missing held-out agreement for {scenario}")
        else:
            samples = _integer(node.get("samples")) or 0
            correct_value = _integer(node.get("correct"))
            correct = correct_value if correct_value is not None else -1
            if samples <= 0:
                errors.append(f"{artifact.name}: held-out sample count must be positive for {scenario}")
            if correct < 0 or correct > samples:
                errors.append(f"{artifact.name}: invalid held-out correct count for {scenario}")
            if samples > 0 and 0 <= correct <= samples:
                agreement = correct / samples
            claimed = _number(node.get("agreement"))
            if claimed is not None and agreement is not None and not math.isclose(
                claimed,
                agreement,
                rel_tol=1e-9,
                abs_tol=1e-12,
            ):
                errors.append(f"{artifact.name}: claimed agreement does not match correct/samples")
            if agreement is not None and agreement < profile.action_agreement:
                errors.append(
                    f"agreement {agreement:.6f} is below threshold {profile.action_agreement:.6f}"
                )
            replayed = replayed_counts.get(scenario)
            if not artifact.name.startswith("<") and replayed != (correct, samples):
                errors.append(
                    f"{artifact.name}: replayed held-out counts {replayed!r} differ from {(correct, samples)!r}"
                )
        scenarios[scenario] = {
            "passed": not errors,
            "agreement": agreement,
            "samples": samples,
            "correct": max(correct, 0),
            "replayed": (
                None
                if scenario not in replayed_counts
                else {
                    "correct": replayed_counts[scenario][0],
                    "samples": replayed_counts[scenario][1],
                }
            ),
            "threshold": profile.action_agreement,
            "errors": errors,
        }

    checkpoints: set[str] = set()
    checkpoint = value.get("checkpoint")
    if isinstance(checkpoint, str) and checkpoint:
        identity = Path(checkpoint).name
        checksum = value.get("checkpoint_sha256")
        if isinstance(checksum, str) and checksum:
            identity += "@" + checksum
        checkpoints.add(identity)
    result = {
        "passed": all(item["passed"] for item in scenarios.values()),
        "split": "heldout",
        "scenarios": scenarios,
        "artifact": artifact.provenance,
    }
    return result, artifact, checkpoints


def _compile_memory(source: Artifact, *, profile: AcceptanceProfile) -> tuple[dict[str, Any], _LoadedArtifact]:
    artifact = _load_artifact(source, label="<memory>")
    node = artifact.value.get("memory", artifact.value)
    errors: list[str] = []
    if not artifact.name.startswith("<"):
        if artifact.value.get("implementation_sha256") != source_tree_sha256():
            errors.append(f"{artifact.name}: implementation fingerprint is stale or missing")
        if artifact.value.get("schema_version") != 2:
            errors.append(f"{artifact.name}: memory benchmark has an invalid schema")
        generator_error = _file_reference_error(
            artifact.value,
            "generator",
            "generator_sha256",
            label=f"{artifact.name}: memory generator",
        )
        if generator_error:
            errors.append(generator_error)
        if artifact.value.get("online_visible_cue") is not True:
            errors.append(f"{artifact.name}: second attempt was not triggered by an online visible cue")
        if artifact.value.get("authority_state_used") is not False:
            errors.append(f"{artifact.name}: memory controller used authority state")
        if artifact.value.get("database_read_only") is not True:
            errors.append(f"{artifact.name}: memory benchmark did not preserve its database")
        for path_key, checksum_key, label in (
            ("database", "database_sha256", "episodic-memory database"),
            ("library_artifact", "library_artifact_sha256", "route-library artifact"),
        ):
            reference_error = _file_reference_error(
                artifact.value,
                path_key,
                checksum_key,
                label=f"{artifact.name}: {label}",
            )
            if reference_error:
                errors.append(reference_error)
        expected_config = (
            (("vision_config", "observation_delay"), profile.observation_delay),
            (("vision_config", "motion_estimation"), "visible_displacement"),
            (("simulation_config", "action_hold_frames"), profile.action_hold_frames),
            (("simulation_config", "reaction_frames"), profile.motor_delay_frames),
            (("rollout_config", "decision_interval"), profile.decision_interval),
            (("rollout_config", "max_frames"), profile.boss4_frames),
        )
        for keys, expected in expected_config:
            actual = _nested(artifact.value, *keys)
            if actual != expected:
                errors.append(
                    f"{artifact.name}: memory {'.'.join(keys)}={actual!r}; expected {expected!r}"
                )
        selection = artifact.value.get("selection")
        if not isinstance(selection, Mapping):
            errors.append(f"{artifact.name}: memory benchmark has no cue selection evidence")
        else:
            memory_id = _integer(selection.get("memory_id"))
            decision = _integer(selection.get("decision"))
            control_frame = _integer(selection.get("control_frame"))
            source_frame = _integer(selection.get("delayed_source_frame"))
            if memory_id is None or memory_id <= 0 or decision is None or decision <= 0:
                errors.append(f"{artifact.name}: memory benchmark selected no valid route")
            if (
                control_frame is None
                or source_frame is None
                or control_frame != decision * profile.decision_interval
                or control_frame - source_frame != profile.observation_delay
            ):
                errors.append(f"{artifact.name}: memory cue did not obey delayed local timing")
            database = artifact.value.get("database")
            library = artifact.value.get("library_artifact")
            if (
                memory_id is not None
                and isinstance(database, str)
                and Path(database).is_file()
                and isinstance(library, str)
                and Path(library).is_file()
            ):
                try:
                    from .memory import EpisodicMemory
                    from .route_memory import load_route_library_artifact

                    route_library = load_route_library_artifact(library)
                    if memory_id not in route_library.memory_ids:
                        errors.append(f"{artifact.name}: selected memory is outside its route library")
                    with EpisodicMemory(database, readonly=True) as store:
                        selected_memory = store.get(memory_id)
                    if selected_memory.scenario != route_library.scenario:
                        errors.append(f"{artifact.name}: selected memory scenario does not match library")
                except Exception as error:
                    errors.append(
                        f"{artifact.name}: external memory validation failed: {type(error).__name__}: {error}"
                    )
    if not isinstance(node, Mapping):
        node = {}
        errors.append(f"{artifact.name}: missing memory result")
    first = node.get("first")
    second = node.get("second")
    if not isinstance(first, Mapping) or not isinstance(second, Mapping):
        first = first if isinstance(first, Mapping) else {}
        second = second if isinstance(second, Mapping) else {}
        errors.append(f"{artifact.name}: memory result must contain first and second attempts")

    first_failed = first.get("survived") is False
    second_survived = second.get("survived") is True
    if not first_failed:
        errors.append("memory first attempt did not fail")
    first_scenario = _scenario_name(first.get("scenario"))
    second_scenario = _scenario_name(second.get("scenario"))
    if first_scenario != "stage5_boss4" or second_scenario != "stage5_boss4":
        errors.append("memory benchmark must contain two Stage 5 Boss #4 attempts")
    first_seed = _integer(first.get("seed"))
    second_seed = _integer(second.get("seed"))
    if first_seed is None or first_seed != second_seed:
        errors.append("memory attempts must use the same valid seed")
    for label, attempt in (("first", first), ("second", second)):
        state = attempt.get("state_hash")
        if not isinstance(state, str) or _STATE_HASH.fullmatch(state) is None:
            errors.append(f"memory {label} attempt has no canonical state hash")
    second_frames = _integer(second.get("frames"))
    if second_survived and second_frames != profile.boss4_frames:
        errors.append(
            f"surviving memory attempt ran {second_frames!r} frames; expected {profile.boss4_frames}"
        )

    first_risk = _number(first.get("peak_risk"))
    second_risk = _number(second.get("peak_risk"))
    risks_meaningful = (
        first_risk is not None
        and second_risk is not None
        and first_risk > 0.0
        and second_risk >= 0.0
    )
    risk_improvement = (
        (first_risk - second_risk) / first_risk
        if risks_meaningful and first_risk is not None and second_risk is not None
        else None
    )
    risk_branch = (
        risks_meaningful
        and risk_improvement is not None
        and risk_improvement >= profile.memory_risk_improvement
    )
    outcome_passed = first_failed and (second_survived or risk_branch)
    if first_failed and not second_survived and not risk_branch:
        errors.append(
            "memory second attempt neither survived nor demonstrated meaningful 30% risk improvement"
        )
    result = {
        "passed": outcome_passed and not errors,
        "first_failed": first_failed,
        "second_survived": second_survived,
        "first_peak_risk": first_risk,
        "second_peak_risk": second_risk,
        "risks_meaningful": risks_meaningful,
        "risk_improvement": risk_improvement,
        "risk_threshold": profile.memory_risk_improvement,
        "artifact": artifact.provenance,
        "errors": errors,
    }
    return result, artifact


def _comparison_hashes(comparison: Mapping[str, Any]) -> tuple[Any, Any, str | None]:
    for first_key, second_key in (
        ("first_hashes", "second_hashes"),
        ("run_a_hashes", "run_b_hashes"),
    ):
        first, second = comparison.get(first_key), comparison.get(second_key)
        if isinstance(first, list) and isinstance(second, list):
            return first, second, "per_frame"
    for first_key, second_key in (
        ("first_trajectory_hash", "second_trajectory_hash"),
        ("first_hash", "second_hash"),
        ("hash_a", "hash_b"),
    ):
        first, second = comparison.get(first_key), comparison.get(second_key)
        if isinstance(first, str) and isinstance(second, str) and first and second:
            implied = "trajectory" if "trajectory" in first_key else None
            return first, second, implied or comparison.get("hash_scope")
    return None, None, None


def _rolling_trajectory_hash(frame_hashes: Sequence[str]) -> str:
    digest = hashlib.blake2s(digest_size=16)
    digest.update(len(frame_hashes).to_bytes(8, "big"))
    for value in frame_hashes:
        digest.update(bytes.fromhex(value))
    return digest.hexdigest()


def _compile_determinism(
    source: Artifact,
    *,
    evaluation_seeds: Mapping[str, set[int]],
    profile: AcceptanceProfile,
) -> tuple[dict[str, Any], _LoadedArtifact]:
    artifact = _load_artifact(source, label="<determinism>")
    raw_comparisons = artifact.value.get("comparisons")
    errors = {scenario: [] for scenario in profile.scenario_frames}
    if not artifact.name.startswith("<"):
        if artifact.value.get("implementation_sha256") != source_tree_sha256():
            for scenario in errors:
                errors[scenario].append(
                    f"{artifact.name}: implementation fingerprint is stale or missing"
                )
        if (
            artifact.value.get("schema_version") != 2
            or artifact.value.get("artifact_kind") != "standalone_simulator_determinism"
            or artifact.value.get("hash_scope") != "per_frame"
        ):
            for scenario in errors:
                errors[scenario].append(f"{artifact.name}: invalid determinism schema or scope")
        generator_error = _file_reference_error(
            artifact.value,
            "generator",
            "generator_sha256",
            label=f"{artifact.name}: determinism generator",
        )
        if generator_error:
            for scenario in errors:
                errors[scenario].append(generator_error)
        contract = artifact.value.get("controller_contract")
        expected_contract = {
            "shield": False,
            "authority_state_used": False,
            "observation_delay": profile.observation_delay,
            "decision_interval": profile.decision_interval,
            "motion_estimation": "visible_displacement",
        }
        if contract != expected_contract:
            for scenario in errors:
                errors[scenario].append(
                    f"{artifact.name}: determinism actions lack the deployable controller contract"
                )
        sources = artifact.value.get("sources")
        if not isinstance(sources, Mapping):
            for scenario in errors:
                errors[scenario].append(f"{artifact.name}: missing determinism action sources")
        else:
            for scenario in profile.scenario_frames:
                node = sources.get(scenario)
                if not isinstance(node, Mapping):
                    errors[scenario].append(f"{artifact.name}: missing action source for {scenario}")
                    continue
                if node.get("kind") not in {
                    "external_route_memory",
                    "external_route_library_memory",
                }:
                    errors[scenario].append(f"{artifact.name}: unsupported action source for {scenario}")
                for path_key, checksum_key, label in (
                    ("artifact", "artifact_sha256", "route artifact"),
                    ("database", "database_sha256", "episodic-memory database"),
                ):
                    reference_error = _file_reference_error(
                        node,
                        path_key,
                        checksum_key,
                        label=f"{artifact.name}: {scenario} {label}",
                    )
                    if reference_error:
                        errors[scenario].append(reference_error)
    comparisons = {scenario: [] for scenario in profile.scenario_frames}
    if not isinstance(raw_comparisons, list):
        raw_comparisons = []
        for scenario in errors:
            errors[scenario].append(f"{artifact.name}: missing determinism comparisons array")

    seen: set[tuple[str, int]] = set()
    for index, raw in enumerate(raw_comparisons):
        if not isinstance(raw, Mapping):
            continue
        scenario = _scenario_name(raw.get("scenario"))
        seed = _integer(raw.get("seed"))
        if scenario is None or seed is None:
            continue
        key = (scenario, seed)
        first = raw.get("first_hashes")
        second = raw.get("second_hashes")
        scope = raw.get("hash_scope")
        item_errors: list[str] = []
        if key in seen:
            item_errors.append("duplicate determinism comparison")
        seen.add(key)
        expected_frames = profile.scenario_frames[scenario]
        expected_hash_count = expected_frames + 1
        if scope != "per_frame" or raw.get("initial_frame_included") is not True:
            item_errors.append("comparison does not include canonical per-frame reset evidence")
        valid_arrays = (
            isinstance(first, list)
            and isinstance(second, list)
            and len(first) == expected_hash_count
            and len(second) == expected_hash_count
            and all(isinstance(value, str) and _STATE_HASH.fullmatch(value) for value in first)
            and all(isinstance(value, str) and _STATE_HASH.fullmatch(value) for value in second)
        )
        if not valid_arrays:
            item_errors.append(
                f"comparison must contain two valid {expected_hash_count}-frame hash arrays"
            )
        elif len(set(first)) < 2 or len(set(second)) < 2:
            item_errors.append("comparison contains a static per-frame hash")
        matched = valid_arrays and first == second
        if not matched:
            item_errors.append("repeated runs produced different hashes")
        actions = raw.get("actions")
        if not isinstance(actions, list) or len(actions) != expected_frames:
            item_errors.append(f"comparison does not retain all {expected_frames} actions")
        for run_name in ("first", "second"):
            run = raw.get(run_name)
            hashes = first if run_name == "first" else second
            if not isinstance(run, Mapping):
                item_errors.append(f"comparison has no {run_name} run summary")
                continue
            if (
                run.get("frames") != expected_frames
                or run.get("actions_consumed") != expected_frames
                or run.get("terminated") is not True
                or run.get("outcome") != "clear"
            ):
                item_errors.append(f"{run_name} run did not complete the canonical survival window")
            if valid_arrays and run.get("frame_hashes") != hashes:
                item_errors.append(f"{run_name} run summary hashes differ from comparison evidence")
            if valid_arrays:
                calculated = _rolling_trajectory_hash(hashes)
                if (
                    run.get("trajectory_hash") != calculated
                    or raw.get(f"{run_name}_trajectory_hash") != calculated
                ):
                    item_errors.append(f"{run_name} trajectory hash is inconsistent")
        if raw.get("survived") is not True or raw.get("passed") is not True:
            item_errors.append("determinism comparison did not report a survived passing replay")
        if seed not in evaluation_seeds.get(scenario, set()):
            item_errors.append("determinism seed is outside the acceptance evaluation set")
        comparisons[scenario].append({
            "seed": seed,
            "hash_scope": "per_frame",
            "matched": matched,
            "frame_count": len(first) if isinstance(first, list) else 0,
            "errors": item_errors,
        })
        errors[scenario].extend(item_errors)

    scenarios: dict[str, Any] = {}
    for scenario in profile.scenario_frames:
        if not comparisons[scenario]:
            errors[scenario].append(f"no determinism evidence for {scenario}")
        scenarios[scenario] = {
            "passed": bool(comparisons[scenario]) and not errors[scenario],
            "comparison_count": len(comparisons[scenario]),
            "comparisons": comparisons[scenario],
            "errors": errors[scenario],
        }
    result = {
        "passed": all(item["passed"] for item in scenarios.values()),
        "scenarios": scenarios,
        "artifact": artifact.provenance,
    }
    return result, artifact


def compile_acceptance_report(
    *,
    planner_artifacts: Artifact | Iterable[Artifact],
    visual_artifacts: Artifact | Iterable[Artifact],
    agreement_artifact: Artifact,
    memory_artifact: Artifact,
    determinism_artifact: Artifact,
    profile: AcceptanceProfile = DEFAULT_PROFILE,
) -> dict[str, Any]:
    """Validate all standalone acceptance evidence and return a JSON-ready report."""

    planner, planner_seeds, planner_loaded, _ = _compile_episode_component(
        planner_artifacts,
        kind="planner",
        profile=profile,
    )
    visual, visual_seeds, visual_loaded, visual_checkpoints = _compile_episode_component(
        visual_artifacts,
        kind="visual",
        profile=profile,
    )
    agreement, agreement_loaded, agreement_checkpoints = _compile_agreement(
        agreement_artifact,
        profile=profile,
    )
    memory, memory_loaded = _compile_memory(memory_artifact, profile=profile)

    seed_errors: list[str] = []
    shared_seeds: dict[str, set[int]] = {}
    for scenario in profile.scenario_frames:
        planner_set = planner_seeds.get(scenario, set())
        visual_set = visual_seeds.get(scenario, set())
        shared_seeds[scenario] = planner_set & visual_set
        if planner_set != visual_set:
            seed_errors.append(
                f"{scenario} planner and visual evaluations use different seed sets"
            )
    seed_check = {
        "passed": not seed_errors,
        "scenarios": {
            scenario: {
                "planner_distinct_seeds": len(planner_seeds.get(scenario, set())),
                "visual_distinct_seeds": len(visual_seeds.get(scenario, set())),
                "shared_distinct_seeds": len(shared_seeds[scenario]),
                "identical": planner_seeds.get(scenario, set()) == visual_seeds.get(scenario, set()),
            }
            for scenario in profile.scenario_frames
        },
        "errors": seed_errors,
    }

    checkpoint_values = visual_checkpoints | agreement_checkpoints
    checkpoint_errors = []
    if not visual_checkpoints:
        checkpoint_errors.append("visual artifacts do not identify a checkpoint")
    if not agreement_checkpoints:
        checkpoint_errors.append("agreement artifact does not identify a checkpoint")
    if len(checkpoint_values) > 1:
        checkpoint_errors.append("visual and agreement artifacts use different checkpoints")
    checkpoint_check = {
        "passed": not checkpoint_errors,
        "checkpoints": sorted(checkpoint_values),
        "errors": checkpoint_errors,
    }

    determinism, determinism_loaded = _compile_determinism(
        determinism_artifact,
        evaluation_seeds=shared_seeds,
        profile=profile,
    )
    checks = {
        "planner": planner,
        "visual": visual,
        "agreement": agreement,
        "memory": memory,
        "determinism": determinism,
        "same_seed_sets": seed_check,
        "checkpoint_consistency": checkpoint_check,
    }
    issues: list[str] = []
    for name, check in checks.items():
        if not check["passed"]:
            issues.append(f"{name} acceptance check failed")

    artifacts = [
        *(artifact.provenance for artifact in planner_loaded),
        *(artifact.provenance for artifact in visual_loaded),
        agreement_loaded.provenance,
        memory_loaded.provenance,
        determinism_loaded.provenance,
    ]
    return {
        "schema_version": 1,
        "implementation_sha256": source_tree_sha256(),
        "passed": not issues,
        "profile": asdict(profile),
        "checks": checks,
        "issues": issues,
        "artifacts": artifacts,
    }


def write_acceptance_report(path: str | Path, report: Mapping[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = [
    "AcceptanceProfile",
    "DEFAULT_PROFILE",
    "compile_acceptance_report",
    "write_acceptance_report",
]
