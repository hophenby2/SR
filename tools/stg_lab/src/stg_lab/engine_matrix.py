"""Strict cross-attack and full-stage evaluation on one live LuaSTG bridge."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import time
from typing import Any, Callable, Mapping, Sequence

from .engine import EngineClient, EngineProtocolError
from .engine_mpc import EngineMPC, MPCConfig
from .engine_mpc_play import EngineMPCPlayConfig, run_engine_mpc_play
from .engine_play import (
    EnginePlayConfig,
    VisualPolicyController,
    VisibleController,
    run_engine_play,
)
from .native_dataset import (
    NativeDemonstrationBuilder,
    NativeEpisodeIdentity,
)
from .provenance import file_sha256, source_tree_sha256
from .vision import VisionConfig


_CURRENT_PROFILE: dict[str, float | int] = {
    "danger_margin_target": 16.0,
    "safe_margin_target": 20.0,
    "region_safe_margin_target": 8.0,
    "minimum_direction_hold_frames": 12,
    "clearance_reward_cap": 48.0,
    "switch_margin_gain": 8.0,
}

_PROFILE_OVERRIDES: dict[str, Mapping[str, float | int]] = {
    # Conservative hysteresis used by the current Boss #3 controller.
    "current": _CURRENT_PROFILE,
    # General-purpose hysteresis recovered from the strict Orin #4 success.
    "general": {
        "danger_margin_target": 16.0,
        "safe_margin_target": 20.0,
        "region_safe_margin_target": 8.0,
        "minimum_direction_hold_frames": 9,
        "clearance_reward_cap": 36.0,
        "switch_margin_gain": 6.0,
    },
    "legacy-clearance-12-1": {
        "danger_margin_target": 12.0,
        "safe_margin_target": 12.0,
        "region_safe_margin_target": 1.0,
        "minimum_direction_hold_frames": 12,
        "clearance_reward_cap": 48.0,
        "switch_margin_gain": 8.0,
    },
    # Controlled bullet-group proficiency profiles. Movement scoring remains
    # identical to "current"; only group perception and gap-route capability
    # change, so matrix differences have an interpretable source.
    "bullet-group-novice": {
        **_CURRENT_PROFILE,
        "gap_direction_tolerance_degrees": 5.0,
        "gap_speed_relative_tolerance": 0.06,
        "gap_speed_absolute_tolerance": 0.15,
        "gap_minimum_group_size": 5,
        "gap_wavefront_depth": 14.0,
        "gap_maximum_lateral_spacing": 80.0,
        "gap_safety_margin": 18.0,
        "gap_minimum_usable_width": 10.0,
        "gap_sample_interval": 12,
        "gap_minimum_lifetime_frames": 24,
        "gap_entry_guard_frames": 12,
        "gap_path_minimum_margin": 8.0,
        "gap_group_coverage_fraction": 0.65,
        "gap_entry_candidate_limit": 2,
        "gap_detour_beam_width": 12,
    },
    "bullet-group-intermediate": {
        **_CURRENT_PROFILE,
        "gap_direction_tolerance_degrees": 8.0,
        "gap_speed_relative_tolerance": 0.12,
        "gap_speed_absolute_tolerance": 0.25,
        "gap_minimum_group_size": 4,
        "gap_wavefront_depth": 20.0,
        "gap_maximum_lateral_spacing": 96.0,
        "gap_safety_margin": 14.0,
        "gap_minimum_usable_width": 6.0,
        "gap_sample_interval": 9,
        "gap_minimum_lifetime_frames": 18,
        "gap_entry_guard_frames": 9,
        "gap_path_minimum_margin": 6.0,
        "gap_group_coverage_fraction": 0.55,
        "gap_entry_candidate_limit": 4,
        "gap_detour_beam_width": 24,
    },
    "bullet-group-expert": {
        **_CURRENT_PROFILE,
        "gap_direction_tolerance_degrees": 12.0,
        "gap_speed_relative_tolerance": 0.20,
        "gap_speed_absolute_tolerance": 0.35,
        "gap_minimum_group_size": 3,
        "gap_wavefront_depth": 24.0,
        "gap_maximum_lateral_spacing": 112.0,
        "gap_safety_margin": 10.0,
        "gap_minimum_usable_width": 4.0,
        "gap_sample_interval": 6,
        "gap_minimum_lifetime_frames": 12,
        "gap_entry_guard_frames": 6,
        "gap_path_minimum_margin": 4.0,
        "gap_group_coverage_fraction": 0.45,
        "gap_entry_candidate_limit": 8,
        "gap_detour_beam_width": 48,
    },
}
_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")
_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class EngineEpisodeTarget:
    episode_kind: str
    scenario: str
    attack: int | None = None
    label: str | None = None
    catalog_entry: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.episode_kind not in {"attack", "stage"}:
            raise ValueError("episode_kind must be attack or stage")
        if not self.scenario:
            raise ValueError("scenario must be nonempty")
        if self.episode_kind == "attack":
            if self.attack is None or self.attack <= 0:
                raise ValueError("attack targets require a positive attack ordinal")
        elif self.attack is not None:
            raise ValueError("stage targets cannot have an attack ordinal")

    @property
    def completion_reason(self) -> str:
        return "attack_complete" if self.episode_kind == "attack" else "stage_complete"

    @property
    def target_id(self) -> str:
        if self.episode_kind == "attack":
            return f"attack:{self.scenario}#{self.attack}"
        return f"stage:{self.scenario}"


@dataclass(frozen=True, slots=True)
class EngineMatrixConfig:
    max_frames: int = 7200
    horizon_frames: int = 60
    observation_delay: int = 5
    boundary_weight: float = 1.0
    boss_alignment_weight: float = 1.0
    stale_track_frames: int = 48
    shoot_minimum_margin: float = 12.0
    render: bool = False
    render_every: int = 1

    def __post_init__(self) -> None:
        if self.max_frames <= 0:
            raise ValueError("max_frames must be positive")
        if self.horizon_frames < 36:
            raise ValueError("horizon_frames must be at least 36")
        if self.observation_delay < 0:
            raise ValueError("observation_delay cannot be negative")
        if self.stale_track_frames <= 0:
            raise ValueError("stale_track_frames must be positive")
        finite = (
            self.boundary_weight,
            self.boss_alignment_weight,
            self.shoot_minimum_margin,
        )
        if not all(math.isfinite(value) for value in finite):
            raise ValueError("matrix numeric values must be finite")
        if self.boundary_weight < 0.0 or self.boss_alignment_weight < 0.0:
            raise ValueError("matrix weights cannot be negative")
        if (
            isinstance(self.render_every, bool)
            or not isinstance(self.render_every, int)
            or not 1 <= self.render_every <= 600
        ):
            raise ValueError("render_every must be an integer in [1, 600]")


def available_engine_profiles() -> tuple[str, ...]:
    return tuple(_PROFILE_OVERRIDES)


def apply_controller_profile(profile: str, config: MPCConfig) -> MPCConfig:
    """Apply one named live-controller profile without losing caller options."""

    try:
        overrides = _PROFILE_OVERRIDES[profile]
    except KeyError as error:
        raise ValueError(f"unknown engine matrix profile: {profile}") from error
    return replace(config, **overrides)


def controller_config_for_profile(
    profile: str,
    config: EngineMatrixConfig,
) -> MPCConfig:
    base = MPCConfig(
        horizon_frames=config.horizon_frames,
        observation_delay=config.observation_delay,
        boundary_weight=config.boundary_weight,
        boss_alignment_weight=config.boss_alignment_weight,
        stale_track_frames=config.stale_track_frames,
    )
    return apply_controller_profile(profile, base)


def _catalog_object(response: Mapping[str, Any]) -> Mapping[str, Any]:
    catalog = response.get("catalog")
    if not isinstance(catalog, Mapping):
        raise EngineProtocolError("engine response has no catalog object")
    return catalog


def select_catalog_targets(
    response: Mapping[str, Any],
    *,
    scenarios: Sequence[str] = (),
    attacks: Sequence[int] = (),
    stages: Sequence[str] = (),
    all_attacks: bool = False,
    all_stages: bool = False,
) -> tuple[EngineEpisodeTarget, ...]:
    """Resolve requested targets in catalog order and reject silent omissions."""

    catalog = _catalog_object(response)
    raw_attacks = catalog.get("attacks")
    raw_stages = catalog.get("stages")
    if not isinstance(raw_attacks, list):
        raise EngineProtocolError("engine catalog has no attacks array")
    if not isinstance(raw_stages, list):
        raw_stages = []
    scenario_filter = tuple(dict.fromkeys(str(value) for value in scenarios))
    attack_filter = tuple(dict.fromkeys(int(value) for value in attacks))
    stage_filter = tuple(dict.fromkeys(str(value) for value in stages))
    if any(value <= 0 for value in attack_filter):
        raise ValueError("attack filters must be positive")
    if attack_filter and not scenario_filter:
        raise ValueError("attack filters require at least one scenario")

    attack_entries: list[Mapping[str, Any]] = []
    seen_attacks: set[tuple[str, int]] = set()
    for index, value in enumerate(raw_attacks):
        if not isinstance(value, Mapping):
            raise EngineProtocolError(f"catalog attack {index} is not an object")
        scenario = value.get("scenario")
        attack = value.get("attack")
        if not isinstance(scenario, str) or not scenario:
            raise EngineProtocolError(f"catalog attack {index} has invalid scenario")
        if isinstance(attack, bool) or not isinstance(attack, int) or attack <= 0:
            raise EngineProtocolError(f"catalog attack {index} has invalid ordinal")
        key = scenario, attack
        if key in seen_attacks:
            raise EngineProtocolError(f"duplicate catalog attack {scenario} #{attack}")
        seen_attacks.add(key)
        attack_entries.append(value)

    stage_entries: list[Mapping[str, Any]] = []
    seen_stages: set[str] = set()
    for index, value in enumerate(raw_stages):
        if not isinstance(value, Mapping) or not isinstance(value.get("stage"), str):
            raise EngineProtocolError(f"catalog stage {index} is invalid")
        stage_name = str(value["stage"])
        if not stage_name or stage_name in seen_stages:
            raise EngineProtocolError(f"catalog stage {index} is empty or duplicated")
        seen_stages.add(stage_name)
        stage_entries.append(value)

    selected: list[EngineEpisodeTarget] = []
    if all_attacks or scenario_filter:
        known_scenarios = {str(value["scenario"]) for value in attack_entries}
        missing_scenarios = sorted(set(scenario_filter) - known_scenarios)
        if missing_scenarios:
            raise ValueError(f"scenarios are absent from the live catalog: {missing_scenarios}")
        for value in attack_entries:
            scenario = str(value["scenario"])
            attack = int(value["attack"])
            if scenario_filter and scenario not in scenario_filter:
                continue
            if attack_filter and attack not in attack_filter:
                continue
            selected.append(EngineEpisodeTarget(
                "attack",
                scenario,
                attack,
                value.get("label") if isinstance(value.get("label"), str) else None,
                dict(value),
            ))
        if attack_filter:
            for scenario in scenario_filter:
                present = {
                    target.attack for target in selected if target.scenario == scenario
                }
                missing = sorted(set(attack_filter) - present)
                if missing:
                    raise ValueError(
                        f"scenario {scenario} lacks requested attacks: {missing}",
                    )

    if all_stages or stage_filter:
        known_stages = {str(value["stage"]) for value in stage_entries}
        missing_stages = sorted(set(stage_filter) - known_stages)
        if missing_stages:
            raise ValueError(f"stages are absent from the live catalog: {missing_stages}")
        for value in stage_entries:
            stage_name = str(value["stage"])
            if stage_filter and stage_name not in stage_filter:
                continue
            selected.append(EngineEpisodeTarget(
                "stage",
                stage_name,
                None,
                value.get("label") if isinstance(value.get("label"), str) else None,
                dict(value),
            ))

    if not selected:
        raise ValueError("matrix selection contains no live catalog targets")
    return tuple(selected)


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n").encode("utf-8")


def _write_trace(path: Path, report: Mapping[str, Any]) -> str:
    raw = _canonical_bytes(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _direction(action: Mapping[str, Any]) -> tuple[int, int]:
    return int(action.get("move_x", 0)), int(action.get("move_y", 0))


def _smoothness(decisions: Any, frames: int) -> dict[str, Any]:
    if not isinstance(decisions, list):
        decisions = []
    actions: list[Mapping[str, Any]] = []
    holds: list[int] = []
    for value in decisions:
        if not isinstance(value, Mapping) or not isinstance(value.get("action"), Mapping):
            continue
        actions.append(value["action"])
        advanced = value.get("advanced_frames")
        holds.append(
            int(advanced)
            if not isinstance(advanced, bool) and isinstance(advanced, int) and advanced > 0
            else 3
        )
    changes = reversals = sharp_turns = aba = slow_changes = 0
    run_lengths: list[int] = []
    run_frames = 0
    previous_direction: tuple[int, int] | None = None
    for index, (action, hold) in enumerate(zip(actions, holds)):
        direction = _direction(action)
        if previous_direction is None or direction == previous_direction:
            run_frames += hold
        else:
            run_lengths.append(run_frames)
            run_frames = hold
            changes += 1
            if (
                direction != (0, 0)
                and previous_direction != (0, 0)
                and direction == (-previous_direction[0], -previous_direction[1])
            ):
                reversals += 1
            if (
                direction != (0, 0)
                and previous_direction != (0, 0)
                and direction[0] * previous_direction[0]
                + direction[1] * previous_direction[1] < 0
            ):
                sharp_turns += 1
        if index >= 2 and direction == _direction(actions[index - 2]) and (
            direction != _direction(actions[index - 1])
        ):
            aba += 1
        if index >= 1 and direction == _direction(actions[index - 1]) and (
            bool(action.get("slow")) != bool(actions[index - 1].get("slow"))
        ):
            slow_changes += 1
        previous_direction = direction
    if actions:
        run_lengths.append(run_frames)
    transition_count = max(0, len(actions) - 1)
    return {
        "decision_count": len(actions),
        "transition_count": transition_count,
        "direction_changes": changes,
        "direction_change_rate": changes / transition_count if transition_count else 0.0,
        "direction_changes_per_1000_frames": (
            1000.0 * changes / frames if frames > 0 else 0.0
        ),
        "exact_reversals": reversals,
        "sharp_turns_over_90_degrees": sharp_turns,
        "aba_changes": aba,
        "slow_mode_changes_without_direction_change": slow_changes,
        "direction_hold_frames": {
            "minimum": min(run_lengths) if run_lengths else None,
            "median": statistics.median(run_lengths) if run_lengths else None,
            "mean": statistics.fmean(run_lengths) if run_lengths else None,
            "maximum": max(run_lengths) if run_lengths else None,
        },
    }


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _verify_policy_checkpoint_metadata(metadata: Mapping[str, Any]) -> None:
    if metadata.get("kind") != "streaming_visual_policy":
        raise ValueError("policy matrix requires streaming_visual_policy metadata")
    checkpoint_value = metadata.get("checkpoint")
    if not isinstance(checkpoint_value, (str, Path)):
        raise ValueError("policy matrix metadata has no checkpoint path")
    checkpoint_path = Path(checkpoint_value)
    if not checkpoint_path.is_file():
        raise ValueError("policy matrix checkpoint does not exist")
    declared_sha256 = metadata.get("checkpoint_sha256")
    if (
        not isinstance(declared_sha256, str)
        or _SHA256.fullmatch(declared_sha256) is None
        or file_sha256(checkpoint_path) != declared_sha256
    ):
        raise ValueError("policy matrix checkpoint SHA-256 evidence is invalid")
    checkpoint_metadata = metadata.get("checkpoint_metadata")
    policy_config = (
        checkpoint_metadata.get("policy_config")
        if isinstance(checkpoint_metadata, Mapping) else
        None
    )
    if (
        not isinstance(policy_config, Mapping)
        or policy_config.get("inference_mode") != "stream"
    ):
        raise ValueError("policy matrix checkpoint metadata is not streaming")


def _episode_evidence(
    report: Mapping[str, Any],
    *,
    target: EngineEpisodeTarget,
    profile: str,
    trace_path: Path | None,
) -> dict[str, Any]:
    terminated = report.get("terminated") is True
    engine_reason = report.get("engine_termination_reason") if terminated else None
    reason_matches = terminated and engine_reason == target.completion_reason
    outcome = report.get("outcome_evidence")
    outcome = outcome if isinstance(outcome, Mapping) else {}
    final_player = outcome.get("final_player")
    final_player = final_player if isinstance(final_player, Mapping) else {}
    death_value = _finite(final_player.get("death"))
    died = engine_reason == "player_hit" or (
        death_value is not None and death_value > 0.0
    )
    zero_death_evidence = death_value == 0.0
    frames_raw = report.get("frames")
    frames = (
        int(frames_raw)
        if not isinstance(frames_raw, bool) and isinstance(frames_raw, int)
        and frames_raw >= 0 else 0
    )
    strict_success = reason_matches and zero_death_evidence and frames > 0
    hp_initial = _finite(outcome.get("boss_hp_initial"))
    hp_last = _finite(outcome.get("boss_hp_last_observed"))
    hp_minimum = _finite(outcome.get("boss_hp_minimum_observed"))
    boss_defeated = bool(
        strict_success
        and hp_minimum is not None
        and hp_minimum <= 0.0
    )
    decisions = report.get("decisions")
    if not isinstance(decisions, list):
        decisions = report.get("action_steps")
    smoothness = _smoothness(decisions, frames)
    raw_sha256 = hashlib.sha256(_canonical_bytes(report)).hexdigest()
    if trace_path is not None:
        raw_sha256 = _write_trace(trace_path, report)
    visible_safety_interventions = report.get("visible_safety_interventions")
    zero_visible_safety_interventions = (
        not isinstance(visible_safety_interventions, bool)
        and isinstance(visible_safety_interventions, int)
        and visible_safety_interventions == 0
    )
    pure_policy = report.get("pure_policy") is True
    pure_policy_eligible = report.get("pure_policy_validation_eligible") is True
    pure_policy_success = (
        strict_success
        and pure_policy
        and pure_policy_eligible
        and zero_visible_safety_interventions
        and report.get("pure_policy_success") is True
    )
    return {
        "target_id": target.target_id,
        "episode_kind": target.episode_kind,
        "scenario": target.scenario,
        "attack": target.attack,
        "profile": profile,
        "seed": report.get("seed"),
        "strict_success": strict_success,
        "pure_policy": pure_policy,
        "pure_policy_success": pure_policy_success,
        "pure_policy_validation_eligible": pure_policy_eligible,
        "visible_safety_interventions": visible_safety_interventions,
        "zero_visible_safety_interventions": zero_visible_safety_interventions,
        "runner_success": report.get("success") is True,
        "runner_success_matches_strict_metric": (
            (report.get("success") is True) == strict_success
        ),
        "expected_completion_reason": target.completion_reason,
        "terminated": terminated,
        "termination_reason": engine_reason if terminated else "max_frames",
        "died": died,
        "frames": frames,
        "engine_advanced_frames": report.get("engine_advanced_frames"),
        "boss_hp_initial": hp_initial,
        "boss_hp_last_observed": hp_last,
        "boss_hp_minimum_observed": hp_minimum,
        "boss_hp_reduction_observed": (
            None if hp_initial is None or hp_last is None else hp_initial - hp_last
        ),
        "boss_defeated_with_zero_hp_evidence": boss_defeated,
        "completion_class": (
            "boss_defeated" if boss_defeated else
            target.completion_reason if strict_success else
            "failed"
        ),
        "player_path_distance": _finite(outcome.get("player_path_distance")),
        "smoothness": smoothness,
        "full_trace_sha256": raw_sha256,
        "trace_path": None if trace_path is None else str(trace_path),
        "engine": report.get("engine"),
        "outcome_evidence": outcome,
    }


def _engine_identity(report: Mapping[str, Any]) -> tuple[Any, Any, Any]:
    engine = report.get("engine")
    if not isinstance(engine, Mapping):
        return None, None, None
    runtime = engine.get("runtime_identity")
    process_id = runtime.get("process_id") if isinstance(runtime, Mapping) else None
    return engine.get("session_id"), engine.get("process_nonce"), process_id


def _validate_engine_identity(identity: tuple[Any, Any, Any]) -> None:
    session_id, process_nonce, process_id = identity
    if not isinstance(session_id, str) or not session_id:
        raise EngineProtocolError("matrix episode has no engine session id")
    if not isinstance(process_nonce, str) or not process_nonce:
        raise EngineProtocolError("matrix episode has no engine process nonce")
    # Win32 bridges expose the OS PID. Portable/headless builds may not, so the
    # connection-bound session and process nonce are the cross-platform floor.
    if process_id is not None and (
        isinstance(process_id, bool)
        or not isinstance(process_id, int)
        or process_id <= 0
    ):
        raise EngineProtocolError("matrix episode has an invalid engine process id")


def _summary(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    attempts = len(episodes)
    successes = sum(value.get("strict_success") is True for value in episodes)
    pure_policy_successes = sum(
        value.get("pure_policy_success") is True for value in episodes
    )
    deaths = sum(value.get("died") is True for value in episodes)
    frames = sum(int(value.get("frames", 0)) for value in episodes)
    frame_values = [int(value.get("frames", 0)) for value in episodes]
    transitions = sum(
        int(value["smoothness"]["transition_count"])
        for value in episodes
        if isinstance(value.get("smoothness"), Mapping)
    )
    changes = sum(
        int(value["smoothness"]["direction_changes"])
        for value in episodes
        if isinstance(value.get("smoothness"), Mapping)
    )
    reversals = sum(
        int(value["smoothness"]["exact_reversals"])
        for value in episodes
        if isinstance(value.get("smoothness"), Mapping)
    )
    aba = sum(
        int(value["smoothness"]["aba_changes"])
        for value in episodes
        if isinstance(value.get("smoothness"), Mapping)
    )
    reasons = Counter(str(value.get("termination_reason")) for value in episodes)
    completion_classes = Counter(str(value.get("completion_class")) for value in episodes)
    hp_reductions = [
        float(value["boss_hp_reduction_observed"])
        for value in episodes
        if _finite(value.get("boss_hp_reduction_observed")) is not None
    ]
    return {
        "attempts": attempts,
        "strict_successes": successes,
        "strict_success_rate": successes / attempts if attempts else 0.0,
        "pure_policy_successes": pure_policy_successes,
        "pure_policy_success_rate": (
            pure_policy_successes / attempts if attempts else 0.0
        ),
        "deaths": deaths,
        "death_rate": deaths / attempts if attempts else 0.0,
        "boss_defeats_with_zero_hp_evidence": sum(
            value.get("boss_defeated_with_zero_hp_evidence") is True
            for value in episodes
        ),
        "completion_classes": dict(sorted(completion_classes.items())),
        "total_frames": frames,
        "frames": {
            "minimum": min(frame_values) if frame_values else None,
            "median": statistics.median(frame_values) if frame_values else None,
            "mean": statistics.fmean(frame_values) if frame_values else None,
            "maximum": max(frame_values) if frame_values else None,
        },
        "observed_boss_hp_reduction": {
            "sample_count": len(hp_reductions),
            "minimum": min(hp_reductions) if hp_reductions else None,
            "median": statistics.median(hp_reductions) if hp_reductions else None,
            "mean": statistics.fmean(hp_reductions) if hp_reductions else None,
            "maximum": max(hp_reductions) if hp_reductions else None,
        },
        "termination_reasons": dict(sorted(reasons.items())),
        "motion": {
            "transition_count": transitions,
            "direction_changes": changes,
            "direction_change_rate": changes / transitions if transitions else 0.0,
            "direction_changes_per_1000_frames": (
                1000.0 * changes / frames if frames else 0.0
            ),
            "exact_reversals": reversals,
            "aba_changes": aba,
        },
    }


EpisodeRunner = Callable[..., Mapping[str, Any]]
ControllerFactory = Callable[[str, EngineMatrixConfig], EngineMPC]
PolicyControllerFactory = Callable[
    [EngineEpisodeTarget, str, int],
    VisibleController,
]


def run_engine_matrix(
    client: EngineClient,
    *,
    targets: Sequence[EngineEpisodeTarget],
    seeds: Sequence[int],
    profiles: Sequence[str] = ("current",),
    player: str = "reimu_player",
    config: EngineMatrixConfig = EngineMatrixConfig(),
    trace_directory: str | Path | None = None,
    episode_runner: EpisodeRunner = run_engine_mpc_play,
    controller_factory: ControllerFactory | None = None,
    demonstration_builder: NativeDemonstrationBuilder | None = None,
) -> dict[str, Any]:
    """Run the Cartesian target/profile/seed matrix on one engine session."""

    target_values = tuple(targets)
    seed_values = tuple(int(value) for value in seeds)
    profile_values = tuple(dict.fromkeys(str(value) for value in profiles))
    if not target_values or not seed_values or not profile_values:
        raise ValueError("targets, seeds, and profiles must all be nonempty")
    if len(set(seed_values)) != len(seed_values):
        raise ValueError("matrix seeds must be unique")
    target_ids = [value.target_id for value in target_values]
    if len(set(target_ids)) != len(target_ids):
        raise ValueError("matrix targets must be unique")
    for profile in profile_values:
        controller_config_for_profile(profile, config)
    traces = None if trace_directory is None else Path(trace_directory)
    create_controller = controller_factory or (
        lambda profile, matrix_config: EngineMPC(
            controller_config_for_profile(profile, matrix_config),
        )
    )
    started = time.perf_counter()
    expected_identity: tuple[Any, Any, Any] | None = None
    episodes: list[dict[str, Any]] = []

    for target in target_values:
        for profile in profile_values:
            for seed in seed_values:
                controller = create_controller(profile, config)
                demonstration_episode = (
                    None
                    if demonstration_builder is None else
                    demonstration_builder.begin(NativeEpisodeIdentity(
                        episode_kind=target.episode_kind,
                        scenario=target.scenario,
                        attack=target.attack,
                        seed=seed,
                        profile=profile,
                    ))
                )
                play_config = EngineMPCPlayConfig(
                    max_frames=config.max_frames,
                    observation_delay=config.observation_delay,
                    shoot_minimum_margin=config.shoot_minimum_margin,
                    render=config.render,
                    render_every=config.render_every,
                )
                runner_options: dict[str, Any] = {}
                if demonstration_episode is not None:
                    runner_options.update({
                        "decision_observer": demonstration_episode.record,
                        "vision_config": VisionConfig(
                            history=1,
                            observation_delay=config.observation_delay,
                        ),
                    })
                raw = episode_runner(
                    client,
                    scenario=target.scenario,
                    attack=target.attack,
                    stage=(target.scenario if target.episode_kind == "stage" else None),
                    seed=seed,
                    player=player,
                    controller=controller,
                    config=play_config,
                    **runner_options,
                )
                if raw.get("seed") != seed:
                    raise EngineProtocolError("matrix episode seed differs from its request")
                if raw.get("scenario") != target.scenario:
                    raise EngineProtocolError("matrix episode scenario differs from its request")
                if raw.get("attack") != target.attack:
                    raise EngineProtocolError("matrix episode attack differs from its request")
                if raw.get("episode_kind") != target.episode_kind:
                    raise EngineProtocolError("matrix episode kind differs from its request")
                if target.episode_kind == "stage" and raw.get("stage") != target.scenario:
                    raise EngineProtocolError("matrix stage differs from its request")
                identity = _engine_identity(raw)
                _validate_engine_identity(identity)
                if expected_identity is None:
                    expected_identity = identity
                elif identity != expected_identity:
                    raise RuntimeError(
                        "matrix episodes did not use one engine process/session",
                    )
                trace_path = None
                if traces is not None:
                    safe_target = _SAFE_NAME.sub("_", target.target_id)
                    trace_path = (
                        traces / safe_target / profile / f"seed-{seed}.json"
                    )
                evidence = _episode_evidence(
                    raw,
                    target=target,
                    profile=profile,
                    trace_path=trace_path,
                )
                episodes.append(evidence)
                if demonstration_episode is not None:
                    demonstration_builder.finish(
                        demonstration_episode,
                        strict_success=evidence["strict_success"] is True,
                        termination_reason=str(evidence["termination_reason"]),
                    )

    groups = []
    for target in target_values:
        for profile in profile_values:
            selected = [
                value for value in episodes
                if value["target_id"] == target.target_id
                and value["profile"] == profile
            ]
            groups.append({
                "target_id": target.target_id,
                "episode_kind": target.episode_kind,
                "scenario": target.scenario,
                "attack": target.attack,
                "profile": profile,
                "summary": _summary(selected),
            })
    overall = _summary(episodes)
    return {
        "schema_version": 2,
        "run_kind": "live_luastg_strict_evaluation_matrix",
        "acceptance_claim": False,
        "implementation_sha256": source_tree_sha256(),
        "passed": overall["strict_successes"] == overall["attempts"],
        "success_criterion": (
            "terminated=true and engine termination reason equals the catalog "
            "target completion reason, with explicit finite final player death=0; "
            "runner success is independently checked"
        ),
        "attack_completion_reason": "attack_complete",
        "stage_completion_reason": "stage_complete",
        "same_engine_connection": True,
        "engine_identity": {
            "session_id": None if expected_identity is None else expected_identity[0],
            "process_nonce": None if expected_identity is None else expected_identity[1],
            "process_id": None if expected_identity is None else expected_identity[2],
        },
        "player": player,
        "seeds": list(seed_values),
        "profiles": list(profile_values),
        "profile_controller_configs": {
            profile: asdict(controller_config_for_profile(profile, config))
            for profile in profile_values
        },
        "config": asdict(config),
        "targets": [
            {
                "target_id": target.target_id,
                "episode_kind": target.episode_kind,
                "scenario": target.scenario,
                "attack": target.attack,
                "label": target.label,
                "completion_reason": target.completion_reason,
                "catalog_entry": dict(target.catalog_entry or {}),
            }
            for target in target_values
        ],
        "overall": overall,
        "groups": groups,
        "episodes": episodes,
        "trace_directory": None if traces is None else str(traces),
        "demonstration_collection": (
            None if demonstration_builder is None else {
                "strict_successes_retained": demonstration_builder.accepted_count,
                "scenario_identity_is_not_a_model_input": True,
                "absolute_frame_is_not_a_model_input": True,
            }
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }


def run_engine_policy_matrix(
    client: EngineClient,
    *,
    targets: Sequence[EngineEpisodeTarget],
    seeds: Sequence[int],
    proficiencies: Sequence[str],
    controller_factory: PolicyControllerFactory,
    controller_metadata: Mapping[str, Any],
    player: str = "reimu_player",
    config: EnginePlayConfig = EnginePlayConfig(),
    trace_directory: str | Path | None = None,
    episode_runner: EpisodeRunner = run_engine_play,
) -> dict[str, Any]:
    """Strictly evaluate one learned policy across native episode targets."""

    target_values = tuple(targets)
    seed_values = tuple(int(value) for value in seeds)
    proficiency_values = tuple(dict.fromkeys(str(value) for value in proficiencies))
    if not target_values or not seed_values or not proficiency_values:
        raise ValueError("targets, seeds, and proficiencies must all be nonempty")
    if len(set(seed_values)) != len(seed_values):
        raise ValueError("matrix seeds must be unique")
    target_ids = [value.target_id for value in target_values]
    if len(set(target_ids)) != len(target_ids):
        raise ValueError("matrix targets must be unique")
    _verify_policy_checkpoint_metadata(controller_metadata)

    traces = None if trace_directory is None else Path(trace_directory)
    expected_identity: tuple[Any, Any, Any] | None = None
    episodes: list[dict[str, Any]] = []
    started = time.perf_counter()

    for target in target_values:
        for proficiency in proficiency_values:
            for seed in seed_values:
                controller = controller_factory(target, proficiency, seed)
                if not isinstance(controller, VisualPolicyController):
                    raise TypeError(
                        "policy matrix controller factory did not return a "
                        "VisualPolicyController"
                    )
                raw = episode_runner(
                    client,
                    scenario=target.scenario,
                    attack=target.attack,
                    stage=(target.scenario if target.episode_kind == "stage" else None),
                    seed=seed,
                    player=player,
                    controller=controller,
                    controller_metadata=controller_metadata,
                    config=config,
                )
                if raw.get("seed") != seed:
                    raise EngineProtocolError("matrix episode seed differs from its request")
                if raw.get("scenario") != target.scenario:
                    raise EngineProtocolError(
                        "matrix episode scenario differs from its request",
                    )
                if raw.get("attack") != target.attack:
                    raise EngineProtocolError("matrix episode attack differs from its request")
                if raw.get("episode_kind") != target.episode_kind:
                    raise EngineProtocolError("matrix episode kind differs from its request")
                if target.episode_kind == "stage" and raw.get("stage") != target.scenario:
                    raise EngineProtocolError("matrix stage differs from its request")

                identity = _engine_identity(raw)
                _validate_engine_identity(identity)
                if expected_identity is None:
                    expected_identity = identity
                elif identity != expected_identity:
                    raise RuntimeError(
                        "matrix episodes did not use one engine process/session",
                    )

                trace_path = None
                if traces is not None:
                    safe_target = _SAFE_NAME.sub("_", target.target_id)
                    trace_path = (
                        traces / safe_target / proficiency / f"seed-{seed}.json"
                    )
                evidence = _episode_evidence(
                    raw,
                    target=target,
                    profile=proficiency,
                    trace_path=trace_path,
                )
                evidence["proficiency"] = evidence.pop("profile")
                evidence["controller"] = raw.get("controller")
                episodes.append(evidence)

    groups = []
    for target in target_values:
        for proficiency in proficiency_values:
            selected = [
                value for value in episodes
                if value["target_id"] == target.target_id
                and value["proficiency"] == proficiency
            ]
            groups.append({
                "target_id": target.target_id,
                "episode_kind": target.episode_kind,
                "scenario": target.scenario,
                "attack": target.attack,
                "proficiency": proficiency,
                "summary": _summary(selected),
            })
    overall = _summary(episodes)
    pure_policy = (
        not config.visible_safety_shield
        and all(
            episode.get("pure_policy") is True
            and episode.get("pure_policy_validation_eligible") is True
            and episode.get("zero_visible_safety_interventions") is True
            for episode in episodes
        )
    )
    pure_policy_success = (
        pure_policy
        and overall["pure_policy_successes"] == overall["attempts"]
    )
    return {
        "schema_version": 2,
        "run_kind": "live_luastg_strict_policy_matrix",
        "acceptance_claim": False,
        "implementation_sha256": source_tree_sha256(),
        "passed": pure_policy_success,
        "success_criterion": (
            "terminated=true and engine termination reason equals the catalog "
            "target completion reason, with explicit finite final player death=0; "
            "runner success is independently checked"
        ),
        "pure_policy": pure_policy,
        "pure_policy_success": pure_policy_success,
        "pure_policy_validation_eligible": pure_policy,
        "checkpoint_evidence_verified": True,
        "strict_outcomes_passed": (
            overall["strict_successes"] == overall["attempts"]
        ),
        "same_engine_connection": True,
        "engine_identity": {
            "session_id": None if expected_identity is None else expected_identity[0],
            "process_nonce": None if expected_identity is None else expected_identity[1],
            "process_id": None if expected_identity is None else expected_identity[2],
        },
        "controller": dict(controller_metadata),
        "player": player,
        "seeds": list(seed_values),
        "proficiencies": list(proficiency_values),
        "config": asdict(config),
        "targets": [
            {
                "target_id": target.target_id,
                "episode_kind": target.episode_kind,
                "scenario": target.scenario,
                "attack": target.attack,
                "label": target.label,
                "completion_reason": target.completion_reason,
                "catalog_entry": dict(target.catalog_entry or {}),
            }
            for target in target_values
        ],
        "overall": overall,
        "groups": groups,
        "episodes": episodes,
        "trace_directory": None if traces is None else str(traces),
        "elapsed_seconds": time.perf_counter() - started,
    }


__all__ = [
    "EngineEpisodeTarget",
    "EngineMatrixConfig",
    "apply_controller_profile",
    "available_engine_profiles",
    "controller_config_for_profile",
    "run_engine_matrix",
    "run_engine_policy_matrix",
    "select_catalog_targets",
]
