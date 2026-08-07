"""Causal visible-policy demonstrations from native THlib replays."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence
import zlib

import numpy as np

from .engine import EngineClient, EngineProtocolError
from .engine_play import (
    EnginePlayConfig,
    _OutcomeTrace,
    _observation,
    visible_shoot_gate,
)
from .engine_replay_analysis import _validated_reset_metadata
from .engine_runtime import verify_runtime_source_fingerprints
from .engine_vision import EngineStreamVision
from .native_dataset import (
    NativeDemonstrationBuilder,
    NativeEpisodeBuffer,
    NativeEpisodeIdentity,
)
from .protocol import Action
from .provenance import file_sha256, source_tree_sha256
from .training import Demonstrations
from .vision import VisionConfig, VisionObservation


_REPLAY_BITS = {
    "up": 128,
    "down": 64,
    "left": 32,
    "right": 16,
    "slow": 8,
    "shoot": 4,
    "spell": 2,
    "special": 1,
}


@dataclass(frozen=True, slots=True)
class ReplayDemonstrationConfig:
    """Collection settings matched to the three-frame live policy primitive."""

    max_frames: int = 120_000
    decision_interval: int = 3
    decision_phase_offset: int = 0
    action_projection: str = "exact-hold"
    vision: VisionConfig = VisionConfig(history=1, observation_delay=5)
    render: bool = False
    render_every: int = 1

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_frames, bool)
            or not isinstance(self.max_frames, int)
            or self.max_frames <= 0
        ):
            raise ValueError("max_frames must be a positive integer")
        if self.decision_interval != 3:
            raise ValueError(
                "replay demonstrations must use the live three-frame motor primitive"
            )
        if (
            isinstance(self.decision_phase_offset, bool)
            or not isinstance(self.decision_phase_offset, int)
            or not 0 <= self.decision_phase_offset < self.decision_interval
        ):
            raise ValueError(
                "decision_phase_offset must be an integer in "
                "[0, decision_interval)"
            )
        if self.action_projection not in {
            "exact-hold", "first", "midpoint", "modal",
        }:
            raise ValueError(
                "action_projection must be 'exact-hold', 'first', 'midpoint', "
                "or 'modal'"
            )
        if (
            isinstance(self.render_every, bool)
            or not isinstance(self.render_every, int)
            or not 1 <= self.render_every <= 600
        ):
            raise ValueError("render_every must be an integer in [1, 600]")


def replay_byte_action(value: int) -> Action:
    """Decode the eight THlib replay key bits into one movement label."""

    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 255:
        raise ValueError("replay input byte must be an integer in [0, 255]")
    move_x = int(bool(value & _REPLAY_BITS["right"])) - int(
        bool(value & _REPLAY_BITS["left"])
    )
    move_y = int(bool(value & _REPLAY_BITS["up"])) - int(
        bool(value & _REPLAY_BITS["down"])
    )
    return Action(
        move_x=move_x,
        move_y=move_y,
        slow=bool(value & _REPLAY_BITS["slow"]),
        shoot=bool(value & _REPLAY_BITS["shoot"]),
        spell=bool(value & _REPLAY_BITS["spell"]),
    )


def aggregate_replay_actions(values: Sequence[Action]) -> Action:
    """Choose the modal three-frame motor primitive, breaking ties at the midpoint."""

    if not values:
        raise ValueError("at least one replay action is required")
    counts = Counter(value.discrete for value in values)
    maximum = max(counts.values())
    candidates = {action for action, count in counts.items() if count == maximum}
    midpoint = values[len(values) // 2].discrete
    selected = midpoint if midpoint in candidates else min(candidates)
    return Action.from_discrete(selected, shoot=True)


def project_replay_actions(values: Sequence[Action], mode: str) -> Action:
    """Project recorded frame actions onto one representative held action.

    ``exact-hold`` only returns an action when the native replay really held
    that movement/speed value for the whole window. Collection keeps mixed
    windows as unsupervised temporal context instead of calling their endpoint
    action an exact three-frame label.
    """

    if not values:
        raise ValueError("at least one replay action is required")
    if mode == "exact-hold":
        selected = values[0].discrete
        if any(value.discrete != selected for value in values[1:]):
            raise ValueError("exact-hold requires one action for the full window")
    elif mode == "first":
        selected = values[0].discrete
    elif mode == "midpoint":
        selected = values[len(values) // 2].discrete
    elif mode == "modal":
        return aggregate_replay_actions(values)
    else:
        raise ValueError(
            "action projection must be 'exact-hold', 'first', 'midpoint', or "
            "'modal'"
        )
    return Action.from_discrete(selected, shoot=True)


class _ReplayEpisodeBuffer(NativeEpisodeBuffer):
    """Native episode buffer with an explicit per-decision action label mask."""

    def __init__(self, identity: NativeEpisodeIdentity) -> None:
        super().__init__(identity)
        self.action_supervision: list[bool] = []

    def record(
        self,
        visible: VisionObservation,
        action: Action,
        risk: float,
        previous_action: Action | None = None,
        *,
        supervised: bool = True,
    ) -> None:
        super().record(
            visible,
            action,
            risk,
            previous_action=previous_action,
        )
        self.action_supervision.append(bool(supervised))


class ReplayDemonstrationBuilder(NativeDemonstrationBuilder):
    """Preserve mixed replay windows as unsupervised recurrent context."""

    def begin(self, identity: NativeEpisodeIdentity) -> _ReplayEpisodeBuffer:
        return _ReplayEpisodeBuffer(identity)

    def build(self) -> Demonstrations:
        demonstrations = super().build()
        flags = [
            supervised
            for episode in self._accepted
            for supervised in episode.action_supervision
        ]
        if len(flags) != demonstrations.actions.shape[0]:
            raise RuntimeError("replay supervision mask does not align with samples")
        demonstrations.supervision_mask = np.asarray(
            flags,
            dtype=bool,
        ).reshape(-1, 1)
        demonstrations.validate()
        return demonstrations


def _read_verified_frame_bytes(
    replay_file: str | Path,
    metadata: Mapping[str, Any],
) -> tuple[bytes, Path]:
    path = Path(replay_file).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"local replay file does not exist: {path}")
    raw = path.read_bytes()
    declared_size = metadata.get("file_size")
    position = metadata.get("frame_data_position")
    count = metadata.get("frame_count")
    if (
        isinstance(declared_size, bool)
        or not isinstance(declared_size, int)
        or len(raw) != declared_size
        or isinstance(position, bool)
        or not isinstance(position, int)
        or isinstance(count, bool)
        or not isinstance(count, int)
        or position < 0
        or count <= 0
        or position + count != len(raw)
    ):
        raise ValueError("local replay extent disagrees with native replay metadata")
    checksum = f"{zlib.crc32(raw) & 0xFFFFFFFF:08x}"
    if checksum != metadata.get("crc32"):
        raise ValueError("local replay checksum disagrees with native replay metadata")
    return raw[position:position + count], path


def _catalog_attack(
    response: Mapping[str, Any],
    *,
    scenario: str,
    card_index: int,
) -> int:
    catalog = response.get("catalog")
    attacks = catalog.get("attacks") if isinstance(catalog, Mapping) else None
    if not isinstance(attacks, list):
        raise EngineProtocolError("engine response has no live attack catalog")
    matches = [
        value.get("attack")
        for value in attacks
        if isinstance(value, Mapping)
        and value.get("scenario") == scenario
        and value.get("card_index") == card_index
    ]
    if (
        len(matches) != 1
        or isinstance(matches[0], bool)
        or not isinstance(matches[0], int)
        or matches[0] <= 0
    ):
        raise EngineProtocolError(
            "live catalog does not uniquely map the replay card to an attack"
        )
    return matches[0]


def _zero_death(outcome: Mapping[str, Any]) -> bool:
    player = outcome.get("final_player")
    death = player.get("death") if isinstance(player, Mapping) else None
    return (
        not isinstance(death, bool)
        and isinstance(death, (int, float))
        and math.isfinite(float(death))
        and float(death) == 0.0
    )


def collect_replay_demonstrations(
    client: EngineClient,
    *,
    replay_path: str,
    replay_file: str | Path,
    config: ReplayDemonstrationConfig = ReplayDemonstrationConfig(),
) -> tuple[dict[str, Any], ReplayDemonstrationBuilder]:
    """Pair delayed visible state with following recorded motor input.

    The phase prefix is advanced before the first decision input is rasterized.
    Every policy input is then fully rasterized before its following replay-input
    window advances, so no future visual state can leak into that input. Under
    the default ``exact-hold`` aggregation, only windows whose three native
    movement/speed inputs are identical receive an action label. Mixed windows
    remain in the recurrent visual trajectory with action supervision masked.
    """

    if not isinstance(replay_path, str) or not replay_path:
        raise ValueError("replay_path must be a nonempty string")
    ping = client.ping()
    commands = ping.get("commands")
    required = {"catalog", "reset_replay", "step", "display"}
    if not isinstance(commands, list) or not required.issubset(commands):
        raise EngineProtocolError("engine bridge lacks replay collection commands")
    runtime_verification = verify_runtime_source_fingerprints(ping)
    catalog_response = client.catalog()
    response = client.reset_replay(replay_path)
    metadata = _validated_reset_metadata(response, requested_path=replay_path)
    frame_bytes, local_path = _read_verified_frame_bytes(replay_file, metadata)
    scenario = metadata.get("scenario")
    card_index = metadata.get("card_index")
    if (
        not isinstance(scenario, str)
        or not scenario
        or isinstance(card_index, bool)
        or not isinstance(card_index, int)
        or card_index <= 0
    ):
        raise EngineProtocolError("native replay metadata has no attack identity")
    attack = _catalog_attack(
        catalog_response,
        scenario=scenario,
        card_index=card_index,
    )
    display = client.set_rendering(config.render, every=config.render_every)
    if display.get("render") is not config.render:
        raise EngineProtocolError("engine did not apply replay rendering state")
    if display.get("every") != config.render_every:
        raise EngineProtocolError("engine did not apply replay render interval")

    raw = _observation(response)
    if raw.get("episode_frame") != 1:
        raise EngineProtocolError("replay reset must consume exactly one input frame")
    stream = EngineStreamVision(config.vision)
    visible = stream.reset(raw)
    outcome_trace = _OutcomeTrace()
    outcome_trace.push(raw)
    builder = ReplayDemonstrationBuilder()
    episode = builder.begin(NativeEpisodeIdentity(
        episode_kind="attack",
        scenario=scenario,
        attack=attack,
        seed=int(metadata["random_seed"]),
        profile="expert",
    ))
    risk_config = EnginePlayConfig(vision=config.vision)
    previous_action = replay_byte_action(frame_bytes[0])
    decision_sources: list[int] = []
    decision_control_frames: list[int] = []
    projected_input_frames = 0
    projection_changed_windows = 0
    projection_consistent_windows = 0
    modal_midpoint_disagreements = 0
    action_frame_first: int | None = None
    action_frame_last: int | None = None
    supervised_windows = 0
    unsupervised_windows = 0
    supervised_execution_mismatches = 0
    phase_prefix_frames = 0
    tail_frames = 0
    incomplete_window_frames = 0
    neutral = Action(shoot=False)

    terminated = raw.get("terminated") is True

    def advance_replay_frame() -> None:
        nonlocal raw, terminated
        before_frame = raw.get("episode_frame")
        response = client.step(neutral, repeat=1)
        raw = _observation(response)
        after_frame = raw.get("episode_frame")
        if (
            isinstance(before_frame, bool)
            or not isinstance(before_frame, int)
            or isinstance(after_frame, bool)
            or not isinstance(after_frame, int)
            or after_frame != before_frame + 1
        ):
            raise EngineProtocolError(
                "native replay did not advance exactly one logical input frame"
            )
        outcome_trace.push(raw)
        stream.push(raw)
        terminated = raw.get("terminated") is True

    for _ in range(config.decision_phase_offset):
        current_frame = raw.get("episode_frame")
        if isinstance(current_frame, bool) or not isinstance(current_frame, int):
            raise EngineProtocolError("replay observation has no integer episode_frame")
        if (
            terminated
            or current_frame >= config.max_frames
            or current_frame >= len(frame_bytes)
        ):
            break
        advance_replay_frame()
        phase_prefix_frames += 1
    if phase_prefix_frames and not terminated:
        visible = stream.observe()
    current_frame = raw.get("episode_frame")
    if (
        not isinstance(current_frame, bool)
        and isinstance(current_frame, int)
        and 0 < current_frame <= len(frame_bytes)
    ):
        previous_action = replay_byte_action(frame_bytes[current_frame - 1])

    while not terminated:
        current_frame = raw.get("episode_frame")
        if isinstance(current_frame, bool) or not isinstance(current_frame, int):
            raise EngineProtocolError("replay observation has no integer episode_frame")
        if current_frame >= config.max_frames:
            break
        next_stop = current_frame + config.decision_interval
        if next_stop > len(frame_bytes) or next_stop > config.max_frames:
            break
        primitive_actions = [
            replay_byte_action(value)
            for value in frame_bytes[current_frame:next_stop]
        ]
        distinct_actions = {action.discrete for action in primitive_actions}
        exact_hold = len(distinct_actions) == 1
        if config.action_projection == "exact-hold":
            # The terminal action is useful recurrent boundary context, but it
            # is never treated as a label when the preceding window was mixed.
            primitive = Action.from_discrete(
                primitive_actions[-1].discrete,
                shoot=True,
            )
            supervised = exact_hold
        else:
            primitive = project_replay_actions(
                primitive_actions,
                config.action_projection,
            )
            supervised = True
        pending_visible = visible
        pending_risk = visible_shoot_gate(pending_visible, risk_config).risk
        advanced = 0
        for _ in range(config.decision_interval):
            advance_replay_frame()
            advanced += 1
            if terminated:
                break
        if advanced == config.decision_interval:
            projection_changed_windows += int(len(distinct_actions) > 1)
            projection_consistent_windows += int(len(distinct_actions) == 1)
            projected_input_frames += sum(
                action.discrete == primitive.discrete for action in primitive_actions
            )
            modal_midpoint_disagreements += int(
                primitive.discrete
                != primitive_actions[len(primitive_actions) // 2].discrete
            )
            episode.record(
                pending_visible,
                primitive,
                pending_risk,
                previous_action=previous_action,
                supervised=supervised,
            )
            previous_action = Action.from_discrete(
                primitive_actions[-1].discrete,
                shoot=True,
            )
            supervised_windows += int(supervised)
            unsupervised_windows += int(not supervised)
            supervised_execution_mismatches += int(supervised and not exact_hold)
            decision_sources.append(int(pending_visible.source_frame))
            decision_control_frames.append(current_frame)
            if action_frame_first is None:
                action_frame_first = current_frame + 1
            action_frame_last = next_stop
        elif advanced:
            incomplete_window_frames += advanced
        if not terminated:
            visible = stream.observe()

    # A shifted phase usually leaves one or two native replay frames after the
    # final complete policy window. They remain outcome evidence and are never
    # turned into an incomplete training sample.
    while not terminated:
        current_frame = raw.get("episode_frame")
        if isinstance(current_frame, bool) or not isinstance(current_frame, int):
            raise EngineProtocolError("replay observation has no integer episode_frame")
        if current_frame >= config.max_frames or current_frame >= len(frame_bytes):
            break
        advance_replay_frame()
        tail_frames += 1

    if not terminated:
        current_frame = raw.get("episode_frame")
        if (
            not isinstance(current_frame, bool)
            and isinstance(current_frame, int)
            and current_frame >= len(frame_bytes)
        ):
            raise EngineProtocolError(
                "native replay consumed all declared frame bytes without termination"
            )

    final_frame = raw.get("episode_frame")
    consumed_frames = (
        final_frame
        if not isinstance(final_frame, bool) and isinstance(final_frame, int)
        else 0
    )
    consumed = frame_bytes[:max(0, min(consumed_frames, len(frame_bytes)))]
    raw_actions = [replay_byte_action(value) for value in consumed]
    spell_frames = sum(action.spell for action in raw_actions)
    special_frames = sum(bool(value & _REPLAY_BITS["special"]) for value in consumed)
    shoot_frames = sum(action.shoot for action in raw_actions)
    termination_reason = (
        raw.get("termination_reason")
        if raw.get("terminated") is True else
        "max_frames"
    )
    outcome = outcome_trace.report(raw)
    strict_success = (
        raw.get("terminated") is True
        and termination_reason == "attack_complete"
        and _zero_death(outcome)
        and shoot_frames == len(raw_actions)
        and spell_frames == 0
        and special_frames == 0
    )
    builder.finish(
        episode,
        strict_success=strict_success and episode.decisions > 0,
        termination_reason=str(termination_reason),
    )
    selected_actions = [Action.from_discrete(value) for value in episode.actions]
    direction_changes = sum(
        (current.move_x, current.move_y) != (previous.move_x, previous.move_y)
        for previous, current in zip(selected_actions, selected_actions[1:])
    )
    exact_reversals = sum(
        (current.move_x, current.move_y)
        == (-previous.move_x, -previous.move_y)
        and (current.move_x, current.move_y) != (0, 0)
        for previous, current in zip(selected_actions, selected_actions[1:])
    )
    report = {
        "schema_version": 1,
        "run_kind": "native_human_replay_demonstration_collection",
        "acceptance_claim": False,
        "training_only": True,
        "implementation_sha256": source_tree_sha256(),
        "engine": {
            "protocol": ping.get("protocol"),
            "session_id": ping.get("session_id"),
            "process_nonce": ping.get("process_nonce"),
            "runtime_identity": ping.get("runtime_identity"),
            "runtime_source_verification": runtime_verification,
        },
        "config": {
            **asdict(config),
            "vision": asdict(config.vision),
        },
        "replay": dict(metadata),
        "local_replay_file": str(local_path),
        "local_replay_sha256": file_sha256(local_path),
        "scenario": scenario,
        "attack": attack,
        "card_index": card_index,
        "seed": int(metadata["random_seed"]),
        "terminated": raw.get("terminated") is True,
        "termination_reason": termination_reason,
        "strict_success": strict_success,
        "passed": strict_success,
        "outcome_evidence": outcome,
        "frames_consumed": consumed_frames,
        "decision_count": episode.decisions,
        "source_frame_range": {
            "first": decision_sources[0] if decision_sources else None,
            "last": decision_sources[-1] if decision_sources else None,
        },
        "decision_phase": {
            "offset_frames": config.decision_phase_offset,
            "interval_frames": config.decision_interval,
            "valid_offset_range": [0, config.decision_interval - 1],
            "prefix_frames_advanced": phase_prefix_frames,
            "tail_outcome_frames_advanced": tail_frames,
            "incomplete_terminal_window_frames": incomplete_window_frames,
            "first_control_episode_frame": (
                decision_control_frames[0] if decision_control_frames else None
            ),
            "last_control_episode_frame": (
                decision_control_frames[-1] if decision_control_frames else None
            ),
            "sampling_only": True,
            "model_input_fields_added": [],
            "offset_is_model_input": False,
        },
        "raw_input": {
            "shoot_frames": shoot_frames,
            "shoot_fraction": shoot_frames / len(raw_actions) if raw_actions else 0.0,
            "continuous_shoot": bool(raw_actions) and shoot_frames == len(raw_actions),
            "spell_frames": spell_frames,
            "special_frames": special_frames,
            "contradictory_horizontal_frames": sum(
                bool(value & _REPLAY_BITS["left"])
                and bool(value & _REPLAY_BITS["right"])
                for value in consumed
            ),
            "contradictory_vertical_frames": sum(
                bool(value & _REPLAY_BITS["up"])
                and bool(value & _REPLAY_BITS["down"])
                for value in consumed
            ),
        },
        "aggregated_motor_labels": {
            "interval_frames": config.decision_interval,
            "decision_phase_offset": config.decision_phase_offset,
            "projection": config.action_projection,
            "aggregation_strategy": config.action_projection,
            "projection_contract": (
                "label_only_native_three_frame_holds"
                if config.action_projection == "exact-hold" else
                "legacy_representative_action_for_native_framewise_window"
            ),
            "projection_is_lossy": projection_changed_windows > 0,
            "complete_windows": episode.decisions,
            "supervised_windows": supervised_windows,
            "unsupervised_context_windows": unsupervised_windows,
            "unchanged_input_windows": projection_consistent_windows,
            "changed_input_windows": projection_changed_windows,
            "exact_hold_windows": projection_consistent_windows,
            "supervised_execution_mismatch_windows": (
                supervised_execution_mismatches
            ),
            "trajectory_execution_contract_satisfied": (
                supervised_execution_mismatches == 0
            ),
            "modal_midpoint_disagreements": modal_midpoint_disagreements,
            "projected_input_frame_agreement": (
                projected_input_frames
                / (episode.decisions * config.decision_interval)
                if episode.decisions else 0.0
            ),
            "action_frame_range": {
                "first": action_frame_first,
                "last": action_frame_last,
            },
            "initial_delay_padding": {
                "used_by_live_policy_too": True,
                "padded_decisions": sum(
                    source == 1 for source in decision_sources
                ),
                "reason": (
                    "the replay begins after its first input; the streaming "
                    "vision buffer repeats that first state during startup"
                ),
            },
            "direction_changes": direction_changes,
            "exact_reversals": exact_reversals,
            "slow_fraction": (
                sum(action.slow for action in selected_actions) / len(selected_actions)
                if selected_actions else 0.0
            ),
        },
        "action_supervision": {
            "mask": "supervision_mask",
            "supervised_decisions": supervised_windows,
            "unsupervised_context_decisions": unsupervised_windows,
            "trajectory_decisions": episode.decisions,
            "native_trajectory_advancement": (
                "all recorded framewise replay inputs in each three-frame window"
            ),
            "label_execution_contract": (
                "the labelled movement/speed action was executed on every native "
                "frame in its three-frame window"
                if config.action_projection == "exact-hold" else
                "legacy representative labels may differ from one or more native "
                "framewise inputs"
            ),
            "mixed_window_context_action": (
                "terminal_native_action_with_supervision_disabled"
                if config.action_projection == "exact-hold" else None
            ),
            "previous_action_contract": (
                "last native replay movement/speed input before the decision boundary"
            ),
        },
        "artifact_compatibility": {
            "legacy_npz_requires_migration": False,
            "legacy_projection_modes": ["first", "midpoint", "modal"],
            "legacy_cli_option": "--action-projection",
            "supervision_mask_is_additive": True,
        },
        "causal_contract": {
            "model_input": (
                "delayed visible geometry and current visible player pose before "
                "the following native three-frame input window"
            ),
            "supervision_only": (
                "exact native three-frame movement/speed holds only"
                if config.action_projection == "exact-hold" else
                "legacy lossy projection of the following native replay input window"
            ),
            "future_replay_action_bytes_are_supervision": True,
            "future_replay_bytes_are_supervision": True,
            "future_replay_bytes_are_model_inputs": False,
            "future_visual_frames_are_model_inputs": False,
            "model_input_rasterized_before_action_window": True,
            "future_actions_only_affect_labels_or_supervision_mask": True,
            "absolute_frame_is_model_input": False,
            "scenario_or_attack_identity_is_model_input": False,
            "recorded_route_is_model_input": False,
            "decision_phase_offset_is_model_input": False,
            "decision_phase_offset_changes_sampling_boundaries_only": True,
            "phase_prefix_frames_are_past_at_first_model_input": True,
        },
    }
    return report, builder


def save_replay_demonstrations(
    builder: NativeDemonstrationBuilder,
    report: Mapping[str, Any],
    output: str | Path,
    manifest_output: str | Path,
) -> dict[str, Any]:
    """Save accepted data with replay-specific provenance and causal claims."""

    manifest = builder.save(output)
    manifest.update({
        "run_kind": "strict_native_human_replay_demonstrations",
        "source_replay": report.get("local_replay_file"),
        "source_replay_sha256": report.get("local_replay_sha256"),
        "source_replay_crc32": (
            report.get("replay", {}).get("crc32")
            if isinstance(report.get("replay"), Mapping) else None
        ),
        "demonstrator": "native_human_replay",
        "demonstrator_proficiency": "expert",
        "decision_interval": report.get("config", {}).get("decision_interval")
        if isinstance(report.get("config"), Mapping) else None,
        "decision_phase_offset": (
            report.get("config", {}).get("decision_phase_offset")
            if isinstance(report.get("config"), Mapping) else None
        ),
        "decision_phase": report.get("decision_phase"),
        "causal_contract": report.get("causal_contract"),
        "raw_input": report.get("raw_input"),
        "aggregated_motor_labels": report.get("aggregated_motor_labels"),
        "action_supervision": report.get("action_supervision"),
        "artifact_compatibility": report.get("artifact_compatibility"),
        "temporal_contract_version": 3,
        "strict_inclusion_criterion": (
            "native attack_complete with finite death=0, continuous shoot, and "
            "zero spell/special input frames"
        ),
    })
    path = Path(manifest_output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


__all__ = [
    "ReplayDemonstrationConfig",
    "ReplayDemonstrationBuilder",
    "aggregate_replay_actions",
    "collect_replay_demonstrations",
    "project_replay_actions",
    "replay_byte_action",
    "save_replay_demonstrations",
]
