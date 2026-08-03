"""Native-engine capture and strict replay for offline teacher routes.

Ghost capture is an iterative training instrument: it records the threat field
conditioned on a route without allowing player collision to delete bullets.
Only :func:`replay_teacher_route_strict` uses ordinary player collision, and
only a native ``attack_complete`` result with explicit zero deaths is counted
as success.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np

from .engine import EngineClient, EngineProtocolError
from .engine_runtime import verify_runtime_source_fingerprints
from .engine_vision import EngineStreamVision
from .native_dataset import (
    NativeDemonstrationBuilder,
    NativeEpisodeIdentity,
    risk_from_clearance,
)
from .protocol import Action
from .vision import VisionConfig


_OBJECT_ARRAYS = (
    "enemy_bullets",
    "enemies",
    "nontjt_enemies",
    "indestructibles",
)
_SQRT_HALF = math.sqrt(0.5)


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _observation(response: Mapping[str, Any]) -> Mapping[str, Any]:
    value = response.get("observation")
    if not isinstance(value, Mapping):
        raise EngineProtocolError("engine response has no observation object")
    return value


def _episode_frame(observation: Mapping[str, Any]) -> int:
    value = observation.get("episode_frame")
    if isinstance(value, bool) or not isinstance(value, int):
        raise EngineProtocolError("engine observation has no integer episode_frame")
    return value


def _player_position(observation: Mapping[str, Any]) -> tuple[float, float] | None:
    player = observation.get("player")
    if not isinstance(player, Mapping):
        return None
    x, y = _finite_number(player.get("x")), _finite_number(player.get("y"))
    return None if x is None or y is None else (x, y)


def _boss_hp(observation: Mapping[str, Any]) -> float:
    records = observation.get("enemies")
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        return math.nan
    candidates: list[tuple[float, float]] = []
    for record in records:
        if not isinstance(record, Mapping):
            continue
        hp = _finite_number(record.get("hp"))
        maximum = _finite_number(record.get("maxhp"))
        if hp is None or maximum is None or maximum <= 0.0 or maximum >= 1e8:
            continue
        candidates.append((maximum, hp))
    return max(candidates)[1] if candidates else math.nan


def _threat_rows(observation: Mapping[str, Any]) -> list[list[float]]:
    result: list[list[float]] = []
    for source in _OBJECT_ARRAYS:
        records = observation.get(source)
        if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
            continue
        for record in records:
            if not isinstance(record, Mapping) or record.get("collidable") is not True:
                continue
            x = _finite_number(record.get("x"))
            y = _finite_number(record.get("y"))
            a = _finite_number(record.get("a"))
            b = _finite_number(record.get("b"))
            angle = _finite_number(record.get("rot"))
            if x is None or y is None or a is None or b is None:
                continue
            a, b = abs(a), abs(b)
            if a <= 0.0 or b <= 0.0:
                continue
            # The final column is reserved for warning/nonlethal geometry.
            # Native collidable object rows captured here are all zero.
            result.append([x, y, a, b, angle or 0.0, 0.0])
    return result


@dataclass(frozen=True, slots=True)
class RouteProgram:
    source_path: Path
    source_sha256: str
    raw: Mapping[str, Any]
    actions: tuple[Action, ...]
    expected_positions: tuple[tuple[float, float], ...]
    decision_count: int

    @property
    def route_frames(self) -> int:
        return len(self.actions)


def load_route_program(path: str | Path) -> RouteProgram:
    source = Path(path)
    source_bytes = source.read_bytes()
    raw = json.loads(source_bytes)
    if not isinstance(raw, Mapping) or raw.get("kind") != "offline_space_time_teacher_route":
        raise ValueError("route JSON is not an offline teacher route")
    decisions = raw.get("decisions")
    config = raw.get("config")
    if not isinstance(decisions, Sequence) or isinstance(decisions, (str, bytes)):
        raise ValueError("teacher route has no decision sequence")
    if not isinstance(config, Mapping):
        raise ValueError("teacher route has no config object")

    actions: list[Action] = []
    for index, decision in enumerate(decisions):
        if not isinstance(decision, Mapping):
            raise ValueError(f"route decision {index} is not an object")
        frame_count = decision.get("frame_count")
        action_raw = decision.get("action")
        if (
            isinstance(frame_count, bool)
            or not isinstance(frame_count, int)
            or not 1 <= frame_count <= 3
        ):
            raise ValueError(f"route decision {index} has an invalid frame_count")
        if not isinstance(action_raw, Mapping):
            raise ValueError(f"route decision {index} has no action")
        action = Action(
            move_x=int(action_raw.get("move_x", 0)),
            move_y=int(action_raw.get("move_y", 0)),
            slow=action_raw.get("slow") is True,
            shoot=action_raw.get("shoot") is not False,
            spell=False,
        )
        actions.extend([action] * frame_count)

    def number(name: str, default: float) -> float:
        value = _finite_number(config.get(name))
        return default if value is None else value

    bounds_raw = config.get("bounds", (-184.0, 184.0, -208.0, 192.0))
    if not isinstance(bounds_raw, Sequence) or len(bounds_raw) != 4:
        raise ValueError("teacher route bounds must have four values")
    left, right, bottom, top = (float(value) for value in bounds_raw)
    padding = number("boundary_padding", 0.0)
    left, right = left + padding, right - padding
    bottom, top = bottom + padding, top - padding
    fast_speed = number("fast_speed", 4.0)
    focus_speed = number("focus_speed", 2.0)
    x, y = number("start_x", 0.0), number("start_y", -176.0)
    positions: list[tuple[float, float]] = [(x, y)]
    for action in actions:
        speed = focus_speed if action.slow else fast_speed
        if action.move_x and action.move_y:
            speed *= _SQRT_HALF
        x = min(max(x + speed * action.move_x, left), right)
        y = min(max(y + speed * action.move_y, bottom), top)
        positions.append((x, y))

    return RouteProgram(
        source_path=source,
        source_sha256=hashlib.sha256(source_bytes).hexdigest(),
        raw=raw,
        actions=tuple(actions),
        expected_positions=tuple(positions),
        decision_count=len(decisions),
    )


@dataclass(frozen=True, slots=True)
class NativeRouteConfig:
    max_frames: int = 4200
    fallback_action: Action = Action(shoot=True)

    def __post_init__(self) -> None:
        if self.max_frames <= 0:
            raise ValueError("max_frames must be positive")
        if self.fallback_action.spell:
            raise ValueError("teacher route fallback cannot use a spell")


def _decision_labels(route: RouteProgram) -> dict[int, tuple[Action, float]]:
    """Index delayed-vision labels by their route-frame decision boundary."""

    raw_decisions = route.raw.get("decisions")
    if not isinstance(raw_decisions, Sequence) or isinstance(
        raw_decisions, (str, bytes),
    ):
        raise ValueError("teacher route has no decision sequence")
    labels: dict[int, tuple[Action, float]] = {}
    frame = 0
    for index, raw in enumerate(raw_decisions):
        if not isinstance(raw, Mapping):
            raise ValueError(f"route decision {index} is not an object")
        frame_count = raw.get("frame_count")
        if isinstance(frame_count, bool) or not isinstance(frame_count, int):
            raise ValueError(f"route decision {index} has an invalid frame_count")
        if frame >= route.route_frames:
            raise ValueError("route decision boundaries exceed expanded actions")
        clearance = _finite_number(raw.get("minimum_clearance"))
        labels[frame] = (
            route.actions[frame],
            risk_from_clearance(math.inf if clearance is None else clearance),
        )
        frame += frame_count
    if frame != route.route_frames:
        raise ValueError("route decision boundaries do not cover expanded actions")
    return labels


@dataclass(slots=True)
class _RunTrace:
    initial_frame: int
    frames: list[int]
    player_positions: list[tuple[float, float]]
    expected_positions: list[tuple[float, float]]
    divergences: list[float]
    boss_hp: list[float]

    @classmethod
    def start(cls, observation: Mapping[str, Any], expected: tuple[float, float]) -> "_RunTrace":
        actual = _player_position(observation)
        if actual is None:
            raise EngineProtocolError("engine observation has no player position")
        return cls(
            initial_frame=_episode_frame(observation),
            frames=[_episode_frame(observation)],
            player_positions=[actual],
            expected_positions=[expected],
            divergences=[math.dist(actual, expected)],
            boss_hp=[_boss_hp(observation)],
        )

    def push(self, observation: Mapping[str, Any], expected: tuple[float, float]) -> None:
        frame = _episode_frame(observation)
        if frame != self.frames[-1] + 1:
            raise EngineProtocolError("engine episode_frame did not advance by one")
        actual = _player_position(observation)
        if actual is None:
            raise EngineProtocolError("engine observation has no player position")
        self.frames.append(frame)
        self.player_positions.append(actual)
        self.expected_positions.append(expected)
        self.divergences.append(math.dist(actual, expected))
        self.boss_hp.append(_boss_hp(observation))

    def report(self) -> dict[str, Any]:
        divergence_index = int(np.argmax(self.divergences))
        finite_hp = [value for value in self.boss_hp if math.isfinite(value)]
        return {
            "initial_episode_frame": self.initial_frame,
            "final_episode_frame": self.frames[-1],
            "frame_samples": len(self.frames),
            "frames_advanced": self.frames[-1] - self.initial_frame,
            "maximum_route_divergence": self.divergences[divergence_index],
            "maximum_route_divergence_frame": self.frames[divergence_index],
            "rms_route_divergence": math.sqrt(
                sum(value * value for value in self.divergences) / len(self.divergences)
            ),
            "boss_hp_initial": finite_hp[0] if finite_hp else None,
            "boss_hp_final": finite_hp[-1] if finite_hp else None,
            "boss_hp_minimum": min(finite_hp) if finite_hp else None,
        }


def _identity(client: EngineClient) -> dict[str, Any]:
    ping = client.ping()
    runtime_source_verification = verify_runtime_source_fingerprints(ping)
    return {
        "protocol": ping.get("protocol"),
        "session_id": ping.get("session_id"),
        "process_nonce": ping.get("process_nonce"),
        "runtime_identity": ping.get("runtime_identity"),
        "runtime_source_verification": runtime_source_verification,
    }


def capture_route_conditioned_field(
    client: EngineClient,
    *,
    route: RouteProgram,
    scenario: str,
    attack: int,
    seed: int,
    player: str,
    output_npz: str | Path,
    output_json: str | Path,
    iteration: int,
    config: NativeRouteConfig = NativeRouteConfig(),
) -> dict[str, Any]:
    """Capture a route-conditioned field with collision-disabled ghost input."""

    if iteration <= 0:
        raise ValueError("iteration must be positive")
    identity = _identity(client)
    response = client.reset(
        scenario,
        attack,
        seed=int(seed),
        player=player,
        options={"player_ghost": True, "player_collidable": False, "lifeleft": 99},
    )
    raw = _observation(response)
    trace = _RunTrace.start(raw, route.expected_positions[0])
    frames: list[int] = []
    offsets: list[int] = [0]
    threats: list[list[float]] = []
    boss_hp: list[float] = []

    def sample(observation: Mapping[str, Any]) -> None:
        frames.append(_episode_frame(observation))
        threats.extend(_threat_rows(observation))
        offsets.append(len(threats))
        boss_hp.append(_boss_hp(observation))

    sample(raw)
    started = time.monotonic()
    route_frames_used = 0
    fallback_frames = 0
    hp_at_route_exhaustion: float | None = None
    while (
        raw.get("terminated") is not True
        and _episode_frame(raw) - trace.initial_frame < config.max_frames
    ):
        if route_frames_used < route.route_frames:
            action = route.actions[route_frames_used]
            route_frames_used += 1
            expected = route.expected_positions[route_frames_used]
        else:
            action = config.fallback_action
            fallback_frames += 1
            expected = trace.expected_positions[-1]
            if hp_at_route_exhaustion is None:
                value = _boss_hp(raw)
                hp_at_route_exhaustion = value if math.isfinite(value) else None
        raw = _observation(client.step(action, repeat=1))
        trace.push(raw, expected)
        sample(raw)

    terminated = raw.get("terminated") is True
    reason = raw.get("termination_reason") if terminated else "max_frames"
    field_complete = terminated and reason == "attack_complete"
    npz_path, json_path = Path(output_npz), Path(output_json)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        frames=np.asarray(frames, dtype=np.int32),
        offsets=np.asarray(offsets, dtype=np.int64),
        threats=np.asarray(threats, dtype=np.float32).reshape((-1, 6)),
        boss_hp=np.asarray(boss_hp, dtype=np.float32),
        player_positions=np.asarray(trace.player_positions, dtype=np.float32),
        expected_positions=np.asarray(trace.expected_positions, dtype=np.float32),
    )
    report = {
        "schema_version": 2,
        "kind": "route_conditioned_ghost_field",
        "training_only": True,
        "strict_success": False,
        "field_complete": field_complete,
        "success_criterion": "training field reaches native attack_complete",
        "scenario": scenario,
        "attack": int(attack),
        "seed": int(seed),
        "iteration": iteration,
        "termination_reason": reason,
        "elapsed_seconds": time.monotonic() - started,
        "route_source": str(route.source_path),
        "route_source_sha256": route.source_sha256,
        "route_frames_available": route.route_frames,
        "route_frames_used": route_frames_used,
        "fallback_frames": fallback_frames,
        "boss_hp_at_route_exhaustion": hp_at_route_exhaustion,
        "threat_samples": len(threats),
        "npz": str(npz_path),
        "engine": identity,
        "trace": trace.report(),
    }
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _nearest_collidables(
    observation: Mapping[str, Any],
    *,
    limit: int = 12,
) -> list[dict[str, Any]]:
    player = _player_position(observation)
    if player is None:
        return []
    result: list[tuple[float, dict[str, Any]]] = []
    for source in _OBJECT_ARRAYS:
        records = observation.get(source)
        if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
            continue
        for record in records:
            if not isinstance(record, Mapping) or record.get("collidable") is not True:
                continue
            x, y = _finite_number(record.get("x")), _finite_number(record.get("y"))
            a, b = _finite_number(record.get("a")), _finite_number(record.get("b"))
            if x is None or y is None or a is None or b is None:
                continue
            margin = math.dist(player, (x, y)) - max(abs(a), abs(b)) - 0.5
            result.append((margin, {
                "source": source,
                "id": record.get("id"),
                "x": x,
                "y": y,
                "a": a,
                "b": b,
                "margin": margin,
            }))
    return [record for _margin, record in sorted(result, key=lambda item: item[0])[:limit]]


def replay_teacher_route_strict(
    client: EngineClient,
    *,
    route: RouteProgram,
    scenario: str,
    attack: int,
    seed: int,
    player: str,
    output_json: str | Path,
    config: NativeRouteConfig = NativeRouteConfig(),
    demonstration_builder: NativeDemonstrationBuilder | None = None,
    vision_config: VisionConfig = VisionConfig(history=1, observation_delay=5),
) -> dict[str, Any]:
    """Replay with ordinary collision; only zero-death completion passes."""

    identity = _identity(client)
    response = client.reset(
        scenario,
        attack,
        seed=int(seed),
        player=player,
        options={},
    )
    raw = _observation(response)
    trace = _RunTrace.start(raw, route.expected_positions[0])
    demonstration = (
        None
        if demonstration_builder is None else
        demonstration_builder.begin(NativeEpisodeIdentity(
            episode_kind="attack",
            scenario=scenario,
            attack=int(attack),
            seed=int(seed),
            profile="offline_teacher_route",
        ))
    )
    stream_vision = (
        None if demonstration is None else EngineStreamVision(vision_config)
    )
    if stream_vision is not None:
        stream_vision.reset(raw)
    labels = _decision_labels(route) if demonstration is not None else {}
    started = time.monotonic()
    route_frames_used = 0
    previous = raw
    while (
        raw.get("terminated") is not True
        and route_frames_used < route.route_frames
        and _episode_frame(raw) - trace.initial_frame < config.max_frames
    ):
        previous = raw
        if demonstration is not None and route_frames_used in labels:
            action_label, risk = labels[route_frames_used]
            assert stream_vision is not None
            demonstration.record(stream_vision.observe(), action_label, risk)
        action = route.actions[route_frames_used]
        route_frames_used += 1
        raw = _observation(client.step(action, repeat=1))
        if stream_vision is not None:
            stream_vision.push(raw)
        trace.push(raw, route.expected_positions[route_frames_used])

    terminated = raw.get("terminated") is True
    reason = raw.get("termination_reason") if terminated else (
        "route_exhausted" if route_frames_used >= route.route_frames else "max_frames"
    )
    final_player = raw.get("player")
    final_player = final_player if isinstance(final_player, Mapping) else {}
    final_death = _finite_number(final_player.get("death"))
    strict_success = (
        terminated
        and reason == "attack_complete"
        and final_death == 0.0
    )
    if demonstration is not None:
        assert demonstration_builder is not None
        demonstration_builder.finish(
            demonstration,
            strict_success=strict_success,
            termination_reason=str(reason),
        )
    report = {
        "schema_version": 2,
        "run_kind": "training_teacher_native_strict_replay",
        "acceptance_claim": False,
        "strict_success": strict_success,
        "success_criterion": (
            "native termination_reason=attack_complete with default collision "
            "and explicit final player death=0"
        ),
        "scenario": scenario,
        "attack": int(attack),
        "seed": int(seed),
        "termination_reason": reason,
        "terminated": terminated,
        "final_death": final_death,
        "elapsed_seconds": time.monotonic() - started,
        "route_source": str(route.source_path),
        "route_source_sha256": route.source_sha256,
        "route_frames_available": route.route_frames,
        "route_frames_used": route_frames_used,
        "engine": identity,
        "trace": trace.report(),
        "terminal_evidence": {
            "before_episode_frame": _episode_frame(previous),
            "after_episode_frame": _episode_frame(raw),
            "player_before": _player_position(previous),
            "player_after": _player_position(raw),
            "final_player_death": final_death,
            "nearest_collidables_before": _nearest_collidables(previous),
        },
        "demonstration_collection": (
            None if demonstration is None else {
                "decisions_recorded": demonstration.decisions,
                "strict_success_retained": strict_success,
                "model_input_excludes_route_and_absolute_frame": True,
            }
        ),
    }
    destination = Path(output_json)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Native teacher-route capture and replay")
    parser.add_argument("mode", choices=("capture", "replay"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=24816)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--route", type=Path, required=True)
    parser.add_argument("--scenario", default="okuu:Lunatic")
    parser.add_argument("--attack", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--player", default="reimu_player")
    parser.add_argument("--max-frames", type=int, default=4200)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--npz", type=Path)
    parser.add_argument("--save-demos", type=Path)
    parser.add_argument("--demos-manifest", type=Path)
    parser.add_argument("--observation-delay", type=int, default=5)
    parser.add_argument("--iteration", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    if arguments.demos_manifest is not None and arguments.save_demos is None:
        parser.error("--demos-manifest requires --save-demos")
    route = load_route_program(arguments.route)
    config = NativeRouteConfig(max_frames=arguments.max_frames)
    with EngineClient.connect(
        arguments.host,
        arguments.port,
        timeout=arguments.timeout,
    ) as client:
        if arguments.mode == "capture":
            if arguments.npz is None:
                parser.error("capture mode requires --npz")
            if arguments.save_demos is not None:
                parser.error("--save-demos is valid only in replay mode")
            report = capture_route_conditioned_field(
                client,
                route=route,
                scenario=arguments.scenario,
                attack=arguments.attack,
                seed=arguments.seed,
                player=arguments.player,
                output_npz=arguments.npz,
                output_json=arguments.output,
                iteration=arguments.iteration,
                config=config,
            )
            print(
                f"field_complete={report['field_complete']} "
                f"reason={report['termination_reason']} "
                f"frames={report['trace']['frames_advanced']} "
                f"fallback={report['fallback_frames']}"
            )
        else:
            if arguments.npz is not None:
                parser.error("--npz is valid only in capture mode")
            if arguments.observation_delay < 0:
                parser.error("--observation-delay cannot be negative")
            builder = (
                None
                if arguments.save_demos is None else
                NativeDemonstrationBuilder()
            )
            report = replay_teacher_route_strict(
                client,
                route=route,
                scenario=arguments.scenario,
                attack=arguments.attack,
                seed=arguments.seed,
                player=arguments.player,
                output_json=arguments.output,
                config=config,
                demonstration_builder=builder,
                vision_config=VisionConfig(
                    history=1,
                    observation_delay=arguments.observation_delay,
                ),
            )
            if builder is not None and builder.accepted_count:
                manifest_path = arguments.demos_manifest or (
                    arguments.save_demos.with_suffix(".manifest.json")
                )
                report["demonstrations"] = builder.save(
                    arguments.save_demos,
                    manifest_path=manifest_path,
                )
                arguments.output.write_text(
                    json.dumps(report, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            elif builder is not None:
                report["demonstrations"] = {
                    "saved": False,
                    "reason": "strict native replay did not complete with death=0",
                }
                arguments.output.write_text(
                    json.dumps(report, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            print(
                f"strict_success={report['strict_success']} "
                f"reason={report['termination_reason']} "
                f"frames={report['trace']['frames_advanced']}"
            )
    if arguments.mode == "capture":
        return 0 if report["field_complete"] is True else 1
    return 0 if report["strict_success"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
