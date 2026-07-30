"""Determinism evidence from replaying explicit per-frame action sequences."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from typing import Any, Callable, Iterable, Sequence

from .metrics import state_hash
from .protocol import Action
from .provenance import source_tree_sha256
from .sim import ActionLike, Observation, STGEnvironment, coerce_action


EnvironmentFactory = Callable[[int], STGEnvironment]


@dataclass(frozen=True, slots=True)
class TrajectoryRun:
    scenario: str
    seed: int
    frame_hashes: tuple[str, ...]
    trajectory_hash: str
    frames: int
    actions_consumed: int
    terminated: bool
    outcome: str

    def summary(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "seed": self.seed,
            "frame_hashes": list(self.frame_hashes),
            "trajectory_hash": self.trajectory_hash,
            "frames": self.frames,
            "actions_consumed": self.actions_consumed,
            "terminated": self.terminated,
            "outcome": self.outcome,
        }


def _scenario_key(environment: STGEnvironment) -> str:
    return str(getattr(environment.scenario, "scenario_key", environment.scenario.name))


def _action_payload(action: Action) -> dict[str, Any]:
    return action.to_dict()


def _initial_payload(environment: STGEnvironment, observation: Observation) -> dict[str, Any]:
    return {
        "record_kind": "initial",
        "scenario": _scenario_key(environment),
        "seed": int(environment.seed),
        "simulation_config": asdict(environment.config),
        "frame": int(environment.frame),
        "observation": observation,
        "terminated": bool(environment.done),
        "outcome": environment.outcome.value,
        "submitted_action": _action_payload(environment.submitted_action),
        "requested_action": _action_payload(environment.requested_action),
        "applied_action": _action_payload(environment.applied_action),
    }


def _frame_payload(
    environment: STGEnvironment,
    observation: Observation,
    action: Action,
    *,
    terminated: bool,
    truncated: bool,
) -> dict[str, Any]:
    return {
        "record_kind": "advance",
        "scenario": _scenario_key(environment),
        "seed": int(environment.seed),
        "frame": int(environment.frame),
        "input_action": _action_payload(action),
        "submitted_action": _action_payload(environment.submitted_action),
        "requested_action": _action_payload(environment.requested_action),
        "applied_action": _action_payload(environment.applied_action),
        "observation": observation,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "outcome": environment.outcome.value,
    }


def _rolling_hash(frame_hashes: Sequence[str]) -> str:
    digest = hashlib.blake2s(digest_size=16)
    digest.update(len(frame_hashes).to_bytes(8, "big"))
    for value in frame_hashes:
        digest.update(bytes.fromhex(value))
    return digest.hexdigest()


def _prepare_actions(
    actions: Iterable[ActionLike],
    max_frames: int | None,
) -> tuple[Action, ...]:
    values = tuple(coerce_action(action) for action in actions)
    if not values:
        raise ValueError("at least one explicit action is required")
    if max_frames is not None:
        if max_frames <= 0:
            raise ValueError("max_frames must be positive or None")
        if len(values) < max_frames:
            raise ValueError("the explicit action sequence is shorter than max_frames")
        values = values[:max_frames]
    return values


def _run_environment(
    environment: STGEnvironment,
    seed: int,
    actions: Sequence[Action],
) -> TrajectoryRun:
    observation = environment.reset(seed=int(seed))
    frame_hashes = [state_hash(_initial_payload(environment, observation))]
    actions_consumed = 0
    for action in actions:
        if environment.done:
            break
        # Semantic grids are a deterministic rendering of authority geometry.
        # Excluding them keeps evidence generation cheap without omitting state.
        result = environment._advance(action, build_semantic=False, detect_collision=True)
        actions_consumed += 1
        frame_hashes.append(state_hash(_frame_payload(
            environment,
            result.observation,
            action,
            terminated=result.terminated,
            truncated=result.truncated,
        )))
    return TrajectoryRun(
        scenario=_scenario_key(environment),
        seed=int(seed),
        frame_hashes=tuple(frame_hashes),
        trajectory_hash=_rolling_hash(frame_hashes),
        frames=int(environment.frame),
        actions_consumed=actions_consumed,
        terminated=bool(environment.done),
        outcome=environment.outcome.value,
    )


def run_explicit_trajectory(
    environment_factory: EnvironmentFactory,
    seed: int,
    actions: Iterable[ActionLike],
    *,
    max_frames: int | None = None,
) -> TrajectoryRun:
    """Run one fresh environment and hash its initial and advanced frames."""

    explicit = _prepare_actions(actions, max_frames)
    environment = environment_factory(int(seed))
    return _run_environment(environment, int(seed), explicit)


def compare_explicit_trajectory(
    environment_factory: EnvironmentFactory,
    seed: int,
    actions: Iterable[ActionLike],
    *,
    max_frames: int | None = None,
) -> dict[str, Any]:
    """Replay one action sequence in two fresh environments and compare it."""

    explicit = _prepare_actions(actions, max_frames)
    first_environment = environment_factory(int(seed))
    second_environment = environment_factory(int(seed))
    if first_environment is second_environment:
        raise ValueError("environment_factory must create a fresh environment for each run")
    first = _run_environment(first_environment, int(seed), explicit)
    second = _run_environment(second_environment, int(seed), explicit)
    if first.scenario != second.scenario:
        raise ValueError("fresh runs produced different scenario identities")

    action_payload = [_action_payload(action) for action in explicit]
    action_sequence_hash = state_hash({"actions": action_payload})
    hashes_matched = first.frame_hashes == second.frame_hashes
    trajectory_matched = first.trajectory_hash == second.trajectory_hash
    survived = (
        first.terminated
        and second.terminated
        and first.outcome == "clear"
        and second.outcome == "clear"
        and first.actions_consumed == len(explicit)
        and second.actions_consumed == len(explicit)
    )
    return {
        "scenario": first.scenario,
        "seed": int(seed),
        "hash_scope": "per_frame",
        "initial_frame_included": True,
        "actions": action_payload,
        "action_sequence_hash": action_sequence_hash,
        "first_hashes": list(first.frame_hashes),
        "second_hashes": list(second.frame_hashes),
        "first_trajectory_hash": first.trajectory_hash,
        "second_trajectory_hash": second.trajectory_hash,
        "first": first.summary(),
        "second": second.summary(),
        "matched": hashes_matched and trajectory_matched,
        "survived": survived,
        "passed": hashes_matched and trajectory_matched and survived,
    }


def merge_determinism_comparisons(
    comparisons: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Create the JSON object consumed by the strict acceptance compiler."""

    values = tuple(dict(comparison) for comparison in comparisons)
    if not values:
        raise ValueError("at least one determinism comparison is required")
    return {
        "schema_version": 2,
        "implementation_sha256": source_tree_sha256(),
        "hash_scope": "per_frame",
        "passed": all(comparison.get("passed") is True for comparison in values),
        "comparisons": list(values),
    }


__all__ = [
    "EnvironmentFactory",
    "TrajectoryRun",
    "compare_explicit_trajectory",
    "merge_determinism_comparisons",
    "run_explicit_trajectory",
]
