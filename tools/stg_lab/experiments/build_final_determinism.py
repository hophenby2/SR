"""Build strict per-frame determinism evidence from visible v2 route control."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from stg_lab.determinism import compare_explicit_trajectory, merge_determinism_comparisons
from stg_lab.memory import EpisodicMemory
from stg_lab.protocol import Action
from stg_lab.provenance import file_sha256
from stg_lab.route_memory import (
    ExternalRouteController,
    ExternalRouteLibraryController,
    RouteControllerConfig,
    load_route_artifact,
    load_route_library_artifact,
    validate_memory_route,
)
from stg_lab.rollout import RolloutConfig, _run_episode
from stg_lab.scenarios import make_environment
from stg_lab.sim import SimulationConfig
from stg_lab.vision import VisionConfig


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
SEED = 5001
DATABASE = ARTIFACTS / "episodic_memory_v2.sqlite"
BOSS3_ROUTE = ARTIFACTS / "route_memory_boss3_v2.json"
BOSS4_LIBRARY = ARTIFACTS / "route_library_boss4_v2.json"
OUTPUT = ARTIFACTS / "determinism_v2.json"
SIMULATION = SimulationConfig(reaction_frames=0, action_hold_frames=3)
VISION = VisionConfig(history=4, observation_delay=5)


def _factory(scenario: str, duration: int):
    def create(seed: int):
        return make_environment(
            scenario,
            difficulty="lunatic",
            seed=seed,
            config=SIMULATION,
            duration_frames=duration,
        )

    return create


def _visible_only(controller: Any):
    def select(_environment, visible, _plan, _memory):
        return controller.select(visible)

    return select


def _expand(actions: tuple[Action, ...], interval: int, frames: int) -> tuple[Action, ...]:
    expanded = tuple(action for action in actions for _ in range(interval))
    if len(expanded) < frames:
        raise ValueError("decision action sequence is shorter than the canonical window")
    return expanded[:frames]


def _capture_actions(
    scenario: str,
    duration: int,
    controller: ExternalRouteController | ExternalRouteLibraryController,
) -> tuple[Action, ...]:
    rollout = RolloutConfig(decision_interval=3, max_frames=duration)
    trace = _run_episode(
        _factory(scenario, duration),
        SEED,
        planner=None,
        vision_config=VISION,
        config=rollout,
        controller=_visible_only(controller),
    )
    if not trace.metrics.survived or trace.metrics.frames != duration:
        raise RuntimeError(f"{scenario} visible route did not survive the canonical window")
    if controller.overrides != 0:
        raise RuntimeError(f"{scenario} route unexpectedly overrode an action")
    if isinstance(controller, ExternalRouteController):
        if not controller.triggered or controller.trigger_decision is None:
            raise RuntimeError(f"{scenario} route never received an online visible cue")
    elif controller.selected_memory is None or controller.selection_decision is None:
        raise RuntimeError(f"{scenario} route library never received an online visible cue")
    return _expand(trace.actions, rollout.decision_interval, duration)


def main() -> None:
    boss3_artifact = load_route_artifact(BOSS3_ROUTE)
    boss4_artifact = load_route_library_artifact(BOSS4_LIBRARY)
    with EpisodicMemory(DATABASE, readonly=True) as store:
        boss3_memory = store.best(
            boss3_artifact.scenario,
            boss3_artifact.cue,
            minimum_similarity=1.0,
        )
        if boss3_memory is None:
            raise RuntimeError("Boss #3 route is absent from the external memory database")
        validate_memory_route(boss3_artifact, boss3_memory)
        boss4_memories = tuple(store.get(memory_id) for memory_id in boss4_artifact.memory_ids)

    route_config = RouteControllerConfig(shield=False, route_origin="episode")
    boss3_controller = ExternalRouteController(boss3_memory, config=route_config)
    boss4_controller = ExternalRouteLibraryController(boss4_memories, config=route_config)
    boss3_actions = _capture_actions("stage5_boss3", 600, boss3_controller)
    boss4_actions = _capture_actions("stage5_boss4", 700, boss4_controller)

    report = merge_determinism_comparisons((
        compare_explicit_trajectory(
            _factory("stage5_boss3", 600),
            SEED,
            boss3_actions,
            max_frames=600,
        ),
        compare_explicit_trajectory(
            _factory("stage5_boss4", 700),
            SEED,
            boss4_actions,
            max_frames=700,
        ),
    ))
    database_sha256 = file_sha256(DATABASE)
    report.update({
        "artifact_kind": "standalone_simulator_determinism",
        "generator": str(Path(__file__).resolve().relative_to(ROOT)),
        "generator_sha256": file_sha256(Path(__file__)),
        "seed": SEED,
        "simulation_config": asdict(SIMULATION),
        "vision_config": asdict(VISION),
        "controller_contract": {
            "shield": False,
            "authority_state_used": False,
            "observation_delay": VISION.observation_delay,
            "decision_interval": 3,
            "motion_estimation": VISION.motion_estimation,
        },
        "sources": {
            "stage5_boss3": {
                "kind": "external_route_memory",
                "artifact": str(BOSS3_ROUTE.relative_to(ROOT)),
                "artifact_sha256": file_sha256(BOSS3_ROUTE),
                "database": str(DATABASE.relative_to(ROOT)),
                "database_sha256": database_sha256,
                "memory_id": boss3_memory.id,
                "trigger_decision": boss3_controller.trigger_decision,
                "trigger_source_frame": boss3_controller.trigger_source_frame,
            },
            "stage5_boss4": {
                "kind": "external_route_library_memory",
                "artifact": str(BOSS4_LIBRARY.relative_to(ROOT)),
                "artifact_sha256": file_sha256(BOSS4_LIBRARY),
                "database": str(DATABASE.relative_to(ROOT)),
                "database_sha256": database_sha256,
                "memory_id": boss4_controller.selected_memory.id,
                "selection_decision": boss4_controller.selection_decision,
                "selection_source_frame": boss4_controller.selection_source_frame,
            },
        },
    })
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(OUTPUT.relative_to(ROOT)),
        "sha256": file_sha256(OUTPUT),
        "passed": report["passed"],
        "frames": [item["first"]["frames"] for item in report["comparisons"]],
    }, indent=2))


if __name__ == "__main__":
    main()
