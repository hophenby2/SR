"""Verify that a failed Boss #4 attempt changes after online external recall."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

from stg_lab.memory import EpisodicMemory
from stg_lab.protocol import Action
from stg_lab.provenance import file_sha256, source_tree_sha256
from stg_lab.route_memory import (
    ExternalRouteLibraryController,
    RouteControllerConfig,
    load_route_library_artifact,
)
from stg_lab.rollout import RolloutConfig, _run_episode
from stg_lab.scenarios import make_environment
from stg_lab.sim import SimulationConfig
from stg_lab.vision import VisionConfig


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
DATABASE = ARTIFACTS / "episodic_memory_v2.sqlite"
LIBRARY_PATH = ARTIFACTS / "route_library_boss4_v2.json"
OUTPUT = ARTIFACTS / "memory_benchmark_v2.json"
SEED = 5001
SIMULATION = SimulationConfig(reaction_frames=0, action_hold_frames=3)
VISION = VisionConfig(history=4, observation_delay=5)
ROLLOUT = RolloutConfig(decision_interval=3, max_frames=700)


def _factory(seed: int):
    return make_environment(
        "stage5_boss4",
        difficulty="lunatic",
        seed=seed,
        config=SIMULATION,
        duration_frames=700,
    )


def _visible_only_route_controller(controller: ExternalRouteLibraryController):
    """Adapt the rollout callback without forwarding authority state."""

    def select(_environment, visible, _plan, _memory):
        return controller.select(visible)

    return select


def _load_route_memories(memory_ids: tuple[int, ...]):
    """Load the persistent artifact through the API's immutable read-only mode."""

    with EpisodicMemory(DATABASE, readonly=True) as store:
        return tuple(store.get(memory_id) for memory_id in memory_ids)


def main() -> None:
    first = _run_episode(
        _factory,
        SEED,
        planner=None,
        vision_config=VISION,
        config=ROLLOUT,
        controller=lambda _environment, _visible, _plan, _memory: Action(),
    )
    artifact = load_route_library_artifact(LIBRARY_PATH)
    database_sha256 = file_sha256(DATABASE)
    memories = _load_route_memories(artifact.memory_ids)
    if file_sha256(DATABASE) != database_sha256:
        raise RuntimeError("read-only memory benchmark modified its database artifact")
    controller = ExternalRouteLibraryController(
        memories,
        config=RouteControllerConfig(shield=False, route_origin="episode"),
    )
    second = _run_episode(
        _factory,
        SEED,
        planner=None,
        vision_config=VISION,
        config=ROLLOUT,
        controller=_visible_only_route_controller(controller),
    )
    if first.metrics.survived or not second.metrics.survived:
        raise RuntimeError("memory benchmark did not change failure into survival")
    if controller.selected_memory is None or controller.selection_source_frame is None:
        raise RuntimeError("memory benchmark never selected an online visible cue")
    if controller.selection_decision is None or controller.selection_decision <= 0:
        raise RuntimeError("memory benchmark selected a route before an online cue appeared")
    selection_control_frame = controller.selection_decision * ROLLOUT.decision_interval
    observed_delay = selection_control_frame - controller.selection_source_frame
    if observed_delay != VISION.observation_delay:
        raise RuntimeError(
            "memory selection did not use the configured delayed visual observation"
        )
    if controller.selection_source_frame >= first.metrics.frames:
        raise RuntimeError("memory cue did not appear before the first-attempt death")
    if controller.overrides != 0:
        raise RuntimeError("unshielded memory controller unexpectedly overrode route actions")
    report = {
        "schema_version": 2,
        "generator": str(Path(__file__).resolve().relative_to(ROOT)),
        "generator_sha256": file_sha256(Path(__file__)),
        "implementation_sha256": source_tree_sha256(),
        "online_visible_cue": True,
        "authority_state_used": False,
        "shield": False,
        "database_read_only": True,
        "controller_input_contract": "delayed VisionObservation only; authority object discarded",
        "database": str(DATABASE.relative_to(ROOT)),
        "database_sha256": database_sha256,
        "library_artifact": str(LIBRARY_PATH.relative_to(ROOT)),
        "library_artifact_sha256": file_sha256(LIBRARY_PATH),
        "selection": {
            "memory_id": controller.selected_memory.id,
            "decision": controller.selection_decision,
            "control_frame": selection_control_frame,
            "delayed_source_frame": controller.selection_source_frame,
            "observed_delay_frames": observed_delay,
            "death_frame": first.metrics.frames,
            "lead_frames": first.metrics.frames - controller.selection_source_frame,
        },
        "vision_config": asdict(VISION),
        "simulation_config": asdict(SIMULATION),
        "rollout_config": asdict(ROLLOUT),
        "memory": {
            "first": {
                "memory_available": False,
                **asdict(first.metrics),
            },
            "second": {
                "memory_available": True,
                **asdict(second.metrics),
            },
        },
    }
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(OUTPUT.relative_to(ROOT)),
        "sha256": file_sha256(OUTPUT),
        **report,
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
