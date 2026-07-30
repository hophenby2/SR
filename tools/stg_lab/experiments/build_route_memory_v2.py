"""Build cue-triggered external route memories for the final human-visible gate."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from stg_lab.memory import EpisodicMemory
from stg_lab.protocol import Action
from stg_lab.provenance import file_sha256, source_tree_sha256
from stg_lab.route_memory import load_route_artifact, semantic_signature
from stg_lab.scenarios import make_environment
from stg_lab.sim import SimulationConfig
from stg_lab.training import Demonstrations
from stg_lab.vision import DelayedVision, VisionConfig


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
DATABASE = ARTIFACTS / "episodic_memory_v2.sqlite"
VISION = VisionConfig(history=4, observation_delay=5)
SIMULATION = SimulationConfig(reaction_frames=0, action_hold_frames=3)
BOSS4_SIGNATURE = {
    "kind": "semantic_signature",
    "trigger_channel": 0,
    "minimum_mass": 23.0,
    "channels": [0, 1, 2],
    "pooled_height": 14,
    "pooled_width": 12,
}


def _capture_signature(seed: int) -> tuple[dict, int, int]:
    environment = make_environment(
        "stage5_boss4",
        difficulty="lunatic",
        seed=seed,
        config=SIMULATION,
        duration_frames=200,
    )
    observation = environment.reset(seed=seed)
    vision = DelayedVision(config=VISION)
    visible = vision.reset(observation)
    while environment.frame < 180:
        if environment.frame % 3 == 0:
            vector = semantic_signature(BOSS4_SIGNATURE, visible)
            if vector is not None:
                return (
                    {**BOSS4_SIGNATURE, "vector": vector.tolist()},
                    environment.frame,
                    visible.source_frame,
                )
        observation = environment._advance(
            Action(), build_semantic=False, detect_collision=False,
        ).observation
        visible = vision.push(observation)
    raise RuntimeError(f"Boss #4 signature did not appear for seed {seed}")


def _canonical_route(seed: int) -> tuple[Action, ...]:
    part = (seed - 1001) // 3
    demonstrations = Demonstrations.load(ARTIFACTS / f"boss4_canonical_{part}.npz")
    episode_position = (seed - 1001) % 3
    episode_id = tuple(np.unique(demonstrations.episode_ids))[episode_position]
    indices = np.flatnonzero(demonstrations.episode_ids == episode_id)
    return tuple(Action.from_discrete(int(value)) for value in demonstrations.actions[indices, -1])


def _development_route(seed: int) -> tuple[Action, ...]:
    path = ARTIFACTS / f"boss4_route_{seed}_visible_v2.npz"
    demonstrations = Demonstrations.load(path)
    return tuple(Action.from_discrete(int(value)) for value in demonstrations.actions[:, -1])


def main() -> None:
    DATABASE.unlink(missing_ok=True)
    boss3_old = load_route_artifact(ARTIFACTS / "route_memory_boss3.json")
    boss3_path = ARTIFACTS / "route_memory_boss3_v2.json"
    boss4_path = ARTIFACTS / "route_library_boss4_v2.json"
    source_routes = {
        1001: _canonical_route(1001),
        1002: _canonical_route(1002),
        1007: _canonical_route(1007),
        3008: _development_route(3008),
        3015: _development_route(3015),
    }

    with EpisodicMemory(DATABASE) as store:
        boss3 = store.remember(
            boss3_old.scenario,
            boss3_old.cue,
            death_point=None,
            trigger_lead=0,
            route=boss3_old.actions,
            confidence=1.0,
        )
        boss4_memories = []
        cue_frames = {}
        for seed, route in source_routes.items():
            cue, decision_frame, source_frame = _capture_signature(seed)
            cue_frames[str(seed)] = {
                "decision_frame": decision_frame,
                "delayed_source_frame": source_frame,
            }
            boss4_memories.append(store.remember(
                "stage5_boss4:lunatic",
                cue,
                death_point=None,
                # Neutral first attempts die around frame 340.  This records
                # how far before that failure the visible signature appeared.
                trigger_lead=max(0, 340 - source_frame),
                route=route,
                confidence=1.0,
            ))

    boss3_artifact = {
        "schema_version": 1,
        "generator": str(Path(__file__).resolve().relative_to(ROOT)),
        "generator_sha256": file_sha256(Path(__file__)),
        "route_id": "sr-stage5-boss3-visible-v2",
        "scenario": boss3.scenario,
        "cue": boss3.cue,
        "trigger_lead": boss3.trigger_lead,
        "decision_interval": 3,
        "actions": [asdict(Action.from_discrete(action.discrete)) for action in boss3_old.actions],
        "source": {
            "kind": "prior_successful_flow",
            "source_artifact": "artifacts/route_memory_boss3.json",
            "source_sha256": file_sha256(ARTIFACTS / "route_memory_boss3.json"),
            "implementation_sha256": source_tree_sha256(),
            "route_origin": "episode",
        },
    }
    boss3_path.write_text(
        json.dumps(boss3_artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    library_artifact = {
        "schema_version": 1,
        "generator": str(Path(__file__).resolve().relative_to(ROOT)),
        "generator_sha256": file_sha256(Path(__file__)),
        "library_id": "sr-stage5-boss4-visible-signatures-v2",
        "scenario": "stage5_boss4:lunatic",
        "memory_ids": [memory.id for memory in boss4_memories],
        "cue_frames": cue_frames,
        "source": {
            "kind": "prior_successful_flows",
            "seeds": list(source_routes),
            "route_origin": "episode",
            "selection": "nearest delayed semantic signature",
            "implementation_sha256": source_tree_sha256(),
            "files": {
                "boss4_canonical_0.npz": file_sha256(ARTIFACTS / "boss4_canonical_0.npz"),
                "boss4_canonical_2.npz": file_sha256(ARTIFACTS / "boss4_canonical_2.npz"),
                "boss4_route_3008_visible_v2.npz": file_sha256(
                    ARTIFACTS / "boss4_route_3008_visible_v2.npz"
                ),
                "boss4_route_3015_visible_v2.npz": file_sha256(
                    ARTIFACTS / "boss4_route_3015_visible_v2.npz"
                ),
            },
        },
    }
    boss4_path.write_text(
        json.dumps(library_artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "database": str(DATABASE.relative_to(ROOT)),
        "database_sha256": file_sha256(DATABASE),
        "boss3_artifact": str(boss3_path.relative_to(ROOT)),
        "boss3_artifact_sha256": file_sha256(boss3_path),
        "boss4_artifact": str(boss4_path.relative_to(ROOT)),
        "boss4_artifact_sha256": file_sha256(boss4_path),
        "boss3_memory_id": boss3.id,
        "boss4_memory_ids": [memory.id for memory in boss4_memories],
    }, indent=2))


if __name__ == "__main__":
    main()
