"""Replay teacher actions through the current human-visible observation pipeline."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import numpy as np

from stg_lab.protocol import Action
from stg_lab.provenance import file_sha256, source_tree_sha256
from stg_lab.scenarios import make_environment
from stg_lab.sim import Outcome, SimulationConfig
from stg_lab.training import Demonstrations
from stg_lab.vision import DelayedVision, VisionConfig


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
VISION = VisionConfig(history=4, observation_delay=5)
SIMULATION = SimulationConfig(reaction_frames=0, action_hold_frames=3)
DECISION_INTERVAL = 3


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"manifest is not an object: {path}")
    return value


def _episode_maps() -> tuple[dict[int, tuple[str, int]], dict[int, tuple[str, int]]]:
    canonical = _load_json(ARTIFACTS / "canonical_dataset_manifest.json")
    expanded = _load_json(ARTIFACTS / "canonical_dataset_expanded_manifest.json")

    def collect(sources: list[dict[str, Any]]) -> dict[int, tuple[str, int]]:
        result: dict[int, tuple[str, int]] = {}
        for source in sources:
            scenario = str(source["scenario"])
            seeds = tuple(int(value) for value in source["seeds"])
            episode_ids = tuple(int(value) for value in source["episode_ids"])
            if len(seeds) != len(episode_ids):
                raise ValueError("manifest seed and episode-id counts differ")
            for episode_id, seed in zip(episode_ids, seeds, strict=True):
                if episode_id in result:
                    raise ValueError(f"duplicate episode id {episode_id}")
                result[episode_id] = scenario, seed
        return result

    train_sources = [*canonical["train"]["sources"], *expanded["added_sources"]]
    heldout_sources = canonical["heldout"]["sources"]
    return collect(train_sources), collect(heldout_sources)


def _replay(
    input_path: Path,
    output_path: Path,
    episodes: dict[int, tuple[str, int]],
) -> dict[str, Any]:
    source = Demonstrations.load(input_path)
    if source.episode_ids is None:
        raise ValueError("source demonstrations have no episode ids")
    observed_ids = {int(value) for value in np.unique(source.episode_ids)}
    if observed_ids != set(episodes):
        raise ValueError("manifest episode ids do not match the demonstration archive")

    global_frames = np.zeros_like(source.global_frames, dtype=np.float16)
    local_frames = np.zeros_like(source.local_frames, dtype=np.float16)
    replayed: list[dict[str, Any]] = []
    for episode_id in sorted(episodes):
        scenario, seed = episodes[episode_id]
        indices = np.flatnonzero(source.episode_ids == episode_id)
        duration = 600 if scenario == "stage5_boss3" else 700
        environment = make_environment(
            scenario,
            difficulty="lunatic",
            seed=seed,
            config=SIMULATION,
            duration_frames=duration,
        )
        observation = environment.reset(seed=seed)
        vision = DelayedVision(config=VISION)
        visible = vision.reset(observation)
        for sample in indices:
            global_frames[sample] = visible.global_frames.astype(np.float16)
            local_frames[sample] = visible.local_frames.astype(np.float16)
            action = Action.from_discrete(int(source.actions[sample, -1]))
            for _ in range(DECISION_INTERVAL):
                if environment.done or environment.frame >= duration:
                    break
                result = environment._advance(
                    action,
                    build_semantic=False,
                    detect_collision=True,
                )
                observation = result.observation
                visible = vision.push(observation)
        survived = environment.outcome is not Outcome.HIT and environment.frame >= duration
        if not survived:
            raise RuntimeError(
                f"teacher replay failed for {scenario} seed {seed} at frame {environment.frame}"
            )
        replayed.append({
            "episode_id": episode_id,
            "scenario": scenario,
            "seed": seed,
            "samples": len(indices),
            "frames": environment.frame,
        })

    result = Demonstrations(
        global_frames=global_frames,
        local_frames=local_frames,
        actions=source.actions.copy(),
        risks=source.risks.copy(),
        memory=None if source.memory is None else source.memory.copy(),
        episode_ids=source.episode_ids.copy(),
        supervision_mask=(
            None if source.supervision_mask is None else source.supervision_mask.copy()
        ),
    )
    result.save(output_path)
    return {
        "input": str(input_path.relative_to(ROOT)),
        "input_sha256": file_sha256(input_path),
        "output": str(output_path.relative_to(ROOT)),
        "output_sha256": file_sha256(output_path),
        "shape": list(result.global_frames.shape),
        "episodes": replayed,
    }


def main() -> None:
    train_map, heldout_map = _episode_maps()
    outputs = [
        _replay(
            ARTIFACTS / "canonical_train_expanded.npz",
            ARTIFACTS / "canonical_train_visible_v2.npz",
            train_map,
        ),
        _replay(
            ARTIFACTS / "canonical_heldout_merged.npz",
            ARTIFACTS / "canonical_heldout_visible_v2.npz",
            heldout_map,
        ),
    ]
    manifest = {
        "schema_version": 1,
        "implementation_sha256": source_tree_sha256(),
        "observation_contract": "blank cold start; delayed visible displacement motion",
        "vision_config": asdict(VISION),
        "simulation_config": asdict(SIMULATION),
        "decision_interval": DECISION_INTERVAL,
        "outputs": outputs,
    }
    path = ARTIFACTS / "visible_dataset_v2_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**manifest, "manifest": str(path.relative_to(ROOT))}, indent=2))


if __name__ == "__main__":
    main()
