"""Extract the verified Boss #3 planner route into external episodic memory."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

from stg_lab.memory import EpisodicMemory
from stg_lab.route_benchmark import file_sha256
from stg_lab.sim import coerce_action


SOURCE = Path("artifacts/determinism_acceptance.json")
OUTPUT = Path("artifacts/route_memory_boss3.json")
DATABASE = Path("artifacts/episodic_memory.sqlite")


def main() -> None:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    comparison = next(
        item for item in source["comparisons"]
        if item["scenario"] == "stage5_boss3:lunatic"
    )
    interval = int(source["route_source"]["decision_interval"])
    expanded = tuple(coerce_action(item) for item in comparison["actions"])
    if len(expanded) % interval:
        raise ValueError("expanded route length is not divisible by decision interval")
    if any(
        expanded[start + offset] != expanded[start]
        for start in range(0, len(expanded), interval)
        for offset in range(interval)
    ):
        raise ValueError("expanded route changes inside an action-hold interval")
    actions = tuple(asdict(expanded[index]) for index in range(0, len(expanded), interval))
    cue = {
        "kind": "semantic_roi_mass",
        "channel": 0,
        "world_bounds": {"left": -192.0, "right": 192.0, "bottom": -224.0, "top": 224.0},
        "roi": {"left": -192.0, "right": 192.0, "bottom": 80.0, "top": 224.0},
        "minimum_mass": 0.5,
    }
    scenario = "stage5_boss3:lunatic"
    with EpisodicMemory(DATABASE) as store:
        memory = None
        for match in store.retrieve(scenario, cue, limit=100, minimum_similarity=1.0):
            candidate = match.memory
            if tuple(candidate.route) == actions and candidate.trigger_lead == 0:
                memory = candidate
                break
        if memory is None:
            memory = store.remember(
                scenario,
                cue,
                death_point=None,
                trigger_lead=0,
                route=actions,
                confidence=1.0,
            )

    artifact = {
        "schema_version": 1,
        "route_id": "sr-stage5-boss3-planner-seed3001-v1",
        "scenario": scenario,
        "cue": cue,
        "trigger_lead": 0,
        "decision_interval": interval,
        "actions": actions,
        "storage": {"database": str(DATABASE), "memory_id": memory.id},
        "source": {
            "artifact": str(SOURCE),
            "artifact_sha256": file_sha256(SOURCE),
            "seed": int(comparison["seed"]),
            "scenario": comparison["scenario"],
            "action_sequence_hash": comparison["action_sequence_hash"],
            "extraction": f"one action per {interval}-frame held block",
        },
    }
    OUTPUT.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "route_artifact": str(OUTPUT),
        "route_artifact_sha256": file_sha256(OUTPUT),
        "database": str(DATABASE),
        "memory_id": memory.id,
        "actions": len(actions),
    }, indent=2))


if __name__ == "__main__":
    main()
