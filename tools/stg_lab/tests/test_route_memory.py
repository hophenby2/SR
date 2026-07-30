from __future__ import annotations

from dataclasses import asdict, replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

from stg_lab.memory import EpisodeMemory, EpisodicMemory
from stg_lab.protocol import Action
from stg_lab.route_memory import (
    ExternalRouteController,
    ExternalRouteLibraryController,
    RouteControllerConfig,
    load_route_artifact,
    semantic_cue_matches,
    semantic_signature,
    validate_memory_route,
)
from stg_lab.route_benchmark import file_sha256, run_route_benchmark
from stg_lab.rollout import RolloutConfig
from stg_lab.sim import Bounds, SimulationConfig, STGEnvironment
from stg_lab.vision import VisionObservation


CUE = {
    "kind": "semantic_roi_mass",
    "channel": 0,
    "world_bounds": {"left": -192.0, "right": 192.0, "bottom": -224.0, "top": 224.0},
    "roi": {"left": -192.0, "right": 192.0, "bottom": 80.0, "top": 224.0},
    "minimum_mass": 0.5,
}


def visible(*, cue: bool) -> VisionObservation:
    global_frames = np.zeros((4, 6, 56, 48), dtype=np.float32)
    if cue:
        global_frames[-1, 0, 50, 20] = 0.75
    return VisionObservation(
        global_frames,
        np.zeros((4, 6, 40, 40), dtype=np.float32),
        source_frame=-999,
    )


def memory(*actions: Action) -> EpisodeMemory:
    return EpisodeMemory(
        id=7,
        scenario="stage5_boss3:lunatic",
        cue=CUE,
        death_point=None,
        trigger_lead=0,
        route=tuple(asdict(action) for action in actions),
        confidence=1.0,
        successes=0,
        failures=0,
        revision=0,
    )


def test_semantic_route_cue_uses_only_delayed_raster() -> None:
    assert not semantic_cue_matches(CUE, visible(cue=False))
    assert semantic_cue_matches(CUE, visible(cue=True))
    # source_frame is deliberately nonsensical; it cannot affect matching.
    changed = VisionObservation(visible(cue=True).global_frames, visible(cue=True).local_frames, 10**9)
    assert semantic_cue_matches(CUE, changed)


def test_route_sequence_reset_and_hold_last_exhaustion() -> None:
    route = memory(Action(move_x=1), Action(move_y=1, slow=True))
    controller = ExternalRouteController(
        route,
        config=RouteControllerConfig(shield=False, route_origin="trigger"),
    )
    opaque_environment = SimpleNamespace(hidden_timer=object())

    assert controller.select(visible(cue=False), environment=opaque_environment) == Action()
    assert controller.select(visible(cue=True), environment=opaque_environment) == Action(move_x=1)
    assert controller.select(visible(cue=True), environment=opaque_environment) == Action(move_y=1, slow=True)
    assert controller.select(visible(cue=True), environment=opaque_environment) == Action(move_y=1, slow=True)
    controller.reset()
    assert controller.select(visible(cue=True), environment=opaque_environment) == Action(move_x=1)


def test_route_exhaustion_can_be_neutral_or_strict() -> None:
    route = memory(Action(move_x=-1))
    neutral = ExternalRouteController(
        route, config=RouteControllerConfig(shield=False, exhaustion="neutral"),
    )
    assert neutral.select(visible(cue=True)) == Action(move_x=-1)
    assert neutral.select(visible(cue=True)) == Action()

    strict = ExternalRouteController(
        route, config=RouteControllerConfig(shield=False, exhaustion="error"),
    )
    assert strict.select(visible(cue=True)) == Action(move_x=-1)
    with pytest.raises(RuntimeError, match="exhausted"):
        strict.select(visible(cue=True))


def test_episode_route_alignment_uses_local_decision_count_after_delayed_cue() -> None:
    route = memory(Action(move_x=-1), Action(), Action(move_x=1))
    controller = ExternalRouteController(route, config=RouteControllerConfig(shield=False))

    assert controller.select(visible(cue=False)) == Action()
    assert controller.select(visible(cue=False)) == Action()
    assert controller.select(visible(cue=True)) == Action(move_x=1)
    assert controller.decision_index == 3


def test_route_library_selects_nearest_delayed_semantic_signature() -> None:
    left = visible(cue=False)
    left.global_frames[-1, 0, 20, 8] = 1.0
    right = visible(cue=False)
    right.global_frames[-1, 0, 20, 40] = 1.0
    base = {
        "kind": "semantic_signature",
        "trigger_channel": 0,
        "minimum_mass": 0.5,
        "channels": [0],
        "pooled_height": 14,
        "pooled_width": 12,
    }
    left_cue = {**base, "vector": semantic_signature(base, left).tolist()}
    right_cue = {**base, "vector": semantic_signature(base, right).tolist()}
    left_memory = replace(memory(Action(move_x=-1)), id=1, cue=left_cue)
    right_memory = replace(memory(Action(move_x=1)), id=2, cue=right_cue)
    controller = ExternalRouteLibraryController(
        (left_memory, right_memory),
        config=RouteControllerConfig(shield=False, route_origin="trigger"),
    )

    assert controller.select(right) == Action(move_x=1)
    assert controller.selected_memory is right_memory


class RightHazard:
    name = "right_hazard"
    duration_frames = 8
    forecast_independent_of_player = True

    def reset(self, environment: STGEnvironment) -> None:
        environment.spawn_circle(8.0, 0.0, 1.0, remove_outside=False)

    def update(self, _environment: STGEnvironment) -> None:
        pass


def test_route_uses_configured_short_toward_shield() -> None:
    environment = STGEnvironment(
        RightHazard(),
        seed=1,
        config=SimulationConfig(
            bounds=Bounds(-64.0, 64.0, -64.0, 64.0),
            player_start=(0.0, 0.0),
            reaction_frames=0,
            action_hold_frames=1,
            semantic_width=8,
            semantic_height=8,
        ),
    )
    environment.reset(seed=1)
    preferred = Action(move_x=1)
    controller = ExternalRouteController(
        memory(preferred),
        config=RouteControllerConfig(shield=True, shield_horizon=4),
    )
    selected = controller.select(visible(cue=True), environment=environment)
    assert selected != preferred
    assert controller.overrides == 1


def test_artifact_and_sqlite_memory_must_describe_the_same_route(tmp_path) -> None:
    actions = (Action(move_x=1), Action(move_y=-1, slow=True))
    path = tmp_path / "route.json"
    path.write_text(json.dumps({
        "schema_version": 1,
        "route_id": "test-route",
        "scenario": "stage5_boss3:lunatic",
        "cue": CUE,
        "trigger_lead": 0,
        "decision_interval": 3,
        "actions": [asdict(action) for action in actions],
        "source": {"kind": "test"},
    }), encoding="utf-8")
    artifact = load_route_artifact(path)
    validate_memory_route(artifact, memory(*actions))
    with pytest.raises(ValueError, match="actions differ"):
        validate_memory_route(artifact, memory(Action()))


def test_same_external_route_emits_same_actions_for_distinct_runs() -> None:
    route = memory(Action(move_x=1), Action(move_y=-1), Action(slow=True))
    outputs = []
    for _seed in (11, 97):
        controller = ExternalRouteController(route, config=RouteControllerConfig(shield=False))
        outputs.append(tuple(controller.select(visible(cue=True)).discrete for _ in range(3)))
    assert outputs[0] == outputs[1]


class VisibleCueScenario:
    name = "stage5_boss3"
    scenario_key = "stage5_boss3:lunatic"
    duration_frames = 6
    forecast_independent_of_player = True

    def reset(self, environment: STGEnvironment) -> None:
        environment.spawn_circle(0.0, 100.0, 8.0, lethal=False, remove_outside=False)

    def update(self, _environment: STGEnvironment) -> None:
        pass


def test_route_benchmark_emits_standard_visual_schema(tmp_path, monkeypatch) -> None:
    route_path = tmp_path / "route.json"
    route_path.write_text("{}", encoding="utf-8")
    checkpoint = tmp_path / "policy.pt"
    checkpoint.write_bytes(b"bound-policy")
    memory_database = tmp_path / "memory.sqlite"
    with EpisodicMemory(memory_database):
        pass

    def make_test_environment(_scenario, *, seed, config, **_kwargs):
        return STGEnvironment(VisibleCueScenario(), seed=seed, config=config)

    monkeypatch.setattr("stg_lab.route_benchmark.make_environment", make_test_environment)
    report = run_route_benchmark(
        "stage5_boss3",
        "lunatic",
        (31, 32),
        memory=memory(Action(), Action()),
        route_artifact=route_path,
        memory_database=memory_database,
        checkpoint=checkpoint,
        checkpoint_metadata={"version": 1, "policy_config": {}, "epochs": 0},
        route_config=RouteControllerConfig(shield=False),
        rollout_config=RolloutConfig(decision_interval=3, max_frames=6),
        simulation_config=SimulationConfig(action_hold_frames=3),
        workers=1,
    )
    assert report["controller_kind"] == "external_route_memory"
    assert report["checkpoint_metadata"]["role"] == "system_checkpoint_reference"
    assert report["checkpoint_metadata"]["policy_actions_used"] is False
    assert report["controller_stack"] == (
        "delayed_semantic_cue",
        "sqlite_external_route",
    )
    assert report["checkpoint_metadata"]["sha256"] == file_sha256(checkpoint)
    assert report["route_memory"]["artifact_sha256"] == file_sha256(route_path)
    assert report["scenarios"]["stage5_boss3"]["survival_rate"] == 1.0
    assert [item.seed for item in report["scenarios"]["stage5_boss3"]["episodes"]] == [31, 32]
