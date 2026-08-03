from __future__ import annotations

from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

from stg_lab.engine_mpc import EngineMPC, MPCConfig, PredictedThreat
from stg_lab.protocol import Action


SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "benchmark_engine_mpc_grid.py"
SPEC = importlib.util.spec_from_file_location("benchmark_engine_mpc_grid", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)


def _observation() -> dict:
    return {
        "episode_frame": 0,
        "world": {"pl": -192.0, "pr": 192.0, "pb": -224.0, "pt": 224.0},
        "player": {
            "x": 0.0,
            "y": -176.0,
            "a": 0.5,
            "b": 0.5,
            "hspeed": 4.0,
            "lspeed": 2.0,
        },
        "enemy_bullets": [],
        "enemies": [],
        "nontjt_enemies": [],
        "indestructibles": [],
        "lasers": [],
    }


def test_grid_forecast_covers_every_logical_frame_in_each_layer() -> None:
    controller = EngineMPC(MPCConfig(observation_delay=0, horizon_frames=36))
    threat = PredictedThreat(
        key="enemy_bullets:1",
        source="enemy_bullets",
        object_id=1,
        x=0.0,
        y=0.0,
        vx=1.0,
        vy=0.0,
        radius=2.0,
        radius_rate=0.0,
        source_frame=0,
        observation_delay=0,
        radius_rate_horizon=0,
        motion_horizon=36,
    )

    forecast = benchmark._GridSource(controller, (threat,)).forecast_swept_threats(6, 3)

    assert [offset for offset, _ in forecast] == [3, 6]
    assert [[record["x"] for record in records] for _, records in forecast] == [
        [0.0, 1.0, 2.0, 3.0],
        [3.0, 4.0, 5.0, 6.0],
    ]
    assert all(
        record["source"] == "enemy_bullets"
        for _, records in forecast
        for record in records
    )


def test_sample_compares_beam_and_grids_over_one_common_horizon(monkeypatch) -> None:
    controller = EngineMPC(MPCConfig(observation_delay=0, horizon_frames=36))
    observation = _observation()
    decision = controller.select(observation)
    decision = replace(decision, planned_actions=decision.planned_actions[:2])
    actions = (Action(move_x=1, slow=True),) * 4
    steps = tuple(
        SimpleNamespace(position=(float(index), -176.0))
        for index in range(5)
    )
    result = SimpleNamespace(
        actions=actions,
        steps=steps,
        field=SimpleNamespace(risk=np.zeros((5, 2, 2))),
        peak_level=0,
        total_risk=0.0,
        reached_goal=False,
        first_action=actions[0],
    )

    monkeypatch.setattr(
        benchmark,
        "_grid_plan",
        lambda *_args, **_kwargs: (result, 0.01, (10.0, -176.0)),
    )
    sample = benchmark._sample(
        controller,
        observation,
        decision,
        {"source_frame": 0},
        (8.0,),
        0.02,
        "survival",
    )

    assert sample["comparison_action_count"] == 2
    assert sample["beam"]["metrics"]["evaluated_frames"] == 6
    assert all(grid["metrics"]["evaluated_frames"] == 6 for grid in sample["grids"])
    assert all(grid["terminal_position"] == [2.0, -176.0] for grid in sample["grids"])
