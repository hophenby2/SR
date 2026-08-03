from __future__ import annotations

import numpy as np

from stg_lab.engine_vision import EngineStreamVision, controller_observation
from stg_lab.vision import VisionConfig


def observation(frame: int, *, player_x: float, bullet_x: float) -> dict:
    return {
        "episode_frame": frame,
        "performance": {"native_fps": 60.0},
        "world": {"pl": -192.0, "pr": 192.0, "pb": -224.0, "pt": 224.0},
        "player": {"x": player_x, "y": -176.0, "a": 0.5, "b": 0.5},
        "enemy_bullets": [{
            "id": 7,
            "x": bullet_x,
            "y": -100.0,
            "a": 3.0,
            "b": 3.0,
            "collidable": True,
        }],
        "enemies": [],
        "nontjt_enemies": [],
        "indestructibles": [],
        "lasers": [],
    }


def test_controller_observation_keeps_current_player_and_delays_hazards() -> None:
    delayed = observation(10, player_x=-40.0, bullet_x=12.0)
    current = observation(15, player_x=24.0, bullet_x=-12.0)

    result = controller_observation(delayed, current)

    assert result["episode_frame"] == 10
    assert result["enemy_bullets"] == delayed["enemy_bullets"]
    assert result["player"]["x"] == 24.0
    assert result["own_player_observation_frame"] == 15
    assert "performance" not in result


def test_stream_vision_delays_bullets_but_renders_current_player_once() -> None:
    config = VisionConfig(
        global_width=24,
        global_height=28,
        local_width=20,
        local_height=20,
        history=4,
        observation_delay=2,
    )
    vision = EngineStreamVision(config)
    initial = vision.reset(observation(0, player_x=0.0, bullet_x=-30.0))
    assert initial.global_frames.shape == (1, 6, 28, 24)
    vision.push(observation(1, player_x=10.0, bullet_x=-20.0))
    vision.push(observation(2, player_x=20.0, bullet_x=-10.0))
    vision.push(observation(3, player_x=30.0, bullet_x=0.0))

    visible = vision.observe()

    assert visible.source_frame == 1
    assert visible.global_frames.shape == (1, 6, 28, 24)
    assert visible.local_frames.shape == (1, 6, 20, 20)
    assert np.max(visible.global_frames[0, 1]) > 0.0
    raw = vision.raw_observation()
    assert raw["enemy_bullets"][0]["x"] == -20.0
    assert raw["player"]["x"] == 30.0
