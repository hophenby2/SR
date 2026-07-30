import math

from stg_lab.adapters import adapt_engine_observation
from stg_lab.vision import SemanticRasterizer, VisionConfig


def test_engine_adapter_converts_units_bounds_and_strips_authority_fields() -> None:
    raw = {
        "observation": {
            "frame": 90,
            "episode_frame": 12,
            "world": {"l": -192, "r": 192, "b": -224, "t": 224,
                      "pl": -192, "pr": 192, "pb": -224, "pt": 224},
            "stage": {"timer": 999, "card_index": 4},
            "player": {"x": 2, "y": -176, "a": 0.5, "b": 0.5,
                       "hspeed": 4, "lspeed": 2, "death": 0},
            "enemy_bullets": [{
                "id": 7, "x": 10, "y": 20, "a": 4.5, "b": 4.5,
                "rot": 90, "omiga": 3, "vx": 1, "vy": -2,
                "collidable": True, "class_name": "secret_class", "timer": 44,
            }],
            "enemies": [],
            "nontjt_enemies": [],
            "indestructibles": [],
            "lasers": [],
            "resources": {"lifeleft": 7},
        }
    }
    adapted = adapt_engine_observation(raw)
    assert "frame" not in adapted
    assert adapted["player_bounds"] == (-184.0, 184.0, -208.0, 192.0)
    assert adapted["player"]["radius"] == 0.5
    threat = adapted["threats"][0]
    assert math.isclose(threat["angle"], math.pi / 2.0)
    assert math.isclose(threat["angular_velocity"], math.radians(3.0))
    assert "class_name" not in threat and "timer" not in threat
    assert all(name not in threat for name in ("vx", "vy", "ax", "ay"))
    assert "stage" not in adapted and "resources" not in adapted

    frame, _ = SemanticRasterizer(VisionConfig(global_width=16, global_height=16)).render(adapted)
    assert frame.shape == (6, 16, 16)
    assert frame[0].max() > 0.0


def test_engine_adapter_converts_straight_laser_to_visible_geometry() -> None:
    adapted = adapt_engine_observation({
        "world": {},
        "player": {},
        "enemy_bullets": [],
        "enemies": [],
        "nontjt_enemies": [],
        "indestructibles": [],
        "lasers": [{
            "id": 2, "kind": "straight_laser", "x": 0, "y": 0,
            "rot": 0, "l1": 10, "l2": 20, "l3": 30, "w": 8,
            "collidable": True,
        }],
    })
    laser = adapted["threats"][0]
    assert laser["x"] == 30.0 and laser["y"] == 0.0
    assert laser["radius_x"] == 30.0 and laser["radius_y"] == 4.0
