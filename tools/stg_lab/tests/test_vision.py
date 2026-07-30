import numpy as np

from stg_lab.vision import DelayedVision, SemanticRasterizer, VisionConfig


def snapshot(frame: int, threat_x: float = 0.0) -> dict:
    return {
        "frame": frame,
        "bounds": (-192.0, 192.0, -224.0, 224.0),
        "player": {"x": 0.0, "y": -160.0, "radius": 3.0},
        "threats": [
            {"x": threat_x, "y": -120.0, "radius": 12.0, "vx": 2.0, "vy": -1.0},
            {"x": 40.0, "y": 80.0, "radius": 20.0, "warning": True},
        ],
    }


def test_semantic_raster_has_visible_channels() -> None:
    config = VisionConfig(history=3, observation_delay=2)
    global_frame, local_frame = SemanticRasterizer(config).render(snapshot(0))
    assert global_frame.shape == (6, config.global_height, config.global_width)
    assert local_frame.shape == (6, config.local_height, config.local_width)
    assert np.max(global_frame[0]) > 0.0
    assert np.max(global_frame[3]) > 0.0
    assert np.max(local_frame[4]) == 1.0


def test_delayed_history_reports_delayed_source_frame() -> None:
    config = VisionConfig(history=3, observation_delay=2)
    delayed = DelayedVision(config=config)
    observation = delayed.reset(snapshot(10))
    assert observation.source_frame == -2
    assert np.count_nonzero(observation.global_frames) == 0
    assert np.count_nonzero(observation.local_frames) == 0
    observation = delayed.push(snapshot(11, 11.0))
    assert observation.source_frame == -1
    assert np.count_nonzero(observation.global_frames) == 0
    observation = delayed.push(snapshot(12, 12.0))
    assert observation.source_frame == 0
    expected_global, expected_local = SemanticRasterizer(config).render(snapshot(10), motion={})
    np.testing.assert_array_equal(observation.global_frames[-1], expected_global)
    np.testing.assert_array_equal(observation.local_frames[-1], expected_local)
    for frame in range(13, 16):
        observation = delayed.push(snapshot(frame, float(frame)))
    assert observation.source_frame == 3
    assert observation.global_frames.shape[0] == config.history
    for frame in range(16, 24):
        observation = delayed.push(snapshot(9999 - frame, float(frame)))
    assert observation.source_frame == 11
    assert np.max(observation.global_frames[-1, 1]) > 0.0


def test_delayed_vision_estimates_motion_from_visible_displacement() -> None:
    config = VisionConfig(history=1, observation_delay=0)
    delayed = DelayedVision(config=config)
    first = snapshot(0, 0.0)
    first["threats"][0]["vx"] = 999.0
    observation = delayed.reset(first)
    assert np.count_nonzero(observation.global_frames[-1, 1:3]) == 0

    stationary = snapshot(1, 0.0)
    stationary["threats"][0]["vx"] = 999.0
    observation = delayed.push(stationary)
    assert np.count_nonzero(observation.global_frames[-1, 1:3]) == 0

    moved = snapshot(2, 8.0)
    moved["threats"][0]["vx"] = 0.0
    observation = delayed.push(moved)
    assert np.max(observation.global_frames[-1, 1]) > 0.0


def test_subpixel_player_and_bullet_remain_visible() -> None:
    config = VisionConfig(global_width=12, global_height=14)
    tiny = {
        "frame": 0,
        "bounds": (-192.0, 192.0, -224.0, 224.0),
        "player": {"x": 1.3, "y": -173.7, "radius": 0.5},
        "threats": [{"x": 3.1, "y": 7.7, "radius": 0.5}],
    }
    global_frame, _ = SemanticRasterizer(config).render(tiny)
    assert np.count_nonzero(global_frame[0]) == 1
    assert np.count_nonzero(global_frame[4]) == 1


def test_density_and_velocity_aggregation_is_order_independent() -> None:
    config = VisionConfig(global_width=24, global_height=28)
    base = snapshot(0)
    threats = [
        {"x": 0.0, "y": -120.0, "radius": 12.0, "vx": 8.0, "vy": 0.0},
        {"x": 0.0, "y": -120.0, "radius": 12.0, "vx": -8.0, "vy": 0.0},
    ]
    base["threats"] = threats
    forward, _ = SemanticRasterizer(config).render(base)
    base["threats"] = list(reversed(threats))
    reverse, _ = SemanticRasterizer(config).render(base)
    np.testing.assert_array_equal(forward, reverse)
    occupied = forward[0] > 0.0
    assert np.max(forward[0]) > np.log1p(1.0) / np.log1p(config.density_saturation)
    assert np.allclose(forward[1, occupied], 0.0)
