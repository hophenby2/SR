from __future__ import annotations

import stg_lab.engine_mpc as engine_mpc


class _UnfilteredEngineMPC(engine_mpc.EngineMPC):
    def _beam_evaluations(self, *args, **kwargs):
        kwargs["_prefilter_threats"] = False
        return super()._beam_evaluations(*args, **kwargs)


def _threat(index: int, *, nearby: bool) -> engine_mpc.PredictedThreat:
    if nearby:
        x = float((index % 5) * 12 - 24)
        y = float(24 + (index // 5) * 16)
        vx = float((index % 3) - 1) * 0.25
        vy = -1.5
    else:
        x = float((index % 25) * 24 - 288)
        y = float(700 + (index // 25) * 20)
        vx = 0.0
        vy = 0.0
    return engine_mpc.PredictedThreat(
        key=f"enemy_bullets:{index}",
        source="enemy_bullets",
        object_id=index,
        x=x,
        y=y,
        vx=vx,
        vy=vy,
        radius=2.0 + float(index % 3),
        radius_rate=0.0,
        source_frame=0,
        observation_delay=5,
        radius_rate_horizon=6,
        motion_horizon=60,
    )


def _run_counting_hypot(
    controller: engine_mpc.EngineMPC,
    threats: tuple[engine_mpc.PredictedThreat, ...],
    *,
    prefilter: bool,
):
    original_hypot = engine_mpc.np.hypot
    evaluated_pairs = 0

    def counting_hypot(x, y):
        nonlocal evaluated_pairs
        evaluated_pairs += int(engine_mpc.np.broadcast(x, y).size)
        return original_hypot(x, y)

    engine_mpc.np.hypot = counting_hypot
    try:
        result = controller._beam_evaluations(
            (0.0, 0.0, 0.5, 4.0, 2.0),
            (-192.0, 192.0, -224.0, 224.0),
            threats,
            0.0,
            _prefilter_threats=prefilter,
        )
    finally:
        engine_mpc.np.hypot = original_hypot
    return result, evaluated_pairs


def test_conservative_threat_prefilter_preserves_complete_beam_result() -> None:
    controller = engine_mpc.EngineMPC(engine_mpc.MPCConfig(
        horizon_frames=60,
        beam_width=32,
    ))
    threats = tuple(
        _threat(index, nearby=index < 10)
        for index in range(310)
    )

    reference, reference_pairs = _run_counting_hypot(
        controller,
        threats,
        prefilter=False,
    )
    optimized, optimized_pairs = _run_counting_hypot(
        controller,
        threats,
        prefilter=True,
    )

    # Equality covers all 17 candidate evaluations, every collision field,
    # exact float margins/penalties, and every action in each 20-segment plan.
    assert optimized == reference
    assert optimized_pairs < reference_pairs // 4


def test_prefilter_preserves_stateful_decisions_and_committed_plans() -> None:
    config = engine_mpc.MPCConfig(horizon_frames=36, beam_width=32)
    optimized = engine_mpc.EngineMPC(config)
    reference = _UnfilteredEngineMPC(config)

    for frame in (0, 3, 6):
        bullets = [
            {
                "id": index,
                "x": threat.x + threat.vx * frame,
                "y": threat.y + threat.vy * frame,
                "dx": threat.vx,
                "dy": threat.vy,
                "a": threat.radius,
                "b": threat.radius,
                "collidable": True,
            }
            for index in range(310)
            for threat in (_threat(index, nearby=index < 10),)
        ]
        observation = {
            "episode_frame": frame,
            "world": {"pl": -192.0, "pr": 192.0, "pb": -224.0, "pt": 224.0},
            "player": {
                "x": 0.0,
                "y": 0.0,
                "a": 0.5,
                "b": 0.5,
                "hspeed": 4.0,
                "lspeed": 2.0,
            },
            "enemy_bullets": bullets,
            "enemies": [],
            "nontjt_enemies": [],
            "indestructibles": [],
        }

        assert optimized.select(observation) == reference.select(observation)
