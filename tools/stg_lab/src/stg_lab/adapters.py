"""Adapters that separate LuaSTG authority telemetry from policy-visible geometry."""

from __future__ import annotations

import math
from typing import Any, Mapping


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _world_bounds(world: Mapping[str, Any]) -> tuple[float, float, float, float]:
    return (
        _number(world.get("l"), -192.0),
        _number(world.get("r"), 192.0),
        _number(world.get("b"), -224.0),
        _number(world.get("t"), 224.0),
    )


def _player_bounds(world: Mapping[str, Any]) -> tuple[float, float, float, float]:
    return (
        _number(world.get("pl", world.get("l")), -192.0) + 8.0,
        _number(world.get("pr", world.get("r")), 192.0) - 8.0,
        _number(world.get("pb", world.get("b")), -224.0) + 16.0,
        _number(world.get("pt", world.get("t")), 224.0) - 32.0,
    )


def _opacity(record: Mapping[str, Any]) -> float:
    alpha = _number(record.get("alpha"), 255.0)
    return min(max(alpha / 255.0 if alpha > 1.0 else alpha, 0.0), 1.0)


def _object_geometry(
    record: Mapping[str, Any],
    *,
    source: str,
    ordinal: int,
) -> dict[str, Any]:
    radius_x = max(0.1, abs(_number(record.get("a"), 2.0)))
    radius_y = max(0.1, abs(_number(record.get("b"), radius_x)))
    return {
        "id": f"{source}:{record.get('id', ordinal)}",
        "x": _number(record.get("x")),
        "y": _number(record.get("y")),
        # Motion is estimated later from successive visible positions.  Raw
        # engine velocity/acceleration telemetry is deliberately discarded.
        "radius_x": radius_x,
        "radius_y": radius_y,
        "angle": math.radians(_number(record.get("rot"))),
        "angular_velocity": math.radians(_number(record.get("omiga"))),
        "lethal": bool(record.get("collidable", True)),
        "visible": True,
        "warning": False,
        "opacity": _opacity(record),
        "danger": 1.0,
        "uncertainty": 0.0,
        "tag": "visible_threat",
        "metadata": {},
    }


def _laser_geometry(record: Mapping[str, Any], ordinal: int) -> list[dict[str, Any]]:
    kind = record.get("kind")
    width = max(0.5, abs(_number(record.get("w", record.get("w0")), 2.0)))
    if kind == "bent_laser":
        points = tuple(record.get("points") or ())
        result: list[dict[str, Any]] = []
        for index, (first, second) in enumerate(zip(points, points[1:])):
            x1, y1 = _number(first.get("x")), _number(first.get("y"))
            x2, y2 = _number(second.get("x")), _number(second.get("y"))
            length = math.hypot(x2 - x1, y2 - y1)
            if length <= 0.0:
                continue
            result.append({
                "id": f"laser:{record.get('id', ordinal)}:{index}",
                "x": (x1 + x2) * 0.5,
                "y": (y1 + y2) * 0.5,
                "vx": 0.0,
                "vy": 0.0,
                "radius_x": length * 0.5,
                "radius_y": width * 0.5,
                "angle": math.atan2(y2 - y1, x2 - x1),
                "lethal": bool(record.get("collidable", True)),
                "visible": True,
                "warning": False,
                "opacity": _opacity(record),
                "danger": 1.0,
                "uncertainty": 0.0,
                "tag": "visible_laser",
                "metadata": {},
            })
        return result

    angle = math.radians(_number(record.get("rot")))
    length = max(0.1, sum(_number(record.get(name)) for name in ("l1", "l2", "l3")))
    if length <= 0.1:
        length = max(0.1, abs(_number(record.get("l"), 0.1)))
    x, y = _number(record.get("x")), _number(record.get("y"))
    return [{
        "id": f"laser:{record.get('id', ordinal)}",
        "x": x + math.cos(angle) * length * 0.5,
        "y": y + math.sin(angle) * length * 0.5,
        "vx": 0.0,
        "vy": 0.0,
        "radius_x": length * 0.5,
        "radius_y": width * 0.5,
        "angle": angle,
        "lethal": bool(record.get("collidable", True)),
        "visible": True,
        "warning": False,
        "opacity": _opacity(record),
        "danger": 1.0,
        "uncertainty": 0.0,
        "tag": "visible_laser",
        "metadata": {},
    }]


def adapt_engine_observation(message: Mapping[str, Any]) -> dict[str, Any]:
    """Return policy-visible geometry from a bridge response or observation.

    Class names, image ids, script timers, card indices, resources, RNG state,
    and other authority-only fields are intentionally not copied.
    """

    observation = message.get("observation", message)
    if not isinstance(observation, Mapping):
        raise ValueError("engine message does not contain an observation object")
    world = observation.get("world") or {}
    if not isinstance(world, Mapping):
        raise ValueError("engine observation world must be an object")
    player_record = observation.get("player") or {}
    if not isinstance(player_record, Mapping):
        raise ValueError("engine observation player must be an object")

    radius = max(
        0.1,
        abs(_number(player_record.get("a"), 0.5)),
        abs(_number(player_record.get("b"), 0.5)),
    )
    player = {
        "x": _number(player_record.get("x")),
        "y": _number(player_record.get("y"), -176.0),
        "radius": radius,
        "speed": max(0.1, _number(player_record.get("hspeed"), 4.0)),
        "focus_speed": max(0.1, _number(player_record.get("lspeed"), 2.0)),
        "alive": _number(player_record.get("death"), 0.0) <= 0.0,
    }

    threats: list[dict[str, Any]] = []
    for source in ("enemy_bullets", "enemies", "nontjt_enemies", "indestructibles"):
        records = observation.get(source) or ()
        for ordinal, record in enumerate(records):
            if not isinstance(record, Mapping) or record.get("kind") in {"straight_laser", "bent_laser"}:
                continue
            threats.append(_object_geometry(record, source=source, ordinal=ordinal))
    for ordinal, record in enumerate(observation.get("lasers") or ()):
        if isinstance(record, Mapping):
            threats.extend(_laser_geometry(record, ordinal))

    return {
        "bounds": _world_bounds(world),
        "player_bounds": _player_bounds(world),
        "player": player,
        "threats": tuple(threats),
        "events": (),
    }


__all__ = ["adapt_engine_observation"]
