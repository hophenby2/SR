from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _pair(value: Any, name: str, fallback: tuple[float, float]) -> tuple[float, float]:
    result = _field(value, name, None)
    if result is not None:
        return float(result[0]), float(result[1])
    x = _field(value, name + "_x", fallback[0])
    y = _field(value, name + "_y", fallback[1])
    return float(x), float(y)


@dataclass(frozen=True, slots=True)
class VisionConfig:
    global_width: int = 48
    global_height: int = 56
    local_width: int = 40
    local_height: int = 40
    local_extent_x: float = 72.0
    local_extent_y: float = 72.0
    history: int = 4
    observation_delay: int = 5
    velocity_scale: float = 8.0
    density_saturation: float = 8.0
    channels: int = 6
    motion_estimation: str = "visible_displacement"

    def __post_init__(self) -> None:
        values = (
            self.global_width,
            self.global_height,
            self.local_width,
            self.local_height,
            self.history,
        )
        if any(value <= 0 for value in values):
            raise ValueError("vision sizes and history must be positive")
        if self.observation_delay < 0:
            raise ValueError("observation delay cannot be negative")
        numeric = (
            self.local_extent_x,
            self.local_extent_y,
            self.velocity_scale,
            self.density_saturation,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in numeric):
            raise ValueError("vision extents and scales must be finite and positive")
        if self.channels != 6:
            raise ValueError("the reference semantic format has six channels")
        if self.motion_estimation != "visible_displacement":
            raise ValueError("motion_estimation must be visible_displacement")


@dataclass(frozen=True, slots=True)
class VisionObservation:
    global_frames: np.ndarray
    local_frames: np.ndarray
    source_frame: int


class SemanticRasterizer:
    """Render visible geometry without exposing hidden script state.

    Channels are threat occupancy, horizontal motion, vertical motion, warning
    geometry, player position, and playfield boundary. Motion channels are
    observable quantities that can also be estimated from a frame stack.
    """

    def __init__(self, config: VisionConfig = VisionConfig()) -> None:
        self.config = config

    def render(
        self,
        snapshot: Any,
        *,
        motion: Mapping[Any, tuple[float, float]] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        bounds = self._bounds(snapshot)
        player = _field(snapshot, "player")
        if player is None:
            raise ValueError("snapshot is missing player")
        player_x = float(_field(player, "x", 0.0))
        player_y = float(_field(player, "y", 0.0))

        global_frame = self._render_region(
            snapshot,
            bounds,
            self.config.global_width,
            self.config.global_height,
            motion,
        )
        local_bounds = (
            player_x - self.config.local_extent_x,
            player_x + self.config.local_extent_x,
            player_y - self.config.local_extent_y,
            player_y + self.config.local_extent_y,
        )
        local_frame = self._render_region(
            snapshot,
            local_bounds,
            self.config.local_width,
            self.config.local_height,
            motion,
        )
        return global_frame, local_frame

    @staticmethod
    def _bounds(snapshot: Any) -> tuple[float, float, float, float]:
        bounds = _field(snapshot, "bounds", (-192.0, 192.0, -224.0, 224.0))
        if isinstance(bounds, Mapping):
            return (
                float(bounds["left"]),
                float(bounds["right"]),
                float(bounds["bottom"]),
                float(bounds["top"]),
            )
        return tuple(float(value) for value in bounds)  # type: ignore[return-value]

    @staticmethod
    def _threats(snapshot: Any) -> Iterable[Any]:
        threats = _field(snapshot, "threats", None)
        if threats is None:
            threats = _field(snapshot, "entities", ())
        return threats

    def _render_region(
        self,
        snapshot: Any,
        bounds: tuple[float, float, float, float],
        width: int,
        height: int,
        motion: Mapping[Any, tuple[float, float]] | None,
    ) -> np.ndarray:
        left, right, bottom, top = bounds
        if right <= left or top <= bottom:
            raise ValueError("invalid raster bounds")
        xs = np.linspace(left, right, width, dtype=np.float32)
        ys = np.linspace(bottom, top, height, dtype=np.float32)
        grid_x, grid_y = np.meshgrid(xs, ys)
        frame = np.zeros((self.config.channels, height, width), dtype=np.float32)
        density = np.zeros((height, width), dtype=np.float32)
        warning_density = np.zeros((height, width), dtype=np.float32)
        velocity_x_sum = np.zeros((height, width), dtype=np.float32)
        velocity_y_sum = np.zeros((height, width), dtype=np.float32)

        for ordinal, threat in enumerate(self._threats(snapshot)):
            if not bool(_field(threat, "visible", True)):
                continue
            x = float(_field(threat, "x", 0.0))
            y = float(_field(threat, "y", 0.0))
            radius = float(_field(threat, "radius", max(
                float(_field(threat, "a", 2.0)),
                float(_field(threat, "b", 2.0)),
            )))
            radius_x = max(0.5, float(_field(threat, "radius_x", radius)))
            radius_y = max(0.5, float(_field(threat, "radius_y", radius)))
            angle = float(_field(threat, "angle", _field(threat, "angle_radians", 0.0)))
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            dx, dy = grid_x - x, grid_y - y
            local_x = cos_a * dx + sin_a * dy
            local_y = -sin_a * dx + cos_a * dy
            distance = (local_x / radius_x) ** 2 + (local_y / radius_y) ** 2
            mask = distance <= 1.0
            if not np.any(mask):
                nearest_y, nearest_x = np.unravel_index(np.argmin(distance), distance.shape)
                mask[nearest_y, nearest_x] = True
            intensity = float(np.clip(_field(threat, "opacity", 1.0), 0.0, 1.0))
            density[mask] += intensity
            key = _field(threat, "id", ("ordinal", ordinal))
            velocity = motion.get(key, (0.0, 0.0)) if motion is not None else (
                float(_field(threat, "vx", 0.0)),
                float(_field(threat, "vy", 0.0)),
            )
            vx = float(velocity[0]) / self.config.velocity_scale
            vy = float(velocity[1]) / self.config.velocity_scale
            velocity_x_sum[mask] += intensity * np.clip(vx, -1.0, 1.0)
            velocity_y_sum[mask] += intensity * np.clip(vy, -1.0, 1.0)
            if bool(_field(threat, "warning", False)):
                warning_density[mask] += intensity

        denominator = math.log1p(self.config.density_saturation)
        occupied = density > 0.0
        frame[0] = np.clip(np.log1p(density) / denominator, 0.0, 1.0)
        frame[1, occupied] = velocity_x_sum[occupied] / density[occupied]
        frame[2, occupied] = velocity_y_sum[occupied] / density[occupied]
        frame[3] = np.clip(np.log1p(warning_density) / denominator, 0.0, 1.0)

        player = _field(snapshot, "player")
        px = float(_field(player, "x", 0.0))
        py = float(_field(player, "y", 0.0))
        pr = max(1.0, float(_field(player, "radius", 2.0)))
        player_mask = (grid_x - px) ** 2 + (grid_y - py) ** 2 <= pr * pr
        if not np.any(player_mask):
            distance = (grid_x - px) ** 2 + (grid_y - py) ** 2
            nearest_y, nearest_x = np.unravel_index(np.argmin(distance), distance.shape)
            player_mask[nearest_y, nearest_x] = True
        frame[4, player_mask] = 1.0

        play_left, play_right, play_bottom, play_top = self._bounds(snapshot)
        pixel_x = (right - left) / max(width - 1, 1)
        pixel_y = (top - bottom) / max(height - 1, 1)
        boundary_mask = (
            (grid_x <= play_left + pixel_x)
            | (grid_x >= play_right - pixel_x)
            | (grid_y <= play_bottom + pixel_y)
            | (grid_y >= play_top - pixel_y)
        )
        frame[5, boundary_mask] = 1.0
        return frame


class DelayedVision:
    def __init__(
        self,
        rasterizer: SemanticRasterizer | None = None,
        config: VisionConfig = VisionConfig(),
    ) -> None:
        self.config = config
        self.rasterizer = rasterizer or SemanticRasterizer(config)
        self._frames: deque[tuple[int, np.ndarray, np.ndarray]] = deque(
            maxlen=config.history + config.observation_delay,
        )
        self._last_positions: dict[Any, tuple[int, float, float]] = {}
        self._capture_index = 0

    def _visible_motion(
        self,
        snapshot: Any,
        capture_index: int,
    ) -> dict[Any, tuple[float, float]]:
        current: dict[Any, tuple[int, float, float]] = {}
        result: dict[Any, tuple[float, float]] = {}
        for ordinal, threat in enumerate(self.rasterizer._threats(snapshot)):
            if not bool(_field(threat, "visible", True)):
                continue
            key = _field(threat, "id", ("ordinal", ordinal))
            x = float(_field(threat, "x", 0.0))
            y = float(_field(threat, "y", 0.0))
            current[key] = (capture_index, x, y)
            previous = self._last_positions.get(key)
            if previous is None or capture_index <= previous[0]:
                result[key] = (0.0, 0.0)
            else:
                elapsed = capture_index - previous[0]
                result[key] = ((x - previous[1]) / elapsed, (y - previous[2]) / elapsed)
        self._last_positions = current
        return result

    def reset(self, snapshot: Any) -> VisionObservation:
        self._frames.clear()
        self._last_positions.clear()
        self._capture_index = 0
        global_frame, local_frame = self.rasterizer.render(
            snapshot,
            motion=self._visible_motion(snapshot, self._capture_index),
        )
        capacity = self._frames.maxlen or 1
        blank_global = np.zeros_like(global_frame)
        blank_local = np.zeros_like(local_frame)
        # Frames before reset were never observed.  Keeping them blank enforces
        # the configured cold-start delay instead of leaking the reset frame
        # backward through the synthetic history.
        for source_frame in range(-capacity + 1, 0):
            self._frames.append((source_frame, blank_global.copy(), blank_local.copy()))
        self._frames.append((self._capture_index, global_frame, local_frame))
        return self.observe()

    def push(self, snapshot: Any) -> VisionObservation:
        self._capture_index += 1
        global_frame, local_frame = self.rasterizer.render(
            snapshot,
            motion=self._visible_motion(snapshot, self._capture_index),
        )
        self._frames.append((self._capture_index, global_frame, local_frame))
        return self.observe()

    def observe(self) -> VisionObservation:
        required = self.config.history + self.config.observation_delay
        if len(self._frames) < required:
            raise RuntimeError("vision history is not initialized")
        end = len(self._frames) - self.config.observation_delay
        start = end - self.config.history
        selected: Sequence[tuple[int, np.ndarray, np.ndarray]] = list(self._frames)[start:end]
        return VisionObservation(
            global_frames=np.stack([item[1] for item in selected], axis=0),
            local_frames=np.stack([item[2] for item in selected], axis=0),
            source_frame=selected[-1][0],
        )
