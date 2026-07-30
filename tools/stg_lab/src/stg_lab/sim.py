"""Deterministic, engine-independent 2D simulation primitives for STG Lab.

The simulator deliberately uses LuaSTG's logical coordinate system and
per-frame velocity convention.  One call to :meth:`STGEnvironment.step` is
exactly one 60 Hz game frame.  It is not intended to reproduce every LuaSTG
object rule; it provides a small, deterministic authority for planners,
automated content checks, and policy pre-training.
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Iterable, Iterator, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from .protocol import Action


MetadataValue = str | int | float | bool | None


@dataclass(frozen=True, slots=True)
class Bounds:
    """Logical playfield bounds in LuaSTG order."""

    left: float = -192.0
    right: float = 192.0
    bottom: float = -224.0
    top: float = 224.0

    def __post_init__(self) -> None:
        if not all(math.isfinite(value) for value in (self.left, self.right, self.bottom, self.top)):
            raise ValueError("bounds must be finite")
        if self.left >= self.right or self.bottom >= self.top:
            raise ValueError("bounds must have positive width and height")

    def __iter__(self) -> Iterator[float]:
        # Vision helpers also accept the historic four-value tuple form.
        return iter((self.left, self.right, self.bottom, self.top))

    @property
    def xmin(self) -> float:
        return self.left

    @property
    def xmax(self) -> float:
        return self.right

    @property
    def ymin(self) -> float:
        return self.bottom

    @property
    def ymax(self) -> float:
        return self.top

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.top - self.bottom

    def clamp(self, x: float, y: float, radius: float = 0.0) -> tuple[float, float]:
        radius = max(0.0, radius)
        return (
            min(max(x, self.left + radius), self.right - radius),
            min(max(y, self.bottom + radius), self.top - radius),
        )


def _point_segment_distance_sq(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    segment = end - start
    length_sq = float(np.dot(segment, segment))
    if length_sq <= 1e-18:
        delta = point - start
        return float(np.dot(delta, delta))
    ratio = float(np.clip(np.dot(point - start, segment) / length_sq, 0.0, 1.0))
    delta = point - (start + ratio * segment)
    return float(np.dot(delta, delta))


def _origin_segment_distance_sq(x0: float, y0: float, x1: float, y1: float) -> float:
    dx, dy = x1 - x0, y1 - y0
    length_sq = dx * dx + dy * dy
    if length_sq <= 1e-18:
        return x0 * x0 + y0 * y0
    ratio = min(max(-(x0 * dx + y0 * dy) / length_sq, 0.0), 1.0)
    closest_x = x0 + ratio * dx
    closest_y = y0 + ratio * dy
    return closest_x * closest_x + closest_y * closest_y


def _cross(first: np.ndarray, second: np.ndarray) -> float:
    return float(first[0] * second[1] - first[1] * second[0])


def _segments_intersect(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
) -> bool:
    first = first_end - first_start
    second = second_end - second_start
    if float(np.dot(first, first)) <= 1e-18 or float(np.dot(second, second)) <= 1e-18:
        return False
    denominator = _cross(first, second)
    delta = second_start - first_start
    if abs(denominator) <= 1e-12:
        if abs(_cross(delta, first)) > 1e-12:
            return False
        axis = int(abs(first[1]) > abs(first[0]))
        first_low, first_high = sorted((float(first_start[axis]), float(first_end[axis])))
        second_low, second_high = sorted((float(second_start[axis]), float(second_end[axis])))
        return max(first_low, second_low) <= min(first_high, second_high) + 1e-12
    along_first = _cross(delta, second) / denominator
    along_second = _cross(delta, first) / denominator
    return -1e-12 <= along_first <= 1.0 + 1e-12 and -1e-12 <= along_second <= 1.0 + 1e-12


def _segment_distance_sq(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
) -> float:
    if _segments_intersect(first_start, first_end, second_start, second_end):
        return 0.0
    return min(
        _point_segment_distance_sq(first_start, second_start, second_end),
        _point_segment_distance_sq(first_end, second_start, second_end),
        _point_segment_distance_sq(second_start, first_start, first_end),
        _point_segment_distance_sq(second_end, first_start, first_end),
    )


def _inside_ellipse(point: np.ndarray, radius_x: float, radius_y: float) -> bool:
    return (point[0] / radius_x) ** 2 + (point[1] / radius_y) ** 2 <= 1.0


@dataclass(slots=True)
class PlayerState:
    x: float = 0.0
    y: float = -176.0
    radius: float = 0.5
    speed: float = 4.0
    focus_speed: float = 2.0
    alive: bool = True

    def snapshot(self) -> "PlayerObservation":
        return PlayerObservation(
            x=self.x,
            y=self.y,
            radius=self.radius,
            speed=self.speed,
            focus_speed=self.focus_speed,
            alive=self.alive,
        )


@dataclass(frozen=True, slots=True)
class PlayerObservation:
    x: float
    y: float
    radius: float
    speed: float
    focus_speed: float
    alive: bool


@dataclass(slots=True)
class Threat:
    """A moving circular or rotated elliptical collision primitive.

    Velocities and accelerations use logical units per frame.  ``managed``
    threats have their position set by their scenario (for example an orbiting
    emitter), while their age and lifetime are still maintained by the core.
    """

    x: float
    y: float
    a: float
    b: float
    vx: float = 0.0
    vy: float = 0.0
    ax: float = 0.0
    ay: float = 0.0
    # Standalone geometry always uses radians.  The LuaSTG bridge converts its
    # degree-valued rot/omiga fields at the adapter boundary.
    angle: float = 0.0
    angular_velocity: float = 0.0
    lifetime: int | None = None
    lethal: bool = True
    visible: bool = True
    warning: bool = False
    opacity: float = 1.0
    danger: float = 1.0
    uncertainty: float = 0.0
    tag: str = "bullet"
    source_id: int | None = None
    managed: bool = False
    remove_outside: bool = True
    metadata: dict[str, MetadataValue] = field(default_factory=dict)
    id: int = -1
    age: int = 0
    prev_x: float = field(init=False)
    prev_y: float = field(init=False)
    prev_a: float = field(init=False)
    prev_b: float = field(init=False)
    prev_angle: float = field(init=False)

    def __post_init__(self) -> None:
        self.x = float(self.x)
        self.y = float(self.y)
        self.a = float(self.a)
        self.b = float(self.b)
        numeric = (
            self.x, self.y, self.a, self.b, self.vx, self.vy, self.ax, self.ay,
            self.angle, self.angular_velocity, self.opacity, self.danger, self.uncertainty,
        )
        if not all(math.isfinite(float(value)) for value in numeric):
            raise ValueError("threat geometry and motion must be finite")
        if self.a <= 0.0 or self.b <= 0.0:
            raise ValueError("threat radii must be positive")
        if self.lifetime is not None and self.lifetime <= 0:
            raise ValueError("threat lifetime must be positive or None")
        if not 0.0 <= self.opacity <= 1.0:
            raise ValueError("threat opacity must be in [0, 1]")
        if self.danger < 0.0 or self.uncertainty < 0.0:
            raise ValueError("threat danger and uncertainty cannot be negative")
        self.prev_x = self.x
        self.prev_y = self.y
        self.prev_a = self.a
        self.prev_b = self.b
        self.prev_angle = self.angle

    @property
    def radius(self) -> float:
        return max(self.a, self.b)

    @property
    def radius_x(self) -> float:
        return self.a

    @property
    def radius_y(self) -> float:
        return self.b

    @property
    def weight(self) -> float:
        return self.danger

    @property
    def expired(self) -> bool:
        return self.lifetime is not None and self.age >= self.lifetime

    def begin_frame(self) -> None:
        self.prev_x = self.x
        self.prev_y = self.y
        self.prev_a = self.a
        self.prev_b = self.b
        self.prev_angle = self.angle

    def advance(self) -> None:
        if not self.managed:
            self.vx += self.ax
            self.vy += self.ay
            self.x += self.vx
            self.y += self.vy
            self.angle += self.angular_velocity
        self.age += 1

    def is_outside(self, bounds: Bounds, margin: float = 32.0) -> bool:
        extent = max(self.a, self.b)
        return (
            self.x + extent < bounds.left - margin
            or self.x - extent > bounds.right + margin
            or self.y + extent < bounds.bottom - margin
            or self.y - extent > bounds.top + margin
        )

    def collides_swept(
        self,
        player_start: tuple[float, float],
        player_end: tuple[float, float],
        player_radius: float,
    ) -> bool:
        """Conservatively test relative swept motion against this ellipse.

        Adding the player radius independently to both ellipse axes is an
        under-approximation of the true Minkowski sum near diagonal normals.
        For a fixed ellipse, use the Euclidean distance to an inscribed
        polygon plus its maximum chord error.  Shape/rotation changes fall
        back to a conservative bounding circle.
        """

        start_x = player_start[0] - self.prev_x
        start_y = player_start[1] - self.prev_y
        end_x = player_end[0] - self.x
        end_y = player_end[1] - self.y
        shape_changed = (
            abs(self.a - self.prev_a) > 1e-9
            or abs(self.b - self.prev_b) > 1e-9
            or abs(self.angle - self.prev_angle) > 1e-9
        )
        if shape_changed:
            radius = max(self.a, self.b, self.prev_a, self.prev_b) + player_radius
            return _origin_segment_distance_sq(start_x, start_y, end_x, end_y) <= radius * radius

        if abs(self.a - self.b) <= 1e-9:
            radius = self.a + player_radius
            return _origin_segment_distance_sq(start_x, start_y, end_x, end_y) <= radius * radius

        relative_start = np.asarray((start_x, start_y), dtype=np.float64)
        relative_end = np.asarray((end_x, end_y), dtype=np.float64)

        cos_a = math.cos(self.angle)
        sin_a = math.sin(self.angle)

        def local(point: np.ndarray) -> np.ndarray:
            return np.asarray(
                (
                    cos_a * point[0] + sin_a * point[1],
                    -sin_a * point[0] + cos_a * point[1],
                ),
                dtype=np.float64,
            )

        start = local(relative_start)
        end = local(relative_end)
        if _inside_ellipse(start, self.a, self.b) or _inside_ellipse(end, self.a, self.b):
            return True

        vertex_count = 48
        angles = np.linspace(0.0, 2.0 * math.pi, vertex_count, endpoint=False)
        vertices = np.column_stack((self.a * np.cos(angles), self.b * np.sin(angles)))
        chord_error = max(self.a, self.b) * (1.0 - math.cos(math.pi / vertex_count))
        threshold_sq = (player_radius + chord_error) ** 2
        for index in range(vertex_count):
            edge_start = vertices[index]
            edge_end = vertices[(index + 1) % vertex_count]
            if _segment_distance_sq(start, end, edge_start, edge_end) <= threshold_sq:
                return True
        return False

    def snapshot(self) -> "ThreatObservation":
        return ThreatObservation(
            id=self.id,
            x=self.x,
            y=self.y,
            vx=self.vx,
            vy=self.vy,
            radius_x=self.a,
            radius_y=self.b,
            angle=self.angle,
            lethal=self.lethal,
            visible=self.visible,
            warning=self.warning,
            opacity=self.opacity,
            danger=self.danger,
            uncertainty=self.uncertainty,
            tag=self.tag,
            source_id=self.source_id,
            age=self.age,
            lifetime=self.lifetime,
            metadata=dict(self.metadata),
        )


class CircleThreat(Threat):
    def __init__(
        self,
        x: float,
        y: float,
        radius: float,
        *,
        vx: float = 0.0,
        vy: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(x=x, y=y, a=radius, b=radius, vx=vx, vy=vy, **kwargs)


class EllipseThreat(Threat):
    def __init__(
        self,
        x: float,
        y: float,
        radius_x: float,
        radius_y: float,
        *,
        vx: float = 0.0,
        vy: float = 0.0,
        angle: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            x=x,
            y=y,
            a=radius_x,
            b=radius_y,
            vx=vx,
            vy=vy,
            angle=angle,
            **kwargs,
        )


@dataclass(frozen=True, slots=True)
class ThreatObservation:
    id: int
    x: float
    y: float
    vx: float
    vy: float
    radius_x: float
    radius_y: float
    angle: float
    lethal: bool
    visible: bool
    warning: bool
    opacity: float
    danger: float
    uncertainty: float
    tag: str
    source_id: int | None
    age: int
    lifetime: int | None
    metadata: Mapping[str, MetadataValue]

    @property
    def radius(self) -> float:
        return max(self.radius_x, self.radius_y)

    @property
    def a(self) -> float:
        return self.radius_x

    @property
    def b(self) -> float:
        return self.radius_y

    @property
    def weight(self) -> float:
        return self.danger


@dataclass(frozen=True, slots=True)
class Event:
    frame: int
    kind: str
    details: Mapping[str, MetadataValue] = field(default_factory=dict)


class Outcome(str, Enum):
    RUNNING = "running"
    HIT = "hit"
    CLEAR = "clear"


SEMANTIC_CHANNELS = (
    "threat_occupancy",
    "threat_velocity_x",
    "threat_velocity_y",
    "warning",
    "player",
    "boundary",
)


@dataclass(frozen=True, slots=True)
class Observation:
    frame: int
    time_seconds: float
    player: PlayerObservation
    bounds: Bounds
    player_bounds: Bounds
    threats: tuple[ThreatObservation, ...]
    events: tuple[Event, ...]
    semantic: np.ndarray
    semantic_channels: tuple[str, ...] = SEMANTIC_CHANNELS

    @property
    def entities(self) -> tuple[ThreatObservation, ...]:
        return self.threats


@dataclass(frozen=True, slots=True)
class ForecastFrame:
    offset: int
    frame: int
    player: PlayerObservation
    threats: tuple[ThreatObservation, ...]
    events: tuple[Event, ...]
    # Conservative first/max-radius/last samples for every threat that existed
    # anywhere since the previous returned forecast frame.
    swept_threats: tuple[ThreatObservation, ...] = ()


@dataclass(frozen=True, slots=True)
class StepResult:
    observation: Observation
    reward: float
    terminated: bool
    truncated: bool
    outcome: Outcome
    events: tuple[Event, ...]
    info: Mapping[str, Any]

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated

    def __iter__(self) -> Iterator[Any]:
        # Gymnasium-compatible unpacking without taking a Gymnasium dependency.
        yield self.observation
        yield self.reward
        yield self.terminated
        yield self.truncated
        yield self.info


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    fps: int = 60
    bounds: Bounds = Bounds()
    player_start: tuple[float, float] = (0.0, -176.0)
    # SR's bundled Reimu/Marisa player image groups use a=b=0.5.
    player_radius: float = 0.5
    player_speed: float = 4.0
    player_focus_speed: float = 2.0
    # LuaSTG applies input immediately.  Motor delay is optional and separate
    # from delayed vision; reaction_frames remains the compatibility name.
    reaction_frames: int = 0
    action_hold_frames: int = 1
    player_left_margin: float = 8.0
    player_right_margin: float = 8.0
    player_bottom_margin: float = 16.0
    player_top_margin: float = 32.0
    semantic_width: int = 48
    semantic_height: int = 56
    velocity_scale: float = 8.0
    survival_reward: float = 1.0 / 3600.0
    hit_reward: float = -1.0
    clear_reward: float = 1.0

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        numeric = (
            *self.player_start,
            self.player_radius,
            self.player_speed,
            self.player_focus_speed,
            self.velocity_scale,
            self.survival_reward,
            self.hit_reward,
            self.clear_reward,
            self.player_left_margin,
            self.player_right_margin,
            self.player_bottom_margin,
            self.player_top_margin,
        )
        if not all(math.isfinite(float(value)) for value in numeric):
            raise ValueError("simulation configuration must be finite")
        if self.player_radius <= 0.0:
            raise ValueError("player radius must be positive")
        if self.player_speed <= 0.0 or self.player_focus_speed <= 0.0:
            raise ValueError("player speeds must be positive")
        if self.reaction_frames < 0:
            raise ValueError("reaction_frames cannot be negative")
        if self.action_hold_frames <= 0:
            raise ValueError("action_hold_frames must be positive")
        margins = (
            self.player_left_margin,
            self.player_right_margin,
            self.player_bottom_margin,
            self.player_top_margin,
        )
        if any(value < 0.0 for value in margins):
            raise ValueError("player movement margins cannot be negative")
        if (
            self.bounds.width <= self.player_left_margin + self.player_right_margin
            or self.bounds.height <= self.player_bottom_margin + self.player_top_margin
        ):
            raise ValueError("player movement margins leave no navigable playfield")
        if self.velocity_scale <= 0.0:
            raise ValueError("velocity_scale must be positive")
        if self.semantic_width <= 1 or self.semantic_height <= 1:
            raise ValueError("semantic dimensions must be greater than one")

    @property
    def player_bounds(self) -> Bounds:
        return Bounds(
            self.bounds.left + self.player_left_margin,
            self.bounds.right - self.player_right_margin,
            self.bounds.bottom + self.player_bottom_margin,
            self.bounds.top - self.player_top_margin,
        )

    @property
    def motor_delay_frames(self) -> int:
        return self.reaction_frames


@runtime_checkable
class ScenarioProtocol(Protocol):
    name: str
    duration_frames: int

    def reset(self, env: "STGEnvironment") -> None: ...

    def update(self, env: "STGEnvironment") -> None: ...


ActionLike = Action | int | Sequence[int] | Mapping[str, Any]


def coerce_action(value: ActionLike) -> Action:
    if isinstance(value, Action):
        return value
    if isinstance(value, (int, np.integer)):
        return Action.from_discrete(int(value))
    if isinstance(value, Mapping):
        return Action(
            move_x=int(value.get("move_x", 0)),
            move_y=int(value.get("move_y", 0)),
            slow=bool(value.get("slow", value.get("focus", False))),
            shoot=bool(value.get("shoot", True)),
            spell=bool(value.get("spell", False)),
        )
    if len(value) not in (2, 3):
        raise ValueError("action sequences must be (x, y) or (x, y, slow)")
    return Action(
        move_x=int(value[0]),
        move_y=int(value[1]),
        slow=bool(value[2]) if len(value) == 3 else False,
    )


class STGEnvironment:
    """A deterministic 60 Hz environment with human-like input latency."""

    def __init__(
        self,
        scenario: ScenarioProtocol,
        *,
        config: SimulationConfig | None = None,
        seed: int = 20260729,
    ) -> None:
        if not isinstance(scenario, ScenarioProtocol):
            raise TypeError("scenario must implement ScenarioProtocol")
        self.scenario = scenario
        self.config = config or SimulationConfig()
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.player = PlayerState()
        self.frame = 0
        self.outcome = Outcome.RUNNING
        self._threats: dict[int, Threat] = {}
        self._next_threat_id = 1
        self._events: list[Event] = []
        self._frame_collision_threats: tuple[Threat, ...] = ()
        self._action_queue: deque[Action] = deque()
        self.submitted_action = Action()
        self.requested_action = Action()
        self.applied_action = Action()
        self._held_action = Action()
        self._hold_remaining = 0
        self.reset(seed=self.seed)

    @property
    def bounds(self) -> Bounds:
        return self.config.bounds

    @property
    def player_bounds(self) -> Bounds:
        return self.config.player_bounds

    @property
    def fps(self) -> int:
        return self.config.fps

    @property
    def done(self) -> bool:
        return self.outcome is not Outcome.RUNNING

    @property
    def threats(self) -> tuple[Threat, ...]:
        return tuple(self._threats.values())

    @property
    def frame_collision_threats(self) -> tuple[Threat, ...]:
        """Threat geometry used by the most recently advanced frame."""

        return self._frame_collision_threats

    def reset(self, *, seed: int | None = None) -> Observation:
        if seed is not None:
            self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.frame = 0
        self.outcome = Outcome.RUNNING
        self._threats.clear()
        self._next_threat_id = 1
        self._events = []
        self._frame_collision_threats = ()
        self.submitted_action = Action()
        self.requested_action = Action()
        self.applied_action = Action()
        self._held_action = Action()
        self._hold_remaining = 0
        self._action_queue = deque(
            Action() for _ in range(self.config.reaction_frames)
        )
        self.player = PlayerState(
            x=float(self.config.player_start[0]),
            y=float(self.config.player_start[1]),
            radius=self.config.player_radius,
            speed=self.config.player_speed,
            focus_speed=self.config.player_focus_speed,
            alive=True,
        )
        self.player.x, self.player.y = self.player_bounds.clamp(self.player.x, self.player.y)
        self.scenario.reset(self)
        return self.observe()

    def clone(self) -> "STGEnvironment":
        """Return an independent environment including RNG and action latency."""

        return deepcopy(self)

    def set_player_position(self, x: float, y: float) -> None:
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("player position must be finite")
        self.player.x, self.player.y = self.player_bounds.clamp(x, y)

    def add_threat(self, threat: Threat) -> int:
        if threat.id >= 0 and threat.id in self._threats:
            raise ValueError(f"threat id {threat.id} already exists")
        threat.id = self._next_threat_id
        self._next_threat_id += 1
        threat.prev_x = threat.x
        threat.prev_y = threat.y
        self._threats[threat.id] = threat
        return threat.id

    def spawn_circle(
        self,
        x: float,
        y: float,
        radius: float,
        *,
        vx: float = 0.0,
        vy: float = 0.0,
        **kwargs: Any,
    ) -> int:
        return self.add_threat(CircleThreat(x, y, radius, vx=vx, vy=vy, **kwargs))

    def spawn_ellipse(
        self,
        x: float,
        y: float,
        radius_x: float,
        radius_y: float,
        *,
        vx: float = 0.0,
        vy: float = 0.0,
        angle: float = 0.0,
        **kwargs: Any,
    ) -> int:
        return self.add_threat(
            EllipseThreat(
                x,
                y,
                radius_x,
                radius_y,
                vx=vx,
                vy=vy,
                angle=angle,
                **kwargs,
            )
        )

    def get_threat(self, threat_id: int) -> Threat | None:
        return self._threats.get(threat_id)

    def iter_threats(self, *, tag: str | None = None) -> Iterable[Threat]:
        if tag is None:
            return tuple(self._threats.values())
        return tuple(threat for threat in self._threats.values() if threat.tag == tag)

    def remove_threat(self, threat_id: int) -> None:
        self._threats.pop(threat_id, None)

    def emit_event(self, kind: str, **details: MetadataValue) -> None:
        self._events.append(Event(self.frame, kind, dict(details)))

    def step(self, action: ActionLike) -> StepResult:
        if self.done:
            raise RuntimeError("step() called after terminal state; call reset()")
        return self._advance(action, build_semantic=True, detect_collision=True)

    def _advance(
        self,
        action: ActionLike,
        *,
        build_semantic: bool,
        detect_collision: bool,
    ) -> StepResult:
        self._events = []
        self.frame += 1
        self.submitted_action = coerce_action(action)
        if self._hold_remaining <= 0:
            self._held_action = self.submitted_action
            self._hold_remaining = self.config.action_hold_frames
        self.requested_action = self._held_action
        self._hold_remaining -= 1
        self._action_queue.append(self.requested_action)
        self.applied_action = self._action_queue.popleft()

        player_start = (self.player.x, self.player.y)
        self._move_player(self.applied_action)
        player_end = (self.player.x, self.player.y)

        for threat in self._threats.values():
            threat.begin_frame()
        self.scenario.update(self)
        for threat in tuple(self._threats.values()):
            threat.advance()
        self._frame_collision_threats = tuple(self._threats.values())

        hit: Threat | None = None
        if detect_collision:
            for threat in self._frame_collision_threats:
                if threat.lethal and threat.collides_swept(
                    player_start,
                    player_end,
                    self.player.radius,
                ):
                    hit = threat
                    break

        for threat_id, threat in tuple(self._threats.items()):
            if threat.expired or (threat.remove_outside and threat.is_outside(self.bounds)):
                del self._threats[threat_id]

        reward = self.config.survival_reward
        if hit is not None:
            self.player.alive = False
            self.outcome = Outcome.HIT
            reward = self.config.hit_reward
            self.emit_event(
                "player_hit",
                threat_id=hit.id,
                threat_tag=hit.tag,
                x=self.player.x,
                y=self.player.y,
            )
        elif self.frame >= self.scenario.duration_frames:
            self.outcome = Outcome.CLEAR
            reward += self.config.clear_reward
            self.emit_event("scenario_clear", scenario=self.scenario.name)

        observation = self.observe(include_semantic=build_semantic)
        info = {
            "frame": self.frame,
            "seed": self.seed,
            "scenario": self.scenario.name,
            "difficulty": getattr(self.scenario, "difficulty", None),
            "outcome": self.outcome.value,
            "submitted_action": self.submitted_action,
            "requested_action": self.requested_action,
            "applied_action": self.applied_action,
        }
        return StepResult(
            observation=observation,
            reward=reward,
            terminated=self.outcome in (Outcome.HIT, Outcome.CLEAR),
            truncated=False,
            outcome=self.outcome,
            events=tuple(self._events),
            info=info,
        )

    def _move_player(self, action: Action) -> None:
        dx = float(action.move_x)
        dy = float(action.move_y)
        if dx != 0.0 and dy != 0.0:
            diagonal = math.sqrt(0.5)
            dx *= diagonal
            dy *= diagonal
        speed = self.player.focus_speed if action.slow else self.player.speed
        self.player.x, self.player.y = self.player_bounds.clamp(
            self.player.x + dx * speed,
            self.player.y + dy * speed,
        )

    def observe(self, *, include_semantic: bool = True) -> Observation:
        semantic = (
            self.semantic_grid()
            if include_semantic
            else np.empty((0, 0, 0), dtype=np.float32)
        )
        semantic.setflags(write=False)
        return Observation(
            frame=self.frame,
            time_seconds=self.frame / self.fps,
            player=self.player.snapshot(),
            bounds=self.bounds,
            player_bounds=self.player_bounds,
            threats=tuple(threat.snapshot() for threat in self._threats.values()),
            events=tuple(self._events),
            semantic=semantic,
        )

    def semantic_grid(self) -> np.ndarray:
        """Rasterize visible state into the same six channels as vision.py."""

        width = self.config.semantic_width
        height = self.config.semantic_height
        xs = np.linspace(self.bounds.left, self.bounds.right, width, dtype=np.float32)
        ys = np.linspace(self.bounds.bottom, self.bounds.top, height, dtype=np.float32)
        grid = np.zeros((len(SEMANTIC_CHANNELS), height, width), dtype=np.float32)
        density = np.zeros((height, width), dtype=np.float32)
        warning_density = np.zeros((height, width), dtype=np.float32)
        velocity_x_sum = np.zeros((height, width), dtype=np.float32)
        velocity_y_sum = np.zeros((height, width), dtype=np.float32)

        for threat in self._threats.values():
            if not threat.visible:
                continue
            cos_a = math.cos(threat.angle)
            sin_a = math.sin(threat.angle)
            extent_x = abs(cos_a) * threat.a + abs(sin_a) * threat.b
            extent_y = abs(sin_a) * threat.a + abs(cos_a) * threat.b
            column_start = max(0, int(np.searchsorted(xs, threat.x - extent_x, side="left")))
            column_end = min(width, int(np.searchsorted(xs, threat.x + extent_x, side="right")))
            row_start = max(0, int(np.searchsorted(ys, threat.y - extent_y, side="left")))
            row_end = min(height, int(np.searchsorted(ys, threat.y + extent_y, side="right")))
            if column_start >= column_end or row_start >= row_end:
                if not (
                    self.bounds.left <= threat.x <= self.bounds.right
                    and self.bounds.bottom <= threat.y <= self.bounds.top
                ):
                    continue
                column_start = int(np.argmin(np.abs(xs - threat.x)))
                row_start = int(np.argmin(np.abs(ys - threat.y)))
                column_end = column_start + 1
                row_end = row_start + 1
            dx = xs[column_start:column_end][None, :] - threat.x
            dy = ys[row_start:row_end, None] - threat.y
            local_x = cos_a * dx + sin_a * dy
            local_y = -sin_a * dx + cos_a * dy
            mask = (local_x / threat.a) ** 2 + (local_y / threat.b) ** 2 <= 1.0
            if not np.any(mask):
                if (
                    self.bounds.left <= threat.x <= self.bounds.right
                    and self.bounds.bottom <= threat.y <= self.bounds.top
                ):
                    column = int(np.argmin(np.abs(xs - threat.x)))
                    row = int(np.argmin(np.abs(ys - threat.y)))
                    column_start, column_end = column, column + 1
                    row_start, row_end = row, row + 1
                    mask = np.ones((1, 1), dtype=bool)
                else:
                    continue
            intensity = float(np.clip(threat.opacity, 0.0, 1.0))
            density_view = density[row_start:row_end, column_start:column_end]
            velocity_x_view = velocity_x_sum[row_start:row_end, column_start:column_end]
            velocity_y_view = velocity_y_sum[row_start:row_end, column_start:column_end]
            density_view[mask] += intensity
            velocity_x_view[mask] += intensity * np.clip(
                threat.vx / self.config.velocity_scale, -1.0, 1.0,
            )
            velocity_y_view[mask] += intensity * np.clip(
                threat.vy / self.config.velocity_scale, -1.0, 1.0,
            )
            if threat.warning:
                warning_view = warning_density[row_start:row_end, column_start:column_end]
                warning_view[mask] += intensity

        occupied = density > 0.0
        grid[0] = np.clip(np.log1p(density) / math.log(9.0), 0.0, 1.0)
        grid[1, occupied] = velocity_x_sum[occupied] / density[occupied]
        grid[2, occupied] = velocity_y_sum[occupied] / density[occupied]
        grid[3] = np.clip(np.log1p(warning_density) / math.log(9.0), 0.0, 1.0)

        # A low-resolution semantic frame must never lose the player between
        # sample points; the visible sprite/hitbox maps to its nearest cell.
        player_column = int(np.argmin(np.abs(xs - self.player.x)))
        player_row = int(np.argmin(np.abs(ys - self.player.y)))
        grid[4, player_row, player_column] = 1.0
        grid[5, 0, :] = 1.0
        grid[5, -1, :] = 1.0
        grid[5, :, 0] = 1.0
        grid[5, :, -1] = 1.0
        return grid

    def forecast(
        self,
        *,
        horizon: int,
        step: int = 1,
        actions: Sequence[ActionLike] | None = None,
    ) -> tuple[ForecastFrame, ...]:
        """Advance a clone and return visible future states without collision death.

        ``horizon`` and ``step`` are measured in frames.  Requested actions are
        still subject to the cloned environment's reaction queue.  Missing
        actions mean no movement, which keeps threat-only forecasting honest.
        """

        if horizon < 0:
            raise ValueError("horizon cannot be negative")
        if step <= 0:
            raise ValueError("forecast step must be positive")
        clone = self.clone()
        forecast: list[ForecastFrame] = []
        action_sequence = actions or ()
        interval_samples: dict[int, list[ThreatObservation]] = {
            threat.id: [threat] for threat in clone.observe(include_semantic=False).threats
        }
        for offset in range(1, horizon + 1):
            action: ActionLike = action_sequence[offset - 1] if offset <= len(action_sequence) else Action()
            result = clone._advance(action, build_semantic=False, detect_collision=False)
            for threat in result.observation.threats:
                interval_samples.setdefault(threat.id, []).append(threat)
            terminal = clone.frame >= clone.scenario.duration_frames
            if offset % step == 0 or offset == horizon or terminal:
                swept: list[ThreatObservation] = []
                for threat_id in sorted(interval_samples):
                    trace = interval_samples[threat_id]
                    selected = {0, len(trace) - 1}
                    selected.add(max(
                        range(len(trace)),
                        key=lambda index: (
                            trace[index].radius,
                            trace[index].danger,
                            -index,
                        ),
                    ))
                    swept.extend(trace[index] for index in sorted(selected))
                forecast.append(
                    ForecastFrame(
                        offset=offset,
                        frame=clone.frame,
                        player=result.observation.player,
                        threats=result.observation.threats,
                        events=result.events,
                        swept_threats=tuple(swept),
                    )
                )
                interval_samples = {
                    threat.id: [threat] for threat in result.observation.threats
                }
            if terminal:
                break
        return tuple(forecast)

    def forecast_threats(
        self,
        horizon: int,
        step: int = 1,
    ) -> tuple[tuple[ThreatObservation, ...], ...]:
        return tuple(frame.threats for frame in self.forecast(horizon=horizon, step=step))

    def forecast_swept_threats(
        self,
        horizon: int,
        step: int = 1,
    ) -> tuple[tuple[int, tuple[ThreatObservation, ...]], ...]:
        """Return real offsets and conservative interval occupancy samples."""

        return tuple(
            (frame.offset, frame.swept_threats)
            for frame in self.forecast(horizon=horizon, step=step)
        )


__all__ = [
    "Action",
    "Bounds",
    "CircleThreat",
    "EllipseThreat",
    "Event",
    "ForecastFrame",
    "Observation",
    "Outcome",
    "PlayerObservation",
    "PlayerState",
    "SEMANTIC_CHANNELS",
    "ScenarioProtocol",
    "SimulationConfig",
    "STGEnvironment",
    "StepResult",
    "Threat",
    "ThreatObservation",
    "coerce_action",
]
