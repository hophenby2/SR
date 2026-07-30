"""Engine-independent approximations of SR's Stage 5 Okuu attacks.

These scenarios preserve the mechanics relevant to navigation tests rather
than attempting frame-perfect asset or task emulation.  Boss #3 models rows of
falling nuclear fields whose radii periodically expand.  Boss #4 models the
large orbiting star and its rotating bullet fans, so reacting only when the
sweeper reaches the player is intentionally too late.
"""

from __future__ import annotations

import math
from typing import Final

from .sim import SimulationConfig, STGEnvironment


_DEG: Final[float] = math.pi / 180.0


def _difficulty(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in {"normal", "lunatic"}:
        raise ValueError("difficulty must be 'normal' or 'lunatic'")
    return normalized


def _velocity(speed: float, angle_degrees: float) -> tuple[float, float]:
    angle = angle_degrees * _DEG
    return speed * math.cos(angle), speed * math.sin(angle)


def _aim_angle(source_x: float, source_y: float, target_x: float, target_y: float) -> float:
    return math.atan2(target_y - source_y, target_x - source_x) / _DEG


class Stage5Boss3Scenario:
    """Approximate Okuu #3: migrating corridors and pulsing nuclear fields."""

    name = "stage5_boss3"
    forecast_independent_of_player = True
    description = "Periodic expansion splits and reconnects moving safe regions."

    def __init__(self, difficulty: str = "lunatic", *, duration_frames: int = 61 * 60) -> None:
        self.difficulty = _difficulty(difficulty)
        self.scenario_key = f"{self.name}:{self.difficulty}"
        if duration_frames <= 0:
            raise ValueError("duration_frames must be positive")
        self.duration_frames = int(duration_frames)
        self._source_ids: list[int] = []
        self._boss_id = -1
        self._last_pulse_state = "low"
        self._wave = 0
        self._motion_phase = 0.0

    @property
    def source_xs(self) -> tuple[float, ...]:
        if self.difficulty == "lunatic":
            return (-216.0, -168.0, -120.0, -72.0, -24.0, 24.0, 72.0, 120.0, 168.0, 216.0)
        return (-224.0, -160.0, -96.0, -32.0, 32.0, 96.0, 160.0, 224.0)

    @property
    def emission_interval(self) -> int:
        return 20 if self.difficulty == "lunatic" else 24

    @property
    def high_radius(self) -> float:
        # 5p3-nuke uses a = b = 70 * master.sc; both difficulties pulse to .4.
        return 28.0

    def reset(self, env: STGEnvironment) -> None:
        self._source_ids = []
        self._last_pulse_state = "low"
        self._wave = 0
        self._motion_phase = float(env.rng.uniform(-8.0, 8.0)) * _DEG
        self._boss_id = env.spawn_ellipse(
            0.0,
            112.0,
            16.0,
            12.0,
            lethal=False,
            warning=False,
            danger=0.1,
            opacity=0.65,
            managed=True,
            remove_outside=False,
            tag="boss",
            metadata={"role": "visible_reference"},
        )
        for index, x in enumerate(self.source_xs):
            source_id = env.spawn_circle(
                x,
                224.0,
                5.0,
                lethal=False,
                warning=False,
                danger=0.1,
                opacity=0.55,
                managed=True,
                remove_outside=False,
                tag="nuke_source",
                metadata={"source_index": index, "role": "visible_reference"},
            )
            self._source_ids.append(source_id)
        env.emit_event(
            "scenario_start",
            scenario=self.name,
            difficulty=self.difficulty,
        )

    def update(self, env: STGEnvironment) -> None:
        frame = env.frame
        self._update_boss(env, frame)
        pulse_state, target_radius = self._pulse(frame)
        if pulse_state != self._last_pulse_state:
            env.emit_event(
                "nuclear_field_" + pulse_state,
                target_radius=target_radius,
                cycle=max(0, (frame - 60) // 180),
            )
            self._last_pulse_state = pulse_state

        radius_rate = 0.7
        for threat in env.iter_threats(tag="expanding_nuke"):
            delta = target_radius - threat.a
            change = max(-radius_rate, min(radius_rate, delta))
            threat.a += change
            threat.b += change
            # SR has no separate pre-expansion warning marker.  Radius changes
            # remain visible through the ordinary occupancy channel.
            threat.warning = False
            threat.danger = 0.35 + 0.65 * (threat.a / self.high_radius)
            threat.metadata["risk_level"] = (
                3 if threat.a >= self.high_radius - 1.0 else 2 if threat.a >= 15.0 else 1
            )

        if frame >= 60 and (frame - 60) % self.emission_interval == 0:
            self._spawn_nuclear_wave(env)
        if frame >= 90 and (frame - 90) % 10 == 0:
            self._spawn_rotating_fans(env, frame)

    def _update_boss(self, env: STGEnvironment, frame: int) -> None:
        boss = env.get_threat(self._boss_id)
        if boss is None:
            return
        old_x, old_y = boss.x, boss.y
        boss.x = 72.0 * math.sin(frame * _DEG * 0.55 + self._motion_phase)
        boss.y = 112.0 + 18.0 * math.sin(frame * _DEG * 0.31 + self._motion_phase * 0.7)
        boss.vx = boss.x - old_x
        boss.vy = boss.y - old_y

    def _pulse(self, frame: int) -> tuple[str, float]:
        if frame < 60:
            return "low", 7.0
        phase = (frame - 60) % 180
        if phase < 120:
            return "low", 7.0
        return "expanded", self.high_radius

    def _spawn_nuclear_wave(self, env: STGEnvironment) -> None:
        angle = (
            -90.0
            + 30.0 * math.cos(max(0, env.frame - 150) * _DEG)
        )
        vx, vy = _velocity(3.0, angle)
        ids: list[int] = []
        for index, x in enumerate(self.source_xs):
            threat_id = env.spawn_circle(
                x,
                224.0,
                7.0,
                vx=vx,
                vy=vy,
                lifetime=190,
                lethal=True,
                danger=0.55,
                uncertainty=0.06,
                tag="expanding_nuke",
                metadata={"wave": self._wave, "source_index": index, "risk_level": 1},
            )
            ids.append(threat_id)
        env.emit_event(
            "nuclear_wave",
            wave=self._wave,
            count=len(ids),
        )
        self._wave += 1

    def _spawn_rotating_fans(self, env: STGEnvironment, frame: int) -> None:
        boss = env.get_threat(self._boss_id)
        if boss is None:
            return
        phase = max(0, frame - 180)
        primary = 112.5 - 20.0 * math.sin(phase * _DEG)
        for center in (primary, primary + 180.0):
            for index in range(10):
                angle_degrees = center + index * 15.0
                vx, vy = _velocity(4.0, angle_degrees)
                # THlib/bullet/bullet.lua loads `ellipse` with a=b=4.5; its
                # sprite is elongated but its collision primitive is circular.
                env.spawn_circle(
                    boss.x,
                    boss.y,
                    4.5,
                    vx=vx,
                    vy=vy,
                    lifetime=180,
                    danger=0.7,
                    uncertainty=0.04,
                    tag="rotating_ellipse",
                )


class Stage5Boss4Scenario:
    """Approximate Okuu #4: an orbiting sweeper with rotating bullet fans."""

    name = "stage5_boss4"
    forecast_independent_of_player = True
    description = "A fast orbiting hazard rewards advance positioning and route memory."

    def __init__(self, difficulty: str = "lunatic", *, duration_frames: int = 69 * 60) -> None:
        self.difficulty = _difficulty(difficulty)
        self.scenario_key = f"{self.name}:{self.difficulty}"
        if duration_frames <= 0:
            raise ValueError("duration_frames must be positive")
        self.duration_frames = int(duration_frames)
        self._boss_id = -1
        self._star_id = -1
        self._last_quadrant = -1
        self._orbit_phase = 0.0
        self._fan_phase = 0.0
        self._radial_phase = 0.0

    def reset(self, env: STGEnvironment) -> None:
        self._last_quadrant = -1
        self._orbit_phase = float(env.rng.uniform(-5.0, 5.0)) * _DEG
        self._fan_phase = float(env.rng.uniform(-3.0, 3.0))
        self._radial_phase = float(env.rng.uniform(-4.0, 4.0))
        self._boss_id = env.spawn_circle(
            0.0,
            0.0,
            15.0,
            lethal=False,
            warning=False,
            danger=0.1,
            opacity=0.65,
            managed=True,
            remove_outside=False,
            tag="boss",
            metadata={"role": "visible_reference"},
        )
        self._star_id = -1
        env.emit_event(
            "scenario_start",
            scenario=self.name,
            difficulty=self.difficulty,
            orbit_phase_deg=self._orbit_phase / _DEG,
            radial_phase_deg=self._radial_phase,
        )

    def update(self, env: STGEnvironment) -> None:
        frame = env.frame
        if frame == 120:
            self._star_id = env.spawn_circle(
                0.0,
                0.0,
                1.0,
                lethal=False,
                warning=True,
                danger=0.2,
                opacity=0.8,
                managed=True,
                remove_outside=False,
                tag="orbiting_star",
                metadata={"role": "sweeper", "risk_level": 1},
            )
            env.emit_event("orbit_star_spawned", phase_deg=self._orbit_phase / _DEG)
        self._update_star(env, frame)
        star_age = frame - 120
        if star_age >= 31 and (star_age - 31) % 5 == 0:
            self._spawn_star_fans(env, frame)
        if frame % 56 == 0:
            self._spawn_radial_volley(env, frame)

    def _update_star(self, env: STGEnvironment, frame: int) -> None:
        star = env.get_threat(self._star_id)
        if star is None:
            return
        age = frame - 120
        old_x, old_y = star.x, star.y
        distance = 66.0 * math.sin(min(age, 90) * _DEG)
        angle = math.pi + self._orbit_phase - age * 0.7 * _DEG
        star.x = distance * math.cos(angle)
        star.y = distance * math.sin(angle)
        star.vx = star.x - old_x
        star.vy = star.y - old_y
        scale = min(age / 30.0, 1.0)
        radius = max(0.5, 72.0 * scale - 6.0)
        star.a = radius
        star.b = radius
        star.angle = angle
        star.lethal = age >= 3
        star.warning = age < 31
        star.danger = min(1.0, 0.2 + 0.8 * scale)
        star.metadata["risk_level"] = 3 if age >= 31 else 1

        if age == 31:
            env.emit_event("orbit_sweep_started", angular_speed_deg=-0.7, radius=66.0)
        quadrant = int((angle % (2.0 * math.pi)) / (math.pi / 2.0))
        if age >= 31 and quadrant != self._last_quadrant:
            self._last_quadrant = quadrant
            env.emit_event("orbit_reference", quadrant=quadrant)

    def _spawn_star_fans(self, env: STGEnvironment, frame: int) -> None:
        star = env.get_threat(self._star_id)
        if star is None:
            return
        age = frame - 120
        primary = 180.0 + self._fan_phase + 0.3 * age
        secondary = primary - 300.0 * math.sin(age / 3.0 * _DEG)
        rays = 9
        for fan_index, center in enumerate((primary, secondary)):
            for index in range(rays):
                offset = (index - (rays - 1) / 2.0) * 5.0
                direction = center + offset
                spawn_direction = center + offset
                spawn_x = star.x + 80.0 * math.cos(spawn_direction * _DEG)
                spawn_y = star.y + 80.0 * math.sin(spawn_direction * _DEG)
                vx, vy = _velocity(5.0, direction)
                # `water_drop` is loaded with a=b=4 in the target THlib.
                env.spawn_circle(
                    spawn_x,
                    spawn_y,
                    4.0,
                    vx=vx,
                    vy=vy,
                    lifetime=145,
                    danger=0.85,
                    uncertainty=0.03,
                    tag="rotating_fan_bullet",
                    source_id=star.id,
                    metadata={"fan": fan_index},
                )

    def _spawn_radial_volley(self, env: STGEnvironment, frame: int) -> None:
        if self.difficulty == "lunatic":
            base_count, cluster = 6, (-6.0, -2.0, 2.0, 6.0)
        else:
            base_count, cluster = 8, (0.0,)
        phase = self._radial_phase
        for ray in range(base_count):
            for offset in cluster:
                direction = phase + ray * 360.0 / base_count + offset
                vx, vy = _velocity(1.5, direction)
                env.spawn_circle(
                    0.0,
                    0.0,
                    4.5,
                    vx=vx,
                    vy=vy,
                    lifetime=245,
                    danger=0.65,
                    uncertainty=0.03,
                    tag="radial_bullet",
                    source_id=self._boss_id,
                )
        env.emit_event(
            "radial_volley",
            count=base_count * len(cluster),
            phase=phase,
        )


_ALIASES: Final[dict[str, str]] = {
    "stage5_boss3": "stage5_boss3",
    "stage5-boss3": "stage5_boss3",
    "stage5 boss #3": "stage5_boss3",
    "5-3": "stage5_boss3",
    "okuu3": "stage5_boss3",
    "stage5_boss4": "stage5_boss4",
    "stage5-boss4": "stage5_boss4",
    "stage5 boss #4": "stage5_boss4",
    "5-4": "stage5_boss4",
    "okuu4": "stage5_boss4",
}


def available_scenarios() -> tuple[str, ...]:
    return ("stage5_boss3", "stage5_boss4")


def make_scenario(
    name: str,
    *,
    difficulty: str = "lunatic",
    duration_frames: int | None = None,
) -> Stage5Boss3Scenario | Stage5Boss4Scenario:
    key = _ALIASES.get(name.strip().lower())
    if key is None:
        choices = ", ".join(available_scenarios())
        raise KeyError(f"unknown scenario {name!r}; expected one of: {choices}")
    if key == "stage5_boss3":
        kwargs = {} if duration_frames is None else {"duration_frames": duration_frames}
        return Stage5Boss3Scenario(difficulty, **kwargs)
    kwargs = {} if duration_frames is None else {"duration_frames": duration_frames}
    return Stage5Boss4Scenario(difficulty, **kwargs)


def make_environment(
    name: str,
    *,
    difficulty: str = "lunatic",
    seed: int = 20260729,
    config: SimulationConfig | None = None,
    duration_frames: int | None = None,
) -> STGEnvironment:
    if config is not None and config.fps != 60:
        raise ValueError("SR Stage 5 approximations are defined only at 60 fps")
    return STGEnvironment(
        make_scenario(name, difficulty=difficulty, duration_frames=duration_frames),
        config=config,
        seed=seed,
    )


# Short alias for scripts that treat environment factories like Gym's make().
make_env = make_environment


__all__ = [
    "Stage5Boss3Scenario",
    "Stage5Boss4Scenario",
    "available_scenarios",
    "make_env",
    "make_environment",
    "make_scenario",
]
