from __future__ import annotations

import json
from pathlib import Path
import statistics

import pytest

from stg_lab.engine_mpc import (
    RegionDynamicsMemory,
    _RegionPhaseMemory,
    load_region_dynamics_memory,
)


REFERENCE_EXPANSION_START = 1862
REFERENCE_CYCLE_FRAMES = 180


def learned_dynamics() -> RegionDynamicsMemory:
    return RegionDynamicsMemory(
        minimum_radius=7.0,
        maximum_radius=28.0,
        growth_rate=0.7,
        contraction_rate=0.7,
        expanding_frames=30.0,
        maximum_hold_frames=30.0,
        contracting_frames=30.0,
        minimum_hold_frames=90.0,
        cycle_frames=180.0,
    )


def reference_radius(frame: int) -> float:
    """Compact reconstruction of the global median in the v16/v21 traces."""

    phase_frame = (
        frame - REFERENCE_EXPANSION_START
    ) % REFERENCE_CYCLE_FRAMES
    if phase_frame < 30:
        return 7.0 + 0.7 * phase_frame
    if phase_frame < 60:
        return 28.0
    if phase_frame < 90:
        return 28.0 - 0.7 * (phase_frame - 60)
    return 7.0


def observed_radii(radius: float, frame: int) -> list[float]:
    """Keep the median stable while object count and newborn outliers vary."""

    values = [radius] * (9 + frame % 3)
    values.extend((6.3, 7.7, 28.0))
    return values if frame % 2 else list(reversed(values))


def replay(
    memory: _RegionPhaseMemory,
    *,
    through_frame: int,
    frame_shift: int = 0,
) -> list[str]:
    transitions: list[str] = []
    previous_phase = memory.phase
    for source_frame in range(1840, through_frame + 1, 3):
        radius = reference_radius(source_frame)
        memory.update(
            source_frame + frame_shift,
            observed_radii(radius, source_frame),
        )
        if memory.phase != previous_phase:
            transitions.append(memory.phase)
            previous_phase = memory.phase
    return transitions


def normalized_phase_starts(
    memory: _RegionPhaseMemory,
    frame_shift: int,
) -> dict[str, list[int]]:
    return {
        phase: [frame - frame_shift for frame in starts]
        for phase, starts in memory.phase_starts.items()
    }


def test_real_trace_learns_four_phase_topology_and_durations() -> None:
    memory = _RegionPhaseMemory()

    transitions = replay(memory, through_frame=2170)

    assert transitions == [
        "expanding",
        "maximum_hold",
        "contracting",
        "minimum_hold",
    ] * 2
    assert memory.learned_cycle_frames is not None
    assert abs(memory.learned_cycle_frames - REFERENCE_CYCLE_FRAMES) <= 3.0

    expected_expansion_starts = (1862, 2042)
    actual_expansion_starts = memory.phase_starts["expanding"][-2:]
    assert all(
        abs(actual - expected) <= 2
        for actual, expected in zip(
            actual_expansion_starts,
            expected_expansion_starts,
            strict=True,
        )
    )

    expected_durations = {
        "expanding": 30.0,
        "maximum_hold": 30.0,
        "contracting": 30.0,
        "minimum_hold": 90.0,
    }
    for phase, expected in expected_durations.items():
        learned = statistics.median(memory.phase_durations[phase])
        assert abs(learned - expected) <= 3.0


def test_phase_forecast_matches_next_topology_change_not_an_absolute_frame() -> None:
    query_frames = (2170, 2200, 2206, 2218)
    expected_expansion = 2222
    expected_portal_close = 2237
    memory = _RegionPhaseMemory()
    forecasts: dict[int, tuple[float | None, float | None]] = {}

    for source_frame in range(1840, query_frames[-1] + 1, 3):
        radius = reference_radius(source_frame)
        memory.update(source_frame, observed_radii(radius, source_frame))
        if source_frame in query_frames:
            forecasts[source_frame] = (
                memory.frames_until_expansion(),
                memory.frames_until_radius(17.5),
            )

    assert forecasts.keys() == set(query_frames)
    for frame, (until_expansion, until_close) in forecasts.items():
        assert until_expansion is not None
        assert abs((frame + until_expansion) - expected_expansion) <= 2.0
        assert until_close is not None
        assert (
            expected_portal_close - 1.0
            <= frame + until_close
            <= expected_portal_close + 2.0
        )

    shifted = _RegionPhaseMemory()
    shift = 431
    replay(shifted, through_frame=query_frames[-1], frame_shift=shift)

    assert shifted.learned_cycle_frames == memory.learned_cycle_frames
    assert shifted.frames_until_expansion() == memory.frames_until_expansion()
    assert shifted.frames_until_radius(17.5) == memory.frames_until_radius(17.5)
    assert normalized_phase_starts(shifted, shift) == normalized_phase_starts(
        memory,
        0,
    )


def test_platform_wobble_and_newborn_outliers_do_not_invent_a_phase() -> None:
    minimum = _RegionPhaseMemory()
    replay(minimum, through_frame=1981)
    assert minimum.phase == "minimum_hold"
    expansion_starts = list(minimum.phase_starts["expanding"])

    low_platform_wobble = (6.3, 7.0, 7.7, 7.7, 7.0, 6.3, 7.0)
    for index, radius in enumerate(low_platform_wobble, start=1):
        frame = 1981 + index * 3
        # A minority of newly spawned large objects must not move the global
        # radius phase away from the established minimum platform.
        radii = [radius] * 9 + [28.0, 28.0]
        minimum.update(frame, radii if index % 2 else list(reversed(radii)))

    assert minimum.phase == "minimum_hold"
    assert minimum.phase_starts["expanding"] == expansion_starts

    maximum = _RegionPhaseMemory()
    replay(maximum, through_frame=1918)
    assert maximum.phase == "maximum_hold"
    for index, radius in enumerate((27.3, 28.0) * 4, start=1):
        maximum.update(1918 + index * 3, [radius] * 9 + [7.0, 7.0])

    assert maximum.phase == "maximum_hold"


def test_prior_forecasts_first_visible_minimum_hold_by_relative_phase() -> None:
    memory = _RegionPhaseMemory(dynamics_memory=learned_dynamics())
    shifted = _RegionPhaseMemory(dynamics_memory=learned_dynamics())

    memory.update(100, [7.0] * 10)
    memory.update(108, [7.0] * 10)
    shifted.update(531, [7.0] * 10)
    shifted.update(539, [7.0] * 10)

    assert memory.phase == shifted.phase == "minimum_hold"
    assert memory.frames_until_expansion() == 82.0
    assert shifted.frames_until_expansion() == 82.0
    assert memory.frames_until_radius(17.5) == pytest.approx(97.0)
    assert shifted.frames_until_radius(17.5) == pytest.approx(97.0)
    assert memory.learned_cycle_frames == shifted.learned_cycle_frames == 180.0


def memory_artifact() -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": "region_dynamics_memory",
        "scenario": "okuu:Lunatic",
        "attack": 3,
        "model": {
            "phase_order": [
                "expanding",
                "maximum_hold",
                "contracting",
                "minimum_hold",
            ],
            "minimum_radius": 7.0,
            "maximum_radius": 28.0,
            "growth_rate": 0.7,
            "contraction_rate": 0.7,
            "phase_durations": {
                "expanding": 30.0,
                "maximum_hold": 30.0,
                "contracting": 30.0,
                "minimum_hold": 90.0,
            },
            "cycle_frames": 180.0,
        },
    }


def write_memory(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_region_memory_loader_accepts_only_dynamics_not_routes(tmp_path: Path) -> None:
    path = tmp_path / "memory.json"
    write_memory(path, memory_artifact())

    loaded = load_region_dynamics_memory(
        path,
        scenario="okuu:Lunatic",
        attack=3,
    )
    assert loaded == learned_dynamics()

    for field, value in (
        ("actions", [0, 1, 2]),
        ("waypoints", [[12.0, -40.0]]),
        ("start_frame", 1200),
    ):
        forbidden = memory_artifact()
        forbidden[field] = value
        write_memory(path, forbidden)
        with pytest.raises(ValueError, match="unsupported top-level fields"):
            load_region_dynamics_memory(path)

    forbidden_model = memory_artifact()
    model = forbidden_model["model"]
    assert isinstance(model, dict)
    model["x"] = -120.0
    write_memory(path, forbidden_model)
    with pytest.raises(ValueError, match="unsupported fields"):
        load_region_dynamics_memory(path)


def test_region_memory_v2_accepts_only_relative_lateral_flow_rule(
    tmp_path: Path,
) -> None:
    path = tmp_path / "memory-v2.json"
    value = memory_artifact()
    value["schema_version"] = 2
    model = value["model"]
    assert isinstance(model, dict)
    model["lateral_flow"] = {
        "cycle_frames": 360.0,
        "safe_side_rule": "opposite_incoming_lateral_flow",
    }
    write_memory(path, value)

    loaded = load_region_dynamics_memory(
        path,
        scenario="okuu:Lunatic",
        attack=3,
    )
    assert loaded.lateral_flow_cycle_frames == 360.0
    assert loaded.safe_side_rule == "opposite_incoming_lateral_flow"

    for forbidden_field, forbidden_value in (
        ("phase_offset", 120),
        ("side_sequence", ["left", "right"]),
        ("waypoints", [[-120.0, -40.0]]),
        ("actions", [1, 2, 3]),
    ):
        forbidden = memory_artifact()
        forbidden["schema_version"] = 2
        forbidden_model = forbidden["model"]
        assert isinstance(forbidden_model, dict)
        forbidden_model["lateral_flow"] = {
            "cycle_frames": 360.0,
            "safe_side_rule": "opposite_incoming_lateral_flow",
            forbidden_field: forbidden_value,
        }
        write_memory(path, forbidden)
        with pytest.raises(ValueError, match="must define only"):
            load_region_dynamics_memory(path)

    wrong_rule = memory_artifact()
    wrong_rule["schema_version"] = 2
    wrong_model = wrong_rule["model"]
    assert isinstance(wrong_model, dict)
    wrong_model["lateral_flow"] = {
        "cycle_frames": 360.0,
        "safe_side_rule": "alternate_fixed_sides",
    }
    write_memory(path, wrong_rule)
    with pytest.raises(ValueError, match="unsupported region safe-side rule"):
        load_region_dynamics_memory(path)


def test_bundled_boss3_region_memory_is_loadable() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "models"
        / "region_dynamics_boss3_v2.json"
    )

    loaded = load_region_dynamics_memory(
        path,
        scenario="okuu:Lunatic",
        attack=3,
    )

    assert loaded.minimum_radius == 7.0
    assert loaded.maximum_radius == 28.0
    assert loaded.cycle_frames == 180.0
    assert loaded.lateral_flow_cycle_frames == 360.0
    assert loaded.safe_side_rule == "opposite_incoming_lateral_flow"


def test_region_memory_rejects_inconsistent_cycle() -> None:
    with pytest.raises(ValueError, match="durations must sum"):
        RegionDynamicsMemory(
            minimum_radius=7.0,
            maximum_radius=28.0,
            growth_rate=0.7,
            contraction_rate=0.7,
            expanding_frames=30.0,
            maximum_hold_frames=30.0,
            contracting_frames=30.0,
            minimum_hold_frames=90.0,
            cycle_frames=240.0,
        )
