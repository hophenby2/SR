from __future__ import annotations

import pytest

from stg_lab.render_performance import RenderPerformanceTrace


def test_render_performance_reports_dense_frame_percentiles() -> None:
    trace = RenderPerformanceTrace(dense_object_threshold=300)
    for frame, fps, objects in (
        (1, 60.0, 100),
        (2, 20.0, 300),
        (3, 10.0, 400),
        (4, 30.0, 350),
    ):
        trace.push({
            "episode_frame": frame,
            "performance": {"native_fps": fps, "object_count": objects},
        })

    report = trace.report()

    assert report["peak_object_count"] == 400
    assert report["all_frames"]["median"] == pytest.approx(25.0)
    assert report["dense_frames"]["sample_count"] == 3
    assert report["dense_frames"]["minimum"] == 10.0
    assert report["dense_frames"]["median"] == 20.0


def test_render_performance_rejects_invalid_samples() -> None:
    trace = RenderPerformanceTrace()
    trace.push({"episode_frame": 1})
    trace.push({
        "episode_frame": 2,
        "performance": {"native_fps": 0.0, "object_count": 300},
    })

    report = trace.report()

    assert report["valid_sample_count"] == 0
    assert report["invalid_sample_count"] == 2
    assert report["dense_frames"]["median"] is None


def test_dense_threshold_must_be_nonnegative() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        RenderPerformanceTrace(dense_object_threshold=-1)
    with pytest.raises(ValueError, match="integer"):
        RenderPerformanceTrace(dense_object_threshold=300.5)  # type: ignore[arg-type]
