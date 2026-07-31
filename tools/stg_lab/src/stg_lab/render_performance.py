"""Reporting-only native renderer telemetry for live-engine regressions."""

from __future__ import annotations

import math
from typing import Any, Mapping


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _fps_summary(samples: list[dict[str, int | float]]) -> dict[str, int | float | None]:
    values = [float(sample["native_fps"]) for sample in samples]
    return {
        "sample_count": len(values),
        "minimum": min(values) if values else None,
        "p10": _percentile(values, 0.10),
        "median": _percentile(values, 0.50),
        "p90": _percentile(values, 0.90),
        "maximum": max(values) if values else None,
        "mean": sum(values) / len(values) if values else None,
    }


class RenderPerformanceTrace:
    """Collect the same averaged FPS and object count shown in the title bar."""

    def __init__(self, *, dense_object_threshold: int = 300) -> None:
        if (
            isinstance(dense_object_threshold, bool)
            or not isinstance(dense_object_threshold, int)
            or dense_object_threshold < 0
        ):
            raise ValueError("dense object threshold must be a nonnegative integer")
        self.dense_object_threshold = dense_object_threshold
        self.samples: list[dict[str, int | float]] = []
        self.invalid_sample_count = 0

    def push(self, observation: Mapping[str, Any]) -> None:
        performance = observation.get("performance")
        frame = observation.get("episode_frame")
        if not isinstance(performance, Mapping):
            self.invalid_sample_count += 1
            return
        fps = _finite_number(performance.get("native_fps"))
        objects = _finite_number(performance.get("object_count"))
        if (
            fps is None
            or fps <= 0.0
            or objects is None
            or objects < 0.0
            or isinstance(frame, bool)
            or not isinstance(frame, int)
        ):
            self.invalid_sample_count += 1
            return
        self.samples.append({
            "episode_frame": frame,
            "native_fps": fps,
            "object_count": int(objects),
        })

    def report(self) -> dict[str, Any]:
        dense = [
            sample for sample in self.samples
            if int(sample["object_count"]) >= self.dense_object_threshold
        ]
        return {
            "reporting_only_not_controller_input": True,
            "metric_source": "lstg.GetFPS 60-native-frame average and lstg.GetnObj",
            "dense_object_threshold": self.dense_object_threshold,
            "valid_sample_count": len(self.samples),
            "invalid_sample_count": self.invalid_sample_count,
            "peak_object_count": max(
                (int(sample["object_count"]) for sample in self.samples),
                default=None,
            ),
            "all_frames": _fps_summary(self.samples),
            "dense_frames": _fps_summary(dense),
            "samples": self.samples,
        }


__all__ = ["RenderPerformanceTrace"]
