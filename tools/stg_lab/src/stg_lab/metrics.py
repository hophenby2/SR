from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


def _canonical(value: Any) -> Any:
    if is_dataclass(value):
        return _canonical(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _canonical(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, np.ndarray):
        return _canonical(value.tolist())
    if isinstance(value, (tuple, list)):
        items = [_canonical(item) for item in value]
        if items and all(isinstance(item, Mapping) and "id" in item for item in items):
            items.sort(key=lambda item: item["id"])
        return items
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        if not np.isfinite(value):
            raise ValueError("state contains non-finite float")
        return round(value, 6)
    if isinstance(value, str) or value is None:
        return value
    if hasattr(value, "value"):
        return _canonical(value.value)
    if hasattr(value, "__dict__"):
        return _canonical(vars(value))
    raise TypeError(f"unsupported state value: {type(value).__name__}")


def state_hash(state: Any) -> str:
    canonical = json.dumps(
        _canonical(state),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.blake2s(canonical, digest_size=16).hexdigest()


@dataclass(frozen=True, slots=True)
class EpisodeMetrics:
    scenario: str
    seed: int
    survived: bool
    frames: int
    peak_risk: float
    total_risk: float
    deaths: int = 0
    action_agreement: float | None = None
    state_hash: str | None = None
    teacher_overrides: int | None = None


@dataclass(frozen=True, slots=True)
class AcceptanceThresholds:
    planner_survival: float = 0.95
    visual_survival: float = 0.90
    action_agreement: float = 0.85
    memory_risk_improvement: float = 0.30


@dataclass(slots=True)
class AcceptanceReport:
    thresholds: AcceptanceThresholds = AcceptanceThresholds()
    planner: list[EpisodeMetrics] = field(default_factory=list)
    visual: list[EpisodeMetrics] = field(default_factory=list)
    memory_first_risk: float | None = None
    memory_second_risk: float | None = None
    deterministic: bool = False
    notes: list[str] = field(default_factory=list)

    @staticmethod
    def _survival(episodes: Iterable[EpisodeMetrics]) -> float:
        episodes = list(episodes)
        if not episodes:
            return 0.0
        return sum(episode.survived for episode in episodes) / len(episodes)

    @staticmethod
    def _agreement(episodes: Iterable[EpisodeMetrics]) -> float:
        values = [episode.action_agreement for episode in episodes if episode.action_agreement is not None]
        return float(np.mean(values)) if values else 0.0

    @property
    def planner_survival(self) -> float:
        return self._survival(self.planner)

    @property
    def visual_survival(self) -> float:
        return self._survival(self.visual)

    @property
    def action_agreement(self) -> float:
        return self._agreement(self.visual)

    @property
    def memory_risk_improvement(self) -> float:
        if self.memory_first_risk is None or self.memory_second_risk is None:
            return 0.0
        if self.memory_first_risk <= 0.0:
            return 1.0 if self.memory_second_risk <= 0.0 else 0.0
        return (self.memory_first_risk - self.memory_second_risk) / self.memory_first_risk

    @property
    def passed(self) -> bool:
        return (
            self.deterministic
            and self.planner_survival >= self.thresholds.planner_survival
            and self.visual_survival >= self.thresholds.visual_survival
            and self.action_agreement >= self.thresholds.action_agreement
            and self.memory_risk_improvement >= self.thresholds.memory_risk_improvement
        )

    def summary(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "deterministic": self.deterministic,
            "planner_survival": self.planner_survival,
            "visual_survival": self.visual_survival,
            "action_agreement": self.action_agreement,
            "memory_risk_improvement": self.memory_risk_improvement,
            "thresholds": asdict(self.thresholds),
            "planner_episodes": len(self.planner),
            "visual_episodes": len(self.visual),
            "notes": self.notes,
        }

    def write(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.summary(), indent=2) + "\n", encoding="utf-8")
