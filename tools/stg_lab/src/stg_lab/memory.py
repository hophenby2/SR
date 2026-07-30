"""SQLite-backed episodic route memory kept outside model weights."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
import math
from pathlib import Path
import sqlite3
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("memory values must be finite")
        return value
    raise TypeError(f"value of type {type(value).__name__} is not JSON serializable")


def _encode(value: Any) -> str:
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _decode(value: str) -> Any:
    return json.loads(value)


def _cue_key(encoded: str) -> str:
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    if isinstance(value, Mapping):
        for key in sorted(value, key=str):
            name = f"{prefix}.{key}" if prefix else str(key)
            result.update(_flatten(value[key], name))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            result.update(_flatten(item, f"{prefix}[{index}]"))
    else:
        result[prefix or "$value"] = value
    return result


def cue_similarity(query: Any, candidate: Any) -> float:
    """Compare observable cue structures on a stable zero-to-one scale."""

    left = _flatten(_jsonable(query))
    right = _flatten(_jsonable(candidate))
    keys = sorted(set(left) | set(right))
    if not keys:
        return 1.0
    score = 0.0
    for key in keys:
        if key not in left or key not in right:
            continue
        a, b = left[key], right[key]
        if isinstance(a, bool) or isinstance(b, bool):
            score += float(a is b)
        elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
            scale = max(1.0, abs(float(a)), abs(float(b)))
            score += max(0.0, 1.0 - abs(float(a) - float(b)) / scale)
        else:
            score += float(a == b)
    return score / len(keys)


@dataclass(frozen=True, slots=True)
class DeathPoint:
    x: float
    y: float
    frame: int | None = None


@dataclass(frozen=True, slots=True)
class EpisodeMemory:
    id: int
    scenario: str
    cue: Any
    death_point: DeathPoint | None
    trigger_lead: int
    route: tuple[Any, ...]
    confidence: float
    successes: int
    failures: int
    revision: int

    @property
    def trigger_lead_frames(self) -> int:
        return self.trigger_lead


@dataclass(frozen=True, slots=True)
class MemoryMatch:
    memory: EpisodeMemory
    cue_similarity: float
    score: float


def _death_point(value: DeathPoint | Mapping[str, Any] | Sequence[float] | None) -> DeathPoint | None:
    if value is None or isinstance(value, DeathPoint):
        return value
    if isinstance(value, Mapping):
        return DeathPoint(
            x=float(value["x"]),
            y=float(value["y"]),
            frame=(None if value.get("frame") is None else int(value["frame"])),
        )
    if len(value) not in (2, 3):
        raise ValueError("death point must be (x, y) or (x, y, frame)")
    return DeathPoint(
        x=float(value[0]),
        y=float(value[1]),
        frame=(None if len(value) == 2 else int(value[2])),
    )


class EpisodicMemory:
    """Persistent scenario memories with deterministic similarity retrieval."""

    SCHEMA_VERSION = 1

    def __init__(self, path: str | Path = ":memory:", *, readonly: bool = False) -> None:
        self.path = str(path) if str(path) == ":memory:" else str(Path(path).expanduser())
        self.readonly = bool(readonly)
        if self.readonly and self.path == ":memory:":
            raise ValueError("an in-memory episodic database cannot be opened read-only")
        if self.path != ":memory:" and not self.readonly:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        if self.readonly:
            database = Path(self.path).resolve(strict=True)
            self._connection = sqlite3.connect(
                database.as_uri() + "?mode=ro&immutable=1",
                uri=True,
            )
        else:
            self._connection = sqlite3.connect(self.path)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._initialize()

    def _initialize(self) -> None:
        version = int(self._connection.execute("PRAGMA user_version").fetchone()[0])
        if self.readonly:
            if version != self.SCHEMA_VERSION:
                raise RuntimeError(f"unsupported episodic-memory schema version {version}")
            try:
                self._connection.execute("SELECT 1 FROM episodes LIMIT 1").fetchone()
            except sqlite3.DatabaseError as error:
                raise RuntimeError("episodic-memory database has no readable schema") from error
            return
        if version not in (0, self.SCHEMA_VERSION):
            raise RuntimeError(f"unsupported episodic-memory schema version {version}")
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY,
                scenario TEXT NOT NULL,
                cue_key TEXT NOT NULL,
                cue_json TEXT NOT NULL,
                death_x REAL,
                death_y REAL,
                death_frame INTEGER,
                trigger_lead INTEGER NOT NULL CHECK (trigger_lead >= 0),
                route_json TEXT NOT NULL,
                confidence REAL NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
                successes INTEGER NOT NULL DEFAULT 0 CHECK (successes >= 0),
                failures INTEGER NOT NULL DEFAULT 0 CHECK (failures >= 0),
                revision INTEGER NOT NULL DEFAULT 0 CHECK (revision >= 0)
            );
            CREATE INDEX IF NOT EXISTS episodes_scenario
                ON episodes (scenario, confidence DESC, id ASC);
            CREATE INDEX IF NOT EXISTS episodes_exact_cue
                ON episodes (scenario, cue_key, id ASC);
            """
        )
        self._connection.execute(f"PRAGMA user_version = {self.SCHEMA_VERSION}")
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "EpisodicMemory":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def _require_writable(self) -> None:
        if self.readonly:
            raise RuntimeError("episodic-memory database is read-only")

    def remember(
        self,
        scenario: str,
        cue: Any,
        *,
        death_point: DeathPoint | Mapping[str, Any] | Sequence[float] | None,
        trigger_lead: int,
        route: Iterable[Any],
        confidence: float = 0.5,
    ) -> EpisodeMemory:
        """Store one observed failure and its proposed anticipatory route."""

        self._require_writable()
        scenario = str(scenario).strip()
        if not scenario:
            raise ValueError("scenario cannot be empty")
        if trigger_lead < 0:
            raise ValueError("trigger_lead cannot be negative")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        point = _death_point(death_point)
        cue_json = _encode(cue)
        route_json = _encode(tuple(route))
        with self._connection:
            cursor = self._connection.execute(
                """
                INSERT INTO episodes (
                    scenario, cue_key, cue_json, death_x, death_y, death_frame,
                    trigger_lead, route_json, confidence
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    scenario, _cue_key(cue_json), cue_json,
                    None if point is None else point.x,
                    None if point is None else point.y,
                    None if point is None else point.frame,
                    int(trigger_lead), route_json, float(confidence),
                ),
            )
        return self.get(int(cursor.lastrowid))

    # ``save`` is a convenient verb for callers that are not modelling death.
    save = remember
    store = remember

    def get(self, memory_id: int) -> EpisodeMemory:
        row = self._connection.execute(
            "SELECT * FROM episodes WHERE id = ?", (int(memory_id),)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown memory id {memory_id}")
        return self._from_row(row)

    @staticmethod
    def _from_row(row: sqlite3.Row) -> EpisodeMemory:
        point = None
        if row["death_x"] is not None and row["death_y"] is not None:
            point = DeathPoint(
                x=float(row["death_x"]),
                y=float(row["death_y"]),
                frame=None if row["death_frame"] is None else int(row["death_frame"]),
            )
        route = _decode(row["route_json"])
        return EpisodeMemory(
            id=int(row["id"]),
            scenario=str(row["scenario"]),
            cue=_decode(row["cue_json"]),
            death_point=point,
            trigger_lead=int(row["trigger_lead"]),
            route=tuple(route),
            confidence=float(row["confidence"]),
            successes=int(row["successes"]),
            failures=int(row["failures"]),
            revision=int(row["revision"]),
        )

    def retrieve(
        self,
        scenario: str,
        cue: Any,
        *,
        limit: int = 5,
        minimum_similarity: float = 0.0,
        minimum_confidence: float = 0.0,
    ) -> tuple[MemoryMatch, ...]:
        if limit <= 0:
            return ()
        if not 0.0 <= minimum_similarity <= 1.0 or not 0.0 <= minimum_confidence <= 1.0:
            raise ValueError("similarity and confidence limits must be in [0, 1]")
        rows = self._connection.execute(
            """
            SELECT * FROM episodes
            WHERE scenario = ? AND confidence >= ?
            ORDER BY id ASC
            """,
            (str(scenario), float(minimum_confidence)),
        ).fetchall()
        matches: list[MemoryMatch] = []
        for row in rows:
            memory = self._from_row(row)
            similarity = cue_similarity(cue, memory.cue)
            if similarity < minimum_similarity:
                continue
            matches.append(MemoryMatch(
                memory=memory,
                cue_similarity=similarity,
                score=similarity * memory.confidence,
            ))
        matches.sort(key=lambda match: (
            -match.score,
            -match.cue_similarity,
            -match.memory.confidence,
            -match.memory.revision,
            match.memory.id,
        ))
        return tuple(matches[:limit])

    def best(
        self,
        scenario: str,
        cue: Any,
        *,
        minimum_similarity: float = 0.0,
        minimum_confidence: float = 0.0,
    ) -> EpisodeMemory | None:
        matches = self.retrieve(
            scenario,
            cue,
            limit=1,
            minimum_similarity=minimum_similarity,
            minimum_confidence=minimum_confidence,
        )
        return matches[0].memory if matches else None

    def update_outcome(
        self,
        memory_id: int,
        *,
        success: bool,
        learning_rate: float = 0.15,
    ) -> EpisodeMemory:
        """Update route confidence after trying the remembered response."""

        self._require_writable()
        if not 0.0 < learning_rate <= 1.0:
            raise ValueError("learning_rate must be in (0, 1]")
        current = self.get(memory_id)
        target = 1.0 if success else 0.0
        confidence = current.confidence + learning_rate * (target - current.confidence)
        with self._connection:
            cursor = self._connection.execute(
                """
                UPDATE episodes SET
                    confidence = ?,
                    successes = successes + ?,
                    failures = failures + ?,
                    revision = revision + 1
                WHERE id = ?
                """,
                (confidence, int(success), int(not success), int(memory_id)),
            )
        if cursor.rowcount != 1:
            raise KeyError(f"unknown memory id {memory_id}")
        return self.get(memory_id)

    def record_success(self, memory_id: int, *, learning_rate: float = 0.15) -> EpisodeMemory:
        return self.update_outcome(memory_id, success=True, learning_rate=learning_rate)

    def record_failure(self, memory_id: int, *, learning_rate: float = 0.15) -> EpisodeMemory:
        return self.update_outcome(memory_id, success=False, learning_rate=learning_rate)

    def delete(self, memory_id: int) -> bool:
        self._require_writable()
        with self._connection:
            cursor = self._connection.execute("DELETE FROM episodes WHERE id = ?", (int(memory_id),))
        return cursor.rowcount == 1

    def __len__(self) -> int:
        return int(self._connection.execute("SELECT COUNT(*) FROM episodes").fetchone()[0])


EpisodicMemoryStore = EpisodicMemory


__all__ = [
    "DeathPoint",
    "EpisodeMemory",
    "EpisodicMemory",
    "EpisodicMemoryStore",
    "MemoryMatch",
    "cue_similarity",
]
