from __future__ import annotations

import hashlib

import pytest

from stg_lab.memory import EpisodicMemory


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_readonly_memory_preserves_database_bytes(tmp_path) -> None:
    path = tmp_path / "memory.sqlite"
    with EpisodicMemory(path) as store:
        memory = store.remember(
            "stage5_boss4:lunatic",
            {"kind": "visible"},
            death_point=None,
            trigger_lead=12,
            route=({"move_x": 1, "move_y": 0},),
        )

    before = _sha256(path)
    with EpisodicMemory(path, readonly=True) as store:
        assert store.readonly
        assert len(store) == 1
        assert store.get(memory.id).route == ({"move_x": 1, "move_y": 0},)
        with pytest.raises(RuntimeError, match="read-only"):
            store.record_success(memory.id)
        with pytest.raises(RuntimeError, match="read-only"):
            store.delete(memory.id)

    assert _sha256(path) == before
    assert sorted(tmp_path.iterdir()) == [path]


def test_readonly_memory_requires_an_existing_initialized_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        EpisodicMemory(tmp_path / "missing.sqlite", readonly=True)

    empty = tmp_path / "empty.sqlite"
    empty.touch()
    with pytest.raises(RuntimeError, match="schema version 0"):
        EpisodicMemory(empty, readonly=True)
