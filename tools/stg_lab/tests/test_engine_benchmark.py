from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from stg_lab.engine_benchmark import (
    _RUNTIME_SOURCE_FILES,
    _local_runtime_sources,
    _resolve_mod_root,
    run_engine_benchmark,
)


class FakeEngineClient:
    def __init__(self) -> None:
        self.episode_frame = 0
        self.scenario = ""
        self.attack = 0
        self.card_index = 0

    def ping(self) -> dict[str, Any]:
        return {
            "id": 1,
            "ok": True,
            "protocol": 2,
            "session_id": "fake-session",
            "process_nonce": "fake-process-nonce",
            "runtime_identity": {
                "process_id": 1234,
                "executable_path": "LuaSTGSub.exe",
                "executable_crc32": "1234abcd",
                "source_crc32": _local_runtime_sources()[0],
            },
        }

    def catalog(self) -> dict[str, Any]:
        attacks = [
            {"scenario": "boss:A", "attack": 1, "card_index": 2, "label": "A #1"},
            {"scenario": "boss:B", "attack": 1, "card_index": 3, "label": "B #1"},
        ]
        return {
            "id": 2,
            "ok": True,
            "catalog": {
                "schema_version": 1,
                "scenario_count": 2,
                "attack_count": 2,
                "attacks": attacks,
            },
        }

    def _observation(self) -> dict[str, Any]:
        enemies = [{"id": 1}] if self.episode_frame >= 1 else []
        enemy_bullets = [
            {"id": index} for index in range(10, 10 + self.episode_frame)
        ]
        return {
            "episode_frame": self.episode_frame,
            "terminated": False,
            "stage": {"scenario": self.scenario, "card_index": self.card_index},
            "player": {"id": 0},
            "enemy_bullets": enemy_bullets,
            "enemies": enemies,
            "nontjt_enemies": [],
            "indestructibles": [],
            "lasers": [],
            "counts": {
                "enemies": len(enemies),
                "enemy_bullets": len(enemy_bullets),
                "nontjt_enemies": 0,
                "indestructibles": 0,
                "lasers": 0,
            },
        }

    def reset(self, scenario, attack, *, seed, player, options=None):
        self.scenario = scenario
        self.attack = attack
        self.card_index = 2 if scenario == "boss:A" else 3
        self.episode_frame = 1
        return {
            "reset": {
                "scenario": scenario,
                "attack": attack,
                "card_index": self.card_index,
                "seed": seed,
                "player": player,
            },
            "observation": self._observation(),
        }

    def step(self, _action, *, repeat):
        self.episode_frame += repeat
        return {"observation": self._observation()}


def _write_runtime_sources(root: Path) -> None:
    for index, relative in enumerate(_RUNTIME_SOURCE_FILES):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"-- runtime source {index}\n", encoding="utf-8")


def test_runtime_sources_resolve_from_environment_after_wheel_install(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod_root = tmp_path / "mod"
    _write_runtime_sources(mod_root)
    monkeypatch.setenv("STG_LAB_MOD_ROOT", str(mod_root))
    monkeypatch.chdir(tmp_path)

    assert _resolve_mod_root() == mod_root
    crc32, sha256 = _local_runtime_sources()
    assert tuple(crc32) == _RUNTIME_SOURCE_FILES
    assert tuple(sha256) == _RUNTIME_SOURCE_FILES
    assert all(len(value) == 8 for value in crc32.values())
    assert all(len(value) == 64 for value in sha256.values())


def test_invalid_explicit_runtime_source_root_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STG_LAB_MOD_ROOT", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="does not identify an SR mod root"):
        _resolve_mod_root()


def test_engine_benchmark_runs_every_catalog_attack() -> None:
    report = run_engine_benchmark(
        FakeEngineClient(),
        seed=100,
        frames_per_attack=4,
        step_batch=1,
        expected_attacks=2,
    )
    assert report["passed"] is True
    assert report["engine_verified"] is False
    assert report["verification_scope"] == "live_attack_catalog_per_frame_content_regression"
    assert report["catalog"]["attack_count"] == 2
    assert report["passed_attacks"] == 2
    assert [item["seed"] for item in report["attacks"]] == [100, 101]
    assert all(item["advanced_frames"] == 4 for item in report["attacks"])
    assert all(len(item["frame_hashes"]) == 5 for item in report["attacks"])
    assert report["active_content_attacks"] == 2
    assert report["engine_process_nonce"] == "fake-process-nonce"


def test_engine_benchmark_rejects_unexpected_catalog_size() -> None:
    report = run_engine_benchmark(FakeEngineClient(), expected_attacks=53)
    assert report["passed"] is False
    assert report["engine_verified"] is False
    assert report["passed_attacks"] == 2
    assert any("expected 53" in error for error in report["errors"])


def test_engine_benchmark_rejects_batched_or_inactive_observations() -> None:
    batched = run_engine_benchmark(
        FakeEngineClient(),
        frames_per_attack=4,
        step_batch=2,
        expected_attacks=2,
    )
    assert batched["passed"] is False
    assert all("one hash per logical frame" in item["error"] for item in batched["attacks"])

    client = FakeEngineClient()
    original = client._observation

    def inactive_observation():
        observation = original()
        observation["enemy_bullets"] = []
        observation["counts"]["enemy_bullets"] = 0
        return observation

    client._observation = inactive_observation
    inactive = run_engine_benchmark(client, frames_per_attack=2, expected_attacks=2)
    assert inactive["passed"] is False
    assert inactive["active_content_attacks"] == 0
    assert all("active hazard" in item["error"] for item in inactive["attacks"])


def test_engine_benchmark_treats_spawned_collidable_enemies_as_active_content() -> None:
    client = FakeEngineClient()
    original = client._observation

    def summoned_enemy_observation():
        observation = original()
        observation["enemy_bullets"] = []
        observation["counts"]["enemy_bullets"] = 0
        observation["enemies"].append({"id": 2})
        observation["counts"]["enemies"] = 2
        return observation

    client._observation = summoned_enemy_observation
    report = run_engine_benchmark(client, frames_per_attack=2, expected_attacks=2)
    assert report["passed"] is True
    assert report["active_content_attacks"] == 2
