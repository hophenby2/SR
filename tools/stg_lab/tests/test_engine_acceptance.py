from __future__ import annotations

from copy import deepcopy

from stg_lab.engine_acceptance import compare_engine_reports
from stg_lab.engine_benchmark import _local_runtime_sources
from stg_lab.provenance import source_tree_sha256


EXPECTED_ATTACKS = 53
FRAMES_PER_ATTACK = 300


def _hash(attack: int, frame: int) -> str:
    return f"{attack * 1000 + frame:032x}"


def report(session: str, process_nonce: str, process_id: int = 1001) -> dict:
    runtime_crc32, local_sha256 = _local_runtime_sources()
    attacks = []
    ordinals = {}
    index = 0
    for scenario_index in range(22):
        scenario = f"boss:{scenario_index:02d}"
        attack_count = 3 if scenario_index < 9 else 2
        for attack in range(1, attack_count + 1):
            ordinals[scenario] = attack
            hashes = [_hash(index + 1, frame) for frame in range(FRAMES_PER_ATTACK + 1)]
            attacks.append({
                "scenario": scenario,
                "attack": attack,
                "card_index": attack + 1,
                "label": f"{scenario} #{attack}",
                "seed": 10 + index,
                "requested_frames": FRAMES_PER_ATTACK,
                "advanced_frames": FRAMES_PER_ATTACK,
                "step_requests": FRAMES_PER_ATTACK,
                "frame_hashes": hashes,
                "final_state_hash": hashes[-1],
                "seen_active_content": True,
                "seen_enemy": True,
                "seen_threat": True,
                "first_active_content_frame": 100,
                "peak_counts": {
                    "enemy_bullets": 1,
                    "enemies": 1,
                    "nontjt_enemies": 0,
                    "indestructibles": 0,
                    "lasers": 0,
                },
                "terminated": False,
                "termination_reason": None,
                "passed": True,
            })
            index += 1
    catalog_scenarios = []
    for scenario in sorted(ordinals):
        nested = [
            {key: item[key] for key in ("scenario", "attack", "card_index", "label")}
            for item in attacks
            if item["scenario"] == scenario
        ]
        catalog_scenarios.append({
            "scenario": scenario,
            "label": scenario,
            "attack_count": len(nested),
            "attacks": nested,
        })
    return {
        "schema_version": 2,
        "implementation_sha256": source_tree_sha256(),
        "run_kind": "live_luastg_spell_practice_regression",
        "verification_scope": "live_attack_catalog_per_frame_content_regression",
        "host_protocol": 2,
        "passed": True,
        "engine_verified": False,
        "engine_session_id": session,
        "engine_process_nonce": process_nonce,
        "engine_process_id": process_id,
        "runtime_identity": {
            "process_id": process_id,
            "executable_path": "LuaSTGSub.exe",
            "executable_crc32": "1234abcd",
            "source_crc32": runtime_crc32,
        },
        "local_source_sha256": local_sha256,
        "catalog": {
            "schema_version": 1,
            "scenario_count": 22,
            "attack_count": EXPECTED_ATTACKS,
            "expected_attacks": EXPECTED_ATTACKS,
            "count_matches": True,
            "attacks": [
                {key: item[key] for key in ("scenario", "attack", "card_index", "label")}
                for item in attacks
            ],
            "scenarios": catalog_scenarios,
        },
        "config": {
            "seed": 10,
            "player": "reimu_player",
            "frames_per_attack": FRAMES_PER_ATTACK,
            "step_batch": 1,
            "action": {
                "move_x": 0,
                "move_y": 0,
                "shoot": False,
                "slow": True,
                "spell": False,
            },
            "reset_options": {
                "lifeleft": 99,
                "player_protect_frames": FRAMES_PER_ATTACK + 600,
            },
        },
        "passed_attacks": EXPECTED_ATTACKS,
        "failed_attacks": 0,
        "active_content_attacks": EXPECTED_ATTACKS,
        "attacks": attacks,
        "errors": [],
    }


def test_engine_acceptance_requires_distinct_matching_processes() -> None:
    result = compare_engine_reports(
        report("run-a", "process-a", 1001),
        report("run-b", "process-b", 1002),
    )
    assert result["passed"] is True
    assert result["engine_verified"] is True
    assert result["matched_attacks"] == EXPECTED_ATTACKS
    assert result["process_nonces"] == ["process-a", "process-b"]


def test_engine_acceptance_rejects_same_process_or_changed_hashes() -> None:
    first = report("run-a", "same-process", 1001)
    second = report("run-a", "same-process", 1001)
    second["attacks"][0]["frame_hashes"][1] = "f" * 32
    result = compare_engine_reports(first, second)
    assert result["passed"] is False
    assert result["engine_verified"] is False
    assert any("same session id" in error for error in result["errors"])
    assert any("same process nonce" in error for error in result["errors"])
    assert any("same operating-system process" in error for error in result["errors"])
    assert any("hashes differ" in error for error in result["errors"])


def test_engine_acceptance_rejects_malformed_matching_evidence() -> None:
    attacks = [
        {
            "scenario": f"boss:{index}",
            "attack": 1,
            "card_index": 1,
            "seed": index,
            "frame_hashes": [None],
        }
        for index in range(EXPECTED_ATTACKS)
    ]
    minimal = {"passed": True, "config": {}, "attacks": attacks}
    result = compare_engine_reports(
        {**minimal, "engine_session_id": "a", "engine_process_nonce": "process-a"},
        {**minimal, "engine_session_id": "b", "engine_process_nonce": "process-b"},
    )
    assert result["passed"] is False
    assert result["engine_verified"] is False
    assert any("invalid schema version" in error for error in result["errors"])
    assert any("did not observe active attack content" in error for error in result["errors"])
    assert any("wrong number of per-frame hashes" in error for error in result["errors"])


def test_engine_acceptance_rejects_batched_or_incomplete_frames() -> None:
    first = report("run-a", "process-a", 1001)
    second = deepcopy(first)
    second["engine_session_id"] = "run-b"
    second["engine_process_nonce"] = "process-b"
    second["engine_process_id"] = 1002
    second["runtime_identity"]["process_id"] = 1002
    for value in (first, second):
        value["config"]["step_batch"] = 2
        value["attacks"][0]["advanced_frames"] = FRAMES_PER_ATTACK - 1
        value["attacks"][0]["seen_active_content"] = False
        value["attacks"][1]["frame_hashes"][0] = "A" * 32
    result = compare_engine_reports(first, second)
    assert result["passed"] is False
    assert result["engine_verified"] is False
    assert any("did not hash every logical frame" in error for error in result["errors"])
    assert any("did not advance every requested frame" in error for error in result["errors"])
    assert any("did not observe active attack content" in error for error in result["errors"])
    assert any("contains an invalid state hash" in error for error in result["errors"])


def test_engine_acceptance_never_verifies_a_reduced_attack_count() -> None:
    result = compare_engine_reports(
        report("run-a", "process-a", 1001),
        report("run-b", "process-b", 1002),
        expected_attacks=2,
    )
    assert result["passed"] is False
    assert result["engine_verified"] is False
    assert any("requires exactly 53 attacks" in error for error in result["errors"])


def test_engine_acceptance_rejects_incomplete_nested_catalog() -> None:
    first = report("run-a", "process-a", 1001)
    second = report("run-b", "process-b", 1002)
    for value in (first, second):
        value["catalog"]["scenarios"][0]["attacks"] = []
        value["catalog"]["scenarios"][0]["attack_count"] = 0
    result = compare_engine_reports(first, second)
    assert result["passed"] is False
    assert any("does not flatten" in error for error in result["errors"])
