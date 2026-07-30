"""Batch regression runner for a live LuaSTG JSONL bridge."""

from __future__ import annotations

import os
import time
from pathlib import Path
import re
from typing import Any, Mapping
import zlib

from .engine import EngineClient, EngineProtocolError
from .metrics import state_hash
from .protocol import Action
from .provenance import file_sha256, source_tree_sha256


_OBSERVATION_ARRAYS = (
    "enemy_bullets",
    "enemies",
    "nontjt_enemies",
    "indestructibles",
    "lasers",
)
_SOURCE_LAYOUT_MOD_ROOT = Path(__file__).resolve().parents[4]
_MOD_ROOT_ENV = "STG_LAB_MOD_ROOT"
_RUNTIME_SOURCE_FILES = (
    "root.lua",
    "_editor_output.lua",
    "compat/init.lua",
    "compat/combo.lua",
    "compat/gameplay.lua",
    "compat/spell_practice.lua",
    "compat/player/marisa.lua",
    "compat/player/reimu.lua",
    "compat/player/sakuya.lua",
    "compat/background/effects.lua",
    "compat/background/stage6bg.lua",
    "compat/background/stg2bg.lua",
    "compat/background/stg3bg.lua",
    "compat/background/stg4bg.lua",
    "compat/background/stg5bg.lua",
    "compat/background/stg6bg.lua",
    "compat/testing/bridge.lua",
    "compat/testing/init.lua",
)
_CRC32 = re.compile(r"[0-9a-f]{8}")


def _file_crc32(path: Path) -> str:
    checksum = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum = zlib.crc32(block, checksum)
    return f"{checksum & 0xffffffff:08x}"


def _looks_like_mod_root(path: Path) -> bool:
    return (path / "root.lua").is_file() and (
        path / "compat/testing/bridge.lua"
    ).is_file()


def _resolve_mod_root() -> Path:
    configured = os.environ.get(_MOD_ROOT_ENV)
    if configured:
        root = Path(configured).expanduser().resolve()
        if not _looks_like_mod_root(root):
            raise FileNotFoundError(
                f"{_MOD_ROOT_ENV} does not identify an SR mod root: {root}",
            )
        return root

    working = Path.cwd().resolve()
    candidates = (working, *working.parents, _SOURCE_LAYOUT_MOD_ROOT)
    for candidate in candidates:
        if _looks_like_mod_root(candidate):
            return candidate
    raise FileNotFoundError(
        f"cannot locate the SR mod root; set {_MOD_ROOT_ENV} to its directory",
    )


def _local_runtime_sources() -> tuple[dict[str, str], dict[str, str]]:
    mod_root = _resolve_mod_root()
    crc32 = {}
    sha256 = {}
    for relative in _RUNTIME_SOURCE_FILES:
        path = mod_root / relative
        crc32[relative] = _file_crc32(path)
        sha256[relative] = file_sha256(path)
    return crc32, sha256


def _runtime_identity(
    ping: Mapping[str, Any],
    expected_sources: Mapping[str, str],
) -> tuple[dict[str, Any], list[str]]:
    value = ping.get("runtime_identity")
    errors: list[str] = []
    if not isinstance(value, Mapping):
        return {}, ["engine ping has no runtime identity"]
    process_id = value.get("process_id")
    if isinstance(process_id, bool) or not isinstance(process_id, int) or process_id <= 0:
        errors.append("engine runtime identity has no operating-system process id")
    executable_path = value.get("executable_path")
    executable_crc32 = value.get("executable_crc32")
    if not isinstance(executable_path, str) or not executable_path:
        errors.append("engine runtime identity has no executable path")
    if not isinstance(executable_crc32, str) or _CRC32.fullmatch(executable_crc32) is None:
        errors.append("engine runtime identity has no executable checksum")
    sources = value.get("source_crc32")
    if not isinstance(sources, Mapping) or dict(sources) != dict(expected_sources):
        errors.append("engine runtime source fingerprints differ from the local test sources")
    return {
        "process_id": process_id,
        "executable_path": executable_path,
        "executable_crc32": executable_crc32,
        "source_crc32": dict(sources) if isinstance(sources, Mapping) else {},
    }, errors


def _observation(response: Mapping[str, Any]) -> Mapping[str, Any]:
    value = response.get("observation")
    if not isinstance(value, Mapping):
        raise EngineProtocolError("engine response has no observation object")
    counts = value.get("counts")
    if not isinstance(counts, Mapping):
        raise EngineProtocolError("engine observation has no counts object")
    for name in _OBSERVATION_ARRAYS:
        records = value.get(name)
        if not isinstance(records, list):
            raise EngineProtocolError(f"engine observation {name} is not an array")
        count = counts.get(name)
        if isinstance(count, bool) or not isinstance(count, int) or count != len(records):
            raise EngineProtocolError(f"engine observation {name} count does not match its array")
    return value


def _count_snapshot(observation: Mapping[str, Any]) -> dict[str, int]:
    counts = observation["counts"]
    assert isinstance(counts, Mapping)
    return {name: int(counts[name]) for name in _OBSERVATION_ARRAYS}


def _active_content(counts: Mapping[str, int]) -> tuple[bool, bool, bool]:
    enemy_count = counts["enemies"] + counts["nontjt_enemies"]
    seen_enemy = enemy_count > 0
    seen_threat = enemy_count > 1 or (
        counts["enemy_bullets"] + counts["indestructibles"] + counts["lasers"] > 0
    )
    return seen_enemy, seen_threat, seen_enemy and seen_threat


def _catalog_attacks(response: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    catalog = response.get("catalog")
    if not isinstance(catalog, Mapping):
        raise EngineProtocolError("engine response has no catalog object")
    values = catalog.get("attacks")
    if not isinstance(values, list):
        raise EngineProtocolError("engine catalog has no attacks array")
    attacks = []
    seen: set[tuple[str, int]] = set()
    for index, value in enumerate(values):
        if not isinstance(value, Mapping):
            raise EngineProtocolError(f"catalog attack {index} is not an object")
        scenario = value.get("scenario")
        attack = value.get("attack")
        card_index = value.get("card_index")
        if not isinstance(scenario, str) or not scenario:
            raise EngineProtocolError(f"catalog attack {index} has invalid scenario")
        if isinstance(attack, bool) or not isinstance(attack, int) or attack <= 0:
            raise EngineProtocolError(f"catalog attack {index} has invalid ordinal")
        if isinstance(card_index, bool) or not isinstance(card_index, int) or card_index <= 0:
            raise EngineProtocolError(f"catalog attack {index} has invalid card index")
        key = (scenario, attack)
        if key in seen:
            raise EngineProtocolError(f"duplicate catalog attack {scenario} #{attack}")
        seen.add(key)
        attacks.append(dict(value))
    return tuple(attacks)


def run_engine_benchmark(
    client: EngineClient,
    *,
    seed: int = 20260729,
    player: str = "reimu_player",
    frames_per_attack: int = 300,
    step_batch: int = 1,
    expected_attacks: int | None = 53,
    action: Action = Action(slow=True),
) -> dict[str, Any]:
    """Reset and advance every runtime-registered spell-practice attack."""

    if frames_per_attack <= 0:
        raise ValueError("frames_per_attack must be positive")
    if step_batch <= 0:
        raise ValueError("step_batch must be positive")
    if expected_attacks is not None and expected_attacks <= 0:
        raise ValueError("expected_attacks must be positive or None")

    started = time.perf_counter()
    ping = client.ping()
    catalog_response = client.catalog()
    attacks = _catalog_attacks(catalog_response)
    catalog = dict(catalog_response["catalog"])
    local_crc32, local_sha256 = _local_runtime_sources()
    runtime_identity, identity_errors = _runtime_identity(ping, local_crc32)
    results: list[dict[str, Any]] = []

    for index, attack_entry in enumerate(attacks):
        scenario = str(attack_entry["scenario"])
        ordinal = int(attack_entry["attack"])
        item: dict[str, Any] = {
            "scenario": scenario,
            "attack": ordinal,
            "card_index": int(attack_entry["card_index"]),
            "label": attack_entry.get("label"),
            "seed": int(seed) + index,
            "requested_frames": frames_per_attack,
            "passed": False,
        }
        try:
            reset_response = client.reset(
                scenario,
                ordinal,
                seed=item["seed"],
                player=player,
                options={
                    "lifeleft": 99,
                    "player_protect_frames": frames_per_attack + 600,
                },
            )
            reset_info = reset_response.get("reset")
            if not isinstance(reset_info, Mapping):
                raise EngineProtocolError("reset response has no reset metadata")
            if reset_info.get("scenario") != scenario or reset_info.get("attack") != ordinal:
                raise EngineProtocolError("reset metadata does not match catalog attack")
            if reset_info.get("card_index") != item["card_index"]:
                raise EngineProtocolError("reset card index does not match catalog")
            if reset_info.get("seed") != item["seed"] or reset_info.get("player") != player:
                raise EngineProtocolError("reset seed or player does not match the request")

            observation = _observation(reset_response)
            if not isinstance(observation.get("player"), Mapping):
                raise EngineProtocolError("reset observation has no active player")
            frame_hashes = [state_hash(observation)]
            peak_counts = _count_snapshot(observation)
            seen_enemy, seen_threat, seen_active_content = _active_content(peak_counts)
            first_active_content_frame = (
                int(observation.get("episode_frame", 0)) if seen_active_content else None
            )
            initial_episode_frame = int(observation.get("episode_frame", 0))
            requested_remaining = frames_per_attack
            step_requests = 0
            while requested_remaining > 0 and observation.get("terminated") is not True:
                repeat = min(step_batch, requested_remaining)
                observation = _observation(client.step(action, repeat=repeat))
                frame_hashes.append(state_hash(observation))
                current_counts = _count_snapshot(observation)
                for name, count in current_counts.items():
                    peak_counts[name] = max(peak_counts[name], count)
                current_enemy, current_threat, current_active = _active_content(current_counts)
                seen_enemy = seen_enemy or current_enemy
                seen_threat = seen_threat or current_threat
                if current_active and first_active_content_frame is None:
                    first_active_content_frame = int(observation.get("episode_frame", 0))
                seen_active_content = seen_active_content or current_active
                requested_remaining -= repeat
                step_requests += 1

            final_episode_frame = int(observation.get("episode_frame", 0))
            stage = observation.get("stage")
            if not isinstance(stage, Mapping) or stage.get("scenario") != scenario:
                raise EngineProtocolError("final observation scenario does not match reset")
            if stage.get("card_index") != item["card_index"]:
                raise EngineProtocolError("final observation card index does not match reset")
            advanced = max(0, final_episode_frame - initial_episode_frame)
            item.update({
                "advanced_frames": advanced,
                "step_requests": step_requests,
                "frame_hashes": frame_hashes,
                "final_state_hash": frame_hashes[-1],
                "terminated": observation.get("terminated") is True,
                "termination_reason": observation.get("termination_reason"),
                "counts": _count_snapshot(observation),
                "peak_counts": peak_counts,
                "seen_enemy": seen_enemy,
                "seen_threat": seen_threat,
                "seen_active_content": seen_active_content,
                "first_active_content_frame": first_active_content_frame,
            })
            if advanced != frames_per_attack:
                raise EngineProtocolError(
                    f"attack advanced {advanced} frames; expected {frames_per_attack}",
                )
            if len(frame_hashes) != frames_per_attack + 1:
                raise EngineProtocolError("attack does not contain one hash per logical frame")
            if not seen_active_content:
                raise EngineProtocolError("attack never exposed both a boss and an active hazard")
            item.update({
                "passed": True,
            })
        except (EngineProtocolError, OSError, TypeError, ValueError) as error:
            item["error"] = str(error)
        results.append(item)

    count_matches = expected_attacks is None or len(attacks) == expected_attacks
    passed_count = sum(item["passed"] is True for item in results)
    errors = list(identity_errors)
    if not count_matches:
        errors.append(f"catalog contains {len(attacks)} attacks; expected {expected_attacks}")
    if catalog.get("attack_count") != len(attacks):
        errors.append("catalog attack_count does not match its attacks array")
    if passed_count != len(results):
        errors.append(f"{len(results) - passed_count} attack regression(s) failed")
    session_id = ping.get("session_id")
    process_nonce = ping.get("process_nonce")
    if not isinstance(session_id, str) or not session_id:
        errors.append("engine ping has no session id")
    if not isinstance(process_nonce, str) or not process_nonce:
        errors.append("engine ping has no process nonce")
    if ping.get("protocol") != 2:
        errors.append(f"engine protocol is {ping.get('protocol')!r}; expected 2")
    scenario_count = len({str(value["scenario"]) for value in attacks})
    if catalog.get("scenario_count") != scenario_count:
        errors.append("catalog scenario_count does not match its attacks array")
    return {
        "schema_version": 2,
        "implementation_sha256": source_tree_sha256(),
        "run_kind": "live_luastg_spell_practice_regression",
        "engine_verified": False,
        "verification_scope": "live_attack_catalog_per_frame_content_regression",
        "passed": not errors,
        "host_protocol": ping.get("protocol"),
        "engine_session_id": session_id,
        "engine_process_nonce": process_nonce,
        "engine_process_id": runtime_identity.get("process_id"),
        "runtime_identity": runtime_identity,
        "local_source_sha256": local_sha256,
        "catalog": {
            "schema_version": catalog.get("schema_version"),
            "scenario_count": scenario_count,
            "attack_count": len(attacks),
            "expected_attacks": expected_attacks,
            "count_matches": count_matches,
            "scenarios": catalog.get("scenarios"),
            "attacks": list(attacks),
        },
        "config": {
            "seed": int(seed),
            "player": player,
            "frames_per_attack": frames_per_attack,
            "step_batch": step_batch,
            "action": action.to_dict(),
            "reset_options": {
                "lifeleft": 99,
                "player_protect_frames": frames_per_attack + 600,
            },
        },
        "passed_attacks": passed_count,
        "failed_attacks": len(results) - passed_count,
        "active_content_attacks": sum(
            item.get("seen_active_content") is True for item in results
        ),
        "attacks": results,
        "errors": errors,
        "elapsed_seconds": time.perf_counter() - started,
    }


__all__ = ["run_engine_benchmark"]
