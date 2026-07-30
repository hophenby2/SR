"""Cross-process acceptance for live LuaSTG attack-regression reports."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping

from .engine_benchmark import _local_runtime_sources
from .provenance import source_tree_sha256


STRICT_EXPECTED_ATTACKS = 53
STRICT_EXPECTED_SCENARIOS = 22
MIN_FRAMES_PER_ATTACK = 300
REPORT_SCHEMA_VERSION = 2
BRIDGE_PROTOCOL_VERSION = 2
CATALOG_SCHEMA_VERSION = 1
BENCHMARK_RUN_KIND = "live_luastg_spell_practice_regression"
BENCHMARK_SCOPE = "live_attack_catalog_per_frame_content_regression"
_STATE_HASH = re.compile(r"[0-9a-f]{32}")
_CRC32 = re.compile(r"[0-9a-f]{8}")


def _canonical_digest(value: Mapping[str, Any]) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _is_int(value: Any, *, minimum: int | None = None) -> bool:
    if isinstance(value, bool) or not isinstance(value, int):
        return False
    return minimum is None or value >= minimum


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _valid_hash(value: Any) -> bool:
    return isinstance(value, str) and _STATE_HASH.fullmatch(value) is not None


def _validate_report(
    report: Mapping[str, Any],
    label: str,
    expected_attacks: int,
) -> tuple[dict[tuple[str, int], Mapping[str, Any]], list[str]]:
    errors: list[str] = []
    schema_version = report.get("schema_version")
    if not _is_int(schema_version) or schema_version != REPORT_SCHEMA_VERSION:
        errors.append(f"{label} has an invalid schema version")
    if report.get("run_kind") != BENCHMARK_RUN_KIND:
        errors.append(f"{label} has an invalid run kind")
    if report.get("verification_scope") != BENCHMARK_SCOPE:
        errors.append(f"{label} has an invalid verification scope")
    if report.get("implementation_sha256") != source_tree_sha256():
        errors.append(f"{label} implementation fingerprint is stale or missing")
    host_protocol = report.get("host_protocol")
    if not _is_int(host_protocol) or host_protocol != BRIDGE_PROTOCOL_VERSION:
        errors.append(f"{label} has an invalid bridge protocol version")
    if report.get("passed") is not True:
        errors.append(f"{label} live engine regression did not pass")

    runtime_identity = report.get("runtime_identity")
    expected_crc32, expected_sha256 = _local_runtime_sources()
    if not isinstance(runtime_identity, Mapping):
        errors.append(f"{label} has no runtime identity")
    else:
        process_id = runtime_identity.get("process_id")
        if not _is_int(process_id, minimum=1):
            errors.append(f"{label} has no operating-system process id")
        executable = runtime_identity.get("executable_crc32")
        if not isinstance(executable, str) or _CRC32.fullmatch(executable) is None:
            errors.append(f"{label} has no executable checksum")
        sources = runtime_identity.get("source_crc32")
        if not isinstance(sources, Mapping) or not sources:
            errors.append(f"{label} has no runtime source fingerprints")
        elif dict(sources) != expected_crc32:
            errors.append(f"{label} runtime Lua fingerprints differ from the current worktree")
        if report.get("engine_process_id") != process_id:
            errors.append(f"{label} process-id summary differs from runtime identity")
    local_sources = report.get("local_source_sha256")
    if not isinstance(local_sources, Mapping) or not local_sources:
        errors.append(f"{label} has no local source fingerprints")
    elif dict(local_sources) != expected_sha256:
        errors.append(f"{label} local Lua fingerprints differ from the current worktree")

    config = report.get("config")
    frames_per_attack: int | None = None
    base_seed: int | None = None
    if not isinstance(config, Mapping):
        errors.append(f"{label} has no test configuration")
    else:
        step_batch = config.get("step_batch")
        if not _is_int(step_batch) or step_batch != 1:
            errors.append(f"{label} did not hash every logical frame")
        raw_frames = config.get("frames_per_attack")
        if not _is_int(raw_frames, minimum=MIN_FRAMES_PER_ATTACK):
            errors.append(
                f"{label} frames_per_attack must be at least {MIN_FRAMES_PER_ATTACK}",
            )
        else:
            frames_per_attack = raw_frames
        raw_seed = config.get("seed")
        if not _is_int(raw_seed, minimum=0):
            errors.append(f"{label} has an invalid base seed")
        else:
            base_seed = raw_seed
        if not _nonempty_string(config.get("player")):
            errors.append(f"{label} has an invalid player configuration")
        action = config.get("action")
        if action != {
            "move_x": 0,
            "move_y": 0,
            "slow": True,
            "shoot": False,
            "spell": False,
        }:
            errors.append(f"{label} has an invalid action configuration")
        reset_options = config.get("reset_options")
        expected_protection = frames_per_attack + 600 if frames_per_attack is not None else None
        if not isinstance(reset_options, Mapping) or (
            reset_options.get("lifeleft") != 99
            or reset_options.get("player_protect_frames") != expected_protection
        ):
            errors.append(f"{label} has an invalid reset configuration")

    catalog = report.get("catalog")
    catalog_attacks: Any = None
    catalog_scenarios: Any = None
    if not isinstance(catalog, Mapping):
        errors.append(f"{label} has no catalog summary")
    else:
        catalog_schema = catalog.get("schema_version")
        if not _is_int(catalog_schema) or catalog_schema != CATALOG_SCHEMA_VERSION:
            errors.append(f"{label} catalog has an invalid schema version")
        attack_count = catalog.get("attack_count")
        if not _is_int(attack_count) or attack_count != expected_attacks:
            errors.append(f"{label} catalog does not contain {expected_attacks} attacks")
        catalog_expected = catalog.get("expected_attacks")
        if not _is_int(catalog_expected) or catalog_expected != expected_attacks:
            errors.append(f"{label} catalog used a different expected attack count")
        if catalog.get("count_matches") is not True:
            errors.append(f"{label} catalog count did not match")
        if catalog.get("scenario_count") != STRICT_EXPECTED_SCENARIOS:
            errors.append(
                f"{label} catalog does not contain {STRICT_EXPECTED_SCENARIOS} scenarios",
            )
        catalog_attacks = catalog.get("attacks")
        catalog_scenarios = catalog.get("scenarios")
        if not isinstance(catalog_attacks, list) or len(catalog_attacks) != expected_attacks:
            errors.append(f"{label} does not retain the complete catalog attack list")
        if not isinstance(catalog_scenarios, list):
            errors.append(f"{label} does not retain the complete catalog scenario list")

    passed_attacks = report.get("passed_attacks")
    if not _is_int(passed_attacks) or passed_attacks != expected_attacks:
        errors.append(f"{label} did not pass all {expected_attacks} attacks")
    failed_attacks = report.get("failed_attacks")
    if not _is_int(failed_attacks) or failed_attacks != 0:
        errors.append(f"{label} contains failed attacks")
    active_content_attacks = report.get("active_content_attacks")
    if not _is_int(active_content_attacks) or active_content_attacks != expected_attacks:
        errors.append(f"{label} did not observe active content in all attacks")
    report_errors = report.get("errors")
    if not isinstance(report_errors, list) or report_errors:
        errors.append(f"{label} has a nonempty or invalid error list")

    values = report.get("attacks")
    if not isinstance(values, list):
        return {}, [*errors, f"{label} has no attacks array"]
    if len(values) != expected_attacks:
        errors.append(f"{label} contains {len(values)} attacks; expected {expected_attacks}")

    result: dict[tuple[str, int], Mapping[str, Any]] = {}
    for index, item in enumerate(values):
        prefix = f"{label} attack {index}"
        if not isinstance(item, Mapping):
            errors.append(f"{prefix} is not an object")
            continue

        scenario = item.get("scenario")
        attack = item.get("attack")
        valid_identity = _nonempty_string(scenario) and _is_int(attack, minimum=1)
        if not valid_identity:
            errors.append(f"{prefix} has an invalid identity")
            continue
        key = (scenario, attack)
        if key in result:
            errors.append(f"{label} duplicates {scenario} #{attack}")
        result[key] = item

        if not _is_int(item.get("card_index"), minimum=1):
            errors.append(f"{prefix} has an invalid card index")
        seed = item.get("seed")
        if not _is_int(seed, minimum=0):
            errors.append(f"{prefix} has an invalid seed")
        elif base_seed is not None and seed != base_seed + index:
            errors.append(f"{prefix} seed does not match catalog order")
        if item.get("passed") is not True:
            errors.append(f"{prefix} did not pass")
        if item.get("seen_active_content") is not True:
            errors.append(f"{prefix} did not observe active attack content")
        peak_counts = item.get("peak_counts")
        if not isinstance(peak_counts, Mapping):
            errors.append(f"{prefix} has no peak object counts")
        else:
            names = (
                "enemy_bullets",
                "enemies",
                "nontjt_enemies",
                "indestructibles",
                "lasers",
            )
            if not all(_is_int(peak_counts.get(name), minimum=0) for name in names):
                errors.append(f"{prefix} has invalid peak object counts")
            else:
                enemy_count = peak_counts["enemies"] + peak_counts["nontjt_enemies"]
                hazard_count = (
                    peak_counts["enemy_bullets"]
                    + peak_counts["indestructibles"]
                    + peak_counts["lasers"]
                )
                if enemy_count < 1 or (enemy_count < 2 and hazard_count < 1):
                    errors.append(f"{prefix} peak counts do not prove active attack content")
        first_active_frame = item.get("first_active_content_frame")
        if not _is_int(first_active_frame, minimum=0):
            errors.append(f"{prefix} has no valid first active-content frame")

        requested = item.get("requested_frames")
        advanced = item.get("advanced_frames")
        if not _is_int(requested, minimum=MIN_FRAMES_PER_ATTACK):
            errors.append(f"{prefix} has an invalid requested frame count")
        if frames_per_attack is not None and requested != frames_per_attack:
            errors.append(f"{prefix} requested frame count differs from the configuration")
        if not _is_int(advanced, minimum=MIN_FRAMES_PER_ATTACK) or advanced != requested:
            errors.append(f"{prefix} did not advance every requested frame")
        if (
            _is_int(first_active_frame, minimum=0)
            and _is_int(advanced, minimum=0)
            and first_active_frame > advanced
        ):
            errors.append(f"{prefix} active-content frame is outside the episode")
        if item.get("step_requests") != requested:
            errors.append(f"{prefix} did not issue one request per logical frame")
        if item.get("terminated") is not False or item.get("termination_reason") is not None:
            errors.append(f"{prefix} terminated before the validation window ended")

        hashes = item.get("frame_hashes")
        expected_hashes = requested + 1 if _is_int(requested, minimum=0) else None
        if not isinstance(hashes, list) or len(hashes) != expected_hashes:
            errors.append(f"{prefix} has the wrong number of per-frame hashes")
        elif not all(_valid_hash(value) for value in hashes):
            errors.append(f"{prefix} contains an invalid state hash")
        elif len(set(hashes)) < 2:
            errors.append(f"{prefix} contains a static per-frame state hash")
        if not isinstance(hashes, list) or not hashes or item.get("final_state_hash") != hashes[-1]:
            errors.append(f"{prefix} final state hash does not match")

    scenario_names = {key[0] for key in result}
    if len(scenario_names) != STRICT_EXPECTED_SCENARIOS:
        errors.append(
            f"{label} attacks contain {len(scenario_names)} distinct scenarios; "
            f"expected {STRICT_EXPECTED_SCENARIOS}"
        )
    if isinstance(catalog_attacks, list):
        catalog_identity = [
            (
                item.get("scenario"),
                item.get("attack"),
                item.get("card_index"),
                item.get("label"),
            )
            if isinstance(item, Mapping) else None
            for item in catalog_attacks
        ]
        report_identity = [
            (
                item.get("scenario"),
                item.get("attack"),
                item.get("card_index"),
                item.get("label"),
            )
            if isinstance(item, Mapping) else None
            for item in values
        ]
        if catalog_identity != report_identity:
            errors.append(f"{label} attack results do not preserve complete catalog order")
    if isinstance(catalog_scenarios, list):
        declared_names = []
        nested_identity = []
        for scenario_index, scenario_item in enumerate(catalog_scenarios):
            prefix = f"{label} catalog scenario {scenario_index}"
            if not isinstance(scenario_item, Mapping):
                errors.append(f"{prefix} is not an object")
                continue
            scenario_name = scenario_item.get("scenario")
            declared_names.append(scenario_name)
            nested_attacks = scenario_item.get("attacks")
            if not _nonempty_string(scenario_name):
                errors.append(f"{prefix} has an invalid identity")
            if not isinstance(nested_attacks, list):
                errors.append(f"{prefix} has no attacks array")
                continue
            if scenario_item.get("attack_count") != len(nested_attacks):
                errors.append(f"{prefix} attack count differs from its array")
            for attack_index, attack_item in enumerate(nested_attacks, start=1):
                if not isinstance(attack_item, Mapping):
                    errors.append(f"{prefix} attack {attack_index} is not an object")
                    nested_identity.append(None)
                    continue
                identity = (
                    attack_item.get("scenario"),
                    attack_item.get("attack"),
                    attack_item.get("card_index"),
                    attack_item.get("label"),
                )
                nested_identity.append(identity)
                if identity[0] != scenario_name or identity[1] != attack_index:
                    errors.append(f"{prefix} does not preserve nested attack order")
        if len(declared_names) != STRICT_EXPECTED_SCENARIOS or set(declared_names) != scenario_names:
            errors.append(f"{label} catalog scenario identities do not match attack results")
        if isinstance(catalog_attacks, list) and nested_identity != catalog_identity:
            errors.append(f"{label} nested catalog does not flatten to the complete attack list")

    return result, errors


def compare_engine_reports(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    *,
    expected_attacks: int = STRICT_EXPECTED_ATTACKS,
) -> dict[str, Any]:
    """Require complete per-frame evidence from two distinct engine processes."""

    errors: list[str] = []
    if expected_attacks != STRICT_EXPECTED_ATTACKS:
        errors.append(
            f"engine verification requires exactly {STRICT_EXPECTED_ATTACKS} attacks",
        )

    first_session = first.get("engine_session_id")
    second_session = second.get("engine_session_id")
    first_process = first.get("engine_process_nonce")
    second_process = second.get("engine_process_nonce")
    first_pid = first.get("engine_process_id")
    second_pid = second.get("engine_process_id")
    if not _nonempty_string(first_session):
        errors.append("first report has no engine session id")
    if not _nonempty_string(second_session):
        errors.append("second report has no engine session id")
    if _nonempty_string(first_session) and first_session == second_session:
        errors.append("engine reports came from the same session id")
    if not _nonempty_string(first_process):
        errors.append("first report has no engine process nonce")
    if not _nonempty_string(second_process):
        errors.append("second report has no engine process nonce")
    if _nonempty_string(first_process) and first_process == second_process:
        errors.append("engine reports came from the same process nonce")
    if not _is_int(first_pid, minimum=1) or not _is_int(second_pid, minimum=1):
        errors.append("engine reports do not contain operating-system process ids")
    elif first_pid == second_pid:
        errors.append("engine reports came from the same operating-system process")
    first_runtime = first.get("runtime_identity")
    second_runtime = second.get("runtime_identity")
    if isinstance(first_runtime, Mapping) and isinstance(second_runtime, Mapping):
        if first_runtime.get("executable_crc32") != second_runtime.get("executable_crc32"):
            errors.append("engine reports used different executable fingerprints")
        if first_runtime.get("source_crc32") != second_runtime.get("source_crc32"):
            errors.append("engine reports used different runtime source fingerprints")
    if first.get("local_source_sha256") != second.get("local_source_sha256"):
        errors.append("engine reports used different local source fingerprints")
    if first.get("config") != second.get("config"):
        errors.append("engine reports used different test configurations")

    first_attacks, first_errors = _validate_report(first, "first report", expected_attacks)
    second_attacks, second_errors = _validate_report(second, "second report", expected_attacks)
    errors.extend(first_errors)
    errors.extend(second_errors)
    if set(first_attacks) != set(second_attacks):
        errors.append("engine reports contain different attack identities")

    comparisons = []
    for key in sorted(set(first_attacks) & set(second_attacks)):
        left, right = first_attacks[key], second_attacks[key]
        item_errors = []
        for field, description in (
            ("card_index", "card index differs"),
            ("seed", "seed differs"),
            ("requested_frames", "requested frame count differs"),
            ("advanced_frames", "advanced frame count differs"),
        ):
            if left.get(field) != right.get(field):
                item_errors.append(description)
        first_hashes = left.get("frame_hashes")
        second_hashes = right.get("frame_hashes")
        valid_hashes = (
            isinstance(first_hashes, list)
            and isinstance(second_hashes, list)
            and bool(first_hashes)
            and all(_valid_hash(value) for value in first_hashes)
            and all(_valid_hash(value) for value in second_hashes)
        )
        if not valid_hashes:
            item_errors.append("per-frame observation hashes are invalid")
        elif first_hashes != second_hashes:
            item_errors.append("per-frame observation hashes differ")
        matched = not item_errors
        comparisons.append({
            "scenario": key[0],
            "attack": key[1],
            "matched": matched,
            "hash_count": len(first_hashes) if isinstance(first_hashes, list) else 0,
            "errors": item_errors,
        })
        errors.extend(f"{key[0]} #{key[1]}: {error}" for error in item_errors)

    passed = (
        not errors
        and expected_attacks == STRICT_EXPECTED_ATTACKS
        and len(comparisons) == STRICT_EXPECTED_ATTACKS
    )
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "implementation_sha256": source_tree_sha256(),
        "run_kind": "live_luastg_cross_process_acceptance",
        "passed": passed,
        "engine_verified": passed,
        "expected_attacks": STRICT_EXPECTED_ATTACKS,
        "sessions": [first_session, second_session],
        "process_nonces": [first_process, second_process],
        "process_ids": [first_pid, second_pid],
        "input_report_sha256": [_canonical_digest(first), _canonical_digest(second)],
        "matched_attacks": sum(item["matched"] for item in comparisons),
        "comparisons": comparisons,
        "errors": errors,
    }


__all__ = ["compare_engine_reports"]
