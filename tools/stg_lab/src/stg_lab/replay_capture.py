"""Shared validation for native THlib replay capture."""

from __future__ import annotations

import re
from typing import Any, Mapping, Protocol

from .engine import EngineProtocolError


_REPLAY_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}")
_WINDOWS_RESERVED_REPLAY_BASENAMES = frozenset({
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
})


class NativeReplayClient(Protocol):
    def save_replay(self, *, finish: bool, reason: str) -> dict[str, Any]: ...


def normalize_replay_name(value: str | None) -> str | None:
    """Return a portable replay basename, removing one optional .rep suffix."""

    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("replay_name must be a string")
    replay_name = value[:-4] if value.lower().endswith(".rep") else value
    if (
        _REPLAY_NAME_PATTERN.fullmatch(replay_name) is None
        or replay_name.endswith(".")
    ):
        raise ValueError("replay_name must be 1-96 portable filename characters")
    windows_basename = replay_name.partition(".")[0].upper()
    if windows_basename in _WINDOWS_RESERVED_REPLAY_BASENAMES:
        raise ValueError("replay_name uses a Windows reserved basename")
    return replay_name


def require_native_replay_capture(ping: Mapping[str, Any]) -> None:
    commands = ping.get("commands")
    if not isinstance(commands, list) or "save_replay" not in commands:
        raise EngineProtocolError(
            "engine bridge does not advertise native replay capture"
        )


def replay_start_metadata(
    response: Mapping[str, Any],
    *,
    expected_name: str,
    expected_episode_kind: str,
    expected_stage_name: str,
    expected_seed: int,
) -> dict[str, Any]:
    """Validate replay metadata returned by reset/reset_stage."""

    reset = response.get("reset")
    replay = reset.get("replay") if isinstance(reset, Mapping) else None
    if not isinstance(replay, Mapping):
        raise EngineProtocolError("engine reset did not start the requested replay")
    result = dict(replay)
    random_seed = result.get("random_seed")
    if (
        result.get("schema_version") != 1
        or result.get("name") != expected_name
        or result.get("episode_kind") != expected_episode_kind
        or result.get("stage_name") != expected_stage_name
        or random_seed != expected_seed
        or result.get("saved") is not False
        or not isinstance(result.get("path"), str)
        or not result["path"]
        or isinstance(random_seed, bool)
        or not isinstance(random_seed, int)
        or not isinstance(result.get("player"), str)
        or not result["player"]
    ):
        raise EngineProtocolError("engine returned invalid replay start metadata")
    return result


def saved_replay_metadata(
    response: Mapping[str, Any],
    *,
    expected_name: str,
    expected_episode_kind: str,
    expected_stage_name: str,
    expected_seed: int,
    expected_player: str,
    expected_path: str,
    expected_finish: bool,
    expected_reason: str,
) -> dict[str, Any]:
    """Validate the bridge's replay file and frame-byte verification evidence."""

    replay = response.get("replay")
    if not isinstance(replay, Mapping):
        raise EngineProtocolError("engine did not return saved replay metadata")
    result = dict(replay)
    frame_count = result.get("frame_count")
    frame_bytes_verified = result.get("frame_bytes_verified")
    file_size = result.get("file_size")
    crc32 = result.get("crc32")
    expected_group_finish = 1 if expected_finish else 0
    if (
        result.get("schema_version") != 1
        or result.get("name") != expected_name
        or result.get("episode_kind") != expected_episode_kind
        or result.get("stage_name") != expected_stage_name
        or result.get("random_seed") != expected_seed
        or result.get("player") != expected_player
        or result.get("path") != expected_path
        or result.get("finish") is not expected_finish
        or result.get("group_finish") != expected_group_finish
        or result.get("reason") != expected_reason
        or result.get("saved") is not True
        or result.get("verified") is not True
        or not isinstance(result.get("path"), str)
        or not result["path"]
        or isinstance(frame_count, bool)
        or not isinstance(frame_count, int)
        or frame_count <= 0
        or isinstance(frame_bytes_verified, bool)
        or not isinstance(frame_bytes_verified, int)
        or frame_bytes_verified != frame_count
        or isinstance(file_size, bool)
        or not isinstance(file_size, int)
        or file_size < frame_bytes_verified
        or not isinstance(crc32, str)
        or re.fullmatch(r"[0-9a-f]{8}", crc32) is None
    ):
        raise EngineProtocolError("engine returned invalid saved replay metadata")
    return result


def save_native_replay(
    client: NativeReplayClient,
    *,
    replay_name: str,
    replay_start: Mapping[str, Any],
    episode_kind: str,
    strict_success: bool,
    termination_reason: Any,
) -> dict[str, Any]:
    """Save an episode replay while keeping strict success separate from capture."""

    replay_reason = (
        termination_reason
        if isinstance(termination_reason, str) and termination_reason else
        "unknown"
    )
    replay_finish = episode_kind == "stage" and strict_success
    response = client.save_replay(finish=replay_finish, reason=replay_reason)
    return saved_replay_metadata(
        response,
        expected_name=replay_name,
        expected_episode_kind=episode_kind,
        expected_stage_name=str(replay_start["stage_name"]),
        expected_seed=int(replay_start["random_seed"]),
        expected_player=str(replay_start["player"]),
        expected_path=str(replay_start["path"]),
        expected_finish=replay_finish,
        expected_reason=replay_reason,
    )


__all__ = [
    "normalize_replay_name",
    "replay_start_metadata",
    "require_native_replay_capture",
    "save_native_replay",
    "saved_replay_metadata",
]
