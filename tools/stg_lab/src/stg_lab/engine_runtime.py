"""Bind live-engine reports to the local SR Lua sources under test."""

from __future__ import annotations

import os
from pathlib import Path
import re
from typing import Any, Mapping
import zlib

from .engine import EngineProtocolError
from .provenance import file_sha256


MOD_ROOT_ENV = "STG_LAB_MOD_ROOT"
RUNTIME_SOURCE_FILES = (
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
_SOURCE_LAYOUT_MOD_ROOT = Path(__file__).resolve().parents[4]
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


def resolve_mod_root() -> Path:
    """Locate the local SR tree whose Lua sources the engine must have loaded."""

    configured = os.environ.get(MOD_ROOT_ENV)
    if configured:
        root = Path(configured).expanduser().resolve()
        if not _looks_like_mod_root(root):
            raise FileNotFoundError(
                f"{MOD_ROOT_ENV} does not identify an SR mod root: {root}",
            )
        return root

    working = Path.cwd().resolve()
    candidates = (working, *working.parents, _SOURCE_LAYOUT_MOD_ROOT)
    for candidate in candidates:
        if _looks_like_mod_root(candidate):
            return candidate
    raise FileNotFoundError(
        f"cannot locate the SR mod root; set {MOD_ROOT_ENV} to its directory",
    )


def local_runtime_source_fingerprints() -> tuple[dict[str, str], dict[str, str]]:
    """Return CRC32 for engine parity and SHA-256 for retained report evidence."""

    mod_root = resolve_mod_root()
    crc32: dict[str, str] = {}
    sha256: dict[str, str] = {}
    for relative in RUNTIME_SOURCE_FILES:
        path = mod_root / relative
        crc32[relative] = _file_crc32(path)
        sha256[relative] = file_sha256(path)
    return crc32, sha256


def verify_runtime_source_fingerprints(ping: Mapping[str, Any]) -> dict[str, Any]:
    """Reject a live engine that did not load the local SR Lua source set."""

    expected_crc32, local_sha256 = local_runtime_source_fingerprints()
    runtime_identity = ping.get("runtime_identity")
    if not isinstance(runtime_identity, Mapping):
        raise EngineProtocolError("engine ping has no runtime identity")
    sources = runtime_identity.get("source_crc32")
    if not isinstance(sources, Mapping):
        raise EngineProtocolError("engine runtime identity has no Lua source fingerprints")
    actual_crc32 = dict(sources)
    if actual_crc32 != expected_crc32:
        missing = sorted(set(expected_crc32) - set(actual_crc32))
        unexpected = sorted(set(actual_crc32) - set(expected_crc32))
        changed = sorted(
            name for name in set(expected_crc32) & set(actual_crc32)
            if actual_crc32[name] != expected_crc32[name]
        )
        details = []
        if missing:
            details.append(f"missing={missing}")
        if unexpected:
            details.append(f"unexpected={unexpected}")
        if changed:
            details.append(f"changed={changed}")
        suffix = "; ".join(details) or "content differs"
        raise EngineProtocolError(
            "engine runtime Lua source fingerprints differ from the local test "
            f"sources ({suffix})",
        )
    if any(
        not isinstance(value, str) or _CRC32.fullmatch(value) is None
        for value in actual_crc32.values()
    ):
        raise EngineProtocolError("engine runtime Lua source fingerprints are invalid")
    return {
        "matched": True,
        "source_count": len(expected_crc32),
        "local_source_sha256": local_sha256,
    }


__all__ = [
    "MOD_ROOT_ENV",
    "RUNTIME_SOURCE_FILES",
    "local_runtime_source_fingerprints",
    "resolve_mod_root",
    "verify_runtime_source_fingerprints",
]
