"""Content fingerprints used to bind reports to the implementation that ran."""

from __future__ import annotations

import hashlib
from pathlib import Path


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_tree_sha256() -> str:
    """Hash every shipped Python module in a stable name/content framing."""

    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(root.glob("*.py"), key=lambda value: value.name):
        name = path.name.encode("utf-8")
        content = path.read_bytes()
        digest.update(len(name).to_bytes(4, "big"))
        digest.update(name)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


__all__ = ["file_sha256", "source_tree_sha256"]
