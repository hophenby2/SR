"""Fail-closed primitives for one-shot formal experiment campaigns."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any

_SHA256_LENGTH = 64
_DIRECTORY_FSYNC_UNSUPPORTED = frozenset(
    error
    for error in (
        errno.EACCES,
        errno.EINVAL,
        getattr(errno, "ENOTSUP", None),
        getattr(errno, "EOPNOTSUPP", None),
        errno.EPERM,
    )
    if error is not None
)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _read_bytes(path: Path) -> bytes:
    return path.read_bytes()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _path_key(path: Path) -> str:
    # macOS commonly uses a case-insensitive filesystem even though its
    # Python platform is POSIX and normcase() therefore preserves case.
    # Conservatively folding here also rejects ambiguous formal paths on
    # case-sensitive filesystems.
    return os.path.normcase(os.path.normpath(os.fspath(path))).casefold()


def _path_spelling_key(path: Path) -> str:
    return os.path.normcase(os.fspath(path))


def _canonical_leaf(path: Path, *, label: str) -> Path:
    if not isinstance(path, Path):
        raise TypeError(f"{label} must be a pathlib.Path")
    if not path.is_absolute():
        raise ValueError(f"{label} must be absolute")
    try:
        parent = path.parent.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"{label} parent must exist") from error
    canonical = parent / path.name
    if _path_spelling_key(path) != _path_spelling_key(canonical):
        raise ValueError(
            f"{label} must be canonical and cannot traverse a parent symlink"
        )
    if path.is_symlink():
        raise ValueError(f"{label} cannot be a symbolic link")
    return canonical


def _require_existing_canonical_file(path: Path, *, label: str) -> Path:
    canonical = _canonical_leaf(path, label=label)
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"{label} must exist") from error
    if _path_key(resolved) != _path_key(canonical):
        raise ValueError(f"{label} cannot resolve through a symbolic link")
    if not path.is_file():
        raise ValueError(f"{label} must be a regular file")
    return canonical


def _same_existing_file(left: Path, right: Path) -> bool:
    try:
        return os.path.samefile(left, right)
    except FileNotFoundError:
        return False


def _reject_path_aliases(paths: Mapping[str, Path]) -> None:
    items = list(paths.items())
    keys: dict[str, str] = {}
    for name, path in items:
        key = _path_key(path)
        if key in keys:
            raise ValueError(f"{name} aliases {keys[key]}")
        keys[key] = name
    for index, (left_name, left) in enumerate(items):
        for right_name, right in items[index + 1 :]:
            if _same_existing_file(left, right):
                raise ValueError(f"{right_name} aliases {left_name}")


def _open_exclusive(path: Path) -> int:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    return os.open(path, flags, 0o600)


def _write_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if written <= 0:
            raise OSError("failed to write formal campaign file")
        offset += written


def _fsync_parent_directory(
    path: Path,
    *,
    platform: str | None = None,
) -> bool:
    """Persist a directory entry where the platform exposes that operation."""

    active_platform = os.name if platform is None else platform
    if active_platform == "nt":
        # Windows FlushFileBuffers on the file handle is the portable guarantee;
        # the CRT cannot open directory handles for os.fsync.
        return False
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path.parent, flags)
    except OSError as error:
        if error.errno in _DIRECTORY_FSYNC_UNSUPPORTED:
            return False
        raise
    try:
        try:
            os.fsync(descriptor)
        except OSError as error:
            if error.errno in _DIRECTORY_FSYNC_UNSUPPORTED:
                return False
            raise
    finally:
        os.close(descriptor)
    return True


@dataclass(frozen=True, slots=True)
class CampaignSpec:
    kind: str
    sequence: int
    output_path: Path
    ledger_path: Path
    preregistration_path: Path
    preregistration_sha256: str
    configuration_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or not self.kind:
            raise ValueError("campaign kind cannot be empty")
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int):
            raise TypeError("campaign sequence must be an integer")
        if self.sequence <= 0:
            raise ValueError("campaign sequence must be positive")
        for name in ("preregistration_sha256", "configuration_sha256"):
            if not _is_sha256(getattr(self, name)):
                raise ValueError(f"{name} must be a lowercase SHA-256")
        for name in ("output_path", "ledger_path", "preregistration_path"):
            path = getattr(self, name)
            if not isinstance(path, Path) or not path.is_absolute():
                raise ValueError(f"{name} must be an absolute pathlib.Path")
        paths = {
            "output": _canonical_leaf(self.output_path, label="output_path"),
            "ledger": _canonical_leaf(self.ledger_path, label="ledger_path"),
            "preregistration": _require_existing_canonical_file(
                self.preregistration_path,
                label="preregistration_path",
            ),
        }
        _reject_path_aliases(paths)

    def public_record(self) -> dict[str, Any]:
        values = asdict(self)
        for name in ("output_path", "ledger_path", "preregistration_path"):
            values[name] = str(values[name])
        return values


def validate_campaign_paths(
    spec: CampaignSpec,
    *,
    requested_output: Path,
    protected_inputs: Mapping[str, Path] | None = None,
) -> None:
    """Reject alternate output names and all known filesystem aliases."""

    output = _canonical_leaf(spec.output_path, label="output")
    requested = _canonical_leaf(requested_output, label="requested output")
    if _path_key(requested) != _path_key(output):
        raise ValueError("formal campaign output path is fixed")
    ledger = _canonical_leaf(spec.ledger_path, label="ledger")
    preregistration = _require_existing_canonical_file(
        spec.preregistration_path,
        label="preregistration",
    )
    paths = {
        "output": output,
        "ledger": ledger,
        "preregistration": preregistration,
    }
    if protected_inputs is not None:
        for name, path in protected_inputs.items():
            if name in paths:
                raise ValueError(f"reserved protected input name: {name}")
            paths[name] = _require_existing_canonical_file(
                path,
                label=f"protected input {name}",
            )
    _reject_path_aliases(paths)


def reserve_one_shot_campaign(
    spec: CampaignSpec,
    *,
    requested_output: Path | None = None,
    extra_ledger_fields: Mapping[str, Any] | None = None,
) -> str:
    """Permanently consume a campaign attempt before protected data is read."""

    validate_campaign_paths(
        spec,
        requested_output=(
            spec.output_path if requested_output is None else requested_output
        ),
    )
    if os.path.lexists(spec.output_path):
        raise FileExistsError(f"formal output already exists: {spec.output_path}")
    record: dict[str, Any] = {
        "schema": "formal_campaign_started/v1",
        "schema_version": 1,
        "kind": spec.kind,
        "record_kind": "formal_campaign_started",
        "start_utc": datetime.now(UTC).isoformat(),
        "pid": os.getpid(),
        "spec": spec.public_record(),
        "preregistration_expected_sha256": spec.preregistration_sha256,
        "failure_or_interruption_permanently_consumes_attempt": True,
    }
    if extra_ledger_fields:
        overlap = set(record) & set(extra_ledger_fields)
        if overlap:
            raise ValueError(
                "extra ledger fields overwrite reserved fields: "
                + ", ".join(sorted(overlap))
            )
        record.update(extra_ledger_fields)
    payload = canonical_json_bytes(record)
    descriptor = _open_exclusive(spec.ledger_path)
    try:
        _write_all(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_parent_directory(spec.ledger_path)
    return hashlib.sha256(payload).hexdigest()


def validate_preregistration(
    spec: CampaignSpec,
    *,
    read_bytes: Callable[[Path], bytes] = _read_bytes,
) -> dict[str, Any]:
    """Hash and parse the exact same preregistration byte string."""

    _require_existing_canonical_file(
        spec.preregistration_path,
        label="preregistration",
    )
    payload = read_bytes(spec.preregistration_path)
    if not isinstance(payload, bytes):
        raise TypeError("preregistration reader must return bytes")
    if hashlib.sha256(payload).hexdigest() != spec.preregistration_sha256:
        raise ValueError("formal preregistration hash differs")
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("formal preregistration is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise TypeError("formal preregistration root must be an object")
    if value.get("schema_version") != 1:
        raise ValueError("formal preregistration schema differs")
    if value.get("kind") != spec.kind:
        raise ValueError("formal preregistration kind differs")
    if value.get("adaptive_development_screen_sequence") != spec.sequence:
        raise ValueError("formal preregistration sequence differs")
    if value.get("configuration_sha256") != spec.configuration_sha256:
        raise ValueError("formal preregistration configuration hash differs")
    if canonical_json_sha256(value.get("configuration")) != spec.configuration_sha256:
        raise ValueError("formal preregistration configuration payload differs")
    _require_existing_canonical_file(
        spec.preregistration_path,
        label="preregistration",
    )
    return value


@dataclass(frozen=True, slots=True)
class IntegritySnapshot:
    paths: Mapping[str, Path]
    sha256: Mapping[str, str]

    def __post_init__(self) -> None:
        paths = dict(self.paths)
        sha256 = dict(self.sha256)
        if not paths or set(paths) != set(sha256):
            raise ValueError("integrity snapshot keys are inconsistent")
        canonical_paths = {
            name: _require_existing_canonical_file(
                path,
                label=f"integrity path {name}",
            )
            for name, path in paths.items()
        }
        _reject_path_aliases(canonical_paths)
        for name, value in sha256.items():
            if not _is_sha256(value):
                raise ValueError(f"integrity SHA-256 is invalid: {name}")
        object.__setattr__(self, "paths", MappingProxyType(canonical_paths))
        object.__setattr__(self, "sha256", MappingProxyType(sha256))

    @classmethod
    def capture(
        cls,
        paths: Mapping[str, Path],
        *,
        digest: Callable[[Path], str] = file_sha256,
    ) -> IntegritySnapshot:
        canonical_paths = {
            name: _require_existing_canonical_file(
                path,
                label=f"integrity path {name}",
            )
            for name, path in paths.items()
        }
        _reject_path_aliases(canonical_paths)
        return cls(
            paths=canonical_paths,
            sha256={name: digest(path) for name, path in canonical_paths.items()},
        )

    def reverify(
        self,
        *,
        digest: Callable[[Path], str] = file_sha256,
    ) -> None:
        changed: list[str] = []
        for name, path in self.paths.items():
            try:
                canonical = _require_existing_canonical_file(
                    path,
                    label=f"integrity path {name}",
                )
                current = digest(canonical)
            except (OSError, ValueError):
                changed.append(name)
                continue
            if current != self.sha256[name]:
                changed.append(name)
        if changed:
            raise ValueError(
                "protected campaign inputs changed: " + ", ".join(sorted(changed))
            )


@dataclass(frozen=True, slots=True)
class StartedCampaign:
    spec: CampaignSpec
    ledger_sha256: str
    preregistration: Mapping[str, Any]
    integrity: IntegritySnapshot

    def reverify_before_audit(
        self,
        *,
        digest: Callable[[Path], str] = file_sha256,
    ) -> None:
        self.integrity.reverify(digest=digest)


def start_guarded_campaign(
    spec: CampaignSpec,
    *,
    requested_output: Path,
    protected_inputs: Mapping[str, Path],
    read_preregistration: Callable[[Path], bytes] = _read_bytes,
    digest: Callable[[Path], str] = file_sha256,
) -> StartedCampaign:
    """Reserve first, then read preregistration and protected inputs."""

    ledger_sha256 = reserve_one_shot_campaign(
        spec,
        requested_output=requested_output,
    )
    preregistration = validate_preregistration(
        spec,
        read_bytes=read_preregistration,
    )
    validate_campaign_paths(
        spec,
        requested_output=requested_output,
        protected_inputs=protected_inputs,
    )
    captured = IntegritySnapshot.capture(protected_inputs, digest=digest)
    integrity = IntegritySnapshot(
        paths={
            "preregistration": spec.preregistration_path,
            **dict(captured.paths),
        },
        sha256={
            "preregistration": spec.preregistration_sha256,
            **dict(captured.sha256),
        },
    )
    return StartedCampaign(
        spec=spec,
        ledger_sha256=ledger_sha256,
        preregistration=MappingProxyType(preregistration),
        integrity=integrity,
    )


def write_json_exclusive(path: Path, value: Any) -> str:
    """Publish canonical JSON without ever replacing an existing path.

    A hard link publishes the complete fsynced temporary file atomically when
    supported.  Filesystems without hard links (notably some Windows mapped
    drives) fall back to an O_EXCL destination; a failed fallback deliberately
    leaves its partial destination in place so the attempt cannot be retried.
    """

    canonical = _canonical_leaf(path, label="output")
    payload = canonical_json_bytes(value)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=canonical.parent,
            prefix=f".{canonical.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())

        try:
            os.link(temporary, canonical)
        except FileExistsError:
            raise
        except OSError:
            descriptor = _open_exclusive(canonical)
            try:
                _write_all(descriptor, payload)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        _fsync_parent_directory(canonical)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return hashlib.sha256(payload).hexdigest()
