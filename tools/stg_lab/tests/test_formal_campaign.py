from __future__ import annotations

import errno
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from experiments import formal_campaign as formal
from experiments.formal_campaign import (
    CampaignSpec,
    IntegritySnapshot,
    canonical_json_bytes,
    canonical_json_sha256,
    file_sha256,
    reserve_one_shot_campaign,
    start_guarded_campaign,
    validate_campaign_paths,
    validate_preregistration,
    write_json_exclusive,
)


def _spec(tmp_path: Path) -> tuple[CampaignSpec, dict[str, object]]:
    root = tmp_path.resolve(strict=True)
    configuration = {"epochs": 12, "seed": 17}
    configuration_sha256 = canonical_json_sha256(configuration)
    preregistration = {
        "schema_version": 1,
        "kind": "test_fifth_candidate_preregistration",
        "adaptive_development_screen_sequence": 5,
        "configuration_sha256": configuration_sha256,
        "configuration": configuration,
    }
    preregistration_path = root / "prereg.json"
    preregistration_path.write_bytes(canonical_json_bytes(preregistration))
    return (
        CampaignSpec(
            kind="test_fifth_candidate_preregistration",
            sequence=5,
            output_path=root / "output.json",
            ledger_path=root / "started.json",
            preregistration_path=preregistration_path,
            preregistration_sha256=file_sha256(preregistration_path),
            configuration_sha256=configuration_sha256,
        ),
        preregistration,
    )


def test_guard_reserves_before_any_input_reader_and_failure_is_permanent(
    tmp_path: Path,
) -> None:
    spec, _preregistration = _spec(tmp_path)
    protected = tmp_path.resolve() / "protected.bin"
    protected.write_bytes(b"protected")
    events: list[str] = []

    def failing_reader(_path: Path) -> bytes:
        assert spec.ledger_path.exists()
        events.append("read")
        raise RuntimeError("injected protected read failure")

    with pytest.raises(RuntimeError, match="injected"):
        start_guarded_campaign(
            spec,
            requested_output=spec.output_path,
            protected_inputs={"protected": protected},
            read_preregistration=failing_reader,
        )
    assert events == ["read"]
    ledger_bytes = spec.ledger_path.read_bytes()
    with pytest.raises(FileExistsError):
        reserve_one_shot_campaign(spec)
    assert spec.ledger_path.read_bytes() == ledger_bytes


def test_guard_captures_inputs_only_after_reservation_and_reverifies(
    tmp_path: Path,
) -> None:
    spec, preregistration = _spec(tmp_path)
    protected = tmp_path.resolve() / "protected.bin"
    protected.write_bytes(b"protected")
    reads: list[str] = []

    def prereg_reader(path: Path) -> bytes:
        assert spec.ledger_path.exists()
        reads.append("preregistration")
        return path.read_bytes()

    def digest(path: Path) -> str:
        assert spec.ledger_path.exists()
        reads.append(path.name)
        return file_sha256(path)

    started = start_guarded_campaign(
        spec,
        requested_output=spec.output_path,
        protected_inputs={"protected": protected},
        read_preregistration=prereg_reader,
        digest=digest,
    )
    assert reads == ["preregistration", "protected.bin"]
    assert dict(started.preregistration) == preregistration
    started.reverify_before_audit(digest=digest)
    protected.write_bytes(b"drift")
    with pytest.raises(ValueError, match="protected"):
        started.reverify_before_audit()


def test_concurrent_reservation_allows_exactly_one_writer(tmp_path: Path) -> None:
    spec, _preregistration = _spec(tmp_path)

    def attempt() -> bool:
        try:
            reserve_one_shot_campaign(spec)
        except FileExistsError:
            return False
        return True

    with ThreadPoolExecutor(max_workers=8) as executor:
        outcomes = list(executor.map(lambda _index: attempt(), range(8)))
    assert outcomes.count(True) == 1
    assert outcomes.count(False) == 7
    payload = json.loads(spec.ledger_path.read_text(encoding="utf-8"))
    assert payload["schema"] == "formal_campaign_started/v1"
    assert payload["kind"] == spec.kind
    assert payload["start_utc"].endswith("+00:00")
    assert payload["pid"] == os.getpid()
    assert payload["spec"]["sequence"] == spec.sequence
    assert (
        payload["preregistration_expected_sha256"]
        == spec.preregistration_sha256
    )
    assert payload["failure_or_interruption_permanently_consumes_attempt"] is True


def test_existing_output_is_preserved_without_consuming_ledger(
    tmp_path: Path,
) -> None:
    spec, _preregistration = _spec(tmp_path)
    spec.output_path.write_bytes(b"preserve")
    with pytest.raises(FileExistsError, match="already exists"):
        reserve_one_shot_campaign(spec)
    assert spec.output_path.read_bytes() == b"preserve"
    assert not spec.ledger_path.exists()


def test_fixed_output_and_noncanonical_paths_are_rejected(tmp_path: Path) -> None:
    spec, _preregistration = _spec(tmp_path)
    with pytest.raises(ValueError, match="fixed"):
        validate_campaign_paths(
            spec,
            requested_output=tmp_path.resolve() / "alternate.json",
        )

    nested = tmp_path.resolve() / "nested"
    nested.mkdir()
    with pytest.raises(ValueError, match="canonical"):
        CampaignSpec(
            kind=spec.kind,
            sequence=spec.sequence,
            output_path=nested / ".." / "same.json",
            ledger_path=tmp_path.resolve() / "same.json",
            preregistration_path=spec.preregistration_path,
            preregistration_sha256=spec.preregistration_sha256,
            configuration_sha256=spec.configuration_sha256,
        )


def test_uncreated_case_only_output_and_ledger_aliases_are_rejected(
    tmp_path: Path,
) -> None:
    spec, _preregistration = _spec(tmp_path)
    root = tmp_path.resolve()
    with pytest.raises(ValueError, match="aliases"):
        CampaignSpec(
            kind=spec.kind,
            sequence=spec.sequence,
            output_path=root / "Formal-Result.json",
            ledger_path=root / "formal-result.JSON",
            preregistration_path=spec.preregistration_path,
            preregistration_sha256=spec.preregistration_sha256,
            configuration_sha256=spec.configuration_sha256,
        )


def test_parent_symlink_is_rejected(tmp_path: Path) -> None:
    spec, _preregistration = _spec(tmp_path)
    root = tmp_path.resolve()
    actual = root / "actual"
    actual.mkdir()
    parent_link = root / "parent-link"
    try:
        parent_link.symlink_to(actual, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"symlinks are unavailable: {error}")
    with pytest.raises(ValueError, match="parent symlink"):
        CampaignSpec(
            kind=spec.kind,
            sequence=spec.sequence,
            output_path=parent_link / "output.json",
            ledger_path=spec.ledger_path,
            preregistration_path=spec.preregistration_path,
            preregistration_sha256=spec.preregistration_sha256,
            configuration_sha256=spec.configuration_sha256,
        )


def test_hardlink_alias_is_rejected(tmp_path: Path) -> None:
    spec, _preregistration = _spec(tmp_path)
    root = tmp_path.resolve()
    protected = root / "protected.bin"
    protected.write_bytes(b"protected")
    try:
        os.link(protected, spec.output_path)
    except OSError as error:
        pytest.skip(f"hard links are unavailable: {error}")
    with pytest.raises(ValueError, match="aliases"):
        validate_campaign_paths(
            spec,
            requested_output=spec.output_path,
            protected_inputs={"protected": protected},
        )


def test_integrity_snapshot_rejects_content_and_path_drift(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    first = root / "first.txt"
    second = root / "second.txt"
    first.write_text("one", encoding="utf-8")
    second.write_text("two", encoding="utf-8")
    snapshot = IntegritySnapshot.capture({"first": first, "second": second})
    snapshot.reverify()
    with pytest.raises(TypeError):
        snapshot.paths["third"] = root / "third.txt"  # type: ignore[index]
    second.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="second"):
        snapshot.reverify()

    second.unlink()
    same_content = root / "same-content.txt"
    same_content.write_text("two", encoding="utf-8")
    try:
        second.symlink_to(same_content)
    except OSError as error:
        pytest.skip(f"symlinks are unavailable: {error}")
    with pytest.raises(ValueError, match="second"):
        snapshot.reverify()


def test_preregistration_hashes_and_parses_the_same_single_read(
    tmp_path: Path,
) -> None:
    spec, preregistration = _spec(tmp_path)
    calls = 0

    def reader(path: Path) -> bytes:
        nonlocal calls
        calls += 1
        return path.read_bytes()

    assert validate_preregistration(spec, read_bytes=reader) == preregistration
    assert calls == 1

    tampered = dict(preregistration)
    tampered["configuration"] = {"epochs": 999, "seed": 17}
    with pytest.raises(ValueError, match="hash differs"):
        validate_preregistration(
            spec,
            read_bytes=lambda _path: canonical_json_bytes(tampered),
        )


def test_json_output_is_exclusive_without_hardlink_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path.resolve() / "result.json"
    value = {"b": 2, "a": [1, True]}

    def hardlink_unavailable(*_args: object, **_kwargs: object) -> None:
        raise OSError(errno.EOPNOTSUPP, "hard links unavailable")

    monkeypatch.setattr(os, "link", hardlink_unavailable)
    digest = write_json_exclusive(path, value)
    assert path.read_bytes() == canonical_json_bytes(value)
    assert file_sha256(path) == digest
    with pytest.raises(FileExistsError):
        write_json_exclusive(path, {"replacement": True})
    assert path.read_bytes() == canonical_json_bytes(value)


def test_json_output_fallback_failure_remains_exclusively_consumed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path.resolve() / "failed-result.json"

    def hardlink_unavailable(*_args: object, **_kwargs: object) -> None:
        raise OSError(errno.EOPNOTSUPP, "hard links unavailable")

    def write_failure(*_args: object, **_kwargs: object) -> None:
        raise OSError("injected write failure")

    monkeypatch.setattr(formal.os, "link", hardlink_unavailable)
    monkeypatch.setattr(formal, "_write_all", write_failure)
    with pytest.raises(OSError, match="injected"):
        write_json_exclusive(path, {"never": "complete"})
    assert path.exists()
    with pytest.raises(FileExistsError):
        write_json_exclusive(path, {"retry": True})


def test_directory_fsync_has_explicit_windows_and_unsupported_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path.resolve() / "target.json"
    assert formal._fsync_parent_directory(target, platform="nt") is False

    def unsupported_open(*_args: object, **_kwargs: object) -> int:
        raise OSError(errno.EINVAL, "directory fsync unavailable")

    monkeypatch.setattr(formal.os, "open", unsupported_open)
    assert formal._fsync_parent_directory(target, platform="posix") is False
