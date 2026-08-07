"""Build parent-filtered human plus one-or-more on-policy DAgger archives.

The historical filename is retained so existing single-source commands keep
working.  The builder itself is checkpoint- and source-agnostic.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from stg_lab.provenance import file_sha256, source_tree_sha256
from stg_lab.training import Demonstrations, load_checkpoint


DEFAULT_HUMAN_DATASET = Path(
    "artifacts/native-humanlike-v37-human-only-misclassified-v1.npz"
)
DEFAULT_DAGGER_DATASET = Path(
    "artifacts/context-humanlike-v37-seed10292-safety-margin20-gain0-v3.npz"
)
DEFAULT_CHECKPOINT = Path(
    "artifacts/policy-humanlike-highres-okuu3-v37-headonly-harderrors-kl20-ft60.pt"
)
DEFAULT_OUTPUT = Path(
    "artifacts/native-humanlike-v37-human-onpolicy-dagger-margin20-gain0-"
    "misclassified-v1.npz"
)
DEFAULT_HUMAN_EPISODE_IDS = tuple(range(7))
DEFAULT_VALIDATION_HUMAN_EPISODE_IDS = (3,)

SELECTION_MASK_FIELDS = frozenset(("supervision_mask", "correction_mask"))
DEMONSTRATION_FIELDS = tuple(Demonstrations.__dataclass_fields__)
HUMAN_MASK_MODES = ("preserve", "parent-misclassified-supervision")
DAGGER_LABEL_MODES = ("hard-corrections", "soft-evaluations")


def _manifest_path(dataset: Path) -> Path:
    return dataset.with_suffix(".manifest.json")


def _load_json_object(path: Path, *, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing {description}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{description} must contain a JSON object: {path}")
    return value


def _load_manifest(path: Path) -> dict[str, Any]:
    return _load_json_object(path, description="source manifest")


def _require_bound_file(
    manifest: Mapping[str, Any],
    *,
    path_field: str,
    sha256_field: str,
) -> Path:
    raw_path = manifest.get(path_field)
    recorded_sha256 = manifest.get(sha256_field)
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"manifest is missing {path_field}")
    if not isinstance(recorded_sha256, str) or len(recorded_sha256) != 64:
        raise ValueError(f"manifest is missing a valid {sha256_field}")
    path = Path(raw_path)
    actual_sha256 = file_sha256(path)
    if actual_sha256 != recorded_sha256:
        raise ValueError(
            f"{path_field} hash mismatch: recorded {recorded_sha256}, "
            f"actual {actual_sha256}"
        )
    return path


def _verify_dataset_binding(
    dataset: Path,
    manifest: Mapping[str, Any],
) -> None:
    recorded_sha256 = manifest.get("output_sha256", manifest.get("dataset_sha256"))
    if recorded_sha256 != file_sha256(dataset):
        raise ValueError(f"source dataset does not match its manifest: {dataset}")


def _verify_archive_schema(path: Path) -> None:
    with np.load(path) as archive:
        unknown = sorted(set(archive.files).difference(DEMONSTRATION_FIELDS))
    if unknown:
        raise ValueError(
            f"archive contains unsupported fields that cannot be preserved: {unknown}"
        )


def _episode_ids_in_order(values: Demonstrations) -> tuple[int, ...]:
    if values.episode_ids is None or len(values.episode_ids) == 0:
        raise ValueError("every source must contain nonempty episode_ids")
    ids = np.asarray(values.episode_ids, dtype=np.int64)
    ordered = tuple(dict.fromkeys(int(value) for value in ids.tolist()))
    for episode_id in ordered:
        indices = np.flatnonzero(ids == episode_id)
        if not np.array_equal(indices, np.arange(indices[0], indices[-1] + 1)):
            raise ValueError(f"source episode {episode_id} is not contiguous")
    return ordered


def _copy_selected(
    values: Demonstrations,
    selected: np.ndarray,
) -> Demonstrations:
    if selected.shape != (len(values.actions),):
        raise ValueError("selection mask must contain one value per decision")
    kwargs: dict[str, np.ndarray | None] = {}
    for name in DEMONSTRATION_FIELDS:
        value = getattr(values, name)
        kwargs[name] = None if value is None else np.asarray(value)[selected].copy()
    result = Demonstrations(**kwargs)
    result.validate()
    return result


def _select_human_episodes(
    values: Demonstrations,
    requested_episode_ids: Sequence[int],
) -> tuple[Demonstrations, tuple[int, ...]]:
    if not requested_episode_ids or len(set(requested_episode_ids)) != len(
        requested_episode_ids
    ):
        raise ValueError("human episode ids must be a nonempty unique list")
    source_order = _episode_ids_in_order(values)
    requested = set(int(value) for value in requested_episode_ids)
    missing = requested.difference(source_order)
    if missing:
        raise ValueError(f"human source is missing episodes {sorted(missing)}")
    selected_order = tuple(value for value in source_order if value in requested)
    assert values.episode_ids is not None
    selected = np.isin(values.episode_ids, selected_order)
    return _copy_selected(values, selected), selected_order


def _copy_with_episode_mapping(
    values: Demonstrations,
    mapping: Mapping[int, int],
) -> Demonstrations:
    kwargs: dict[str, np.ndarray | None] = {}
    for name in DEMONSTRATION_FIELDS:
        value = getattr(values, name)
        if value is None:
            kwargs[name] = None
        elif name == "episode_ids":
            kwargs[name] = np.asarray(
                [mapping[int(episode_id)] for episode_id in value],
                dtype=np.int64,
            )
        else:
            kwargs[name] = np.asarray(value).copy()
    result = Demonstrations(**kwargs)
    result.validate()
    return result


def _concatenate_demonstrations(values: Sequence[Demonstrations]) -> Demonstrations:
    if not values:
        raise ValueError("at least one demonstration source is required")
    kwargs: dict[str, np.ndarray | None] = {}
    for name in DEMONSTRATION_FIELDS:
        fields = [getattr(item, name) for item in values]
        if all(value is None for value in fields):
            kwargs[name] = None
            continue
        if any(value is None for value in fields):
            raise ValueError(f"optional field mismatch across sources for {name}")
        reference = np.asarray(fields[0])
        for value in fields[1:]:
            candidate = np.asarray(value)
            if candidate.dtype != reference.dtype:
                raise ValueError(f"source dtype mismatch for {name}")
            if candidate.shape[1:] != reference.shape[1:]:
                raise ValueError(f"source trailing-shape mismatch for {name}")
        kwargs[name] = np.concatenate(fields, axis=0)  # type: ignore[arg-type]
    result = Demonstrations(**kwargs)
    result.validate()
    return result


def _merge_sources(
    human: Demonstrations,
    daggers: Sequence[Demonstrations],
    *,
    human_episode_ids: Sequence[int],
) -> tuple[
    Demonstrations,
    Demonstrations,
    list[dict[str, int]],
    list[list[dict[str, int]]],
    list[tuple[int, int]],
]:
    human_selected, human_source_ids = _select_human_episodes(
        human, human_episode_ids,
    )
    human_mapping = {
        source_id: output_id
        for output_id, source_id in enumerate(human_source_ids)
    }
    chunks = [_copy_with_episode_mapping(human_selected, human_mapping)]
    human_mappings = [
        {
            "source_episode_id": source_id,
            "output_episode_id": output_id,
            "decisions": int(np.count_nonzero(
                human_selected.episode_ids == source_id
            )),
        }
        for source_id, output_id in human_mapping.items()
    ]
    source_spans: list[tuple[int, int]] = [(0, len(human_selected.actions))]
    next_episode_id = len(human_mapping)
    next_row = len(human_selected.actions)
    dagger_mappings: list[list[dict[str, int]]] = []
    for dagger in daggers:
        source_ids = _episode_ids_in_order(dagger)
        mapping = {
            source_id: next_episode_id + offset
            for offset, source_id in enumerate(source_ids)
        }
        remapped = _copy_with_episode_mapping(dagger, mapping)
        chunks.append(remapped)
        mappings = [
            {
                "source_episode_id": source_id,
                "output_episode_id": mapping[source_id],
                "decisions": int(np.count_nonzero(
                    dagger.episode_ids == source_id
                )),
            }
            for source_id in source_ids
        ]
        dagger_mappings.append(mappings)
        source_spans.append((next_row, next_row + len(dagger.actions)))
        next_row += len(dagger.actions)
        next_episode_id += len(source_ids)

    result = _concatenate_demonstrations(chunks)
    assert result.episode_ids is not None
    if not np.array_equal(np.unique(result.episode_ids), np.arange(next_episode_id)):
        raise RuntimeError("merged archive episode ids are not contiguous")
    if result.actions.shape[1] != 1:
        raise ValueError("streaming DAgger merge currently requires history=1")
    return (
        result,
        human_selected,
        human_mappings,
        dagger_mappings,
        source_spans,
    )


def _validate_strict_dagger_source(
    dataset_path: Path,
    demonstrations: Demonstrations,
) -> dict[str, Any]:
    context_manifest_path = _manifest_path(dataset_path)
    context_manifest = _load_manifest(context_manifest_path)
    _verify_dataset_binding(dataset_path, context_manifest)
    if context_manifest.get("run_kind") != "scenario_context_annotation":
        raise ValueError(f"DAgger source is not contextualized: {dataset_path}")

    native_manifest_path = _require_bound_file(
        context_manifest,
        path_field="source_manifest",
        sha256_field="source_manifest_sha256",
    )
    native_manifest = _load_manifest(native_manifest_path)
    native_dataset_path = _require_bound_file(
        context_manifest,
        path_field="source",
        sha256_field="source_sha256",
    )
    _verify_dataset_binding(native_dataset_path, native_manifest)
    report_path = _require_bound_file(
        native_manifest,
        path_field="source_dagger_report",
        sha256_field="source_dagger_report_sha256",
    )
    report = _load_json_object(report_path, description="DAgger report")

    source_episode_ids = _episode_ids_in_order(demonstrations)
    accepted = native_manifest.get("accepted_episodes")
    outcome = native_manifest.get("source_outcome_evidence")
    context_episodes = context_manifest.get("episode_contexts")
    if (
        not isinstance(accepted, list)
        or len(accepted) != len(source_episode_ids)
        or not isinstance(context_episodes, list)
        or len(context_episodes) != len(source_episode_ids)
    ):
        raise ValueError("DAgger manifests do not align with dataset episodes")
    outcome_death = outcome.get("final_player_death") if isinstance(outcome, dict) else None
    if (
        not isinstance(outcome, dict)
        or outcome.get("termination_reason") != "attack_complete"
        or isinstance(outcome_death, bool)
        or outcome_death != 0.0
        or outcome.get("terminated") is not True
    ):
        raise ValueError("DAgger relabel manifest lacks attack_complete/death=0")

    for source_episode_id, accepted_episode, context_episode in zip(
        source_episode_ids, accepted, context_episodes, strict=True,
    ):
        if (
            not isinstance(accepted_episode, dict)
            or accepted_episode.get("strict_success") is not True
            or accepted_episode.get("termination_reason") != "attack_complete"
            or not isinstance(context_episode, dict)
            or context_episode.get("episode_kind") != "attack"
        ):
            raise ValueError("every DAgger episode must be a strict successful attack")
        decisions = int(np.count_nonzero(
            demonstrations.episode_ids == source_episode_id
        ))
        if accepted_episode.get("decisions") != decisions:
            raise ValueError("accepted DAgger episode decision count mismatch")
        for identity_field in ("scenario", "attack", "seed"):
            if accepted_episode.get(identity_field) != context_episode.get(identity_field):
                raise ValueError(
                    f"DAgger context identity mismatch for {identity_field}"
                )

    final_player = (
        report.get("outcome_evidence", {}).get("final_player", {})
        if isinstance(report.get("outcome_evidence"), dict) else {}
    )
    report_death = final_player.get("death") if isinstance(final_player, dict) else None
    if (
        report.get("run_kind") != "live_luastg_native_dagger"
        or report.get("success") is not True
        or report.get("passed") is not True
        or report.get("terminated") is not True
        or report.get("termination_reason") != "attack_complete"
        or report.get("engine_termination_reason") != "attack_complete"
        or not isinstance(final_player, dict)
        or isinstance(report_death, bool)
        or report_death != 0
        or report.get("decision_count") != len(demonstrations.actions)
    ):
        raise ValueError("DAgger report is not a strict attack_complete/death=0 run")
    for accepted_episode in accepted:
        if (
            report.get("episode_kind") != accepted_episode.get("episode_kind")
            or report.get("scenario") != accepted_episode.get("scenario")
            or report.get("attack") != accepted_episode.get("attack")
            or report.get("seed") != accepted_episode.get("seed")
        ):
            raise ValueError("DAgger report identity does not match accepted episode")

    return {
        "dataset": dataset_path,
        "context_manifest_path": context_manifest_path,
        "context_manifest": context_manifest,
        "native_dataset_path": native_dataset_path,
        "native_manifest_path": native_manifest_path,
        "native_manifest": native_manifest,
        "report_path": report_path,
        "report": report,
        "accepted_episodes": accepted,
        "source_outcome_evidence": outcome,
        "context_episodes": context_episodes,
        "source_episode_ids": source_episode_ids,
    }


def _streaming_argmax(
    values: Demonstrations,
    checkpoint: Path,
    *,
    chunk_length: int,
    device: str,
) -> tuple[np.ndarray, Any]:
    if values.episode_ids is None:
        raise ValueError("episode_ids are required for streaming inference")
    if chunk_length <= 0:
        raise ValueError("chunk_length must be positive")
    model, _checkpoint_payload = load_checkpoint(checkpoint, device=device)
    if model.config.inference_mode != "stream":
        raise ValueError("parent checkpoint must use stream inference")
    if values.memory is None and model.config.memory_size:
        raise ValueError("archive memory is required by the parent checkpoint")
    if values.memory is not None and values.memory.shape[-1] != model.config.memory_size:
        raise ValueError("archive memory width does not match the parent checkpoint")
    if (
        values.proficiency is not None
        and values.proficiency.shape[-1] != model.config.proficiency_size
    ):
        raise ValueError("archive proficiency width does not match the checkpoint")

    predictions = np.full(values.actions.shape, -1, dtype=np.int64)
    model.eval()
    with torch.inference_mode():
        for episode_id in _episode_ids_in_order(values):
            indices = np.flatnonzero(values.episode_ids == episode_id)
            hidden = None
            for offset in range(0, len(indices), chunk_length):
                selected = indices[offset:offset + chunk_length]
                global_frames = torch.as_tensor(
                    values.global_frames[selected, -1],
                    dtype=torch.float32,
                    device=device,
                )[None]
                local_frames = torch.as_tensor(
                    values.local_frames[selected, -1],
                    dtype=torch.float32,
                    device=device,
                )[None]
                memory = (
                    None if values.memory is None else torch.as_tensor(
                        values.memory[selected, -1],
                        dtype=torch.float32,
                        device=device,
                    )[None]
                )
                proficiency = (
                    None if values.proficiency is None else torch.as_tensor(
                        values.proficiency[selected, -1],
                        dtype=torch.float32,
                        device=device,
                    )[None]
                )
                logits, _risk, hidden = model(
                    global_frames,
                    local_frames,
                    memory,
                    proficiency,
                    hidden=hidden,
                )
                hidden = hidden.detach()
                predictions[selected, -1] = (
                    logits[0].argmax(dim=-1).detach().cpu().numpy()
                )
    if np.any(predictions < 0):
        raise RuntimeError("streaming inference left decisions without predictions")
    return predictions, model


def _array_metadata(path: Path) -> tuple[list[str], dict[str, list[int]], dict[str, str]]:
    with np.load(path) as archive:
        fields = sorted(archive.files)
        shapes = {name: list(archive[name].shape) for name in fields}
        dtypes = {name: str(archive[name].dtype) for name in fields}
    return fields, shapes, dtypes


def _verify_preserved_arrays(
    expected: Demonstrations,
    persisted: Demonstrations,
    *,
    mutable_fields: Iterable[str],
) -> list[str]:
    mutable = set(mutable_fields)
    preserved: list[str] = []
    for name in DEMONSTRATION_FIELDS:
        if name in mutable:
            continue
        expected_value = getattr(expected, name)
        persisted_value = getattr(persisted, name)
        if (expected_value is None) != (persisted_value is None):
            raise RuntimeError(f"optional field changed while saving: {name}")
        if expected_value is None:
            continue
        if not np.array_equal(expected_value, persisted_value, equal_nan=True):
            raise RuntimeError(f"preserved array changed while saving: {name}")
        preserved.append(name)
    return sorted(preserved)


def _verify_source_span(
    persisted: Demonstrations,
    expected: Demonstrations,
    span: tuple[int, int],
    episode_mapping: Sequence[Mapping[str, int]],
    *,
    mutable_fields: Iterable[str],
) -> list[str]:
    mutable = set(mutable_fields)
    start, stop = span
    if stop - start != len(expected.actions):
        raise RuntimeError("source span length changed")
    mapping = {
        int(item["source_episode_id"]): int(item["output_episode_id"])
        for item in episode_mapping
    }
    verified: list[str] = []
    for name in DEMONSTRATION_FIELDS:
        if name in mutable:
            continue
        expected_value = getattr(expected, name)
        actual_value = getattr(persisted, name)
        if (expected_value is None) != (actual_value is None):
            raise RuntimeError(f"source optional field mismatch for {name}")
        if expected_value is None:
            continue
        expected_array = np.asarray(expected_value)
        if name == "episode_ids":
            expected_array = np.asarray(
                [mapping[int(value)] for value in expected_array],
                dtype=np.int64,
            )
        actual_array = np.asarray(actual_value)[start:stop]
        if actual_array.dtype != expected_array.dtype:
            raise RuntimeError(f"source span changed dtype for {name}")
        if not np.array_equal(
            actual_array, expected_array, equal_nan=True,
        ):
            raise RuntimeError(f"source span changed preserved array {name}")
        verified.append(name)
    return sorted(verified)


def _action_counts(actions: np.ndarray, mask: np.ndarray) -> dict[str, int]:
    return {
        str(action): int(count)
        for action, count in sorted(Counter(actions[mask].tolist()).items())
    }


def _argument_dagger_paths(arguments: argparse.Namespace) -> list[Path]:
    values = getattr(arguments, "dagger_datasets", None)
    if values is None:
        legacy = getattr(arguments, "dagger_dataset", None)
        values = [legacy] if legacy is not None else [DEFAULT_DAGGER_DATASET]
    if not values:
        values = [DEFAULT_DAGGER_DATASET]
    paths = [Path(value) for value in values]
    resolved = [path.expanduser().resolve() for path in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError("DAgger dataset paths must be unique")
    return paths


def _verify_checkpoint_context(
    model: Any,
    dagger_provenance: Sequence[Mapping[str, Any]],
) -> None:
    model_vocabulary = getattr(model, "scenario_vocabulary", None)
    model_previous_action_size = getattr(model, "previous_action_size", 0)
    model_previous_action_offset = getattr(
        model,
        "previous_action_offset",
        len(model_vocabulary) if model_vocabulary is not None else 0,
    )
    expected_signature: tuple[Any, ...] | None = None
    for provenance in dagger_provenance:
        manifest = provenance["context_manifest"]
        vocabulary = manifest.get("scenario_vocabulary")
        if model_vocabulary is None or tuple(vocabulary or ()) != tuple(model_vocabulary):
            raise ValueError(
                "DAgger context vocabulary does not match parent checkpoint"
            )
        if (
            manifest.get("previous_action_size") != model_previous_action_size
            or manifest.get("previous_action_offset") != model_previous_action_offset
        ):
            raise ValueError(
                "DAgger previous-action context does not match parent checkpoint"
            )
        if manifest.get("proficiency_size") != model.config.proficiency_size:
            raise ValueError(
                "DAgger proficiency context does not match parent checkpoint"
            )
        signature = (
            tuple(vocabulary or ()),
            manifest.get("previous_action_offset"),
            manifest.get("previous_action_size"),
            manifest.get("proficiency_size"),
            tuple(manifest.get("proficiency_profiles") or ()),
            manifest.get("context_semantics"),
        )
        if expected_signature is None:
            expected_signature = signature
        elif signature != expected_signature:
            raise ValueError("DAgger contextual sources use incompatible semantics")


def _verify_human_context_values(
    model: Any,
    human: Demonstrations,
    daggers: Sequence[Demonstrations],
) -> dict[str, Any]:
    if human.memory is None:
        if model.config.memory_size:
            raise ValueError("human source lacks checkpoint identity context")
        return {"mode": "no_memory", "verified": True}
    if getattr(model, "previous_action_size", 0) != 0:
        return {
            "mode": "width_only_previous_action_conditioned",
            "verified": human.memory.shape[-1] == model.config.memory_size,
        }
    human_values = np.unique(
        human.memory.reshape(-1, human.memory.shape[-1]), axis=0,
    )
    dagger_memory = [
        dagger.memory.reshape(-1, dagger.memory.shape[-1])
        for dagger in daggers
        if dagger.memory is not None
    ]
    if not dagger_memory:
        raise ValueError("DAgger sources lack checkpoint identity context")
    dagger_values = np.unique(np.concatenate(dagger_memory, axis=0), axis=0)
    human_tokens = {tuple(float(item) for item in row) for row in human_values}
    dagger_tokens = {tuple(float(item) for item in row) for row in dagger_values}
    if not human_tokens.issubset(dagger_tokens):
        raise ValueError("human identity context is not represented by DAgger sources")
    return {
        "mode": "identity_token_values",
        "verified": True,
        "human_unique_tokens": [list(value) for value in sorted(human_tokens)],
        "dagger_unique_tokens": [list(value) for value in sorted(dagger_tokens)],
    }


def _reject_output_aliases(
    output: Path,
    manifest: Path,
    *,
    inputs: Iterable[Path],
) -> None:
    if output.suffix.lower() != ".npz":
        raise ValueError("output dataset must use the .npz suffix")
    if manifest.suffix.lower() != ".json":
        raise ValueError("output manifest must use the .json suffix")
    output_resolved = output.expanduser().resolve()
    manifest_resolved = manifest.expanduser().resolve()
    temporary_resolved = output.with_name(
        output.name + ".tmp.npz"
    ).expanduser().resolve()
    if output_resolved == manifest_resolved:
        raise ValueError("output dataset and manifest must be different paths")
    protected = {path.expanduser().resolve() for path in inputs}
    if (
        output_resolved in protected
        or manifest_resolved in protected
        or temporary_resolved in protected
    ):
        raise ValueError("output paths must not overwrite an input or provenance file")


def _capture_hashes(paths: Iterable[Path]) -> dict[Path, str]:
    return {
        path.expanduser().resolve(): file_sha256(path)
        for path in paths
    }


def _verify_stable_hashes(captured: Mapping[Path, str]) -> None:
    for path, expected in captured.items():
        if file_sha256(path) != expected:
            raise RuntimeError(f"input changed while constructing archive: {path}")


def _argument_episode_ids(
    arguments: argparse.Namespace,
    name: str,
    default: Sequence[int],
) -> tuple[int, ...]:
    values = getattr(arguments, name, None)
    return tuple(default if values is None else (int(value) for value in values))


def build(arguments: argparse.Namespace) -> dict[str, Any]:
    builder_path = Path(__file__).resolve()
    builder_sha256 = file_sha256(builder_path)
    implementation_sha256 = source_tree_sha256()
    checkpoint_sha256 = file_sha256(arguments.checkpoint)
    dagger_paths = _argument_dagger_paths(arguments)
    human_episode_ids = _argument_episode_ids(
        arguments, "human_episode_ids", DEFAULT_HUMAN_EPISODE_IDS,
    )
    validation_human_episode_ids = _argument_episode_ids(
        arguments,
        "validation_human_episode_ids",
        DEFAULT_VALIDATION_HUMAN_EPISODE_IDS,
    )
    human_mask_mode = getattr(arguments, "human_mask_mode", "preserve")
    dagger_label_mode = getattr(arguments, "dagger_label_mode", "hard-corrections")
    if human_mask_mode not in HUMAN_MASK_MODES:
        raise ValueError(f"unknown human mask mode {human_mask_mode}")
    if dagger_label_mode not in DAGGER_LABEL_MODES:
        raise ValueError(f"unknown DAgger label mode {dagger_label_mode}")
    if not set(validation_human_episode_ids).issubset(human_episode_ids):
        raise ValueError("validation human episodes must be selected human episodes")

    human_manifest_path = _manifest_path(arguments.human_dataset)
    human_manifest = _load_manifest(human_manifest_path)
    _verify_dataset_binding(arguments.human_dataset, human_manifest)
    _verify_archive_schema(arguments.human_dataset)
    human = Demonstrations.load(arguments.human_dataset)

    daggers: list[Demonstrations] = []
    dagger_provenance: list[dict[str, Any]] = []
    for dagger_path in dagger_paths:
        _verify_archive_schema(dagger_path)
        dagger = Demonstrations.load(dagger_path)
        daggers.append(dagger)
        dagger_provenance.append(
            _validate_strict_dagger_source(dagger_path, dagger)
        )

    manifest_path = arguments.manifest or _manifest_path(arguments.output)
    protected_paths = [
        arguments.human_dataset,
        human_manifest_path,
        arguments.checkpoint,
        builder_path,
    ]
    for provenance in dagger_provenance:
        protected_paths.extend((
            provenance["dataset"],
            provenance["context_manifest_path"],
            provenance["native_dataset_path"],
            provenance["native_manifest_path"],
            provenance["report_path"],
        ))
    _reject_output_aliases(
        arguments.output,
        manifest_path,
        inputs=protected_paths,
    )
    captured_input_hashes = _capture_hashes(protected_paths)

    (
        merged,
        human_selected,
        human_mappings,
        dagger_mappings,
        source_spans,
    ) = _merge_sources(
        human,
        daggers,
        human_episode_ids=human_episode_ids,
    )
    assert merged.episode_ids is not None
    assert merged.supervision_mask is not None
    assert merged.correction_mask is not None
    if merged.teacher_action_evaluation_mask is None:
        raise ValueError("teacher action evaluation masks are required")

    original_supervision = merged.supervision_mask.copy()
    original_corrections = merged.correction_mask.copy()
    original_teacher_evaluation_mask = (
        merged.teacher_action_evaluation_mask.copy()
    )
    predictions, model = _streaming_argmax(
        merged,
        arguments.checkpoint,
        chunk_length=arguments.chunk_length,
        device=arguments.device,
    )
    _verify_checkpoint_context(model, dagger_provenance)
    human_context_validation = _verify_human_context_values(
        model, human_selected, daggers,
    )
    disagreement = predictions != merged.actions

    human_rows = np.zeros(len(merged.actions), dtype=np.bool_)
    human_rows[slice(*source_spans[0])] = True
    dagger_rows = ~human_rows
    human_rows_2d = human_rows[:, None]
    dagger_rows_2d = dagger_rows[:, None]
    if human_mask_mode == "parent-misclassified-supervision":
        human_eligible = original_supervision
        human_supervision = human_eligible & disagreement & human_rows_2d
        human_corrections = human_supervision.copy()
    else:
        human_supervision = original_supervision & disagreement & human_rows_2d
        human_corrections = original_corrections & disagreement & human_rows_2d

    dagger_selected = original_corrections & disagreement & dagger_rows_2d
    if dagger_label_mode == "hard-corrections":
        dagger_supervision = original_supervision & disagreement & dagger_rows_2d
        dagger_corrections = dagger_selected
        merged.teacher_action_evaluation_mask = original_teacher_evaluation_mask
    else:
        dagger_supervision = np.zeros_like(original_supervision)
        dagger_corrections = np.zeros_like(original_corrections)
        merged.teacher_action_evaluation_mask = np.zeros_like(
            original_teacher_evaluation_mask,
        )
        merged.teacher_action_evaluation_mask[dagger_selected] = True

    merged.supervision_mask = human_supervision | dagger_supervision
    merged.correction_mask = human_corrections | dagger_corrections
    merged.validate()

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = arguments.output.with_name(arguments.output.name + ".tmp.npz")
    merged.save(temporary_output)
    temporary_output.replace(arguments.output)
    persisted = Demonstrations.load(arguments.output)
    mutable_fields = set(SELECTION_MASK_FIELDS)
    if dagger_label_mode == "soft-evaluations":
        mutable_fields.add("teacher_action_evaluation_mask")
    preserved_arrays = _verify_preserved_arrays(
        merged, persisted, mutable_fields=mutable_fields,
    )
    for name in mutable_fields:
        if not np.array_equal(getattr(persisted, name), getattr(merged, name)):
            raise RuntimeError(f"persisted mutable mask differs: {name}")

    source_array_checks: list[dict[str, Any]] = []
    verified = _verify_source_span(
        persisted,
        human_selected,
        source_spans[0],
        human_mappings,
        mutable_fields=mutable_fields,
    )
    source_array_checks.append({
        "kind": "human",
        "all_preserved_arrays_exact": True,
        "verified_arrays": verified,
    })
    for index, (dagger, mappings, span) in enumerate(zip(
        daggers, dagger_mappings, source_spans[1:], strict=True,
    )):
        verified = _verify_source_span(
            persisted,
            dagger,
            span,
            mappings,
            mutable_fields=mutable_fields,
        )
        source_array_checks.append({
            "kind": "dagger",
            "source_index": index,
            "all_preserved_arrays_exact": True,
            "verified_arrays": verified,
        })

    fields, shapes, dtypes = _array_metadata(arguments.output)
    expected_fields = sorted(
        name for name in DEMONSTRATION_FIELDS
        if getattr(merged, name) is not None
    )
    if fields != expected_fields:
        raise RuntimeError("persisted NPZ field set does not match Demonstrations fields")

    human_source_to_output = {
        item["source_episode_id"]: item["output_episode_id"]
        for item in human_mappings
    }
    validation_episode_ids = tuple(
        human_source_to_output[value] for value in validation_human_episode_ids
    )
    validation = np.isin(merged.episode_ids, validation_episode_ids)
    train = ~validation
    labels = merged.actions[:, -1]
    disagreement_flat = disagreement[:, -1]
    original_supervision_flat = original_supervision[:, -1]
    original_corrections_flat = original_corrections[:, -1]
    selected_supervision_flat = merged.supervision_mask[:, -1]
    selected_corrections_flat = merged.correction_mask[:, -1]
    soft_evaluation_flat = (
        merged.teacher_action_evaluation_mask[:, -1]
        if dagger_label_mode == "soft-evaluations" else
        np.zeros(len(merged.actions), dtype=np.bool_)
    )

    origin: dict[int, dict[str, Any]] = {}
    for mapping in human_mappings:
        origin[mapping["output_episode_id"]] = {
            "kind": "human",
            "source_episode_id": mapping["source_episode_id"],
        }
    for source_index, mappings in enumerate(dagger_mappings):
        for accepted_index, mapping in enumerate(mappings):
            origin[mapping["output_episode_id"]] = {
                "kind": "dagger",
                "source_index": source_index,
                "source_episode_id": mapping["source_episode_id"],
                "accepted_episode": dagger_provenance[source_index][
                    "accepted_episodes"
                ][accepted_index],
            }

    episode_stats: dict[str, Any] = {}
    for episode_id in _episode_ids_in_order(merged):
        episode = merged.episode_ids == episode_id
        source_eligible = episode & (
            original_supervision_flat | original_corrections_flat
        )
        selected_hard = episode & (
            selected_supervision_flat | selected_corrections_flat
        )
        selected_soft = episode & soft_evaluation_flat
        origin_value = origin[episode_id]
        if origin_value["kind"] == "dagger":
            role = "strict_success_onpolicy_dagger_train"
        elif episode_id in validation_episode_ids:
            role = "successful_human_validation"
        elif np.any(selected_hard):
            role = "successful_human_train"
        else:
            role = "recurrent_and_frozen_parent_kl_context_only"
        eligible_count = int(np.count_nonzero(source_eligible))
        agreements = int(np.count_nonzero(source_eligible & ~disagreement_flat))
        episode_stats[str(episode_id)] = {
            **origin_value,
            "role": role,
            "decisions": int(np.count_nonzero(episode)),
            "source_supervision_labels": int(np.count_nonzero(
                episode & original_supervision_flat
            )),
            "source_correction_labels": int(np.count_nonzero(
                episode & original_corrections_flat
            )),
            "parent_raw_argmax_disagreements_on_all_rows": int(
                np.count_nonzero(episode & disagreement_flat)
            ),
            "parent_agreements_removed_from_source_eligible": agreements,
            "retained_hard_supervision_labels": int(np.count_nonzero(
                episode & selected_supervision_flat
            )),
            "retained_hard_correction_labels": int(np.count_nonzero(
                episode & selected_corrections_flat
            )),
            "retained_soft_evaluation_labels": int(np.count_nonzero(
                selected_soft
            )),
            "parent_exact_accuracy_on_source_eligible": (
                agreements / eligible_count if eligible_count else None
            ),
            "selected_hard_action_counts": _action_counts(labels, selected_hard),
            "selected_soft_teacher_action_counts": _action_counts(
                labels, selected_soft,
            ),
        }

    dagger_entries: list[dict[str, Any]] = []
    for index, (path, provenance, mappings, span) in enumerate(zip(
        dagger_paths,
        dagger_provenance,
        dagger_mappings,
        source_spans[1:],
        strict=True,
    )):
        start, stop = span
        source_rows = np.zeros(len(merged.actions), dtype=np.bool_)
        source_rows[start:stop] = True
        entry = {
            "source_index": index,
            "dataset": str(path),
            "dataset_sha256": file_sha256(path),
            "manifest": str(provenance["context_manifest_path"]),
            "manifest_sha256": file_sha256(provenance["context_manifest_path"]),
            "run_kind": provenance["context_manifest"].get("run_kind"),
            "episode_mappings": mappings,
            "context_episode_identities": provenance["context_episodes"],
            "native_dataset": str(provenance["native_dataset_path"]),
            "native_dataset_sha256": file_sha256(
                provenance["native_dataset_path"]
            ),
            "native_manifest": str(provenance["native_manifest_path"]),
            "native_manifest_sha256": file_sha256(
                provenance["native_manifest_path"]
            ),
            "native_run_kind": provenance["native_manifest"].get("run_kind"),
            "minimum_safety_margin_gain": provenance["native_manifest"].get(
                "minimum_safety_margin_gain"
            ),
            "dagger_report": str(provenance["report_path"]),
            "dagger_report_sha256": file_sha256(provenance["report_path"]),
            "accepted_episodes": provenance["accepted_episodes"],
            "source_outcome_evidence": provenance["source_outcome_evidence"],
            "strict_inclusion_criterion": provenance["native_manifest"].get(
                "strict_inclusion_criterion"
            ),
            "source_correction_labels": int(original_corrections[
                source_rows
            ].sum()),
            "retained_hard_correction_labels": int(merged.correction_mask[
                source_rows
            ].sum()),
            "retained_soft_evaluation_labels": int(
                merged.teacher_action_evaluation_mask[source_rows].sum()
                if dagger_label_mode == "soft-evaluations" else 0
            ),
            "parent_agreement_labels_removed": int(np.count_nonzero(
                original_corrections[source_rows]
                & ~disagreement[source_rows]
            )),
            "complete_recurrent_context_decisions": stop - start,
            "source_array_preservation": source_array_checks[index + 1],
        }
        dagger_entries.append(entry)

    human_entry = {
        "dataset": str(arguments.human_dataset),
        "dataset_sha256": file_sha256(arguments.human_dataset),
        "manifest": str(human_manifest_path),
        "manifest_sha256": file_sha256(human_manifest_path),
        "run_kind": human_manifest.get("run_kind"),
        "episode_mappings": human_mappings,
        "selected_source_episode_ids": list(human_episode_ids),
        "excluded_source_episode_ids": [
            value for value in _episode_ids_in_order(human)
            if value not in human_episode_ids
        ],
        "mask_mode": human_mask_mode,
        "source_mask_candidate_scope": (
            "all source supervision_mask eligible human labels"
            if human_mask_mode == "parent-misclassified-supervision" else
            "preexisting source supervision/correction masks; this may be a "
            "strict subset of all errors for a different parent"
        ),
        "source_array_preservation": source_array_checks[0],
    }
    sources: dict[str, Any] = {
        "human_context": human_entry,
        "dagger_contexts": dagger_entries,
    }
    if (
        len(dagger_entries) == 1
        and arguments.checkpoint.expanduser().resolve()
        == DEFAULT_CHECKPOINT.expanduser().resolve()
        and dagger_paths[0].expanduser().resolve()
        == DEFAULT_DAGGER_DATASET.expanduser().resolve()
    ):
        # Preserve the old manifest lookup used by prior one-source experiments.
        sources["v37_onpolicy_dagger"] = dagger_entries[0]

    report: dict[str, Any] = {
        "schema_version": 2,
        "run_kind": "successful_human_and_multi_onpolicy_dagger_parent_filtered",
        "acceptance_claim": False,
        "training_only": True,
        "output": str(arguments.output),
        "output_sha256": file_sha256(arguments.output),
        "implementation_sha256": implementation_sha256,
        "builder": str(builder_path),
        "builder_sha256": builder_sha256,
        "parent_checkpoint": str(arguments.checkpoint),
        "parent_checkpoint_sha256": checkpoint_sha256,
        "parent_policy_config": asdict(model.config),
        "sources": sources,
        "label_semantics": {
            "human": (
                "successful-human supervision eligibility intersected with "
                "frozen-parent raw-argmax disagreement; retained as matching "
                "hard supervision_mask and correction_mask"
                if human_mask_mode == "parent-misclassified-supervision" else
                "source human supervision/correction masks independently "
                "intersected with frozen-parent raw-argmax disagreement"
            ),
            "dagger": (
                "source intervention hard supervision/correction masks "
                "intersected with frozen-parent raw-argmax disagreement"
                if dagger_label_mode == "hard-corrections" else
                "all teacher_action_evaluation_mask rows first cleared, including "
                "human rows; DAgger hard masks cleared; evaluation mask true only "
                "for source correction interventions that disagree with the "
                "frozen-parent raw argmax"
            ),
            "dagger_label_mode": dagger_label_mode,
            "soft_training_intent": (
                "human top1/minimal-edit hard correction plus DAgger soft_action "
                "and collision-rank supervision"
                if dagger_label_mode == "soft-evaluations" else None
            ),
        },
        "selection": {
            "parent_filter": "frozen parent raw joint 18-way argmax differs from stored action label",
            "action_selection": "raw torch.argmax; no factorization, safety shield, proficiency runtime, or postprocessing",
            "streaming_hidden_state": (
                "reset at every remapped episode; carried and detached across "
                f"{arguments.chunk_length}-decision chunks within each episode"
            ),
            "inference_chunk_length": arguments.chunk_length,
            "human_mask_mode": human_mask_mode,
            "dagger_label_mode": dagger_label_mode,
            "rows_removed": 0,
            "rows_reordered": False,
            "action_labels_rewritten": False,
            "validation_episode_ids": list(validation_episode_ids),
            "train_episode_ids": [
                int(value) for value in _episode_ids_in_order(merged)
                if value not in validation_episode_ids
            ],
        },
        "samples": int(len(merged.actions)),
        "history": int(merged.actions.shape[1]),
        "episode_groups": len(_episode_ids_in_order(merged)),
        "context_decisions_retained": int(len(merged.actions)),
        "source_supervision_labels": int(original_supervision.sum()),
        "source_correction_labels": int(original_corrections.sum()),
        "retained_hard_supervision_labels": int(merged.supervision_mask.sum()),
        "retained_hard_correction_labels": int(merged.correction_mask.sum()),
        "retained_dagger_soft_evaluation_labels": int(
            merged.teacher_action_evaluation_mask[dagger_rows].sum()
            if dagger_label_mode == "soft-evaluations" else 0
        ),
        "retained_human_teacher_action_evaluation_labels": int(
            merged.teacher_action_evaluation_mask[human_rows].sum()
        ),
        "retained_total_teacher_action_evaluation_labels": int(
            merged.teacher_action_evaluation_mask.sum()
        ),
        "train_hard_supervision_labels": int(merged.supervision_mask[train].sum()),
        "train_hard_correction_labels": int(merged.correction_mask[train].sum()),
        "validation_hard_supervision_labels": int(
            merged.supervision_mask[validation].sum()
        ),
        "validation_hard_correction_labels": int(
            merged.correction_mask[validation].sum()
        ),
        "all_retained_hard_supervision_differs_from_parent_raw_argmax": bool(
            np.all(disagreement[merged.supervision_mask])
        ),
        "all_retained_hard_corrections_differ_from_parent_raw_argmax": bool(
            np.all(disagreement[merged.correction_mask])
        ),
        "all_retained_dagger_soft_evaluations_differs_from_parent_raw_argmax": (
            bool(np.all(disagreement[
                merged.teacher_action_evaluation_mask & dagger_rows_2d
            ])) if dagger_label_mode == "soft-evaluations" else None
        ),
        "strict_success_provenance": {
            "failed_dagger_episode_rows_added": 0,
            "all_dagger_sources_verified_attack_complete_death_zero": True,
            "dagger_sources": [
                {
                    "source_index": entry["source_index"],
                    "dataset": entry["dataset"],
                    "dataset_sha256": entry["dataset_sha256"],
                    "dagger_report": entry["dagger_report"],
                    "dagger_report_sha256": entry["dagger_report_sha256"],
                    "accepted_episodes": entry["accepted_episodes"],
                    "source_outcome_evidence": entry["source_outcome_evidence"],
                    "strict_inclusion_criterion": entry[
                        "strict_inclusion_criterion"
                    ],
                }
                for entry in dagger_entries
            ],
            "human_source_claim": human_manifest.get(
                "strict_success_provenance",
                human_manifest.get("human_episode_identity_provenance"),
            ),
        },
        "excluded_model_inputs": [
            "absolute_frame",
            "script_phase",
            "fixed_route",
            "waypoint",
            "external_region_dynamics_memory",
        ],
        "context_validation": {
            "dagger_contexts_match_parent_checkpoint": True,
            "dagger_contexts_mutually_compatible": True,
            "human_context": human_context_validation,
        },
        "array_preservation": {
            "all_preserved_arrays_exactly_equal": True,
            "intentionally_modified_arrays": sorted(
                mutable_fields | {"episode_ids"}
            ),
            "preserved_arrays": preserved_arrays,
            "field_set_exact_match": True,
            "fields": fields,
            "shapes": shapes,
            "dtypes": dtypes,
            "per_source": source_array_checks,
        },
        "per_episode": episode_stats,
    }
    _verify_stable_hashes(captured_input_hashes)
    if file_sha256(builder_path) != builder_sha256:
        raise RuntimeError("builder changed while constructing the archive")
    if source_tree_sha256() != implementation_sha256:
        raise RuntimeError("stg_lab implementation changed while constructing archive")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    persisted_report = json.loads(manifest_path.read_text(encoding="utf-8"))
    if persisted_report != report:
        raise RuntimeError("persisted manifest differs from generated report")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human-dataset", type=Path, default=DEFAULT_HUMAN_DATASET)
    parser.add_argument(
        "--human-episode-id",
        dest="human_episode_ids",
        action="append",
        type=int,
        help="source human episode to retain; repeatable (default: 0..6)",
    )
    parser.add_argument(
        "--validation-human-episode-id",
        dest="validation_human_episode_ids",
        action="append",
        type=int,
        help="retained source human validation episode; repeatable (default: 3)",
    )
    parser.add_argument(
        "--human-mask-mode",
        choices=HUMAN_MASK_MODES,
        default="preserve",
    )
    parser.add_argument(
        "--dagger-dataset",
        dest="dagger_datasets",
        action="append",
        type=Path,
        help="contextual strict-success DAgger archive; repeatable",
    )
    parser.add_argument(
        "--dagger-label-mode",
        choices=DAGGER_LABEL_MODES,
        default="hard-corrections",
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--chunk-length", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    arguments = parser.parse_args()
    print(json.dumps(build(arguments), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
