"""Recompute the v2 policy's action agreement on the external held-out split."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from stg_lab.provenance import file_sha256, source_tree_sha256
from stg_lab.training import Demonstrations, load_checkpoint


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
CHECKPOINT = ARTIFACTS / "policy_visible_v2.pt"
TRAIN_DATASET = ARTIFACTS / "canonical_train_visible_v2.npz"
HELDOUT_DATASET = ARTIFACTS / "canonical_heldout_visible_v2.npz"
VISIBLE_MANIFEST = ARTIFACTS / "visible_dataset_v2_manifest.json"
CANONICAL_MANIFEST = ARTIFACTS / "canonical_dataset_manifest.json"
EXPANDED_MANIFEST = ARTIFACTS / "canonical_dataset_expanded_manifest.json"
OUTPUT = ARTIFACTS / "agreement_visible_v2.json"
THRESHOLD = 0.85


@dataclass(frozen=True, slots=True)
class Partition:
    episode_scenario: Mapping[int, str]
    scenario_episode_ids: Mapping[str, tuple[int, ...]]
    scenario_seeds: Mapping[str, tuple[int, ...]]
    scenario_samples: Mapping[str, int]


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"manifest is not an object: {path}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _manifest_output(manifest: Mapping[str, Any], path: Path) -> Mapping[str, Any]:
    relative = _relative(path)
    matches = [
        item
        for item in manifest.get("outputs", ())
        if isinstance(item, Mapping) and item.get("output") == relative
    ]
    _require(len(matches) == 1, f"visible manifest must bind exactly one {relative} output")
    record = matches[0]
    _require(
        record.get("output_sha256") == file_sha256(path),
        f"visible manifest checksum does not match {relative}",
    )
    input_value = record.get("input")
    _require(isinstance(input_value, str) and not Path(input_value).is_absolute(), "invalid input path")
    input_path = ROOT / input_value
    _require(input_path.is_file(), f"visible dataset input does not exist: {input_value}")
    _require(
        record.get("input_sha256") == file_sha256(input_path),
        f"visible manifest input checksum does not match {input_value}",
    )
    return record


def _partition(sources: Iterable[Mapping[str, Any]]) -> Partition:
    episode_scenario: dict[int, str] = {}
    scenario_episode_ids: dict[str, list[int]] = {}
    scenario_seeds: dict[str, list[int]] = {}
    scenario_samples: dict[str, int] = {}
    for source in sources:
        scenario = str(source.get("scenario", ""))
        _require(scenario in {"stage5_boss3", "stage5_boss4"}, "unknown scenario in split")
        episode_ids = tuple(int(value) for value in source.get("episode_ids", ()))
        seeds = tuple(int(value) for value in source.get("seeds", ()))
        _require(episode_ids and len(episode_ids) == len(seeds), "split seed/id counts differ")
        samples = int(source.get("samples", 0))
        _require(samples > 0, "split source has no samples")
        for episode_id in episode_ids:
            _require(episode_id not in episode_scenario, f"duplicate split episode id {episode_id}")
            episode_scenario[episode_id] = scenario
        scenario_episode_ids.setdefault(scenario, []).extend(episode_ids)
        scenario_seeds.setdefault(scenario, []).extend(seeds)
        scenario_samples[scenario] = scenario_samples.get(scenario, 0) + samples
    _require(set(scenario_episode_ids) == {"stage5_boss3", "stage5_boss4"}, "split is incomplete")
    return Partition(
        episode_scenario=episode_scenario,
        scenario_episode_ids={key: tuple(value) for key, value in scenario_episode_ids.items()},
        scenario_seeds={key: tuple(value) for key, value in scenario_seeds.items()},
        scenario_samples=scenario_samples,
    )


def _validate_archive(demonstrations: Demonstrations, partition: Partition, *, label: str) -> None:
    _require(demonstrations.episode_ids is not None, f"{label} archive has no episode ids")
    observed = {int(value) for value in np.unique(demonstrations.episode_ids)}
    expected = set(partition.episode_scenario)
    _require(observed == expected, f"{label} archive episode ids do not match its split manifest")
    for scenario, episode_ids in partition.scenario_episode_ids.items():
        count = int(np.isin(demonstrations.episode_ids, episode_ids).sum())
        _require(
            count == partition.scenario_samples[scenario],
            f"{label} {scenario} sample count does not match its split manifest",
        )


def validate_bindings(
    checkpoint_metadata: Mapping[str, Any],
    train: Demonstrations,
    heldout: Demonstrations,
) -> dict[str, Any]:
    """Validate hashes and the declared external split before any inference."""

    visible = _load_object(VISIBLE_MANIFEST)
    canonical = _load_object(CANONICAL_MANIFEST)
    expanded = _load_object(EXPANDED_MANIFEST)
    train_record = _manifest_output(visible, TRAIN_DATASET)
    heldout_record = _manifest_output(visible, HELDOUT_DATASET)

    training_data = checkpoint_metadata.get("training_data")
    _require(int(checkpoint_metadata.get("version", 0)) == 2, "checkpoint must use schema version 2")
    _require(isinstance(training_data, Mapping), "checkpoint has no training-data binding")
    _require(training_data.get("path") == _relative(TRAIN_DATASET), "checkpoint names another training dataset")
    _require(
        training_data.get("sha256") == file_sha256(TRAIN_DATASET),
        "checkpoint training-data checksum does not match the archive",
    )

    canonical_train = canonical.get("train", {})
    canonical_heldout = canonical.get("heldout", {})
    _require(isinstance(canonical_train, Mapping), "canonical train split is missing")
    _require(isinstance(canonical_heldout, Mapping), "canonical held-out split is missing")
    train_sources = [
        *canonical_train.get("sources", ()),
        *expanded.get("added_sources", ()),
    ]
    heldout_sources = list(canonical_heldout.get("sources", ()))
    _require(all(isinstance(value, Mapping) for value in train_sources), "invalid train source")
    _require(all(isinstance(value, Mapping) for value in heldout_sources), "invalid held-out source")
    train_partition = _partition(train_sources)
    heldout_partition = _partition(heldout_sources)
    _validate_archive(train, train_partition, label="training")
    _validate_archive(heldout, heldout_partition, label="held-out")

    checkpoint_episode_ids = training_data.get("episode_ids")
    _require(isinstance(checkpoint_episode_ids, Sequence), "checkpoint has no training episode ids")
    _require(
        sorted(int(value) for value in checkpoint_episode_ids)
        == sorted(train_partition.episode_scenario),
        "checkpoint training episode ids do not match the training split",
    )
    _require(
        train_record.get("input") == expanded.get("path"),
        "visible training archive was not rebuilt from the expanded train split",
    )
    _require(
        heldout_record.get("input") == canonical_heldout.get("path"),
        "visible held-out archive was not rebuilt from the canonical held-out split",
    )

    overlaps: dict[str, list[int]] = {}
    for scenario in ("stage5_boss3", "stage5_boss4"):
        overlap = sorted(
            set(train_partition.scenario_seeds[scenario])
            & set(heldout_partition.scenario_seeds[scenario])
        )
        overlaps[scenario] = overlap
        _require(not overlap, f"{scenario} training and held-out seeds overlap")

    return {
        "verified": True,
        "checkpoint_training_dataset": _relative(TRAIN_DATASET),
        "checkpoint_training_dataset_sha256": file_sha256(TRAIN_DATASET),
        "training_episode_ids": sorted(train_partition.episode_scenario),
        "training_seeds": {
            key: list(value) for key, value in train_partition.scenario_seeds.items()
        },
        "heldout_episode_ids": sorted(heldout_partition.episode_scenario),
        "heldout_seeds": {
            key: list(value) for key, value in heldout_partition.scenario_seeds.items()
        },
        "seed_overlap": overlaps,
        "scenario_episode_ids": {
            key: list(value) for key, value in heldout_partition.scenario_episode_ids.items()
        },
    }


def _agreement_counts(
    model: Any,
    demonstrations: Demonstrations,
    episode_ids: Sequence[int],
    *,
    batch_size: int = 64,
) -> tuple[int, int]:
    import torch

    _require(demonstrations.episode_ids is not None, "held-out archive has no episode ids")
    indices = np.flatnonzero(np.isin(demonstrations.episode_ids, episode_ids))
    correct = 0
    total = 0
    memory_size = int(model.config.memory_size)
    model.to("cpu")
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(indices), batch_size):
            selected = indices[offset:offset + batch_size]
            global_frames = torch.from_numpy(demonstrations.global_frames[selected]).float()
            local_frames = torch.from_numpy(demonstrations.local_frames[selected]).float()
            if demonstrations.memory is None:
                memory = torch.zeros((*demonstrations.actions[selected].shape, memory_size))
            else:
                memory = torch.from_numpy(demonstrations.memory[selected]).float()
            logits, _risk, _hidden = model(global_frames, local_frames, memory)
            labels = torch.from_numpy(demonstrations.actions[selected]).long()
            if demonstrations.supervision_mask is None:
                mask = torch.zeros_like(labels, dtype=torch.bool)
                mask[:, -1] = True
            else:
                mask = torch.from_numpy(demonstrations.supervision_mask[selected]).bool()
            correct += int((logits.argmax(dim=-1)[mask] == labels[mask]).sum().item())
            total += int(mask.sum().item())
    return correct, total


def main() -> None:
    train = Demonstrations.load(TRAIN_DATASET)
    heldout = Demonstrations.load(HELDOUT_DATASET)
    model, checkpoint_metadata = load_checkpoint(CHECKPOINT, device="cpu")
    binding = validate_bindings(checkpoint_metadata, train, heldout)

    heldout_results: dict[str, Any] = {}
    for scenario, episode_ids in binding["scenario_episode_ids"].items():
        correct, samples = _agreement_counts(model, heldout, episode_ids)
        agreement = correct / samples if samples else 0.0
        heldout_results[scenario] = {
            "episode_ids": episode_ids,
            "seeds": binding["heldout_seeds"][scenario],
            "samples": samples,
            "correct": correct,
            "agreement": agreement,
            "threshold": THRESHOLD,
            "passed": agreement >= THRESHOLD,
        }
    overall_correct = sum(value["correct"] for value in heldout_results.values())
    overall_samples = sum(value["samples"] for value in heldout_results.values())
    final_history = checkpoint_metadata.get("history", ())
    report = {
        "schema_version": 2,
        "run_kind": "external_heldout_action_agreement",
        "generator": _relative(Path(__file__).resolve()),
        "generator_sha256": file_sha256(Path(__file__)),
        "implementation_sha256": source_tree_sha256(),
        "split": "heldout",
        "checkpoint": _relative(CHECKPOINT),
        "checkpoint_sha256": file_sha256(CHECKPOINT),
        "checkpoint_metadata": {
            "version": checkpoint_metadata.get("version"),
            "policy_config": checkpoint_metadata.get("policy_config"),
            "training_config": checkpoint_metadata.get("training_config"),
            "training_data": checkpoint_metadata.get("training_data"),
            "epochs": len(final_history),
            "final_training_metrics": final_history[-1] if final_history else None,
        },
        "dataset": _relative(HELDOUT_DATASET),
        "dataset_sha256": file_sha256(HELDOUT_DATASET),
        "evidence_manifests": {
            _relative(VISIBLE_MANIFEST): file_sha256(VISIBLE_MANIFEST),
            _relative(CANONICAL_MANIFEST): file_sha256(CANONICAL_MANIFEST),
            _relative(EXPANDED_MANIFEST): file_sha256(EXPANDED_MANIFEST),
        },
        "split_binding": binding,
        "evaluation_contract": {
            "device": "cpu",
            "policy_input": "blank-cold-start delayed visible-displacement semantic windows",
            "memory_input": "archive-provided scenario identity; no held-out cue backfill",
            "supervision": "archive supervision mask, or final visible timestep when absent",
            "authority_state_used": False,
            "shield_used": False,
        },
        "heldout": heldout_results,
        "overall": {
            "samples": overall_samples,
            "correct": overall_correct,
            "agreement": overall_correct / overall_samples if overall_samples else 0.0,
        },
        "passed": all(value["passed"] for value in heldout_results.values()),
    }
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": _relative(OUTPUT),
        "sha256": file_sha256(OUTPUT),
        **report,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
