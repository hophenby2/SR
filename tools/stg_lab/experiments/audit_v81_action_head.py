"""Read-only v81 action-head audit on the fit and calibration sources.

This script deliberately never accepts arbitrary episode seeds.  It filters the
failure inventory by the fixed audit whitelist before resolving, hashing, or
opening any source path.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
import math
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import torch

from stg_lab.provenance import file_sha256
from stg_lab.residual_adapter import (
    ResidualAdapterConfig,
    ResidualCorrectionAdapter,
    finite_action_probabilities,
)
from stg_lab.training import load_checkpoint

from train_temporal_residual_adapter import (
    EARLY_ONSET_MINIMUM_LEAD_DECISIONS,
    FUTURE_ONSET_HORIZON_DECISIONS,
    EpisodeFeatures,
    _apply_existing_normalization,
    _load_episode,
    _load_fit_checkpoint,
)


TRAINING_SEEDS = (
    20260730,
    20260731,
    20260732,
    10294,
    10295,
    20260733,
    10296,
    10298,
    10292,
    10297,
    10301,
    10302,
    10303,
    10304,
    10305,
)
CALIBRATION_SEED = 10306
PROHIBITED_SOURCE_SEEDS = (10307, 10308, 10310, 10309)
ALLOWED_SEEDS = frozenset((*TRAINING_SEEDS, CALIBRATION_SEED))
ACTION_COUNT = 18
DIRECTION_COUNT = 9

DEFAULT_FIT = Path(
    "artifacts/"
    "policy-humanlike-highres-okuu3-v81-internal-joint-pos-separate.fit.pt"
)
DEFAULT_FAILURE = Path(
    "artifacts/"
    "policy-humanlike-highres-okuu3-v81-internal-joint-pos-separate-"
    "calibration-failure.json"
)
DEFAULT_PARENT = Path(
    "artifacts/"
    "policy-humanlike-highres-okuu3-v54-v37-onpolicy-gain025-minedit20-"
    "top1w2-kl20-ft60.pt"
)
DEFAULT_OUTPUT = Path(
    "artifacts/"
    "policy-humanlike-highres-okuu3-v81-internal-joint-pos-action-audit.json"
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            mode="w",
            encoding="utf-8",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(dict(payload), handle, indent=2, allow_nan=False)
            handle.write("\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _required_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _required_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a nonempty string")
    return value


def _select_allowed_inventory(
    failure: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Select by seed and role before touching any inventory source path."""

    raw_inventory = failure.get("source_inventory")
    if not isinstance(raw_inventory, list):
        raise ValueError("failure diagnostics have no source inventory")

    selected_by_seed: dict[int, dict[str, Any]] = {}
    for index, raw in enumerate(raw_inventory):
        if not isinstance(raw, Mapping):
            raise ValueError(f"source_inventory[{index}] must be an object")
        seed = _required_int(raw.get("seed"), field=f"source_inventory[{index}].seed")
        if seed not in ALLOWED_SEEDS:
            # Do not inspect, resolve, hash, or open any path in this record.
            continue
        expected_role = "calibration" if seed == CALIBRATION_SEED else "training"
        if raw.get("role") != expected_role:
            raise ValueError(f"seed {seed} must have role={expected_role!r}")
        if seed in selected_by_seed:
            raise ValueError(f"duplicate allowed source seed: {seed}")
        selected_by_seed[seed] = dict(raw)

    missing = ALLOWED_SEEDS - selected_by_seed.keys()
    unexpected = selected_by_seed.keys() - ALLOWED_SEEDS
    if missing or unexpected:
        raise ValueError(
            f"allowed inventory mismatch: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
    return [selected_by_seed[seed] for seed in (*TRAINING_SEEDS, CALIBRATION_SEED)]


def _selected_fit_inventory(
    fit_metadata: Mapping[str, Any],
) -> dict[int, dict[str, Any]]:
    training_metadata = fit_metadata.get("training_metadata")
    if not isinstance(training_metadata, Mapping):
        raise ValueError("fit checkpoint training metadata are invalid")
    raw_inventory = training_metadata.get("source_inventory")
    if not isinstance(raw_inventory, list):
        raise ValueError("fit checkpoint has no source inventory")
    result: dict[int, dict[str, Any]] = {}
    for index, raw in enumerate(raw_inventory):
        if not isinstance(raw, Mapping):
            raise ValueError(f"fit source_inventory[{index}] must be an object")
        seed = _required_int(
            raw.get("seed"),
            field=f"fit source_inventory[{index}].seed",
        )
        if seed not in ALLOWED_SEEDS:
            # Apply the same guard before inspecting any path-bearing fields.
            continue
        if seed in result:
            raise ValueError(f"duplicate allowed fit source seed: {seed}")
        result[seed] = dict(raw)
    if result.keys() != ALLOWED_SEEDS:
        raise ValueError("fit checkpoint does not contain the complete audit whitelist")
    return result


def _verify_allowed_sources(
    inventory: Sequence[Mapping[str, Any]],
    fit_inventory: Mapping[int, Mapping[str, Any]],
) -> tuple[list[tuple[Path, Path, Path]], list[dict[str, Any]]]:
    triplets: list[tuple[Path, Path, Path]] = []
    provenance: list[dict[str, Any]] = []
    for raw in inventory:
        seed = _required_int(raw.get("seed"), field="source seed")
        fit_raw = fit_inventory.get(seed)
        if fit_raw is None or dict(raw) != dict(fit_raw):
            raise ValueError(
                f"failure and fit inventories disagree for allowed seed {seed}"
            )
        paths: dict[str, Path] = {}
        hashes: dict[str, str] = {}
        for kind in ("dataset", "report", "manifest"):
            path_text = _required_string(raw.get(kind), field=f"seed {seed} {kind}")
            declared = _required_string(
                raw.get(f"{kind}_sha256"),
                field=f"seed {seed} {kind}_sha256",
            )
            path = Path(path_text)
            actual = file_sha256(path)
            if actual != declared:
                raise ValueError(
                    f"seed {seed} {kind} hash mismatch: "
                    f"declared={declared}, actual={actual}"
                )
            paths[kind] = path
            hashes[kind] = actual
        triplets.append((paths["dataset"], paths["report"], paths["manifest"]))
        provenance.append({
            "seed": seed,
            "role": raw["role"],
            "dataset": str(paths["dataset"]),
            "dataset_sha256": hashes["dataset"],
            "report": str(paths["report"]),
            "report_sha256": hashes["report"],
            "manifest": str(paths["manifest"]),
            "manifest_sha256": hashes["manifest"],
            "declared_hashes_verified": True,
        })
    return triplets, provenance


def _predict_action_probabilities(
    adapter: ResidualCorrectionAdapter,
    episode: EpisodeFeatures,
    *,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ensemble-mean probabilities and all-member finite masks."""

    hidden: tuple[torch.Tensor, ...] | None = None
    probability_chunks: list[torch.Tensor] = []
    finite_chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, episode.decisions, 256):
            stop = min(start + 256, episode.decisions)
            features = episode.features[:, start:stop].to(device)
            outputs = [
                member.forward_with_all_safety(
                    features,
                    None if hidden is None else hidden[index],
                )
                for index, member in enumerate(adapter.members)
            ]
            parent_logits = episode.parent_logits[0, start:stop].to(device)
            action_logits = torch.stack([
                adapter.decode_action_logits(actions[0], parent_logits)
                for _gate, actions, _collision, _margin, _physical, _next_hidden
                in outputs
            ], dim=0)
            probabilities, member_finite = finite_action_probabilities(action_logits)
            probability_chunks.append(probabilities.mean(dim=0).cpu())
            finite_chunks.append(member_finite.all(dim=0).cpu())
            hidden = tuple(
                next_hidden.detach()
                for _gate, _actions, _collision, _margin, _physical, next_hidden
                in outputs
            )
    mean_probabilities = torch.cat(probability_chunks, dim=0)
    all_members_finite = torch.cat(finite_chunks, dim=0)
    if mean_probabilities.shape != (episode.decisions, ACTION_COUNT):
        raise ValueError(
            f"unexpected action probability shape for seed {episode.seed}: "
            f"{tuple(mean_probabilities.shape)}"
        )
    return mean_probabilities, all_members_finite


def _speed_name(slow: bool) -> str:
    return "slow" if slow else "fast"


def _empty_confusion() -> dict[str, dict[str, int]]:
    return {
        "fast": {"fast": 0, "slow": 0},
        "slow": {"fast": 0, "slow": 0},
    }


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _audit_split(
    episodes: Sequence[EpisodeFeatures],
    probabilities: Mapping[int, torch.Tensor],
    finite_masks: Mapping[int, torch.Tensor],
    *,
    include_details: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    counts: Counter[str] = Counter()
    marginal_confusion = _empty_confusion()
    raw_confusion = _empty_confusion()
    equivalent_set_sizes: Counter[int] = Counter()
    by_lead: dict[str, Counter[str]] = {
        str(lead): Counter()
        for lead in range(
            EARLY_ONSET_MINIMUM_LEAD_DECISIONS,
            FUTURE_ONSET_HORIZON_DECISIONS + 1,
        )
    }
    details: list[dict[str, Any]] = []

    for episode in episodes:
        values = probabilities[episode.seed]
        finite = finite_masks[episode.seed]
        early = (
            episode.anticipatory
            & (
                episode.anticipatory_lead_decisions
                >= EARLY_ONSET_MINIMUM_LEAD_DECISIONS
            )
            & (
                episode.anticipatory_lead_decisions
                <= FUTURE_ONSET_HORIZON_DECISIONS
            )
        )
        for decision in torch.nonzero(early, as_tuple=False).flatten().tolist():
            lead = int(episode.anticipatory_lead_decisions[decision])
            target = int(episode.preferred_actions[decision])
            if not 0 <= target < ACTION_COUNT:
                raise ValueError(
                    f"seed {episode.seed} decision {decision} has no preferred action"
                )
            row_probabilities = values[decision]
            if not bool(finite[decision]) or not bool(torch.isfinite(row_probabilities).all()):
                raise ValueError(
                    f"seed {episode.seed} decision {decision} has nonfinite action output"
                )
            if not math.isclose(
                float(row_probabilities.sum()),
                1.0,
                rel_tol=1e-5,
                abs_tol=1e-5,
            ):
                raise ValueError(
                    f"seed {episode.seed} decision {decision} probabilities do not sum to one"
                )

            top5 = torch.topk(row_probabilities, k=5, largest=True, sorted=True).indices
            top_ids = [int(value) for value in top5.tolist()]
            top1 = top_ids[0]
            target_direction = target % DIRECTION_COUNT
            target_slow = target >= DIRECTION_COUNT
            raw_direction = top1 % DIRECTION_COUNT
            raw_slow = top1 >= DIRECTION_COUNT
            direction_probabilities = (
                row_probabilities[:DIRECTION_COUNT]
                + row_probabilities[DIRECTION_COUNT:]
            )
            direction_prediction = int(direction_probabilities.argmax())
            speed_probabilities = torch.stack((
                row_probabilities[:DIRECTION_COUNT].sum(),
                row_probabilities[DIRECTION_COUNT:].sum(),
            ))
            speed_prediction_slow = bool(int(speed_probabilities.argmax()))

            exact_top1 = top1 == target
            exact_top3 = target in top_ids[:3]
            exact_top5 = target in top_ids
            raw_direction_hit = raw_direction == target_direction
            marginal_direction_hit = direction_prediction == target_direction
            raw_speed_hit = raw_slow == target_slow
            marginal_speed_hit = speed_prediction_slow == target_slow
            preferred_candidate_path = (
                exact_top1
                and top1 != int(episode.parent_actions[decision])
            )
            correction_required = bool(
                episode.preferred_correction_required[decision]
            )
            equivalent_actions = episode.preferred_equivalent_actions[
                decision
            ].nonzero(as_tuple=False).flatten().tolist()
            equivalent_top1 = correction_required and top1 in equivalent_actions
            speed_sibling = target + 9 if target < 9 else target - 9

            def first_strict_failure(action: int) -> int | None:
                for offset in range(FUTURE_ONSET_HORIZON_DECISIONS + 1):
                    future = decision + offset
                    if future >= episode.decisions or not bool(
                        episode.evaluation_safe_actions[future, action]
                    ):
                        return offset
                return None

            preferred_first_failure = first_strict_failure(target)
            sibling_first_failure = first_strict_failure(speed_sibling)
            preferred_survives_onset = (
                preferred_first_failure is None
                or preferred_first_failure > lead
            )
            sibling_survives_onset = (
                sibling_first_failure is None
                or sibling_first_failure > lead
            )

            counts["targets"] += 1
            counts["all_members_finite"] += 1
            counts["exact_top1"] += int(exact_top1)
            counts["exact_top3"] += int(exact_top3)
            counts["exact_top5"] += int(exact_top5)
            counts["raw_top1_direction"] += int(raw_direction_hit)
            counts["direction_marginal_top1"] += int(marginal_direction_hit)
            counts["raw_top1_speed"] += int(raw_speed_hit)
            counts["speed_marginal_top1"] += int(marginal_speed_hit)
            counts["preferred_candidate_path"] += int(preferred_candidate_path)
            counts["correction_required"] += int(correction_required)
            counts["equivalent_top1"] += int(equivalent_top1)
            counts["preferred_survives_onset"] += int(
                preferred_survives_onset
            )
            counts["speed_sibling_current_strict_safe"] += int(
                bool(episode.evaluation_safe_actions[decision, speed_sibling])
            )
            counts["speed_sibling_certified_equivalent"] += int(
                speed_sibling in equivalent_actions
            )
            counts["speed_sibling_survives_onset"] += int(
                sibling_survives_onset
            )
            equivalent_set_sizes[len(equivalent_actions)] += 1
            target_speed = _speed_name(target_slow)
            marginal_speed = _speed_name(speed_prediction_slow)
            raw_speed = _speed_name(raw_slow)
            marginal_confusion[target_speed][marginal_speed] += 1
            raw_confusion[target_speed][raw_speed] += 1

            lead_counts = by_lead[str(lead)]
            lead_counts["targets"] += 1
            lead_counts["exact_top1"] += int(exact_top1)
            lead_counts["exact_top3"] += int(exact_top3)
            lead_counts["exact_top5"] += int(exact_top5)
            lead_counts["raw_top1_direction"] += int(raw_direction_hit)
            lead_counts["direction_marginal_top1"] += int(marginal_direction_hit)
            lead_counts["raw_top1_speed"] += int(raw_speed_hit)
            lead_counts["speed_marginal_top1"] += int(marginal_speed_hit)
            lead_counts["correction_required"] += int(correction_required)
            lead_counts["equivalent_top1"] += int(equivalent_top1)

            if include_details:
                details.append({
                    "seed": episode.seed,
                    "decision": int(decision),
                    "decision_indexing": "zero_based",
                    "lead": lead,
                    "preferred_action": target,
                    "target_direction": target_direction,
                    "target_speed": target_speed,
                    "parent_action": int(episode.parent_actions[decision]),
                    "top1": top1,
                    "top3": top_ids[:3],
                    "top5": top_ids,
                    "probabilities": [float(value) for value in row_probabilities.tolist()],
                    "raw_top1_direction": raw_direction,
                    "direction_prediction": direction_prediction,
                    "direction_probabilities": [
                        float(value) for value in direction_probabilities.tolist()
                    ],
                    "raw_top1_speed": raw_speed,
                    "speed_prediction": marginal_speed,
                    "speed_probabilities": {
                        "fast": float(speed_probabilities[0]),
                        "slow": float(speed_probabilities[1]),
                    },
                    "exact_top1_hit": exact_top1,
                    "exact_top3_hit": exact_top3,
                    "exact_top5_hit": exact_top5,
                    "raw_top1_direction_hit": raw_direction_hit,
                    "direction_marginal_hit": marginal_direction_hit,
                    "raw_top1_speed_hit": raw_speed_hit,
                    "speed_marginal_hit": marginal_speed_hit,
                    "preferred_candidate_path": preferred_candidate_path,
                    "correction_required": correction_required,
                    "certified_equivalent_actions": equivalent_actions,
                    "equivalent_top1_hit": equivalent_top1,
                    "speed_sibling_action": speed_sibling,
                    "preferred_first_strict_failure_offset": (
                        preferred_first_failure
                    ),
                    "speed_sibling_first_strict_failure_offset": (
                        sibling_first_failure
                    ),
                    "preferred_survives_through_onset": (
                        preferred_survives_onset
                    ),
                    "speed_sibling_survives_through_onset": (
                        sibling_survives_onset
                    ),
                })

    targets = counts["targets"]

    def metrics_for(counter: Counter[str]) -> dict[str, Any]:
        denominator = counter["targets"]
        return {
            "targets": denominator,
            "exact_top1": counter["exact_top1"],
            "exact_top1_accuracy": _ratio(counter["exact_top1"], denominator),
            "exact_top3": counter["exact_top3"],
            "exact_top3_accuracy": _ratio(counter["exact_top3"], denominator),
            "exact_top5": counter["exact_top5"],
            "exact_top5_accuracy": _ratio(counter["exact_top5"], denominator),
            "raw_top1_direction": counter["raw_top1_direction"],
            "raw_top1_direction_accuracy": _ratio(
                counter["raw_top1_direction"], denominator
            ),
            "direction_marginal_top1": counter["direction_marginal_top1"],
            "direction_marginal_top1_accuracy": _ratio(
                counter["direction_marginal_top1"], denominator
            ),
            "raw_top1_speed": counter["raw_top1_speed"],
            "raw_top1_speed_accuracy": _ratio(
                counter["raw_top1_speed"], denominator
            ),
            "speed_marginal_top1": counter["speed_marginal_top1"],
            "speed_marginal_top1_accuracy": _ratio(
                counter["speed_marginal_top1"], denominator
            ),
            "correction_required": counter["correction_required"],
            "equivalent_top1": counter["equivalent_top1"],
            "equivalent_top1_accuracy_on_required": _ratio(
                counter["equivalent_top1"],
                counter["correction_required"],
            ),
        }

    summary = metrics_for(counts)
    summary.update({
        "all_members_finite": counts["all_members_finite"],
        "preferred_candidate_path": counts["preferred_candidate_path"],
        "certified_equivalent_set_size_distribution": {
            str(size): count
            for size, count in sorted(equivalent_set_sizes.items())
        },
        "preferred_survives_through_onset": counts[
            "preferred_survives_onset"
        ],
        "speed_sibling_current_strict_safe": counts[
            "speed_sibling_current_strict_safe"
        ],
        "speed_sibling_certified_equivalent": counts[
            "speed_sibling_certified_equivalent"
        ],
        "speed_sibling_survives_through_onset": counts[
            "speed_sibling_survives_onset"
        ],
        "speed_marginal_confusion_matrix": marginal_confusion,
        "raw_top1_speed_confusion_matrix": raw_confusion,
        "confusion_matrix_axes": {
            "rows": "target speed",
            "columns": "predicted speed",
            "order": ["fast", "slow"],
        },
        "by_lead": {
            lead: metrics_for(by_lead[lead])
            for lead in sorted(by_lead, key=int)
        },
    })
    if targets != sum(value["targets"] for value in summary["by_lead"].values()):
        raise AssertionError("lead summaries do not add up")
    return summary, details


def _failure_expected_counts(
    failure: Mapping[str, Any],
) -> dict[str, dict[str, int]]:
    splits = failure.get("splits")
    if not isinstance(splits, Mapping):
        raise ValueError("failure diagnostics have no split funnel")
    result: dict[str, dict[str, int]] = {}
    for split in ("training", "calibration"):
        raw = splits.get(split)
        if not isinstance(raw, Mapping):
            raise ValueError(f"failure diagnostics have no {split} funnel")
        stages = raw.get("stage_counts")
        if not isinstance(stages, Mapping):
            raise ValueError(f"failure diagnostics have no {split} stage counts")
        result[split] = {
            "early_4_10_targets": _required_int(
                stages.get("early_4_10_targets"),
                field=f"splits.{split}.stage_counts.early_4_10_targets",
            ),
            "preferred_candidate_possible": _required_int(
                stages.get("preferred_candidate_possible"),
                field=f"splits.{split}.stage_counts.preferred_candidate_possible",
            ),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only top-k audit for the fixed v81 fit source whitelist."
    )
    parser.add_argument("--fit-checkpoint", type=Path, default=DEFAULT_FIT)
    parser.add_argument("--failure", type=Path, default=DEFAULT_FAILURE)
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    args = parser.parse_args()

    output_resolved = args.output.resolve()
    protected = {
        args.fit_checkpoint.resolve(),
        args.failure.resolve(),
        args.parent.resolve(),
        Path(__file__).resolve(),
    }
    if output_resolved in protected:
        raise ValueError("audit output must not overwrite an input or this script")

    failure = _read_json(args.failure)
    selected_inventory = _select_allowed_inventory(failure)
    failure_fit = failure.get("fit_checkpoint")
    if not isinstance(failure_fit, Mapping):
        raise ValueError("failure diagnostics have no fit checkpoint record")
    expected_adapter_raw = failure_fit.get("adapter_config")
    if not isinstance(expected_adapter_raw, Mapping):
        raise ValueError("failure diagnostics have no adapter configuration")
    expected_adapter = ResidualAdapterConfig(**dict(expected_adapter_raw))
    if expected_adapter.action_count != ACTION_COUNT:
        raise ValueError("this audit requires the 18-action STG action space")

    parent, _parent_metadata = load_checkpoint(args.parent, device=args.device)
    parent.eval()
    if asdict(parent.config).get("action_count") != ACTION_COUNT:
        raise ValueError("parent checkpoint does not use the 18-action space")
    adapter, fit_metadata = _load_fit_checkpoint(
        args.fit_checkpoint,
        parent_checkpoint=args.parent,
        parent_policy_config=asdict(parent.config),
        expected_adapter_config=expected_adapter,
        device=args.device,
    )
    if fit_metadata["fit_checkpoint_sha256"] != failure_fit.get(
        "fit_checkpoint_sha256"
    ):
        raise ValueError("fit checkpoint hash does not match failure diagnostics")
    if fit_metadata["verified_parent_checkpoint_sha256"] != failure.get(
        "parent_checkpoint_sha256"
    ):
        raise ValueError("parent checkpoint hash does not match failure diagnostics")

    fit_inventory = _selected_fit_inventory(fit_metadata)
    triplets, source_provenance = _verify_allowed_sources(
        selected_inventory,
        fit_inventory,
    )
    training_metadata = fit_metadata["training_metadata"]
    label_metadata = training_metadata.get("label_metadata")
    if not isinstance(label_metadata, Mapping):
        raise ValueError("fit checkpoint has no label metadata")
    if label_metadata.get("future_onset_gate") is not True:
        raise ValueError("fit checkpoint is not a future-onset fit")
    if label_metadata.get("future_onset_horizon_decisions") != (
        FUTURE_ONSET_HORIZON_DECISIONS
    ):
        raise ValueError("fit checkpoint future-onset horizon is incompatible")

    episodes = [
        _load_episode(
            parent,
            adapter,
            dataset,
            report,
            manifest,
            parent_checkpoint_sha256=fit_metadata[
                "verified_parent_checkpoint_sha256"
            ],
            device=args.device,
            chunk_length=256,
            safe_regret=float(label_metadata["safe_regret"]),
            minimum_parent_margin=float(label_metadata["minimum_parent_margin"]),
            minimum_margin_gain=float(label_metadata["minimum_margin_gain"]),
            predecessor_decisions=int(label_metadata["predecessor_decisions"]),
            future_onset_gate=True,
        )
        for dataset, report, manifest in triplets
    ]
    if [episode.seed for episode in episodes] != [
        *TRAINING_SEEDS,
        CALIBRATION_SEED,
    ]:
        raise ValueError("loaded episodes do not match the ordered audit whitelist")
    _apply_existing_normalization(adapter, episodes)
    adapter.to(args.device).eval()

    probabilities: dict[int, torch.Tensor] = {}
    finite_masks: dict[int, torch.Tensor] = {}
    for episode in episodes:
        probabilities[episode.seed], finite_masks[episode.seed] = (
            _predict_action_probabilities(
                adapter,
                episode,
                device=args.device,
            )
        )

    training = [episode for episode in episodes if episode.seed in TRAINING_SEEDS]
    calibration = [episode for episode in episodes if episode.seed == CALIBRATION_SEED]
    training_summary, _training_details = _audit_split(
        training,
        probabilities,
        finite_masks,
        include_details=False,
    )
    calibration_summary, calibration_details = _audit_split(
        calibration,
        probabilities,
        finite_masks,
        include_details=True,
    )
    expected_counts = _failure_expected_counts(failure)
    cross_checks = {
        "training": {
            "computed_early_4_10_targets": training_summary["targets"],
            "failure_early_4_10_targets": expected_counts["training"][
                "early_4_10_targets"
            ],
            "computed_preferred_candidate_path": training_summary[
                "preferred_candidate_path"
            ],
            "failure_preferred_candidate_possible": expected_counts["training"][
                "preferred_candidate_possible"
            ],
        },
        "calibration": {
            "computed_early_4_10_targets": calibration_summary["targets"],
            "failure_early_4_10_targets": expected_counts["calibration"][
                "early_4_10_targets"
            ],
            "computed_preferred_candidate_path": calibration_summary[
                "preferred_candidate_path"
            ],
            "failure_preferred_candidate_possible": expected_counts[
                "calibration"
            ]["preferred_candidate_possible"],
        },
    }
    if any(
        values["computed_early_4_10_targets"]
        != values["failure_early_4_10_targets"]
        or values["computed_preferred_candidate_path"]
        != values["failure_preferred_candidate_possible"]
        for values in cross_checks.values()
    ):
        raise ValueError("action audit does not reproduce the failure funnel")
    if len(calibration_details) != 27:
        raise ValueError(
            f"calibration detail count must be 27, got {len(calibration_details)}"
        )

    script_path = Path(__file__)
    helper_path = script_path.with_name("train_temporal_residual_adapter.py")
    residual_path = script_path.parent.parent / "src/stg_lab/residual_adapter.py"
    payload = {
        "schema_version": 1,
        "kind": "v81_action_head_read_only_audit",
        "read_only": True,
        "training_performed": False,
        "deployment_claim": False,
        "protocol": {
            "training_seed_whitelist": list(TRAINING_SEEDS),
            "calibration_seed_whitelist": [CALIBRATION_SEED],
            "prohibited_source_seeds_not_opened_or_hashed": list(
                PROHIBITED_SOURCE_SEEDS
            ),
            "inventory_guard": (
                "Seed and expected role are checked before any source path "
                "field is inspected, resolved, hashed, or opened."
            ),
            "target_mask": (
                "anticipatory == true and 4 <= anticipatory_lead_decisions <= 10"
            ),
            "decision_indexing": "zero_based",
            "action_semantics": {
                "action_ids": "0..17",
                "direction": "action_id % 9",
                "speed": "fast for action_id < 9; slow for action_id >= 9",
                "paired_actions": "direction d uses fast=d and slow=d+9",
            },
            "probability_protocol": (
                "For each recurrent ensemble member, decode its action logits "
                "against the frozen parent logits, apply finite softmax, then "
                "average the three member probability vectors."
            ),
            "metric_definitions": {
                "exact_top_k": (
                    "preferred_action appears in the k largest entries of the "
                    "ensemble-mean 18-action probability vector"
                ),
                "raw_top1_direction": "top1_action % 9 equals target % 9",
                "direction_marginal_top1": (
                    "argmax_d(probability[d] + probability[d+9]) equals "
                    "target % 9"
                ),
                "raw_top1_speed": (
                    "the fast/slow half containing raw top1 equals the target half"
                ),
                "speed_marginal_top1": (
                    "argmax(sum probabilities[0:9], sum probabilities[9:18]) "
                    "equals the target speed"
                ),
                "speed_confusion_matrix": (
                    "rows are target speed and columns are predicted speed"
                ),
                "preferred_candidate_path": (
                    "raw top1 equals preferred_action and differs from parent_action; "
                    "this reproduces the calibration-failure funnel field"
                ),
            },
        },
        "provenance": {
            "audit_script": str(script_path),
            "audit_script_sha256": file_sha256(script_path),
            "implementation_sources": [
                {
                    "path": str(helper_path),
                    "sha256": file_sha256(helper_path),
                },
                {
                    "path": str(residual_path),
                    "sha256": file_sha256(residual_path),
                },
            ],
            "fit_checkpoint": str(args.fit_checkpoint),
            "fit_checkpoint_sha256": fit_metadata["fit_checkpoint_sha256"],
            "failure_diagnostics": str(args.failure),
            "failure_diagnostics_sha256": file_sha256(args.failure),
            "parent_checkpoint": str(args.parent),
            "parent_checkpoint_sha256": fit_metadata[
                "verified_parent_checkpoint_sha256"
            ],
            "allowed_episode_sources": source_provenance,
        },
        "model": {
            "fit_checkpoint_version": fit_metadata["version"],
            "fit_checkpoint_kind": fit_metadata["kind"],
            "parent_policy_config": asdict(parent.config),
            "adapter_config": asdict(adapter.config),
            "label_metadata": dict(label_metadata),
            "device": args.device,
        },
        "cross_checks": cross_checks,
        "results": {
            "training": training_summary,
            "calibration": calibration_summary,
            "calibration_seed_10306_early_lead_4_10_details": (
                calibration_details
            ),
        },
    }
    _write_json_atomic(args.output, payload)
    print(json.dumps({
        "output": str(args.output),
        "output_sha256": file_sha256(args.output),
        "training": training_summary,
        "calibration": calibration_summary,
        "calibration_detail_rows": len(calibration_details),
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
