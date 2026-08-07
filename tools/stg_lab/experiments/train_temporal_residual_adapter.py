"""Train a temporal correction gate from strict successful native DAgger runs."""

from __future__ import annotations

import argparse
import json
import math
import random
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from stg_lab.provenance import file_sha256
from stg_lab.residual_adapter import (
    ResidualAdapterConfig,
    ResidualCorrectionAdapter,
    ResidualRuntimeConfig,
    ensemble_action_summary,
    finite_action_probabilities,
    finite_sigmoid,
    load_residual_adapter,
    residual_candidate_selection,
    residual_future_onset_mask,
    residual_override_masks,
    save_residual_adapter,
    semantic_player_position_features,
)
from stg_lab.rollout import scenario_memory_vector
from stg_lab.training import (
    TEACHER_ACTION_COLLIDED_INDEX,
    TEACHER_ACTION_MINIMUM_MARGIN_INDEX,
    TEACHER_ACTION_SELECTED_INDEX,
    Demonstrations,
    load_checkpoint,
)

FUTURE_ONSET_HORIZON_DECISIONS = 10
EARLY_ONSET_MINIMUM_LEAD_DECISIONS = 4
FIT_CHECKPOINT_VERSION = 2
MEMBERSHIP_CONFIDENCE_FIT_CHECKPOINT_VERSION = 3
LEGACY_FIT_CHECKPOINT_VERSIONS = (1,)
FIT_CHECKPOINT_KIND = "temporal_residual_fit_checkpoint"
PREFERRED_ACTION_TARGET_SEMANTICS = "certified_equivalence_set_v1"
PREFERRED_ACTION_TIEBREAK_SEMANTICS = "conditional_certified_previous_action_v1"
PREFERRED_ACTION_UNIFORM_SOFT_TARGET_SEMANTICS = (
    "uniform_within_certified_equivalence_set_v1"
)
MEMBERSHIP_LOSS_MODES = ("balanced", "unweighted")
MEMBERSHIP_CONFIDENCE_LOSS_MODES = ("balanced", "unweighted")
MEMBERSHIP_LOSS_SEMANTICS = {
    "balanced": (
        "per-row balanced independent membership BCE with equal total "
        "positive and negative weight"
    ),
    "unweighted": (
        "unweighted independent membership BCE, mean over action cells within "
        "each labelled row then mean over labelled rows"
    ),
}
MEMBERSHIP_CONFIDENCE_TRAINING_SEMANTICS = {
    "version": 1,
    "target": "certified_equivalence_membership_on_valid_positive_rows",
    "cells_per_row": 18,
    "head_input": "detached_selector_recurrent_latent",
    "optimizer": "independent_adamw_with_base_lr_and_weight_decay",
    "gradient_clip": "independent_parameter_group_max_norm_5",
    "selector_gradient_from_membership_loss": False,
}
GRADIENT_CLIP_MAX_NORM = 5.0
ACTION_BRANCH_MODULE_NAMES = (
    "action_input_projection",
    "action_recurrent",
    "action_head",
)
GLOBAL_GRADIENT_CLIP_SEMANTICS = "single_global_trainable_parameter_group"
SEPARATE_ACTION_GRADIENT_CLIP_SEMANTICS = (
    "independent_action_and_shared_safety_trainable_parameter_groups"
)
ANTICIPATORY_LEAD_BUCKETS = (
    ("1_3", 1, 3),
    ("4_6", 4, 6),
    ("7_10", 7, 10),
)


@dataclass(slots=True)
class EpisodeFeatures:
    seed: int
    dataset: str
    report: str
    manifest: str
    features: torch.Tensor
    parent_logits: torch.Tensor
    parent_actions: torch.Tensor
    previous_actions: torch.Tensor
    gate_targets: torch.Tensor
    gate_valid: torch.Tensor
    hard_positive: torch.Tensor
    correctable_hard_positive: torch.Tensor
    anticipatory: torch.Tensor
    future_onset_valid: torch.Tensor
    anticipatory_lead_decisions: torch.Tensor
    preferred_actions: torch.Tensor
    preferred_action_set: torch.Tensor
    preferred_equivalent_actions: torch.Tensor
    preferred_correction_required: torch.Tensor
    safety_candidate_actions: torch.Tensor
    safety_candidate_valid: torch.Tensor
    safe_actions: torch.Tensor
    evaluation_safe_actions: torch.Tensor
    parent_evaluation_danger: torch.Tensor
    collided_actions: torch.Tensor
    minimum_margins: torch.Tensor
    minimum_margin_mask: torch.Tensor
    teacher_selected_collision: torch.Tensor
    global_frame_dtype: str = "unknown"
    local_frame_dtype: str = "unknown"

    @property
    def decisions(self) -> int:
        return int(self.features.shape[1])


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(dict(payload), temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


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


def _save_fit_checkpoint(
    adapter: ResidualCorrectionAdapter,
    path: Path,
    *,
    parent_checkpoint: Path,
    parent_policy_config: Mapping[str, Any],
    training_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Atomically save fitted weights without making a deployment claim."""

    output = Path(path)
    parent_path = Path(parent_checkpoint)
    if not parent_path.is_file():
        raise FileNotFoundError(f"parent checkpoint does not exist: {parent_path}")
    if output.resolve() == parent_path.resolve():
        raise ValueError("fit checkpoint cannot overwrite its parent checkpoint")
    state = {
        name: value.detach().cpu().clone()
        for name, value in adapter.state_dict().items()
    }
    if not all(bool(torch.isfinite(value).all()) for value in state.values()):
        raise ValueError("fit checkpoint adapter state must be finite")
    mean = adapter.feature_mean.detach().cpu().clone()
    scale = adapter.feature_scale.detach().cpu().clone()
    if not bool(torch.isfinite(mean).all()):
        raise ValueError("fit checkpoint feature mean must be finite")
    if not bool(torch.isfinite(scale).all()) or not bool((scale > 0.0).all()):
        raise ValueError("fit checkpoint feature scale must be finite and positive")
    fit_checkpoint_version = (
        MEMBERSHIP_CONFIDENCE_FIT_CHECKPOINT_VERSION
        if adapter.config.per_action_membership_confidence else
        FIT_CHECKPOINT_VERSION
    )
    payload = {
        "version": fit_checkpoint_version,
        "kind": FIT_CHECKPOINT_KIND,
        "deployment_artifact": False,
        "deployment_eligible": False,
        "calibration_complete": False,
        "acceptance_claim": False,
        "parent_checkpoint": str(parent_path),
        "parent_checkpoint_sha256": file_sha256(parent_path),
        "parent_policy_config": dict(parent_policy_config),
        "adapter_config": asdict(adapter.config),
        "feature_normalization": {
            "mean": mean,
            "scale": scale,
        },
        "state_dict": state,
        "training_metadata": dict(training_metadata),
    }
    _atomic_torch_save(payload, output)
    return {
        "fit_checkpoint": str(output),
        "fit_checkpoint_sha256": file_sha256(output),
        "kind": FIT_CHECKPOINT_KIND,
        "deployment_artifact": False,
        "parent_checkpoint_sha256": payload["parent_checkpoint_sha256"],
        "adapter_config": payload["adapter_config"],
    }


def _load_fit_checkpoint(
    path: Path,
    *,
    parent_checkpoint: Path,
    parent_policy_config: Mapping[str, Any],
    expected_adapter_config: ResidualAdapterConfig | Mapping[str, Any],
    device: str = "cpu",
) -> tuple[ResidualCorrectionAdapter, dict[str, Any]]:
    """Load only an uncalibrated fit checkpoint bound to its exact parent."""

    checkpoint_path = Path(path)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("fit checkpoint must be a mapping")
    version = payload.get("version")
    if (
        version not in (
            *LEGACY_FIT_CHECKPOINT_VERSIONS,
            FIT_CHECKPOINT_VERSION,
            MEMBERSHIP_CONFIDENCE_FIT_CHECKPOINT_VERSION,
        )
        or payload.get("kind") != FIT_CHECKPOINT_KIND
    ):
        raise ValueError("unsupported temporal residual fit checkpoint")
    if any(
        payload.get(name) is not False
        for name in (
            "deployment_artifact",
            "deployment_eligible",
            "calibration_complete",
            "acceptance_claim",
        )
    ):
        raise ValueError("fit checkpoint must not contain deployment claims")
    actual_parent_sha = file_sha256(parent_checkpoint)
    if payload.get("parent_checkpoint_sha256") != actual_parent_sha:
        raise ValueError("fit checkpoint parent hash does not match")
    if payload.get("parent_policy_config") != dict(parent_policy_config):
        raise ValueError("fit checkpoint parent policy config does not match")
    expected_values = (
        asdict(expected_adapter_config)
        if isinstance(expected_adapter_config, ResidualAdapterConfig) else
        dict(expected_adapter_config)
    )
    raw_adapter_values = payload.get("adapter_config")
    if not isinstance(raw_adapter_values, Mapping):
        raise ValueError("fit checkpoint adapter config is invalid")
    adapter_values = dict(raw_adapter_values)
    if version in LEGACY_FIT_CHECKPOINT_VERSIONS:
        if adapter_values.get("action_logit_mode", "absolute") != "absolute":
            raise ValueError("legacy fit checkpoints cannot contain residual logits")
        if adapter_values.get("semantic_player_position", False) is not False:
            raise ValueError(
                "legacy fit checkpoints cannot contain semantic position inputs"
            )
        if adapter_values.get("separate_action_recurrent", False) is not False:
            raise ValueError(
                "legacy fit checkpoints cannot contain separate action recurrence"
            )
    adapter_values.setdefault("action_logit_mode", "absolute")
    adapter_values.setdefault("semantic_player_position", False)
    adapter_values.setdefault("separate_action_recurrent", False)
    adapter_values.setdefault("per_action_membership_confidence", False)
    if (
        version != MEMBERSHIP_CONFIDENCE_FIT_CHECKPOINT_VERSION
        and adapter_values["per_action_membership_confidence"] is not False
    ):
        raise ValueError(
            "fit checkpoint version predates per-action membership confidence"
        )
    if (
        version == MEMBERSHIP_CONFIDENCE_FIT_CHECKPOINT_VERSION
        and adapter_values["per_action_membership_confidence"] is not True
    ):
        raise ValueError(
            "membership-confidence fit checkpoint must enable its auxiliary head"
        )
    if adapter_values != expected_values:
        raise ValueError("fit checkpoint adapter config does not match")
    state = payload.get("state_dict")
    normalization = payload.get("feature_normalization")
    training_metadata = payload.get("training_metadata")
    if not isinstance(state, Mapping) or not all(
        isinstance(value, torch.Tensor) and bool(torch.isfinite(value).all())
        for value in state.values()
    ):
        raise ValueError("fit checkpoint state dictionary must be finite")
    if not isinstance(normalization, Mapping):
        raise ValueError("fit checkpoint normalization metadata is invalid")
    mean = normalization.get("mean")
    scale = normalization.get("scale")
    if not isinstance(mean, torch.Tensor) or not isinstance(scale, torch.Tensor):
        raise ValueError("fit checkpoint normalization tensors are missing")
    if not bool(torch.isfinite(mean).all()):
        raise ValueError("fit checkpoint feature mean must be finite")
    if not bool(torch.isfinite(scale).all()) or not bool((scale > 0.0).all()):
        raise ValueError("fit checkpoint feature scale must be finite and positive")
    state_mean = state.get("feature_mean")
    state_scale = state.get("feature_scale")
    if (
        not isinstance(state_mean, torch.Tensor)
        or not isinstance(state_scale, torch.Tensor)
        or not torch.equal(state_mean, mean)
        or not torch.equal(state_scale, scale)
    ):
        raise ValueError("fit checkpoint normalization does not match adapter state")
    if not isinstance(training_metadata, Mapping):
        raise ValueError("fit checkpoint training metadata is invalid")
    adapter = ResidualCorrectionAdapter(
        ResidualAdapterConfig(**dict(adapter_values)),
    ).to(device)
    adapter.load_state_dict(dict(state), strict=True)
    adapter.eval()
    metadata = dict(payload)
    metadata["adapter_config"] = adapter_values
    metadata.pop("state_dict", None)
    metadata.update({
        "fit_checkpoint": str(checkpoint_path),
        "fit_checkpoint_sha256": file_sha256(checkpoint_path),
        "verified_parent_checkpoint": str(parent_checkpoint),
        "verified_parent_checkpoint_sha256": actual_parent_sha,
    })
    return adapter, metadata


def _validate_fit_resume_metadata(
    metadata: Mapping[str, Any],
    *,
    source_inventory: list[dict[str, Any]],
    label_metadata: Mapping[str, Any],
) -> None:
    training = metadata.get("training_metadata")
    if not isinstance(training, Mapping):
        raise ValueError("fit checkpoint training metadata is invalid")
    if training.get("source_inventory") != source_inventory:
        raise ValueError("fit checkpoint source inventory or split roles do not match")
    if training.get("label_metadata") != dict(label_metadata):
        raise ValueError("fit checkpoint label configuration does not match")


def _strict_protocol_mapping(
    value: Any,
    field: str,
    dataset: Path,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"DAgger report protocol field {field} must be an object: {dataset}")
    return value


def _strict_protocol_integer(
    value: Any,
    expected: int,
    field: str,
    dataset: Path,
) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise ValueError(
            f"DAgger report protocol field {field} must be integer {expected}: {dataset}"
        )


def _strict_protocol_number(
    value: Any,
    expected: float,
    field: str,
    dataset: Path,
) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) != expected
    ):
        raise ValueError(
            f"DAgger report protocol field {field} must be finite {expected}: {dataset}"
        )


def _strict_success(report: dict[str, Any], dataset: Path, manifest: Path) -> int:
    if report.get("success") is not True:
        raise ValueError(f"DAgger report is not a strict success: {dataset}")
    if report.get("termination_reason") != "attack_complete":
        raise ValueError(f"DAgger report did not complete the attack: {dataset}")
    final = report.get("outcome_evidence", {}).get("final_player", {})
    death = final.get("death")
    if (
        isinstance(death, bool)
        or not isinstance(death, (int, float))
        or not math.isfinite(float(death))
        or float(death) != 0.0
    ):
        raise ValueError(f"DAgger report has nonzero death evidence: {dataset}")
    if report.get("continuous_fire") is not True:
        raise ValueError(f"DAgger report did not require continuous fire: {dataset}")
    shoot_rate = report.get("shoot_command_rate")
    if (
        isinstance(shoot_rate, bool)
        or not isinstance(shoot_rate, (int, float))
        or not math.isfinite(float(shoot_rate))
        or float(shoot_rate) != 1.0
    ):
        raise ValueError(f"DAgger report did not fire on every frame: {dataset}")
    config = _strict_protocol_mapping(report.get("config"), "config", dataset)
    if config.get("spell_forced_off") is not True:
        raise ValueError(f"DAgger report did not force spell off: {dataset}")
    if config.get("record_teacher_evaluations") is not True:
        raise ValueError(f"DAgger report lacks full teacher evaluations: {dataset}")
    if config.get("supervision_mode") != "corrective":
        raise ValueError(f"DAgger report is not corrective: {dataset}")
    _strict_protocol_integer(
        config.get("decision_interval"), 3, "config.decision_interval", dataset,
    )
    _strict_protocol_integer(
        config.get("observation_delay"), 5, "config.observation_delay", dataset,
    )
    vision = _strict_protocol_mapping(config.get("vision"), "config.vision", dataset)
    for field, expected in (
        ("global_width", 48),
        ("global_height", 56),
        ("local_width", 40),
        ("local_height", 40),
        ("history", 1),
        ("observation_delay", 5),
        ("channels", 6),
    ):
        _strict_protocol_integer(
            vision.get(field), expected, f"config.vision.{field}", dataset,
        )
    for field in ("local_extent_x", "local_extent_y"):
        _strict_protocol_number(
            vision.get(field), 72.0, f"config.vision.{field}", dataset,
        )
    controller = _strict_protocol_mapping(
        report.get("controller"), "controller", dataset,
    )
    student = _strict_protocol_mapping(
        controller.get("student"), "controller.student", dataset,
    )
    if student.get("action_selection") != "joint":
        raise ValueError(
            "DAgger report protocol field controller.student.action_selection "
            f"must be 'joint': {dataset}"
        )
    if student.get("action_selection_uses_safety_state") is not False:
        raise ValueError(
            "DAgger report protocol field "
            "controller.student.action_selection_uses_safety_state must be false: "
            f"{dataset}"
        )
    demonstrations = report.get("demonstrations", {})
    if demonstrations.get("dataset_sha256") != file_sha256(dataset):
        raise ValueError(f"DAgger dataset hash does not match its report: {dataset}")
    manifest_value = _read_json(manifest)
    if manifest_value.get("dataset_sha256") != file_sha256(dataset):
        raise ValueError(f"DAgger dataset hash does not match its manifest: {dataset}")
    accepted = manifest_value.get("accepted_episodes")
    if not isinstance(accepted, list) or len(accepted) != 1:
        raise ValueError(f"DAgger manifest must contain one accepted episode: {dataset}")
    accepted_episode = accepted[0]
    if not isinstance(accepted_episode, dict):
        raise ValueError(f"DAgger manifest episode is invalid: {dataset}")
    if accepted_episode.get("strict_success") is not True:
        raise ValueError(f"DAgger manifest episode is not a strict success: {dataset}")
    seed = report.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"DAgger report has an invalid seed: {dataset}")
    manifest_seed = accepted_episode.get("seed")
    if isinstance(manifest_seed, bool) or not isinstance(manifest_seed, int):
        raise ValueError(f"DAgger manifest has an invalid episode seed: {dataset}")
    if manifest_seed != seed:
        raise ValueError(f"DAgger manifest seed does not match its report: {dataset}")
    return seed


def _source_triplets(args: argparse.Namespace) -> list[tuple[Path, Path, Path]]:
    datasets = list(args.dagger_dataset)
    reports = list(args.dagger_report)
    manifests = list(args.dagger_manifest)
    if not datasets or not (len(datasets) == len(reports) == len(manifests)):
        raise ValueError(
            "--dagger-dataset, --dagger-report, and --dagger-manifest must align"
        )
    return list(zip(datasets, reports, manifests, strict=True))


def _validate_distinct_workflow_paths(
    *,
    outputs: Mapping[str, Path | None],
    protected_inputs: Mapping[str, Path | None],
    source_paths: list[Path],
) -> None:
    resolved_outputs: dict[Path, str] = {}
    for name, path in outputs.items():
        if path is None:
            continue
        resolved = path.resolve()
        previous = resolved_outputs.get(resolved)
        if previous is not None:
            raise ValueError(f"{name} and {previous} must use different paths")
        resolved_outputs[resolved] = name
    protected = {
        path.resolve(): name
        for name, path in protected_inputs.items()
        if path is not None
    }
    resolved_sources = {path.resolve() for path in source_paths}
    for output, name in resolved_outputs.items():
        if output in protected:
            raise ValueError(f"{name} cannot overwrite {protected[output]}")
        if output in resolved_sources:
            raise ValueError(f"{name} cannot overwrite a DAgger source file")
    for path, name in protected.items():
        if name in {"resume fit checkpoint", "frozen adapter"} and (
            path in resolved_sources
        ):
            raise ValueError(f"{name} cannot also be a DAgger source file")


def _episode_memory(model: Any, report: dict[str, Any]) -> np.ndarray:
    student = report.get("controller", {}).get("student", {})
    scenario_key = student.get("scenario_key")
    if not isinstance(scenario_key, str) or not scenario_key:
        raise ValueError("DAgger report has no student scenario key")
    vocabulary = getattr(model, "scenario_vocabulary", None)
    if vocabulary is None:
        raise ValueError("parent checkpoint has no scenario vocabulary")
    if int(getattr(model, "previous_action_size", 0)) != 0:
        raise ValueError("temporal residual training does not accept action-conditioned parents")
    return scenario_memory_vector(
        scenario_key,
        model.config.memory_size,
        vocabulary,
    )


def _parent_stream(
    model: Any,
    demonstrations: Demonstrations,
    memory_vector: np.ndarray,
    *,
    device: str,
    chunk_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    recurrent_values: list[torch.Tensor] = []
    logit_values: list[torch.Tensor] = []
    visual_values: list[torch.Tensor] = []
    hidden = None
    model.eval()
    with torch.no_grad():
        for start in range(0, len(demonstrations.actions), chunk_length):
            stop = min(start + chunk_length, len(demonstrations.actions))
            global_frames = torch.from_numpy(
                demonstrations.global_frames[start:stop, -1],
            ).float().unsqueeze(0).to(device)
            local_frames = torch.from_numpy(
                demonstrations.local_frames[start:stop, -1],
            ).float().unsqueeze(0).to(device)
            memory = torch.from_numpy(
                np.broadcast_to(
                    memory_vector,
                    (stop - start, len(memory_vector)),
                ).copy(),
            ).float().unsqueeze(0).to(device)
            logits, _risk, hidden, recurrent, visual = (
                model.forward_with_visual_features(
                global_frames,
                local_frames,
                memory,
                None,
                hidden,
                )
            )
            recurrent_values.append(recurrent.detach().cpu())
            logit_values.append(logits.detach().cpu())
            visual_values.append(visual.detach().cpu())
            hidden = hidden.detach()
    return (
        torch.cat(recurrent_values, dim=1),
        torch.cat(logit_values, dim=1),
        torch.cat(visual_values, dim=1),
    )


def _movement_mask(device: torch.device | str = "cpu") -> torch.Tensor:
    return torch.arange(18, device=device).remainder(9) != 4


def _executed_action_context(
    demonstrations: Demonstrations,
    *,
    action_count: int = 18,
) -> torch.Tensor:
    """Recreate the live previous-action state before every recorded decision."""

    if demonstrations.actions.shape[1] != 1:
        raise ValueError("residual action context requires one-step native streams")
    if demonstrations.previous_actions is None:
        raise ValueError("residual action context requires recorded previous actions")
    previous = np.asarray(demonstrations.previous_actions[:, -1], dtype=np.int64)
    executed = np.asarray(demonstrations.actions[:, -1], dtype=np.int64)
    if previous.size == 0 or int(previous[0]) != -1:
        raise ValueError("native residual episodes must start without a previous action")
    if not np.array_equal(previous[1:], executed[:-1]):
        raise ValueError("recorded previous actions do not match executed actions")

    context = torch.zeros(
        (1, len(previous), action_count + 2),
        dtype=torch.float32,
    )
    held_action = -1
    held_decisions = 0
    for index, raw_action in enumerate(previous.tolist()):
        action = int(raw_action)
        if action < 0:
            held_action = -1
            held_decisions = 0
            continue
        if action == held_action:
            held_decisions += 1
        else:
            held_action = action
            held_decisions = 1
        context[0, index, action] = 1.0
        context[0, index, -2] = 1.0
        context[0, index, -1] = math.log1p(held_decisions)
    return context


def _labels_from_evidence(
    demonstrations: Demonstrations,
    parent_logits: torch.Tensor,
    *,
    safe_regret: float,
    minimum_parent_margin: float,
    minimum_margin_gain: float,
    predecessor_decisions: int,
    future_onset_gate: bool = False,
) -> dict[str, torch.Tensor]:
    evaluations = torch.from_numpy(
        demonstrations.teacher_action_evaluations[:, -1],
    ).float()
    regrets = torch.from_numpy(
        demonstrations.teacher_action_regrets[:, -1],
    ).float()
    available = torch.from_numpy(
        demonstrations.teacher_action_evaluation_mask[:, -1],
    ).bool()
    if not bool(available.all()):
        raise ValueError("raw DAgger sources must evaluate every decision")
    collided = evaluations[..., TEACHER_ACTION_COLLIDED_INDEX] >= 0.5
    selected = evaluations[..., TEACHER_ACTION_SELECTED_INDEX] >= 0.5
    if not bool((selected.sum(dim=-1) == 1).all()):
        raise ValueError("every DAgger decision must select one teacher action")
    teacher_actions = selected.to(torch.int64).argmax(dim=-1)
    teacher_moving = _movement_mask()[teacher_actions]
    same_motion = _movement_mask().unsqueeze(0) == teacher_moving.unsqueeze(-1)
    safe_actions = (~collided) & (regrets <= safe_regret) & same_motion
    teacher_selected_collision = collided.gather(
        -1, teacher_actions.unsqueeze(-1),
    ).squeeze(-1)
    parent_actions = parent_logits[0].argmax(dim=-1)
    parent_collided = collided.gather(
        -1, parent_actions.unsqueeze(-1),
    ).squeeze(-1)
    margins = evaluations[..., TEACHER_ACTION_MINIMUM_MARGIN_INDEX]
    minimum_margin_mask = torch.isfinite(margins)
    evaluation_safe_actions = (
        ~collided
        & minimum_margin_mask
        & (margins >= minimum_parent_margin)
    )
    teacher_margin = margins.gather(
        -1, teacher_actions.unsqueeze(-1),
    ).squeeze(-1)
    parent_margin = margins.gather(
        -1, parent_actions.unsqueeze(-1),
    ).squeeze(-1)
    margin_gain = teacher_margin - parent_margin
    evidence_valid = available & ~teacher_selected_collision & safe_actions.any(dim=-1)
    hard_positive = (
        evidence_valid
        & (parent_collided | (parent_margin < minimum_parent_margin))
        & (margin_gain >= minimum_margin_gain)
    )
    reliable_negative = evidence_valid & ~hard_positive
    gate_valid = hard_positive | reliable_negative
    gate_targets = hard_positive.to(torch.float32)
    anticipatory = torch.zeros_like(hard_positive)
    anticipatory_lead_decisions = torch.zeros_like(teacher_actions)
    preferred_actions = torch.full_like(teacher_actions, -1)
    safety_candidate_actions = torch.full_like(teacher_actions, -1)
    if demonstrations.previous_actions is None:
        raise ValueError("preferred residual labels require recorded previous actions")
    previous_actions = torch.from_numpy(
        demonstrations.previous_actions[:, -1],
    ).to(torch.int64)
    safe_previous = (
        (previous_actions >= 0)
        & safe_actions.gather(
            -1,
            previous_actions.clamp_min(0).unsqueeze(-1),
        ).squeeze(-1)
    )
    preferred_safe_previous = safe_previous
    if future_onset_gate:
        preferred_safe_previous = (
            safe_previous
            & evaluation_safe_actions.gather(
                -1,
                previous_actions.clamp_min(0).unsqueeze(-1),
            ).squeeze(-1)
        )
    safety_candidate_valid = (
        evidence_valid
        & (parent_collided | (parent_margin < minimum_parent_margin))
    )
    if future_onset_gate:
        strict_safe_actions = safe_actions & evaluation_safe_actions
        strict_safe_available = strict_safe_actions.any(dim=-1)
        strict_safe_margins = margins.masked_fill(~strict_safe_actions, -torch.inf)
        safest_actions = strict_safe_margins.argmax(dim=-1)
        strict_safe_teacher = strict_safe_actions.gather(
            -1,
            teacher_actions.unsqueeze(-1),
        ).squeeze(-1)
        safety_candidate_valid &= strict_safe_available
        safety_candidate_actions[safety_candidate_valid] = torch.where(
            preferred_safe_previous[safety_candidate_valid],
            previous_actions[safety_candidate_valid],
            torch.where(
                strict_safe_teacher[safety_candidate_valid],
                teacher_actions[safety_candidate_valid],
                safest_actions[safety_candidate_valid],
            ),
        )
    else:
        safety_candidate_actions[safety_candidate_valid] = torch.where(
            preferred_safe_previous[safety_candidate_valid],
            previous_actions[safety_candidate_valid],
            teacher_actions[safety_candidate_valid],
        )
    preferred_actions[hard_positive] = torch.where(
        preferred_safe_previous[hard_positive],
        previous_actions[hard_positive],
        teacher_actions[hard_positive],
    )
    correctable_hard_positive = hard_positive.clone()
    if future_onset_gate:
        preferred_indices = preferred_actions.clamp_min(0).unsqueeze(-1)
        preferred_evaluation_safe = evaluation_safe_actions.gather(
            -1,
            preferred_indices,
        ).squeeze(-1)
        correctable_hard_positive &= (
            (preferred_actions >= 0) & preferred_evaluation_safe
        )
        # Current danger remains available to the physical critic and audit,
        # but the future gate must not learn an onset with no certified escape.
        gate_targets = correctable_hard_positive.to(torch.float32)

    propagation_decisions = (
        FUTURE_ONSET_HORIZON_DECISIONS
        if future_onset_gate else
        predecessor_decisions
    )
    onset_positive = (
        correctable_hard_positive if future_onset_gate else hard_positive
    )
    starts = torch.nonzero(
        onset_positive
        & ~torch.cat((torch.zeros(1, dtype=torch.bool), onset_positive[:-1])),
        as_tuple=False,
    ).flatten().tolist()
    for start in starts:
        preferred_action = int(preferred_actions[start])
        future_safe_actions = safe_actions[start].clone()
        if future_onset_gate:
            future_safe_actions &= evaluation_safe_actions[start]
            if not bool(future_safe_actions[preferred_action]):
                continue
        for distance in range(1, propagation_decisions + 1):
            index = start - distance
            if (
                index < 0
                or bool(teacher_selected_collision[index])
                or bool(hard_positive[index])
            ):
                break
            propagated_safe = future_safe_actions & ~collided[index]
            if future_onset_gate:
                propagated_safe &= evaluation_safe_actions[index]
            if not bool(propagated_safe[preferred_action]):
                break
            target = (
                1.0
                if future_onset_gate else
                (predecessor_decisions + 1 - distance)
                / (predecessor_decisions + 1)
            )
            gate_targets[index] = max(float(gate_targets[index]), target)
            gate_valid[index] = True
            anticipatory[index] = True
            anticipatory_lead_decisions[index] = distance
            safe_actions[index] = propagated_safe
            preferred_actions[index] = preferred_action
            future_safe_actions = propagated_safe

    if future_onset_gate:
        # A negative close to episode termination has less than the complete
        # 30-frame future window and is therefore right-censored. Positives are
        # observable events and remain valid regardless of their location.
        indices = torch.arange(len(gate_targets))
        full_horizon = (
            indices + FUTURE_ONSET_HORIZON_DECISIONS < len(gate_targets)
        )
        future_onset_valid = gate_valid & (
            full_horizon | (gate_targets > 0.0)
        )
        gate_valid = future_onset_valid
    else:
        future_onset_valid = gate_valid.clone()

    positive = gate_valid & (gate_targets > 0.0) & (preferred_actions >= 0)
    action_ids = torch.arange(
        safe_actions.shape[-1],
        dtype=parent_actions.dtype,
    ).unsqueeze(0)
    preferred_correction_required = (
        positive & (preferred_actions != parent_actions)
    )
    if future_onset_gate:
        certified_actions = safe_actions & evaluation_safe_actions
        preferred_equivalent_actions = (
            certified_actions
            & (action_ids != parent_actions.unsqueeze(-1))
            & preferred_correction_required.unsqueeze(-1)
        )
        preferred_action_set = preferred_equivalent_actions.clone()
        no_correction_required = positive & ~preferred_correction_required
        preferred_action_set.scatter_(
            -1,
            parent_actions.unsqueeze(-1),
            no_correction_required.unsqueeze(-1),
        )
    else:
        preferred_action_set = torch.zeros_like(safe_actions)
        preferred_action_set.scatter_(
            -1,
            preferred_actions.clamp_min(0).unsqueeze(-1),
            positive.unsqueeze(-1),
        )
        preferred_equivalent_actions = (
            preferred_action_set
            & (action_ids != parent_actions.unsqueeze(-1))
            & preferred_correction_required.unsqueeze(-1)
        )
    missing_preferred_correction = (
        preferred_correction_required
        & ~preferred_equivalent_actions.any(dim=-1)
    )
    if bool(missing_preferred_correction.any()):
        raise RuntimeError(
            "a preferred correction has no independently certified equivalent action"
        )

    return {
        "parent_actions": parent_actions,
        "previous_actions": previous_actions,
        "gate_targets": gate_targets,
        "gate_valid": gate_valid,
        "hard_positive": hard_positive,
        "correctable_hard_positive": correctable_hard_positive,
        "anticipatory": anticipatory,
        "future_onset_valid": future_onset_valid,
        "anticipatory_lead_decisions": anticipatory_lead_decisions,
        "preferred_actions": preferred_actions,
        "preferred_action_set": preferred_action_set,
        "preferred_equivalent_actions": preferred_equivalent_actions,
        "preferred_correction_required": preferred_correction_required,
        "safety_candidate_actions": safety_candidate_actions,
        "safety_candidate_valid": safety_candidate_valid,
        "safe_actions": safe_actions,
        "evaluation_safe_actions": evaluation_safe_actions,
        "parent_evaluation_danger": (
            parent_collided | (parent_margin < minimum_parent_margin)
        ),
        "collided_actions": collided,
        "minimum_margins": margins,
        "minimum_margin_mask": minimum_margin_mask,
        "teacher_selected_collision": teacher_selected_collision,
    }


def _load_episode(
    model: Any,
    adapter: ResidualCorrectionAdapter,
    dataset: Path,
    report_path: Path,
    manifest: Path,
    *,
    parent_checkpoint_sha256: str,
    device: str,
    chunk_length: int,
    safe_regret: float,
    minimum_parent_margin: float,
    minimum_margin_gain: float,
    predecessor_decisions: int,
    future_onset_gate: bool = False,
) -> EpisodeFeatures:
    report = _read_json(report_path)
    seed = _strict_success(report, dataset, manifest)
    checkpoint_sha = report.get("controller", {}).get("student", {}).get(
        "checkpoint_sha256"
    )
    if checkpoint_sha != parent_checkpoint_sha256:
        raise ValueError("DAgger source was not collected from the requested parent")
    if checkpoint_sha != report["controller"]["student"]["checkpoint_metadata"].get(
        "checkpoint_sha256", checkpoint_sha,
    ):
        raise ValueError("DAgger report contains inconsistent checkpoint hashes")
    demonstrations = Demonstrations.load(dataset)
    demonstrations.validate()
    if len(np.unique(demonstrations.episode_ids)) != 1:
        raise ValueError("each raw DAgger source must contain one episode")
    if demonstrations.teacher_action_evaluations is None:
        raise ValueError("raw DAgger source has no teacher evaluations")
    recurrent, parent_logits, visual_features = _parent_stream(
        model,
        demonstrations,
        _episode_memory(model, report),
        device=device,
        chunk_length=chunk_length,
    )
    labels = _labels_from_evidence(
        demonstrations,
        parent_logits,
        safe_regret=safe_regret,
        minimum_parent_margin=minimum_parent_margin,
        minimum_margin_gain=minimum_margin_gain,
        predecessor_decisions=predecessor_decisions,
        future_onset_gate=future_onset_gate,
    )
    features = adapter.raw_features(
        recurrent,
        parent_logits,
        _executed_action_context(
            demonstrations,
            action_count=adapter.config.action_count,
        ),
        (
            visual_features
            if adapter.config.visual_latent_size else
            None
        ),
        (
            semantic_player_position_features(
                torch.from_numpy(
                    demonstrations.global_frames[:, -1],
                ).float().unsqueeze(0)
            )
            if adapter.config.semantic_player_position else
            None
        ),
    ).detach().cpu()
    return EpisodeFeatures(
        seed=seed,
        dataset=str(dataset),
        report=str(report_path),
        manifest=str(manifest),
        features=features,
        parent_logits=parent_logits,
        global_frame_dtype=str(demonstrations.global_frames.dtype),
        local_frame_dtype=str(demonstrations.local_frames.dtype),
        **labels,
    )


def _fit_source_inventory(
    episodes: list[EpisodeFeatures],
    *,
    calibration_seeds: set[int],
    validation_seeds: set[int],
) -> list[dict[str, Any]]:
    inventory = []
    for episode in episodes:
        role = (
            "validation"
            if episode.seed in validation_seeds else
            "calibration"
            if episode.seed in calibration_seeds else
            "training"
        )
        inventory.append({
            "seed": episode.seed,
            "role": role,
            "dataset": episode.dataset,
            "dataset_sha256": file_sha256(episode.dataset),
            "report": episode.report,
            "report_sha256": file_sha256(episode.report),
            "manifest": episode.manifest,
            "manifest_sha256": file_sha256(episode.manifest),
        })
    return inventory


def _normalize(
    adapter: ResidualCorrectionAdapter,
    fitting_episodes: list[EpisodeFeatures],
    episodes: list[EpisodeFeatures],
) -> None:
    values = torch.cat(
        [episode.features[0] for episode in fitting_episodes],
        dim=0,
    )
    mean = values.mean(dim=0)
    scale = values.std(dim=0).clamp_min(1e-4)
    adapter.set_feature_normalization(mean, scale)
    for episode in episodes:
        episode.features = (
            episode.features - adapter.feature_mean.cpu()
        ) / adapter.feature_scale.cpu()


def _apply_existing_normalization(
    adapter: ResidualCorrectionAdapter,
    episodes: list[EpisodeFeatures],
) -> None:
    """Apply normalization already bound to immutable pretrained weights."""

    for episode in episodes:
        episode.features = (
            episode.features - adapter.feature_mean.cpu()
        ) / adapter.feature_scale.cpu()


def _action_set_loss(
    logits: torch.Tensor,
    safe_actions: torch.Tensor,
    collided_actions: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if not bool(mask.any()):
        return logits.sum() * 0.0
    probabilities = torch.softmax(logits, dim=-1)
    accepted = (probabilities * safe_actions.to(probabilities.dtype)).sum(dim=-1)
    set_loss = -torch.log(accepted.clamp_min(torch.finfo(logits.dtype).tiny))
    negative_infinity = torch.finfo(logits.dtype).min
    best_safe = logits.masked_fill(~safe_actions, negative_infinity).amax(dim=-1)
    best_collision = logits.masked_fill(
        ~collided_actions,
        negative_infinity,
    ).amax(dim=-1)
    has_collision = collided_actions.any(dim=-1)
    rank = F.relu(best_collision - best_safe + 1.0)
    terms = set_loss + torch.where(has_collision, rank, torch.zeros_like(rank))
    return terms[mask].mean()


def _preferred_action_loss(
    logits: torch.Tensor,
    preferred_actions: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    valid = mask & (preferred_actions >= 0)
    if not bool(valid.any()):
        return logits.sum() * 0.0
    return F.cross_entropy(logits[valid], preferred_actions[valid])


def _preferred_action_set_loss(
    logits: torch.Tensor,
    accepted_actions: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if accepted_actions.shape != logits.shape:
        raise ValueError("preferred action sets must align with action logits")
    if accepted_actions.dtype != torch.bool:
        raise ValueError("preferred action sets must be Boolean")
    valid = mask & accepted_actions.any(dim=-1)
    if not bool(valid.any()):
        return logits.sum() * 0.0
    log_probabilities = F.log_softmax(logits[valid], dim=-1)
    accepted_log_probability = torch.logsumexp(
        log_probabilities.masked_fill(
            ~accepted_actions[valid],
            -torch.inf,
        ),
        dim=-1,
    )
    return -accepted_log_probability.mean()


def _preferred_action_membership_loss(
    logits: torch.Tensor,
    accepted_actions: torch.Tensor,
    mask: torch.Tensor,
    *,
    mode: str = "balanced",
) -> torch.Tensor:
    """Multi-label BCE for independently certified actions."""

    if accepted_actions.shape != logits.shape:
        raise ValueError("preferred action sets must align with action logits")
    if accepted_actions.dtype != torch.bool:
        raise ValueError("preferred action sets must be Boolean")
    if mask.shape != logits.shape[:-1] or mask.dtype != torch.bool:
        raise ValueError("preferred action membership mask must be aligned and Boolean")
    if mode not in MEMBERSHIP_LOSS_MODES:
        raise ValueError(
            "membership loss mode must be one of: "
            + ", ".join(MEMBERSHIP_LOSS_MODES)
        )
    positive_count = accepted_actions.sum(dim=-1)
    if mode == "unweighted":
        valid = mask & (positive_count > 0)
        if not bool(valid.any()):
            return logits.sum() * 0.0
        terms = F.binary_cross_entropy_with_logits(
            logits,
            accepted_actions.to(logits.dtype),
            reduction="none",
        )
        return terms.mean(dim=-1)[valid].mean()

    negative_count = (~accepted_actions).sum(dim=-1)
    valid = mask & (positive_count > 0) & (negative_count > 0)
    if not bool(valid.any()):
        return logits.sum() * 0.0
    targets = accepted_actions.to(logits.dtype)
    positive_weight = 0.5 / positive_count.clamp_min(1).to(logits.dtype)
    negative_weight = 0.5 / negative_count.clamp_min(1).to(logits.dtype)
    weights = torch.where(
        accepted_actions,
        positive_weight.unsqueeze(-1),
        negative_weight.unsqueeze(-1),
    )
    terms = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
    )
    return (terms * weights).sum(dim=-1)[valid].mean()


def _preferred_action_uniform_conditional_loss(
    logits: torch.Tensor,
    accepted_actions: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Regularize multi-member certified sets without selecting a unique action."""

    if accepted_actions.shape != logits.shape:
        raise ValueError("preferred action sets must align with action logits")
    if accepted_actions.dtype != torch.bool:
        raise ValueError("preferred action sets must be Boolean")
    if mask.shape != logits.shape[:-1] or mask.dtype != torch.bool:
        raise ValueError("preferred action uniform mask must be aligned and Boolean")
    cardinality = accepted_actions.sum(dim=-1)
    eligible = mask & (cardinality > 1)
    if not bool(eligible.any()):
        return logits.sum() * 0.0
    eligible_sets = accepted_actions[eligible]
    restricted_log_probabilities = F.log_softmax(
        logits[eligible].masked_fill(~eligible_sets, -torch.inf),
        dim=-1,
    )
    member_log_probabilities = restricted_log_probabilities.masked_fill(
        ~eligible_sets,
        0.0,
    )
    eligible_cardinality = cardinality[eligible].to(logits.dtype)
    uniform_cross_entropy = (
        -member_log_probabilities.sum(dim=-1) / eligible_cardinality
    )
    return (uniform_cross_entropy - eligible_cardinality.log()).mean()


def _preferred_action_conditional_tiebreak_loss(
    logits: torch.Tensor,
    accepted_actions: torch.Tensor,
    preferred_actions: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Break multi-action ties without changing rejected-action pressure."""

    if accepted_actions.shape != logits.shape:
        raise ValueError("preferred action sets must align with action logits")
    if accepted_actions.dtype != torch.bool:
        raise ValueError("preferred action sets must be Boolean")
    if preferred_actions.shape != logits.shape[:-1]:
        raise ValueError("preferred actions must align with action logits")
    if mask.shape != logits.shape[:-1] or mask.dtype != torch.bool:
        raise ValueError("preferred action tiebreak mask must be aligned and Boolean")
    eligible = mask & (accepted_actions.sum(dim=-1) > 1)
    if not bool(eligible.any()):
        return logits.sum() * 0.0
    eligible_targets = preferred_actions[eligible]
    action_count = logits.shape[-1]
    in_range = (eligible_targets >= 0) & (eligible_targets < action_count)
    if not bool(in_range.all()):
        raise ValueError("preferred tiebreak target is outside the action vocabulary")
    eligible_sets = accepted_actions[eligible]
    target_accepted = eligible_sets.gather(
        -1,
        eligible_targets.unsqueeze(-1),
    ).squeeze(-1)
    if not bool(target_accepted.all()):
        raise ValueError("preferred tiebreak target is outside its certified set")
    restricted_logits = logits[eligible].masked_fill(~eligible_sets, -torch.inf)
    conditional_log_probabilities = F.log_softmax(restricted_logits, dim=-1)
    return -conditional_log_probabilities.gather(
        -1,
        eligible_targets.unsqueeze(-1),
    ).mean()


def _preferred_action_tiebreak_mask(
    accepted_actions: torch.Tensor,
    previous_actions: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Select rows with a row-local, certified previous-action target."""

    if accepted_actions.dtype != torch.bool:
        raise ValueError("preferred action sets must be Boolean")
    if previous_actions.shape != accepted_actions.shape[:-1]:
        raise ValueError("previous actions must align with preferred action sets")
    if mask.shape != previous_actions.shape or mask.dtype != torch.bool:
        raise ValueError("preferred action tiebreak mask must be aligned and Boolean")
    action_count = accepted_actions.shape[-1]
    in_range = (previous_actions >= 0) & (previous_actions < action_count)
    previous_accepted = accepted_actions.gather(
        -1,
        previous_actions.clamp(0, action_count - 1).unsqueeze(-1),
    ).squeeze(-1)
    return (
        mask
        & (accepted_actions.sum(dim=-1) > 1)
        & in_range
        & previous_accepted
    )


def _preferred_action_set_rank_loss(
    logits: torch.Tensor,
    accepted_actions: torch.Tensor,
    mask: torch.Tensor,
    *,
    margin: float = 1.0,
) -> torch.Tensor:
    """Require the best accepted action to outrank every rejected action."""

    if accepted_actions.shape != logits.shape:
        raise ValueError("preferred action sets must align with action logits")
    if accepted_actions.dtype != torch.bool:
        raise ValueError("preferred action sets must be Boolean")
    if not math.isfinite(margin) or margin < 0.0:
        raise ValueError("preferred action rank margin must be finite and nonnegative")
    valid = (
        mask
        & accepted_actions.any(dim=-1)
        & (~accepted_actions).any(dim=-1)
    )
    if not bool(valid.any()):
        return logits.sum() * 0.0
    valid_logits = logits[valid]
    valid_accepted = accepted_actions[valid]
    negative_infinity = torch.finfo(logits.dtype).min
    best_accepted = valid_logits.masked_fill(
        ~valid_accepted,
        negative_infinity,
    ).amax(dim=-1)
    best_rejected = valid_logits.masked_fill(
        valid_accepted,
        negative_infinity,
    ).amax(dim=-1)
    return F.relu(best_rejected - best_accepted + margin).mean()


def _dense_safety_losses(
    collision_logits: torch.Tensor,
    normalized_margin_predictions: torch.Tensor,
    collided_actions: torch.Tensor,
    minimum_margins: torch.Tensor,
    minimum_margin_mask: torch.Tensor,
    *,
    collision_positive_weights: torch.Tensor | None = None,
    all_collision_row_weight: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Supervise all candidate actions without letting all-collision rows dominate."""

    collision_targets = collided_actions.to(collision_logits.dtype)
    differentiable = ~collided_actions.all(dim=-1, keepdim=True)
    if collision_positive_weights is None:
        collision_positive_weights = torch.ones(
            collision_logits.shape[-1],
            dtype=collision_logits.dtype,
            device=collision_logits.device,
        )
    else:
        collision_positive_weights = collision_positive_weights.to(
            collision_logits,
        )
    positive_weights = collision_positive_weights.unsqueeze(0).expand_as(
        collision_logits,
    )
    collision_weights = torch.where(
        collided_actions & differentiable,
        positive_weights,
        torch.ones_like(collision_logits),
    )
    collision_weights = collision_weights * torch.where(
        differentiable,
        torch.ones_like(collision_weights),
        torch.full_like(collision_weights, all_collision_row_weight),
    )
    collision_terms = F.binary_cross_entropy_with_logits(
        collision_logits,
        collision_targets,
        reduction="none",
    )
    collision_loss = (
        (collision_terms * collision_weights).sum()
        / collision_weights.sum().clamp_min(1.0)
    )

    normalized_margin_targets = torch.where(
        minimum_margin_mask,
        minimum_margins.clamp(-64.0, 64.0) / 16.0,
        torch.zeros_like(minimum_margins),
    )
    margin_terms = F.smooth_l1_loss(
        normalized_margin_predictions,
        normalized_margin_targets,
        reduction="none",
    )
    margin_loss = (
        margin_terms[minimum_margin_mask].mean()
        if bool(minimum_margin_mask.any()) else
        normalized_margin_predictions.sum() * 0.0
    )
    return collision_loss, margin_loss


def _collision_positive_weights(
    episodes: list[EpisodeFeatures],
    *,
    maximum_weight: float,
) -> torch.Tensor:
    """Balance distinguishable collision labels independently per action."""

    collided = torch.cat([episode.collided_actions for episode in episodes], dim=0)
    distinguishable = ~collided.all(dim=-1, keepdim=True)
    positives = (collided & distinguishable).sum(dim=0).to(torch.float32)
    negatives = ((~collided) & distinguishable).sum(dim=0).to(torch.float32)
    weights = negatives / positives.clamp_min(1.0)
    return weights.clamp(1.0, maximum_weight)


def _physical_danger_positive_weights(
    episodes: list[EpisodeFeatures],
    *,
    maximum_weight: float,
) -> torch.Tensor:
    """Balance the direct clearance<8/collision target per action."""

    danger = torch.cat(
        [~episode.evaluation_safe_actions for episode in episodes],
        dim=0,
    )
    positives = danger.sum(dim=0).to(torch.float32)
    negatives = (~danger).sum(dim=0).to(torch.float32)
    return (negatives / positives.clamp_min(1.0)).clamp(1.0, maximum_weight)


def _physical_danger_loss(
    logits: torch.Tensor,
    evaluation_safe_actions: torch.Tensor,
    *,
    positive_weights: torch.Tensor,
) -> torch.Tensor:
    target = (~evaluation_safe_actions).to(logits.dtype)
    return F.binary_cross_entropy_with_logits(
        logits,
        target,
        pos_weight=positive_weights.to(logits),
    )


def _parent_copy_loss(
    action_logits: torch.Tensor,
    parent_logits: torch.Tensor,
    mask: torch.Tensor,
    *,
    action_logit_mode: str,
) -> torch.Tensor:
    """Keep residual negatives close to the parent's complete distribution."""

    if action_logit_mode == "certified_membership":
        return action_logits.sum() * 0.0
    if not bool(mask.any()):
        return action_logits.sum() * 0.0
    if action_logit_mode == "absolute":
        parent_actions = parent_logits.argmax(dim=-1)
        return F.cross_entropy(action_logits[mask], parent_actions[mask])
    parent_probabilities = torch.softmax(parent_logits[mask].detach(), dim=-1)
    return F.kl_div(
        torch.log_softmax(action_logits[mask], dim=-1),
        parent_probabilities,
        reduction="batchmean",
    )


def _validate_action_training_semantics(
    *,
    action_logit_mode: str,
    action_loss_weight: float,
    parent_copy_weight: float,
    preferred_action_uniform_loss_weight: float,
    preferred_action_tiebreak_loss_weight: float,
    preferred_action_rank_loss_weight: float,
    safety_candidate_loss_weight: float,
    membership_loss_mode: str = "balanced",
) -> None:
    if membership_loss_mode not in MEMBERSHIP_LOSS_MODES:
        raise ValueError(
            "membership loss mode must be one of: "
            + ", ".join(MEMBERSHIP_LOSS_MODES)
        )
    if (
        action_logit_mode != "certified_membership"
        and membership_loss_mode != "balanced"
    ):
        raise ValueError(
            "unweighted membership loss is only applicable to "
            "certified_membership action logits"
        )
    if action_logit_mode != "certified_membership":
        return
    incompatible = {
        "safe-action softmax": action_loss_weight,
        "parent copy": parent_copy_weight,
        "preferred-action uniform": preferred_action_uniform_loss_weight,
        "preferred-action tiebreak": preferred_action_tiebreak_loss_weight,
        "preferred-action rank": preferred_action_rank_loss_weight,
        "single safety-candidate cross entropy": safety_candidate_loss_weight,
    }
    enabled = [name for name, value in incompatible.items() if value != 0.0]
    if enabled:
        raise ValueError(
            "certified_membership requires zero inapplicable loss weights: "
            + ", ".join(enabled)
        )


def _preferred_action_loss_semantics(
    *,
    action_logit_mode: str,
    membership_loss_mode: str,
) -> str:
    _validate_action_training_semantics(
        action_logit_mode=action_logit_mode,
        action_loss_weight=0.0,
        parent_copy_weight=0.0,
        preferred_action_uniform_loss_weight=0.0,
        preferred_action_tiebreak_loss_weight=0.0,
        preferred_action_rank_loss_weight=0.0,
        safety_candidate_loss_weight=0.0,
        membership_loss_mode=membership_loss_mode,
    )
    if action_logit_mode == "certified_membership":
        return MEMBERSHIP_LOSS_SEMANTICS[membership_loss_mode]
    return "negative log softmax mass on the certified action set"


def _membership_loss_metadata(
    *,
    action_logit_mode: str,
    membership_loss_mode: str,
) -> dict[str, str | None]:
    return {
        "membership_loss_mode": (
            membership_loss_mode
            if action_logit_mode == "certified_membership" else
            None
        ),
        "preferred_action_loss_semantics": _preferred_action_loss_semantics(
            action_logit_mode=action_logit_mode,
            membership_loss_mode=membership_loss_mode,
        ),
    }


def _validate_restored_membership_loss_mode(
    training_metadata: Mapping[str, Any],
    *,
    action_logit_mode: str,
    requested_mode: str,
) -> str | None:
    """Bind recalibration to the loss semantics that produced frozen weights."""

    _preferred_action_loss_semantics(
        action_logit_mode=action_logit_mode,
        membership_loss_mode=requested_mode,
    )
    if action_logit_mode != "certified_membership":
        return None

    current = training_metadata
    recorded_modes: list[str] = []
    explicit_mode_count = 0
    visited: set[int] = set()
    for _ in range(8):
        identity = id(current)
        if identity in visited:
            raise ValueError("restored adapter weight source metadata contains a cycle")
        visited.add(identity)

        controls = current.get("training_controls")
        if "training_controls" in current and not isinstance(controls, Mapping):
            raise TypeError("restored adapter training controls must be a mapping")
        explicit_mode = False
        if isinstance(controls, Mapping) and "membership_loss_mode" in controls:
            recorded = controls["membership_loss_mode"]
            if recorded not in MEMBERSHIP_LOSS_MODES:
                raise ValueError(
                    "restored adapter has invalid membership loss mode metadata"
                )
            recorded_modes.append(str(recorded))
            explicit_mode_count += 1
            explicit_mode = True

        nested: list[Mapping[str, Any]] = []
        for field in (
            "fit_checkpoint_weight_source",
            "frozen_adapter_weight_source",
        ):
            if field not in current:
                continue
            source = current[field]
            if not isinstance(source, Mapping):
                raise TypeError(
                    f"restored adapter {field} metadata must be a mapping"
                )
            source_training = source.get("training_metadata")
            if not isinstance(source_training, Mapping):
                raise TypeError(
                    "restored adapter weight source has invalid training metadata"
                )
            nested.append(source_training)
        if len(nested) > 1:
            raise ValueError("restored adapter has ambiguous weight source metadata")
        if not nested:
            if not explicit_mode:
                # Every membership artifact predating this field used balanced BCE.
                recorded_modes.append("balanced")
            break
        current = nested[0]
    else:
        raise ValueError("restored adapter weight source metadata is too deeply nested")

    distinct_modes = set(recorded_modes)
    if len(distinct_modes) != 1:
        raise ValueError(
            "restored adapter has inconsistent membership loss mode provenance: "
            + ", ".join(sorted(distinct_modes))
        )
    recorded_mode = recorded_modes[0]
    if requested_mode != recorded_mode:
        if explicit_mode_count == 0:
            raise ValueError(
                "--membership-loss-mode must match the restored legacy "
                "adapter weight value 'balanced'"
            )
        raise ValueError(
            "--membership-loss-mode must match the restored adapter "
            f"weight value {recorded_mode!r}"
        )
    return recorded_mode


def _membership_confidence_loss(
    logits: torch.Tensor,
    accepted_actions: torch.Tensor,
    mask: torch.Tensor,
    *,
    mode: str = "unweighted",
) -> torch.Tensor:
    """Independent certified-membership BCE for the detached confidence head."""

    if accepted_actions.shape != logits.shape:
        raise ValueError("membership confidence targets must align with logits")
    if accepted_actions.dtype != torch.bool:
        raise ValueError("membership confidence targets must be Boolean")
    if mask.shape != logits.shape[:-1] or mask.dtype != torch.bool:
        raise ValueError("membership confidence mask must be aligned and Boolean")
    if logits.shape[-1] != 18:
        raise ValueError("membership confidence requires all 18 action cells")
    if not torch.is_floating_point(logits):
        raise ValueError("membership confidence logits must be floating point")
    if mode not in MEMBERSHIP_CONFIDENCE_LOSS_MODES:
        raise ValueError(
            "membership confidence loss mode must be one of: "
            + ", ".join(MEMBERSHIP_CONFIDENCE_LOSS_MODES)
        )

    positive_count = accepted_actions.sum(dim=-1)
    valid = mask & (positive_count > 0)
    if not bool(valid.any()):
        finite_logits = torch.where(
            torch.isfinite(logits),
            logits,
            torch.zeros_like(logits),
        )
        return finite_logits.sum() * 0.0
    valid_logits = logits[valid]
    if not bool(torch.isfinite(valid_logits).all()):
        raise ValueError(
            "membership confidence logits must be finite on labelled rows"
        )
    valid_targets = accepted_actions[valid]
    terms = F.binary_cross_entropy_with_logits(
        valid_logits,
        valid_targets.to(valid_logits.dtype),
        reduction="none",
    )
    if mode == "unweighted":
        result = terms.mean(dim=-1).mean()
    else:
        valid_positive_count = positive_count[valid]
        valid_negative_count = (~valid_targets).sum(dim=-1)
        positive_total_weight = torch.where(
            valid_negative_count > 0,
            torch.full_like(valid_positive_count, 0.5, dtype=valid_logits.dtype),
            torch.ones_like(valid_positive_count, dtype=valid_logits.dtype),
        )
        positive_weight = (
            positive_total_weight
            / valid_positive_count.to(valid_logits.dtype)
        )
        negative_weight = torch.where(
            valid_negative_count > 0,
            0.5 / valid_negative_count.clamp_min(1).to(valid_logits.dtype),
            torch.zeros_like(valid_negative_count, dtype=valid_logits.dtype),
        )
        weights = torch.where(
            valid_targets,
            positive_weight.unsqueeze(-1),
            negative_weight.unsqueeze(-1),
        )
        result = (terms * weights).sum(dim=-1).mean()
    if not bool(torch.isfinite(result)):
        raise ValueError("membership confidence loss must be finite")
    return result


def _validate_membership_confidence_training_semantics(
    *,
    enabled: bool,
    action_logit_mode: str,
    loss_weight: float,
    loss_mode: str,
) -> None:
    if not isinstance(enabled, bool):
        raise ValueError("per-action membership confidence must be a Boolean")
    if loss_mode not in MEMBERSHIP_CONFIDENCE_LOSS_MODES:
        raise ValueError(
            "membership confidence loss mode must be one of: "
            + ", ".join(MEMBERSHIP_CONFIDENCE_LOSS_MODES)
        )
    if (
        isinstance(loss_weight, bool)
        or not isinstance(loss_weight, (int, float))
        or not math.isfinite(float(loss_weight))
        or float(loss_weight) < 0.0
    ):
        raise ValueError(
            "membership confidence loss weight must be finite and nonnegative"
        )
    if not enabled:
        if float(loss_weight) != 0.0:
            raise ValueError(
                "membership confidence loss weight must be zero when the head "
                "is disabled"
            )
        return
    if action_logit_mode == "certified_membership":
        raise ValueError(
            "per-action membership confidence requires a selector action head, "
            "not certified_membership action logits"
        )
    if float(loss_weight) <= 0.0:
        raise ValueError(
            "membership confidence loss weight must be positive when the head "
            "is enabled"
        )


def _membership_confidence_training_metadata(
    *,
    enabled: bool,
    action_logit_mode: str,
    loss_weight: float,
    loss_mode: str,
) -> dict[str, Any]:
    _validate_membership_confidence_training_semantics(
        enabled=enabled,
        action_logit_mode=action_logit_mode,
        loss_weight=loss_weight,
        loss_mode=loss_mode,
    )
    if not enabled:
        return {}
    return {
        "per_action_membership_confidence": True,
        "membership_confidence_loss_weight": float(loss_weight),
        "membership_confidence_loss_mode": loss_mode,
        "membership_confidence_loss_semantics": MEMBERSHIP_LOSS_SEMANTICS[
            loss_mode
        ],
        "membership_confidence_training_semantics": dict(
            MEMBERSHIP_CONFIDENCE_TRAINING_SEMANTICS
        ),
    }


def _validate_restored_membership_confidence_training(
    training_metadata: Mapping[str, Any],
    *,
    enabled: bool,
    action_logit_mode: str,
    requested_weight: float,
    requested_mode: str,
) -> dict[str, Any] | None:
    """Bind frozen auxiliary-head weights to their exact training semantics."""

    requested = _membership_confidence_training_metadata(
        enabled=enabled,
        action_logit_mode=action_logit_mode,
        loss_weight=requested_weight,
        loss_mode=requested_mode,
    )
    current = training_metadata
    recorded: list[dict[str, Any] | None] = []
    visited: set[int] = set()
    fields = {
        "per_action_membership_confidence",
        "membership_confidence_loss_weight",
        "membership_confidence_loss_mode",
        "membership_confidence_loss_semantics",
        "membership_confidence_training_semantics",
    }
    for _ in range(8):
        identity = id(current)
        if identity in visited:
            raise ValueError(
                "restored membership confidence provenance contains a cycle"
            )
        visited.add(identity)

        controls = current.get("training_controls")
        if "training_controls" in current and not isinstance(controls, Mapping):
            raise TypeError("restored adapter training controls must be a mapping")
        explicit = False
        if isinstance(controls, Mapping) and fields & controls.keys():
            explicit = True
            missing = fields - controls.keys()
            if missing:
                raise ValueError(
                    "restored adapter has incomplete membership confidence "
                    "training provenance: " + ", ".join(sorted(missing))
                )
            recorded_enabled = controls["per_action_membership_confidence"]
            recorded_weight = controls["membership_confidence_loss_weight"]
            recorded_mode = controls["membership_confidence_loss_mode"]
            if recorded_enabled is not True:
                raise ValueError(
                    "restored adapter has invalid membership confidence enabled "
                    "metadata"
                )
            _validate_membership_confidence_training_semantics(
                enabled=True,
                action_logit_mode=action_logit_mode,
                loss_weight=recorded_weight,
                loss_mode=recorded_mode,
            )
            expected_semantics = _membership_confidence_training_metadata(
                enabled=True,
                action_logit_mode=action_logit_mode,
                loss_weight=float(recorded_weight),
                loss_mode=str(recorded_mode),
            )
            actual = {name: controls[name] for name in fields}
            if actual != expected_semantics:
                raise ValueError(
                    "restored adapter has invalid membership confidence training "
                    "semantics"
                )
            recorded.append(expected_semantics)

        nested: list[Mapping[str, Any]] = []
        for field in (
            "fit_checkpoint_weight_source",
            "frozen_adapter_weight_source",
        ):
            if field not in current:
                continue
            source = current[field]
            if not isinstance(source, Mapping):
                raise TypeError(
                    f"restored adapter {field} metadata must be a mapping"
                )
            source_training = source.get("training_metadata")
            if not isinstance(source_training, Mapping):
                raise TypeError(
                    "restored adapter weight source has invalid training metadata"
                )
            nested.append(source_training)
        if len(nested) > 1:
            raise ValueError("restored adapter has ambiguous weight source metadata")
        if not nested:
            if not explicit:
                recorded.append(None)
            break
        current = nested[0]
    else:
        raise ValueError(
            "restored membership confidence weight source metadata is too deeply "
            "nested"
        )

    normalized = {
        json.dumps(value, sort_keys=True, allow_nan=False)
        if value is not None else
        "null"
        for value in recorded
    }
    if len(normalized) != 1:
        raise ValueError(
            "restored adapter has inconsistent membership confidence provenance"
        )
    restored = recorded[0]
    if restored != (requested or None):
        if restored is None:
            raise ValueError(
                "restored legacy adapter has no per-action membership confidence "
                "head training provenance"
            )
        raise ValueError(
            "membership confidence loss mode and weight must match the restored "
            "adapter weights"
        )
    return restored


def _validate_training_loss_weights(**weights: float) -> None:
    invalid = [
        name
        for name, value in weights.items()
        if not math.isfinite(value) or value < 0.0
    ]
    if invalid:
        raise ValueError(
            "training loss weights must be finite and nonnegative: "
            + ", ".join(name.replace("_", " ") for name in invalid)
        )


def _training_optimizer_parameter_groups(
    member: torch.nn.Module,
    *,
    per_action_membership_confidence: bool,
) -> dict[str, tuple[torch.nn.Parameter, ...]]:
    """Partition parameters without changing the base optimizer's order."""

    all_parameters = tuple(
        parameter for parameter in member.parameters() if parameter.requires_grad
    )
    all_ids = {id(parameter) for parameter in all_parameters}
    if len(all_ids) != len(all_parameters):
        raise RuntimeError("trainable member parameters contain duplicate identities")
    if not per_action_membership_confidence:
        return {"base": all_parameters, "membership_confidence": ()}

    membership_head = getattr(member, "membership_head", None)
    if not isinstance(membership_head, torch.nn.Module):
        raise RuntimeError(
            "per-action membership confidence member is missing membership_head"
        )
    membership_parameters = tuple(
        parameter
        for parameter in membership_head.parameters()
        if parameter.requires_grad
    )
    membership_ids = {id(parameter) for parameter in membership_parameters}
    if not membership_parameters:
        raise RuntimeError("membership confidence head has no trainable parameters")
    if len(membership_ids) != len(membership_parameters):
        raise RuntimeError(
            "membership confidence parameters contain duplicate identities"
        )
    if not membership_ids <= all_ids:
        raise RuntimeError(
            "membership confidence parameters are absent from the member"
        )

    parameter_paths: dict[int, list[str]] = {}
    for name, parameter in member.named_parameters(remove_duplicate=False):
        if parameter.requires_grad:
            parameter_paths.setdefault(id(parameter), []).append(name)
    for parameter_id in membership_ids:
        outside = [
            path
            for path in parameter_paths.get(parameter_id, [])
            if path.split(".", 1)[0] != "membership_head"
        ]
        if outside:
            raise RuntimeError(
                "membership confidence and base parameter groups overlap at "
                f"{outside[0]!r}"
            )

    base_parameters = tuple(
        parameter
        for parameter in all_parameters
        if id(parameter) not in membership_ids
    )
    if not base_parameters:
        raise RuntimeError("base optimizer parameter group is empty")
    if len(base_parameters) + len(membership_parameters) != len(all_parameters):
        raise RuntimeError("optimizer parameter groups omit or duplicate parameters")
    return {
        "base": base_parameters,
        "membership_confidence": membership_parameters,
    }


def _gradient_clip_parameter_groups(
    member: torch.nn.Module,
    *,
    excluded_parameters: tuple[torch.nn.Parameter, ...] = (),
) -> dict[str, tuple[torch.nn.Parameter, ...]]:
    """Return complete, disjoint trainable groups for gradient clipping."""

    if not hasattr(member, "action_recurrent"):
        raise RuntimeError("residual member is missing action_recurrent")
    excluded_ids = {id(parameter) for parameter in excluded_parameters}
    if len(excluded_ids) != len(excluded_parameters):
        raise RuntimeError("excluded gradient parameters contain duplicate identities")
    member_trainable = tuple(
        parameter for parameter in member.parameters() if parameter.requires_grad
    )
    member_trainable_ids = {id(parameter) for parameter in member_trainable}
    if not excluded_ids <= member_trainable_ids:
        raise RuntimeError("excluded gradient parameters are absent from member")
    trainable = tuple(
        parameter
        for parameter in member_trainable
        if id(parameter) not in excluded_ids
    )
    trainable_ids = {id(parameter) for parameter in trainable}
    if len(trainable_ids) != len(trainable):
        raise RuntimeError("trainable member parameters contain duplicate identities")
    if member.action_recurrent is None:
        return {"global": trainable}

    action_modules: dict[str, torch.nn.Module] = {}
    for name in ACTION_BRANCH_MODULE_NAMES:
        module = getattr(member, name, None)
        if module is None or not isinstance(module, torch.nn.Module):
            raise RuntimeError(
                f"separate action recurrent member is missing module {name}"
            )
        action_modules[name] = module

    action_ids: set[int] = set()
    action_parameter_names: dict[int, str] = {}
    for module_name, module in action_modules.items():
        for parameter_name, parameter in module.named_parameters():
            if not parameter.requires_grad:
                continue
            parameter_id = id(parameter)
            qualified_name = f"{module_name}.{parameter_name}"
            if parameter_id in action_ids:
                previous = action_parameter_names[parameter_id]
                raise RuntimeError(
                    "separate action gradient group contains overlapping parameter "
                    f"{previous!r} / {qualified_name!r}"
                )
            action_ids.add(parameter_id)
            action_parameter_names[parameter_id] = qualified_name

    if not action_ids:
        raise RuntimeError("separate action gradient group has no trainable parameters")
    missing_from_member = action_ids - trainable_ids
    if missing_from_member:
        raise RuntimeError(
            "separate action gradient group contains parameters absent from member"
        )

    # Detect weight aliases between the named action modules and any other
    # member path. named_parameters() normally de-duplicates these aliases.
    parameter_paths: dict[int, list[str]] = {}
    for name, parameter in member.named_parameters(remove_duplicate=False):
        if parameter.requires_grad:
            parameter_paths.setdefault(id(parameter), []).append(name)
    for parameter_id in action_ids:
        paths = parameter_paths.get(parameter_id, [])
        outside = [
            path
            for path in paths
            if path.split(".", 1)[0] not in ACTION_BRANCH_MODULE_NAMES
        ]
        if outside:
            raise RuntimeError(
                "action and shared/safety gradient groups overlap at "
                f"{action_parameter_names[parameter_id]!r} / {outside[0]!r}"
            )

    action = tuple(parameter for parameter in trainable if id(parameter) in action_ids)
    shared_safety = tuple(
        parameter for parameter in trainable if id(parameter) not in action_ids
    )
    action_result_ids = {id(parameter) for parameter in action}
    shared_result_ids = {id(parameter) for parameter in shared_safety}
    if action_result_ids & shared_result_ids:
        raise RuntimeError("action and shared/safety gradient groups overlap")
    if action_result_ids | shared_result_ids != trainable_ids:
        raise RuntimeError("gradient clip parameter groups omit trainable parameters")
    if len(action) + len(shared_safety) != len(trainable):
        raise RuntimeError("gradient clip parameter groups duplicate trainable parameters")
    return {"shared_safety": shared_safety, "action": action}


def _clip_member_gradients(
    member: torch.nn.Module,
    *,
    max_norm: float = GRADIENT_CLIP_MAX_NORM,
    excluded_parameters: tuple[torch.nn.Parameter, ...] = (),
) -> dict[str, torch.Tensor]:
    """Clip independent action/shared groups when the architecture separates them."""

    if not math.isfinite(max_norm) or max_norm <= 0.0:
        raise ValueError("gradient clip max norm must be finite and positive")
    groups = _gradient_clip_parameter_groups(
        member,
        excluded_parameters=excluded_parameters,
    )
    return {
        name: torch.nn.utils.clip_grad_norm_(parameters, max_norm)
        for name, parameters in groups.items()
    }


def _clip_membership_confidence_gradients(
    parameters: tuple[torch.nn.Parameter, ...],
    *,
    max_norm: float = GRADIENT_CLIP_MAX_NORM,
) -> torch.Tensor:
    if not parameters:
        raise RuntimeError("membership confidence gradient group is empty")
    if not math.isfinite(max_norm) or max_norm <= 0.0:
        raise ValueError("gradient clip max norm must be finite and positive")
    return torch.nn.utils.clip_grad_norm_(parameters, max_norm)


def _early_selected_candidate_confidence_mask(
    episode: EpisodeFeatures,
) -> torch.Tensor:
    """Rows consumed by fail-closed future-onset calibration."""

    mask = (
        episode.gate_valid
        & (episode.gate_targets > 0.0)
        & episode.anticipatory
        & (
            episode.anticipatory_lead_decisions
            >= EARLY_ONSET_MINIMUM_LEAD_DECISIONS
        )
        & (
            episode.anticipatory_lead_decisions
            <= FUTURE_ONSET_HORIZON_DECISIONS
        )
    )
    if mask.shape != episode.parent_actions.shape or mask.dtype != torch.bool:
        raise ValueError("early selected-candidate mask is invalid")
    return mask


def _selected_candidate_confidence_targets(
    episode: EpisodeFeatures,
    candidates: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return early-row mask and frozen-selector beneficial-candidate labels."""

    if candidates.shape != episode.parent_actions.shape:
        raise ValueError("selected confidence candidates do not align with episode")
    if candidates.dtype != torch.int64:
        raise ValueError("selected confidence candidates must be int64")
    if bool(((candidates < 0) | (candidates >= 18)).any()):
        raise ValueError("selected confidence candidate is outside the vocabulary")
    equivalent = episode.preferred_equivalent_actions
    if equivalent.shape != (*candidates.shape, 18) or equivalent.dtype != torch.bool:
        raise ValueError("preferred equivalent actions do not align with candidates")
    required = episode.preferred_correction_required
    if required.shape != candidates.shape or required.dtype != torch.bool:
        raise ValueError("preferred correction-required rows do not align")
    parent_actions = episode.parent_actions
    if parent_actions.shape != candidates.shape or parent_actions.dtype != torch.int64:
        raise ValueError("parent actions do not align with selected candidates")
    if bool(((parent_actions < 0) | (parent_actions >= 18)).any()):
        raise ValueError("parent action is outside the vocabulary")
    has_equivalent = equivalent.any(dim=-1)
    if bool((has_equivalent & ~required).any()):
        raise ValueError(
            "a no-correction row cannot contain preferred equivalent actions"
        )
    if bool((required & ~has_equivalent).any()):
        raise ValueError(
            "a correction-required row must contain an equivalent action"
        )
    parent_is_equivalent = equivalent.gather(
        -1,
        parent_actions.unsqueeze(-1),
    ).squeeze(-1)
    if bool(parent_is_equivalent.any()):
        raise ValueError("the parent action cannot be an equivalent correction")
    mask = _early_selected_candidate_confidence_mask(episode)
    targets = (
        equivalent.gather(-1, candidates.unsqueeze(-1)).squeeze(-1)
        & required
    )
    return mask, targets


def _frozen_selector_episode(
    adapter: ResidualCorrectionAdapter,
    episode: EpisodeFeatures,
    *,
    chunk_length: int,
) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
    """Evaluate the frozen action branches on canonical CPU float32 tensors."""

    if chunk_length <= 0:
        raise ValueError("frozen selector chunk length must be positive")
    if any(parameter.device.type != "cpu" for parameter in adapter.parameters()):
        raise ValueError("frozen selected-candidate training requires a CPU adapter")
    if episode.features.device.type != "cpu" or episode.parent_logits.device.type != "cpu":
        raise ValueError("frozen selected-candidate inputs must be on CPU")
    if episode.features.dtype != torch.float32 or episode.parent_logits.dtype != torch.float32:
        raise ValueError("frozen selected-candidate inputs must be float32")

    action_hidden: list[torch.Tensor | None] = [None] * len(adapter.members)
    candidate_parts: list[torch.Tensor] = []
    finite_parts: list[torch.Tensor] = []
    latent_parts: list[list[torch.Tensor]] = [
        [] for _member in adapter.members
    ]
    with torch.no_grad():
        for start in range(0, episode.decisions, chunk_length):
            stop = min(start + chunk_length, episode.decisions)
            features = episode.features[:, start:stop]
            parent_logits = episode.parent_logits[0, start:stop]
            member_logits: list[torch.Tensor] = []
            for index, member in enumerate(adapter.members):
                projection = member.action_input_projection
                recurrent = member.action_recurrent
                if projection is None or recurrent is None:
                    raise RuntimeError(
                        "selected confidence requires a separate action recurrent"
                    )
                encoded = projection(features)
                latent, next_hidden = recurrent(encoded, action_hidden[index])
                action_hidden[index] = next_hidden.detach()
                latent = latent[0].detach()
                latent_parts[index].append(latent.clone())
                member_logits.append(adapter.decode_action_logits(
                    member.action_head(latent),
                    parent_logits,
                ))
            decoded = torch.stack(member_logits, dim=0)
            probabilities, member_finite = finite_action_probabilities(
                decoded,
                adapter.config.action_logit_mode,
            )
            selector = ensemble_action_summary(probabilities, member_finite)
            candidate_parts.append(selector["candidates"].detach().cpu())
            finite_parts.append(
                selector["action_all_members_finite"].detach().cpu()
            )

    candidates = torch.cat(candidate_parts, dim=0)
    finite = torch.cat(finite_parts, dim=0)
    latents = tuple(torch.cat(parts, dim=0) for parts in latent_parts)
    expected = (episode.decisions,)
    if candidates.shape != expected or finite.shape != expected:
        raise RuntimeError("frozen selector output does not align with episode")
    if any(value.shape[0] != episode.decisions for value in latents):
        raise RuntimeError("frozen selector latent does not align with episode")
    return candidates, finite, latents


def _frozen_ensemble_selected_candidates(
    adapter: ResidualCorrectionAdapter,
    episodes: list[EpisodeFeatures],
    *,
    chunk_length: int,
) -> dict[int, torch.Tensor]:
    """Freeze the runtime global candidate before confidence-head fitting."""

    if not episodes:
        raise ValueError("frozen candidate extraction requires fitting episodes")
    if adapter.config.action_logit_mode != "parent_residual_joint":
        raise ValueError("frozen candidate extraction requires the plain selector")
    if any(parameter.device.type != "cpu" for parameter in adapter.parameters()):
        raise ValueError("frozen candidate extraction requires a CPU adapter")
    module_training_state = [
        (module, module.training) for module in adapter.modules()
    ]
    adapter.eval()
    result: dict[int, torch.Tensor] = {}
    try:
        for episode in episodes:
            if episode.seed in result:
                raise ValueError("frozen candidate episodes contain duplicate seeds")
            candidates, finite, _latents = _frozen_selector_episode(
                adapter,
                episode,
                chunk_length=chunk_length,
            )
            labelled = _early_selected_candidate_confidence_mask(episode)
            if bool(labelled.any()) and not bool(finite[labelled].all()):
                raise ValueError(
                    "frozen ensemble selector is nonfinite on labelled rows"
                )
            result[episode.seed] = candidates.clone()
    finally:
        for module, training in module_training_state:
            module.train(training)
    return result


def _train_frozen_ensemble_selected_confidence_heads(
    adapter: ResidualCorrectionAdapter,
    episodes: list[EpisodeFeatures],
    *,
    frozen_candidates: Mapping[int, torch.Tensor],
    seed: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    chunk_length: int,
    loss_weight: float,
    device: str,
    max_norm: float = GRADIENT_CLIP_MAX_NORM,
) -> list[dict[str, float]]:
    """Train only confidence heads against immutable ensemble-global candidates."""

    if not adapter.config.per_action_membership_confidence:
        raise ValueError("selected confidence training requires membership heads")
    if adapter.config.action_logit_mode != "parent_residual_joint":
        raise ValueError("selected confidence training requires the plain selector")
    if not adapter.config.separate_action_recurrent:
        raise ValueError("selected confidence training requires action recurrence")
    if not episodes:
        raise ValueError("selected confidence training requires fitting episodes")
    expected_seeds = [episode.seed for episode in episodes]
    if len(set(expected_seeds)) != len(expected_seeds):
        raise ValueError("selected confidence fitting episodes contain duplicate seeds")
    if set(frozen_candidates) != set(expected_seeds):
        raise ValueError("frozen candidate map does not exactly match fitting seeds")
    if epochs <= 0 or chunk_length <= 0:
        raise ValueError("selected confidence epochs and chunk length must be positive")
    for name, value in (
        ("learning rate", learning_rate),
        ("weight decay", weight_decay),
        ("loss weight", loss_weight),
        ("gradient clip", max_norm),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"selected confidence {name} must be finite and positive")

    if device != "cpu":
        raise ValueError(
            "selected confidence fitting is fixed to deterministic CPU"
        )
    if any(parameter.device.type != "cpu" for parameter in adapter.parameters()):
        raise ValueError("selected confidence fitting requires a CPU adapter")

    base_state_before = {
        name: value.detach().cpu().clone()
        for name, value in adapter.state_dict().items()
        if ".membership_head." not in name
    }
    head_state_before = {
        name: value.detach().cpu().clone()
        for name, value in adapter.state_dict().items()
        if ".membership_head." in name
    }
    parameter_grad_state = [
        (
            parameter,
            parameter.requires_grad,
            None if parameter.grad is None else parameter.grad.detach().clone(),
        )
        for parameter in adapter.parameters()
    ]
    module_training_state = [
        (module, module.training) for module in adapter.modules()
    ]

    def restore_training_state(
        *,
        rollback_head: bool,
        rollback_base: bool = False,
    ) -> None:
        if rollback_head or rollback_base:
            with torch.no_grad():
                current_state = adapter.state_dict()
                if rollback_head:
                    for name, value in head_state_before.items():
                        current_state[name].copy_(value)
                if rollback_base:
                    for name, value in base_state_before.items():
                        current_state[name].copy_(value)
        for parameter, requires_grad, gradient in parameter_grad_state:
            parameter.requires_grad_(requires_grad)
            parameter.grad = None if gradient is None else gradient.clone()
        for module, training in module_training_state:
            module.train(training)

    parameter_paths: dict[int, list[str]] = {}
    for name, parameter in adapter.named_parameters(remove_duplicate=False):
        parameter_paths.setdefault(id(parameter), []).append(name)
    aliases = [paths for paths in parameter_paths.values() if len(paths) != 1]
    if aliases:
        raise RuntimeError(
            "selected confidence adapter contains aliased parameters: "
            + "; ".join(" / ".join(paths) for paths in aliases)
        )

    membership_parameters: list[tuple[torch.nn.Parameter, ...]] = []
    optimizers: list[torch.optim.AdamW] = []
    try:
        adapter.eval()
        for member in adapter.members:
            if member.membership_head is None:
                raise RuntimeError("adapter membership head is missing")
            for name, parameter in member.named_parameters():
                parameter.requires_grad_(name.startswith("membership_head."))
            member.membership_head.train()
            parameters = tuple(member.membership_head.parameters())
            if not parameters:
                raise RuntimeError("selected confidence parameter group is empty")
            membership_parameters.append(parameters)
            optimizers.append(torch.optim.AdamW(
                parameters,
                lr=learning_rate,
                weight_decay=weight_decay,
            ))
    except Exception:
        restore_training_state(rollback_head=True)
        raise

    member_latents: list[list[torch.Tensor]] = [
        [] for _member in adapter.members
    ]
    selected_candidates: list[torch.Tensor] = []
    selected_targets: list[torch.Tensor] = []
    try:
        for episode in episodes:
            candidates = frozen_candidates[episode.seed]
            if candidates.device.type != "cpu" or candidates.dtype != torch.int64:
                raise ValueError("frozen candidates must be CPU int64 tensors")
            if candidates.shape != episode.parent_actions.shape:
                raise ValueError("frozen candidates do not align with fitting episode")
            actual_candidates, finite, latents = _frozen_selector_episode(
                adapter,
                episode,
                chunk_length=chunk_length,
            )
            if not torch.equal(actual_candidates, candidates):
                raise AssertionError(
                    "frozen selector candidate map drifted before fitting"
                )
            mask, targets = _selected_candidate_confidence_targets(
                episode,
                candidates,
            )
            if bool(mask.any()) and not bool(finite[mask].all()):
                raise ValueError(
                    "frozen ensemble selector is nonfinite on labelled rows"
                )
            selected_candidates.append(candidates[mask])
            selected_targets.append(targets[mask])
            for index, latent in enumerate(latents):
                member_latents[index].append(latent[mask])

        candidates = torch.cat(selected_candidates, dim=0)
        targets = torch.cat(selected_targets, dim=0)
        latents = tuple(torch.cat(parts, dim=0) for parts in member_latents)
        labelled_rows = int(targets.numel())
        positive_rows = int(targets.sum())
        if labelled_rows == 0:
            raise ValueError("selected confidence fit has no early 4-10 rows")
        if any(value.shape[0] != labelled_rows for value in latents):
            raise RuntimeError("selected confidence latent inventory is misaligned")
    except Exception:
        restore_training_state(rollback_head=True)
        raise

    rng = random.Random(seed)
    history: list[dict[str, float]] = []
    try:
        for epoch in range(1, epochs + 1):
            order = list(range(labelled_rows))
            rng.shuffle(order)
            loss_sum = 0.0
            batches = 0
            for start in range(0, labelled_rows, chunk_length):
                indices = torch.tensor(
                    order[start:start + chunk_length],
                    dtype=torch.int64,
                )
                batch_candidates = candidates[indices]
                batch_targets = targets[indices].to(torch.float32)
                losses: list[torch.Tensor] = []
                for member, latent in zip(adapter.members, latents, strict=True):
                    assert member.membership_head is not None
                    logits = member.membership_head(latent[indices])
                    if not bool(torch.isfinite(logits).all()):
                        raise ValueError(
                            "membership confidence is nonfinite on labelled rows"
                        )
                    selected = logits.gather(
                        -1,
                        batch_candidates.unsqueeze(-1),
                    ).squeeze(-1)
                    losses.append(F.binary_cross_entropy_with_logits(
                        selected,
                        batch_targets,
                    ))
                for optimizer in optimizers:
                    optimizer.zero_grad(set_to_none=True)
                weighted_loss = loss_weight * torch.stack(losses).sum()
                if not bool(torch.isfinite(weighted_loss)):
                    raise ValueError(
                        "selected candidate confidence loss is nonfinite"
                    )
                weighted_loss.backward()
                for parameters in membership_parameters:
                    gradients = [
                        parameter.grad for parameter in parameters
                        if parameter.grad is not None
                    ]
                    if len(gradients) != len(parameters) or not all(
                        bool(torch.isfinite(gradient).all())
                        for gradient in gradients
                    ):
                        raise ValueError(
                            "membership confidence gradient is nonfinite or missing"
                        )
                    gradient_norm = _clip_membership_confidence_gradients(
                        parameters,
                        max_norm=max_norm,
                    )
                    if not bool(torch.isfinite(gradient_norm)):
                        raise ValueError(
                            "membership confidence gradient norm is nonfinite"
                        )
                for optimizer in optimizers:
                    optimizer.step()
                if not all(
                    bool(torch.isfinite(parameter).all())
                    for parameters in membership_parameters
                    for parameter in parameters
                ):
                    raise ValueError(
                        "membership confidence parameter became nonfinite"
                    )
                mean_loss = float(
                    torch.stack([loss.detach() for loss in losses]).mean()
                )
                if not math.isfinite(mean_loss):
                    raise ValueError(
                        "selected candidate confidence history loss is nonfinite"
                    )
                loss_sum += mean_loss
                batches += 1
            history.append({
                "epoch": float(epoch),
                "mean_selected_candidate_confidence_loss": loss_sum / batches,
                "membership_confidence_loss_weight": float(loss_weight),
                "labelled_rows": float(labelled_rows),
                "positive_rows": float(positive_rows),
                "fixed_label_batch_size": float(chunk_length),
                "batches": float(batches),
            })
    except Exception:
        restore_training_state(rollback_head=True)
        raise
    finally:
        restore_training_state(rollback_head=False)

    base_state_after = {
        name: value.detach().cpu()
        for name, value in adapter.state_dict().items()
        if ".membership_head." not in name
    }
    changed = [
        name
        for name, before in base_state_before.items()
        if name not in base_state_after or not torch.equal(before, base_state_after[name])
    ]
    if changed or set(base_state_before) != set(base_state_after):
        restore_training_state(rollback_head=True, rollback_base=True)
        raise AssertionError(
            "selected confidence training changed frozen base tensors: "
            + ", ".join(changed)
        )
    return history


def _train_member(
    adapter: ResidualCorrectionAdapter,
    member_index: int,
    episodes: list[EpisodeFeatures],
    *,
    seed: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    chunk_length: int,
    gate_positive_weight: float,
    action_loss_weight: float,
    parent_copy_weight: float,
    device: str,
    preferred_action_loss_weight: float = 1.0,
    membership_loss_mode: str = "balanced",
    membership_confidence_loss_weight: float = 0.0,
    membership_confidence_loss_mode: str = "unweighted",
    preferred_action_uniform_loss_weight: float = 0.0,
    preferred_action_tiebreak_loss_weight: float = 0.0,
    preferred_action_rank_loss_weight: float = 0.0,
    preferred_action_rank_margin: float = 1.0,
    safety_candidate_loss_weight: float = 2.0,
    collision_loss_weight: float = 1.0,
    minimum_margin_loss_weight: float = 0.5,
    physical_danger_loss_weight: float = 2.0,
    collision_positive_weights: torch.Tensor | None = None,
    physical_danger_positive_weights: torch.Tensor | None = None,
    all_collision_row_weight: float = 0.25,
    episode_bootstrap: bool = False,
) -> list[dict[str, float]]:
    _validate_training_loss_weights(
        gate_positive_weight=gate_positive_weight,
        action_loss_weight=action_loss_weight,
        preferred_action_loss_weight=preferred_action_loss_weight,
        membership_confidence_loss_weight=membership_confidence_loss_weight,
        preferred_action_uniform_loss_weight=preferred_action_uniform_loss_weight,
        preferred_action_tiebreak_loss_weight=preferred_action_tiebreak_loss_weight,
        preferred_action_rank_loss_weight=preferred_action_rank_loss_weight,
        safety_candidate_loss_weight=safety_candidate_loss_weight,
        parent_copy_weight=parent_copy_weight,
        collision_loss_weight=collision_loss_weight,
        minimum_margin_loss_weight=minimum_margin_loss_weight,
        physical_danger_loss_weight=physical_danger_loss_weight,
    )
    if not math.isfinite(preferred_action_rank_margin) or (
        preferred_action_rank_margin < 0.0
    ):
        raise ValueError("preferred action rank margin must be finite and nonnegative")
    _validate_action_training_semantics(
        action_logit_mode=adapter.config.action_logit_mode,
        action_loss_weight=action_loss_weight,
        parent_copy_weight=parent_copy_weight,
        preferred_action_uniform_loss_weight=(
            preferred_action_uniform_loss_weight
        ),
        preferred_action_tiebreak_loss_weight=(
            preferred_action_tiebreak_loss_weight
        ),
        preferred_action_rank_loss_weight=preferred_action_rank_loss_weight,
        safety_candidate_loss_weight=safety_candidate_loss_weight,
        membership_loss_mode=membership_loss_mode,
    )
    _validate_membership_confidence_training_semantics(
        enabled=adapter.config.per_action_membership_confidence,
        action_logit_mode=adapter.config.action_logit_mode,
        loss_weight=membership_confidence_loss_weight,
        loss_mode=membership_confidence_loss_mode,
    )
    rng = random.Random(seed)
    member = adapter.members[member_index].to(device)
    optimizer_parameters = _training_optimizer_parameter_groups(
        member,
        per_action_membership_confidence=(
            adapter.config.per_action_membership_confidence
        ),
    )
    optimizer = torch.optim.AdamW(
        optimizer_parameters["base"],
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    membership_optimizer = (
        torch.optim.AdamW(
            optimizer_parameters["membership_confidence"],
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        if adapter.config.per_action_membership_confidence else
        None
    )
    epoch_episodes = (
        [rng.choice(episodes) for _ in episodes]
        if episode_bootstrap else
        list(episodes)
    )
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        rng.shuffle(epoch_episodes)
        loss_sum = 0.0
        membership_confidence_loss_sum = 0.0
        chunks = 0
        member.train()
        for episode in epoch_episodes:
            hidden = None
            for start in range(0, episode.decisions, chunk_length):
                stop = min(start + chunk_length, episode.decisions)
                features = episode.features[:, start:stop].to(device)
                targets = episode.gate_targets[start:stop].to(device)
                valid = episode.gate_valid[start:stop].to(device)
                safe = episode.safe_actions[start:stop].to(device)
                collided = episode.collided_actions[start:stop].to(device)
                minimum_margins = episode.minimum_margins[start:stop].to(device)
                minimum_margin_mask = episode.minimum_margin_mask[start:stop].to(
                    device
                )
                preferred_action_set = episode.preferred_action_set[
                    start:stop
                ].to(device)
                previous_actions = episode.previous_actions[start:stop].to(device)
                safety_candidates = episode.safety_candidate_actions[
                    start:stop
                ].to(device)
                safety_candidate_valid = episode.safety_candidate_valid[
                    start:stop
                ].to(device)
                parent_action_logits = episode.parent_logits[
                    0, start:stop
                ].to(device)
                evaluation_safe = episode.evaluation_safe_actions[start:stop].to(device)
                if adapter.config.per_action_membership_confidence:
                    (
                        gate_logits,
                        action_logits,
                        collision_logits,
                        normalized_margin_predictions,
                        physical_danger_logits,
                        membership_confidence_logits,
                        hidden,
                    ) = member.forward_with_all_safety_and_membership(
                        features,
                        hidden,
                    )
                    if membership_confidence_logits is None:
                        raise RuntimeError(
                            "enabled membership confidence head returned no logits"
                        )
                else:
                    (
                        gate_logits,
                        action_logits,
                        collision_logits,
                        normalized_margin_predictions,
                        physical_danger_logits,
                        hidden,
                    ) = member.forward_with_all_safety(features, hidden)
                    membership_confidence_logits = None
                hidden = hidden.detach()
                gate_logits = gate_logits[0]
                action_logits = adapter.decode_action_logits(
                    action_logits[0],
                    parent_action_logits,
                )
                optimizer.zero_grad(set_to_none=True)
                if membership_optimizer is not None:
                    membership_optimizer.zero_grad(set_to_none=True)
                gate_terms = F.binary_cross_entropy_with_logits(
                    gate_logits,
                    targets,
                    reduction="none",
                )
                weights = torch.where(
                    targets > 0.0,
                    torch.full_like(targets, gate_positive_weight),
                    torch.ones_like(targets),
                )
                gate_loss = (
                    (gate_terms * weights)[valid].sum()
                    / weights[valid].sum().clamp_min(1.0)
                )
                action_mask = valid & (targets > 0.0) & safe.any(dim=-1)
                action_loss = _action_set_loss(
                    action_logits,
                    safe,
                    collided,
                    action_mask,
                )
                preferred_action_loss = (
                    _preferred_action_membership_loss(
                        action_logits,
                        preferred_action_set,
                        valid & (targets > 0.0),
                        mode=membership_loss_mode,
                    )
                    if adapter.config.action_logit_mode
                    == "certified_membership" else
                    _preferred_action_set_loss(
                        action_logits,
                        preferred_action_set,
                        valid & (targets > 0.0),
                    )
                )
                preferred_action_uniform_loss = (
                    _preferred_action_uniform_conditional_loss(
                        action_logits,
                        preferred_action_set,
                        valid & (targets > 0.0),
                    )
                    if preferred_action_uniform_loss_weight > 0.0 else
                    action_logits.sum() * 0.0
                )
                preferred_action_tiebreak_loss = (
                    _preferred_action_conditional_tiebreak_loss(
                        action_logits,
                        preferred_action_set,
                        previous_actions,
                        _preferred_action_tiebreak_mask(
                            preferred_action_set,
                            previous_actions,
                            valid & (targets > 0.0),
                        ),
                    )
                    if preferred_action_tiebreak_loss_weight > 0.0 else
                    action_logits.sum() * 0.0
                )
                preferred_action_rank_loss = (
                    _preferred_action_set_rank_loss(
                        action_logits,
                        preferred_action_set,
                        valid & (targets > 0.0),
                        margin=preferred_action_rank_margin,
                    )
                    if preferred_action_rank_loss_weight > 0.0 else
                    action_logits.sum() * 0.0
                )
                safety_candidate_loss = (
                    _preferred_action_loss(
                        action_logits,
                        safety_candidates,
                        safety_candidate_valid,
                    )
                    if adapter.config.per_action_safety_critic else
                    action_logits.sum() * 0.0
                )
                negative = valid & (targets == 0.0)
                copy_loss = _parent_copy_loss(
                    action_logits,
                    parent_action_logits,
                    negative,
                    action_logit_mode=adapter.config.action_logit_mode,
                )
                if collision_logits is None or normalized_margin_predictions is None:
                    collision_loss = action_logits.sum() * 0.0
                    minimum_margin_loss = action_logits.sum() * 0.0
                else:
                    collision_loss, minimum_margin_loss = _dense_safety_losses(
                        collision_logits[0],
                        normalized_margin_predictions[0],
                        collided,
                        minimum_margins,
                        minimum_margin_mask,
                        collision_positive_weights=collision_positive_weights,
                        all_collision_row_weight=all_collision_row_weight,
                    )
                if physical_danger_logits is None:
                    physical_danger_loss = action_logits.sum() * 0.0
                else:
                    if physical_danger_positive_weights is None:
                        raise ValueError(
                            "physical danger heads require positive class weights"
                        )
                    physical_danger_loss = _physical_danger_loss(
                        physical_danger_logits[0],
                        evaluation_safe,
                        positive_weights=physical_danger_positive_weights,
                    )
                membership_confidence_loss = (
                    _membership_confidence_loss(
                        membership_confidence_logits[0],
                        preferred_action_set,
                        valid & (targets > 0.0),
                        mode=membership_confidence_loss_mode,
                    )
                    if membership_confidence_logits is not None else
                    action_logits.sum() * 0.0
                )
                loss = (
                    gate_loss
                    + action_loss_weight * action_loss
                    + preferred_action_loss_weight * preferred_action_loss
                    + preferred_action_uniform_loss_weight
                    * preferred_action_uniform_loss
                    + preferred_action_tiebreak_loss_weight
                    * preferred_action_tiebreak_loss
                    + preferred_action_rank_loss_weight
                    * preferred_action_rank_loss
                    + safety_candidate_loss_weight * safety_candidate_loss
                    + parent_copy_weight * copy_loss
                    + collision_loss_weight * collision_loss
                    + minimum_margin_loss_weight * minimum_margin_loss
                    + physical_danger_loss_weight * physical_danger_loss
                )
                loss.backward()
                if membership_optimizer is not None:
                    (
                        membership_confidence_loss_weight
                        * membership_confidence_loss
                    ).backward()
                _clip_member_gradients(
                    member,
                    excluded_parameters=optimizer_parameters[
                        "membership_confidence"
                    ],
                )
                if membership_optimizer is not None:
                    _clip_membership_confidence_gradients(
                        optimizer_parameters["membership_confidence"],
                    )
                optimizer.step()
                if membership_optimizer is not None:
                    membership_optimizer.step()
                loss_sum += float(loss.detach())
                membership_confidence_loss_sum += float(
                    membership_confidence_loss.detach()
                )
                chunks += 1
        epoch_history = {
            "epoch": float(epoch),
            "mean_chunk_loss": loss_sum / max(chunks, 1),
        }
        if adapter.config.per_action_membership_confidence:
            epoch_history.update({
                "mean_membership_confidence_loss": (
                    membership_confidence_loss_sum / max(chunks, 1)
                ),
                "membership_confidence_loss_weight": float(
                    membership_confidence_loss_weight
                ),
            })
        history.append(epoch_history)
    member.eval()
    return history


def _predict_episode(
    adapter: ResidualCorrectionAdapter,
    episode: EpisodeFeatures,
    *,
    device: str,
) -> dict[str, torch.Tensor]:
    hidden = None
    gate_values = []
    action_values = []
    action_finite_values = []
    membership_values = []
    membership_finite_values = []
    collision_values = []
    margin_values = []
    physical_danger_values = []
    with torch.no_grad():
        for start in range(0, episode.decisions, 256):
            stop = min(start + 256, episode.decisions)
            features = episode.features[:, start:stop].to(device)
            if adapter.config.per_action_membership_confidence:
                outputs = [
                    member.forward_with_all_safety_and_membership(
                        features,
                        None if hidden is None else hidden[index],
                    )
                    for index, member in enumerate(adapter.members)
                ]
            else:
                legacy_outputs = [
                    member.forward_with_all_safety(
                        features,
                        None if hidden is None else hidden[index],
                    )
                    for index, member in enumerate(adapter.members)
                ]
                outputs = [
                    (*values[:-1], None, values[-1])
                    for values in legacy_outputs
                ]
            gate_values.append(torch.stack([
                finite_sigmoid(gate[0])
                for gate, _actions, _collision, _margin, _physical,
                _membership, _hidden in outputs
            ], dim=0).cpu())
            action_logits = torch.stack([
                adapter.decode_action_logits(
                    actions[0],
                    episode.parent_logits[0, start:stop].to(device),
                )
                for _gate, actions, _collision, _margin, _physical,
                _membership, _hidden in outputs
            ], dim=0)
            action_probabilities, action_member_finite = (
                finite_action_probabilities(
                    action_logits,
                    adapter.config.action_logit_mode,
                )
            )
            action_values.append(action_probabilities.cpu())
            action_finite_values.append(action_member_finite.cpu())
            if outputs[0][5] is not None:
                membership_logits = torch.stack([
                    membership[0]
                    for _gate, _actions, _collision, _margin, _physical,
                    membership, _hidden in outputs
                    if membership is not None
                ], dim=0)
                membership_probabilities, membership_member_finite = (
                    finite_action_probabilities(
                        membership_logits,
                        "certified_membership",
                    )
                )
                membership_values.append(membership_probabilities.cpu())
                membership_finite_values.append(
                    membership_member_finite.cpu()
                )
            if outputs[0][2] is not None:
                collision_values.append(torch.stack([
                    finite_sigmoid(collision[0])
                    for _gate, _actions, collision, _margin, _physical,
                    _membership, _hidden
                    in outputs
                    if collision is not None
                ], dim=0).cpu())
                margin_values.append(torch.stack([
                    margin[0] * 16.0
                    for _gate, _actions, _collision, margin, _physical,
                    _membership, _hidden
                    in outputs
                    if margin is not None
                ], dim=0).cpu())
            if outputs[0][4] is not None:
                physical_danger_values.append(torch.stack([
                    finite_sigmoid(physical[0])
                    for _gate, _actions, _collision, _margin, physical,
                    _membership, _hidden
                    in outputs
                    if physical is not None
                ], dim=0).cpu())
            hidden = tuple(
                next_hidden.detach()
                for _gate, _actions, _collision, _margin, _physical,
                _membership, next_hidden
                in outputs
            )
    gates = torch.cat(gate_values, dim=1)
    action_probabilities = torch.cat(action_values, dim=1)
    action_member_finite = torch.cat(action_finite_values, dim=1)
    membership_probabilities = (
        torch.cat(membership_values, dim=1)
        if membership_values else
        None
    )
    membership_member_finite = (
        torch.cat(membership_finite_values, dim=1)
        if membership_finite_values else
        None
    )
    action_summary = (
        ensemble_action_summary(
            action_probabilities,
            action_member_finite,
            membership_probabilities,
            membership_member_finite,
        )
        if membership_probabilities is not None else
        ensemble_action_summary(
            action_probabilities,
            action_member_finite,
        )
    )
    mean_actions = action_summary["mean_action_probabilities"]
    candidates = action_summary["candidates"]
    result = {
        "mean_gate": gates.mean(dim=0),
        "minimum_gate": gates.amin(dim=0),
        "action_all_members_finite": action_summary[
            "action_all_members_finite"
        ],
        "action_member_finite": action_member_finite,
        "action_probabilities": action_probabilities,
        "mean_action_probabilities": mean_actions,
        "action_confidence": action_summary["action_confidence"],
        "candidates": candidates,
        "agreement": action_summary["agreement"],
    }
    if collision_values:
        result["collision_probabilities"] = torch.cat(collision_values, dim=1)
        result["minimum_margins"] = torch.cat(margin_values, dim=1)
    if physical_danger_values:
        result["physical_danger_probabilities"] = torch.cat(
            physical_danger_values,
            dim=1,
        )
    if membership_probabilities is not None:
        assert membership_member_finite is not None
        result.update({
            "membership_probabilities": membership_probabilities,
            "mean_membership_probabilities": action_summary[
                "mean_membership_probabilities"
            ],
            "membership_member_finite": membership_member_finite,
            "selector_all_members_finite": action_summary[
                "selector_all_members_finite"
            ],
            "membership_all_members_finite": action_summary[
                "membership_all_members_finite"
            ],
        })
    return result


def _preferred_equivalence(
    episode: EpisodeFeatures | Any,
    positive: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    preferred = getattr(
        episode,
        "preferred_actions",
        torch.full_like(episode.parent_actions, -1),
    )
    required = getattr(
        episode,
        "preferred_correction_required",
        positive & (preferred >= 0) & (preferred != episode.parent_actions),
    )
    required = positive & required
    equivalent = getattr(episode, "preferred_equivalent_actions", None)
    if equivalent is None:
        safe = getattr(episode, "safe_actions", None)
        if safe is None:
            safe = episode.evaluation_safe_actions
        evaluation_safe = getattr(
            episode,
            "evaluation_safe_actions",
            safe,
        )
        action_ids = torch.arange(
            safe.shape[-1],
            dtype=episode.parent_actions.dtype,
            device=episode.parent_actions.device,
        ).unsqueeze(0)
        equivalent = (
            safe
            & evaluation_safe
            & (action_ids != episode.parent_actions.unsqueeze(-1))
            & required.unsqueeze(-1)
        )
    if equivalent.shape != (*episode.parent_actions.shape, 18):
        raise ValueError("preferred equivalent actions do not align with episode")
    if equivalent.dtype != torch.bool:
        raise ValueError("preferred equivalent actions must be Boolean")
    return preferred, required, equivalent


def _future_onset_split_diagnostics(
    predictions: dict[int, dict[str, torch.Tensor]],
    episodes: list[EpisodeFeatures],
    *,
    ensemble_size: int,
) -> dict[str, Any]:
    minimum_ensemble_agreement = (
        math.ceil(ensemble_size * 2 / 3) / ensemble_size
    )
    stage_names = (
        "early_4_10_targets",
        "early_correction_required_targets",
        "gate_outputs_finite",
        "raw_gate_positive_at_0_5",
        "raw_action_all_members_finite",
        "exact_preferred_candidate_possible",
        "certified_equivalent_candidate_possible",
        "physical_all_members_finite",
        "physical_predicted_safe_at_0_5",
        "prethreshold_intersection_upper_bound",
        "minimum_search_threshold_intersection_upper_bound",
    )
    blocking_names = (
        "gate_output_nonfinite",
        "mean_gate_below_0_5",
        "minimum_member_gate_below_0_25",
        "raw_action_member_nonfinite",
        "action_summary_nonfinite",
        "action_confidence_below_0_2",
        "action_agreement_below_minimum",
        "candidate_not_certified_equivalent",
        "candidate_physical_nonfinite",
        "candidate_predicted_unsafe",
    )
    totals = {name: 0 for name in stage_names}
    blocking = {name: 0 for name in blocking_names}
    by_lead = {
        str(lead): {name: 0 for name in stage_names}
        for lead in range(
            EARLY_ONSET_MINIMUM_LEAD_DECISIONS,
            FUTURE_ONSET_HORIZON_DECISIONS + 1,
        )
    }

    for episode in episodes:
        values = predictions[episode.seed]
        physical = values.get("physical_danger_probabilities")
        decisions = episode.parent_actions.shape
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
        gate_finite = (
            torch.isfinite(values["mean_gate"])
            & torch.isfinite(values["minimum_gate"])
        )
        raw_gate_positive = gate_finite & (values["mean_gate"] >= 0.5)
        raw_action_finite = values["action_all_members_finite"]
        action_summary_finite = (
            torch.isfinite(values["action_confidence"])
            & torch.isfinite(values["agreement"])
        )
        action_finite = raw_action_finite & action_summary_finite
        raw_actions = values["candidates"]
        preferred_actions, correction_required, equivalent_actions = (
            _preferred_equivalence(episode, episode.anticipatory)
        )
        exact_path = (
            correction_required & (raw_actions == preferred_actions)
        )
        equivalent_path = equivalent_actions.gather(
            -1,
            raw_actions.unsqueeze(-1),
        ).squeeze(-1)

        raw_evaluation_safe = episode.evaluation_safe_actions.gather(
            -1,
            raw_actions.unsqueeze(-1),
        ).squeeze(-1)
        evaluation_safe_possible = equivalent_path & raw_evaluation_safe

        if physical is None:
            raw_physical_finite = torch.zeros(decisions, dtype=torch.bool)
            raw_physical_safe = torch.zeros(decisions, dtype=torch.bool)
        else:
            raw_indices = raw_actions.unsqueeze(0).unsqueeze(-1).expand(
                physical.shape[0],
                *raw_actions.shape,
                1,
            )
            raw_scores = physical.gather(-1, raw_indices).squeeze(-1)
            raw_physical_finite = torch.isfinite(raw_scores).all(dim=0)
            raw_physical_safe = raw_physical_finite & (
                (raw_scores <= 0.5).to(torch.float32).mean(dim=0)
                >= minimum_ensemble_agreement
            )
        physical_finite_possible = (
            equivalent_path & raw_evaluation_safe & raw_physical_finite
        )
        physical_safe_possible = (
            equivalent_path & raw_evaluation_safe & raw_physical_safe
        )
        prethreshold_upper = (
            gate_finite & action_finite & physical_safe_possible
        )
        raw_quality_floor = (
            action_finite
            & (values["action_confidence"] >= 0.2)
            & (values["agreement"] >= minimum_ensemble_agreement)
        )
        minimum_threshold_upper = (
            raw_gate_positive
            & (values["minimum_gate"] >= 0.25)
            & action_finite
            & equivalent_path
            & raw_evaluation_safe
            & raw_physical_safe
            & raw_quality_floor
        )
        stages = {
            "early_4_10_targets": early,
            "early_correction_required_targets": early & correction_required,
            "gate_outputs_finite": early & correction_required & gate_finite,
            "raw_gate_positive_at_0_5": (
                early & correction_required & raw_gate_positive
            ),
            "raw_action_all_members_finite": (
                early & correction_required & raw_action_finite
            ),
            "exact_preferred_candidate_possible": early & exact_path,
            "certified_equivalent_candidate_possible": (
                early & equivalent_path
            ),
            "physical_all_members_finite": (
                early & correction_required & physical_finite_possible
            ),
            "physical_predicted_safe_at_0_5": (
                early & correction_required & physical_safe_possible
            ),
            "prethreshold_intersection_upper_bound": (
                early & correction_required & prethreshold_upper
            ),
            "minimum_search_threshold_intersection_upper_bound": (
                early & correction_required & minimum_threshold_upper
            ),
        }
        for name, mask in stages.items():
            totals[name] += int(mask.sum())
            for lead in range(
                EARLY_ONSET_MINIMUM_LEAD_DECISIONS,
                FUTURE_ONSET_HORIZON_DECISIONS + 1,
            ):
                lead_mask = episode.anticipatory_lead_decisions == lead
                by_lead[str(lead)][name] += int((mask & lead_mask).sum())

        reasons = {
            "gate_output_nonfinite": ~gate_finite,
            "mean_gate_below_0_5": gate_finite & (values["mean_gate"] < 0.5),
            "minimum_member_gate_below_0_25": (
                gate_finite & (values["minimum_gate"] < 0.25)
            ),
            "raw_action_member_nonfinite": ~raw_action_finite,
            "action_summary_nonfinite": (
                raw_action_finite & ~action_summary_finite
            ),
            "action_confidence_below_0_2": (
                action_finite & (values["action_confidence"] < 0.2)
            ),
            "action_agreement_below_minimum": (
                action_finite
                & (values["agreement"] < minimum_ensemble_agreement)
            ),
            "candidate_not_certified_equivalent": ~equivalent_path,
            "candidate_physical_nonfinite": (
                evaluation_safe_possible & ~physical_finite_possible
            ),
            "candidate_predicted_unsafe": (
                physical_finite_possible & ~physical_safe_possible
            ),
        }
        for name, mask in reasons.items():
            blocking[name] += int((early & correction_required & mask).sum())

    return {
        "episodes": len(episodes),
        "threshold_search_floors": {
            "mean_gate": 0.5,
            "minimum_member_gate": 0.25,
            "action_confidence": 0.2,
            "ensemble_agreement": minimum_ensemble_agreement,
            "all_action_members_must_be_finite": True,
            "candidate_physical_danger": 0.5,
            "all_physical_members_must_be_finite": True,
        },
        "stage_counts": totals,
        "by_lead_decisions": by_lead,
        "blocking_reasons": blocking,
    }


def _future_onset_calibration_diagnostics(
    predictions: dict[int, dict[str, torch.Tensor]],
    training_episodes: list[EpisodeFeatures],
    calibration_episodes: list[EpisodeFeatures],
    *,
    ensemble_size: int,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "kind": "future_onset_calibration_failure_diagnostics",
        "read_only_audit": True,
        "threshold_selection_uses_validation": False,
        "future_onset_horizon_decisions": FUTURE_ONSET_HORIZON_DECISIONS,
        "early_lead_decisions": [
            EARLY_ONSET_MINIMUM_LEAD_DECISIONS,
            FUTURE_ONSET_HORIZON_DECISIONS,
        ],
        "splits": {
            "training": _future_onset_split_diagnostics(
                predictions,
                training_episodes,
                ensemble_size=ensemble_size,
            ),
            "calibration": _future_onset_split_diagnostics(
                predictions,
                calibration_episodes,
                ensemble_size=ensemble_size,
            ),
        },
    }


def _calibrate(
    predictions: dict[int, dict[str, torch.Tensor]],
    training_episodes: list[EpisodeFeatures],
    calibration_episodes: list[EpisodeFeatures],
    *,
    ensemble_size: int,
    per_action_safety_critic: bool,
    per_action_physical_danger: bool,
    future_onset_gate: bool = False,
) -> ResidualRuntimeConfig:
    agreement_values = sorted({
        math.ceil(ensemble_size * 2 / 3) / ensemble_size,
        1.0,
    })
    if future_onset_gate:
        if not (per_action_safety_critic and per_action_physical_danger):
            raise ValueError(
                "future-onset calibration requires direct visual physical heads"
            )

        calibration_pool = training_episodes + calibration_episodes
        future_candidates: list[
            tuple[tuple[Any, ...], ResidualRuntimeConfig]
        ] = []
        for (
            minimum_gate,
            action_confidence,
            action_agreement,
            candidate_danger,
            candidate_agreement,
        ) in product(
            (0.25, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
            (0.2, 0.4, 0.6, 0.8),
            agreement_values,
            (0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5),
            agreement_values,
        ):
            selection_runtime = ResidualRuntimeConfig(
                gate_probability_threshold=0.5,
                minimum_member_gate_probability=minimum_gate,
                action_probability_threshold=action_confidence,
                ensemble_agreement_threshold=action_agreement,
                override_logit_margin=1.0,
                legacy_gate_enabled=False,
                critic_enabled=True,
                current_critic_request_enabled=False,
                prefer_safe_previous_action=False,
                critic_signal="physical_danger",
                candidate_physical_danger_probability_threshold=candidate_danger,
                candidate_safety_agreement_threshold=candidate_agreement,
                future_onset_gate_enabled=True,
            )
            # For each ensemble/action/safety setting, place the mean-gate
            # threshold immediately above every observed disallowed request.
            # Candidate selection runs with onset active so this prefilter has
            # exactly the same escape/hold semantics as offline and live use.
            disallowed_scores: list[torch.Tensor] = []
            for episode in calibration_pool:
                values = predictions[episode.seed]
                physical = values.get("physical_danger_probabilities")
                if physical is None:
                    raise ValueError(
                        "future-onset calibration lacks physical danger predictions"
                    )
                selection = residual_candidate_selection(
                    correction_actions=values["candidates"],
                    correction_confidence=values["action_confidence"],
                    agreement=values["agreement"],
                    previous_actions=episode.previous_actions,
                    runtime_config=selection_runtime,
                    physical_danger_probabilities=physical,
                    parent_actions=episode.parent_actions,
                    future_onset=torch.ones_like(
                        values["mean_gate"],
                        dtype=torch.bool,
                    ),
                )
                candidates = selection["correction_actions"]
                candidate_indices = candidates.unsqueeze(0).unsqueeze(-1).expand(
                    physical.shape[0],
                    *candidates.shape,
                    1,
                )
                candidate_scores = physical.gather(
                    -1,
                    candidate_indices,
                ).squeeze(-1)
                predicted_candidate_safe = (
                    torch.isfinite(candidate_scores).all(dim=0)
                    & (
                    (
                        torch.isfinite(candidate_scores)
                        & (candidate_scores <= candidate_danger)
                    )
                    .to(values["mean_gate"].dtype)
                    .mean(dim=0)
                    >= candidate_agreement
                    )
                )
                actual_candidate_safe = episode.evaluation_safe_actions.gather(
                    -1,
                    candidates.unsqueeze(-1),
                ).squeeze(-1)
                positive = episode.gate_valid & (episode.gate_targets > 0.0)
                _preferred, correction_required, equivalent_actions = (
                    _preferred_equivalence(episode, positive)
                )
                candidate_equivalent = equivalent_actions.gather(
                    -1,
                    candidates.unsqueeze(-1),
                ).squeeze(-1)
                eligible = (
                    torch.isfinite(values["mean_gate"])
                    & torch.isfinite(values["minimum_gate"])
                    & values["action_all_members_finite"]
                    & (values["minimum_gate"] >= minimum_gate)
                    & (
                        selection["correction_confidence"]
                        >= action_confidence
                    )
                    & (selection["agreement"] >= action_agreement)
                    & (candidates != episode.parent_actions)
                    & predicted_candidate_safe
                )
                disallowed = (
                    ~correction_required
                    | ~actual_candidate_safe
                    | ~candidate_equivalent
                )
                disallowed_scores.append(
                    values["mean_gate"][eligible & disallowed]
                )
            available = [value for value in disallowed_scores if value.numel()]
            maximum_disallowed = (
                max(float(value.max()) for value in available)
                if available else
                0.4999
            )
            gate_threshold = min(
                1.0,
                max(0.5, maximum_disallowed + 1e-4),
            )
            runtime_values = asdict(selection_runtime)
            runtime_values["gate_probability_threshold"] = gate_threshold
            runtime = ResidualRuntimeConfig(**runtime_values)
            train = _metrics(predictions, training_episodes, runtime)["total"]
            calibration = _metrics(
                predictions,
                calibration_episodes,
                runtime,
            )["total"]
            fail_closed = all(
                metrics[name] == 0
                for metrics in (train, calibration)
                for name in (
                    "unbeneficial_overrides",
                    "false_overrides",
                    "unsafe_overrides",
                    "non_equivalent_overrides",
                )
            )
            early_covered = (
                train["early_beneficial_overrides"] > 0
                and calibration["early_beneficial_overrides"] > 0
            )
            if not (fail_closed and early_covered):
                continue
            score = (
                min(
                    train["early_anticipatory_opportunity_recall"],
                    calibration["early_anticipatory_opportunity_recall"],
                ),
                min(
                    train["early_danger_event_cluster_recall"],
                    calibration["early_danger_event_cluster_recall"],
                ),
                train["early_beneficial_overrides"]
                + calibration["early_beneficial_overrides"],
                train["beneficial_overrides"] + calibration["beneficial_overrides"],
                train["mean_covered_early_lead_decisions"]
                + calibration["mean_covered_early_lead_decisions"],
                -train["candidate_safety_vetoes"]
                - calibration["candidate_safety_vetoes"],
            )
            future_candidates.append((score, runtime))

        if not future_candidates:
            raise ValueError(
                "no fail-closed future-onset calibration covers early events "
                "in both training and calibration episodes"
            )
        _onset_score, onset_runtime = max(
            future_candidates,
            key=lambda item: item[0],
        )

        # Recover useful current-danger reactions where they do not compromise
        # the already-certified anticipatory behavior.
        expanded_candidates: list[
            tuple[tuple[Any, ...], ResidualRuntimeConfig]
        ] = []
        expansion_settings = [(False, 1.0, 1.0)]
        expansion_settings.extend(
            (True, parent_danger, parent_agreement)
            for parent_danger, parent_agreement in product(
                (0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 1.0),
                agreement_values,
            )
        )
        for current_enabled, parent_danger, parent_agreement in expansion_settings:
            runtime_values = asdict(onset_runtime)
            runtime_values.update({
                "current_critic_request_enabled": current_enabled,
                "parent_physical_danger_probability_threshold": parent_danger,
                "parent_danger_agreement_threshold": parent_agreement,
            })
            runtime = ResidualRuntimeConfig(**runtime_values)
            train = _metrics(predictions, training_episodes, runtime)["total"]
            calibration = _metrics(
                predictions,
                calibration_episodes,
                runtime,
            )["total"]
            fail_closed = all(
                metrics[name] == 0
                for metrics in (train, calibration)
                for name in (
                    "unbeneficial_overrides",
                    "false_overrides",
                    "unsafe_overrides",
                    "non_equivalent_overrides",
                )
            )
            early_covered = (
                train["early_beneficial_overrides"] > 0
                and calibration["early_beneficial_overrides"] > 0
            )
            if not (fail_closed and early_covered):
                continue
            score = (
                min(
                    train["early_anticipatory_opportunity_recall"],
                    calibration["early_anticipatory_opportunity_recall"],
                ),
                min(
                    train["early_danger_event_cluster_recall"],
                    calibration["early_danger_event_cluster_recall"],
                ),
                train["beneficial_overrides"] + calibration["beneficial_overrides"],
                train["beneficial_current_only_overrides"]
                + calibration["beneficial_current_only_overrides"],
                int(current_enabled),
                -train["candidate_safety_vetoes"]
                - calibration["candidate_safety_vetoes"],
            )
            expanded_candidates.append((score, runtime))
        if not expanded_candidates:
            # The conservative onset runtime was already admitted above, so
            # this would indicate an internal inconsistency rather than a
            # reason to emit a legacy artifact.
            raise RuntimeError("future-onset calibration expansion lost its baseline")
        return max(expanded_candidates, key=lambda item: item[0])[1]

    candidates: list[tuple[tuple[Any, ...], ResidualRuntimeConfig]] = []
    calibration_pool = training_episodes + calibration_episodes
    for minimum_gate in (0.25, 0.4, 0.5, 0.6, 0.7):
        for action_confidence in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
            for agreement in agreement_values:
                negative_scores = []
                for episode in calibration_pool:
                    values = predictions[episode.seed]
                    candidate_safe = episode.safe_actions.gather(
                        -1,
                        values["candidates"].unsqueeze(-1),
                    ).squeeze(-1)
                    positive = episode.gate_valid & (episode.gate_targets > 0.0)
                    _preferred, correction_required, equivalent_actions = (
                        _preferred_equivalence(episode, positive)
                    )
                    candidate_equivalent = equivalent_actions.gather(
                        -1,
                        values["candidates"].unsqueeze(-1),
                    ).squeeze(-1)
                    eligible = (
                        (values["minimum_gate"] >= minimum_gate)
                        & (values["action_confidence"] >= action_confidence)
                        & (values["agreement"] >= agreement)
                        & (values["candidates"] != episode.parent_actions)
                    )
                    disallowed = (
                        ~correction_required
                        | ~candidate_safe
                        | ~candidate_equivalent
                    )
                    negative_scores.append(
                        values["mean_gate"][eligible & disallowed]
                    )
                available = [values for values in negative_scores if values.numel()]
                maximum_negative = (
                    max(float(values.max()) for values in available)
                    if available else
                    0.5
                )
                runtime = ResidualRuntimeConfig(
                    # Float32 runtime comparison needs a material epsilon.
                    gate_probability_threshold=min(
                        1.0,
                        max(0.5, maximum_negative) + 1e-4,
                    ),
                    minimum_member_gate_probability=minimum_gate,
                    action_probability_threshold=action_confidence,
                    ensemble_agreement_threshold=agreement,
                    override_logit_margin=1.0,
                )
                train = _metrics(predictions, training_episodes, runtime)["total"]
                calibration = _metrics(
                    predictions,
                    calibration_episodes,
                    runtime,
                )["total"]
                admissible = int(
                    train["false_overrides"] == 0
                    and calibration["false_overrides"] == 0
                    and train["unsafe_overrides"] == 0
                    and calibration["unsafe_overrides"] == 0
                    and train["non_equivalent_overrides"] == 0
                    and calibration["non_equivalent_overrides"] == 0
                )
                train_coverage = (
                    train["equivalent_action_overrides"]
                    / max(train["equivalent_action_targets"], 1)
                )
                calibration_coverage = (
                    calibration["equivalent_action_overrides"]
                    / max(calibration["equivalent_action_targets"], 1)
                )
                score = (
                    admissible,
                    int(
                        calibration["equivalent_action_overrides"] > 0
                        and train["equivalent_action_overrides"] > 0
                    ),
                    min(train_coverage, calibration_coverage),
                    (
                        train["equivalent_action_overrides"]
                        + calibration["equivalent_action_overrides"]
                    ),
                    -calibration["overrides"],
                    -train["overrides"],
                )
                candidates.append((score, runtime))
    admissible = [item for item in candidates if item[0][0] == 1]
    if not admissible:
        raise ValueError("no fail-closed residual calibration is available")
    legacy_runtime = max(admissible, key=lambda item: item[0])[1]
    if not per_action_safety_critic:
        return legacy_runtime

    if per_action_physical_danger:
        physical_fallback_values = asdict(legacy_runtime)
        physical_fallback_values["critic_signal"] = "physical_danger"
        physical_fallback_values["legacy_gate_enabled"] = False
        physical_fallback = ResidualRuntimeConfig(**physical_fallback_values)
        critic_candidates: list[
            tuple[tuple[Any, ...], ResidualRuntimeConfig]
        ] = []
        for (
            action_confidence,
            action_agreement,
            parent_danger,
            candidate_danger,
            parent_agreement,
            candidate_agreement,
        ) in product(
            (0.2, 0.4, 0.6, 0.8),
            agreement_values,
            (0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75),
            (0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4),
            agreement_values,
            agreement_values,
        ):
            runtime_values = asdict(legacy_runtime)
            runtime_values.update({
                "legacy_gate_enabled": False,
                "critic_enabled": True,
                "prefer_safe_previous_action": True,
                "critic_signal": "physical_danger",
                "action_probability_threshold": action_confidence,
                "ensemble_agreement_threshold": action_agreement,
                "parent_physical_danger_probability_threshold": parent_danger,
                "candidate_physical_danger_probability_threshold": candidate_danger,
                "parent_danger_agreement_threshold": parent_agreement,
                "candidate_safety_agreement_threshold": candidate_agreement,
            })
            runtime = ResidualRuntimeConfig(**runtime_values)
            train = _metrics(predictions, training_episodes, runtime)["total"]
            calibration = _metrics(
                predictions,
                calibration_episodes,
                runtime,
            )["total"]
            admissible = int(
                train["unbeneficial_overrides"] == 0
                and calibration["unbeneficial_overrides"] == 0
                and train["false_overrides"] == 0
                and calibration["false_overrides"] == 0
                and train["unsafe_overrides"] == 0
                and calibration["unsafe_overrides"] == 0
                and train["non_equivalent_overrides"] == 0
                and calibration["non_equivalent_overrides"] == 0
            )
            train_coverage = (
                train["beneficial_overrides"]
                / max(train["safety_critic_opportunities"], 1)
            )
            calibration_coverage = (
                calibration["beneficial_overrides"]
                / max(calibration["safety_critic_opportunities"], 1)
            )
            score = (
                admissible,
                int(
                    train["beneficial_overrides"] > 0
                    and calibration["beneficial_overrides"] > 0
                ),
                min(train_coverage, calibration_coverage),
                min(
                    train["danger_event_cluster_recall"],
                    calibration["danger_event_cluster_recall"],
                ),
                train["beneficial_overrides"]
                + calibration["beneficial_overrides"],
                -train["candidate_safety_vetoes"]
                - calibration["candidate_safety_vetoes"],
            )
            critic_candidates.append((score, runtime))
        safe_critic = [item for item in critic_candidates if item[0][0] == 1]
        if not safe_critic:
            return physical_fallback
        best_score, best_runtime = max(safe_critic, key=lambda item: item[0])
        if best_score[1] == 0:
            return physical_fallback
        return best_runtime

    critic_candidates: list[tuple[tuple[Any, ...], ResidualRuntimeConfig]] = []
    action_confidence_thresholds = (0.2, 0.4, 0.6, 0.8)
    parent_collision_thresholds = (0.5, 0.7, 0.85, 0.95)
    candidate_collision_thresholds = (0.1, 0.35, 0.5)
    parent_margin_thresholds = (0.0, 4.0, 8.0, 12.0)
    candidate_margin_thresholds = (4.0, 8.0, 12.0)
    for (
        action_confidence,
        action_agreement,
        parent_collision,
        candidate_collision,
        parent_margin,
        candidate_margin,
        parent_agreement,
        candidate_agreement,
    ) in product(
        action_confidence_thresholds,
        agreement_values,
        parent_collision_thresholds,
        candidate_collision_thresholds,
        parent_margin_thresholds,
        candidate_margin_thresholds,
        agreement_values,
        agreement_values,
    ):
        runtime_values = asdict(legacy_runtime)
        runtime_values.update({
            # The critic replaces, rather than compounds, the legacy request
            # after independent safety calibration.
            "legacy_gate_enabled": False,
            "critic_enabled": True,
            "prefer_safe_previous_action": True,
            "action_probability_threshold": action_confidence,
            "ensemble_agreement_threshold": action_agreement,
            "parent_collision_probability_threshold": parent_collision,
            "candidate_collision_probability_threshold": candidate_collision,
            "parent_minimum_margin_threshold": parent_margin,
            "candidate_minimum_margin_threshold": candidate_margin,
            "parent_danger_agreement_threshold": parent_agreement,
            "candidate_safety_agreement_threshold": candidate_agreement,
        })
        runtime = ResidualRuntimeConfig(**runtime_values)
        train = _metrics(predictions, training_episodes, runtime)["total"]
        calibration = _metrics(
            predictions,
            calibration_episodes,
            runtime,
        )["total"]
        admissible = int(
            train["unbeneficial_overrides"] == 0
            and calibration["unbeneficial_overrides"] == 0
        )
        train_coverage = (
            train["beneficial_overrides"]
            / max(train["safety_critic_opportunities"], 1)
        )
        calibration_coverage = (
            calibration["beneficial_overrides"]
            / max(calibration["safety_critic_opportunities"], 1)
        )
        score = (
            admissible,
            int(
                train["beneficial_overrides"] > 0
                and calibration["beneficial_overrides"] > 0
            ),
            min(train_coverage, calibration_coverage),
            min(
                train["danger_event_cluster_recall"],
                calibration["danger_event_cluster_recall"],
            ),
            train["beneficial_overrides"]
            + calibration["beneficial_overrides"],
            train["critic_only_overrides"]
            + calibration["critic_only_overrides"],
            -train["candidate_safety_vetoes"]
            - calibration["candidate_safety_vetoes"],
        )
        critic_candidates.append((score, runtime))
    safe_critic = [item for item in critic_candidates if item[0][0] == 1]
    if not safe_critic:
        return legacy_runtime
    best_score, best_runtime = max(safe_critic, key=lambda item: item[0])
    # Dense heads are advisory until they add a separately certified correction
    # on both fitting and calibration data.
    if best_score[1] == 0:
        return legacy_runtime
    return best_runtime


def _override_masks(
    values: dict[str, torch.Tensor],
    episode: EpisodeFeatures,
    runtime: ResidualRuntimeConfig,
) -> dict[str, torch.Tensor]:
    future_onset = residual_future_onset_mask(
        values["mean_gate"],
        values["minimum_gate"],
        runtime,
    )
    selection = residual_candidate_selection(
        correction_actions=values["candidates"],
        correction_confidence=values["action_confidence"],
        agreement=values["agreement"],
        previous_actions=getattr(episode, "previous_actions", None),
        runtime_config=runtime,
        collision_probabilities=values.get("collision_probabilities"),
        minimum_margins=values.get("minimum_margins"),
        physical_danger_probabilities=values.get(
            "physical_danger_probabilities"
        ),
        parent_actions=episode.parent_actions,
        future_onset=future_onset,
    )
    masks = residual_override_masks(
        mean_gate=values["mean_gate"],
        minimum_gate=values["minimum_gate"],
        action_all_members_finite=values["action_all_members_finite"],
        correction_confidence=selection["correction_confidence"],
        agreement=selection["agreement"],
        correction_actions=selection["correction_actions"],
        parent_actions=episode.parent_actions,
        runtime_config=runtime,
        collision_probabilities=values.get("collision_probabilities"),
        minimum_margins=values.get("minimum_margins"),
        physical_danger_probabilities=values.get(
            "physical_danger_probabilities"
        ),
    )
    masks.update(selection)
    return masks


def _event_cluster_counts(
    targets: torch.Tensor,
    overrides: torch.Tensor,
) -> tuple[int, int]:
    clusters = 0
    covered = 0
    active = False
    cluster_covered = False
    for target, override in zip(
        targets.to(torch.bool).tolist(),
        overrides.to(torch.bool).tolist(),
        strict=True,
    ):
        if target:
            if not active:
                clusters += 1
                active = True
                cluster_covered = False
            cluster_covered = cluster_covered or override
        elif active:
            covered += int(cluster_covered)
            active = False
    if active:
        covered += int(cluster_covered)
    return clusters, covered


def _early_event_cluster_metrics(
    targets: torch.Tensor,
    overrides: torch.Tensor,
    lead_decisions: torch.Tensor,
) -> tuple[int, int, int, int]:
    """Count early windows and the earliest accepted lead in each one."""

    clusters = 0
    covered = 0
    covered_lead_sum = 0
    maximum_lead = 0
    active = False
    active_leads: list[int] = []
    for target, override, lead in zip(
        targets.to(torch.bool).tolist(),
        overrides.to(torch.bool).tolist(),
        lead_decisions.to(torch.int64).tolist(),
        strict=True,
    ):
        if target:
            if not active:
                clusters += 1
                active = True
                active_leads = []
            if override:
                active_leads.append(int(lead))
        elif active:
            if active_leads:
                earliest = max(active_leads)
                covered += 1
                covered_lead_sum += earliest
                maximum_lead = max(maximum_lead, earliest)
            active = False
            active_leads = []
    if active and active_leads:
        earliest = max(active_leads)
        covered += 1
        covered_lead_sum += earliest
        maximum_lead = max(maximum_lead, earliest)
    return clusters, covered, covered_lead_sum, maximum_lead


def _metrics(
    predictions: dict[int, dict[str, torch.Tensor]],
    episodes: list[EpisodeFeatures],
    runtime: ResidualRuntimeConfig,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    totals = {
        "decisions": 0,
        "reliable_negatives": 0,
        "hard_positives": 0,
        "correctable_hard_positives": 0,
        "uncorrectable_hard_positives": 0,
        "anticipatory_targets": 0,
        "early_anticipatory_targets": 0,
        "overrides": 0,
        "false_overrides": 0,
        "positive_overrides": 0,
        "safe_positive_overrides": 0,
        "unsafe_overrides": 0,
        "hard_positive_overrides": 0,
        "safe_hard_positive_overrides": 0,
        "anticipatory_overrides": 0,
        "safe_anticipatory_overrides": 0,
        "early_anticipatory_overrides": 0,
        "preferred_action_targets": 0,
        "preferred_action_overrides": 0,
        "equivalent_action_targets": 0,
        "equivalent_action_overrides": 0,
        "nonpreferred_overrides": 0,
        "non_equivalent_overrides": 0,
        "raw_candidate_changes": 0,
        "raw_safe_candidate_changes": 0,
        "raw_preferred_candidate_changes": 0,
        "raw_equivalent_candidate_changes": 0,
        "legacy_requests": 0,
        "legacy_accepted_overrides": 0,
        "critic_requests": 0,
        "critic_accepted_overrides": 0,
        "future_onset_requests": 0,
        "future_onset_accepted_overrides": 0,
        "legacy_only_overrides": 0,
        "critic_only_overrides": 0,
        "critic_future_overlap": 0,
        "beneficial_current_only_overrides": 0,
        "legacy_critic_overlap": 0,
        "candidate_safety_vetoes": 0,
        "predicted_parent_danger": 0,
        "predicted_candidate_safe": 0,
        "safe_previous_candidates": 0,
        "safe_previous_overrides": 0,
        "safety_critic_opportunities": 0,
        "early_anticipatory_opportunities": 0,
        "beneficial_overrides": 0,
        "early_beneficial_overrides": 0,
        "unbeneficial_overrides": 0,
        "danger_event_clusters": 0,
        "covered_danger_event_clusters": 0,
        "early_danger_event_clusters": 0,
        "early_covered_danger_event_clusters": 0,
        "covered_early_lead_decisions_sum": 0,
        "covered_early_lead_decisions_maximum": 0,
    }
    for bucket_name, _minimum, _maximum in ANTICIPATORY_LEAD_BUCKETS:
        totals[f"anticipatory_targets_lead_{bucket_name}"] = 0
        totals[f"anticipatory_overrides_lead_{bucket_name}"] = 0
    for episode in episodes:
        values = predictions[episode.seed]
        masks = _override_masks(values, episode, runtime)
        candidates = masks["correction_actions"]
        override = masks["override"]
        reliable_negatives = episode.gate_valid & (episode.gate_targets == 0.0)
        positive = episode.gate_valid & (episode.gate_targets > 0.0)
        hard = episode.hard_positive
        correctable_hard = getattr(
            episode,
            "correctable_hard_positive",
            hard,
        )
        anticipatory = episode.anticipatory
        anticipatory_lead = getattr(
            episode,
            "anticipatory_lead_decisions",
            torch.zeros_like(episode.parent_actions),
        )
        early_anticipatory = (
            anticipatory
            & (anticipatory_lead >= EARLY_ONSET_MINIMUM_LEAD_DECISIONS)
            & (anticipatory_lead <= FUTURE_ONSET_HORIZON_DECISIONS)
        )
        preferred_actions, preferred_targets, equivalent_actions = (
            _preferred_equivalence(episode, positive)
        )
        preferred_selected = preferred_targets & (candidates == preferred_actions)
        equivalent_selected = equivalent_actions.gather(
            -1,
            candidates.unsqueeze(-1),
        ).squeeze(-1)
        safe_selected = episode.safe_actions.gather(
            -1, candidates.unsqueeze(-1),
        ).squeeze(-1)
        evaluation_safe_actions = getattr(
            episode,
            "evaluation_safe_actions",
            episode.safe_actions,
        )
        evaluation_safe_selected = evaluation_safe_actions.gather(
            -1,
            candidates.unsqueeze(-1),
        ).squeeze(-1)
        parent_evaluation_danger = getattr(
            episode,
            "parent_evaluation_danger",
            positive,
        )
        current_critic_opportunity = (
            parent_evaluation_danger
            & positive
            & evaluation_safe_selected
            & equivalent_selected
            & (candidates != episode.parent_actions)
        )
        raw_changes = candidates != episode.parent_actions
        anticipatory_opportunity = (
            anticipatory
            & evaluation_safe_selected
            & equivalent_selected
            & raw_changes
        )
        critic_opportunity = (
            current_critic_opportunity | anticipatory_opportunity
            if runtime.future_onset_gate_enabled else
            current_critic_opportunity
        )
        early_anticipatory_opportunity = (
            early_anticipatory & anticipatory_opportunity
        )
        danger_clusters, covered_danger_clusters = _event_cluster_counts(
            hard | anticipatory,
            override,
        )
        (
            early_clusters,
            covered_early_clusters,
            covered_early_lead_sum,
            covered_early_lead_maximum,
        ) = _early_event_cluster_metrics(
            early_anticipatory,
            override,
            anticipatory_lead,
        )
        legacy_only = masks["legacy_accepted"] & ~masks["critic_accepted"]
        critic_only = (
            masks["critic_accepted"]
            & ~masks["legacy_accepted"]
            & ~masks["future_onset_accepted"]
        )
        critic_future_overlap = (
            masks["critic_accepted"] & masks["future_onset_accepted"]
        )
        overlap = masks["legacy_accepted"] & masks["critic_accepted"]
        episode_metrics = {
            "decisions": episode.decisions,
            "reliable_negatives": int(reliable_negatives.sum()),
            "hard_positives": int(hard.sum()),
            "correctable_hard_positives": int(correctable_hard.sum()),
            "uncorrectable_hard_positives": int(
                (hard & ~correctable_hard).sum()
            ),
            "anticipatory_targets": int(anticipatory.sum()),
            "early_anticipatory_targets": int(early_anticipatory.sum()),
            "teacher_selected_collision_rows_dropped": int(
                episode.teacher_selected_collision.sum()
            ),
            "overrides": int(override.sum()),
            "false_overrides": int((override & ~positive).sum()),
            "positive_overrides": int((override & positive).sum()),
            "safe_positive_overrides": int(
                (override & positive & safe_selected).sum()
            ),
            "unsafe_overrides": int((override & ~safe_selected).sum()),
            "hard_positive_overrides": int((override & hard).sum()),
            "safe_hard_positive_overrides": int(
                (override & hard & safe_selected).sum()
            ),
            "anticipatory_overrides": int((override & anticipatory).sum()),
            "safe_anticipatory_overrides": int(
                (override & anticipatory & safe_selected).sum()
            ),
            "early_anticipatory_overrides": int(
                (override & early_anticipatory).sum()
            ),
            "preferred_action_targets": int(preferred_targets.sum()),
            "preferred_action_overrides": int(
                (override & preferred_selected).sum()
            ),
            "equivalent_action_targets": int(preferred_targets.sum()),
            "equivalent_action_overrides": int(
                (override & equivalent_selected).sum()
            ),
            "nonpreferred_overrides": int(
                (override & ~preferred_selected).sum()
            ),
            "non_equivalent_overrides": int(
                (override & ~equivalent_selected).sum()
            ),
            "raw_candidate_changes": int(raw_changes.sum()),
            "raw_safe_candidate_changes": int(
                (raw_changes & safe_selected).sum()
            ),
            "raw_preferred_candidate_changes": int(
                (raw_changes & preferred_selected).sum()
            ),
            "raw_equivalent_candidate_changes": int(
                (raw_changes & equivalent_selected).sum()
            ),
            "legacy_requests": int(masks["legacy_request"].sum()),
            "legacy_accepted_overrides": int(
                masks["legacy_accepted"].sum()
            ),
            "critic_requests": int(masks["critic_request"].sum()),
            "critic_accepted_overrides": int(
                masks["critic_accepted"].sum()
            ),
            "future_onset_requests": int(
                masks["future_onset_request"].sum()
            ),
            "future_onset_accepted_overrides": int(
                masks["future_onset_accepted"].sum()
            ),
            "legacy_only_overrides": int(legacy_only.sum()),
            "critic_only_overrides": int(critic_only.sum()),
            "critic_future_overlap": int(critic_future_overlap.sum()),
            "beneficial_current_only_overrides": int(
                (critic_only & current_critic_opportunity).sum()
            ),
            "legacy_critic_overlap": int(overlap.sum()),
            "candidate_safety_vetoes": int(
                masks["candidate_safety_veto"].sum()
            ),
            "predicted_parent_danger": int(masks["parent_danger"].sum()),
            "predicted_candidate_safe": int(masks["candidate_safe"].sum()),
            "safe_previous_candidates": int(masks["used_previous"].sum()),
            "safe_previous_overrides": int(
                (masks["used_previous"] & override).sum()
            ),
            "safety_critic_opportunities": int(critic_opportunity.sum()),
            "early_anticipatory_opportunities": int(
                early_anticipatory_opportunity.sum()
            ),
            "beneficial_overrides": int(
                (override & critic_opportunity).sum()
            ),
            "early_beneficial_overrides": int(
                (override & early_anticipatory_opportunity).sum()
            ),
            "unbeneficial_overrides": int(
                (override & ~critic_opportunity).sum()
            ),
            "danger_event_clusters": danger_clusters,
            "covered_danger_event_clusters": covered_danger_clusters,
            "early_danger_event_clusters": early_clusters,
            "early_covered_danger_event_clusters": covered_early_clusters,
            "covered_early_lead_decisions_sum": covered_early_lead_sum,
            "covered_early_lead_decisions_maximum": (
                covered_early_lead_maximum
            ),
        }
        for bucket_name, minimum_lead, maximum_lead in ANTICIPATORY_LEAD_BUCKETS:
            bucket = (
                anticipatory
                & (anticipatory_lead >= minimum_lead)
                & (anticipatory_lead <= maximum_lead)
            )
            episode_metrics[f"anticipatory_targets_lead_{bucket_name}"] = int(
                bucket.sum()
            )
            episode_metrics[f"anticipatory_overrides_lead_{bucket_name}"] = int(
                (override & bucket).sum()
            )
        result[str(episode.seed)] = episode_metrics
        for name in totals:
            if name == "covered_early_lead_decisions_maximum":
                totals[name] = max(totals[name], int(episode_metrics[name]))
            else:
                totals[name] += int(episode_metrics[name])
    totals["hard_positive_recall"] = (
        totals["hard_positive_overrides"] / totals["hard_positives"]
        if totals["hard_positives"] else 0.0
    )
    totals["anticipatory_recall"] = (
        totals["anticipatory_overrides"] / totals["anticipatory_targets"]
        if totals["anticipatory_targets"] else 0.0
    )
    totals["early_anticipatory_recall"] = (
        totals["early_anticipatory_overrides"]
        / totals["early_anticipatory_targets"]
        if totals["early_anticipatory_targets"] else 0.0
    )
    totals["preferred_action_override_rate"] = (
        totals["preferred_action_overrides"] / totals["positive_overrides"]
        if totals["positive_overrides"] else 0.0
    )
    totals["equivalent_action_override_rate"] = (
        totals["equivalent_action_overrides"] / totals["positive_overrides"]
        if totals["positive_overrides"] else 0.0
    )
    totals["safe_override_precision_on_hard_positives"] = (
        totals["safe_hard_positive_overrides"]
        / totals["hard_positive_overrides"]
        if totals["hard_positive_overrides"] else 0.0
    )
    totals["safe_override_precision"] = (
        totals["safe_positive_overrides"] / totals["overrides"]
        if totals["overrides"] else 0.0
    )
    totals["raw_candidate_safety_rate"] = (
        totals["raw_safe_candidate_changes"] / totals["raw_candidate_changes"]
        if totals["raw_candidate_changes"] else 0.0
    )
    totals["raw_candidate_preferred_rate"] = (
        totals["raw_preferred_candidate_changes"]
        / totals["raw_candidate_changes"]
        if totals["raw_candidate_changes"] else 0.0
    )
    totals["raw_candidate_equivalent_rate"] = (
        totals["raw_equivalent_candidate_changes"]
        / totals["raw_candidate_changes"]
        if totals["raw_candidate_changes"] else 0.0
    )
    totals["danger_event_cluster_recall"] = (
        totals["covered_danger_event_clusters"] / totals["danger_event_clusters"]
        if totals["danger_event_clusters"] else 0.0
    )
    totals["safety_critic_opportunity_recall"] = (
        totals["beneficial_overrides"] / totals["safety_critic_opportunities"]
        if totals["safety_critic_opportunities"] else 0.0
    )
    totals["early_anticipatory_opportunity_recall"] = (
        totals["early_beneficial_overrides"]
        / totals["early_anticipatory_opportunities"]
        if totals["early_anticipatory_opportunities"] else 0.0
    )
    totals["early_danger_event_cluster_recall"] = (
        totals["early_covered_danger_event_clusters"]
        / totals["early_danger_event_clusters"]
        if totals["early_danger_event_clusters"] else 0.0
    )
    totals["mean_covered_early_lead_decisions"] = (
        totals["covered_early_lead_decisions_sum"]
        / totals["early_covered_danger_event_clusters"]
        if totals["early_covered_danger_event_clusters"] else 0.0
    )
    for bucket_name, _minimum_lead, _maximum_lead in ANTICIPATORY_LEAD_BUCKETS:
        targets = totals[f"anticipatory_targets_lead_{bucket_name}"]
        overrides = totals[f"anticipatory_overrides_lead_{bucket_name}"]
        totals[f"anticipatory_recall_lead_{bucket_name}"] = (
            overrides / targets if targets else 0.0
        )
    result["total"] = totals
    result["runtime_config"] = asdict(runtime)
    return result


def _offline_deployment_eligible(metrics: dict[str, Any]) -> bool:
    total = metrics["total"]
    if metrics.get("runtime_config", {}).get("future_onset_gate_enabled") is True:
        return bool(
            total["unbeneficial_overrides"] == 0
            and total["early_beneficial_overrides"] > 0
            and total["false_overrides"] == 0
            and total["unsafe_overrides"] == 0
            and total["non_equivalent_overrides"] == 0
        )
    if metrics.get("runtime_config", {}).get("critic_enabled") is True:
        return bool(
            total["unbeneficial_overrides"] == 0
            and total["beneficial_overrides"] > 0
            and total["false_overrides"] == 0
            and total["unsafe_overrides"] == 0
            and total["non_equivalent_overrides"] == 0
        )
    return bool(
        total["false_overrides"] == 0
        and total["unsafe_overrides"] == 0
        and total["non_equivalent_overrides"] == 0
        and (
            total["safe_hard_positive_overrides"] > 0
            or total["safe_anticipatory_overrides"] > 0
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument(
        "--frozen-adapter-weights",
        type=Path,
        help=(
            "reuse an existing adapter's immutable weights and feature "
            "normalization, training only new fail-closed runtime thresholds"
        ),
    )
    parser.add_argument(
        "--resume-fit-checkpoint",
        type=Path,
        help=(
            "restore an uncalibrated fit checkpoint and rerun only episode "
            "prediction, calibration, and audit"
        ),
    )
    parser.add_argument(
        "--fit-checkpoint-output",
        type=Path,
        help=(
            "atomically save fitted adapter weights and normalization before "
            "calibration; this is not a deployment artifact"
        ),
    )
    parser.add_argument(
        "--calibration-diagnostics",
        type=Path,
        help=(
            "write read-only train/calibration funnel diagnostics when "
            "fail-closed calibration has no admissible runtime"
        ),
    )
    parser.add_argument("--dagger-dataset", type=Path, action="append", required=True)
    parser.add_argument("--dagger-report", type=Path, action="append", required=True)
    parser.add_argument("--dagger-manifest", type=Path, action="append", required=True)
    parser.add_argument("--calibration-seed", type=int, action="append", required=True)
    parser.add_argument("--validation-seed", type=int, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--chunk-length", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--ensemble-size", type=int, default=3)
    parser.add_argument(
        "--action-logit-mode",
        choices=(
            "absolute",
            "parent_residual_joint",
            "parent_residual_factorized",
            "certified_membership",
        ),
        default="absolute",
        help=(
            "interpret the learned action output as absolute logits, as a "
            "joint/factorized correction to the frozen parent logits, or as "
            "independent certified-action membership logits"
        ),
    )
    parser.add_argument(
        "--semantic-player-position",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "append normalized player coordinates decoded only from global "
            "semantic channel four to the residual feature stream"
        ),
    )
    parser.add_argument(
        "--separate-action-recurrent",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "give the future preferred-action head an independent projection "
            "and GRU instead of sharing the gate/physical-safety recurrent"
        ),
    )
    parser.add_argument(
        "--per-action-membership-confidence",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "train an independent certified-membership confidence head on a "
            "detached selector latent"
        ),
    )
    parser.add_argument("--safe-regret", type=float, default=1.0)
    parser.add_argument("--minimum-parent-margin", type=float, default=8.0)
    parser.add_argument("--minimum-margin-gain", type=float, default=1.0)
    parser.add_argument("--predecessor-decisions", type=int, default=3)
    parser.add_argument("--gate-positive-weight", type=float, default=4.0)
    parser.add_argument("--action-loss-weight", type=float, default=2.0)
    parser.add_argument("--preferred-action-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--membership-loss-mode",
        choices=MEMBERSHIP_LOSS_MODES,
        default="balanced",
        help=(
            "balance certified and rejected actions within each row, or use "
            "ordinary unweighted per-action BCE; only applicable to "
            "certified_membership logits"
        ),
    )
    parser.add_argument(
        "--membership-confidence-loss-weight",
        type=float,
        default=0.0,
        help=(
            "weight for the independently optimized per-action membership "
            "confidence head; must be positive when that head is enabled"
        ),
    )
    parser.add_argument(
        "--membership-confidence-loss-mode",
        choices=MEMBERSHIP_CONFIDENCE_LOSS_MODES,
        default="unweighted",
        help=(
            "balanced or ordinary unweighted BCE for the detached confidence "
            "head, independent of --membership-loss-mode"
        ),
    )
    parser.add_argument(
        "--preferred-action-uniform-loss-weight",
        type=float,
        default=0.0,
        help=(
            "optional training-only KL from a uniform distribution over each "
            "multi-member certified preferred-action set"
        ),
    )
    parser.add_argument(
        "--preferred-action-tiebreak-loss-weight",
        type=float,
        default=0.0,
        help=(
            "optional training-only continuity loss toward the row-local previous "
            "action when it belongs to a multi-member certified set"
        ),
    )
    parser.add_argument(
        "--preferred-action-rank-loss-weight",
        type=float,
        default=0.0,
        help=(
            "optional hinge weight requiring one certified preferred action to "
            "outrank every rejected action"
        ),
    )
    parser.add_argument("--preferred-action-rank-margin", type=float, default=1.0)
    parser.add_argument("--safety-candidate-loss-weight", type=float, default=2.0)
    parser.add_argument("--parent-copy-weight", type=float, default=0.05)
    parser.add_argument(
        "--per-action-safety-critic",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "train version-3 dense collision and minimum-margin heads and "
            "calibrate their fail-closed runtime"
        ),
    )
    parser.add_argument("--collision-loss-weight", type=float, default=1.0)
    parser.add_argument("--minimum-margin-loss-weight", type=float, default=0.5)
    parser.add_argument(
        "--direct-visual-physical-critic",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "train a version-4 direct clearance/collision danger head from the "
            "frozen parent's current visual latent plus recurrent state"
        ),
    )
    parser.add_argument(
        "--future-onset-gate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "train a version-5 binary 10-decision future-danger onset gate; "
            "requires --direct-visual-physical-critic"
        ),
    )
    parser.add_argument("--physical-danger-loss-weight", type=float, default=2.0)
    parser.add_argument(
        "--maximum-physical-danger-positive-weight",
        type=float,
        default=24.0,
    )
    parser.add_argument(
        "--maximum-collision-positive-weight",
        type=float,
        default=24.0,
    )
    parser.add_argument("--all-collision-row-weight", type=float, default=0.25)
    parser.add_argument(
        "--episode-bootstrap",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "sample source episodes with replacement per ensemble member; "
            "disabled by default so every member sees every scarce success"
        ),
    )
    args = parser.parse_args()

    if args.epochs <= 0 or args.chunk_length <= 0:
        raise ValueError("epochs and chunk length must be positive")
    if (
        args.safe_regret < 0.0
        or args.minimum_parent_margin < 0.0
        or args.minimum_margin_gain < 0.0
    ):
        raise ValueError("evidence thresholds must be nonnegative")
    if not 0 <= args.predecessor_decisions <= 12:
        raise ValueError("predecessor decisions must be in [0, 12]")
    _validate_training_loss_weights(
        gate_positive_weight=args.gate_positive_weight,
        action_loss_weight=args.action_loss_weight,
        preferred_action_loss_weight=args.preferred_action_loss_weight,
        membership_confidence_loss_weight=(
            args.membership_confidence_loss_weight
        ),
        preferred_action_uniform_loss_weight=(
            args.preferred_action_uniform_loss_weight
        ),
        preferred_action_tiebreak_loss_weight=(
            args.preferred_action_tiebreak_loss_weight
        ),
        preferred_action_rank_loss_weight=args.preferred_action_rank_loss_weight,
        safety_candidate_loss_weight=args.safety_candidate_loss_weight,
        parent_copy_weight=args.parent_copy_weight,
        collision_loss_weight=args.collision_loss_weight,
        minimum_margin_loss_weight=args.minimum_margin_loss_weight,
        physical_danger_loss_weight=args.physical_danger_loss_weight,
    )
    if not math.isfinite(args.preferred_action_rank_margin) or (
        args.preferred_action_rank_margin < 0.0
    ):
        raise ValueError(
            "preferred action rank margin must be finite and nonnegative"
        )
    _validate_action_training_semantics(
        action_logit_mode=args.action_logit_mode,
        action_loss_weight=args.action_loss_weight,
        parent_copy_weight=args.parent_copy_weight,
        preferred_action_uniform_loss_weight=(
            args.preferred_action_uniform_loss_weight
        ),
        preferred_action_tiebreak_loss_weight=(
            args.preferred_action_tiebreak_loss_weight
        ),
        preferred_action_rank_loss_weight=(
            args.preferred_action_rank_loss_weight
        ),
        safety_candidate_loss_weight=args.safety_candidate_loss_weight,
        membership_loss_mode=args.membership_loss_mode,
    )
    _validate_membership_confidence_training_semantics(
        enabled=args.per_action_membership_confidence,
        action_logit_mode=args.action_logit_mode,
        loss_weight=args.membership_confidence_loss_weight,
        loss_mode=args.membership_confidence_loss_mode,
    )
    if (
        not math.isfinite(args.maximum_collision_positive_weight)
        or args.maximum_collision_positive_weight < 1.0
    ):
        raise ValueError("maximum collision positive weight must be at least one")
    if (
        not math.isfinite(args.maximum_physical_danger_positive_weight)
        or args.maximum_physical_danger_positive_weight < 1.0
    ):
        raise ValueError(
            "maximum physical danger positive weight must be at least one"
        )
    if not 0.0 < args.all_collision_row_weight <= 1.0:
        raise ValueError("all-collision row weight must be in (0, 1]")
    if args.future_onset_gate and not args.direct_visual_physical_critic:
        raise ValueError(
            "--future-onset-gate requires --direct-visual-physical-critic"
        )
    effective_predecessor_decisions = (
        FUTURE_ONSET_HORIZON_DECISIONS
        if args.future_onset_gate else
        args.predecessor_decisions
    )
    if (
        args.frozen_adapter_weights is not None
        and args.resume_fit_checkpoint is not None
    ):
        raise ValueError(
            "--frozen-adapter-weights and --resume-fit-checkpoint are mutually exclusive"
        )
    source_triplets = _source_triplets(args)
    diagnostic_output = (
        args.calibration_diagnostics
        if args.calibration_diagnostics is not None else
        args.report.with_name(f"{args.report.stem}-calibration-failure.json")
    )
    _validate_distinct_workflow_paths(
        outputs={
            "deployment output": args.output,
            "report output": args.report,
            "fit checkpoint output": args.fit_checkpoint_output,
            "calibration diagnostics": diagnostic_output,
        },
        protected_inputs={
            "parent checkpoint": args.parent,
            "frozen adapter": args.frozen_adapter_weights,
            "resume fit checkpoint": args.resume_fit_checkpoint,
        },
        source_paths=[path for triplet in source_triplets for path in triplet],
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    parent, checkpoint = load_checkpoint(args.parent, device=args.device)
    parent_sha256 = file_sha256(args.parent)
    requested_adapter_config = ResidualAdapterConfig(
        recurrent_size=parent.config.recurrent_size,
        action_count=parent.config.action_count,
        hidden_size=args.hidden_size,
        ensemble_size=args.ensemble_size,
        executed_action_context=True,
        per_action_safety_critic=(
            args.per_action_safety_critic
            or args.direct_visual_physical_critic
        ),
        visual_latent_size=(
            parent.config.feature_size * 2
            if args.direct_visual_physical_critic else
            0
        ),
        per_action_physical_danger=args.direct_visual_physical_critic,
        action_logit_mode=args.action_logit_mode,
        semantic_player_position=args.semantic_player_position,
        separate_action_recurrent=args.separate_action_recurrent,
        per_action_membership_confidence=(
            args.per_action_membership_confidence
        ),
    )
    frozen_adapter_metadata: dict[str, Any] | None = None
    resumed_fit_metadata: dict[str, Any] | None = None
    if args.resume_fit_checkpoint is not None:
        adapter, resumed_fit_metadata = _load_fit_checkpoint(
            args.resume_fit_checkpoint,
            parent_checkpoint=args.parent,
            parent_policy_config=asdict(parent.config),
            expected_adapter_config=requested_adapter_config,
            device=args.device,
        )
    elif args.frozen_adapter_weights is not None:
        wrapper, frozen_adapter_metadata = load_residual_adapter(
            parent,
            args.frozen_adapter_weights,
            parent_checkpoint=args.parent,
            device=args.device,
        )
        adapter = wrapper.adapter
        if asdict(adapter.config) != asdict(requested_adapter_config):
            raise ValueError(
                "frozen adapter config does not match the requested training "
                "and runtime semantics"
            )
        if not adapter.config.executed_action_context:
            raise ValueError(
                "frozen recalibration requires executed-action context weights"
            )
        source_runtime = frozen_adapter_metadata.get("runtime_config", {})
        if bool(source_runtime.get("future_onset_gate_enabled", False)) != bool(
            args.future_onset_gate
        ):
            raise ValueError(
                "--future-onset-gate must match the frozen adapter runtime mode"
            )
        source_training = frozen_adapter_metadata.get("training_metadata", {})
        if args.future_onset_gate and source_training.get(
            "preferred_action_target_semantics"
        ) != PREFERRED_ACTION_TARGET_SEMANTICS:
            raise ValueError(
                "frozen future-onset adapter preferred-action target semantics "
                "do not match the requested certified equivalence set"
            )
        if args.future_onset_gate and source_training.get(
            "future_onset_horizon_decisions"
        ) != FUTURE_ONSET_HORIZON_DECISIONS:
            raise ValueError(
                "frozen future-onset adapter must use the 10-decision horizon"
            )
        for argument_name, metadata_name, actual in (
            ("safe_regret", "safe_regret", args.safe_regret),
            (
                "minimum_parent_margin",
                "minimum_parent_margin",
                args.minimum_parent_margin,
            ),
            (
                "minimum_margin_gain",
                "minimum_margin_gain",
                args.minimum_margin_gain,
            ),
            (
                "predecessor_decisions",
                "predecessor_decisions",
                effective_predecessor_decisions,
            ),
        ):
            expected = source_training.get(metadata_name)
            if expected is None or actual != expected:
                raise ValueError(
                    f"--{argument_name.replace('_', '-')} must match the "
                    f"frozen adapter label value {expected!r}"
                )
    else:
        adapter = ResidualCorrectionAdapter(requested_adapter_config)
    restored_weight_metadata = (
        resumed_fit_metadata
        if resumed_fit_metadata is not None else
        frozen_adapter_metadata
    )
    if restored_weight_metadata is not None:
        restored_training_metadata = restored_weight_metadata.get(
            "training_metadata"
        )
        if not isinstance(restored_training_metadata, Mapping):
            raise ValueError("restored adapter training metadata is invalid")
        _validate_restored_membership_loss_mode(
            restored_training_metadata,
            action_logit_mode=adapter.config.action_logit_mode,
            requested_mode=args.membership_loss_mode,
        )
        _validate_restored_membership_confidence_training(
            restored_training_metadata,
            enabled=adapter.config.per_action_membership_confidence,
            action_logit_mode=adapter.config.action_logit_mode,
            requested_weight=args.membership_confidence_loss_weight,
            requested_mode=args.membership_confidence_loss_mode,
        )
    episodes = [
        _load_episode(
            parent,
            adapter,
            dataset,
            report,
            manifest,
            parent_checkpoint_sha256=parent_sha256,
            device=args.device,
            chunk_length=256,
            safe_regret=args.safe_regret,
            minimum_parent_margin=args.minimum_parent_margin,
            minimum_margin_gain=args.minimum_margin_gain,
            predecessor_decisions=effective_predecessor_decisions,
            future_onset_gate=args.future_onset_gate,
        )
        for dataset, report, manifest in source_triplets
    ]
    seeds = [episode.seed for episode in episodes]
    if len(set(seeds)) != len(seeds):
        raise ValueError("DAgger source seeds must be unique")
    calibration_seeds = set(args.calibration_seed)
    validation_seeds = set(args.validation_seed)
    if calibration_seeds & validation_seeds:
        raise ValueError("calibration and validation seeds must be disjoint")
    calibration = [
        episode for episode in episodes if episode.seed in calibration_seeds
    ]
    validation = [episode for episode in episodes if episode.seed in validation_seeds]
    held_out_seeds = calibration_seeds | validation_seeds
    training = [episode for episode in episodes if episode.seed not in held_out_seeds]
    if {episode.seed for episode in calibration} != calibration_seeds:
        raise ValueError("every calibration seed must identify one DAgger source")
    if {episode.seed for episode in validation} != validation_seeds:
        raise ValueError("every validation seed must identify one DAgger source")
    if len(training) < 2 or not calibration or not validation:
        raise ValueError(
            "temporal residual training needs two train sources plus separate "
            "calibration and validation sources"
        )
    source_inventory = _fit_source_inventory(
        episodes,
        calibration_seeds=calibration_seeds,
        validation_seeds=validation_seeds,
    )
    fit_label_metadata = {
        "safe_regret": args.safe_regret,
        "minimum_parent_margin": args.minimum_parent_margin,
        "minimum_margin_gain": args.minimum_margin_gain,
        "predecessor_decisions": effective_predecessor_decisions,
        "future_onset_gate": args.future_onset_gate,
        "future_onset_horizon_decisions": (
            FUTURE_ONSET_HORIZON_DECISIONS
            if args.future_onset_gate else
            None
        ),
        "future_onset_binary_target": bool(args.future_onset_gate),
        "right_censor_incomplete_negative_tail": bool(args.future_onset_gate),
        "preferred_action_target_semantics": (
            PREFERRED_ACTION_TARGET_SEMANTICS
        ),
        "preferred_correction_required": (
            "positive_and_unique_preferred_differs_from_parent"
        ),
    }
    if adapter.config.per_action_membership_confidence:
        fit_label_metadata.update({
            "membership_confidence_target": (
                "preferred_action_set_on_valid_positive_rows"
            ),
            "membership_confidence_action_cells": 18,
        })
    if resumed_fit_metadata is not None:
        _validate_fit_resume_metadata(
            resumed_fit_metadata,
            source_inventory=source_inventory,
            label_metadata=fit_label_metadata,
        )

    # New weights fit normalization only on their training episodes. Frozen
    # weights must retain the exact normalization they were trained against.
    histories: list[list[dict[str, float]]]
    collision_positive_weights: torch.Tensor | None = None
    physical_danger_positive_weights: torch.Tensor | None = None
    if frozen_adapter_metadata is None and resumed_fit_metadata is None:
        _normalize(adapter, training, episodes)
        if adapter.config.per_action_safety_critic:
            collision_positive_weights = _collision_positive_weights(
                training,
                maximum_weight=args.maximum_collision_positive_weight,
            )
        if adapter.config.per_action_physical_danger:
            physical_danger_positive_weights = _physical_danger_positive_weights(
                training,
                maximum_weight=args.maximum_physical_danger_positive_weight,
            )
        histories = []
        for member_index in range(args.ensemble_size):
            histories.append(_train_member(
                adapter,
                member_index,
                training,
                seed=args.seed + member_index * 1009,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                chunk_length=args.chunk_length,
                gate_positive_weight=args.gate_positive_weight,
                action_loss_weight=args.action_loss_weight,
                parent_copy_weight=args.parent_copy_weight,
                device=args.device,
                preferred_action_loss_weight=args.preferred_action_loss_weight,
                membership_loss_mode=args.membership_loss_mode,
                membership_confidence_loss_weight=(
                    args.membership_confidence_loss_weight
                ),
                membership_confidence_loss_mode=(
                    args.membership_confidence_loss_mode
                ),
                preferred_action_uniform_loss_weight=(
                    args.preferred_action_uniform_loss_weight
                ),
                preferred_action_tiebreak_loss_weight=(
                    args.preferred_action_tiebreak_loss_weight
                ),
                preferred_action_rank_loss_weight=(
                    args.preferred_action_rank_loss_weight
                ),
                preferred_action_rank_margin=args.preferred_action_rank_margin,
                safety_candidate_loss_weight=args.safety_candidate_loss_weight,
                collision_loss_weight=args.collision_loss_weight,
                minimum_margin_loss_weight=args.minimum_margin_loss_weight,
                physical_danger_loss_weight=args.physical_danger_loss_weight,
                collision_positive_weights=collision_positive_weights,
                physical_danger_positive_weights=(
                    physical_danger_positive_weights
                ),
                all_collision_row_weight=args.all_collision_row_weight,
                episode_bootstrap=args.episode_bootstrap,
            ))
    else:
        _apply_existing_normalization(adapter, episodes)
        weight_metadata = (
            resumed_fit_metadata
            if resumed_fit_metadata is not None else
            frozen_adapter_metadata
        )
        assert weight_metadata is not None
        source_training = weight_metadata.get("training_metadata", {})
        source_histories = source_training.get("member_histories", [])
        if not isinstance(source_histories, list):
            raise ValueError("restored adapter has invalid member training history")
        histories = source_histories
    adapter.to(args.device).eval()
    if resumed_fit_metadata is not None:
        fit_checkpoint_training_metadata = dict(
            resumed_fit_metadata["training_metadata"]
        )
        fit_checkpoint_training_metadata["resumed_from"] = {
            "path": str(args.resume_fit_checkpoint),
            "sha256": resumed_fit_metadata["fit_checkpoint_sha256"],
            "weights_unchanged": True,
        }
    else:
        fit_checkpoint_training_metadata = {
            "run_kind": "temporal_residual_weight_fit",
            "acceptance_claim": False,
            "calibration_complete": False,
            "parent_parameters_frozen": True,
            "parent_checkpoint_sha256": parent_sha256,
            "source_inventory": source_inventory,
            "label_metadata": fit_label_metadata,
            "member_histories": histories,
            "training_controls": {
                "seed": args.seed,
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "chunk_length": args.chunk_length,
                "gate_positive_weight": args.gate_positive_weight,
                "action_loss_weight": args.action_loss_weight,
                "preferred_action_loss_weight": (
                    args.preferred_action_loss_weight
                ),
                **_membership_loss_metadata(
                    action_logit_mode=args.action_logit_mode,
                    membership_loss_mode=args.membership_loss_mode,
                ),
                **_membership_confidence_training_metadata(
                    enabled=adapter.config.per_action_membership_confidence,
                    action_logit_mode=adapter.config.action_logit_mode,
                    loss_weight=args.membership_confidence_loss_weight,
                    loss_mode=args.membership_confidence_loss_mode,
                ),
                "preferred_action_uniform_loss_weight": (
                    args.preferred_action_uniform_loss_weight
                ),
                "preferred_action_uniform_semantics": (
                    PREFERRED_ACTION_UNIFORM_SOFT_TARGET_SEMANTICS
                ),
                "preferred_action_tiebreak_loss_weight": (
                    args.preferred_action_tiebreak_loss_weight
                ),
                "preferred_action_tiebreak_semantics": (
                    PREFERRED_ACTION_TIEBREAK_SEMANTICS
                ),
                "preferred_action_rank_loss_weight": (
                    args.preferred_action_rank_loss_weight
                ),
                "preferred_action_rank_margin": (
                    args.preferred_action_rank_margin
                ),
                "safety_candidate_loss_weight": (
                    args.safety_candidate_loss_weight
                ),
                "parent_copy_weight": args.parent_copy_weight,
                "parent_copy_semantics": (
                    "inapplicable; required to be exactly zero"
                    if args.action_logit_mode == "certified_membership" else
                    "negative-row parent distribution preservation"
                ),
                "action_logit_mode": args.action_logit_mode,
                "semantic_player_position": args.semantic_player_position,
                "separate_action_recurrent": args.separate_action_recurrent,
                "collision_loss_weight": args.collision_loss_weight,
                "minimum_margin_loss_weight": args.minimum_margin_loss_weight,
                "physical_danger_loss_weight": args.physical_danger_loss_weight,
                "all_collision_row_weight": args.all_collision_row_weight,
                "episode_bootstrap": args.episode_bootstrap,
            },
        }
    fit_checkpoint_record: dict[str, Any] | None = None
    if args.fit_checkpoint_output is not None:
        fit_checkpoint_record = _save_fit_checkpoint(
            adapter,
            args.fit_checkpoint_output,
            parent_checkpoint=args.parent,
            parent_policy_config=asdict(parent.config),
            training_metadata=fit_checkpoint_training_metadata,
        )
    elif resumed_fit_metadata is not None:
        fit_checkpoint_record = {
            "fit_checkpoint": str(args.resume_fit_checkpoint),
            "fit_checkpoint_sha256": resumed_fit_metadata[
                "fit_checkpoint_sha256"
            ],
            "kind": FIT_CHECKPOINT_KIND,
            "deployment_artifact": False,
            "resumed_without_weight_changes": True,
        }
    predictions = {
        episode.seed: _predict_episode(adapter, episode, device=args.device)
        for episode in episodes
    }
    try:
        runtime = _calibrate(
            predictions,
            training,
            calibration,
            ensemble_size=adapter.config.ensemble_size,
            per_action_safety_critic=adapter.config.per_action_safety_critic,
            per_action_physical_danger=adapter.config.per_action_physical_danger,
            future_onset_gate=args.future_onset_gate,
        )
    except (ValueError, RuntimeError) as error:
        diagnostics = (
            _future_onset_calibration_diagnostics(
                predictions,
                training,
                calibration,
                ensemble_size=adapter.config.ensemble_size,
            )
            if args.future_onset_gate else
            {
                "schema_version": 1,
                "kind": "residual_calibration_failure_diagnostics",
                "read_only_audit": True,
                "threshold_selection_uses_validation": False,
            }
        )
        diagnostics.update({
            "error_type": type(error).__name__,
            "error": str(error),
            "parent_checkpoint_sha256": parent_sha256,
            "source_inventory": source_inventory,
            "fit_checkpoint": fit_checkpoint_record,
            "formal_deployment_artifact_written": False,
        })
        _write_json_atomic(diagnostic_output, diagnostics)
        raise ValueError(
            f"{error}; calibration diagnostics: {diagnostic_output}"
        ) from error
    training_metrics = _metrics(predictions, training, runtime)
    calibration_metrics = _metrics(predictions, calibration, runtime)
    validation_metrics = _metrics(predictions, validation, runtime)
    offline_deployment_eligible = _offline_deployment_eligible(validation_metrics)
    validation_metrics["offline_deployment_eligible"] = (
        offline_deployment_eligible
    )
    source_metadata = [
        {
            "seed": episode.seed,
            "dataset": episode.dataset,
            "dataset_sha256": file_sha256(episode.dataset),
            "report": episode.report,
            "report_sha256": file_sha256(episode.report),
            "global_frame_dtype": episode.global_frame_dtype,
            "local_frame_dtype": episode.local_frame_dtype,
            "role": (
                "validation"
                if episode.seed in validation_seeds else
                "calibration"
                if episode.seed in calibration_seeds else
                "training"
            ),
        }
        for episode in episodes
    ]
    training_metadata = {
        "run_kind": (
            "temporal_residual_runtime_recalibration"
            if (
                frozen_adapter_metadata is not None
                or resumed_fit_metadata is not None
            ) else
            "temporal_frozen_parent_residual_correction_training"
        ),
        "acceptance_claim": False,
        "parent_parameters_frozen": True,
        "adapter_weights_frozen": (
            frozen_adapter_metadata is not None
            or resumed_fit_metadata is not None
        ),
        "parent_checkpoint_sha256": parent_sha256,
        "strict_success_sources_only": True,
        "teacher_evidence_model_input": False,
        "preferred_action_target_semantics": (
            PREFERRED_ACTION_TARGET_SEMANTICS
        ),
        "excluded_model_inputs": [
            "absolute_frame",
            "script_phase",
            "fixed_route",
            "waypoint",
            "external_region_dynamics_memory",
        ],
        "model_inputs": [
            "frozen_parent_gru_state",
            "frozen_parent_action_logits",
            "frozen_parent_top1_action_token",
            "previous_executed_motor_action_one_hot",
            "previous_executed_action_validity",
            "log1p_consecutive_executed_action_decisions",
            "learned_temporal_residual_hidden_state",
            *(
                [
                    "learned_per_action_collision_scores",
                    "learned_per_action_minimum_margin",
                ]
                if adapter.config.per_action_safety_critic else
                []
            ),
            *(
                [
                    "frozen_parent_current_global_visual_latent",
                    "frozen_parent_current_local_visual_latent",
                    "learned_per_action_physical_danger_score",
                ]
                if adapter.config.per_action_physical_danger else
                []
            ),
            *(
                [
                    "normalized_player_xy_decoded_from_global_semantic_channel_4",
                ]
                if adapter.config.semantic_player_position else
                []
            ),
        ],
        **(
            {
                "auxiliary_model_outputs": [
                    "per_action_certified_membership_confidence",
                ],
                "auxiliary_head_input": "detached_selector_recurrent_latent",
            }
            if adapter.config.per_action_membership_confidence else
            {}
        ),
        "label_rule": {
            "safe_action": (
                "noncolliding and clearance_regret<=safe_regret and same "
                "moving/stationary class as noncolliding selected teacher"
            ),
            "drop": "selected teacher action predicts collision",
            "hard_positive": (
                "parent predicts collision or falls below minimum_parent_margin, "
                "and selected-teacher minus parent minimum-margin gain is at "
                "least minimum_margin_gain"
            ),
            "predecessor_target": (
                "binary target equal to one when a hard-positive danger onset "
                "occurs within the next 10 decisions; propagate the onset's "
                "preferred action only while it is independently noncolliding "
                "and has at least minimum_parent_margin at every predecessor"
                if args.future_onset_gate else
                "discounted gate target plus the future hard-positive onset's "
                "preferred action and cumulative safe action set, re-certified "
                "noncolliding at each predecessor"
            ),
            "multi_action_target": (
                _preferred_action_loss_semantics(
                    action_logit_mode=args.action_logit_mode,
                    membership_loss_mode=args.membership_loss_mode,
                )
                if args.action_logit_mode == "certified_membership" else
                "set-valued safe probability plus collision rank"
            ),
            "preferred_action_target": (
                "derive the unique teacher/previous action only to identify "
                "whether a correction is required; train probability mass on "
                "every non-parent action that remains in the cumulative "
                "safe-regret set and independently clears minimum_parent_margin; "
                "when the parent already equals the unique preferred action, "
                "train a no-change parent target instead"
            ),
            "preferred_equivalence_set": (
                "safe_actions intersect evaluation_safe_actions, excluding "
                "the parent action on required corrections; each action has a "
                "three-control-frame MPC certificate followed by a stationary "
                "forecast through the configured teacher horizon"
            ),
            "preferred_action_tiebreak": (
                "disabled by default; when explicitly weighted, apply a weak "
                "conditional cross-entropy normalized only across members of a "
                "positive certified preferred set having multiple members, toward "
                "the current row's recorded previous action only when that action "
                "belongs to the set; there is no teacher or propagated-preferred "
                "fallback. Rejected-action logits have zero gradient from this "
                "training-only continuity supervision, which adds no runtime hold "
                "logic"
            ),
            "preferred_action_uniform_soft_target": (
                "disabled by default; when explicitly weighted, add KL from a "
                "uniform distribution to the action softmax restricted to each "
                "positive multi-member certified preferred set. Singleton rows "
                "receive zero auxiliary loss, rejected-action logits receive zero "
                "auxiliary gradient, and no unique teacher, previous action, route, "
                "frame, phase, or waypoint target is selected"
            ),
            "safety_candidate_target": (
                "when the parent action predicts collision or low margin, keep "
                "the previous executed action if it remains in the strict safe "
                "set; otherwise use the noncolliding selected teacher action"
            ),
            "physical_danger_target": (
                "per-action collision or finite predicted clearance below "
                "minimum_parent_margin"
            ),
            **(
                {
                    "membership_confidence_target": (
                        "all 18 Boolean cells of preferred_action_set on valid "
                        "positive rows"
                    ),
                }
                if adapter.config.per_action_membership_confidence else
                {}
            ),
        },
        "safe_regret": args.safe_regret,
        "minimum_parent_margin": args.minimum_parent_margin,
        "minimum_margin_gain": args.minimum_margin_gain,
        "predecessor_decisions": effective_predecessor_decisions,
        "future_onset_gate": args.future_onset_gate,
        "future_onset_horizon_decisions": (
            FUTURE_ONSET_HORIZON_DECISIONS
            if args.future_onset_gate else
            None
        ),
        "future_onset_target": (
            {
                "type": "binary_any_onset_within_horizon",
                "horizon_decisions": FUTURE_ONSET_HORIZON_DECISIONS,
                "early_audit_lead_decisions": [
                    EARLY_ONSET_MINIMUM_LEAD_DECISIONS,
                    FUTURE_ONSET_HORIZON_DECISIONS,
                ],
                "early_audit_lead_control_frames": [12, 30],
                "right_censor_negative_episode_tail": True,
                "positive_events_remain_observed_at_tail": True,
                "candidate_recertification": (
                    "current decision noncollision and finite minimum clearance "
                    ">= minimum_parent_margin; future requests never relax the "
                    "runtime physical candidate-safety gate"
                ),
            }
            if args.future_onset_gate else
            None
        ),
        "member_training": (
            "unchanged frozen adapter weights"
            if (
                frozen_adapter_metadata is not None
                or resumed_fit_metadata is not None
            ) else
            "all strict-success episodes per member with distinct initialization "
            "and epoch order"
            if not args.episode_bootstrap else
            "independent episode bootstrap with distinct seeds"
        ),
        "train_seeds": [episode.seed for episode in training],
        "calibration_seeds": [episode.seed for episode in calibration],
        "validation_seeds": [episode.seed for episode in validation],
        "runtime_selection": (
            "thresholds selected on training plus calibration episodes with "
            "zero false, unsafe, non-equivalent, and unbeneficial overrides"
            + (
                ", and at least one beneficial 4-10-decision early override "
                "in each selection split"
                if args.future_onset_gate else
                ""
            )
            + "; validation episodes are audit-only"
        ),
        "offline_metric_semantics": {
            "state_transition": (
                "teacher-forced conditional audit using each DAgger source's "
                "recorded previous executed action"
            ),
            "limitation": (
                "residual overrides can change the subsequent previous action "
                "in live closed loop, so offline multi-decision trajectories "
                "are not claimed statefully equivalent"
            ),
            "final_acceptance": (
                "strict native closed-loop clear on independent validation seeds"
            ),
        },
        "sources": source_metadata,
        "training_controls": {
            "seed": args.seed,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "chunk_length": args.chunk_length,
            "gate_positive_weight": args.gate_positive_weight,
            "action_loss_weight": args.action_loss_weight,
            "preferred_action_loss_weight": args.preferred_action_loss_weight,
            **_membership_loss_metadata(
                action_logit_mode=adapter.config.action_logit_mode,
                membership_loss_mode=args.membership_loss_mode,
            ),
            **_membership_confidence_training_metadata(
                enabled=adapter.config.per_action_membership_confidence,
                action_logit_mode=adapter.config.action_logit_mode,
                loss_weight=args.membership_confidence_loss_weight,
                loss_mode=args.membership_confidence_loss_mode,
            ),
            "preferred_action_uniform_loss_weight": (
                args.preferred_action_uniform_loss_weight
            ),
            "preferred_action_uniform_semantics": (
                PREFERRED_ACTION_UNIFORM_SOFT_TARGET_SEMANTICS
            ),
            "preferred_action_tiebreak_loss_weight": (
                args.preferred_action_tiebreak_loss_weight
            ),
            "preferred_action_tiebreak_semantics": (
                PREFERRED_ACTION_TIEBREAK_SEMANTICS
            ),
            "preferred_action_rank_loss_weight": (
                args.preferred_action_rank_loss_weight
            ),
            "preferred_action_rank_margin": args.preferred_action_rank_margin,
            "safety_candidate_loss_weight": args.safety_candidate_loss_weight,
            "parent_copy_weight": args.parent_copy_weight,
            "parent_copy_semantics": (
                "inapplicable; required to be exactly zero"
                if adapter.config.action_logit_mode
                == "certified_membership" else
                "negative-row parent distribution preservation"
            ),
            "action_logit_mode": adapter.config.action_logit_mode,
            "semantic_player_position": adapter.config.semantic_player_position,
            "separate_action_recurrent": adapter.config.separate_action_recurrent,
            "per_action_safety_critic": (
                adapter.config.per_action_safety_critic
            ),
            "per_action_physical_danger": (
                adapter.config.per_action_physical_danger
            ),
            "future_onset_gate": args.future_onset_gate,
            "visual_latent_size": adapter.config.visual_latent_size,
            "collision_loss_weight": args.collision_loss_weight,
            "minimum_margin_loss_weight": args.minimum_margin_loss_weight,
            "physical_danger_loss_weight": args.physical_danger_loss_weight,
            "maximum_collision_positive_weight": (
                args.maximum_collision_positive_weight
            ),
            "collision_positive_weights": (
                None
                if collision_positive_weights is None else
                collision_positive_weights.tolist()
            ),
            "maximum_physical_danger_positive_weight": (
                args.maximum_physical_danger_positive_weight
            ),
            "physical_danger_positive_weights": (
                None
                if physical_danger_positive_weights is None else
                physical_danger_positive_weights.tolist()
            ),
            "all_collision_row_weight": args.all_collision_row_weight,
            "minimum_margin_transform": "clamp[-64,64]/16",
            "episode_bootstrap": args.episode_bootstrap,
        },
        "member_histories": histories,
        "offline_training_metrics": training_metrics,
        "offline_calibration_metrics": calibration_metrics,
        "offline_validation_metrics": validation_metrics,
    }
    if frozen_adapter_metadata is not None:
        training_metadata["frozen_adapter_weight_source"] = {
            "path": str(args.frozen_adapter_weights),
            "sha256": frozen_adapter_metadata["adapter_sha256"],
            "training_metadata": frozen_adapter_metadata.get("training_metadata"),
        }
        training_metadata["training_controls"] = {
            "recalibration_only": True,
            "weights_unchanged": True,
            "threshold_search_uses_validation": False,
            **_membership_loss_metadata(
                action_logit_mode=adapter.config.action_logit_mode,
                membership_loss_mode=args.membership_loss_mode,
            ),
            **_membership_confidence_training_metadata(
                enabled=adapter.config.per_action_membership_confidence,
                action_logit_mode=adapter.config.action_logit_mode,
                loss_weight=args.membership_confidence_loss_weight,
                loss_mode=args.membership_confidence_loss_mode,
            ),
        }
    elif resumed_fit_metadata is not None:
        training_metadata["fit_checkpoint_weight_source"] = {
            "path": str(args.resume_fit_checkpoint),
            "sha256": resumed_fit_metadata["fit_checkpoint_sha256"],
            "training_metadata": resumed_fit_metadata.get("training_metadata"),
        }
        training_metadata["training_controls"] = {
            "recalibration_only": True,
            "weights_unchanged": True,
            "normalization_unchanged": True,
            "threshold_search_uses_validation": False,
            **_membership_loss_metadata(
                action_logit_mode=adapter.config.action_logit_mode,
                membership_loss_mode=args.membership_loss_mode,
            ),
            **_membership_confidence_training_metadata(
                enabled=adapter.config.per_action_membership_confidence,
                action_logit_mode=adapter.config.action_logit_mode,
                loss_weight=args.membership_confidence_loss_weight,
                loss_mode=args.membership_confidence_loss_mode,
            ),
        }
    if fit_checkpoint_record is not None:
        training_metadata["fit_checkpoint_before_calibration"] = (
            fit_checkpoint_record
        )
    artifact = save_residual_adapter(
        adapter,
        args.output,
        parent_checkpoint=args.parent,
        parent_policy_config=asdict(parent.config),
        runtime_config=runtime,
        training_metadata=training_metadata,
    )
    report = {
        "schema_version": 1,
        **training_metadata,
        "parent_checkpoint": str(args.parent),
        "parent_checkpoint_sha256": parent_sha256,
        "parent_policy_config": checkpoint["policy_config"],
        "runtime_config": asdict(runtime),
        "artifact": artifact,
        "offline_deployment_eligible": offline_deployment_eligible,
        "deployment_accepted": False,
        "deployment_gate": (
            "requires zero validation false/unsafe overrides and strict native "
            "clears on independent validation seeds"
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
