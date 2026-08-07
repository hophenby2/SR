from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
import random
from typing import Any, Iterable, Mapping

import numpy as np

try:
    import torch
    from torch import Tensor
    from torch.nn import functional as F
    from torch.utils.data import DataLoader, Dataset
except ImportError:  # pragma: no cover
    torch = None
    Tensor = object  # type: ignore[assignment,misc]
    Dataset = object  # type: ignore[assignment,misc]

from .policy import (
    PROFICIENCY_VECTOR_SIZE,
    HumanVisionPolicy,
    PolicyConfig,
    proficiency_vector,
)


TEACHER_ACTION_EVALUATION_FIELDS = (
    "collided",
    "collision_frames",
    "earliest_collision_frame",
    "minimum_margin",
    "boundary_penalty",
    "boss_alignment",
    "motion_penalty",
    "minimum_nonregion_margin",
    "minimum_region_margin",
    "immediate_corner_clearance",
    "selected_teacher_action",
)
TEACHER_ACTION_COLLIDED_INDEX = TEACHER_ACTION_EVALUATION_FIELDS.index("collided")
TEACHER_ACTION_MINIMUM_MARGIN_INDEX = TEACHER_ACTION_EVALUATION_FIELDS.index(
    "minimum_margin"
)
TEACHER_ACTION_SELECTED_INDEX = TEACHER_ACTION_EVALUATION_FIELDS.index(
    "selected_teacher_action"
)


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    seed: int = 20260729
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    validation_fraction: float = 0.2
    risk_loss_weight: float = 0.2
    class_balance: bool = True
    class_balance_power: float = 0.5
    soft_action_loss_weight: float = 0.0
    soft_action_collision_rank_weight: float = 0.0
    soft_action_collision_rank_margin: float = 1.0
    soft_action_temperature: float = 4.0
    soft_action_safety_margin: float = 12.0
    device: str = "auto"


@dataclass(frozen=True, slots=True)
class TrainingMetrics:
    epoch: int
    train_loss: float
    validation_loss: float
    action_accuracy: float
    risk_mae: float
    train_future_visual_loss: float = 0.0
    validation_future_visual_loss: float = 0.0
    train_future_visual_labels: int = 0
    validation_future_visual_labels: int = 0
    train_transition_action_rank_loss: float = 0.0
    validation_transition_action_rank_loss: float = 0.0
    train_transition_action_rank_labels: int = 0
    validation_transition_action_rank_labels: int = 0
    train_transition_action_rank_margin_satisfaction: float = 0.0
    validation_transition_action_rank_margin_satisfaction: float = 0.0
    train_movement_onset_rank_loss: float = 0.0
    validation_movement_onset_rank_loss: float = 0.0
    train_movement_onset_rank_labels: int = 0
    validation_movement_onset_rank_labels: int = 0
    train_movement_onset_rank_margin_satisfaction: float = 0.0
    validation_movement_onset_rank_margin_satisfaction: float = 0.0
    train_movement_speed_change_rank_loss: float = 0.0
    validation_movement_speed_change_rank_loss: float = 0.0
    train_movement_speed_change_rank_labels: int = 0
    validation_movement_speed_change_rank_labels: int = 0
    train_movement_speed_change_rank_margin_satisfaction: float = 0.0
    validation_movement_speed_change_rank_margin_satisfaction: float = 0.0
    train_motion_boundary_rank_loss: float = 0.0
    validation_motion_boundary_rank_loss: float = 0.0
    train_motion_boundary_rank_events: int = 0
    validation_motion_boundary_rank_events: int = 0
    train_motion_boundary_rank_pairs: int = 0
    validation_motion_boundary_rank_pairs: int = 0
    train_motion_boundary_rank_margin_satisfaction: float = 0.0
    validation_motion_boundary_rank_margin_satisfaction: float = 0.0
    train_safety_correction_pairwise_rank_loss: float = 0.0
    validation_safety_correction_pairwise_rank_loss: float = 0.0
    train_safety_correction_pairwise_rank_labels: int = 0
    validation_safety_correction_pairwise_rank_labels: int = 0
    train_safety_correction_pairwise_rank_margin_satisfaction: float = 0.0
    validation_safety_correction_pairwise_rank_margin_satisfaction: float = 0.0
    train_safety_correction_top1_rank_loss: float = 0.0
    validation_safety_correction_top1_rank_loss: float = 0.0
    train_safety_correction_top1_rank_labels: int = 0
    validation_safety_correction_top1_rank_labels: int = 0
    train_safety_correction_top1_rank_margin_satisfaction: float = 0.0
    validation_safety_correction_top1_rank_margin_satisfaction: float = 0.0
    train_safety_correction_minimal_edit_loss: float = 0.0
    validation_safety_correction_minimal_edit_loss: float = 0.0
    train_safety_correction_minimal_edit_labels: int = 0
    validation_safety_correction_minimal_edit_labels: int = 0
    train_safety_correction_minimal_edit_margin_satisfaction: float = 0.0
    validation_safety_correction_minimal_edit_margin_satisfaction: float = 0.0
    train_initial_policy_kl_loss: float = 0.0
    validation_initial_policy_kl_loss: float = 0.0
    train_initial_policy_kl_labels: int = 0
    validation_initial_policy_kl_labels: int = 0


@dataclass(slots=True)
class Demonstrations:
    global_frames: np.ndarray
    local_frames: np.ndarray
    actions: np.ndarray
    risks: np.ndarray
    previous_actions: np.ndarray | None = None
    memory: np.ndarray | None = None
    proficiency: np.ndarray | None = None
    episode_ids: np.ndarray | None = None
    supervision_mask: np.ndarray | None = None
    teacher_action_evaluations: np.ndarray | None = None
    teacher_action_regrets: np.ndarray | None = None
    teacher_action_evaluation_mask: np.ndarray | None = None
    correction_mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        # Corrections are an explicit negative-default annotation.  Filling the
        # mask here keeps direct/legacy constructors compatible while ensuring
        # every in-memory dataset has an unambiguous value for every action.
        if self.correction_mask is None:
            self.correction_mask = np.zeros_like(self.actions, dtype=np.bool_)

    def validate(self) -> None:
        if self.global_frames.ndim != 5 or self.local_frames.ndim != 5:
            raise ValueError("demonstration frames must have [sample, time, channel, height, width]")
        samples, steps = self.global_frames.shape[:2]
        if self.local_frames.shape[:2] != (samples, steps):
            raise ValueError("global and local sequences are not aligned")
        if self.actions.shape != (samples, steps):
            raise ValueError("actions must have [sample, time]")
        if self.risks.shape != (samples, steps):
            raise ValueError("risks must have [sample, time]")
        if self.previous_actions is not None:
            if self.previous_actions.shape != (samples, steps):
                raise ValueError("previous_actions must align with actions")
            if (
                not np.issubdtype(self.previous_actions.dtype, np.integer)
                or np.issubdtype(self.previous_actions.dtype, np.bool_)
                or np.any(self.previous_actions < -1)
                or np.any(self.previous_actions >= 18)
            ):
                raise ValueError(
                    "previous_actions must contain -1 or action ids in [0, 18)"
                )
        if self.memory is not None and self.memory.shape[:2] != (samples, steps):
            raise ValueError("memory must align with visual sequences")
        if self.proficiency is not None:
            if self.proficiency.shape != (samples, steps, PROFICIENCY_VECTOR_SIZE):
                raise ValueError(
                    "proficiency must have "
                    f"[sample, time, {PROFICIENCY_VECTOR_SIZE}]"
                )
        if self.episode_ids is not None and self.episode_ids.shape != (samples,):
            raise ValueError("episode_ids must contain one group id per sample")
        if self.supervision_mask is not None and self.supervision_mask.shape != (samples, steps):
            raise ValueError("supervision_mask must align with actions")
        assert self.correction_mask is not None
        if self.correction_mask.shape != (samples, steps):
            raise ValueError("correction_mask must align with actions")
        if not np.issubdtype(self.correction_mask.dtype, np.bool_):
            raise ValueError("correction_mask must be boolean")
        teacher_fields = (
            self.teacher_action_evaluations,
            self.teacher_action_regrets,
            self.teacher_action_evaluation_mask,
        )
        if any(value is not None for value in teacher_fields):
            if not all(value is not None for value in teacher_fields):
                raise ValueError(
                    "teacher action evaluations, regrets, and mask must be provided together"
                )
            assert self.teacher_action_evaluations is not None
            assert self.teacher_action_regrets is not None
            assert self.teacher_action_evaluation_mask is not None
            expected_evaluations = (
                samples,
                steps,
                18,
                len(TEACHER_ACTION_EVALUATION_FIELDS),
            )
            if self.teacher_action_evaluations.shape != expected_evaluations:
                raise ValueError(
                    "teacher_action_evaluations must have "
                    f"[sample, time, 18, {len(TEACHER_ACTION_EVALUATION_FIELDS)}]"
                )
            if self.teacher_action_regrets.shape != (samples, steps, 18):
                raise ValueError(
                    "teacher_action_regrets must have [sample, time, 18]"
                )
            if self.teacher_action_evaluation_mask.shape != (samples, steps):
                raise ValueError(
                    "teacher_action_evaluation_mask must align with actions"
                )
            available = np.asarray(
                self.teacher_action_evaluation_mask, dtype=np.bool_,
            )
            if available.any():
                evaluations = self.teacher_action_evaluations[available]
                regrets = self.teacher_action_regrets[available]
                if np.isnan(evaluations).any():
                    raise ValueError("available teacher action evaluations cannot be NaN")
                if not np.isfinite(regrets).all() or np.any(regrets < 0.0):
                    raise ValueError(
                        "available teacher action regrets must be finite and nonnegative"
                    )
                collisions = evaluations[:, :, TEACHER_ACTION_COLLIDED_INDEX]
                if np.any((collisions != 0.0) & (collisions != 1.0)):
                    raise ValueError(
                        "teacher action collided evaluations must be zero or one"
                    )
                selected = evaluations[:, :, TEACHER_ACTION_SELECTED_INDEX]
                if np.any((selected != 0.0) & (selected != 1.0)):
                    raise ValueError(
                        "selected teacher action evaluations must be zero or one"
                    )
                if np.any(selected.sum(axis=-1) != 1.0):
                    raise ValueError(
                        "every available evaluation must select one teacher action"
                    )
        if np.any(self.actions < 0) or np.any(self.actions >= 18):
            raise ValueError("action labels must be in [0, 18)")

    def save(self, path: str | Path) -> None:
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, np.ndarray] = {
            # Preserve sub-pixel semantic geometry used by native/DAgger
            # training.  Loading remains dtype-agnostic for legacy float16
            # archives, while every newly saved archive uses float32.
            "global_frames": self.global_frames.astype(np.float32, copy=False),
            "local_frames": self.local_frames.astype(np.float32, copy=False),
            "actions": self.actions.astype(np.int64),
            "risks": self.risks.astype(np.float32),
        }
        if self.memory is not None:
            payload["memory"] = self.memory.astype(np.float32)
        if self.previous_actions is not None:
            payload["previous_actions"] = self.previous_actions.astype(np.int64)
        if self.proficiency is not None:
            payload["proficiency"] = self.proficiency.astype(np.float32)
        if self.episode_ids is not None:
            payload["episode_ids"] = self.episode_ids.astype(np.int64)
        if self.supervision_mask is not None:
            payload["supervision_mask"] = self.supervision_mask.astype(np.uint8)
        assert self.correction_mask is not None
        payload["correction_mask"] = self.correction_mask.astype(np.uint8)
        if self.teacher_action_evaluations is not None:
            assert self.teacher_action_regrets is not None
            assert self.teacher_action_evaluation_mask is not None
            payload["teacher_action_evaluations"] = (
                self.teacher_action_evaluations.astype(np.float32)
            )
            payload["teacher_action_regrets"] = self.teacher_action_regrets.astype(
                np.float32
            )
            payload["teacher_action_evaluation_mask"] = (
                self.teacher_action_evaluation_mask.astype(np.uint8)
            )
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: str | Path) -> "Demonstrations":
        with np.load(path) as data:
            demonstrations = cls(
                global_frames=data["global_frames"],
                local_frames=data["local_frames"],
                actions=data["actions"],
                risks=data["risks"],
                previous_actions=(
                    data["previous_actions"]
                    if "previous_actions" in data.files else None
                ),
                memory=data["memory"] if "memory" in data.files else None,
                proficiency=(
                    data["proficiency"] if "proficiency" in data.files else None
                ),
                episode_ids=data["episode_ids"] if "episode_ids" in data.files else None,
                supervision_mask=(
                    data["supervision_mask"].astype(bool)
                    if "supervision_mask" in data.files else None
                ),
                teacher_action_evaluations=(
                    data["teacher_action_evaluations"]
                    if "teacher_action_evaluations" in data.files else None
                ),
                teacher_action_regrets=(
                    data["teacher_action_regrets"]
                    if "teacher_action_regrets" in data.files else None
                ),
                teacher_action_evaluation_mask=(
                    data["teacher_action_evaluation_mask"].astype(bool)
                    if "teacher_action_evaluation_mask" in data.files else None
                ),
                correction_mask=(
                    data["correction_mask"].astype(bool)
                    if "correction_mask" in data.files else
                    np.zeros_like(data["actions"], dtype=np.bool_)
                ),
            )
        demonstrations.validate()
        return demonstrations


def previous_actions_from_targets(demonstrations: Demonstrations) -> np.ndarray:
    """Infer prior executed actions for teacher-executed legacy archives."""

    demonstrations.validate()
    if demonstrations.episode_ids is None:
        raise ValueError("episode_ids are required to infer previous actions")
    if demonstrations.actions.shape[1] != 1:
        raise ValueError(
            "previous actions cannot be inferred from multi-frame windows; "
            "provide recorded previous_actions"
        )
    result = np.full(demonstrations.actions.shape, -1, dtype=np.int64)
    previous_episode: int | None = None
    previous_action = -1
    for sample, raw_episode_id in enumerate(demonstrations.episode_ids):
        episode_id = int(raw_episode_id)
        if episode_id != previous_episode:
            previous_action = -1
            previous_episode = episode_id
        result[sample, :] = previous_action
        previous_action = int(demonstrations.actions[sample, -1])
    return result


def teacher_action_acceptance_weights(
    evaluations: Tensor,
    regrets: Tensor,
    available: Tensor,
    actions: Tensor,
    *,
    temperature: float,
    safety_margin: float,
) -> tuple[Tensor, Tensor]:
    """Build regret-weighted acceptable sets without averaging route modes.

    Every non-colliding action above ``safety_margin`` with the teacher's same
    moving/stationary state is acceptable, and the selected teacher action is
    always retained. This preserves movement onsets while allowing alternate
    safe route directions. The returned weights peak at one inside each set and
    decay with clearance regret. Training maximizes the total policy probability
    in this set instead of matching a multimodal probability vector with cross
    entropy.
    """

    if torch is None:
        raise RuntimeError("PyTorch is required for soft teacher targets")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("soft action temperature must be finite and positive")
    if not math.isfinite(safety_margin) or safety_margin < 0.0:
        raise ValueError("soft action safety margin must be finite and nonnegative")
    if evaluations.shape[:-2] != regrets.shape[:-1]:
        raise ValueError("teacher evaluations and regrets do not align")
    if evaluations.shape[-2] != 18 or regrets.shape[-1] != 18:
        raise ValueError("teacher action evidence must use the 18-action vocabulary")
    if evaluations.shape[-1] != len(TEACHER_ACTION_EVALUATION_FIELDS):
        raise ValueError("teacher action evaluation field count is invalid")
    if available.shape != regrets.shape[:-1] or actions.shape != available.shape:
        raise ValueError("teacher action evidence must align with action labels")

    collisions = (
        evaluations[..., TEACHER_ACTION_COLLIDED_INDEX] >= 0.5
    )
    margins = evaluations[..., TEACHER_ACTION_MINIMUM_MARGIN_INDEX]
    acceptable = (~collisions) & (margins >= safety_margin)
    selected = evaluations[..., TEACHER_ACTION_SELECTED_INDEX] >= 0.5
    candidate_moving = (
        torch.arange(18, device=actions.device).remainder(9) != 4
    )
    teacher_actions = selected.to(dtype=torch.int64).argmax(dim=-1)
    teacher_moving = teacher_actions.remainder(9) != 4
    acceptable = acceptable & (
        candidate_moving == teacher_moving.unsqueeze(-1)
    )
    acceptable = (acceptable | selected) & available.unsqueeze(-1)

    masked_regrets = regrets.masked_fill(~acceptable, math.inf)
    minimum_regret = masked_regrets.amin(dim=-1, keepdim=True)
    # Unavailable rows contain no acceptable action. Keep their arithmetic
    # finite; the returned availability mask excludes them from the loss.
    minimum_regret = torch.where(
        available.unsqueeze(-1),
        minimum_regret,
        torch.zeros_like(minimum_regret),
    )
    weights = torch.exp(-(regrets - minimum_regret) / temperature)
    weights = weights.masked_fill(~acceptable, 0.0)
    return weights, available.bool()


def teacher_set_valued_action_loss(
    logits: Tensor,
    evaluations: Tensor,
    regrets: Tensor,
    available: Tensor,
    actions: Tensor,
    *,
    temperature: float,
    safety_margin: float,
) -> tuple[Tensor, Tensor]:
    """Return per-decision loss and the rows carrying teacher evaluations."""

    weights, target_mask = teacher_action_acceptance_weights(
        evaluations,
        regrets,
        available,
        actions,
        temperature=temperature,
        safety_margin=safety_margin,
    )
    if logits.shape != regrets.shape:
        raise ValueError("policy logits and teacher action regrets do not align")
    probabilities = torch.softmax(logits, dim=-1)
    accepted_probability = (probabilities * weights).sum(dim=-1)
    losses = -torch.log(accepted_probability.clamp_min(
        torch.finfo(logits.dtype).tiny
    ))
    return losses, target_mask


def teacher_action_collision_ranking_loss(
    logits: Tensor,
    evaluations: Tensor,
    regrets: Tensor,
    available: Tensor,
    actions: Tensor,
    *,
    temperature: float,
    safety_margin: float,
    ranking_margin: float,
) -> tuple[Tensor, Tensor]:
    """Rank at least one acceptable action above every predicted collision.

    The set-valued objective can assign most probability to its acceptable set
    while leaving one colliding action as the discrete argmax. This hinge term
    closes that gap without selecting a unique safe route.
    """

    if torch is None:
        raise RuntimeError("PyTorch is required for soft teacher targets")
    if not math.isfinite(ranking_margin) or ranking_margin < 0.0:
        raise ValueError("soft action collision rank margin must be finite and nonnegative")
    weights, target_mask = teacher_action_acceptance_weights(
        evaluations,
        regrets,
        available,
        actions,
        temperature=temperature,
        safety_margin=safety_margin,
    )
    if logits.shape != regrets.shape:
        raise ValueError("policy logits and teacher action regrets do not align")
    acceptable = weights > 0.0
    collisions = evaluations[..., TEACHER_ACTION_COLLIDED_INDEX] >= 0.5
    unsafe_collisions = collisions & ~acceptable & available.unsqueeze(-1)
    ranking_mask = target_mask & unsafe_collisions.any(dim=-1)

    negative_infinity = torch.finfo(logits.dtype).min
    best_acceptable = logits.masked_fill(~acceptable, negative_infinity).amax(dim=-1)
    best_collision = logits.masked_fill(
        ~unsafe_collisions, negative_infinity,
    ).amax(dim=-1)
    losses = F.relu(best_collision - best_acceptable + ranking_margin)
    losses = torch.where(ranking_mask, losses, torch.zeros_like(losses))
    return losses, ranking_mask


if torch is not None:

    class _DemonstrationDataset(Dataset):
        def __init__(
            self,
            demonstrations: Demonstrations,
            indices: np.ndarray,
            memory_size: int,
            proficiency_size: int,
        ) -> None:
            self.demonstrations = demonstrations
            self.indices = indices
            self.memory_size = memory_size
            self.proficiency_size = proficiency_size

        def __len__(self) -> int:
            return len(self.indices)

        def __getitem__(
            self, index: int,
        ) -> tuple[
            Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,
            Tensor, Tensor, Tensor,
        ]:
            item = int(self.indices[index])
            demo = self.demonstrations
            if demo.memory is None:
                memory = np.zeros((demo.actions.shape[1], self.memory_size), dtype=np.float32)
            else:
                memory = demo.memory[item]
            if demo.proficiency is None:
                if self.proficiency_size == 0:
                    proficiency = np.empty(
                        (demo.actions.shape[1], 0), dtype=np.float32,
                    )
                else:
                    proficiency = np.broadcast_to(
                        proficiency_vector("expert"),
                        (demo.actions.shape[1], PROFICIENCY_VECTOR_SIZE),
                    ).copy()
            else:
                proficiency = demo.proficiency[item]
            if demo.supervision_mask is None:
                mask = np.zeros(demo.actions.shape[1], dtype=bool)
                mask[-1] = True
            else:
                mask = demo.supervision_mask[item]
            if demo.teacher_action_evaluations is None:
                teacher_evaluations = np.zeros(
                    (
                        demo.actions.shape[1],
                        18,
                        len(TEACHER_ACTION_EVALUATION_FIELDS),
                    ),
                    dtype=np.float32,
                )
                teacher_regrets = np.zeros(
                    (demo.actions.shape[1], 18), dtype=np.float32,
                )
                teacher_evaluation_mask = np.zeros(
                    demo.actions.shape[1], dtype=bool,
                )
            else:
                assert demo.teacher_action_regrets is not None
                assert demo.teacher_action_evaluation_mask is not None
                teacher_evaluations = demo.teacher_action_evaluations[item]
                teacher_regrets = demo.teacher_action_regrets[item]
                teacher_evaluation_mask = demo.teacher_action_evaluation_mask[item]
            return (
                torch.from_numpy(demo.global_frames[item]).float(),
                torch.from_numpy(demo.local_frames[item]).float(),
                torch.from_numpy(memory).float(),
                torch.from_numpy(proficiency).float(),
                torch.from_numpy(demo.actions[item]).long(),
                torch.from_numpy(demo.risks[item]).float(),
                torch.from_numpy(mask).bool(),
                torch.from_numpy(teacher_evaluations).float(),
                torch.from_numpy(teacher_regrets).float(),
                torch.from_numpy(teacher_evaluation_mask).bool(),
            )


def to_recurrent_sequences(
    demonstrations: Demonstrations,
    *,
    sequence_length: int = 32,
) -> Demonstrations:
    """Convert delayed decision windows into contiguous per-episode sequences."""

    demonstrations.validate()
    if sequence_length <= 1:
        raise ValueError("sequence_length must be greater than one")
    if demonstrations.episode_ids is None:
        raise ValueError("episode_ids are required for recurrent sequence conversion")

    global_sequences: list[np.ndarray] = []
    local_sequences: list[np.ndarray] = []
    action_sequences: list[np.ndarray] = []
    risk_sequences: list[np.ndarray] = []
    previous_action_sequences: list[np.ndarray] = []
    memory_sequences: list[np.ndarray] = []
    proficiency_sequences: list[np.ndarray] = []
    teacher_evaluation_sequences: list[np.ndarray] = []
    teacher_regret_sequences: list[np.ndarray] = []
    teacher_evaluation_mask_sequences: list[np.ndarray] = []
    correction_mask_sequences: list[np.ndarray] = []
    episode_ids: list[int] = []
    for episode_id in np.unique(demonstrations.episode_ids):
        indices = np.flatnonzero(demonstrations.episode_ids == episode_id)
        if len(indices) < sequence_length:
            continue
        starts = list(range(0, len(indices) - sequence_length + 1, sequence_length))
        final_start = len(indices) - sequence_length
        if starts[-1] != final_start:
            starts.append(final_start)
        for start in starts:
            selected = indices[start:start + sequence_length]
            global_sequences.append(demonstrations.global_frames[selected, -1])
            local_sequences.append(demonstrations.local_frames[selected, -1])
            action_sequences.append(demonstrations.actions[selected, -1])
            risk_sequences.append(demonstrations.risks[selected, -1])
            assert demonstrations.correction_mask is not None
            correction_mask_sequences.append(
                demonstrations.correction_mask[selected, -1]
            )
            if demonstrations.previous_actions is not None:
                previous_action_sequences.append(
                    demonstrations.previous_actions[selected, -1]
                )
            if demonstrations.memory is not None:
                memory_sequences.append(demonstrations.memory[selected, -1])
            if demonstrations.proficiency is not None:
                proficiency_sequences.append(
                    demonstrations.proficiency[selected, -1]
                )
            if demonstrations.teacher_action_evaluations is not None:
                assert demonstrations.teacher_action_regrets is not None
                assert demonstrations.teacher_action_evaluation_mask is not None
                teacher_evaluation_sequences.append(
                    demonstrations.teacher_action_evaluations[selected, -1]
                )
                teacher_regret_sequences.append(
                    demonstrations.teacher_action_regrets[selected, -1]
                )
                teacher_evaluation_mask_sequences.append(
                    demonstrations.teacher_action_evaluation_mask[selected, -1]
                )
            episode_ids.append(int(episode_id))
    if not global_sequences:
        raise ValueError("no episode is long enough for the requested sequence length")
    result = Demonstrations(
        global_frames=np.stack(global_sequences).astype(np.float32, copy=False),
        local_frames=np.stack(local_sequences).astype(np.float32, copy=False),
        actions=np.stack(action_sequences).astype(np.int64, copy=False),
        risks=np.stack(risk_sequences).astype(np.float32, copy=False),
        previous_actions=(
            np.stack(previous_action_sequences).astype(np.int64, copy=False)
            if previous_action_sequences else None
        ),
        memory=(np.stack(memory_sequences).astype(np.float32, copy=False) if memory_sequences else None),
        proficiency=(
            np.stack(proficiency_sequences).astype(np.float32, copy=False)
            if proficiency_sequences else None
        ),
        episode_ids=np.asarray(episode_ids, dtype=np.int64),
        supervision_mask=np.ones((len(global_sequences), sequence_length), dtype=bool),
        teacher_action_evaluations=(
            np.stack(teacher_evaluation_sequences).astype(np.float32, copy=False)
            if teacher_evaluation_sequences else None
        ),
        teacher_action_regrets=(
            np.stack(teacher_regret_sequences).astype(np.float32, copy=False)
            if teacher_regret_sequences else None
        ),
        teacher_action_evaluation_mask=(
            np.stack(teacher_evaluation_mask_sequences).astype(bool, copy=False)
            if teacher_evaluation_mask_sequences else None
        ),
        correction_mask=np.stack(correction_mask_sequences).astype(
            bool, copy=False,
        ),
    )
    result.validate()
    return result


def _select_device(requested: str) -> str:
    if torch is None:
        raise RuntimeError("PyTorch is required for policy training")
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)


def train_behavior_cloning(
    demonstrations: Demonstrations,
    *,
    policy_config: PolicyConfig = PolicyConfig(),
    training_config: TrainingConfig = TrainingConfig(),
    output: str | Path | None = None,
    training_data: Mapping[str, Any] | None = None,
) -> tuple[HumanVisionPolicy, list[TrainingMetrics]]:
    if torch is None:
        raise RuntimeError("PyTorch is required for policy training")
    demonstrations.validate()
    if (
        demonstrations.proficiency is not None
        and demonstrations.proficiency.shape[-1] != policy_config.proficiency_size
    ):
        raise ValueError(
            "demonstration proficiency width does not match policy_config"
        )
    if not 0.0 <= training_config.class_balance_power <= 1.0:
        raise ValueError("class_balance_power must be in [0, 1]")
    if (
        not math.isfinite(training_config.soft_action_loss_weight)
        or training_config.soft_action_loss_weight < 0.0
    ):
        raise ValueError("soft_action_loss_weight must be finite and nonnegative")
    if (
        not math.isfinite(training_config.soft_action_collision_rank_weight)
        or training_config.soft_action_collision_rank_weight < 0.0
    ):
        raise ValueError(
            "soft_action_collision_rank_weight must be finite and nonnegative"
        )
    if (
        not math.isfinite(training_config.soft_action_collision_rank_margin)
        or training_config.soft_action_collision_rank_margin < 0.0
    ):
        raise ValueError(
            "soft_action_collision_rank_margin must be finite and nonnegative"
        )
    if (
        training_config.soft_action_collision_rank_weight > 0.0
        and training_config.soft_action_loss_weight <= 0.0
    ):
        raise ValueError("soft action collision ranking requires soft action loss")
    if (
        not math.isfinite(training_config.soft_action_temperature)
        or training_config.soft_action_temperature <= 0.0
    ):
        raise ValueError("soft_action_temperature must be finite and positive")
    if (
        not math.isfinite(training_config.soft_action_safety_margin)
        or training_config.soft_action_safety_margin < 0.0
    ):
        raise ValueError("soft_action_safety_margin must be finite and nonnegative")
    if training_config.soft_action_loss_weight > 0.0 and (
        demonstrations.teacher_action_evaluations is None
        or demonstrations.teacher_action_evaluation_mask is None
        or not demonstrations.teacher_action_evaluation_mask.any()
    ):
        raise ValueError(
            "soft action loss requires recorded teacher action evaluations"
        )
    _seed_everything(training_config.seed)
    sample_count = demonstrations.actions.shape[0]
    if sample_count < 2:
        raise ValueError("at least two demonstration sequences are required")
    generator = np.random.default_rng(training_config.seed)
    if demonstrations.episode_ids is not None and len(np.unique(demonstrations.episode_ids)) >= 2:
        groups = generator.permutation(np.unique(demonstrations.episode_ids))
        validation_group_count = max(1, int(round(len(groups) * training_config.validation_fraction)))
        validation_groups = groups[:validation_group_count]
        validation_mask = np.isin(demonstrations.episode_ids, validation_groups)
        validation_indices = np.flatnonzero(validation_mask)
        train_indices = np.flatnonzero(~validation_mask)
    else:
        indices = generator.permutation(sample_count)
        validation_count = max(1, int(round(sample_count * training_config.validation_fraction)))
        validation_indices = indices[:validation_count]
        train_indices = indices[validation_count:]
    if len(train_indices) == 0:
        raise ValueError("validation split leaves no training samples")

    train_dataset = _DemonstrationDataset(
        demonstrations,
        train_indices,
        policy_config.memory_size,
        policy_config.proficiency_size,
    )
    validation_dataset = _DemonstrationDataset(
        demonstrations,
        validation_indices,
        policy_config.memory_size,
        policy_config.proficiency_size,
    )
    loader_generator = torch.Generator().manual_seed(training_config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        generator=loader_generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
    )

    device = _select_device(training_config.device)
    model = HumanVisionPolicy(policy_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    history: list[TrainingMetrics] = []
    action_weights = None
    if training_config.class_balance:
        train_actions = demonstrations.actions[train_indices]
        if demonstrations.supervision_mask is None:
            hard_mask = np.ones(train_actions.shape, dtype=np.bool_)
        else:
            hard_mask = demonstrations.supervision_mask[train_indices].copy()
        if (
            training_config.soft_action_loss_weight > 0.0
            and demonstrations.teacher_action_evaluation_mask is not None
        ):
            hard_mask &= ~demonstrations.teacher_action_evaluation_mask[
                train_indices
            ]
        labels = train_actions[hard_mask]
        if len(labels):
            counts = np.bincount(
                labels, minlength=policy_config.action_count,
            ).astype(np.float64)
            present = counts > 0
            weights = np.zeros(policy_config.action_count, dtype=np.float32)
            inverse_frequency = len(labels) / (present.sum() * counts[present])
            weights[present] = (
                inverse_frequency ** training_config.class_balance_power
            )
            # A rare teacher correction should matter without destabilizing the
            # small imitation runs used during iteration.
            weights[present] = np.minimum(weights[present], 10.0)
            action_weights = torch.from_numpy(weights).to(device)

    for epoch in range(1, training_config.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_items = 0
        for (
            global_frames,
            local_frames,
            memory,
            proficiency,
            actions,
            risks,
            mask,
            teacher_evaluations,
            teacher_regrets,
            teacher_evaluation_mask,
        ) in train_loader:
            global_frames = global_frames.to(device)
            local_frames = local_frames.to(device)
            memory = memory.to(device)
            proficiency = proficiency.to(device)
            actions = actions.to(device)
            risks = risks.to(device)
            mask = mask.to(device)
            teacher_evaluations = teacher_evaluations.to(device)
            teacher_regrets = teacher_regrets.to(device)
            teacher_evaluation_mask = teacher_evaluation_mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, predicted_risk, _ = model(
                global_frames,
                local_frames,
                memory,
                proficiency=proficiency,
            )
            soft_mask = (
                teacher_evaluation_mask
                if training_config.soft_action_loss_weight > 0.0 else
                torch.zeros_like(mask)
            )
            hard_mask = mask & ~soft_mask
            action_loss = logits.sum() * 0.0
            if hard_mask.any():
                action_loss = F.cross_entropy(
                    logits[hard_mask], actions[hard_mask], weight=action_weights,
                )
            if training_config.soft_action_loss_weight > 0.0:
                soft_terms, available_soft_mask = teacher_set_valued_action_loss(
                    logits,
                    teacher_evaluations,
                    teacher_regrets,
                    teacher_evaluation_mask,
                    actions,
                    temperature=training_config.soft_action_temperature,
                    safety_margin=training_config.soft_action_safety_margin,
                )
                if available_soft_mask.any():
                    action_loss = action_loss + (
                        training_config.soft_action_loss_weight
                        * soft_terms[available_soft_mask].mean()
                    )
                if training_config.soft_action_collision_rank_weight > 0.0:
                    rank_terms, rank_mask = teacher_action_collision_ranking_loss(
                        logits,
                        teacher_evaluations,
                        teacher_regrets,
                        teacher_evaluation_mask,
                        actions,
                        temperature=training_config.soft_action_temperature,
                        safety_margin=training_config.soft_action_safety_margin,
                        ranking_margin=(
                            training_config.soft_action_collision_rank_margin
                        ),
                    )
                    if rank_mask.any():
                        action_loss = action_loss + (
                            training_config.soft_action_collision_rank_weight
                            * rank_terms[rank_mask].mean()
                        )
            risk_loss = F.smooth_l1_loss(predicted_risk[mask], risks[mask])
            loss = action_loss + training_config.risk_loss_weight * risk_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_loss_sum += float(loss.detach()) * len(actions)
            train_items += len(actions)

        model.eval()
        validation_loss_sum = 0.0
        validation_items = 0
        correct = 0
        labels = 0
        risk_error = 0.0
        with torch.no_grad():
            for (
                global_frames,
                local_frames,
                memory,
                proficiency,
                actions,
                risks,
                mask,
                teacher_evaluations,
                teacher_regrets,
                teacher_evaluation_mask,
            ) in validation_loader:
                global_frames = global_frames.to(device)
                local_frames = local_frames.to(device)
                memory = memory.to(device)
                proficiency = proficiency.to(device)
                actions = actions.to(device)
                risks = risks.to(device)
                mask = mask.to(device)
                teacher_evaluations = teacher_evaluations.to(device)
                teacher_regrets = teacher_regrets.to(device)
                teacher_evaluation_mask = teacher_evaluation_mask.to(device)
                logits, predicted_risk, _ = model(
                    global_frames,
                    local_frames,
                    memory,
                    proficiency=proficiency,
                )
                soft_mask = (
                    teacher_evaluation_mask
                    if training_config.soft_action_loss_weight > 0.0 else
                    torch.zeros_like(mask)
                )
                hard_mask = mask & ~soft_mask
                action_loss = logits.sum() * 0.0
                if hard_mask.any():
                    action_loss = F.cross_entropy(
                        logits[hard_mask], actions[hard_mask],
                    )
                if training_config.soft_action_loss_weight > 0.0:
                    soft_terms, available_soft_mask = teacher_set_valued_action_loss(
                        logits,
                        teacher_evaluations,
                        teacher_regrets,
                        teacher_evaluation_mask,
                        actions,
                        temperature=training_config.soft_action_temperature,
                        safety_margin=training_config.soft_action_safety_margin,
                    )
                    if available_soft_mask.any():
                        action_loss = action_loss + (
                            training_config.soft_action_loss_weight
                            * soft_terms[available_soft_mask].mean()
                        )
                    if training_config.soft_action_collision_rank_weight > 0.0:
                        rank_terms, rank_mask = teacher_action_collision_ranking_loss(
                            logits,
                            teacher_evaluations,
                            teacher_regrets,
                            teacher_evaluation_mask,
                            actions,
                            temperature=training_config.soft_action_temperature,
                            safety_margin=training_config.soft_action_safety_margin,
                            ranking_margin=(
                                training_config.soft_action_collision_rank_margin
                            ),
                        )
                        if rank_mask.any():
                            action_loss = action_loss + (
                                training_config.soft_action_collision_rank_weight
                                * rank_terms[rank_mask].mean()
                            )
                risk_loss = F.smooth_l1_loss(predicted_risk[mask], risks[mask])
                loss = action_loss + training_config.risk_loss_weight * risk_loss
                validation_loss_sum += float(loss) * len(actions)
                validation_items += len(actions)
                action_metric_mask = hard_mask | soft_mask
                correct += int((
                    logits.argmax(dim=-1)[action_metric_mask]
                    == actions[action_metric_mask]
                ).sum())
                labels += int(action_metric_mask.sum())
                risk_error += float(torch.abs(predicted_risk[mask] - risks[mask]).sum())

        metrics = TrainingMetrics(
            epoch=epoch,
            train_loss=train_loss_sum / max(train_items, 1),
            validation_loss=validation_loss_sum / max(validation_items, 1),
            action_accuracy=correct / max(labels, 1),
            risk_mae=risk_error / max(labels, 1),
        )
        history.append(metrics)

    if output is not None:
        save_checkpoint(
            model,
            output,
            policy_config=policy_config,
            history=history,
            training_config=training_config,
            training_data=training_data,
        )
    return model, history


def save_checkpoint(
    model: HumanVisionPolicy,
    path: str | Path,
    *,
    policy_config: PolicyConfig,
    history: Iterable[TrainingMetrics] = (),
    training_config: TrainingConfig | None = None,
    training_data: Mapping[str, Any] | None = None,
) -> None:
    if torch is None:
        raise RuntimeError("PyTorch is required for policy checkpoints")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version": 3,
            "policy_config": asdict(policy_config),
            "state_dict": model.state_dict(),
            "history": [asdict(item) for item in history],
            "training_config": (
                asdict(training_config) if training_config is not None else None
            ),
            "training_data": dict(training_data or {}),
        },
        path,
    )


def expand_checkpoint_with_previous_action_context(
    source: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Add a zero-initialized 18-way previous-action input to a checkpoint.

    This migration is intentionally narrow: the source must have a declared
    two-entry scenario vocabulary occupying its complete two-value memory and
    no existing previous-action input. The new columns are inserted between
    scenario memory and proficiency, matching ``HumanVisionPolicy.forward``.
    Zero GRU input weights make every previous-action token inert until the
    expanded checkpoint is trained.
    """

    if torch is None:
        raise RuntimeError("PyTorch is required for policy checkpoints")
    source_path = Path(source)
    output_path = Path(output)
    if not source_path.is_file():
        raise FileNotFoundError(f"source checkpoint does not exist: {source_path}")
    if source_path.resolve() == output_path.resolve():
        raise ValueError("source and output checkpoint paths must be different")

    from .provenance import file_sha256

    source_sha256 = file_sha256(source_path)
    source_model, source_checkpoint = load_checkpoint(source_path, device="cpu")
    source_config = source_model.config
    vocabulary = getattr(source_model, "scenario_vocabulary", None)
    previous_action_size = int(
        getattr(source_model, "previous_action_size", 0)
    )
    previous_action_offset = int(
        getattr(source_model, "previous_action_offset", 0)
    )
    if source_config.memory_size != 2:
        raise ValueError(
            "source checkpoint memory_size must be 2 for previous-action expansion"
        )
    if vocabulary is None:
        raise ValueError(
            "source checkpoint must declare a scenario_vocabulary"
        )
    if len(vocabulary) != 2:
        raise ValueError(
            "source checkpoint scenario_vocabulary must contain exactly 2 entries"
        )
    if previous_action_size != 0:
        raise ValueError(
            "source checkpoint must not already use previous-action context"
        )
    if previous_action_offset != 2:
        raise ValueError(
            "source checkpoint scenario context must occupy memory columns [0, 2)"
        )

    target_previous_action_size = 18
    target_memory_size = source_config.memory_size + target_previous_action_size
    target_config = replace(source_config, memory_size=target_memory_size)
    target_model = HumanVisionPolicy(target_config)
    source_state = source_model.state_dict()
    target_template = target_model.state_dict()
    recurrent_name = "recurrent.weight_ih_l0"
    if set(source_state) != set(target_template):
        raise ValueError("source and expanded policy state dictionaries do not align")

    visual_width = source_config.feature_size * 2
    inserted_column_start = visual_width + source_config.memory_size
    inserted_column_stop = inserted_column_start + target_previous_action_size
    source_recurrent = source_state[recurrent_name]
    target_recurrent_shape = target_template[recurrent_name].shape
    if (
        source_recurrent.ndim != 2
        or source_recurrent.shape[0] != target_recurrent_shape[0]
        or source_recurrent.shape[1] + target_previous_action_size
        != target_recurrent_shape[1]
    ):
        raise ValueError("source recurrent input weight has an incompatible shape")

    expanded_state: dict[str, Tensor] = {}
    for name, source_value in source_state.items():
        if name != recurrent_name:
            target_value = target_template[name]
            if source_value.shape != target_value.shape:
                raise ValueError(
                    f"source parameter {name!r} has an incompatible shape"
                )
            expanded_state[name] = source_value.detach().clone()
            continue
        expanded = source_value.new_zeros(target_recurrent_shape)
        expanded[:, :inserted_column_start] = source_value[
            :, :inserted_column_start
        ]
        expanded[:, inserted_column_stop:] = source_value[
            :, inserted_column_start:
        ]
        expanded_state[name] = expanded

    target_model.load_state_dict(expanded_state, strict=True)
    verified_state = target_model.state_dict()
    unchanged_tensors_equal = all(
        torch.equal(verified_state[name], source_state[name])
        for name in source_state
        if name != recurrent_name
    )
    recurrent_prefix_equal = torch.equal(
        verified_state[recurrent_name][:, :inserted_column_start],
        source_recurrent[:, :inserted_column_start],
    )
    recurrent_suffix_equal = torch.equal(
        verified_state[recurrent_name][:, inserted_column_stop:],
        source_recurrent[:, inserted_column_start:],
    )
    inserted_columns = verified_state[recurrent_name][
        :, inserted_column_start:inserted_column_stop
    ]
    inserted_nonzero_count = int(torch.count_nonzero(inserted_columns).item())
    if not (
        unchanged_tensors_equal
        and recurrent_prefix_equal
        and recurrent_suffix_equal
        and inserted_nonzero_count == 0
    ):
        raise RuntimeError("expanded checkpoint failed exact weight-copy verification")

    source_training_data = source_checkpoint.get("training_data")
    if not isinstance(source_training_data, Mapping):
        raise ValueError("source checkpoint training_data must be a mapping")
    training_data = dict(source_training_data)
    expansion_metadata = {
        "source_memory_size": source_config.memory_size,
        "target_memory_size": target_memory_size,
        "previous_action_offset": previous_action_offset,
        "previous_action_size": target_previous_action_size,
        "input_layout": [
            "visual_features",
            "scenario_memory",
            "previous_action_one_hot",
            "proficiency",
        ],
        "weight_copy_proof": {
            "state_tensor_count": len(source_state),
            "all_source_values_copied_exactly": True,
            "unchanged_tensors_equal": unchanged_tensors_equal,
            "recurrent_prefix_equal": recurrent_prefix_equal,
            "recurrent_suffix_equal": recurrent_suffix_equal,
        },
        "zero_initialization_proof": {
            "parameter": recurrent_name,
            "column_start": inserted_column_start,
            "column_stop_exclusive": inserted_column_stop,
            "column_count": target_previous_action_size,
            "row_count": int(inserted_columns.shape[0]),
            "nonzero_count": inserted_nonzero_count,
            "maximum_absolute_value": float(
                inserted_columns.abs().max().item()
            ),
            "verified_exact": inserted_nonzero_count == 0,
        },
        "epoch_zero_equivalence": {
            "guaranteed": True,
            "scope": (
                "identical visual, original scenario memory, proficiency, and "
                "initial hidden inputs with any previous-action one-hot token"
            ),
            "reason": (
                "the only new model inputs multiply exact-zero GRU input weights"
            ),
        },
    }
    training_data.update({
        "parent_checkpoint": str(source_path),
        "parent_checkpoint_sha256": source_sha256,
        "parent_checkpoint_policy_config": asdict(source_config),
        "initialization": (
            "complete_policy_state_with_zero_initialized_previous_action_input"
        ),
        "scenario_vocabulary": list(vocabulary),
        "previous_action_offset": previous_action_offset,
        "previous_action_size": target_previous_action_size,
        "previous_action_expansion": expansion_metadata,
    })
    expanded_checkpoint = dict(source_checkpoint)
    expanded_checkpoint.update({
        "version": 3,
        "policy_config": asdict(target_config),
        "state_dict": verified_state,
        "training_data": training_data,
    })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(expanded_checkpoint, output_path)

    # Reload through the ordinary path so the persisted context contract, not
    # only the in-memory tensors above, is verified before reporting success.
    persisted_model, persisted_checkpoint = load_checkpoint(
        output_path, device="cpu",
    )
    if (
        persisted_model.config != target_config
        or getattr(persisted_model, "scenario_vocabulary", None)
        != tuple(vocabulary)
        or getattr(persisted_model, "previous_action_offset", None) != 2
        or getattr(persisted_model, "previous_action_size", None) != 18
    ):
        raise RuntimeError("persisted expanded checkpoint failed context verification")

    return {
        "checkpoint": str(output_path),
        "checkpoint_sha256": file_sha256(output_path),
        "parent_checkpoint": str(source_path),
        "parent_checkpoint_sha256": source_sha256,
        "policy_config": persisted_checkpoint["policy_config"],
        "initialization": training_data["initialization"],
        "previous_action_expansion": expansion_metadata,
    }


def load_checkpoint(path: str | Path, *, device: str = "cpu") -> tuple[HumanVisionPolicy, dict[str, Any]]:
    if torch is None:
        raise RuntimeError("PyTorch is required for policy checkpoints")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config_values = dict(checkpoint["policy_config"])
    # Checkpoints created before inference modes used one-frame recurrent
    # streaming during evaluation. Preserve that behavior when loading them.
    config_values.setdefault("inference_mode", "stream")
    if "proficiency_size" not in config_values:
        recurrent_input = int(checkpoint["state_dict"]["recurrent.weight_ih_l0"].shape[1])
        base_input = (
            int(config_values["feature_size"]) * 2
            + int(config_values.get("memory_size", 4))
        )
        config_values["proficiency_size"] = recurrent_input - base_input
    config = PolicyConfig(**config_values)
    model = HumanVisionPolicy(config).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    training_data = checkpoint.get("training_data")
    vocabulary = (
        training_data.get("scenario_vocabulary")
        if isinstance(training_data, Mapping) else
        None
    )
    previous_action_size = (
        training_data.get("previous_action_size", 0)
        if isinstance(training_data, Mapping) else
        0
    )
    previous_action_offset = (
        training_data.get(
            "previous_action_offset",
            len(vocabulary) if isinstance(vocabulary, (tuple, list)) else 0,
        )
        if isinstance(training_data, Mapping) else
        0
    )
    if previous_action_size not in {0, 18}:
        raise ValueError("checkpoint previous_action_size must be 0 or 18")
    if vocabulary is not None:
        if (
            not isinstance(vocabulary, (tuple, list))
            or not all(isinstance(value, str) and value for value in vocabulary)
        ):
            raise ValueError("checkpoint scenario_vocabulary must be a string list")
        if previous_action_offset != len(vocabulary):
            raise ValueError(
                "checkpoint previous action features must follow scenario context"
            )
        if previous_action_offset + previous_action_size != config.memory_size:
            raise ValueError(
                "checkpoint scenario/action context width does not match memory_size"
            )
        if len(set(vocabulary)) != len(vocabulary):
            raise ValueError("checkpoint scenario_vocabulary entries must be unique")
        model.scenario_vocabulary = tuple(vocabulary)
        model.previous_action_size = previous_action_size
        model.previous_action_offset = previous_action_offset
    elif previous_action_size:
        raise ValueError(
            "checkpoint previous action context requires scenario_vocabulary metadata"
        )
    model.eval()
    checkpoint["policy_config"] = asdict(config)
    return model, checkpoint


def write_metrics(path: str | Path, history: Iterable[TrainingMetrics]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([asdict(item) for item in history], indent=2) + "\n",
        encoding="utf-8",
    )
