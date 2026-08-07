"""Episode-contiguous truncated backpropagation for streaming policies."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import math
from pathlib import Path
import random
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

try:
    import torch
    from torch import Tensor
    from torch.nn import functional as F
except ImportError:  # pragma: no cover - the base install intentionally omits torch
    torch = None
    Tensor = object  # type: ignore[assignment,misc]

from .policy import HumanVisionPolicy, PolicyConfig, proficiency_vector
from .protocol import Action
from .training import (
    TEACHER_ACTION_EVALUATION_FIELDS,
    Demonstrations,
    TrainingMetrics,
    save_checkpoint,
    teacher_action_collision_ranking_loss,
    teacher_set_valued_action_loss,
)


DEFAULT_FUTURE_VISUAL_HORIZONS = (20, 40, 80)


def _normalize_future_visual_horizons(values: Iterable[int]) -> tuple[int, ...]:
    horizons = tuple(values)
    if not horizons:
        raise ValueError("future visual horizons cannot be empty")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in horizons
    ):
        raise ValueError("future visual horizons must be positive integers")
    if len(set(horizons)) != len(horizons):
        raise ValueError("future visual horizons cannot contain duplicates")
    if tuple(sorted(horizons)) != horizons:
        raise ValueError("future visual horizons must be strictly increasing")
    return horizons


@dataclass(frozen=True, slots=True)
class StatefulTrainingConfig:
    """Optimization controls for latest-frame streaming TBPTT."""

    seed: int = 20260731
    epochs: int = 20
    chunk_length: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    validation_fraction: float = 0.2
    risk_loss_weight: float = 0.2
    class_balance: bool = True
    class_balance_power: float = 0.5
    gradient_clip: float = 5.0
    device: str = "auto"
    validation_episode_ids: tuple[int, ...] | None = None
    horizontal_reflection_probability: float = 0.0
    restore_best_validation: bool = True
    movement_onset_weight: float = 1.0
    movement_stop_weight: float = 1.0
    movement_speed_change_weight: float = 1.0
    direction_change_weight: float = 1.0
    episode_balanced: bool = False
    exact_action_loss_weight: float = 1.0
    direction_loss_weight: float = 0.0
    speed_loss_weight: float = 0.0
    direction_consistency_weight: float = 0.0
    action_consistency_weight: float = 0.0
    transition_action_rank_weight: float = 0.0
    transition_action_rank_margin: float = 1.0
    movement_onset_rank_weight: float = 0.0
    movement_speed_change_rank_weight: float = 0.0
    motion_boundary_rank_weight: float = 0.0
    motion_boundary_rank_margin: float = 1.0
    motion_boundary_rank_lookback: int = 3
    safety_correction_pairwise_rank_weight: float = 0.0
    safety_correction_pairwise_rank_margin: float = 0.25
    safety_correction_top1_rank_weight: float = 0.0
    safety_correction_top1_rank_margin: float = 0.25
    safety_correction_minimal_edit_weight: float = 0.0
    safety_correction_minimal_edit_margin: float = 0.25
    soft_action_loss_weight: float = 0.0
    soft_action_collision_rank_weight: float = 0.0
    soft_action_collision_rank_margin: float = 1.0
    soft_action_temperature: float = 4.0
    soft_action_safety_margin: float = 12.0
    initial_policy_kl_weight: float = 0.0
    correction_only: bool = False
    policy_head_only: bool = False
    previous_action_dropout_probability: float = 0.0
    future_visual_loss_weight: float = 0.0
    future_visual_horizons: tuple[int, ...] = DEFAULT_FUTURE_VISUAL_HORIZONS

    def __post_init__(self) -> None:
        if not isinstance(self.policy_head_only, bool):
            raise ValueError("policy_head_only must be a boolean")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.chunk_length <= 0:
            raise ValueError("chunk_length must be positive")
        positive = (
            self.learning_rate,
            self.gradient_clip,
            self.movement_onset_weight,
            self.movement_stop_weight,
            self.movement_speed_change_weight,
            self.direction_change_weight,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("positive training controls must be finite and positive")
        nonnegative = (
            self.weight_decay,
            self.risk_loss_weight,
            self.future_visual_loss_weight,
        )
        if not all(math.isfinite(value) and value >= 0.0 for value in nonnegative):
            raise ValueError(
                "nonnegative training controls must be finite and nonnegative"
            )
        action_loss_weights = (
            self.exact_action_loss_weight,
            self.direction_loss_weight,
            self.speed_loss_weight,
            self.direction_consistency_weight,
            self.action_consistency_weight,
            self.transition_action_rank_weight,
            self.movement_onset_rank_weight,
            self.movement_speed_change_rank_weight,
            self.motion_boundary_rank_weight,
            self.safety_correction_pairwise_rank_weight,
            self.safety_correction_top1_rank_weight,
            self.safety_correction_minimal_edit_weight,
            self.soft_action_loss_weight,
            self.soft_action_collision_rank_weight,
            self.initial_policy_kl_weight,
        )
        if not all(
            math.isfinite(value) and value >= 0.0
            for value in action_loss_weights
        ):
            raise ValueError("action-loss weights must be finite and nonnegative")
        if not any((
            self.exact_action_loss_weight,
            self.direction_loss_weight,
            self.speed_loss_weight,
            self.soft_action_loss_weight,
            self.safety_correction_pairwise_rank_weight,
            self.safety_correction_top1_rank_weight,
            self.safety_correction_minimal_edit_weight,
        )):
            raise ValueError("at least one supervised action-loss weight must be positive")
        if (
            not math.isfinite(self.transition_action_rank_margin)
            or self.transition_action_rank_margin < 0.0
        ):
            raise ValueError(
                "transition_action_rank_margin must be finite and nonnegative"
            )
        if (
            not math.isfinite(self.motion_boundary_rank_margin)
            or self.motion_boundary_rank_margin < 0.0
        ):
            raise ValueError(
                "motion_boundary_rank_margin must be finite and nonnegative"
            )
        if (
            isinstance(self.motion_boundary_rank_lookback, bool)
            or not isinstance(self.motion_boundary_rank_lookback, int)
            or not 1 <= self.motion_boundary_rank_lookback <= 3
        ):
            raise ValueError("motion_boundary_rank_lookback must be in [1, 3]")
        if self.motion_boundary_rank_weight > 0.0 and not self.episode_balanced:
            raise ValueError(
                "motion boundary ranking requires episode-balanced optimization"
            )
        if (
            not math.isfinite(self.safety_correction_pairwise_rank_margin)
            or self.safety_correction_pairwise_rank_margin < 0.0
        ):
            raise ValueError(
                "safety_correction_pairwise_rank_margin must be finite and "
                "nonnegative"
            )
        if (
            self.safety_correction_pairwise_rank_weight > 0.0
            and not self.episode_balanced
        ):
            raise ValueError(
                "safety correction pairwise ranking requires episode-balanced "
                "optimization"
            )
        if (
            not math.isfinite(self.safety_correction_top1_rank_margin)
            or self.safety_correction_top1_rank_margin < 0.0
        ):
            raise ValueError(
                "safety_correction_top1_rank_margin must be finite and "
                "nonnegative"
            )
        if (
            self.safety_correction_top1_rank_weight > 0.0
            and not self.episode_balanced
        ):
            raise ValueError(
                "safety correction top-1 ranking requires episode-balanced "
                "optimization"
            )
        if (
            not math.isfinite(self.safety_correction_minimal_edit_margin)
            or self.safety_correction_minimal_edit_margin < 0.0
        ):
            raise ValueError(
                "safety_correction_minimal_edit_margin must be finite and "
                "nonnegative"
            )
        if (
            self.safety_correction_minimal_edit_weight > 0.0
            and not self.episode_balanced
        ):
            raise ValueError(
                "safety correction minimal-edit training requires episode-balanced "
                "optimization"
            )
        if (
            self.soft_action_collision_rank_weight > 0.0
            and self.soft_action_loss_weight <= 0.0
        ):
            raise ValueError("soft action collision ranking requires soft action loss")
        if (
            not math.isfinite(self.soft_action_collision_rank_margin)
            or self.soft_action_collision_rank_margin < 0.0
        ):
            raise ValueError(
                "soft_action_collision_rank_margin must be finite and nonnegative"
            )
        if (
            not math.isfinite(self.soft_action_temperature)
            or self.soft_action_temperature <= 0.0
        ):
            raise ValueError("soft_action_temperature must be finite and positive")
        if (
            not math.isfinite(self.soft_action_safety_margin)
            or self.soft_action_safety_margin < 0.0
        ):
            raise ValueError(
                "soft_action_safety_margin must be finite and nonnegative"
            )
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in (0, 1)")
        if not 0.0 <= self.class_balance_power <= 1.0:
            raise ValueError("class_balance_power must be in [0, 1]")
        if not 0.0 <= self.horizontal_reflection_probability <= 1.0:
            raise ValueError("horizontal_reflection_probability must be in [0, 1]")
        if not 0.0 <= self.previous_action_dropout_probability <= 1.0:
            raise ValueError("previous_action_dropout_probability must be in [0, 1]")
        if self.policy_head_only and self.future_visual_loss_weight > 0.0:
            raise ValueError(
                "policy-head-only training cannot be combined with future visual "
                "prediction"
            )
        object.__setattr__(
            self,
            "future_visual_horizons",
            _normalize_future_visual_horizons(self.future_visual_horizons),
        )
        if self.validation_episode_ids is not None:
            normalized = tuple(int(value) for value in self.validation_episode_ids)
            if not normalized:
                raise ValueError("validation_episode_ids cannot be empty")
            if len(set(normalized)) != len(normalized):
                raise ValueError("validation_episode_ids cannot contain duplicates")
            object.__setattr__(self, "validation_episode_ids", normalized)


@dataclass(frozen=True, slots=True)
class EpisodeSequence:
    """One contiguous, archive-ordered episode block."""

    episode_id: int
    start: int
    stop: int

    @property
    def decisions(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True, slots=True)
class EpisodeSplit:
    train_episode_ids: tuple[int, ...]
    validation_episode_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _MotionBoundaryRankConstraints:
    """Two-sided old/new action comparisons grouped by motion event."""

    state_indices: np.ndarray
    preferred_actions: np.ndarray
    rejected_actions: np.ndarray
    pair_weights: np.ndarray
    event_ids: np.ndarray
    event_indices: np.ndarray
    event_episode_ids: np.ndarray
    event_kinds: tuple[str, ...]

    @property
    def pairs(self) -> int:
        return int(len(self.state_indices))

    @property
    def events(self) -> int:
        return int(len(self.event_indices))


@dataclass(frozen=True, slots=True)
class StatefulPassMetrics:
    loss: float
    action_accuracy: float
    risk_mae: float
    labels: int
    risk_labels: int
    decisions: int
    chunks: int
    episodes: int
    optimizer_steps: int
    movement_onsets: int
    direction_changes: int
    future_visual_loss: float
    future_visual_labels: int
    movement_stops: int = 0
    transition_action_rank_loss: float = 0.0
    transition_action_rank_labels: int = 0
    transition_action_rank_margin_satisfaction: float = 0.0
    movement_onset_rank_loss: float = 0.0
    movement_onset_rank_labels: int = 0
    movement_onset_rank_margin_satisfaction: float = 0.0
    movement_speed_changes: int = 0
    movement_speed_change_rank_loss: float = 0.0
    movement_speed_change_rank_labels: int = 0
    movement_speed_change_rank_margin_satisfaction: float = 0.0
    motion_boundary_rank_loss: float = 0.0
    motion_boundary_rank_events: int = 0
    motion_boundary_rank_pairs: int = 0
    motion_boundary_rank_margin_satisfaction: float = 0.0
    safety_correction_pairwise_rank_loss: float = 0.0
    safety_correction_pairwise_rank_labels: int = 0
    safety_correction_pairwise_rank_margin_satisfaction: float = 0.0
    safety_correction_top1_rank_loss: float = 0.0
    safety_correction_top1_rank_labels: int = 0
    safety_correction_top1_rank_margin_satisfaction: float = 0.0
    safety_correction_minimal_edit_loss: float = 0.0
    safety_correction_minimal_edit_labels: int = 0
    safety_correction_minimal_edit_margin_satisfaction: float = 0.0
    initial_policy_kl_loss: float = 0.0
    initial_policy_kl_labels: int = 0


if torch is not None:

    class _FutureVisualPredictor(torch.nn.Module):
        """Training-only linear probes from GRU state to future visual latents."""

        def __init__(
            self,
            recurrent_size: int,
            visual_size: int,
            horizons: tuple[int, ...],
        ) -> None:
            super().__init__()
            self.horizons = _normalize_future_visual_horizons(horizons)
            self.heads = torch.nn.ModuleDict({
                str(horizon): torch.nn.Linear(recurrent_size, visual_size)
                for horizon in self.horizons
            })

        def forward(self, recurrent: Tensor, horizon: int) -> Tensor:
            try:
                head = self.heads[str(horizon)]
            except KeyError as error:
                raise ValueError(
                    f"unconfigured future visual horizon {horizon}"
                ) from error
            return head(recurrent)


def ordered_episode_sequences(
    demonstrations: Demonstrations,
) -> tuple[EpisodeSequence, ...]:
    """Return episode blocks and reject archives that cannot be streamed online."""

    demonstrations.validate()
    episode_ids = demonstrations.episode_ids
    if episode_ids is None:
        raise ValueError("episode_ids are required for stateful training")
    if not np.issubdtype(episode_ids.dtype, np.integer) or np.issubdtype(
        episode_ids.dtype, np.bool_,
    ):
        raise ValueError("episode_ids must be an integer array")
    if len(episode_ids) == 0:
        raise ValueError("stateful training requires at least one decision")

    sequences: list[EpisodeSequence] = []
    seen: set[int] = set()
    block_start = 0
    current = int(episode_ids[0])
    seen.add(current)
    for sample in range(1, len(episode_ids)):
        episode_id = int(episode_ids[sample])
        if episode_id == current:
            continue
        sequences.append(EpisodeSequence(current, block_start, sample))
        if episode_id in seen:
            raise ValueError(
                f"samples for episode_id {episode_id} must form one contiguous block"
            )
        seen.add(episode_id)
        current = episode_id
        block_start = sample
    sequences.append(EpisodeSequence(current, block_start, len(episode_ids)))
    return tuple(sequences)


def split_episode_ids(
    demonstrations: Demonstrations,
    *,
    validation_fraction: float,
    seed: int,
    validation_episode_ids: Iterable[int] | None = None,
) -> EpisodeSplit:
    """Create a deterministic episode-level split without sequence leakage."""

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")
    sequences = ordered_episode_sequences(demonstrations)
    if len(sequences) < 2:
        raise ValueError("stateful training requires at least two complete episodes")
    ordered_ids = np.asarray([sequence.episode_id for sequence in sequences], dtype=np.int64)
    if validation_episode_ids is not None:
        validation_values = tuple(int(value) for value in validation_episode_ids)
        if not validation_values:
            raise ValueError("validation_episode_ids cannot be empty")
        if len(set(validation_values)) != len(validation_values):
            raise ValueError("validation_episode_ids cannot contain duplicates")
        known = {int(value) for value in ordered_ids}
        unknown = sorted(set(validation_values) - known)
        if unknown:
            raise ValueError(f"unknown validation episode_ids: {unknown}")
        validation_set = set(validation_values)
        if validation_set == known:
            raise ValueError("validation episodes cannot consume the complete dataset")
        return EpisodeSplit(
            train_episode_ids=tuple(
                int(value) for value in ordered_ids if int(value) not in validation_set
            ),
            validation_episode_ids=tuple(
                int(value) for value in ordered_ids if int(value) in validation_set
            ),
        )
    shuffled = np.random.default_rng(seed).permutation(ordered_ids)
    validation_count = min(
        len(ordered_ids) - 1,
        max(1, int(round(len(ordered_ids) * validation_fraction))),
    )
    validation_set = {int(value) for value in shuffled[:validation_count]}
    return EpisodeSplit(
        train_episode_ids=tuple(
            int(value) for value in ordered_ids if int(value) not in validation_set
        ),
        validation_episode_ids=tuple(
            int(value) for value in ordered_ids if int(value) in validation_set
        ),
    )


def _select_device(requested: str) -> str:
    if torch is None:
        raise RuntimeError("PyTorch is required for stateful policy training")
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


def _policy_dimensions(model: Any) -> tuple[int, int, int]:
    config = getattr(model, "config", None)
    if config is None:
        raise ValueError("stateful model must expose its policy config")
    if getattr(config, "inference_mode", None) != "stream":
        raise ValueError("stateful TBPTT requires policy inference_mode='stream'")
    memory_size = int(getattr(config, "memory_size", 0))
    proficiency_size = int(getattr(config, "proficiency_size", 0))
    action_count = int(getattr(config, "action_count", 18))
    if memory_size < 0 or proficiency_size < 0 or action_count <= 0:
        raise ValueError("policy feature and action dimensions are invalid")
    return memory_size, proficiency_size, action_count


def _validate_demonstration_features(
    demonstrations: Demonstrations,
    *,
    memory_size: int,
    proficiency_size: int,
) -> None:
    if demonstrations.memory is not None and demonstrations.memory.shape[-1] != memory_size:
        raise ValueError("demonstration memory width does not match the policy")
    if demonstrations.proficiency is not None:
        if demonstrations.proficiency.shape[-1] != proficiency_size:
            raise ValueError("demonstration proficiency width does not match the policy")
    elif proficiency_size not in {0, len(proficiency_vector("expert"))}:
        raise ValueError("missing demonstrations cannot supply this proficiency width")


def _episode_selection(
    sequences: Sequence[EpisodeSequence],
    episode_ids: Iterable[int],
) -> tuple[EpisodeSequence, ...]:
    by_id = {sequence.episode_id: sequence for sequence in sequences}
    selected: list[EpisodeSequence] = []
    seen: set[int] = set()
    for raw_episode_id in episode_ids:
        episode_id = int(raw_episode_id)
        if episode_id in seen:
            raise ValueError(f"episode_id {episode_id} was selected more than once")
        try:
            selected.append(by_id[episode_id])
        except KeyError as error:
            raise ValueError(f"unknown episode_id {episode_id}") from error
        seen.add(episode_id)
    if not selected:
        raise ValueError("at least one episode must be selected")
    return tuple(selected)


def _teacher_motion_transition_masks(
    demonstrations: Demonstrations,
    episodes: Sequence[EpisodeSequence],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Locate reliable transitions between consecutive exact-hold labels."""

    samples = demonstrations.actions.shape[0]
    movement_onsets = np.zeros(samples, dtype=np.bool_)
    movement_stops = np.zeros(samples, dtype=np.bool_)
    direction_changes = np.zeros(samples, dtype=np.bool_)
    movement_speed_changes = np.zeros(samples, dtype=np.bool_)
    supervised = (
        np.ones(samples, dtype=np.bool_)
        if demonstrations.supervision_mask is None else
        np.asarray(demonstrations.supervision_mask[:, -1], dtype=np.bool_)
    )
    supervised &= ~_latest_correction_mask(demonstrations, required=False)
    for episode in episodes:
        supervised_indices = np.flatnonzero(
            supervised[episode.start:episode.stop]
        ) + episode.start
        if len(supervised_indices) < 2:
            continue
        previous_indices = supervised_indices[:-1]
        current_indices = supervised_indices[1:]
        # exact-hold masks the one decision window containing an input change.
        # Bridge only that single mixed window; wider gaps (for example sparse
        # DAgger interventions) do not establish a temporal transition.
        reliable = current_indices - previous_indices <= 2
        previous_actions = np.asarray(
            demonstrations.actions[previous_indices, -1],
            dtype=np.int64,
        )
        current_actions = np.asarray(
            demonstrations.actions[current_indices, -1],
            dtype=np.int64,
        )
        previous = previous_actions % 9
        current = current_actions % 9
        previous_moving = previous != 4
        current_moving = current != 4
        movement_onsets[current_indices] = (
            reliable & current_moving & ~previous_moving
        )
        movement_stops[current_indices] = (
            reliable & ~current_moving & previous_moving
        )
        direction_changes[current_indices] = (
            reliable & current_moving & previous_moving & (current != previous)
        )
        movement_speed_changes[current_indices] = (
            reliable
            & current_moving
            & previous_moving
            & (current == previous)
            & (current_actions // 9 != previous_actions // 9)
        )
    return (
        movement_onsets,
        movement_stops,
        direction_changes,
        movement_speed_changes,
    )


def _teacher_transition_masks(
    demonstrations: Demonstrations,
    episodes: Sequence[EpisodeSequence],
) -> tuple[np.ndarray, np.ndarray]:
    """Return onset/change masks retained by the original public helper."""

    movement_onsets, _movement_stops, direction_changes, _movement_speed_changes = (
        _teacher_motion_transition_masks(demonstrations, episodes)
    )
    return movement_onsets, direction_changes


def _motion_boundary_kind(old_action: int, new_action: int) -> str | None:
    old_direction = old_action % 9
    new_direction = new_action % 9
    old_moving = old_direction != 4
    new_moving = new_direction != 4
    if new_moving and not old_moving:
        return "onset"
    if old_moving and not new_moving:
        return "stop"
    if old_moving and new_moving and old_direction != new_direction:
        return "turn"
    if (
        old_moving
        and new_moving
        and old_direction == new_direction
        and old_action // 9 != new_action // 9
    ):
        return "speed_change"
    return None


def _motion_boundary_rank_constraints(
    demonstrations: Demonstrations,
    episodes: Sequence[EpisodeSequence],
    *,
    lookback: int,
) -> _MotionBoundaryRankConstraints:
    """Build visual-state comparisons around reliable hard motion boundaries.

    Stored demonstration episodes must already have passed the caller's strict
    success admission. Teacher-evaluated rows are excluded even when the soft
    action objective is disabled: they are set-valued evidence, not hard motor
    labels. One intervening non-hard row may be bridged, matching exact-hold
    transition semantics.
    """

    if (
        isinstance(lookback, bool)
        or not isinstance(lookback, int)
        or not 1 <= lookback <= 3
    ):
        raise ValueError("motion boundary lookback must be in [1, 3]")
    demonstrations.validate()
    samples = demonstrations.actions.shape[0]
    supervised = (
        np.ones(samples, dtype=np.bool_)
        if demonstrations.supervision_mask is None else
        np.asarray(demonstrations.supervision_mask[:, -1], dtype=np.bool_)
    )
    teacher_evaluated = (
        np.zeros(samples, dtype=np.bool_)
        if demonstrations.teacher_action_evaluation_mask is None else
        np.asarray(
            demonstrations.teacher_action_evaluation_mask[:, -1],
            dtype=np.bool_,
        )
    )
    hard_supervised = (
        supervised
        & ~teacher_evaluated
        & ~_latest_correction_mask(demonstrations, required=False)
    )
    actions = np.asarray(demonstrations.actions[:, -1], dtype=np.int64)
    state_indices: list[int] = []
    preferred_actions: list[int] = []
    rejected_actions: list[int] = []
    pair_weights: list[float] = []
    event_ids: list[int] = []
    event_indices: list[int] = []
    event_episode_ids: list[int] = []
    event_kinds: list[str] = []

    for episode in episodes:
        hard_indices = (
            np.flatnonzero(hard_supervised[episode.start:episode.stop])
            + episode.start
        )
        for position in range(1, len(hard_indices)):
            current_index = int(hard_indices[position])
            previous_index = int(hard_indices[position - 1])
            if current_index - previous_index > 2:
                continue
            old_action = int(actions[previous_index])
            new_action = int(actions[current_index])
            kind = _motion_boundary_kind(old_action, new_action)
            if kind is None:
                continue

            prior_indices: list[int] = []
            next_index = current_index
            cursor = position - 1
            while cursor >= 0 and len(prior_indices) < lookback:
                prior_index = int(hard_indices[cursor])
                if (
                    next_index - prior_index > 2
                    or int(actions[prior_index]) != old_action
                ):
                    break
                prior_indices.append(prior_index)
                next_index = prior_index
                cursor -= 1
            if not prior_indices:
                continue

            event_id = len(event_indices)
            # Keep the two sides of the boundary equally important.  Giving
            # every pair the same weight would let a three-state lookback put
            # 3/4 of the event mass on "keep holding" and systematically
            # delay the transition that the event-side pair is meant to learn.
            prior_pair_weight = 0.5 / len(prior_indices)
            for prior_index in reversed(prior_indices):
                state_indices.append(prior_index)
                preferred_actions.append(old_action)
                rejected_actions.append(new_action)
                pair_weights.append(prior_pair_weight)
                event_ids.append(event_id)
            state_indices.append(current_index)
            preferred_actions.append(new_action)
            rejected_actions.append(old_action)
            pair_weights.append(0.5)
            event_ids.append(event_id)
            event_indices.append(current_index)
            event_episode_ids.append(episode.episode_id)
            event_kinds.append(kind)

    return _MotionBoundaryRankConstraints(
        state_indices=np.asarray(state_indices, dtype=np.int64),
        preferred_actions=np.asarray(preferred_actions, dtype=np.int64),
        rejected_actions=np.asarray(rejected_actions, dtype=np.int64),
        pair_weights=np.asarray(pair_weights, dtype=np.float32),
        event_ids=np.asarray(event_ids, dtype=np.int64),
        event_indices=np.asarray(event_indices, dtype=np.int64),
        event_episode_ids=np.asarray(event_episode_ids, dtype=np.int64),
        event_kinds=tuple(event_kinds),
    )


def teacher_transition_sample_weights(
    demonstrations: Demonstrations,
    *,
    episodes: Sequence[EpisodeSequence] | None = None,
    movement_onset_weight: float = 1.0,
    movement_stop_weight: float = 1.0,
    movement_speed_change_weight: float = 1.0,
    direction_change_weight: float = 1.0,
) -> np.ndarray:
    """Weight reliable teacher motion transitions per episode.

    Slow-mode changes do not count as direction changes, and the first action in
    an episode has no inferred transition. One masked mixed-action window may be
    bridged; sparse interventions are not treated as adjacent motor transitions.
    """

    demonstrations.validate()
    for name, value in (
        ("movement_onset_weight", movement_onset_weight),
        ("movement_stop_weight", movement_stop_weight),
        ("movement_speed_change_weight", movement_speed_change_weight),
        ("direction_change_weight", direction_change_weight),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    selected = ordered_episode_sequences(demonstrations) if episodes is None else episodes
    movement_onsets, movement_stops, direction_changes, movement_speed_changes = (
        _teacher_motion_transition_masks(
            demonstrations, selected,
        )
    )
    weights = np.ones(demonstrations.actions.shape[0], dtype=np.float32)
    weights[movement_onsets] = movement_onset_weight
    weights[movement_stops] = movement_stop_weight
    weights[direction_changes] = direction_change_weight
    weights[movement_speed_changes] = movement_speed_change_weight
    return weights


def _hard_action_ranking_terms(
    logits: Tensor,
    actions: Tensor,
    *,
    margin: float,
) -> Tensor:
    """Require the labelled action to outrank every alternative action."""

    if not math.isfinite(margin) or margin < 0.0:
        raise ValueError("hard action ranking margin must be finite and nonnegative")
    if logits.shape[:-1] != actions.shape:
        raise ValueError("hard action ranking logits and labels do not align")
    if logits.shape[-1] < 2:
        raise ValueError("hard action ranking requires at least two actions")
    labelled = logits.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    alternatives = logits.masked_fill(
        F.one_hot(actions, num_classes=logits.shape[-1]).to(torch.bool),
        -torch.inf,
    ).amax(dim=-1)
    return F.relu(margin + alternatives - labelled)


def _motion_boundary_ranking_terms(
    logits: Tensor,
    preferred_actions: Tensor,
    rejected_actions: Tensor,
    *,
    margin: float,
) -> Tensor:
    """Require the old/new action ordering appropriate to one boundary side."""

    if not math.isfinite(margin) or margin < 0.0:
        raise ValueError(
            "motion boundary ranking margin must be finite and nonnegative"
        )
    if logits.ndim != 2:
        raise ValueError("motion boundary logits must have [pair, action]")
    if (
        preferred_actions.shape != logits.shape[:1]
        or rejected_actions.shape != logits.shape[:1]
    ):
        raise ValueError("motion boundary actions must align with pair logits")
    if (
        preferred_actions.dtype == torch.bool
        or preferred_actions.is_floating_point()
    ):
        raise ValueError("preferred motion boundary actions must be integer ids")
    if (
        rejected_actions.dtype == torch.bool
        or rejected_actions.is_floating_point()
    ):
        raise ValueError("rejected motion boundary actions must be integer ids")
    if (
        torch.any(preferred_actions < 0)
        or torch.any(preferred_actions >= logits.shape[-1])
        or torch.any(rejected_actions < 0)
        or torch.any(rejected_actions >= logits.shape[-1])
    ):
        raise ValueError("motion boundary action ids are outside the policy vocabulary")
    preferred = logits.gather(-1, preferred_actions[:, None]).squeeze(-1)
    rejected = logits.gather(-1, rejected_actions[:, None]).squeeze(-1)
    return F.relu(margin + rejected - preferred)


def _safety_correction_pairwise_ranking_terms(
    logits: Tensor,
    preferred_actions: Tensor,
    rejected_actions: Tensor,
    *,
    margin: float,
) -> Tensor:
    """Rank each safety correction only against the frozen parent's choice."""

    if not math.isfinite(margin) or margin < 0.0:
        raise ValueError(
            "safety correction pairwise ranking margin must be finite and "
            "nonnegative"
        )
    if logits.ndim != 2:
        raise ValueError("safety correction logits must have [correction, action]")
    if (
        preferred_actions.shape != logits.shape[:1]
        or rejected_actions.shape != logits.shape[:1]
    ):
        raise ValueError(
            "safety correction actions must align with correction logits"
        )
    for name, actions in (
        ("preferred", preferred_actions),
        ("rejected", rejected_actions),
    ):
        if actions.dtype == torch.bool or actions.is_floating_point():
            raise ValueError(f"{name} safety correction actions must be integer ids")
        if torch.any(actions < 0) or torch.any(actions >= logits.shape[-1]):
            raise ValueError(
                f"{name} safety correction action ids are outside the policy "
                "vocabulary"
            )
    preferred = logits.gather(-1, preferred_actions[:, None]).squeeze(-1)
    rejected = logits.gather(-1, rejected_actions[:, None]).squeeze(-1)
    return F.relu(margin - preferred + rejected)


def _safety_correction_top1_ranking_terms(
    logits: Tensor,
    preferred_actions: Tensor,
    *,
    margin: float,
) -> Tensor:
    """Require each correction to outrank its strongest current alternative."""

    if logits.ndim != 2:
        raise ValueError(
            "safety correction top-1 logits must have [correction, action]"
        )
    return _hard_action_ranking_terms(
        logits,
        preferred_actions,
        margin=margin,
    )


def _safety_correction_minimal_edit_target_logits(
    reference_logits: Tensor,
    preferred_actions: Tensor,
    *,
    margin: float,
) -> Tensor:
    """Raise only the correction action above the frozen parent's maximum."""

    if not math.isfinite(margin) or margin < 0.0:
        raise ValueError(
            "safety correction minimal-edit margin must be finite and nonnegative"
        )
    if reference_logits.ndim != 2:
        raise ValueError(
            "safety correction reference logits must have [correction, action]"
        )
    if preferred_actions.shape != reference_logits.shape[:1]:
        raise ValueError(
            "preferred safety correction actions must align with reference logits"
        )
    if preferred_actions.dtype == torch.bool or preferred_actions.is_floating_point():
        raise ValueError("preferred safety correction actions must be integer ids")
    if (
        torch.any(preferred_actions < 0)
        or torch.any(preferred_actions >= reference_logits.shape[-1])
    ):
        raise ValueError(
            "preferred safety correction action ids are outside the policy vocabulary"
        )
    frozen = reference_logits.detach()
    target = frozen.clone()
    preferred_targets = frozen.amax(dim=-1) + margin
    target.scatter_(1, preferred_actions[:, None], preferred_targets[:, None])
    return target


def _safety_correction_minimal_edit_terms(
    logits: Tensor,
    reference_logits: Tensor,
    preferred_actions: Tensor,
    *,
    margin: float,
) -> Tensor:
    """Match a minimally edited frozen-parent distribution on corrections."""

    if logits.shape != reference_logits.shape:
        raise ValueError(
            "current and reference safety correction logits must align"
        )
    target_logits = _safety_correction_minimal_edit_target_logits(
        reference_logits,
        preferred_actions,
        margin=margin,
    )
    return F.kl_div(
        F.log_softmax(logits, dim=-1),
        F.softmax(target_logits, dim=-1),
        reduction="none",
    ).sum(dim=-1)


def _latest_correction_mask(
    demonstrations: Demonstrations,
    *,
    required: bool,
) -> np.ndarray:
    """Return explicit latest-frame safety corrections without inferring episodes."""

    value = getattr(demonstrations, "correction_mask", None)
    if value is None:
        if required:
            raise ValueError(
                "safety correction training requires a correction_mask"
            )
        return np.zeros(demonstrations.actions.shape[0], dtype=np.bool_)
    mask = np.asarray(value)
    if mask.shape != demonstrations.actions.shape:
        raise ValueError("correction_mask must align with actions")
    if not np.issubdtype(mask.dtype, np.bool_):
        raise ValueError("correction_mask must be boolean")
    latest = np.asarray(mask[:, -1], dtype=np.bool_)
    if required and not latest.any():
        raise ValueError(
            "safety correction training requires at least one marked "
            "latest-frame correction"
        )
    return latest


def _episode_supervised_labels(
    demonstrations: Demonstrations,
    episode: EpisodeSequence,
) -> int:
    if demonstrations.supervision_mask is None:
        return episode.decisions
    return int(
        np.asarray(
            demonstrations.supervision_mask[episode.start:episode.stop, -1],
            dtype=np.bool_,
        ).sum()
    )


def _episode_future_visual_labels(
    episode: EpisodeSequence,
    horizons: tuple[int, ...],
) -> int:
    return sum(max(episode.decisions - horizon, 0) for horizon in horizons)


def _chunk_future_visual_labels(
    episode: EpisodeSequence,
    start: int,
    stop: int,
    horizons: tuple[int, ...],
) -> int:
    return sum(
        max(min(stop, episode.stop - horizon) - start, 0)
        for horizon in horizons
    )


def _optimizer_steps_per_epoch(
    demonstrations: Demonstrations,
    episodes: Sequence[EpisodeSequence],
    *,
    chunk_length: int,
    episode_balanced: bool,
    risk_loss_weight: float,
    risk_on_all_decisions: bool,
    future_visual_loss_weight: float,
    future_visual_horizons: tuple[int, ...],
    initial_policy_kl_weight: float = 0.0,
    hard_action_terms_enabled: bool = True,
    soft_action_loss_enabled: bool = False,
    transition_action_rank_weight: float = 0.0,
    movement_onset_rank_weight: float = 0.0,
    movement_speed_change_rank_weight: float = 0.0,
    motion_boundary_rank_weight: float = 0.0,
    motion_boundary_rank_lookback: int = 3,
    safety_correction_pairwise_rank_weight: float = 0.0,
    safety_correction_top1_rank_weight: float = 0.0,
    safety_correction_minimal_edit_weight: float = 0.0,
) -> int:
    correction_objective_enabled = (
        safety_correction_pairwise_rank_weight > 0.0
        or safety_correction_top1_rank_weight > 0.0
        or safety_correction_minimal_edit_weight > 0.0
    )
    if correction_objective_enabled and not episode_balanced:
        raise ValueError(
            "safety correction training requires episode-balanced "
            "optimization"
        )
    correction_mask = _latest_correction_mask(
        demonstrations,
        required=correction_objective_enabled,
    )
    motion_transitions = np.zeros(
        demonstrations.actions.shape[0], dtype=np.bool_,
    )
    movement_onsets = np.zeros_like(motion_transitions)
    movement_speed_changes = np.zeros_like(motion_transitions)
    if (
        transition_action_rank_weight > 0.0
        or movement_onset_rank_weight > 0.0
        or movement_speed_change_rank_weight > 0.0
    ):
        movement_onsets, stops, direction_changes, movement_speed_changes = (
            _teacher_motion_transition_masks(demonstrations, episodes)
        )
        motion_transitions = (
            movement_onsets | stops | direction_changes | movement_speed_changes
        )
    motion_boundary_constraints = (
        _motion_boundary_rank_constraints(
            demonstrations,
            episodes,
            lookback=motion_boundary_rank_lookback,
        )
        if motion_boundary_rank_weight > 0.0 else
        None
    )
    steps = 0
    for episode in episodes:
        episode_has_objective = False
        for start in range(episode.start, episode.stop, chunk_length):
            stop = min(start + chunk_length, episode.stop)
            if demonstrations.supervision_mask is None:
                supervision_mask = np.ones(stop - start, dtype=np.bool_)
            else:
                supervision_mask = np.asarray(
                    demonstrations.supervision_mask[start:stop, -1],
                    dtype=np.bool_,
                ).copy()
            soft_action_mask = np.zeros(stop - start, dtype=np.bool_)
            if soft_action_loss_enabled:
                assert demonstrations.teacher_action_evaluation_mask is not None
                soft_action_mask = np.asarray(
                    demonstrations.teacher_action_evaluation_mask[start:stop, -1],
                    dtype=np.bool_,
                ) & ~correction_mask[start:stop]
            hard_action_mask = (
                supervision_mask
                & ~soft_action_mask
                & ~correction_mask[start:stop]
                if hard_action_terms_enabled else
                np.zeros_like(supervision_mask)
            )
            action_labels = int(np.count_nonzero(
                hard_action_mask | soft_action_mask
            ))
            rank_mask = (
                supervision_mask
                & ~soft_action_mask
                & ~correction_mask[start:stop]
                & (
                (
                    motion_transitions[start:stop]
                    if transition_action_rank_weight > 0.0 else
                    False
                )
                | (
                    movement_onsets[start:stop]
                    if movement_onset_rank_weight > 0.0 else
                    False
                )
                | (
                    movement_speed_changes[start:stop]
                    if movement_speed_change_rank_weight > 0.0 else
                    False
                )
                )
            )
            rank_labels = int(np.count_nonzero(rank_mask))
            motion_boundary_pairs = (
                0
                if motion_boundary_constraints is None else
                int(np.count_nonzero(
                    (motion_boundary_constraints.state_indices >= start)
                    & (motion_boundary_constraints.state_indices < stop)
                ))
            )
            correction_labels = (
                int(np.count_nonzero(correction_mask[start:stop]))
                if correction_objective_enabled else
                0
            )
            initial_policy_kl_labels = int(np.count_nonzero(
                ~correction_mask[start:stop]
            ))
            risk_labels = (
                stop - start
                if risk_on_all_decisions else
                int(np.count_nonzero(supervision_mask))
            )
            future_labels = _chunk_future_visual_labels(
                episode,
                start,
                stop,
                future_visual_horizons,
            )
            chunk_has_objective = bool(
                action_labels
                or rank_labels
                or motion_boundary_pairs
                or correction_labels
                or (risk_loss_weight > 0.0 and risk_labels)
                or (future_visual_loss_weight > 0.0 and future_labels)
                or (
                    initial_policy_kl_weight > 0.0
                    and initial_policy_kl_labels
                )
            )
            if episode_balanced:
                episode_has_objective |= chunk_has_objective
            else:
                steps += int(chunk_has_objective)
        if episode_balanced:
            steps += int(episode_has_objective)
    return steps


def _chunk_tensors(
    demonstrations: Demonstrations,
    start: int,
    stop: int,
    *,
    memory_size: int,
    proficiency_size: int,
    device: str,
) -> tuple[
    Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,
    Tensor, Tensor, Tensor,
]:
    if torch is None:  # pragma: no cover - guarded by public entry points
        raise RuntimeError("PyTorch is required for stateful policy training")
    # Each archive sample is one decision window. Streaming inference consumes
    # exactly its newest visible frame and carries GRU state to the next sample.
    global_frames = torch.as_tensor(
        demonstrations.global_frames[start:stop, -1], dtype=torch.float32,
        device=device,
    ).unsqueeze(0)
    local_frames = torch.as_tensor(
        demonstrations.local_frames[start:stop, -1], dtype=torch.float32,
        device=device,
    ).unsqueeze(0)
    decisions = stop - start
    if demonstrations.memory is None:
        memory = torch.zeros(
            (1, decisions, memory_size), dtype=torch.float32, device=device,
        )
    else:
        memory = torch.as_tensor(
            demonstrations.memory[start:stop, -1], dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
    if demonstrations.proficiency is not None:
        proficiency = torch.as_tensor(
            demonstrations.proficiency[start:stop, -1], dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
    elif proficiency_size == 0:
        proficiency = torch.empty(
            (1, decisions, 0), dtype=torch.float32, device=device,
        )
    else:
        default = torch.as_tensor(
            proficiency_vector("expert"), dtype=torch.float32, device=device,
        )
        proficiency = default.view(1, 1, -1).expand(1, decisions, -1)
    actions = torch.as_tensor(
        demonstrations.actions[start:stop, -1], dtype=torch.long, device=device,
    ).unsqueeze(0)
    risks = torch.as_tensor(
        demonstrations.risks[start:stop, -1], dtype=torch.float32, device=device,
    ).unsqueeze(0)
    if demonstrations.supervision_mask is None:
        mask = torch.ones((1, decisions), dtype=torch.bool, device=device)
    else:
        mask = torch.as_tensor(
            demonstrations.supervision_mask[start:stop, -1],
            dtype=torch.bool,
            device=device,
        ).unsqueeze(0)
    if demonstrations.teacher_action_evaluations is None:
        teacher_evaluations = torch.zeros(
            (
                1,
                decisions,
                18,
                len(TEACHER_ACTION_EVALUATION_FIELDS),
            ),
            dtype=torch.float32,
            device=device,
        )
        teacher_regrets = torch.zeros(
            (1, decisions, 18), dtype=torch.float32, device=device,
        )
        teacher_evaluation_mask = torch.zeros(
            (1, decisions), dtype=torch.bool, device=device,
        )
    else:
        assert demonstrations.teacher_action_regrets is not None
        assert demonstrations.teacher_action_evaluation_mask is not None
        teacher_evaluations = torch.as_tensor(
            demonstrations.teacher_action_evaluations[start:stop, -1],
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
        teacher_regrets = torch.as_tensor(
            demonstrations.teacher_action_regrets[start:stop, -1],
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
        teacher_evaluation_mask = torch.as_tensor(
            demonstrations.teacher_action_evaluation_mask[start:stop, -1],
            dtype=torch.bool,
            device=device,
        ).unsqueeze(0)
    return (
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
    )


_HORIZONTAL_ACTIONS = tuple(
    Action(
        move_x=-Action.from_discrete(value).move_x,
        move_y=Action.from_discrete(value).move_y,
        slow=Action.from_discrete(value).slow,
    ).discrete
    for value in range(18)
)


def _reflect_horizontal_visual_batch(
    global_frames: Tensor,
    local_frames: Tensor,
) -> tuple[Tensor, Tensor]:
    if torch is None:
        raise RuntimeError("PyTorch is required for stream augmentation")
    if global_frames.ndim != 5 or local_frames.ndim != 5:
        raise ValueError("stream frames must have [batch, time, channel, height, width]")
    if global_frames.shape[:3] != local_frames.shape[:3]:
        raise ValueError("global and local stream frames must align")
    if global_frames.shape[2] < 2:
        raise ValueError("stream frames must include horizontal motion channel 1")
    reflected_global = torch.flip(global_frames, dims=(-1,)).clone()
    reflected_local = torch.flip(local_frames, dims=(-1,)).clone()
    reflected_global[:, :, 1].neg_()
    reflected_local[:, :, 1].neg_()
    return reflected_global, reflected_local


def reflect_horizontal_stream_batch(
    global_frames: Tensor,
    local_frames: Tensor,
    actions: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Mirror visible geometry, observable x motion, and supervised actions."""

    if actions.shape != global_frames.shape[:2] or actions.shape != local_frames.shape[:2]:
        raise ValueError("stream actions must align with frame batch and time")
    reflected_global, reflected_local = _reflect_horizontal_visual_batch(
        global_frames,
        local_frames,
    )
    lookup = torch.as_tensor(
        _HORIZONTAL_ACTIONS,
        dtype=torch.long,
        device=actions.device,
    )
    return reflected_global, reflected_local, lookup[actions]


def reflect_horizontal_teacher_action_evidence(
    evaluations: Tensor,
    regrets: Tensor,
) -> tuple[Tensor, Tensor]:
    """Remap the 18-way candidate axis for horizontal augmentation."""

    if evaluations.shape[:-2] != regrets.shape[:-1]:
        raise ValueError("teacher evaluations and regrets do not align")
    if evaluations.shape[-2] != 18 or regrets.shape[-1] != 18:
        raise ValueError("teacher action evidence must use 18 actions")
    lookup = torch.as_tensor(
        _HORIZONTAL_ACTIONS,
        dtype=torch.long,
        device=regrets.device,
    )
    return (
        evaluations.index_select(-2, lookup),
        regrets.index_select(-1, lookup),
    )


def _future_visual_loss(
    model: Any,
    predictor: Any,
    recurrent: Tensor,
    demonstrations: Demonstrations,
    episode: EpisodeSequence,
    start: int,
    stop: int,
    *,
    horizons: tuple[int, ...],
    device: str,
    reflect_episode: bool,
) -> tuple[Tensor, int]:
    """Predict detached future visual latents without crossing an episode edge."""

    if torch is None:  # pragma: no cover - guarded by the training entry point
        raise RuntimeError("PyTorch is required for future visual prediction")
    encoder = getattr(model, "encode_visual", None)
    if not callable(encoder):
        raise ValueError(
            "future visual prediction requires a policy encode_visual method"
        )
    total = recurrent.sum() * 0.0
    labels = 0
    ranges: list[tuple[int, int, int]] = []
    for horizon in horizons:
        source_stop = min(stop, episode.stop - horizon)
        if source_stop <= start:
            continue
        target_start = start + horizon
        target_stop = source_stop + horizon
        ranges.append((horizon, target_start, target_stop))
    if not ranges:
        return total, labels

    bank_start = min(target_start for _horizon, target_start, _stop in ranges)
    bank_stop = max(target_stop for _horizon, _start, target_stop in ranges)
    target_global = torch.as_tensor(
        demonstrations.global_frames[bank_start:bank_stop, -1],
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)
    target_local = torch.as_tensor(
        demonstrations.local_frames[bank_start:bank_stop, -1],
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)
    if reflect_episode:
        target_global, target_local = _reflect_horizontal_visual_batch(
            target_global,
            target_local,
        )
    with torch.no_grad():
        target_bank = encoder(target_global, target_local).detach()

    for horizon, target_start, target_stop in ranges:
        target = target_bank[
            :,
            target_start - bank_start:target_stop - bank_start,
        ]
        source_stop = start + (target_stop - target_start)
        source = recurrent[:, :source_stop - start]
        predicted = predictor(source, horizon)
        if predicted.shape != target.shape:
            raise ValueError(
                "future visual predictor output does not match detached visual target"
            )
        terms = F.smooth_l1_loss(predicted, target, reduction="none").mean(dim=-1)
        total = total + terms.sum()
        labels += terms.numel()
    return total, labels


def reflect_horizontal_action_context(
    memory: Tensor,
    previous_actions: Tensor,
    *,
    previous_action_offset: int,
) -> Tensor:
    """Mirror verified previous-action one-hot context without changing identity."""

    if torch is None:
        raise RuntimeError("PyTorch is required for stream augmentation")
    if memory.ndim != 3:
        raise ValueError("stream memory must have [batch, time, feature]")
    if previous_actions.shape != memory.shape[:2]:
        raise ValueError("previous actions must align with stream memory")
    if previous_actions.dtype == torch.bool or previous_actions.is_floating_point():
        raise ValueError("previous actions must contain integer action ids")
    if torch.any(previous_actions < -1) or torch.any(previous_actions >= 18):
        raise ValueError("previous actions must contain -1 or action ids in [0, 18)")
    if (
        isinstance(previous_action_offset, bool)
        or not isinstance(previous_action_offset, int)
        or previous_action_offset < 0
        or previous_action_offset + 18 != memory.shape[-1]
    ):
        raise ValueError("previous action context must be the final 18 memory entries")
    lookup = torch.as_tensor(
        _HORIZONTAL_ACTIONS,
        dtype=torch.long,
        device=memory.device,
    )
    reflected = memory.clone()
    context = memory[..., previous_action_offset:previous_action_offset + 18]
    known = previous_actions >= 0
    expected = F.one_hot(previous_actions.clamp_min(0), num_classes=18).to(
        dtype=context.dtype,
    )
    expected = expected * known.unsqueeze(-1).to(dtype=context.dtype)
    if not torch.allclose(context, expected, rtol=0.0, atol=1e-6):
        raise ValueError(
            "previous action memory does not match recorded previous_actions"
        )
    reflected[..., previous_action_offset:previous_action_offset + 18] = (
        context.index_select(-1, lookup)
    )
    return reflected


def drop_previous_action_context(
    memory: Tensor,
    dropout_mask: Tensor,
    *,
    previous_action_offset: int,
) -> Tensor:
    """Hide selected executed-action inputs while preserving scenario identity.

    This is training-only input corruption. It prevents the recurrent policy from
    treating a teacher-forced previous action as an infallible phase cue, while the
    GRU still receives the complete visual episode and learns its own latent state.
    """

    if torch is None:
        raise RuntimeError("PyTorch is required for stream augmentation")
    if memory.ndim != 3:
        raise ValueError("stream memory must have [batch, time, feature]")
    if dropout_mask.shape != memory.shape[:2] or dropout_mask.dtype != torch.bool:
        raise ValueError("previous action dropout mask must be boolean [batch, time]")
    if (
        isinstance(previous_action_offset, bool)
        or not isinstance(previous_action_offset, int)
        or previous_action_offset < 0
        or previous_action_offset + 18 != memory.shape[-1]
    ):
        raise ValueError("previous action context must be the final 18 memory entries")
    dropped = memory.clone()
    action_context = dropped[..., previous_action_offset:previous_action_offset + 18]
    action_context.masked_fill_(dropout_mask.unsqueeze(-1), 0.0)
    return dropped


def _detach_hidden(hidden: Any) -> Any:
    if hidden is None:
        return None
    if torch is not None and isinstance(hidden, torch.Tensor):
        return hidden.detach()
    if isinstance(hidden, tuple):
        return tuple(_detach_hidden(value) for value in hidden)
    if isinstance(hidden, list):
        return [_detach_hidden(value) for value in hidden]
    raise TypeError("policy hidden state must be a tensor or tensor sequence")


def _stateful_pass(
    model: Any,
    demonstrations: Demonstrations,
    episodes: Sequence[EpisodeSequence],
    *,
    chunk_length: int,
    risk_loss_weight: float,
    gradient_clip: float,
    device: str,
    optimizer: Any | None,
    action_weights: Tensor | None = None,
    horizontal_reflection_probability: float = 0.0,
    augmentation_rng: random.Random | None = None,
    movement_onset_weight: float = 1.0,
    movement_stop_weight: float = 1.0,
    movement_speed_change_weight: float = 1.0,
    direction_change_weight: float = 1.0,
    episode_balanced: bool = False,
    exact_action_loss_weight: float = 1.0,
    direction_loss_weight: float = 0.0,
    speed_loss_weight: float = 0.0,
    direction_consistency_weight: float = 0.0,
    action_consistency_weight: float = 0.0,
    transition_action_rank_weight: float = 0.0,
    transition_action_rank_margin: float = 1.0,
    movement_onset_rank_weight: float = 0.0,
    movement_speed_change_rank_weight: float = 0.0,
    motion_boundary_rank_weight: float = 0.0,
    motion_boundary_rank_margin: float = 1.0,
    motion_boundary_rank_lookback: int = 3,
    safety_correction_pairwise_rank_weight: float = 0.0,
    safety_correction_pairwise_rank_margin: float = 0.25,
    safety_correction_top1_rank_weight: float = 0.0,
    safety_correction_top1_rank_margin: float = 0.25,
    safety_correction_minimal_edit_weight: float = 0.0,
    safety_correction_minimal_edit_margin: float = 0.25,
    soft_action_loss_weight: float = 0.0,
    soft_action_collision_rank_weight: float = 0.0,
    soft_action_collision_rank_margin: float = 1.0,
    soft_action_temperature: float = 4.0,
    soft_action_safety_margin: float = 12.0,
    initial_policy_kl_weight: float = 0.0,
    reference_model: Any | None = None,
    risk_on_all_decisions: bool = False,
    previous_action_dropout_probability: float = 0.0,
    future_visual_loss_weight: float = 0.0,
    future_visual_horizons: tuple[int, ...] = DEFAULT_FUTURE_VISUAL_HORIZONS,
    future_visual_predictor: Any | None = None,
) -> StatefulPassMetrics:
    if torch is None:
        raise RuntimeError("PyTorch is required for stateful policy training")
    memory_size, proficiency_size, action_count = _policy_dimensions(model)
    horizons = _normalize_future_visual_horizons(future_visual_horizons)
    if (
        not math.isfinite(future_visual_loss_weight)
        or future_visual_loss_weight < 0.0
    ):
        raise ValueError("future_visual_loss_weight must be finite and nonnegative")
    if future_visual_loss_weight > 0.0:
        if future_visual_predictor is None:
            raise ValueError(
                "future visual loss requires a future visual predictor"
            )
        if not callable(getattr(model, "forward_with_recurrent", None)):
            raise ValueError(
                "future visual prediction requires per-decision GRU hidden states"
            )
    if not math.isfinite(initial_policy_kl_weight) or initial_policy_kl_weight < 0.0:
        raise ValueError("initial_policy_kl_weight must be finite and nonnegative")
    if initial_policy_kl_weight > 0.0 and reference_model is None:
        raise ValueError("initial policy KL requires a frozen reference model")
    if (
        not math.isfinite(safety_correction_pairwise_rank_weight)
        or safety_correction_pairwise_rank_weight < 0.0
    ):
        raise ValueError(
            "safety_correction_pairwise_rank_weight must be finite and nonnegative"
        )
    if (
        not math.isfinite(safety_correction_pairwise_rank_margin)
        or safety_correction_pairwise_rank_margin < 0.0
    ):
        raise ValueError(
            "safety_correction_pairwise_rank_margin must be finite and nonnegative"
        )
    if (
        not math.isfinite(safety_correction_top1_rank_weight)
        or safety_correction_top1_rank_weight < 0.0
    ):
        raise ValueError(
            "safety_correction_top1_rank_weight must be finite and nonnegative"
        )
    if (
        not math.isfinite(safety_correction_top1_rank_margin)
        or safety_correction_top1_rank_margin < 0.0
    ):
        raise ValueError(
            "safety_correction_top1_rank_margin must be finite and nonnegative"
        )
    if (
        not math.isfinite(safety_correction_minimal_edit_weight)
        or safety_correction_minimal_edit_weight < 0.0
    ):
        raise ValueError(
            "safety_correction_minimal_edit_weight must be finite and nonnegative"
        )
    if (
        not math.isfinite(safety_correction_minimal_edit_margin)
        or safety_correction_minimal_edit_margin < 0.0
    ):
        raise ValueError(
            "safety_correction_minimal_edit_margin must be finite and nonnegative"
        )
    correction_objective_enabled = (
        safety_correction_pairwise_rank_weight > 0.0
        or safety_correction_top1_rank_weight > 0.0
        or safety_correction_minimal_edit_weight > 0.0
    )
    if correction_objective_enabled:
        if not episode_balanced:
            raise ValueError(
                "safety correction training requires episode-balanced "
                "optimization"
            )
        if (
            reference_model is None
            and (
                safety_correction_pairwise_rank_weight > 0.0
                or safety_correction_minimal_edit_weight > 0.0
            )
        ):
            raise ValueError(
                "pairwise or minimal-edit safety correction training requires "
                "a frozen reference model"
            )
    if (
        not math.isfinite(transition_action_rank_weight)
        or transition_action_rank_weight < 0.0
    ):
        raise ValueError(
            "transition_action_rank_weight must be finite and nonnegative"
        )
    if (
        not math.isfinite(movement_onset_rank_weight)
        or movement_onset_rank_weight < 0.0
    ):
        raise ValueError(
            "movement_onset_rank_weight must be finite and nonnegative"
        )
    if (
        not math.isfinite(movement_speed_change_rank_weight)
        or movement_speed_change_rank_weight < 0.0
    ):
        raise ValueError(
            "movement_speed_change_rank_weight must be finite and nonnegative"
        )
    if (
        not math.isfinite(motion_boundary_rank_weight)
        or motion_boundary_rank_weight < 0.0
    ):
        raise ValueError(
            "motion_boundary_rank_weight must be finite and nonnegative"
        )
    if (
        not math.isfinite(transition_action_rank_margin)
        or transition_action_rank_margin < 0.0
    ):
        raise ValueError(
            "transition_action_rank_margin must be finite and nonnegative"
        )
    if (
        not math.isfinite(motion_boundary_rank_margin)
        or motion_boundary_rank_margin < 0.0
    ):
        raise ValueError(
            "motion_boundary_rank_margin must be finite and nonnegative"
        )
    if (
        isinstance(motion_boundary_rank_lookback, bool)
        or not isinstance(motion_boundary_rank_lookback, int)
        or not 1 <= motion_boundary_rank_lookback <= 3
    ):
        raise ValueError("motion_boundary_rank_lookback must be in [1, 3]")
    if (
        motion_boundary_rank_weight > 0.0
        and optimizer is not None
        and not episode_balanced
    ):
        raise ValueError(
            "motion boundary ranking requires episode-balanced optimization"
        )
    if (
        action_count != 18
        and (
            direction_loss_weight > 0.0
            or speed_loss_weight > 0.0
            or direction_consistency_weight > 0.0
            or action_consistency_weight > 0.0
            or soft_action_loss_weight > 0.0
            or soft_action_collision_rank_weight > 0.0
            or movement_onset_rank_weight > 0.0
            or movement_speed_change_rank_weight > 0.0
            or motion_boundary_rank_weight > 0.0
        )
    ):
        raise ValueError("factorized action losses require the 18-action vocabulary")
    if soft_action_loss_weight > 0.0 and (
        demonstrations.teacher_action_evaluations is None
        or demonstrations.teacher_action_evaluation_mask is None
        or not demonstrations.teacher_action_evaluation_mask.any()
    ):
        raise ValueError(
            "soft action loss requires recorded teacher action evaluations"
        )
    _validate_demonstration_features(
        demonstrations,
        memory_size=memory_size,
        proficiency_size=proficiency_size,
    )
    correction_mask = _latest_correction_mask(
        demonstrations,
        required=correction_objective_enabled,
    )
    training = optimizer is not None
    model.train(training)
    if reference_model is not None:
        reference_model.eval()
    if future_visual_predictor is not None:
        future_visual_predictor.train(training)
    gradient_parameters = list(model.parameters())
    if future_visual_predictor is not None:
        gradient_parameters.extend(future_visual_predictor.parameters())
    transition_weights = teacher_transition_sample_weights(
        demonstrations,
        episodes=episodes,
        movement_onset_weight=movement_onset_weight,
        movement_stop_weight=movement_stop_weight,
        movement_speed_change_weight=movement_speed_change_weight,
        direction_change_weight=direction_change_weight,
    )
    (
        movement_onsets,
        movement_stops,
        direction_changes,
        movement_speed_changes,
    ) = (
        _teacher_motion_transition_masks(demonstrations, episodes)
    )
    motion_transitions = (
        movement_onsets
        | movement_stops
        | direction_changes
        | movement_speed_changes
    )
    motion_boundary_constraints = _motion_boundary_rank_constraints(
        demonstrations,
        episodes,
        lookback=motion_boundary_rank_lookback,
    )
    hard_action_terms_enabled = any((
        exact_action_loss_weight,
        direction_loss_weight,
        speed_loss_weight,
        direction_consistency_weight,
        action_consistency_weight,
    ))
    total_action_loss = 0.0
    total_action_weight = 0.0
    balanced_objective = 0.0
    balanced_episodes = 0
    total_risk_error = 0.0
    total_risk_loss = 0.0
    total_future_visual_loss = 0.0
    total_initial_policy_kl = 0.0
    transition_action_rank_objective = 0.0
    transition_action_rank_episodes = 0
    total_transition_action_rank_labels = 0
    total_transition_action_rank_margin_satisfied = 0
    movement_onset_rank_objective = 0.0
    movement_onset_rank_episodes = 0
    total_movement_onset_rank_labels = 0
    total_movement_onset_rank_margin_satisfied = 0
    movement_speed_change_rank_objective = 0.0
    movement_speed_change_rank_episodes = 0
    total_movement_speed_change_rank_labels = 0
    total_movement_speed_change_rank_margin_satisfied = 0
    motion_boundary_rank_objective = 0.0
    motion_boundary_rank_episodes = 0
    total_motion_boundary_rank_events = 0
    total_motion_boundary_rank_pairs = 0
    total_motion_boundary_rank_margin_satisfied = 0
    safety_correction_pairwise_rank_objective = 0.0
    safety_correction_pairwise_rank_episodes = 0
    total_safety_correction_pairwise_rank_labels = 0
    total_safety_correction_pairwise_rank_margin_satisfied = 0
    safety_correction_top1_rank_objective = 0.0
    safety_correction_top1_rank_episodes = 0
    total_safety_correction_top1_rank_labels = 0
    total_safety_correction_top1_rank_margin_satisfied = 0
    safety_correction_minimal_edit_objective = 0.0
    safety_correction_minimal_edit_episodes = 0
    total_safety_correction_minimal_edit_labels = 0
    total_safety_correction_minimal_edit_margin_satisfied = 0
    total_initial_policy_kl_labels = 0
    correct = 0
    labels = 0
    risk_labels = 0
    future_visual_labels = 0
    decisions = 0
    chunks = 0
    optimizer_steps = 0

    context = torch.enable_grad if training else torch.no_grad
    with context():
        for episode in episodes:
            hidden: Any | None = None
            reference_hidden: Any | None = None
            previous_direction_probabilities: Tensor | None = None
            previous_teacher_direction: Tensor | None = None
            previous_action_probabilities: Tensor | None = None
            previous_teacher_action: Tensor | None = None
            episode_labels = _episode_supervised_labels(demonstrations, episode)
            if demonstrations.supervision_mask is None:
                episode_supervised = np.ones(episode.decisions, dtype=np.bool_)
            else:
                episode_supervised = np.asarray(
                    demonstrations.supervision_mask[
                        episode.start:episode.stop, -1
                    ],
                    dtype=np.bool_,
                )
            episode_soft_supervised = np.zeros_like(episode_supervised)
            if soft_action_loss_weight > 0.0:
                assert demonstrations.teacher_action_evaluation_mask is not None
                episode_soft_supervised = np.asarray(
                    demonstrations.teacher_action_evaluation_mask[
                        episode.start:episode.stop, -1
                    ],
                    dtype=np.bool_,
                )
            episode_corrections = correction_mask[episode.start:episode.stop]
            episode_soft_supervised &= ~episode_corrections
            episode_hard_supervised = (
                episode_supervised
                & ~episode_soft_supervised
                & ~episode_corrections
            )
            episode_action_supervised = (
                episode_hard_supervised
                if hard_action_terms_enabled else
                np.zeros_like(episode_supervised)
            ) | episode_soft_supervised
            episode_transition_supervised = (
                motion_transitions[episode.start:episode.stop]
                & episode_hard_supervised
                if transition_action_rank_weight > 0.0 else
                np.zeros_like(episode_supervised)
            )
            episode_action_labels = int(episode_action_supervised.sum())
            episode_transition_action_rank_labels = int(
                episode_transition_supervised.sum()
            )
            episode_movement_onset_rank_supervised = (
                movement_onsets[episode.start:episode.stop]
                & episode_hard_supervised
                if movement_onset_rank_weight > 0.0 else
                np.zeros_like(episode_supervised)
            )
            episode_movement_onset_rank_labels = int(
                episode_movement_onset_rank_supervised.sum()
            )
            episode_movement_speed_change_rank_supervised = (
                movement_speed_changes[episode.start:episode.stop]
                & episode_hard_supervised
                if movement_speed_change_rank_weight > 0.0 else
                np.zeros_like(episode_supervised)
            )
            episode_movement_speed_change_rank_labels = int(
                episode_movement_speed_change_rank_supervised.sum()
            )
            episode_motion_boundary_event_mask = (
                motion_boundary_constraints.event_episode_ids
                == episode.episode_id
            )
            episode_motion_boundary_rank_events = (
                int(episode_motion_boundary_event_mask.sum())
                if motion_boundary_rank_weight > 0.0 else
                0
            )
            episode_safety_correction_pairwise_rank_labels = (
                int(correction_mask[episode.start:episode.stop].sum())
                if safety_correction_pairwise_rank_weight > 0.0 else
                0
            )
            episode_safety_correction_top1_rank_labels = (
                int(correction_mask[episode.start:episode.stop].sum())
                if safety_correction_top1_rank_weight > 0.0 else
                0
            )
            episode_safety_correction_minimal_edit_labels = (
                int(correction_mask[episode.start:episode.stop].sum())
                if safety_correction_minimal_edit_weight > 0.0 else
                0
            )
            episode_initial_policy_kl_labels = (
                int((~correction_mask[episode.start:episode.stop]).sum())
                if initial_policy_kl_weight > 0.0 else
                0
            )
            episode_action_weight = float(
                transition_weights[episode.start:episode.stop][
                    episode_action_supervised
                ].sum()
            )
            episode_risk_labels = (
                episode.decisions if risk_on_all_decisions else episode_labels
            )
            episode_future_visual_labels = _episode_future_visual_labels(
                episode,
                horizons,
            )
            episode_action_loss = 0.0
            episode_risk_loss = 0.0
            episode_future_visual_loss = 0.0
            episode_initial_policy_kl = 0.0
            episode_transition_action_rank_loss = 0.0
            episode_movement_onset_rank_loss = 0.0
            episode_movement_speed_change_rank_loss = 0.0
            episode_motion_boundary_rank_loss = 0.0
            episode_safety_correction_pairwise_rank_loss = 0.0
            episode_safety_correction_top1_rank_loss = 0.0
            episode_safety_correction_minimal_edit_loss = 0.0
            reflect_episode = (
                training
                and horizontal_reflection_probability > 0.0
                and (augmentation_rng or random).random()
                < horizontal_reflection_probability
            )
            episode_has_objective = bool(
                episode_action_labels
                or episode_transition_action_rank_labels
                or episode_movement_onset_rank_labels
                or episode_movement_speed_change_rank_labels
                or episode_motion_boundary_rank_events
                or episode_safety_correction_pairwise_rank_labels
                or episode_safety_correction_top1_rank_labels
                or episode_safety_correction_minimal_edit_labels
                or (risk_loss_weight > 0.0 and episode_risk_labels)
                or (
                    future_visual_loss_weight > 0.0
                    and episode_future_visual_labels
                )
                or episode_initial_policy_kl_labels
            )
            if training and episode_balanced and episode_has_objective:
                optimizer.zero_grad(set_to_none=True)
            for start in range(episode.start, episode.stop, chunk_length):
                stop = min(start + chunk_length, episode.stop)
                batch = _chunk_tensors(
                    demonstrations,
                    start,
                    stop,
                    memory_size=memory_size,
                    proficiency_size=proficiency_size,
                    device=device,
                )
                (
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
                ) = batch
                reference_global_frames = global_frames
                reference_local_frames = local_frames
                reference_memory = memory
                if reflect_episode:
                    global_frames, local_frames, actions = reflect_horizontal_stream_batch(
                        global_frames,
                        local_frames,
                        actions,
                    )
                    teacher_evaluations, teacher_regrets = (
                        reflect_horizontal_teacher_action_evidence(
                            teacher_evaluations,
                            teacher_regrets,
                        )
                    )
                    previous_action_size = int(
                        getattr(model, "previous_action_size", 0)
                    )
                    if previous_action_size:
                        if previous_action_size != 18:
                            raise ValueError(
                                "previous action context must have 18 entries"
                            )
                        memory = reflect_horizontal_action_context(
                            memory,
                            torch.as_tensor(
                                demonstrations.previous_actions[start:stop, -1],
                                dtype=torch.long,
                                device=device,
                            ).unsqueeze(0),
                            previous_action_offset=int(
                                getattr(model, "previous_action_offset", -1)
                            ),
                        )
                if training and previous_action_dropout_probability > 0.0:
                    previous_action_size = int(
                        getattr(model, "previous_action_size", 0)
                    )
                    if previous_action_size != 18:
                        raise ValueError(
                            "previous action dropout requires 18-entry action context"
                        )
                    rng = augmentation_rng or random
                    dropout_mask = torch.as_tensor(
                        [
                            rng.random() < previous_action_dropout_probability
                            for _ in range(stop - start)
                        ],
                        dtype=torch.bool,
                        device=device,
                    ).unsqueeze(0)
                    memory = drop_previous_action_context(
                        memory,
                        dropout_mask,
                        previous_action_offset=int(
                            getattr(model, "previous_action_offset", -1)
                        ),
                    )
                    if reference_model is not None:
                        reference_memory = drop_previous_action_context(
                            reference_memory,
                            dropout_mask,
                            previous_action_offset=int(
                                getattr(model, "previous_action_offset", -1)
                            ),
                        )
                if training and not episode_balanced:
                    optimizer.zero_grad(set_to_none=True)
                recurrent = None
                if future_visual_loss_weight > 0.0:
                    logits, predicted_risk, next_hidden, recurrent = (
                        model.forward_with_recurrent(
                            global_frames,
                            local_frames,
                            memory,
                            proficiency=proficiency,
                            hidden=hidden,
                        )
                    )
                else:
                    logits, predicted_risk, next_hidden = model(
                        global_frames,
                        local_frames,
                        memory,
                        proficiency=proficiency,
                        hidden=hidden,
                    )
                initial_policy_kl = logits.sum() * 0.0
                chunk_initial_policy_kl_labels = 0
                next_reference_hidden = None
                reference_logits = None
                if (
                    initial_policy_kl_weight > 0.0
                    or safety_correction_pairwise_rank_weight > 0.0
                    or safety_correction_minimal_edit_weight > 0.0
                ):
                    assert reference_model is not None
                    with torch.no_grad():
                        reference_logits, _reference_risk, next_reference_hidden = (
                            reference_model(
                                reference_global_frames,
                                reference_local_frames,
                                reference_memory,
                                proficiency=proficiency,
                                hidden=reference_hidden,
                            )
                        )
                        if reflect_episode:
                            horizontal_lookup = torch.as_tensor(
                                _HORIZONTAL_ACTIONS,
                                dtype=torch.long,
                                device=device,
                            )
                            reference_logits = reference_logits.index_select(
                                -1,
                                horizontal_lookup,
                            )
                chunk_correction_mask = torch.as_tensor(
                    correction_mask[start:stop],
                    dtype=torch.bool,
                    device=device,
                ).unsqueeze(0)
                if initial_policy_kl_weight > 0.0:
                    assert reference_logits is not None
                    initial_policy_kl_mask = ~chunk_correction_mask
                    chunk_initial_policy_kl_labels = int(
                        initial_policy_kl_mask.sum().item()
                    )
                    initial_policy_kl_terms = F.kl_div(
                        F.log_softmax(logits, dim=-1),
                        F.softmax(reference_logits, dim=-1),
                        reduction="none",
                    ).sum(dim=-1)
                    initial_policy_kl = initial_policy_kl_terms[
                        initial_policy_kl_mask
                    ].sum()
                expected_logits = (*actions.shape, action_count)
                if tuple(logits.shape) != expected_logits:
                    raise ValueError(
                        f"policy logits have shape {tuple(logits.shape)}; "
                        f"expected {expected_logits}"
                    )
                if tuple(predicted_risk.shape) != tuple(risks.shape):
                    raise ValueError("policy risk output does not align with decisions")
                soft_action_terms = torch.zeros_like(risks)
                soft_action_rank_terms = torch.zeros_like(risks)
                soft_action_mask = torch.zeros_like(mask)
                if soft_action_loss_weight > 0.0:
                    soft_action_terms, soft_action_mask = (
                        teacher_set_valued_action_loss(
                            logits,
                            teacher_evaluations,
                            teacher_regrets,
                            teacher_evaluation_mask,
                            actions,
                            temperature=soft_action_temperature,
                            safety_margin=soft_action_safety_margin,
                        )
                    )
                    if soft_action_collision_rank_weight > 0.0:
                        soft_action_rank_terms, _ = (
                            teacher_action_collision_ranking_loss(
                                logits,
                                teacher_evaluations,
                                teacher_regrets,
                                teacher_evaluation_mask,
                                actions,
                                temperature=soft_action_temperature,
                                safety_margin=soft_action_safety_margin,
                                ranking_margin=soft_action_collision_rank_margin,
                            )
                        )
                soft_action_mask &= ~chunk_correction_mask
                hard_action_mask = (
                    mask & ~soft_action_mask & ~chunk_correction_mask
                    if hard_action_terms_enabled else
                    torch.zeros_like(mask)
                )
                action_mask = hard_action_mask | soft_action_mask
                risk_mask = torch.ones_like(mask) if risk_on_all_decisions else mask
                chunk_risk_labels = int(risk_mask.sum().item())
                directions = actions % 9
                factorized = logits.reshape(*logits.shape[:-1], 2, 9)
                direction_logits = torch.logsumexp(factorized, dim=-2)
                direction_probabilities = torch.softmax(direction_logits, dim=-1)
                action_probabilities = torch.softmax(logits, dim=-1)
                transition_action_rank_loss = logits.sum() * 0.0
                chunk_transition_action_rank_labels = 0
                chunk_transition_action_rank_margin_satisfied = 0
                transition_action_rank_mask = torch.zeros_like(mask)
                movement_onset_rank_loss = logits.sum() * 0.0
                chunk_movement_onset_rank_labels = 0
                chunk_movement_onset_rank_margin_satisfied = 0
                movement_onset_rank_mask = torch.zeros_like(mask)
                movement_speed_change_rank_loss = logits.sum() * 0.0
                chunk_movement_speed_change_rank_labels = 0
                chunk_movement_speed_change_rank_margin_satisfied = 0
                movement_speed_change_rank_mask = torch.zeros_like(mask)
                motion_boundary_rank_loss = logits.sum() * 0.0
                chunk_motion_boundary_rank_pairs = 0
                chunk_motion_boundary_rank_margin_satisfied = 0
                motion_boundary_rank_mask = torch.zeros_like(mask)
                safety_correction_pairwise_rank_loss = logits.sum() * 0.0
                chunk_safety_correction_pairwise_rank_labels = 0
                chunk_safety_correction_pairwise_rank_margin_satisfied = 0
                safety_correction_pairwise_rank_mask = torch.zeros_like(mask)
                safety_correction_top1_rank_loss = logits.sum() * 0.0
                chunk_safety_correction_top1_rank_labels = 0
                chunk_safety_correction_top1_rank_margin_satisfied = 0
                safety_correction_top1_rank_mask = torch.zeros_like(mask)
                safety_correction_minimal_edit_loss = logits.sum() * 0.0
                chunk_safety_correction_minimal_edit_labels = 0
                chunk_safety_correction_minimal_edit_margin_satisfied = 0
                safety_correction_minimal_edit_mask = torch.zeros_like(mask)
                if transition_action_rank_weight > 0.0:
                    transition_action_rank_mask = torch.as_tensor(
                        motion_transitions[start:stop],
                        dtype=torch.bool,
                        device=device,
                    ).unsqueeze(0) & mask & ~soft_action_mask
                    transition_action_rank_mask &= ~chunk_correction_mask
                if movement_onset_rank_weight > 0.0:
                    movement_onset_rank_mask = torch.as_tensor(
                        movement_onsets[start:stop],
                        dtype=torch.bool,
                        device=device,
                    ).unsqueeze(0) & mask & ~soft_action_mask
                    movement_onset_rank_mask &= ~chunk_correction_mask
                if movement_speed_change_rank_weight > 0.0:
                    movement_speed_change_rank_mask = torch.as_tensor(
                        movement_speed_changes[start:stop],
                        dtype=torch.bool,
                        device=device,
                    ).unsqueeze(0) & mask & ~soft_action_mask
                    movement_speed_change_rank_mask &= ~chunk_correction_mask
                ranking_mask = (
                    transition_action_rank_mask
                    | movement_onset_rank_mask
                    | movement_speed_change_rank_mask
                )
                if ranking_mask.any():
                    ranking_terms = _hard_action_ranking_terms(
                        logits,
                        actions,
                        margin=transition_action_rank_margin,
                    )
                    if transition_action_rank_mask.any():
                        transition_terms = ranking_terms[
                            transition_action_rank_mask
                        ]
                        transition_action_rank_loss = transition_terms.sum()
                        chunk_transition_action_rank_labels = int(
                            transition_action_rank_mask.sum().item()
                        )
                        chunk_transition_action_rank_margin_satisfied = int(
                            (transition_terms <= 0.0).sum().item()
                        )
                    if movement_onset_rank_mask.any():
                        onset_terms = ranking_terms[movement_onset_rank_mask]
                        movement_onset_rank_loss = onset_terms.sum()
                        chunk_movement_onset_rank_labels = int(
                            movement_onset_rank_mask.sum().item()
                        )
                        chunk_movement_onset_rank_margin_satisfied = int(
                            (onset_terms <= 0.0).sum().item()
                        )
                    if movement_speed_change_rank_mask.any():
                        speed_change_terms = ranking_terms[
                            movement_speed_change_rank_mask
                        ]
                        movement_speed_change_rank_loss = speed_change_terms.sum()
                        chunk_movement_speed_change_rank_labels = int(
                            movement_speed_change_rank_mask.sum().item()
                        )
                        chunk_movement_speed_change_rank_margin_satisfied = int(
                            (speed_change_terms <= 0.0).sum().item()
                        )
                if motion_boundary_rank_weight > 0.0:
                    selected_pairs = np.flatnonzero(
                        (motion_boundary_constraints.state_indices >= start)
                        & (motion_boundary_constraints.state_indices < stop)
                    )
                    if len(selected_pairs):
                        state_offsets = torch.as_tensor(
                            motion_boundary_constraints.state_indices[selected_pairs]
                            - start,
                            dtype=torch.long,
                            device=device,
                        )
                        preferred_actions = torch.as_tensor(
                            motion_boundary_constraints.preferred_actions[
                                selected_pairs
                            ],
                            dtype=torch.long,
                            device=device,
                        )
                        rejected_actions = torch.as_tensor(
                            motion_boundary_constraints.rejected_actions[
                                selected_pairs
                            ],
                            dtype=torch.long,
                            device=device,
                        )
                        if reflect_episode:
                            horizontal_lookup = torch.as_tensor(
                                _HORIZONTAL_ACTIONS,
                                dtype=torch.long,
                                device=device,
                            )
                            preferred_actions = horizontal_lookup[preferred_actions]
                            rejected_actions = horizontal_lookup[rejected_actions]
                        boundary_terms = _motion_boundary_ranking_terms(
                            logits[0, state_offsets],
                            preferred_actions,
                            rejected_actions,
                            margin=motion_boundary_rank_margin,
                        )
                        pair_weights = torch.as_tensor(
                            motion_boundary_constraints.pair_weights[selected_pairs],
                            dtype=logits.dtype,
                            device=device,
                        )
                        motion_boundary_rank_loss = (
                            boundary_terms * pair_weights
                        ).sum()
                        chunk_motion_boundary_rank_pairs = len(selected_pairs)
                        chunk_motion_boundary_rank_margin_satisfied = int(
                            (boundary_terms <= 0.0).sum().item()
                        )
                        motion_boundary_rank_mask[0, state_offsets] = True
                if safety_correction_pairwise_rank_weight > 0.0:
                    assert reference_logits is not None
                    safety_correction_pairwise_rank_mask = chunk_correction_mask
                    if safety_correction_pairwise_rank_mask.any():
                        preferred_actions = actions[
                            safety_correction_pairwise_rank_mask
                        ]
                        rejected_actions = reference_logits.detach().argmax(dim=-1)[
                            safety_correction_pairwise_rank_mask
                        ]
                        correction_terms = (
                            _safety_correction_pairwise_ranking_terms(
                                logits[safety_correction_pairwise_rank_mask],
                                preferred_actions,
                                rejected_actions,
                                margin=safety_correction_pairwise_rank_margin,
                            )
                        )
                        safety_correction_pairwise_rank_loss = correction_terms.sum()
                        chunk_safety_correction_pairwise_rank_labels = int(
                            safety_correction_pairwise_rank_mask.sum().item()
                        )
                        chunk_safety_correction_pairwise_rank_margin_satisfied = int(
                            (correction_terms <= 0.0).sum().item()
                        )
                if safety_correction_top1_rank_weight > 0.0:
                    safety_correction_top1_rank_mask = chunk_correction_mask
                    if safety_correction_top1_rank_mask.any():
                        preferred_actions = actions[
                            safety_correction_top1_rank_mask
                        ]
                        top1_terms = _safety_correction_top1_ranking_terms(
                            logits[safety_correction_top1_rank_mask],
                            preferred_actions,
                            margin=safety_correction_top1_rank_margin,
                        )
                        safety_correction_top1_rank_loss = top1_terms.sum()
                        chunk_safety_correction_top1_rank_labels = int(
                            safety_correction_top1_rank_mask.sum().item()
                        )
                        chunk_safety_correction_top1_rank_margin_satisfied = int(
                            (top1_terms <= 0.0).sum().item()
                        )
                if safety_correction_minimal_edit_weight > 0.0:
                    assert reference_logits is not None
                    safety_correction_minimal_edit_mask = chunk_correction_mask
                    if safety_correction_minimal_edit_mask.any():
                        preferred_actions = actions[
                            safety_correction_minimal_edit_mask
                        ]
                        selected_reference_logits = reference_logits.detach()[
                            safety_correction_minimal_edit_mask
                        ]
                        selected_logits = logits[safety_correction_minimal_edit_mask]
                        minimal_edit_terms = _safety_correction_minimal_edit_terms(
                            selected_logits,
                            selected_reference_logits,
                            preferred_actions,
                            margin=safety_correction_minimal_edit_margin,
                        )
                        safety_correction_minimal_edit_loss = minimal_edit_terms.sum()
                        chunk_safety_correction_minimal_edit_labels = int(
                            safety_correction_minimal_edit_mask.sum().item()
                        )
                        parent_actions = selected_reference_logits.argmax(dim=-1)
                        margin_terms = _safety_correction_pairwise_ranking_terms(
                            selected_logits,
                            preferred_actions,
                            parent_actions,
                            margin=safety_correction_minimal_edit_margin,
                        )
                        chunk_safety_correction_minimal_edit_margin_satisfied = int(
                            (margin_terms <= 0.0).sum().item()
                        )
                weighted_action_loss = logits.sum() * 0.0
                chunk_action_weight = 0.0
                chunk_action_labels = int(action_mask.sum().item())
                metric_action_mask = (
                    action_mask
                    | transition_action_rank_mask
                    | movement_onset_rank_mask
                    | movement_speed_change_rank_mask
                    | motion_boundary_rank_mask
                    | safety_correction_pairwise_rank_mask
                    | safety_correction_top1_rank_mask
                    | safety_correction_minimal_edit_mask
                )
                chunk_labels = int(metric_action_mask.sum().item())
                if chunk_action_labels:
                    action_terms = torch.zeros_like(
                        risks,
                        dtype=logits.dtype,
                        device=logits.device,
                    )
                    if exact_action_loss_weight > 0.0 and hard_action_mask.any():
                        action_terms[hard_action_mask] += exact_action_loss_weight * (
                            F.cross_entropy(
                                logits[hard_action_mask],
                                actions[hard_action_mask],
                                weight=action_weights,
                                reduction="none",
                            )
                        )
                    if direction_loss_weight > 0.0 and hard_action_mask.any():
                        action_terms[hard_action_mask] += direction_loss_weight * (
                            F.cross_entropy(
                                direction_logits[hard_action_mask],
                                directions[hard_action_mask],
                                reduction="none",
                            )
                        )
                    if speed_loss_weight > 0.0 and hard_action_mask.any():
                        speed_logits = torch.logsumexp(factorized, dim=-1)
                        action_terms[hard_action_mask] += speed_loss_weight * (
                            F.cross_entropy(
                                speed_logits[hard_action_mask],
                                actions[hard_action_mask] // 9,
                                reduction="none",
                            )
                        )
                    if direction_consistency_weight > 0.0:
                        if previous_direction_probabilities is None:
                            prior_probabilities = direction_probabilities[:, :1].detach()
                            prior_directions = directions[:, :1]
                            has_prior = torch.zeros_like(mask[:, :1])
                        else:
                            prior_probabilities = previous_direction_probabilities[:, None]
                            assert previous_teacher_direction is not None
                            prior_directions = previous_teacher_direction[:, None]
                            has_prior = torch.ones_like(mask[:, :1])
                        prior_probabilities = torch.cat((
                            prior_probabilities,
                            direction_probabilities[:, :-1],
                        ), dim=1)
                        prior_directions = torch.cat((
                            prior_directions,
                            directions[:, :-1],
                        ), dim=1)
                        has_prior = torch.cat((
                            has_prior,
                            torch.ones_like(mask[:, 1:]),
                        ), dim=1)
                        stable_teacher = has_prior & (directions == prior_directions)
                        consistency = torch.square(
                            direction_probabilities - prior_probabilities,
                        ).sum(dim=-1)
                        action_terms[hard_action_mask] += (
                            direction_consistency_weight
                            * consistency[hard_action_mask]
                            * stable_teacher[hard_action_mask]
                        )
                    if action_consistency_weight > 0.0:
                        if previous_action_probabilities is None:
                            prior_action_probabilities = (
                                action_probabilities[:, :1].detach()
                            )
                            prior_actions = actions[:, :1]
                            has_action_prior = torch.zeros_like(mask[:, :1])
                        else:
                            prior_action_probabilities = (
                                previous_action_probabilities[:, None]
                            )
                            assert previous_teacher_action is not None
                            prior_actions = previous_teacher_action[:, None]
                            has_action_prior = torch.ones_like(mask[:, :1])
                        prior_action_probabilities = torch.cat((
                            prior_action_probabilities,
                            action_probabilities[:, :-1],
                        ), dim=1)
                        prior_actions = torch.cat((
                            prior_actions,
                            actions[:, :-1],
                        ), dim=1)
                        has_action_prior = torch.cat((
                            has_action_prior,
                            torch.ones_like(mask[:, 1:]),
                        ), dim=1)
                        stable_teacher_action = has_action_prior & (
                            actions == prior_actions
                        )
                        action_consistency = torch.square(
                            action_probabilities - prior_action_probabilities,
                        ).sum(dim=-1)
                        action_terms[hard_action_mask] += (
                            action_consistency_weight
                            * action_consistency[hard_action_mask]
                            * stable_teacher_action[hard_action_mask]
                        )
                    if soft_action_loss_weight > 0.0:
                        action_terms[soft_action_mask] += soft_action_loss_weight * (
                            soft_action_terms[soft_action_mask]
                        )
                    if soft_action_collision_rank_weight > 0.0:
                        action_terms[soft_action_mask] += (
                            soft_action_collision_rank_weight
                            * soft_action_rank_terms[soft_action_mask]
                        )
                    sample_weights = torch.as_tensor(
                        transition_weights[start:stop],
                        dtype=torch.float32,
                        device=device,
                    ).unsqueeze(0)[action_mask]
                    weighted_action_loss = (
                        action_terms[action_mask] * sample_weights
                    ).sum()
                    chunk_action_weight = float(sample_weights.detach().sum())
                risk_terms = F.smooth_l1_loss(
                    predicted_risk[risk_mask],
                    risks[risk_mask],
                    reduction="none",
                )
                risk_loss = risk_terms.sum()
                future_visual_loss = logits.sum() * 0.0
                chunk_future_visual_labels = 0
                if future_visual_loss_weight > 0.0:
                    assert recurrent is not None
                    future_visual_loss, chunk_future_visual_labels = (
                        _future_visual_loss(
                            model,
                            future_visual_predictor,
                            recurrent,
                            demonstrations,
                            episode,
                            start,
                            stop,
                            horizons=horizons,
                            device=device,
                            reflect_episode=reflect_episode,
                        )
                    )
                chunk_has_objective = bool(
                    chunk_labels
                    or (risk_loss_weight > 0.0 and chunk_risk_labels)
                    or (
                        future_visual_loss_weight > 0.0
                        and chunk_future_visual_labels
                    )
                    or chunk_safety_correction_pairwise_rank_labels
                    or chunk_safety_correction_top1_rank_labels
                    or chunk_safety_correction_minimal_edit_labels
                    or (
                        initial_policy_kl_weight > 0.0
                        and chunk_initial_policy_kl_labels
                    )
                )
                if chunk_has_objective:
                    loss = logits.sum() * 0.0
                    if chunk_action_labels:
                        loss = loss + (
                            weighted_action_loss / (
                                episode_action_weight
                                if episode_balanced else chunk_action_weight
                            )
                        )
                    if chunk_transition_action_rank_labels:
                        loss = loss + (
                            transition_action_rank_weight
                            * transition_action_rank_loss
                            / episode_transition_action_rank_labels
                        )
                    if chunk_movement_onset_rank_labels:
                        loss = loss + (
                            movement_onset_rank_weight
                            * movement_onset_rank_loss
                            / episode_movement_onset_rank_labels
                        )
                    if chunk_movement_speed_change_rank_labels:
                        loss = loss + (
                            movement_speed_change_rank_weight
                            * movement_speed_change_rank_loss
                            / episode_movement_speed_change_rank_labels
                        )
                    if chunk_motion_boundary_rank_pairs:
                        loss = loss + (
                            motion_boundary_rank_weight
                            * motion_boundary_rank_loss
                            / episode_motion_boundary_rank_events
                        )
                    if chunk_safety_correction_pairwise_rank_labels:
                        loss = loss + (
                            safety_correction_pairwise_rank_weight
                            * safety_correction_pairwise_rank_loss
                            / episode_safety_correction_pairwise_rank_labels
                        )
                    if chunk_safety_correction_top1_rank_labels:
                        loss = loss + (
                            safety_correction_top1_rank_weight
                            * safety_correction_top1_rank_loss
                            / episode_safety_correction_top1_rank_labels
                        )
                    if chunk_safety_correction_minimal_edit_labels:
                        loss = loss + (
                            safety_correction_minimal_edit_weight
                            * safety_correction_minimal_edit_loss
                            / episode_safety_correction_minimal_edit_labels
                        )
                    if risk_loss_weight > 0.0 and chunk_risk_labels:
                        loss = loss + risk_loss_weight * risk_loss / (
                            episode_risk_labels
                            if episode_balanced else chunk_risk_labels
                        )
                    if (
                        future_visual_loss_weight > 0.0
                        and chunk_future_visual_labels
                    ):
                        loss = loss + (
                            future_visual_loss_weight * future_visual_loss / (
                                episode_future_visual_labels
                                if episode_balanced else
                                chunk_future_visual_labels
                            )
                        )
                    if (
                        initial_policy_kl_weight > 0.0
                        and chunk_initial_policy_kl_labels
                    ):
                        loss = loss + (
                            initial_policy_kl_weight * initial_policy_kl / (
                                episode_initial_policy_kl_labels
                                if episode_balanced else
                                chunk_initial_policy_kl_labels
                            )
                        )
                    if training:
                        loss.backward()
                        if not episode_balanced:
                            torch.nn.utils.clip_grad_norm_(
                                gradient_parameters, gradient_clip,
                            )
                            optimizer.step()
                            optimizer_steps += 1
                detached_risk_loss = float(risk_loss.detach())
                episode_risk_loss += detached_risk_loss
                total_risk_loss += detached_risk_loss
                total_risk_error += float(
                    torch.abs(
                        predicted_risk.detach()[risk_mask] - risks[risk_mask]
                    ).sum()
                )
                risk_labels += chunk_risk_labels
                detached_future_visual_loss = float(future_visual_loss.detach())
                episode_future_visual_loss += detached_future_visual_loss
                total_future_visual_loss += detached_future_visual_loss
                future_visual_labels += chunk_future_visual_labels
                detached_initial_policy_kl = float(initial_policy_kl.detach())
                episode_initial_policy_kl += detached_initial_policy_kl
                total_initial_policy_kl += detached_initial_policy_kl
                total_initial_policy_kl_labels += chunk_initial_policy_kl_labels
                if chunk_transition_action_rank_labels:
                    detached_transition_action_rank_loss = float(
                        transition_action_rank_loss.detach()
                    )
                    episode_transition_action_rank_loss += (
                        detached_transition_action_rank_loss
                    )
                    total_transition_action_rank_labels += (
                        chunk_transition_action_rank_labels
                    )
                    total_transition_action_rank_margin_satisfied += (
                        chunk_transition_action_rank_margin_satisfied
                    )
                if chunk_movement_onset_rank_labels:
                    detached_movement_onset_rank_loss = float(
                        movement_onset_rank_loss.detach()
                    )
                    episode_movement_onset_rank_loss += (
                        detached_movement_onset_rank_loss
                    )
                    total_movement_onset_rank_labels += (
                        chunk_movement_onset_rank_labels
                    )
                    total_movement_onset_rank_margin_satisfied += (
                        chunk_movement_onset_rank_margin_satisfied
                    )
                if chunk_movement_speed_change_rank_labels:
                    detached_movement_speed_change_rank_loss = float(
                        movement_speed_change_rank_loss.detach()
                    )
                    episode_movement_speed_change_rank_loss += (
                        detached_movement_speed_change_rank_loss
                    )
                    total_movement_speed_change_rank_labels += (
                        chunk_movement_speed_change_rank_labels
                    )
                    total_movement_speed_change_rank_margin_satisfied += (
                        chunk_movement_speed_change_rank_margin_satisfied
                    )
                if chunk_motion_boundary_rank_pairs:
                    detached_motion_boundary_rank_loss = float(
                        motion_boundary_rank_loss.detach()
                    )
                    episode_motion_boundary_rank_loss += (
                        detached_motion_boundary_rank_loss
                    )
                    total_motion_boundary_rank_pairs += (
                        chunk_motion_boundary_rank_pairs
                    )
                    total_motion_boundary_rank_margin_satisfied += (
                        chunk_motion_boundary_rank_margin_satisfied
                    )
                if chunk_safety_correction_pairwise_rank_labels:
                    detached_safety_correction_pairwise_rank_loss = float(
                        safety_correction_pairwise_rank_loss.detach()
                    )
                    episode_safety_correction_pairwise_rank_loss += (
                        detached_safety_correction_pairwise_rank_loss
                    )
                    total_safety_correction_pairwise_rank_labels += (
                        chunk_safety_correction_pairwise_rank_labels
                    )
                    total_safety_correction_pairwise_rank_margin_satisfied += (
                        chunk_safety_correction_pairwise_rank_margin_satisfied
                    )
                if chunk_safety_correction_top1_rank_labels:
                    detached_safety_correction_top1_rank_loss = float(
                        safety_correction_top1_rank_loss.detach()
                    )
                    episode_safety_correction_top1_rank_loss += (
                        detached_safety_correction_top1_rank_loss
                    )
                    total_safety_correction_top1_rank_labels += (
                        chunk_safety_correction_top1_rank_labels
                    )
                    total_safety_correction_top1_rank_margin_satisfied += (
                        chunk_safety_correction_top1_rank_margin_satisfied
                    )
                if chunk_safety_correction_minimal_edit_labels:
                    detached_safety_correction_minimal_edit_loss = float(
                        safety_correction_minimal_edit_loss.detach()
                    )
                    episode_safety_correction_minimal_edit_loss += (
                        detached_safety_correction_minimal_edit_loss
                    )
                    total_safety_correction_minimal_edit_labels += (
                        chunk_safety_correction_minimal_edit_labels
                    )
                    total_safety_correction_minimal_edit_margin_satisfied += (
                        chunk_safety_correction_minimal_edit_margin_satisfied
                    )
                if chunk_action_labels:
                    detached_action_loss = float(weighted_action_loss.detach())
                    episode_action_loss += detached_action_loss
                    total_action_loss += detached_action_loss
                    total_action_weight += chunk_action_weight
                if chunk_labels:
                    correct += int(
                        (
                            logits.detach().argmax(dim=-1)[metric_action_mask]
                            == actions[metric_action_mask]
                        ).sum()
                    )
                labels += chunk_labels
                previous_direction_probabilities = (
                    direction_probabilities[:, -1].detach()
                )
                previous_teacher_direction = directions[:, -1].detach()
                previous_action_probabilities = action_probabilities[:, -1].detach()
                previous_teacher_action = actions[:, -1].detach()
                # This is the TBPTT boundary: history remains numerically
                # continuous across the attack, while its graph is truncated.
                hidden = _detach_hidden(next_hidden)
                reference_hidden = _detach_hidden(next_reference_hidden)
                decisions += stop - start
                chunks += 1
            if episode_has_objective:
                if episode_action_labels:
                    balanced_objective += episode_action_loss / episode_action_weight
                if episode_transition_action_rank_labels:
                    episode_transition_action_rank_mean = (
                        episode_transition_action_rank_loss
                        / episode_transition_action_rank_labels
                    )
                    transition_action_rank_objective += (
                        episode_transition_action_rank_mean
                    )
                    transition_action_rank_episodes += 1
                    balanced_objective += (
                        transition_action_rank_weight
                        * episode_transition_action_rank_mean
                    )
                if episode_movement_onset_rank_labels:
                    episode_movement_onset_rank_mean = (
                        episode_movement_onset_rank_loss
                        / episode_movement_onset_rank_labels
                    )
                    movement_onset_rank_objective += (
                        episode_movement_onset_rank_mean
                    )
                    movement_onset_rank_episodes += 1
                    balanced_objective += (
                        movement_onset_rank_weight
                        * episode_movement_onset_rank_mean
                    )
                if episode_movement_speed_change_rank_labels:
                    episode_movement_speed_change_rank_mean = (
                        episode_movement_speed_change_rank_loss
                        / episode_movement_speed_change_rank_labels
                    )
                    movement_speed_change_rank_objective += (
                        episode_movement_speed_change_rank_mean
                    )
                    movement_speed_change_rank_episodes += 1
                    balanced_objective += (
                        movement_speed_change_rank_weight
                        * episode_movement_speed_change_rank_mean
                    )
                if episode_motion_boundary_rank_events:
                    episode_motion_boundary_rank_mean = (
                        episode_motion_boundary_rank_loss
                        / episode_motion_boundary_rank_events
                    )
                    motion_boundary_rank_objective += (
                        episode_motion_boundary_rank_mean
                    )
                    motion_boundary_rank_episodes += 1
                    total_motion_boundary_rank_events += (
                        episode_motion_boundary_rank_events
                    )
                    balanced_objective += (
                        motion_boundary_rank_weight
                        * episode_motion_boundary_rank_mean
                    )
                if episode_safety_correction_pairwise_rank_labels:
                    episode_safety_correction_pairwise_rank_mean = (
                        episode_safety_correction_pairwise_rank_loss
                        / episode_safety_correction_pairwise_rank_labels
                    )
                    safety_correction_pairwise_rank_objective += (
                        episode_safety_correction_pairwise_rank_mean
                    )
                    safety_correction_pairwise_rank_episodes += 1
                    balanced_objective += (
                        safety_correction_pairwise_rank_weight
                        * episode_safety_correction_pairwise_rank_mean
                    )
                if episode_safety_correction_top1_rank_labels:
                    episode_safety_correction_top1_rank_mean = (
                        episode_safety_correction_top1_rank_loss
                        / episode_safety_correction_top1_rank_labels
                    )
                    safety_correction_top1_rank_objective += (
                        episode_safety_correction_top1_rank_mean
                    )
                    safety_correction_top1_rank_episodes += 1
                    balanced_objective += (
                        safety_correction_top1_rank_weight
                        * episode_safety_correction_top1_rank_mean
                    )
                if episode_safety_correction_minimal_edit_labels:
                    episode_safety_correction_minimal_edit_mean = (
                        episode_safety_correction_minimal_edit_loss
                        / episode_safety_correction_minimal_edit_labels
                    )
                    safety_correction_minimal_edit_objective += (
                        episode_safety_correction_minimal_edit_mean
                    )
                    safety_correction_minimal_edit_episodes += 1
                    balanced_objective += (
                        safety_correction_minimal_edit_weight
                        * episode_safety_correction_minimal_edit_mean
                    )
                if risk_loss_weight > 0.0 and episode_risk_labels:
                    balanced_objective += (
                        risk_loss_weight * episode_risk_loss / episode_risk_labels
                    )
                if (
                    future_visual_loss_weight > 0.0
                    and episode_future_visual_labels
                ):
                    balanced_objective += (
                        future_visual_loss_weight
                        * episode_future_visual_loss
                        / episode_future_visual_labels
                    )
                if (
                    initial_policy_kl_weight > 0.0
                    and episode_initial_policy_kl_labels
                ):
                    balanced_objective += (
                        initial_policy_kl_weight
                        * episode_initial_policy_kl
                        / episode_initial_policy_kl_labels
                    )
                balanced_episodes += 1
                if training and episode_balanced:
                    torch.nn.utils.clip_grad_norm_(
                        gradient_parameters,
                        gradient_clip,
                    )
                    optimizer.step()
                    optimizer_steps += 1

    if labels == 0 and not correction_objective_enabled:
        raise ValueError("selected episodes contain no supervised latest-frame labels")
    if risk_loss_weight > 0.0 and risk_labels == 0:
        raise ValueError("selected episodes contain no valid risk targets")
    if future_visual_loss_weight > 0.0 and future_visual_labels == 0:
        raise ValueError(
            "selected episodes contain no future visual targets at the requested horizons"
        )
    loss = (
        (
            balanced_objective / balanced_episodes
            if balanced_episodes else
            0.0
        ) if episode_balanced else
        (total_action_loss / total_action_weight if total_action_weight else 0.0)
        + (
            transition_action_rank_weight
            * transition_action_rank_objective
            / transition_action_rank_episodes
            if transition_action_rank_episodes else
            0.0
        )
        + (
            movement_onset_rank_weight
            * movement_onset_rank_objective
            / movement_onset_rank_episodes
            if movement_onset_rank_episodes else
            0.0
        )
        + (
            movement_speed_change_rank_weight
            * movement_speed_change_rank_objective
            / movement_speed_change_rank_episodes
            if movement_speed_change_rank_episodes else
            0.0
        )
        + (
            safety_correction_pairwise_rank_weight
            * safety_correction_pairwise_rank_objective
            / safety_correction_pairwise_rank_episodes
            if safety_correction_pairwise_rank_episodes else
            0.0
        )
        + (
            safety_correction_top1_rank_weight
            * safety_correction_top1_rank_objective
            / safety_correction_top1_rank_episodes
            if safety_correction_top1_rank_episodes else
            0.0
        )
        + (
            safety_correction_minimal_edit_weight
            * safety_correction_minimal_edit_objective
            / safety_correction_minimal_edit_episodes
            if safety_correction_minimal_edit_episodes else
            0.0
        )
        + (
            motion_boundary_rank_weight
            * motion_boundary_rank_objective
            / motion_boundary_rank_episodes
            if motion_boundary_rank_episodes else
            0.0
        )
        + (
            risk_loss_weight * total_risk_loss / risk_labels
            if risk_labels else
            0.0
        )
        + (
            future_visual_loss_weight
            * total_future_visual_loss
            / future_visual_labels
            if future_visual_labels else
            0.0
        )
        + (
            initial_policy_kl_weight
            * total_initial_policy_kl
            / total_initial_policy_kl_labels
            if total_initial_policy_kl_labels else
            0.0
        )
    )
    return StatefulPassMetrics(
        loss=loss,
        action_accuracy=correct / labels if labels else 0.0,
        risk_mae=total_risk_error / risk_labels if risk_labels else 0.0,
        labels=labels,
        risk_labels=risk_labels,
        decisions=decisions,
        chunks=chunks,
        episodes=len(episodes),
        optimizer_steps=optimizer_steps,
        movement_onsets=int(movement_onsets.sum()),
        movement_stops=int(movement_stops.sum()),
        direction_changes=int(direction_changes.sum()),
        movement_speed_changes=int(movement_speed_changes.sum()),
        future_visual_loss=(
            total_future_visual_loss / future_visual_labels
            if future_visual_labels else
            0.0
        ),
        future_visual_labels=future_visual_labels,
        transition_action_rank_loss=(
            transition_action_rank_objective / transition_action_rank_episodes
            if transition_action_rank_episodes else
            0.0
        ),
        transition_action_rank_labels=total_transition_action_rank_labels,
        transition_action_rank_margin_satisfaction=(
            total_transition_action_rank_margin_satisfied
            / total_transition_action_rank_labels
            if total_transition_action_rank_labels else
            0.0
        ),
        movement_onset_rank_loss=(
            movement_onset_rank_objective / movement_onset_rank_episodes
            if movement_onset_rank_episodes else
            0.0
        ),
        movement_onset_rank_labels=total_movement_onset_rank_labels,
        movement_onset_rank_margin_satisfaction=(
            total_movement_onset_rank_margin_satisfied
            / total_movement_onset_rank_labels
            if total_movement_onset_rank_labels else
            0.0
        ),
        movement_speed_change_rank_loss=(
            movement_speed_change_rank_objective
            / movement_speed_change_rank_episodes
            if movement_speed_change_rank_episodes else
            0.0
        ),
        movement_speed_change_rank_labels=(
            total_movement_speed_change_rank_labels
        ),
        movement_speed_change_rank_margin_satisfaction=(
            total_movement_speed_change_rank_margin_satisfied
            / total_movement_speed_change_rank_labels
            if total_movement_speed_change_rank_labels else
            0.0
        ),
        motion_boundary_rank_loss=(
            motion_boundary_rank_objective / motion_boundary_rank_episodes
            if motion_boundary_rank_episodes else
            0.0
        ),
        motion_boundary_rank_events=total_motion_boundary_rank_events,
        motion_boundary_rank_pairs=total_motion_boundary_rank_pairs,
        motion_boundary_rank_margin_satisfaction=(
            total_motion_boundary_rank_margin_satisfied
            / total_motion_boundary_rank_pairs
            if total_motion_boundary_rank_pairs else
            0.0
        ),
        safety_correction_pairwise_rank_loss=(
            safety_correction_pairwise_rank_objective
            / safety_correction_pairwise_rank_episodes
            if safety_correction_pairwise_rank_episodes else
            0.0
        ),
        safety_correction_pairwise_rank_labels=(
            total_safety_correction_pairwise_rank_labels
        ),
        safety_correction_pairwise_rank_margin_satisfaction=(
            total_safety_correction_pairwise_rank_margin_satisfied
            / total_safety_correction_pairwise_rank_labels
            if total_safety_correction_pairwise_rank_labels else
            0.0
        ),
        safety_correction_top1_rank_loss=(
            safety_correction_top1_rank_objective
            / safety_correction_top1_rank_episodes
            if safety_correction_top1_rank_episodes else
            0.0
        ),
        safety_correction_top1_rank_labels=(
            total_safety_correction_top1_rank_labels
        ),
        safety_correction_top1_rank_margin_satisfaction=(
            total_safety_correction_top1_rank_margin_satisfied
            / total_safety_correction_top1_rank_labels
            if total_safety_correction_top1_rank_labels else
            0.0
        ),
        safety_correction_minimal_edit_loss=(
            safety_correction_minimal_edit_objective
            / safety_correction_minimal_edit_episodes
            if safety_correction_minimal_edit_episodes else
            0.0
        ),
        safety_correction_minimal_edit_labels=(
            total_safety_correction_minimal_edit_labels
        ),
        safety_correction_minimal_edit_margin_satisfaction=(
            total_safety_correction_minimal_edit_margin_satisfied
            / total_safety_correction_minimal_edit_labels
            if total_safety_correction_minimal_edit_labels else
            0.0
        ),
        initial_policy_kl_loss=(
            total_initial_policy_kl / total_initial_policy_kl_labels
            if total_initial_policy_kl_labels else
            0.0
        ),
        initial_policy_kl_labels=total_initial_policy_kl_labels,
    )


def evaluate_stateful_policy(
    model: Any,
    demonstrations: Demonstrations,
    *,
    episode_ids: Iterable[int] | None = None,
    chunk_length: int = 32,
    risk_loss_weight: float = 0.2,
    device: str = "auto",
    movement_onset_weight: float = 1.0,
    movement_stop_weight: float = 1.0,
    movement_speed_change_weight: float = 1.0,
    direction_change_weight: float = 1.0,
    episode_balanced: bool = False,
    exact_action_loss_weight: float = 1.0,
    direction_loss_weight: float = 0.0,
    speed_loss_weight: float = 0.0,
    direction_consistency_weight: float = 0.0,
    action_consistency_weight: float = 0.0,
    transition_action_rank_weight: float = 0.0,
    transition_action_rank_margin: float = 1.0,
    movement_onset_rank_weight: float = 0.0,
    movement_speed_change_rank_weight: float = 0.0,
    motion_boundary_rank_weight: float = 0.0,
    motion_boundary_rank_margin: float = 1.0,
    motion_boundary_rank_lookback: int = 3,
    soft_action_loss_weight: float = 0.0,
    soft_action_temperature: float = 4.0,
    soft_action_safety_margin: float = 12.0,
) -> StatefulPassMetrics:
    """Evaluate in archive order while preserving hidden state inside episodes."""

    if chunk_length <= 0:
        raise ValueError("chunk_length must be positive")
    if not math.isfinite(risk_loss_weight) or risk_loss_weight < 0.0:
        raise ValueError("risk_loss_weight must be finite and nonnegative")
    resolved_device = _select_device(device)
    model.to(resolved_device)
    sequences = ordered_episode_sequences(demonstrations)
    selected = (
        sequences
        if episode_ids is None else
        _episode_selection(sequences, episode_ids)
    )
    return _stateful_pass(
        model,
        demonstrations,
        selected,
        chunk_length=chunk_length,
        risk_loss_weight=risk_loss_weight,
        gradient_clip=1.0,
        device=resolved_device,
        optimizer=None,
        movement_onset_weight=movement_onset_weight,
        movement_stop_weight=movement_stop_weight,
        movement_speed_change_weight=movement_speed_change_weight,
        direction_change_weight=direction_change_weight,
        episode_balanced=episode_balanced,
        exact_action_loss_weight=exact_action_loss_weight,
        direction_loss_weight=direction_loss_weight,
        speed_loss_weight=speed_loss_weight,
        direction_consistency_weight=direction_consistency_weight,
        action_consistency_weight=action_consistency_weight,
        transition_action_rank_weight=transition_action_rank_weight,
        transition_action_rank_margin=transition_action_rank_margin,
        movement_onset_rank_weight=movement_onset_rank_weight,
        movement_speed_change_rank_weight=movement_speed_change_rank_weight,
        motion_boundary_rank_weight=motion_boundary_rank_weight,
        motion_boundary_rank_margin=motion_boundary_rank_margin,
        motion_boundary_rank_lookback=motion_boundary_rank_lookback,
        soft_action_loss_weight=soft_action_loss_weight,
        soft_action_temperature=soft_action_temperature,
        soft_action_safety_margin=soft_action_safety_margin,
    )


def initialize_visual_encoders(
    model: HumanVisionPolicy,
    source: HumanVisionPolicy,
) -> None:
    """Transfer only visible-geometry encoders into a fresh recurrent policy."""

    target_config = getattr(model, "config", None)
    source_config = getattr(source, "config", None)
    for field in (
        "channels",
        "feature_size",
        "local_feature_grid_size",
        "local_downsample_stages",
    ):
        if getattr(target_config, field, None) != getattr(source_config, field, None):
            raise ValueError(
                f"visual encoder source does not match policy_config.{field}"
            )
    for name in ("global_encoder", "local_encoder"):
        target_encoder = getattr(model, name, None)
        source_encoder = getattr(source, name, None)
        if target_encoder is None or source_encoder is None:
            raise ValueError(f"visual policies must expose {name}")
        target_encoder.load_state_dict(source_encoder.state_dict(), strict=True)


def _class_weights(
    demonstrations: Demonstrations,
    episodes: Sequence[EpisodeSequence],
    *,
    action_count: int,
    power: float,
    device: str,
    exclude_teacher_evaluated: bool = False,
) -> Tensor:
    if torch is None:  # pragma: no cover - guarded by the training entry point
        raise RuntimeError("PyTorch is required for stateful policy training")
    labels: list[np.ndarray] = []
    correction_mask = _latest_correction_mask(demonstrations, required=False)
    for episode in episodes:
        if demonstrations.supervision_mask is not None:
            supervised = np.asarray(
                demonstrations.supervision_mask[episode.start:episode.stop, -1],
                dtype=np.bool_,
            )
        else:
            supervised = np.ones(episode.decisions, dtype=np.bool_)
        supervised &= ~correction_mask[episode.start:episode.stop]
        if exclude_teacher_evaluated:
            assert demonstrations.teacher_action_evaluation_mask is not None
            supervised &= ~np.asarray(
                demonstrations.teacher_action_evaluation_mask[
                    episode.start:episode.stop, -1
                ],
                dtype=np.bool_,
            )
        actions = demonstrations.actions[episode.start:episode.stop, -1][supervised]
        labels.append(actions)
    selected = np.concatenate(labels) if labels else np.empty(0, dtype=np.int64)
    if not len(selected):
        raise ValueError("training episodes contain no supervised latest-frame labels")
    counts = np.bincount(selected, minlength=action_count).astype(np.float64)
    present = counts > 0
    weights = np.zeros(action_count, dtype=np.float32)
    inverse_frequency = len(selected) / (present.sum() * counts[present])
    weights[present] = np.minimum(inverse_frequency ** power, 10.0)
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


def train_stateful_behavior_cloning(
    demonstrations: Demonstrations,
    *,
    policy_config: PolicyConfig = PolicyConfig(inference_mode="stream"),
    training_config: StatefulTrainingConfig = StatefulTrainingConfig(),
    model: HumanVisionPolicy | None = None,
    output: str | Path | None = None,
    training_data: Mapping[str, Any] | None = None,
) -> tuple[HumanVisionPolicy, list[TrainingMetrics]]:
    """Train one streaming GRU across complete, ordered attack episodes."""

    if torch is None:
        raise RuntimeError("PyTorch is required for stateful policy training")
    demonstrations.validate()
    if policy_config.inference_mode != "stream":
        raise ValueError("stateful TBPTT requires policy_config.inference_mode='stream'")
    _latest_correction_mask(
        demonstrations,
        required=(
            training_config.safety_correction_pairwise_rank_weight > 0.0
            or training_config.safety_correction_top1_rank_weight > 0.0
            or training_config.safety_correction_minimal_edit_weight > 0.0
        ),
    )
    if training_config.correction_only:
        if demonstrations.supervision_mask is None:
            raise ValueError(
                "correction-only training requires a supervision_mask"
            )
        latest_mask = np.asarray(
            demonstrations.supervision_mask[:, -1], dtype=np.bool_,
        )
        if not latest_mask.any():
            raise ValueError(
                "correction-only training requires at least one supervised correction"
            )
    split = split_episode_ids(
        demonstrations,
        validation_fraction=training_config.validation_fraction,
        seed=training_config.seed,
        validation_episode_ids=training_config.validation_episode_ids,
    )
    sequences = ordered_episode_sequences(demonstrations)
    train_episodes = _episode_selection(sequences, split.train_episode_ids)
    validation_episodes = _episode_selection(
        sequences, split.validation_episode_ids,
    )

    _seed_everything(training_config.seed)
    resolved_device = _select_device(training_config.device)
    if model is None:
        model = HumanVisionPolicy(policy_config)
    metadata_input = dict(training_data or {})
    previous_action_size = int(metadata_input.get("previous_action_size", 0))
    previous_action_offset = int(metadata_input.get("previous_action_offset", 0))
    if previous_action_size not in {0, 18}:
        raise ValueError("training previous_action_size must be 0 or 18")
    if previous_action_size and (
        previous_action_offset < 0
        or previous_action_offset + previous_action_size != policy_config.memory_size
    ):
        raise ValueError("training previous action context must end at memory_size")
    inherited_previous_action_size = int(
        getattr(model, "previous_action_size", previous_action_size)
    )
    inherited_previous_action_offset = int(
        getattr(model, "previous_action_offset", previous_action_offset)
    )
    if (
        inherited_previous_action_size != previous_action_size
        or inherited_previous_action_offset != previous_action_offset
    ):
        raise ValueError("model and training previous action context metadata differ")
    model.previous_action_size = previous_action_size
    model.previous_action_offset = previous_action_offset
    if (
        (
            training_config.horizontal_reflection_probability > 0.0
            or training_config.previous_action_dropout_probability > 0.0
        )
        and previous_action_size
    ):
        if demonstrations.memory is None or demonstrations.previous_actions is None:
            raise ValueError(
                "previous action augmentation requires "
                "recorded memory and previous_actions"
            )
        reflect_horizontal_action_context(
            torch.as_tensor(
                demonstrations.memory[:, -1], dtype=torch.float32,
            ).unsqueeze(0),
            torch.as_tensor(
                demonstrations.previous_actions[:, -1], dtype=torch.long,
            ).unsqueeze(0),
            previous_action_offset=previous_action_offset,
        )
    if (
        training_config.previous_action_dropout_probability > 0.0
        and previous_action_size != 18
    ):
        raise ValueError(
            "previous action dropout requires checkpoint-declared action context"
        )
    model_config = getattr(model, "config", None)
    for field in (
        "feature_size",
        "recurrent_size",
        "memory_size",
        "proficiency_size",
        "action_count",
        "inference_mode",
        "local_feature_grid_size",
        "local_downsample_stages",
    ):
        if getattr(model_config, field, None) != getattr(policy_config, field):
            raise ValueError(f"model config does not match policy_config.{field}")
    model.to(resolved_device)
    reference_model = None
    if (
        training_config.initial_policy_kl_weight > 0.0
        or training_config.safety_correction_pairwise_rank_weight > 0.0
        or training_config.safety_correction_minimal_edit_weight > 0.0
    ):
        reference_model = copy.deepcopy(model).to(resolved_device)
        reference_model.eval()
        for parameter in reference_model.parameters():
            parameter.requires_grad_(False)
    if training_config.policy_head_only:
        policy_head = getattr(model, "policy_head", None)
        if not isinstance(policy_head, torch.nn.Module):
            raise ValueError(
                "policy-head-only training requires a model policy_head module"
            )
        policy_head_parameter_ids = {
            id(parameter) for parameter in policy_head.parameters()
        }
        if not policy_head_parameter_ids:
            raise ValueError(
                "policy-head-only training requires trainable policy_head parameters"
            )
        for parameter in model.parameters():
            parameter.requires_grad_(id(parameter) in policy_head_parameter_ids)
    future_visual_predictor = None
    if training_config.future_visual_loss_weight > 0.0:
        if not callable(getattr(model, "forward_with_recurrent", None)):
            raise ValueError(
                "future visual prediction requires per-decision GRU hidden states"
            )
        if not callable(getattr(model, "encode_visual", None)):
            raise ValueError(
                "future visual prediction requires a policy encode_visual method"
            )
        future_visual_predictor = _FutureVisualPredictor(
            policy_config.recurrent_size,
            policy_config.feature_size * 2,
            training_config.future_visual_horizons,
        ).to(resolved_device)
    optimized_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if future_visual_predictor is not None:
        optimized_parameters.extend(future_visual_predictor.parameters())
    optimizer = torch.optim.AdamW(
        optimized_parameters,
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    action_weights = None
    if training_config.class_balance and training_config.exact_action_loss_weight > 0.0:
        action_weights = _class_weights(
            demonstrations,
            train_episodes,
            action_count=policy_config.action_count,
            power=training_config.class_balance_power,
            device=resolved_device,
            exclude_teacher_evaluated=(
                training_config.soft_action_loss_weight > 0.0
            ),
        )

    history: list[TrainingMetrics] = []
    augmentation_rng = random.Random(training_config.seed ^ 0x5EED5EED)
    episode_order_rng = random.Random(training_config.seed ^ 0xE9150DE)
    best_validation_loss = math.inf
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None
    for epoch in range(1, training_config.epochs + 1):
        epoch_train_episodes = train_episodes
        if training_config.episode_balanced:
            shuffled = list(train_episodes)
            episode_order_rng.shuffle(shuffled)
            epoch_train_episodes = tuple(shuffled)
        training = _stateful_pass(
            model,
            demonstrations,
            epoch_train_episodes,
            chunk_length=training_config.chunk_length,
            risk_loss_weight=training_config.risk_loss_weight,
            gradient_clip=training_config.gradient_clip,
            device=resolved_device,
            optimizer=optimizer,
            action_weights=action_weights,
            horizontal_reflection_probability=(
                training_config.horizontal_reflection_probability
            ),
            augmentation_rng=augmentation_rng,
            movement_onset_weight=training_config.movement_onset_weight,
            movement_stop_weight=training_config.movement_stop_weight,
            movement_speed_change_weight=(
                training_config.movement_speed_change_weight
            ),
            direction_change_weight=training_config.direction_change_weight,
            episode_balanced=training_config.episode_balanced,
            exact_action_loss_weight=training_config.exact_action_loss_weight,
            direction_loss_weight=training_config.direction_loss_weight,
            speed_loss_weight=training_config.speed_loss_weight,
            direction_consistency_weight=(
                training_config.direction_consistency_weight
            ),
            action_consistency_weight=training_config.action_consistency_weight,
            transition_action_rank_weight=(
                training_config.transition_action_rank_weight
            ),
            transition_action_rank_margin=(
                training_config.transition_action_rank_margin
            ),
            movement_onset_rank_weight=(
                training_config.movement_onset_rank_weight
            ),
            movement_speed_change_rank_weight=(
                training_config.movement_speed_change_rank_weight
            ),
            motion_boundary_rank_weight=(
                training_config.motion_boundary_rank_weight
            ),
            motion_boundary_rank_margin=(
                training_config.motion_boundary_rank_margin
            ),
            motion_boundary_rank_lookback=(
                training_config.motion_boundary_rank_lookback
            ),
            safety_correction_pairwise_rank_weight=(
                training_config.safety_correction_pairwise_rank_weight
            ),
            safety_correction_pairwise_rank_margin=(
                training_config.safety_correction_pairwise_rank_margin
            ),
            safety_correction_top1_rank_weight=(
                training_config.safety_correction_top1_rank_weight
            ),
            safety_correction_top1_rank_margin=(
                training_config.safety_correction_top1_rank_margin
            ),
            safety_correction_minimal_edit_weight=(
                training_config.safety_correction_minimal_edit_weight
            ),
            safety_correction_minimal_edit_margin=(
                training_config.safety_correction_minimal_edit_margin
            ),
            soft_action_loss_weight=training_config.soft_action_loss_weight,
            soft_action_collision_rank_weight=(
                training_config.soft_action_collision_rank_weight
            ),
            soft_action_collision_rank_margin=(
                training_config.soft_action_collision_rank_margin
            ),
            soft_action_temperature=training_config.soft_action_temperature,
            soft_action_safety_margin=training_config.soft_action_safety_margin,
            initial_policy_kl_weight=training_config.initial_policy_kl_weight,
            reference_model=reference_model,
            risk_on_all_decisions=training_config.correction_only,
            previous_action_dropout_probability=(
                training_config.previous_action_dropout_probability
            ),
            future_visual_loss_weight=(
                training_config.future_visual_loss_weight
            ),
            future_visual_horizons=training_config.future_visual_horizons,
            future_visual_predictor=future_visual_predictor,
        )
        validation = _stateful_pass(
            model,
            demonstrations,
            validation_episodes,
            chunk_length=training_config.chunk_length,
            risk_loss_weight=training_config.risk_loss_weight,
            gradient_clip=training_config.gradient_clip,
            device=resolved_device,
            optimizer=None,
            movement_onset_weight=training_config.movement_onset_weight,
            movement_stop_weight=training_config.movement_stop_weight,
            movement_speed_change_weight=(
                training_config.movement_speed_change_weight
            ),
            direction_change_weight=training_config.direction_change_weight,
            episode_balanced=training_config.episode_balanced,
            exact_action_loss_weight=training_config.exact_action_loss_weight,
            direction_loss_weight=training_config.direction_loss_weight,
            speed_loss_weight=training_config.speed_loss_weight,
            direction_consistency_weight=(
                training_config.direction_consistency_weight
            ),
            action_consistency_weight=training_config.action_consistency_weight,
            transition_action_rank_weight=(
                training_config.transition_action_rank_weight
            ),
            transition_action_rank_margin=(
                training_config.transition_action_rank_margin
            ),
            movement_onset_rank_weight=(
                training_config.movement_onset_rank_weight
            ),
            movement_speed_change_rank_weight=(
                training_config.movement_speed_change_rank_weight
            ),
            motion_boundary_rank_weight=(
                training_config.motion_boundary_rank_weight
            ),
            motion_boundary_rank_margin=(
                training_config.motion_boundary_rank_margin
            ),
            motion_boundary_rank_lookback=(
                training_config.motion_boundary_rank_lookback
            ),
            safety_correction_pairwise_rank_weight=(
                training_config.safety_correction_pairwise_rank_weight
            ),
            safety_correction_pairwise_rank_margin=(
                training_config.safety_correction_pairwise_rank_margin
            ),
            safety_correction_top1_rank_weight=(
                training_config.safety_correction_top1_rank_weight
            ),
            safety_correction_top1_rank_margin=(
                training_config.safety_correction_top1_rank_margin
            ),
            safety_correction_minimal_edit_weight=(
                training_config.safety_correction_minimal_edit_weight
            ),
            safety_correction_minimal_edit_margin=(
                training_config.safety_correction_minimal_edit_margin
            ),
            soft_action_loss_weight=training_config.soft_action_loss_weight,
            soft_action_collision_rank_weight=(
                training_config.soft_action_collision_rank_weight
            ),
            soft_action_collision_rank_margin=(
                training_config.soft_action_collision_rank_margin
            ),
            soft_action_temperature=training_config.soft_action_temperature,
            soft_action_safety_margin=training_config.soft_action_safety_margin,
            initial_policy_kl_weight=training_config.initial_policy_kl_weight,
            reference_model=reference_model,
            risk_on_all_decisions=training_config.correction_only,
            future_visual_loss_weight=(
                training_config.future_visual_loss_weight
            ),
            future_visual_horizons=training_config.future_visual_horizons,
            future_visual_predictor=future_visual_predictor,
        )
        metrics = TrainingMetrics(
            epoch=epoch,
            train_loss=training.loss,
            validation_loss=validation.loss,
            action_accuracy=validation.action_accuracy,
            risk_mae=validation.risk_mae,
            train_future_visual_loss=training.future_visual_loss,
            validation_future_visual_loss=validation.future_visual_loss,
            train_future_visual_labels=training.future_visual_labels,
            validation_future_visual_labels=validation.future_visual_labels,
            train_transition_action_rank_loss=(
                training.transition_action_rank_loss
            ),
            validation_transition_action_rank_loss=(
                validation.transition_action_rank_loss
            ),
            train_transition_action_rank_labels=(
                training.transition_action_rank_labels
            ),
            validation_transition_action_rank_labels=(
                validation.transition_action_rank_labels
            ),
            train_transition_action_rank_margin_satisfaction=(
                training.transition_action_rank_margin_satisfaction
            ),
            validation_transition_action_rank_margin_satisfaction=(
                validation.transition_action_rank_margin_satisfaction
            ),
            train_movement_onset_rank_loss=(
                training.movement_onset_rank_loss
            ),
            validation_movement_onset_rank_loss=(
                validation.movement_onset_rank_loss
            ),
            train_movement_onset_rank_labels=(
                training.movement_onset_rank_labels
            ),
            validation_movement_onset_rank_labels=(
                validation.movement_onset_rank_labels
            ),
            train_movement_onset_rank_margin_satisfaction=(
                training.movement_onset_rank_margin_satisfaction
            ),
            validation_movement_onset_rank_margin_satisfaction=(
                validation.movement_onset_rank_margin_satisfaction
            ),
            train_movement_speed_change_rank_loss=(
                training.movement_speed_change_rank_loss
            ),
            validation_movement_speed_change_rank_loss=(
                validation.movement_speed_change_rank_loss
            ),
            train_movement_speed_change_rank_labels=(
                training.movement_speed_change_rank_labels
            ),
            validation_movement_speed_change_rank_labels=(
                validation.movement_speed_change_rank_labels
            ),
            train_movement_speed_change_rank_margin_satisfaction=(
                training.movement_speed_change_rank_margin_satisfaction
            ),
            validation_movement_speed_change_rank_margin_satisfaction=(
                validation.movement_speed_change_rank_margin_satisfaction
            ),
            train_motion_boundary_rank_loss=(
                training.motion_boundary_rank_loss
            ),
            validation_motion_boundary_rank_loss=(
                validation.motion_boundary_rank_loss
            ),
            train_motion_boundary_rank_events=(
                training.motion_boundary_rank_events
            ),
            validation_motion_boundary_rank_events=(
                validation.motion_boundary_rank_events
            ),
            train_motion_boundary_rank_pairs=(
                training.motion_boundary_rank_pairs
            ),
            validation_motion_boundary_rank_pairs=(
                validation.motion_boundary_rank_pairs
            ),
            train_motion_boundary_rank_margin_satisfaction=(
                training.motion_boundary_rank_margin_satisfaction
            ),
            validation_motion_boundary_rank_margin_satisfaction=(
                validation.motion_boundary_rank_margin_satisfaction
            ),
            train_safety_correction_pairwise_rank_loss=(
                training.safety_correction_pairwise_rank_loss
            ),
            validation_safety_correction_pairwise_rank_loss=(
                validation.safety_correction_pairwise_rank_loss
            ),
            train_safety_correction_pairwise_rank_labels=(
                training.safety_correction_pairwise_rank_labels
            ),
            validation_safety_correction_pairwise_rank_labels=(
                validation.safety_correction_pairwise_rank_labels
            ),
            train_safety_correction_pairwise_rank_margin_satisfaction=(
                training.safety_correction_pairwise_rank_margin_satisfaction
            ),
            validation_safety_correction_pairwise_rank_margin_satisfaction=(
                validation.safety_correction_pairwise_rank_margin_satisfaction
            ),
            train_safety_correction_top1_rank_loss=(
                training.safety_correction_top1_rank_loss
            ),
            validation_safety_correction_top1_rank_loss=(
                validation.safety_correction_top1_rank_loss
            ),
            train_safety_correction_top1_rank_labels=(
                training.safety_correction_top1_rank_labels
            ),
            validation_safety_correction_top1_rank_labels=(
                validation.safety_correction_top1_rank_labels
            ),
            train_safety_correction_top1_rank_margin_satisfaction=(
                training.safety_correction_top1_rank_margin_satisfaction
            ),
            validation_safety_correction_top1_rank_margin_satisfaction=(
                validation.safety_correction_top1_rank_margin_satisfaction
            ),
            train_safety_correction_minimal_edit_loss=(
                training.safety_correction_minimal_edit_loss
            ),
            validation_safety_correction_minimal_edit_loss=(
                validation.safety_correction_minimal_edit_loss
            ),
            train_safety_correction_minimal_edit_labels=(
                training.safety_correction_minimal_edit_labels
            ),
            validation_safety_correction_minimal_edit_labels=(
                validation.safety_correction_minimal_edit_labels
            ),
            train_safety_correction_minimal_edit_margin_satisfaction=(
                training.safety_correction_minimal_edit_margin_satisfaction
            ),
            validation_safety_correction_minimal_edit_margin_satisfaction=(
                validation.safety_correction_minimal_edit_margin_satisfaction
            ),
            train_initial_policy_kl_loss=training.initial_policy_kl_loss,
            validation_initial_policy_kl_loss=(
                validation.initial_policy_kl_loss
            ),
            train_initial_policy_kl_labels=training.initial_policy_kl_labels,
            validation_initial_policy_kl_labels=(
                validation.initial_policy_kl_labels
            ),
        )
        history.append(metrics)
        if validation.loss < best_validation_loss:
            best_validation_loss = validation.loss
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

    if training_config.restore_best_validation:
        if best_state is None:
            raise RuntimeError("stateful training produced no validation checkpoint")
        model.load_state_dict(best_state)

    if output is not None:
        (
            train_onsets,
            train_stops,
            train_direction_changes,
            train_speed_changes,
        ) = (
            _teacher_motion_transition_masks(demonstrations, train_episodes)
        )
        (
            validation_onsets,
            validation_stops,
            validation_direction_changes,
            validation_speed_changes,
        ) = (
            _teacher_motion_transition_masks(demonstrations, validation_episodes)
        )
        train_motion_boundaries = (
            _motion_boundary_rank_constraints(
                demonstrations,
                train_episodes,
                lookback=training_config.motion_boundary_rank_lookback,
            )
            if training_config.motion_boundary_rank_weight > 0.0 else
            None
        )
        validation_motion_boundaries = (
            _motion_boundary_rank_constraints(
                demonstrations,
                validation_episodes,
                lookback=training_config.motion_boundary_rank_lookback,
            )
            if training_config.motion_boundary_rank_weight > 0.0 else
            None
        )
        train_future_visual_labels = sum(
            _episode_future_visual_labels(
                episode,
                training_config.future_visual_horizons,
            )
            for episode in train_episodes
        )
        validation_future_visual_labels = sum(
            _episode_future_visual_labels(
                episode,
                training_config.future_visual_horizons,
            )
            for episode in validation_episodes
        )
        latest_corrections = _latest_correction_mask(
            demonstrations,
            required=(
                training_config.safety_correction_pairwise_rank_weight > 0.0
                or training_config.safety_correction_top1_rank_weight > 0.0
                or training_config.safety_correction_minimal_edit_weight > 0.0
            ),
        )
        train_safety_correction_labels = sum(
            int(latest_corrections[episode.start:episode.stop].sum())
            for episode in train_episodes
        )
        validation_safety_correction_labels = sum(
            int(latest_corrections[episode.start:episode.stop].sum())
            for episode in validation_episodes
        )
        metadata = dict(training_data or {})
        metadata.update({
            "training_mode": "episode_stateful_tbptt",
            "inference_semantics": "latest_visible_frame_stream",
            "policy_head_only": training_config.policy_head_only,
            "tbptt_chunk_length": training_config.chunk_length,
            "train_episode_ids": list(split.train_episode_ids),
            "validation_episode_ids": list(split.validation_episode_ids),
            "loss_weighting": {
                "class_balance": training_config.class_balance,
                "class_balance_power": training_config.class_balance_power,
                "movement_onset_weight": training_config.movement_onset_weight,
                "movement_stop_weight": training_config.movement_stop_weight,
                "movement_speed_change_weight": (
                    training_config.movement_speed_change_weight
                ),
                "direction_change_weight": training_config.direction_change_weight,
                "exact_action_loss_weight": (
                    training_config.exact_action_loss_weight
                ),
                "direction_loss_weight": training_config.direction_loss_weight,
                "speed_loss_weight": training_config.speed_loss_weight,
                "direction_consistency_weight": (
                    training_config.direction_consistency_weight
                ),
                "action_consistency_weight": (
                    training_config.action_consistency_weight
                ),
                **(
                    {
                        "safety_correction_top1_rank_weight": (
                            training_config.safety_correction_top1_rank_weight
                        ),
                        "safety_correction_top1_rank_margin": (
                            training_config.safety_correction_top1_rank_margin
                        ),
                        "safety_correction_top1_rank_preferred": (
                            "demonstrations.actions_on_explicit_correction_mask"
                        ),
                        "safety_correction_top1_rank_rejected": (
                            "strongest_current_nonpreferred_policy_logit"
                        ),
                        "safety_correction_top1_rank_loss": (
                            "relu(margin+max_nonpreferred_logit-preferred_logit)"
                        ),
                        "safety_correction_top1_rank_reduction": (
                            "mean_corrections_per_episode_then_mean_episodes"
                        ),
                        "safety_correction_top1_rank_optimizer_step_unit": (
                            "complete_episode"
                        ),
                        "safety_correction_top1_rank_other_action_losses": (
                            "exclude_correction_mask_rows"
                        ),
                        "safety_correction_top1_rank_horizontal_reflection": (
                            "map_preferred_action_id"
                        ),
                        "train_safety_correction_top1_rank_labels": (
                            train_safety_correction_labels
                        ),
                        "validation_safety_correction_top1_rank_labels": (
                            validation_safety_correction_labels
                        ),
                    }
                    if training_config.safety_correction_top1_rank_weight > 0.0
                    else {}
                ),
                **(
                    {
                        "transition_action_rank_weight": (
                            training_config.transition_action_rank_weight
                        ),
                        "transition_action_rank_margin": (
                            training_config.transition_action_rank_margin
                        ),
                        "transition_action_rank_scope": (
                            "reliable_supervised_movement_action_transitions"
                        ),
                        "transition_action_rank_reduction": (
                            "mean_per_episode_over_transition_labels"
                        ),
                        "transition_action_rank_sample_weighting": (
                            "independent_of_transition_sample_weights"
                        ),
                        "transition_action_rank_soft_evaluation_policy": (
                            "exclude_teacher_evaluated_rows"
                        ),
                    }
                    if training_config.transition_action_rank_weight > 0.0 else {}
                ),
                **(
                    {
                        "movement_onset_rank_weight": (
                            training_config.movement_onset_rank_weight
                        ),
                        "movement_onset_rank_margin": (
                            training_config.transition_action_rank_margin
                        ),
                        "movement_onset_rank_margin_source": (
                            "shared_transition_action_rank_margin"
                        ),
                        "movement_onset_rank_scope": (
                            "reliable_hard_supervised_stationary_to_moving_"
                            "transitions"
                        ),
                        "movement_onset_rank_reduction": (
                            "mean_per_episode_over_onset_labels"
                        ),
                        "movement_onset_rank_sample_weighting": (
                            "independent_of_transition_sample_weights"
                        ),
                        "movement_onset_rank_soft_evaluation_policy": (
                            "exclude_teacher_evaluated_rows"
                        ),
                        "movement_onset_rank_interaction": (
                            "additive_with_transition_action_rank_when_enabled"
                        ),
                    }
                    if training_config.movement_onset_rank_weight > 0.0 else {}
                ),
                **(
                    {
                        "movement_speed_change_rank_weight": (
                            training_config.movement_speed_change_rank_weight
                        ),
                        "movement_speed_change_rank_margin": (
                            training_config.transition_action_rank_margin
                        ),
                        "movement_speed_change_rank_margin_source": (
                            "shared_transition_action_rank_margin"
                        ),
                        "movement_speed_change_rank_scope": (
                            "reliable_supervised_focused_speed_changes_while_"
                            "movement_direction_is_held"
                        ),
                        "movement_speed_change_rank_reduction": (
                            "mean_per_episode_over_speed_change_labels"
                        ),
                        "movement_speed_change_rank_sample_weighting": (
                            "independent_of_transition_sample_weights"
                        ),
                        "movement_speed_change_rank_soft_evaluation_policy": (
                            "exclude_teacher_evaluated_rows"
                        ),
                        "movement_speed_change_rank_interaction": (
                            "additive_with_transition_action_rank_when_enabled"
                        ),
                    }
                    if training_config.movement_speed_change_rank_weight > 0.0 else
                    {}
                ),
                **(
                    {
                        "motion_boundary_rank_weight": (
                            training_config.motion_boundary_rank_weight
                        ),
                        "motion_boundary_rank_margin": (
                            training_config.motion_boundary_rank_margin
                        ),
                        "motion_boundary_rank_lookback": (
                            training_config.motion_boundary_rank_lookback
                        ),
                        "motion_boundary_rank_event_types": [
                            "onset",
                            "stop",
                            "turn",
                            "speed_change",
                        ],
                        "motion_boundary_rank_pairing": (
                            "preceding_old_action_over_future_new_action_and_"
                            "event_new_action_over_old_action"
                        ),
                        "motion_boundary_rank_side_weighting": (
                            "0.5_pre_event_total_and_0.5_event"
                        ),
                        "motion_boundary_rank_reduction": (
                            "equal_side_weighted_pairs_per_event_then_mean_"
                            "events_per_episode_then_mean_episodes"
                        ),
                        "motion_boundary_rank_optimizer_step_unit": (
                            "complete_episode"
                        ),
                        "motion_boundary_rank_hard_state_policy": (
                            "supervision_mask_and_not_teacher_evaluated_with_"
                            "at_most_one_intervening_nonhard_row"
                        ),
                        "motion_boundary_rank_soft_evaluation_policy": (
                            "exclude_teacher_evaluated_rows_from_events_and_"
                            "lookback_even_when_soft_loss_is_disabled"
                        ),
                        "motion_boundary_rank_episode_admission": (
                            "input_episode_blocks_must_be_strict_successes;_"
                            "outcome_is_not_a_model_input_or_npz_field"
                        ),
                        "train_motion_boundary_rank_events": (
                            train_motion_boundaries.events
                        ),
                        "train_motion_boundary_rank_pairs": (
                            train_motion_boundaries.pairs
                        ),
                        "train_motion_boundary_rank_event_counts": {
                            kind: train_motion_boundaries.event_kinds.count(kind)
                            for kind in (
                                "onset",
                                "stop",
                                "turn",
                                "speed_change",
                            )
                        },
                        "validation_motion_boundary_rank_events": (
                            validation_motion_boundaries.events
                        ),
                        "validation_motion_boundary_rank_pairs": (
                            validation_motion_boundaries.pairs
                        ),
                        "validation_motion_boundary_rank_event_counts": {
                            kind: validation_motion_boundaries.event_kinds.count(kind)
                            for kind in (
                                "onset",
                                "stop",
                                "turn",
                                "speed_change",
                            )
                        },
                    }
                    if training_config.motion_boundary_rank_weight > 0.0 else
                    {}
                ),
                **(
                    {
                        "safety_correction_pairwise_rank_weight": (
                            training_config.safety_correction_pairwise_rank_weight
                        ),
                        "safety_correction_pairwise_rank_margin": (
                            training_config.safety_correction_pairwise_rank_margin
                        ),
                        "safety_correction_pairwise_rank_preferred": (
                            "demonstrations.actions_on_explicit_correction_mask"
                        ),
                        "safety_correction_pairwise_rank_rejected": (
                            "frozen_initial_checkpoint_policy_argmax"
                        ),
                        "safety_correction_pairwise_rank_loss": (
                            "relu(margin-preferred_logit+rejected_logit)"
                        ),
                        "safety_correction_pairwise_rank_reduction": (
                            "mean_corrections_per_episode_then_mean_episodes"
                        ),
                        "safety_correction_pairwise_rank_optimizer_step_unit": (
                            "complete_episode"
                        ),
                        "safety_correction_pairwise_rank_other_action_losses": (
                            "exclude_correction_mask_rows"
                        ),
                        "safety_correction_pairwise_rank_horizontal_reflection": (
                            "map_preferred_and_frozen_rejected_action_ids"
                        ),
                        "train_safety_correction_pairwise_rank_labels": (
                            train_safety_correction_labels
                        ),
                        "validation_safety_correction_pairwise_rank_labels": (
                            validation_safety_correction_labels
                        ),
                    }
                    if training_config.safety_correction_pairwise_rank_weight > 0.0
                    else {}
                ),
                **(
                    {
                        "safety_correction_minimal_edit_weight": (
                            training_config.safety_correction_minimal_edit_weight
                        ),
                        "safety_correction_minimal_edit_margin": (
                            training_config.safety_correction_minimal_edit_margin
                        ),
                        "safety_correction_minimal_edit_reference": (
                            "frozen_initial_checkpoint_policy_logits"
                        ),
                        "safety_correction_minimal_edit_target": (
                            "copy_reference_logits_then_set_only_preferred_to_"
                            "reference_max_plus_margin"
                        ),
                        "safety_correction_minimal_edit_loss": (
                            "kl(target_distribution||current_distribution)"
                        ),
                        "safety_correction_minimal_edit_other_actions": (
                            "retain_frozen_reference_logits"
                        ),
                        "safety_correction_minimal_edit_reduction": (
                            "mean_corrections_per_episode_then_mean_episodes"
                        ),
                        "safety_correction_minimal_edit_optimizer_step_unit": (
                            "complete_episode"
                        ),
                        "safety_correction_minimal_edit_horizontal_reflection": (
                            "map_reference_logit_axis_and_preferred_action_id"
                        ),
                        "train_safety_correction_minimal_edit_labels": (
                            train_safety_correction_labels
                        ),
                        "validation_safety_correction_minimal_edit_labels": (
                            validation_safety_correction_labels
                        ),
                    }
                    if training_config.safety_correction_minimal_edit_weight > 0.0
                    else {}
                ),
                **(
                    {
                        "initial_policy_kl_weight": (
                            training_config.initial_policy_kl_weight
                        ),
                        "initial_policy_kl_reference": (
                            "frozen_initial_checkpoint_policy"
                        ),
                        "initial_policy_kl_mask": "not_correction_mask",
                        "initial_policy_kl_reduction": (
                            "mean_actual_anchor_rows_per_episode_then_mean_episodes"
                        ),
                    }
                    if training_config.initial_policy_kl_weight > 0.0 else {}
                ),
                **(
                    {
                        "soft_action_loss_weight": (
                            training_config.soft_action_loss_weight
                        ),
                        "soft_action_collision_rank_weight": (
                            training_config.soft_action_collision_rank_weight
                        ),
                        "soft_action_collision_rank_margin": (
                            training_config.soft_action_collision_rank_margin
                        ),
                        "soft_action_temperature": (
                            training_config.soft_action_temperature
                        ),
                        "soft_action_safety_margin": (
                            training_config.soft_action_safety_margin
                        ),
                        "soft_action_objective": (
                            "negative_log_regret_weighted_acceptable_probability_mass"
                        ),
                        "soft_action_collision_ranking": (
                            "best_acceptable_logit_above_best_predicted_collision"
                            if training_config.soft_action_collision_rank_weight > 0.0
                            else "disabled"
                        ),
                        "soft_action_acceptance": (
                            "noncolliding_and_minimum_margin_threshold_and_"
                            "teacher_moving_state_plus_selected_teacher"
                        ),
                        "soft_action_mask": (
                            "teacher_action_evaluation_mask_independent_of_"
                            "hard_supervision_mask"
                        ),
                        "hard_action_policy": (
                            "disabled_on_teacher_evaluated_rows"
                        ),
                    }
                    if training_config.soft_action_loss_weight > 0.0 else
                    {}
                ),
                "previous_action_dropout_probability": (
                    training_config.previous_action_dropout_probability
                ),
                "future_visual_loss_weight": (
                    training_config.future_visual_loss_weight
                ),
                "future_visual_horizons_decisions": list(
                    training_config.future_visual_horizons
                ),
                **(
                    {
                        "action_supervision": "supervision_mask",
                        "risk_supervision": "all_decisions",
                        "recurrent_context": "complete_episode",
                    }
                    if training_config.correction_only else {}
                ),
                "transition_source": (
                    "consecutive_supervised_teacher_actions_with_at_most_"
                    "one_masked_mixed_window_within_episode"
                ),
                "direction_semantics": "move_xy_ignoring_slow_mode",
                "train_movement_onsets": int(train_onsets.sum()),
                "train_movement_stops": int(train_stops.sum()),
                "train_direction_changes": int(train_direction_changes.sum()),
                "train_movement_speed_changes": int(train_speed_changes.sum()),
                "validation_movement_onsets": int(validation_onsets.sum()),
                "validation_movement_stops": int(validation_stops.sum()),
                "validation_direction_changes": int(
                    validation_direction_changes.sum()
                ),
                "validation_movement_speed_changes": int(
                    validation_speed_changes.sum()
                ),
            },
            "future_visual_prediction": {
                "enabled": training_config.future_visual_loss_weight > 0.0,
                "source": "per_decision_gru_hidden",
                "target": (
                    "stop_gradient_concatenated_global_local_visual_encoding"
                ),
                "target_observations": (
                    "future_latest_semantic_frames_within_same_episode"
                ),
                "horizons_decisions": list(
                    training_config.future_visual_horizons
                ),
                "loss": "mean_feature_smooth_l1_then_mean_valid_target",
                "loss_weight": training_config.future_visual_loss_weight,
                "predictor": "independent_linear_head_per_horizon",
                "predictor_checkpointed": False,
                "predictor_lifecycle": "training_and_validation_only",
                "validation_loss_included": True,
                "reported_metrics": [
                    "train_future_visual_loss",
                    "validation_future_visual_loss",
                    "train_future_visual_labels",
                    "validation_future_visual_labels",
                ],
                "episode_boundary_policy": "discard_cross_episode_targets",
                "horizontal_reflection": "matches_source_episode_augmentation",
                "target_inputs": [
                    "global_semantic_observation",
                    "local_semantic_observation",
                ],
                "auxiliary_inputs_added": [],
                "excluded_target_inputs": [
                    "explicit_absolute_frame_index",
                    "explicit_absolute_position_scalar",
                    "scenario_or_attack_token",
                    "phase",
                    "route",
                    "external_memory",
                    "action",
                    "risk",
                ],
                "train_labels": train_future_visual_labels,
                "validation_labels": validation_future_visual_labels,
                "visual_latent_size": policy_config.feature_size * 2,
            },
            **(
                {
                    "correction_only_supervision": {
                        "action_loss_mask": "supervision_mask",
                        "risk_loss_mask": "all_latest_frame_decisions",
                        "recurrent_context": "complete_episode",
                        "train_action_labels": sum(
                            _episode_supervised_labels(demonstrations, episode)
                            for episode in train_episodes
                        ),
                        "train_risk_labels": sum(
                            episode.decisions for episode in train_episodes
                        ),
                        "validation_action_labels": sum(
                            _episode_supervised_labels(demonstrations, episode)
                            for episode in validation_episodes
                        ),
                        "validation_risk_labels": sum(
                            episode.decisions for episode in validation_episodes
                        ),
                    },
                }
                if training_config.correction_only else {}
            ),
            "episode_balance": {
                "enabled": training_config.episode_balanced,
                "optimizer_step_unit": (
                    "complete_episode" if training_config.episode_balanced else "tbptt_chunk"
                ),
                "episode_order": (
                    "deterministic_epoch_shuffle"
                    if training_config.episode_balanced else
                    "archive_order"
                ),
                "optimizer_steps_per_epoch": _optimizer_steps_per_epoch(
                    demonstrations,
                    train_episodes,
                    chunk_length=training_config.chunk_length,
                    episode_balanced=training_config.episode_balanced,
                    risk_loss_weight=training_config.risk_loss_weight,
                    risk_on_all_decisions=training_config.correction_only,
                    future_visual_loss_weight=(
                        training_config.future_visual_loss_weight
                    ),
                    future_visual_horizons=(
                        training_config.future_visual_horizons
                    ),
                    initial_policy_kl_weight=(
                        training_config.initial_policy_kl_weight
                    ),
                    hard_action_terms_enabled=any((
                        training_config.exact_action_loss_weight,
                        training_config.direction_loss_weight,
                        training_config.speed_loss_weight,
                        training_config.direction_consistency_weight,
                        training_config.action_consistency_weight,
                    )),
                    soft_action_loss_enabled=(
                        training_config.soft_action_loss_weight > 0.0
                    ),
                    transition_action_rank_weight=(
                        training_config.transition_action_rank_weight
                    ),
                    movement_onset_rank_weight=(
                        training_config.movement_onset_rank_weight
                    ),
                    movement_speed_change_rank_weight=(
                        training_config.movement_speed_change_rank_weight
                    ),
                    motion_boundary_rank_weight=(
                        training_config.motion_boundary_rank_weight
                    ),
                    motion_boundary_rank_lookback=(
                        training_config.motion_boundary_rank_lookback
                    ),
                    safety_correction_pairwise_rank_weight=(
                        training_config.safety_correction_pairwise_rank_weight
                    ),
                    safety_correction_top1_rank_weight=(
                        training_config.safety_correction_top1_rank_weight
                    ),
                    safety_correction_minimal_edit_weight=(
                        training_config.safety_correction_minimal_edit_weight
                    ),
                ),
                "optimizer_loss_reduction": (
                    "mean_of_per_episode_weighted_means"
                    if training_config.episode_balanced else
                    "per_tbptt_chunk_weighted_mean"
                ),
                "validation_loss_reduction": (
                    "mean_of_per_episode_weighted_means"
                    if training_config.episode_balanced else
                    "global_weighted_label_mean"
                ),
                "train_supervised_labels": {
                    str(episode.episode_id): _episode_supervised_labels(
                        demonstrations, episode,
                    )
                    for episode in train_episodes
                },
            },
            "checkpoint_selection": (
                "minimum_complete_episode_validation_loss"
                if training_config.restore_best_validation else
                "final_epoch"
            ),
            "selected_epoch": (
                best_epoch if training_config.restore_best_validation else history[-1].epoch
            ),
            "selected_validation_loss": (
                best_validation_loss
                if training_config.restore_best_validation else
                history[-1].validation_loss
            ),
        })
        save_checkpoint(
            model,
            output,
            policy_config=policy_config,
            history=history,
            training_config=training_config,  # type: ignore[arg-type]
            training_data=metadata,
        )
    return model, history


__all__ = [
    "EpisodeSequence",
    "EpisodeSplit",
    "StatefulPassMetrics",
    "StatefulTrainingConfig",
    "drop_previous_action_context",
    "evaluate_stateful_policy",
    "initialize_visual_encoders",
    "ordered_episode_sequences",
    "reflect_horizontal_action_context",
    "reflect_horizontal_stream_batch",
    "reflect_horizontal_teacher_action_evidence",
    "split_episode_ids",
    "teacher_transition_sample_weights",
    "train_stateful_behavior_cloning",
]
