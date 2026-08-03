"""Episode-contiguous truncated backpropagation for streaming policies."""

from __future__ import annotations

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
    Demonstrations,
    TrainingMetrics,
    save_checkpoint,
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
    direction_change_weight: float = 1.0
    episode_balanced: bool = False
    exact_action_loss_weight: float = 1.0
    direction_loss_weight: float = 0.0
    speed_loss_weight: float = 0.0
    direction_consistency_weight: float = 0.0
    correction_only: bool = False
    previous_action_dropout_probability: float = 0.0
    future_visual_loss_weight: float = 0.0
    future_visual_horizons: tuple[int, ...] = DEFAULT_FUTURE_VISUAL_HORIZONS

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.chunk_length <= 0:
            raise ValueError("chunk_length must be positive")
        positive = (
            self.learning_rate,
            self.gradient_clip,
            self.movement_onset_weight,
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
        )
        if not all(
            math.isfinite(value) and value >= 0.0
            for value in action_loss_weights
        ):
            raise ValueError("action-loss weights must be finite and nonnegative")
        if not any(action_loss_weights[:3]):
            raise ValueError("at least one supervised action-loss weight must be positive")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in (0, 1)")
        if not 0.0 <= self.class_balance_power <= 1.0:
            raise ValueError("class_balance_power must be in [0, 1]")
        if not 0.0 <= self.horizontal_reflection_probability <= 1.0:
            raise ValueError("horizontal_reflection_probability must be in [0, 1]")
        if not 0.0 <= self.previous_action_dropout_probability <= 1.0:
            raise ValueError("previous_action_dropout_probability must be in [0, 1]")
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


def _teacher_transition_masks(
    demonstrations: Demonstrations,
    episodes: Sequence[EpisodeSequence],
) -> tuple[np.ndarray, np.ndarray]:
    """Locate supervised teacher motion transitions without scene-derived inputs."""

    samples = demonstrations.actions.shape[0]
    movement_onsets = np.zeros(samples, dtype=np.bool_)
    direction_changes = np.zeros(samples, dtype=np.bool_)
    supervised = (
        np.ones(samples, dtype=np.bool_)
        if demonstrations.supervision_mask is None else
        np.asarray(demonstrations.supervision_mask[:, -1], dtype=np.bool_)
    )
    for episode in episodes:
        directions = np.asarray(
            demonstrations.actions[episode.start:episode.stop, -1] % 9,
            dtype=np.int64,
        )
        if len(directions) < 2:
            continue
        previous = directions[:-1]
        current = directions[1:]
        previous_moving = previous != 4
        current_moving = current != 4
        indices = np.arange(episode.start + 1, episode.stop)
        movement_onsets[indices] = current_moving & ~previous_moving
        direction_changes[indices] = (
            current_moving & previous_moving & (current != previous)
        )
    movement_onsets &= supervised
    direction_changes &= supervised
    return movement_onsets, direction_changes


def teacher_transition_sample_weights(
    demonstrations: Demonstrations,
    *,
    episodes: Sequence[EpisodeSequence] | None = None,
    movement_onset_weight: float = 1.0,
    direction_change_weight: float = 1.0,
) -> np.ndarray:
    """Weight teacher movement starts and moving-direction changes per episode.

    Slow-mode changes do not count as direction changes, and the first action in
    an episode has no inferred transition. Unsupervised actions may provide the
    preceding teacher direction but never receive a non-unit loss weight.
    """

    demonstrations.validate()
    for name, value in (
        ("movement_onset_weight", movement_onset_weight),
        ("direction_change_weight", direction_change_weight),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    selected = ordered_episode_sequences(demonstrations) if episodes is None else episodes
    movement_onsets, direction_changes = _teacher_transition_masks(
        demonstrations, selected,
    )
    weights = np.ones(demonstrations.actions.shape[0], dtype=np.float32)
    weights[movement_onsets] = movement_onset_weight
    weights[direction_changes] = direction_change_weight
    return weights


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
) -> int:
    steps = 0
    for episode in episodes:
        episode_has_objective = False
        for start in range(episode.start, episode.stop, chunk_length):
            stop = min(start + chunk_length, episode.stop)
            if demonstrations.supervision_mask is None:
                action_labels = stop - start
            else:
                action_labels = int(np.count_nonzero(
                    demonstrations.supervision_mask[start:stop, -1]
                ))
            risk_labels = (stop - start) if risk_on_all_decisions else action_labels
            future_labels = _chunk_future_visual_labels(
                episode,
                start,
                stop,
                future_visual_horizons,
            )
            chunk_has_objective = bool(
                action_labels
                or (risk_loss_weight > 0.0 and risk_labels)
                or (future_visual_loss_weight > 0.0 and future_labels)
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
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
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
    return global_frames, local_frames, memory, proficiency, actions, risks, mask


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
    direction_change_weight: float = 1.0,
    episode_balanced: bool = False,
    exact_action_loss_weight: float = 1.0,
    direction_loss_weight: float = 0.0,
    speed_loss_weight: float = 0.0,
    direction_consistency_weight: float = 0.0,
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
    if (
        action_count != 18
        and (
            direction_loss_weight > 0.0
            or speed_loss_weight > 0.0
            or direction_consistency_weight > 0.0
        )
    ):
        raise ValueError("factorized action losses require the 18-action vocabulary")
    _validate_demonstration_features(
        demonstrations,
        memory_size=memory_size,
        proficiency_size=proficiency_size,
    )
    training = optimizer is not None
    model.train(training)
    if future_visual_predictor is not None:
        future_visual_predictor.train(training)
    gradient_parameters = list(model.parameters())
    if future_visual_predictor is not None:
        gradient_parameters.extend(future_visual_predictor.parameters())
    transition_weights = teacher_transition_sample_weights(
        demonstrations,
        episodes=episodes,
        movement_onset_weight=movement_onset_weight,
        direction_change_weight=direction_change_weight,
    )
    movement_onsets, direction_changes = _teacher_transition_masks(
        demonstrations, episodes,
    )
    total_action_loss = 0.0
    total_action_weight = 0.0
    balanced_objective = 0.0
    balanced_episodes = 0
    total_risk_error = 0.0
    total_risk_loss = 0.0
    total_future_visual_loss = 0.0
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
            previous_direction_probabilities: Tensor | None = None
            previous_teacher_direction: Tensor | None = None
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
            episode_action_weight = float(
                transition_weights[episode.start:episode.stop][episode_supervised].sum()
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
            reflect_episode = (
                training
                and horizontal_reflection_probability > 0.0
                and (augmentation_rng or random).random()
                < horizontal_reflection_probability
            )
            episode_has_objective = bool(
                episode_labels
                or (risk_loss_weight > 0.0 and episode_risk_labels)
                or (
                    future_visual_loss_weight > 0.0
                    and episode_future_visual_labels
                )
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
                ) = batch
                if reflect_episode:
                    global_frames, local_frames, actions = reflect_horizontal_stream_batch(
                        global_frames,
                        local_frames,
                        actions,
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
                expected_logits = (*actions.shape, action_count)
                if tuple(logits.shape) != expected_logits:
                    raise ValueError(
                        f"policy logits have shape {tuple(logits.shape)}; "
                        f"expected {expected_logits}"
                    )
                if tuple(predicted_risk.shape) != tuple(risks.shape):
                    raise ValueError("policy risk output does not align with decisions")
                chunk_labels = int(mask.sum().item())
                risk_mask = torch.ones_like(mask) if risk_on_all_decisions else mask
                chunk_risk_labels = int(risk_mask.sum().item())
                directions = actions % 9
                factorized = logits.reshape(*logits.shape[:-1], 2, 9)
                direction_logits = torch.logsumexp(factorized, dim=-2)
                direction_probabilities = torch.softmax(direction_logits, dim=-1)
                weighted_action_loss = logits.sum() * 0.0
                chunk_action_weight = 0.0
                if chunk_labels:
                    action_terms = torch.zeros(
                        chunk_labels,
                        dtype=logits.dtype,
                        device=logits.device,
                    )
                    if exact_action_loss_weight > 0.0:
                        action_terms = action_terms + exact_action_loss_weight * (
                            F.cross_entropy(
                                logits[mask],
                                actions[mask],
                                weight=action_weights,
                                reduction="none",
                            )
                        )
                    if direction_loss_weight > 0.0:
                        action_terms = action_terms + direction_loss_weight * (
                            F.cross_entropy(
                                direction_logits[mask],
                                directions[mask],
                                reduction="none",
                            )
                        )
                    if speed_loss_weight > 0.0:
                        speed_logits = torch.logsumexp(factorized, dim=-1)
                        action_terms = action_terms + speed_loss_weight * (
                            F.cross_entropy(
                                speed_logits[mask],
                                actions[mask] // 9,
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
                        action_terms = action_terms + direction_consistency_weight * (
                            consistency[mask] * stable_teacher[mask]
                        )
                    sample_weights = torch.as_tensor(
                        transition_weights[start:stop],
                        dtype=torch.float32,
                        device=device,
                    ).unsqueeze(0)[mask]
                    weighted_action_loss = (action_terms * sample_weights).sum()
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
                )
                if chunk_has_objective:
                    loss = logits.sum() * 0.0
                    if chunk_labels:
                        loss = loss + (
                            weighted_action_loss / (
                                episode_action_weight
                                if episode_balanced else chunk_action_weight
                            )
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
                if chunk_labels:
                    detached_action_loss = float(weighted_action_loss.detach())
                    episode_action_loss += detached_action_loss
                    total_action_loss += detached_action_loss
                    total_action_weight += chunk_action_weight
                    correct += int(
                        (logits.detach().argmax(dim=-1)[mask] == actions[mask]).sum()
                    )
                    labels += chunk_labels
                previous_direction_probabilities = (
                    direction_probabilities[:, -1].detach()
                )
                previous_teacher_direction = directions[:, -1].detach()
                # This is the TBPTT boundary: history remains numerically
                # continuous across the attack, while its graph is truncated.
                hidden = _detach_hidden(next_hidden)
                decisions += stop - start
                chunks += 1
            if episode_has_objective:
                if episode_labels:
                    balanced_objective += episode_action_loss / episode_action_weight
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
                balanced_episodes += 1
                if training and episode_balanced:
                    torch.nn.utils.clip_grad_norm_(
                        gradient_parameters,
                        gradient_clip,
                    )
                    optimizer.step()
                    optimizer_steps += 1

    if labels == 0:
        raise ValueError("selected episodes contain no supervised latest-frame labels")
    if risk_labels == 0:
        raise ValueError("selected episodes contain no valid risk targets")
    if future_visual_loss_weight > 0.0 and future_visual_labels == 0:
        raise ValueError(
            "selected episodes contain no future visual targets at the requested horizons"
        )
    loss = (
        balanced_objective / balanced_episodes
        if episode_balanced else
        total_action_loss / total_action_weight
        + risk_loss_weight * total_risk_loss / risk_labels
        + (
            future_visual_loss_weight
            * total_future_visual_loss
            / future_visual_labels
            if future_visual_labels else
            0.0
        )
    )
    return StatefulPassMetrics(
        loss=loss,
        action_accuracy=correct / labels,
        risk_mae=total_risk_error / risk_labels,
        labels=labels,
        risk_labels=risk_labels,
        decisions=decisions,
        chunks=chunks,
        episodes=len(episodes),
        optimizer_steps=optimizer_steps,
        movement_onsets=int(movement_onsets.sum()),
        direction_changes=int(direction_changes.sum()),
        future_visual_loss=(
            total_future_visual_loss / future_visual_labels
            if future_visual_labels else
            0.0
        ),
        future_visual_labels=future_visual_labels,
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
    direction_change_weight: float = 1.0,
    episode_balanced: bool = False,
    exact_action_loss_weight: float = 1.0,
    direction_loss_weight: float = 0.0,
    speed_loss_weight: float = 0.0,
    direction_consistency_weight: float = 0.0,
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
        direction_change_weight=direction_change_weight,
        episode_balanced=episode_balanced,
        exact_action_loss_weight=exact_action_loss_weight,
        direction_loss_weight=direction_loss_weight,
        speed_loss_weight=speed_loss_weight,
        direction_consistency_weight=direction_consistency_weight,
    )


def initialize_visual_encoders(
    model: HumanVisionPolicy,
    source: HumanVisionPolicy,
) -> None:
    """Transfer only visible-geometry encoders into a fresh recurrent policy."""

    target_config = getattr(model, "config", None)
    source_config = getattr(source, "config", None)
    for field in ("channels", "feature_size"):
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
) -> Tensor:
    if torch is None:  # pragma: no cover - guarded by the training entry point
        raise RuntimeError("PyTorch is required for stateful policy training")
    labels: list[np.ndarray] = []
    for episode in episodes:
        actions = demonstrations.actions[episode.start:episode.stop, -1]
        if demonstrations.supervision_mask is not None:
            actions = actions[
                demonstrations.supervision_mask[episode.start:episode.stop, -1]
            ]
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
    ):
        if getattr(model_config, field, None) != getattr(policy_config, field):
            raise ValueError(f"model config does not match policy_config.{field}")
    model.to(resolved_device)
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
    optimized_parameters = list(model.parameters())
    if future_visual_predictor is not None:
        optimized_parameters.extend(future_visual_predictor.parameters())
    optimizer = torch.optim.AdamW(
        optimized_parameters,
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    action_weights = None
    if training_config.class_balance:
        action_weights = _class_weights(
            demonstrations,
            train_episodes,
            action_count=policy_config.action_count,
            power=training_config.class_balance_power,
            device=resolved_device,
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
            direction_change_weight=training_config.direction_change_weight,
            episode_balanced=training_config.episode_balanced,
            exact_action_loss_weight=training_config.exact_action_loss_weight,
            direction_loss_weight=training_config.direction_loss_weight,
            speed_loss_weight=training_config.speed_loss_weight,
            direction_consistency_weight=(
                training_config.direction_consistency_weight
            ),
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
            direction_change_weight=training_config.direction_change_weight,
            episode_balanced=training_config.episode_balanced,
            exact_action_loss_weight=training_config.exact_action_loss_weight,
            direction_loss_weight=training_config.direction_loss_weight,
            speed_loss_weight=training_config.speed_loss_weight,
            direction_consistency_weight=(
                training_config.direction_consistency_weight
            ),
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
        train_onsets, train_direction_changes = _teacher_transition_masks(
            demonstrations, train_episodes,
        )
        validation_onsets, validation_direction_changes = _teacher_transition_masks(
            demonstrations, validation_episodes,
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
        metadata = dict(training_data or {})
        metadata.update({
            "training_mode": "episode_stateful_tbptt",
            "inference_semantics": "latest_visible_frame_stream",
            "tbptt_chunk_length": training_config.chunk_length,
            "train_episode_ids": list(split.train_episode_ids),
            "validation_episode_ids": list(split.validation_episode_ids),
            "loss_weighting": {
                "class_balance": training_config.class_balance,
                "class_balance_power": training_config.class_balance_power,
                "movement_onset_weight": training_config.movement_onset_weight,
                "direction_change_weight": training_config.direction_change_weight,
                "exact_action_loss_weight": (
                    training_config.exact_action_loss_weight
                ),
                "direction_loss_weight": training_config.direction_loss_weight,
                "speed_loss_weight": training_config.speed_loss_weight,
                "direction_consistency_weight": (
                    training_config.direction_consistency_weight
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
                "transition_source": "adjacent_teacher_actions_within_episode",
                "direction_semantics": "move_xy_ignoring_slow_mode",
                "train_movement_onsets": int(train_onsets.sum()),
                "train_direction_changes": int(train_direction_changes.sum()),
                "validation_movement_onsets": int(validation_onsets.sum()),
                "validation_direction_changes": int(
                    validation_direction_changes.sum()
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
    "split_episode_ids",
    "teacher_transition_sample_weights",
    "train_stateful_behavior_cloning",
]
