from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

from .protocol import Action

try:
    import torch
    from torch import Tensor, nn
except ImportError:  # pragma: no cover - exercised in the base, non-training environment
    torch = None
    Tensor = object  # type: ignore[assignment,misc]
    nn = None


PROFICIENCY_VECTOR_SIZE = 5


@dataclass(frozen=True, slots=True)
class PlayerProficiencyProfile:
    """Interpretable human execution limits applied around one shared policy."""

    name: str
    reaction_delay_frames: int
    direction_hold_frames: int
    prediction_horizon_frames: int
    shield_probability: float
    suboptimal_action_probability: float

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("proficiency profile name cannot be empty")
        for field_name in (
            "reaction_delay_frames",
            "direction_hold_frames",
            "prediction_horizon_frames",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a nonnegative integer")
        for field_name in ("shield_probability", "suboptimal_action_probability"):
            value = getattr(self, field_name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be in [0, 1]")

    def vector(self) -> np.ndarray:
        return np.asarray((
            min(self.reaction_delay_frames / 12.0, 1.0),
            min(self.direction_hold_frames / 12.0, 1.0),
            min(self.prediction_horizon_frames / 24.0, 1.0),
            self.shield_probability,
            self.suboptimal_action_probability,
        ), dtype=np.float32)


PROFICIENCY_PROFILES = {
    "novice": PlayerProficiencyProfile(
        name="novice",
        reaction_delay_frames=6,
        direction_hold_frames=9,
        prediction_horizon_frames=3,
        shield_probability=0.25,
        suboptimal_action_probability=0.04,
    ),
    "intermediate": PlayerProficiencyProfile(
        name="intermediate",
        reaction_delay_frames=3,
        direction_hold_frames=6,
        prediction_horizon_frames=6,
        shield_probability=0.65,
        suboptimal_action_probability=0.01,
    ),
    "expert": PlayerProficiencyProfile(
        name="expert",
        reaction_delay_frames=0,
        direction_hold_frames=3,
        prediction_horizon_frames=12,
        shield_probability=1.0,
        suboptimal_action_probability=0.0,
    ),
}


def available_proficiencies() -> tuple[str, ...]:
    return tuple(PROFICIENCY_PROFILES)


def resolve_proficiency(
    value: str | PlayerProficiencyProfile,
) -> PlayerProficiencyProfile:
    if isinstance(value, PlayerProficiencyProfile):
        return value
    normalized = str(value).strip().lower()
    try:
        return PROFICIENCY_PROFILES[normalized]
    except KeyError as error:
        choices = ", ".join(available_proficiencies())
        raise ValueError(f"unknown proficiency profile {value!r}; choose {choices}") from error


def proficiency_vector(
    value: str | PlayerProficiencyProfile = "expert",
) -> np.ndarray:
    return resolve_proficiency(value).vector()


class ProficiencyRuntime:
    """Seeded motor/noise state for one policy episode."""

    def __init__(
        self,
        profile: str | PlayerProficiencyProfile = "expert",
        *,
        seed: int = 0,
    ) -> None:
        self.profile = resolve_proficiency(profile)
        self.reset(seed)

    def reset(self, seed: int) -> None:
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)
        self._reaction_queue: deque[Action] = deque()
        self._last_action: Action | None = None
        self._direction_held_frames = 0

    def _sample_discrete(self, logits: np.ndarray) -> int:
        values = np.asarray(logits, dtype=np.float64)
        if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
            raise ValueError("policy logits must be a finite one-dimensional array")
        ranking = np.argsort(-values, kind="stable")
        probability = self.profile.suboptimal_action_probability
        choose_suboptimal = (
            len(ranking) > 1
            and probability > 0.0
            and (probability >= 1.0 or self._rng.random() < probability)
        )
        return int(ranking[1 if choose_suboptimal else 0])

    def preferred_action(self, logits: np.ndarray, *, decision_interval: int) -> Action:
        if decision_interval <= 0:
            raise ValueError("decision_interval must be positive")
        sampled = Action.from_discrete(self._sample_discrete(logits))
        self._reaction_queue.append(sampled)
        delay = math.ceil(self.profile.reaction_delay_frames / decision_interval)
        candidate = (
            self._reaction_queue.popleft()
            if len(self._reaction_queue) > delay else
            Action()
        )
        if (
            self._last_action is not None
            and (candidate.move_x, candidate.move_y)
            != (self._last_action.move_x, self._last_action.move_y)
            and self._direction_held_frames < self.profile.direction_hold_frames
        ):
            return self._last_action
        return candidate

    def commit(self, action: Action, *, decision_interval: int) -> None:
        if decision_interval <= 0:
            raise ValueError("decision_interval must be positive")
        direction = action.move_x, action.move_y
        if (
            self._last_action is not None
            and direction == (self._last_action.move_x, self._last_action.move_y)
        ):
            self._direction_held_frames += decision_interval
        else:
            self._direction_held_frames = decision_interval
        self._last_action = action

    def should_apply_shield(self) -> bool:
        probability = self.profile.shield_probability
        return probability >= 1.0 or (
            probability > 0.0 and self._rng.random() < probability
        )


@dataclass(frozen=True, slots=True)
class PolicyConfig:
    channels: int = 6
    feature_size: int = 96
    recurrent_size: int = 128
    memory_size: int = 4
    proficiency_size: int = PROFICIENCY_VECTOR_SIZE
    action_count: int = 18
    inference_mode: str = "window"

    def __post_init__(self) -> None:
        if self.proficiency_size not in {0, PROFICIENCY_VECTOR_SIZE}:
            raise ValueError(
                f"proficiency_size must be 0 or {PROFICIENCY_VECTOR_SIZE}"
            )
        if self.inference_mode not in {"window", "stream"}:
            raise ValueError("inference_mode must be 'window' or 'stream'")


if nn is not None:

    class _VisualEncoder(nn.Module):
        def __init__(self, channels: int, output_size: int) -> None:
            super().__init__()
            self.network = nn.Sequential(
                nn.Conv2d(channels, 24, kernel_size=5, stride=2, padding=2),
                nn.SiLU(),
                nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1),
                nn.SiLU(),
                nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1),
                nn.SiLU(),
                # MPS does not implement every non-divisible adaptive-pool
                # shape (global input reaches 7x6, local reaches 5x5). Fixed
                # bilinear resampling retains coarse spatial layout on all
                # supported devices.
                nn.Upsample(size=(4, 4), mode="bilinear", align_corners=False),
                nn.Flatten(),
                nn.Linear(64 * 16, output_size),
                nn.LayerNorm(output_size),
                nn.SiLU(),
            )

        def forward(self, frames: Tensor) -> Tensor:
            return self.network(frames)


    class HumanVisionPolicy(nn.Module):
        def __init__(self, config: PolicyConfig = PolicyConfig()) -> None:
            super().__init__()
            self.config = config
            self.global_encoder = _VisualEncoder(config.channels, config.feature_size)
            self.local_encoder = _VisualEncoder(config.channels, config.feature_size)
            recurrent_input = (
                config.feature_size * 2
                + config.memory_size
                + config.proficiency_size
            )
            self.recurrent = nn.GRU(recurrent_input, config.recurrent_size, batch_first=True)
            self.policy_head = nn.Linear(config.recurrent_size, config.action_count)
            self.risk_head = nn.Sequential(nn.Linear(config.recurrent_size, 1), nn.Sigmoid())

        def encode_visual(
            self,
            global_frames: Tensor,
            local_frames: Tensor,
        ) -> Tensor:
            """Encode aligned semantic observations without nonvisual context."""

            if global_frames.ndim != 5 or local_frames.ndim != 5:
                raise ValueError(
                    "visual inputs must have shape [batch, time, channel, height, width]"
                )
            if global_frames.shape[:3] != local_frames.shape[:3]:
                raise ValueError("global and local visual inputs must align")
            batch, steps = global_frames.shape[:2]
            global_features = self.global_encoder(
                global_frames.flatten(0, 1)
            ).reshape(batch, steps, -1)
            local_features = self.local_encoder(
                local_frames.flatten(0, 1)
            ).reshape(batch, steps, -1)
            return torch.cat((global_features, local_features), dim=-1)

        def forward_with_recurrent(
            self,
            global_frames: Tensor,
            local_frames: Tensor,
            memory: Tensor | None = None,
            proficiency: Tensor | None = None,
            hidden: Tensor | None = None,
        ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
            """Run the policy and expose each decision's GRU hidden state."""

            visual_features = self.encode_visual(global_frames, local_frames)
            batch, steps = global_frames.shape[:2]
            if memory is None:
                memory = torch.zeros(
                    (batch, steps, self.config.memory_size),
                    dtype=global_frames.dtype,
                    device=global_frames.device,
                )
            elif memory.ndim == 2:
                memory = memory[:, None, :].expand(-1, steps, -1)
            if self.config.proficiency_size == 0:
                proficiency = torch.empty(
                    (batch, steps, 0),
                    dtype=global_frames.dtype,
                    device=global_frames.device,
                )
            elif proficiency is None:
                default = torch.as_tensor(
                    proficiency_vector("expert"),
                    dtype=global_frames.dtype,
                    device=global_frames.device,
                )
                proficiency = default[None, None, :].expand(batch, steps, -1)
            elif proficiency.ndim == 2:
                proficiency = proficiency[:, None, :].expand(-1, steps, -1)
            if proficiency.shape != (batch, steps, self.config.proficiency_size):
                raise ValueError(
                    "proficiency input must have shape "
                    f"[batch, time, {self.config.proficiency_size}]"
                )
            features = torch.cat(
                (visual_features, memory, proficiency), dim=-1,
            )
            recurrent, hidden = self.recurrent(features, hidden)
            return (
                self.policy_head(recurrent),
                self.risk_head(recurrent).squeeze(-1),
                hidden,
                recurrent,
            )

        def forward(
            self,
            global_frames: Tensor,
            local_frames: Tensor,
            memory: Tensor | None = None,
            proficiency: Tensor | None = None,
            hidden: Tensor | None = None,
        ) -> tuple[Tensor, Tensor, Tensor]:
            logits, risk, hidden, _recurrent = self.forward_with_recurrent(
                global_frames,
                local_frames,
                memory,
                proficiency,
                hidden,
            )
            return logits, risk, hidden

else:

    class HumanVisionPolicy:  # type: ignore[no-redef]
        def __init__(self, config: PolicyConfig = PolicyConfig()) -> None:
            raise RuntimeError("PyTorch is required for HumanVisionPolicy; install stg-lab[train]")


def safety_shield(logits: np.ndarray, allowed_actions: Iterable[int]) -> int:
    allowed = np.asarray(sorted(set(allowed_actions)), dtype=np.int64)
    if allowed.size == 0:
        return int(np.argmax(logits))
    return int(allowed[np.argmax(logits[allowed])])
