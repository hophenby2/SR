from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

try:
    import torch
    from torch import Tensor, nn
except ImportError:  # pragma: no cover - exercised in the base, non-training environment
    torch = None
    Tensor = object  # type: ignore[assignment,misc]
    nn = None


@dataclass(frozen=True, slots=True)
class PolicyConfig:
    channels: int = 6
    feature_size: int = 96
    recurrent_size: int = 128
    memory_size: int = 4
    action_count: int = 18
    inference_mode: str = "window"

    def __post_init__(self) -> None:
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
            recurrent_input = config.feature_size * 2 + config.memory_size
            self.recurrent = nn.GRU(recurrent_input, config.recurrent_size, batch_first=True)
            self.policy_head = nn.Linear(config.recurrent_size, config.action_count)
            self.risk_head = nn.Sequential(nn.Linear(config.recurrent_size, 1), nn.Sigmoid())

        def forward(
            self,
            global_frames: Tensor,
            local_frames: Tensor,
            memory: Tensor | None = None,
            hidden: Tensor | None = None,
        ) -> tuple[Tensor, Tensor, Tensor]:
            if global_frames.ndim != 5 or local_frames.ndim != 5:
                raise ValueError("visual inputs must have shape [batch, time, channel, height, width]")
            batch, steps = global_frames.shape[:2]
            global_features = self.global_encoder(global_frames.flatten(0, 1)).reshape(batch, steps, -1)
            local_features = self.local_encoder(local_frames.flatten(0, 1)).reshape(batch, steps, -1)
            if memory is None:
                memory = torch.zeros(
                    (batch, steps, self.config.memory_size),
                    dtype=global_frames.dtype,
                    device=global_frames.device,
                )
            elif memory.ndim == 2:
                memory = memory[:, None, :].expand(-1, steps, -1)
            features = torch.cat((global_features, local_features, memory), dim=-1)
            recurrent, hidden = self.recurrent(features, hidden)
            return self.policy_head(recurrent), self.risk_head(recurrent).squeeze(-1), hidden

else:

    class HumanVisionPolicy:  # type: ignore[no-redef]
        def __init__(self, config: PolicyConfig = PolicyConfig()) -> None:
            raise RuntimeError("PyTorch is required for HumanVisionPolicy; install stg-lab[train]")


def safety_shield(logits: np.ndarray, allowed_actions: Iterable[int]) -> int:
    allowed = np.asarray(sorted(set(allowed_actions)), dtype=np.int64)
    if allowed.size == 0:
        return int(np.argmax(logits))
    return int(allowed[np.argmax(logits[allowed])])
