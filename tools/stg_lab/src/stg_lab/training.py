from __future__ import annotations

from dataclasses import asdict, dataclass
import json
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

from .policy import HumanVisionPolicy, PolicyConfig


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
    device: str = "auto"


@dataclass(frozen=True, slots=True)
class TrainingMetrics:
    epoch: int
    train_loss: float
    validation_loss: float
    action_accuracy: float
    risk_mae: float


@dataclass(slots=True)
class Demonstrations:
    global_frames: np.ndarray
    local_frames: np.ndarray
    actions: np.ndarray
    risks: np.ndarray
    memory: np.ndarray | None = None
    episode_ids: np.ndarray | None = None
    supervision_mask: np.ndarray | None = None

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
        if self.memory is not None and self.memory.shape[:2] != (samples, steps):
            raise ValueError("memory must align with visual sequences")
        if self.episode_ids is not None and self.episode_ids.shape != (samples,):
            raise ValueError("episode_ids must contain one group id per sample")
        if self.supervision_mask is not None and self.supervision_mask.shape != (samples, steps):
            raise ValueError("supervision_mask must align with actions")
        if np.any(self.actions < 0) or np.any(self.actions >= 18):
            raise ValueError("action labels must be in [0, 18)")

    def save(self, path: str | Path) -> None:
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, np.ndarray] = {
            # Semantic channels are normalized; float16 halves large planner
            # demonstration archives without changing action/risk labels.
            "global_frames": self.global_frames.astype(np.float16),
            "local_frames": self.local_frames.astype(np.float16),
            "actions": self.actions.astype(np.int64),
            "risks": self.risks.astype(np.float32),
        }
        if self.memory is not None:
            payload["memory"] = self.memory.astype(np.float32)
        if self.episode_ids is not None:
            payload["episode_ids"] = self.episode_ids.astype(np.int64)
        if self.supervision_mask is not None:
            payload["supervision_mask"] = self.supervision_mask.astype(np.uint8)
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: str | Path) -> "Demonstrations":
        with np.load(path) as data:
            demonstrations = cls(
                global_frames=data["global_frames"],
                local_frames=data["local_frames"],
                actions=data["actions"],
                risks=data["risks"],
                memory=data["memory"] if "memory" in data.files else None,
                episode_ids=data["episode_ids"] if "episode_ids" in data.files else None,
                supervision_mask=(
                    data["supervision_mask"].astype(bool)
                    if "supervision_mask" in data.files else None
                ),
            )
        demonstrations.validate()
        return demonstrations


if torch is not None:

    class _DemonstrationDataset(Dataset):
        def __init__(self, demonstrations: Demonstrations, indices: np.ndarray, memory_size: int) -> None:
            self.demonstrations = demonstrations
            self.indices = indices
            self.memory_size = memory_size

        def __len__(self) -> int:
            return len(self.indices)

        def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
            item = int(self.indices[index])
            demo = self.demonstrations
            if demo.memory is None:
                memory = np.zeros((demo.actions.shape[1], self.memory_size), dtype=np.float32)
            else:
                memory = demo.memory[item]
            if demo.supervision_mask is None:
                mask = np.zeros(demo.actions.shape[1], dtype=bool)
                mask[-1] = True
            else:
                mask = demo.supervision_mask[item]
            return (
                torch.from_numpy(demo.global_frames[item]).float(),
                torch.from_numpy(demo.local_frames[item]).float(),
                torch.from_numpy(memory).float(),
                torch.from_numpy(demo.actions[item]).long(),
                torch.from_numpy(demo.risks[item]).float(),
                torch.from_numpy(mask).bool(),
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
    memory_sequences: list[np.ndarray] = []
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
            if demonstrations.memory is not None:
                memory_sequences.append(demonstrations.memory[selected, -1])
            episode_ids.append(int(episode_id))
    if not global_sequences:
        raise ValueError("no episode is long enough for the requested sequence length")
    result = Demonstrations(
        global_frames=np.stack(global_sequences).astype(np.float16, copy=False),
        local_frames=np.stack(local_sequences).astype(np.float16, copy=False),
        actions=np.stack(action_sequences).astype(np.int64, copy=False),
        risks=np.stack(risk_sequences).astype(np.float32, copy=False),
        memory=(np.stack(memory_sequences).astype(np.float32, copy=False) if memory_sequences else None),
        episode_ids=np.asarray(episode_ids, dtype=np.int64),
        supervision_mask=np.ones((len(global_sequences), sequence_length), dtype=bool),
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
    if not 0.0 <= training_config.class_balance_power <= 1.0:
        raise ValueError("class_balance_power must be in [0, 1]")
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

    train_dataset = _DemonstrationDataset(demonstrations, train_indices, policy_config.memory_size)
    validation_dataset = _DemonstrationDataset(demonstrations, validation_indices, policy_config.memory_size)
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
            labels = train_actions[:, -1]
        else:
            labels = train_actions[demonstrations.supervision_mask[train_indices]]
        counts = np.bincount(labels, minlength=policy_config.action_count).astype(np.float64)
        present = counts > 0
        weights = np.zeros(policy_config.action_count, dtype=np.float32)
        inverse_frequency = len(labels) / (present.sum() * counts[present])
        weights[present] = inverse_frequency ** training_config.class_balance_power
        # A rare teacher correction should matter without destabilizing the
        # small imitation runs used during iteration.
        weights[present] = np.minimum(weights[present], 10.0)
        action_weights = torch.from_numpy(weights).to(device)

    for epoch in range(1, training_config.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_items = 0
        for global_frames, local_frames, memory, actions, risks, mask in train_loader:
            global_frames = global_frames.to(device)
            local_frames = local_frames.to(device)
            memory = memory.to(device)
            actions = actions.to(device)
            risks = risks.to(device)
            mask = mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, predicted_risk, _ = model(global_frames, local_frames, memory)
            action_loss = F.cross_entropy(logits[mask], actions[mask], weight=action_weights)
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
            for global_frames, local_frames, memory, actions, risks, mask in validation_loader:
                global_frames = global_frames.to(device)
                local_frames = local_frames.to(device)
                memory = memory.to(device)
                actions = actions.to(device)
                risks = risks.to(device)
                mask = mask.to(device)
                logits, predicted_risk, _ = model(global_frames, local_frames, memory)
                action_loss = F.cross_entropy(logits[mask], actions[mask])
                risk_loss = F.smooth_l1_loss(predicted_risk[mask], risks[mask])
                loss = action_loss + training_config.risk_loss_weight * risk_loss
                validation_loss_sum += float(loss) * len(actions)
                validation_items += len(actions)
                correct += int((logits.argmax(dim=-1)[mask] == actions[mask]).sum())
                labels += int(mask.sum())
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
            "version": 2,
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


def load_checkpoint(path: str | Path, *, device: str = "cpu") -> tuple[HumanVisionPolicy, dict[str, Any]]:
    if torch is None:
        raise RuntimeError("PyTorch is required for policy checkpoints")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config_values = dict(checkpoint["policy_config"])
    # Checkpoints created before inference modes used one-frame recurrent
    # streaming during evaluation. Preserve that behavior when loading them.
    config_values.setdefault("inference_mode", "stream")
    config = PolicyConfig(**config_values)
    model = HumanVisionPolicy(config).to(device)
    model.load_state_dict(checkpoint["state_dict"])
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
