"""Compare frozen parent state and visual latents for physical-safety prediction.

This is a development probe, not deployment training.  It uses the same strict
episode roles and physical clearance labels as the temporal residual trainer,
then reports whether a zero-false-override threshold exists on fitting and
calibration episodes.  Validation is audit-only.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments.train_temporal_residual_adapter import (
    EpisodeFeatures,
    _apply_existing_normalization,
    _load_episode,
)
from stg_lab.residual_adapter import load_residual_adapter
from stg_lab.training import Demonstrations, load_checkpoint


@dataclass(slots=True)
class ProbeEpisode:
    labels: EpisodeFeatures
    role: str
    visual: torch.Tensor


class _PhysicalDangerProbe(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.SiLU(),
            nn.LayerNorm(hidden_size),
        )
        self.recurrent = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.danger_head = nn.Linear(hidden_size, 18)

    def forward(
        self,
        values: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        recurrent, hidden = self.recurrent(self.projection(values), hidden)
        return self.danger_head(recurrent), hidden


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _visual_stream(
    parent: Any,
    demonstrations: Demonstrations,
    *,
    device: str,
    chunk_length: int,
) -> torch.Tensor:
    values: list[torch.Tensor] = []
    parent.eval()
    with torch.no_grad():
        for start in range(0, len(demonstrations.actions), chunk_length):
            stop = min(start + chunk_length, len(demonstrations.actions))
            global_frames = torch.from_numpy(
                demonstrations.global_frames[start:stop, -1],
            ).float().unsqueeze(0).to(device)
            local_frames = torch.from_numpy(
                demonstrations.local_frames[start:stop, -1],
            ).float().unsqueeze(0).to(device)
            values.append(
                parent.encode_visual(global_frames, local_frames).detach().cpu()
            )
    return torch.cat(values, dim=1)


def _features(episode: ProbeEpisode, mode: str) -> torch.Tensor:
    if mode == "parent-state":
        return episode.labels.features
    if mode == "visual":
        return episode.visual
    if mode == "combined":
        return torch.cat((episode.labels.features, episode.visual), dim=-1)
    raise ValueError(f"unsupported probe mode: {mode}")


def _danger_positive_weights(
    episodes: list[ProbeEpisode],
    *,
    maximum: float,
) -> torch.Tensor:
    danger = torch.cat(
        [~episode.labels.evaluation_safe_actions for episode in episodes],
        dim=0,
    )
    positives = danger.sum(dim=0).to(torch.float32)
    negatives = (~danger).sum(dim=0).to(torch.float32)
    return (negatives / positives.clamp_min(1.0)).clamp(1.0, maximum)


def _train(
    episodes: list[ProbeEpisode],
    *,
    mode: str,
    device: str,
    hidden_size: int,
    epochs: int,
    chunk_length: int,
    learning_rate: float,
    seed: int,
) -> tuple[_PhysicalDangerProbe, list[dict[str, float]]]:
    torch.manual_seed(seed)
    model = _PhysicalDangerProbe(
        _features(episodes[0], mode).shape[-1],
        hidden_size,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
    )
    positive_weights = _danger_positive_weights(
        episodes,
        maximum=24.0,
    ).to(device)
    rng = random.Random(seed)
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        order = list(episodes)
        rng.shuffle(order)
        loss_sum = 0.0
        chunks = 0
        model.train()
        for episode in order:
            hidden = None
            inputs = _features(episode, mode)
            for start in range(0, episode.labels.decisions, chunk_length):
                stop = min(start + chunk_length, episode.labels.decisions)
                logits, hidden = model(inputs[:, start:stop].to(device), hidden)
                hidden = hidden.detach()
                target = (~episode.labels.evaluation_safe_actions[start:stop]).to(
                    dtype=torch.float32,
                    device=device,
                ).unsqueeze(0)
                loss = F.binary_cross_entropy_with_logits(
                    logits,
                    target,
                    pos_weight=positive_weights,
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                loss_sum += float(loss.detach().cpu())
                chunks += 1
        item = {"epoch": float(epoch), "mean_chunk_loss": loss_sum / chunks}
        history.append(item)
        if epoch == 1 or epoch == epochs or epoch % 5 == 0:
            print(mode, item, flush=True)
    model.eval()
    return model, history


def _predict(
    model: _PhysicalDangerProbe,
    episodes: list[ProbeEpisode],
    *,
    mode: str,
    device: str,
    chunk_length: int,
) -> dict[int, torch.Tensor]:
    result: dict[int, torch.Tensor] = {}
    with torch.no_grad():
        for episode in episodes:
            values: list[torch.Tensor] = []
            hidden = None
            inputs = _features(episode, mode)
            for start in range(0, episode.labels.decisions, chunk_length):
                stop = min(start + chunk_length, episode.labels.decisions)
                logits, hidden = model(inputs[:, start:stop].to(device), hidden)
                hidden = hidden.detach()
                values.append(torch.sigmoid(logits[0]).cpu())
            result[episode.labels.seed] = torch.cat(values, dim=0)
    return result


def _auc(labels: list[torch.Tensor], scores: list[torch.Tensor]) -> float:
    target = torch.cat(labels).numpy().astype(np.int64, copy=False)
    value = torch.cat(scores).numpy()
    order = np.argsort(value, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(value) + 1, dtype=np.float64)
    sorted_values = value[order]
    start = 0
    while start < len(sorted_values):
        stop = start + 1
        while stop < len(sorted_values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = (start + stop + 1) / 2.0
        start = stop
    positive = int(target.sum())
    negative = len(target) - positive
    if not positive or not negative:
        return 0.0
    return float(
        (
            ranks[target == 1].sum()
            - positive * (positive + 1) / 2.0
        )
        / (positive * negative)
    )


def _metrics(
    episodes: list[ProbeEpisode],
    predictions: dict[int, torch.Tensor],
    *,
    parent_threshold: float,
    candidate_threshold: float,
) -> dict[str, float | int]:
    opportunities = 0
    requests = 0
    beneficial = 0
    unbeneficial = 0
    for episode in episodes:
        labels = episode.labels
        danger = predictions[labels.seed]
        parent_score = danger.gather(
            -1, labels.parent_actions.unsqueeze(-1),
        ).squeeze(-1)
        candidates = labels.safety_candidate_actions.clamp_min(0)
        candidate_score = danger.gather(
            -1, candidates.unsqueeze(-1),
        ).squeeze(-1)
        candidate_true_safe = labels.evaluation_safe_actions.gather(
            -1, candidates.unsqueeze(-1),
        ).squeeze(-1)
        opportunity = (
            labels.parent_evaluation_danger
            & labels.safety_candidate_valid
            & candidate_true_safe
            & (candidates != labels.parent_actions)
        )
        request = (
            labels.safety_candidate_valid
            & (candidates != labels.parent_actions)
            & (parent_score >= parent_threshold)
            & (candidate_score <= candidate_threshold)
        )
        opportunities += int(opportunity.sum())
        requests += int(request.sum())
        beneficial += int((request & opportunity).sum())
        unbeneficial += int((request & ~opportunity).sum())
    return {
        "opportunities": opportunities,
        "requests": requests,
        "beneficial": beneficial,
        "unbeneficial": unbeneficial,
        "recall": beneficial / max(opportunities, 1),
        "precision": beneficial / max(requests, 1),
    }


def _select_thresholds(
    training: list[ProbeEpisode],
    calibration: list[ProbeEpisode],
    predictions: dict[int, torch.Tensor],
) -> tuple[float, float, dict[str, Any]]:
    ranked: list[tuple[tuple[float, ...], float, float, dict[str, Any]]] = []
    for parent_threshold in np.linspace(0.1, 0.95, 18):
        for candidate_threshold in np.linspace(0.05, 0.9, 18):
            train = _metrics(
                training,
                predictions,
                parent_threshold=float(parent_threshold),
                candidate_threshold=float(candidate_threshold),
            )
            calibrate = _metrics(
                calibration,
                predictions,
                parent_threshold=float(parent_threshold),
                candidate_threshold=float(candidate_threshold),
            )
            admissible = int(
                train["unbeneficial"] == 0
                and calibrate["unbeneficial"] == 0
                and train["beneficial"] > 0
                and calibrate["beneficial"] > 0
            )
            score = (
                float(admissible),
                min(float(train["recall"]), float(calibrate["recall"])),
                float(train["beneficial"] + calibrate["beneficial"]),
                -float(train["requests"] + calibrate["requests"]),
            )
            ranked.append((score, float(parent_threshold), float(candidate_threshold), {
                "training": train,
                "calibration": calibrate,
            }))
    score, parent_threshold, candidate_threshold, metrics = max(
        ranked,
        key=lambda item: item[0],
    )
    metrics["admissible"] = bool(score[0])
    return parent_threshold, candidate_threshold, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--residual-adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--chunk-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=20260818)
    args = parser.parse_args()

    source_report = _read_json(args.source_report)
    parent, _metadata = load_checkpoint(args.parent, device=args.device)
    wrapper, _adapter_metadata = load_residual_adapter(
        parent,
        args.residual_adapter,
        parent_checkpoint=args.parent,
        device=args.device,
    )
    episodes: list[ProbeEpisode] = []
    for index, source in enumerate(source_report["sources"], start=1):
        dataset = Path(source["dataset"])
        labels = _load_episode(
            parent,
            wrapper.adapter,
            dataset,
            Path(source["report"]),
            dataset.with_suffix(".manifest.json"),
            parent_checkpoint_sha256=source_report["parent_checkpoint_sha256"],
            device=args.device,
            chunk_length=args.chunk_length,
            safe_regret=source_report["safe_regret"],
            minimum_parent_margin=source_report["minimum_parent_margin"],
            minimum_margin_gain=source_report["minimum_margin_gain"],
            predecessor_decisions=source_report["predecessor_decisions"],
        )
        visual = _visual_stream(
            parent,
            Demonstrations.load(dataset),
            device=args.device,
            chunk_length=args.chunk_length,
        )
        episodes.append(ProbeEpisode(labels, str(source["role"]), visual))
        print(
            f"loaded {index}/{len(source_report['sources'])}: "
            f"seed={labels.seed} role={source['role']}",
            flush=True,
        )

    _apply_existing_normalization(
        wrapper.adapter,
        [episode.labels for episode in episodes],
    )
    training = [episode for episode in episodes if episode.role == "training"]
    calibration = [episode for episode in episodes if episode.role == "calibration"]
    validation = [episode for episode in episodes if episode.role == "validation"]
    if not training or not calibration or not validation:
        raise ValueError("probe requires training, calibration, and validation episodes")
    fitting_visual = torch.cat([episode.visual[0] for episode in training], dim=0)
    visual_mean = fitting_visual.mean(dim=0)
    visual_scale = fitting_visual.std(dim=0).clamp_min(1e-4)
    for episode in episodes:
        episode.visual = (episode.visual - visual_mean) / visual_scale

    report: dict[str, Any] = {
        "run_kind": "residual_visual_physical_safety_probe",
        "source_report": str(args.source_report),
        "source_frame_storage": "legacy sources may be float16; probe is not deployment evidence",
        "training_seeds": [episode.labels.seed for episode in training],
        "calibration_seeds": [episode.labels.seed for episode in calibration],
        "validation_seeds": [episode.labels.seed for episode in validation],
        "config": vars(args) | {"output": str(args.output), "parent": str(args.parent), "residual_adapter": str(args.residual_adapter), "source_report": str(args.source_report)},
        "modes": {},
    }
    for mode in ("parent-state", "visual", "combined"):
        model, history = _train(
            training,
            mode=mode,
            device=args.device,
            hidden_size=args.hidden_size,
            epochs=args.epochs,
            chunk_length=args.chunk_length,
            learning_rate=args.learning_rate,
            seed=args.seed,
        )
        predictions = _predict(
            model,
            episodes,
            mode=mode,
            device=args.device,
            chunk_length=args.chunk_length,
        )
        auc_values: dict[str, float] = {}
        for name, split in (
            ("training", training),
            ("calibration", calibration),
            ("validation", validation),
        ):
            labels: list[torch.Tensor] = []
            scores: list[torch.Tensor] = []
            for episode in split:
                value = episode.labels
                labels.append(value.parent_evaluation_danger)
                scores.append(
                    predictions[value.seed].gather(
                        -1, value.parent_actions.unsqueeze(-1),
                    ).squeeze(-1)
                )
            auc_values[name] = _auc(labels, scores)
        parent_threshold, candidate_threshold, selected = _select_thresholds(
            training,
            calibration,
            predictions,
        )
        selected["validation"] = _metrics(
            validation,
            predictions,
            parent_threshold=parent_threshold,
            candidate_threshold=candidate_threshold,
        )
        report["modes"][mode] = {
            "input_size": _features(training[0], mode).shape[-1],
            "history": history,
            "parent_danger_auc": auc_values,
            "thresholds": {
                "parent_danger_minimum": parent_threshold,
                "candidate_danger_maximum": candidate_threshold,
            },
            "physical_override": selected,
        }
        print(mode, json.dumps(report["modes"][mode], sort_keys=True), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
