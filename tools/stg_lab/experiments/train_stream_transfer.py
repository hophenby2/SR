"""Train a bounded latest-frame streaming transfer baseline."""

from __future__ import annotations

from dataclasses import asdict
import argparse
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn import functional as F

from stg_lab.policy import PolicyConfig
from stg_lab.rollout import teacher_action_agreement
from stg_lab.training import Demonstrations, load_checkpoint, to_recurrent_sequences


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def subset(demos: Demonstrations, mask: np.ndarray) -> Demonstrations:
    return Demonstrations(
        global_frames=demos.global_frames[mask],
        local_frames=demos.local_frames[mask],
        actions=demos.actions[mask],
        risks=demos.risks[mask],
        memory=None if demos.memory is None else demos.memory[mask],
        episode_ids=None if demos.episode_ids is None else demos.episode_ids[mask],
        supervision_mask=(
            None if demos.supervision_mask is None else demos.supervision_mask[mask]
        ),
    )


def encode(model, demos: Demonstrations, device: str, batch_size: int = 2):
    global_values = []
    local_values = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(demos.actions), batch_size):
            end = min(start + batch_size, len(demos.actions))
            global_frames = torch.from_numpy(demos.global_frames[start:end]).float().to(device)
            local_frames = torch.from_numpy(demos.local_frames[start:end]).float().to(device)
            batch, steps = global_frames.shape[:2]
            global_values.append(
                model.global_encoder(global_frames.flatten(0, 1)).reshape(batch, steps, -1).cpu()
            )
            local_values.append(
                model.local_encoder(local_frames.flatten(0, 1)).reshape(batch, steps, -1).cpu()
            )
    return torch.cat(global_values), torch.cat(local_values)


def recurrent_agreement(model, demos: Demonstrations, device: str) -> float:
    global_features, local_features = encode(model, demos, device)
    memory = torch.from_numpy(demos.memory).float() if demos.memory is not None else torch.zeros(
        (*demos.actions.shape, model.config.memory_size), dtype=torch.float32,
    )
    actions = torch.from_numpy(demos.actions).long()
    mask = torch.from_numpy(demos.supervision_mask).bool()
    correct = total = 0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(actions), 8):
            end = min(start + 8, len(actions))
            features = torch.cat((
                global_features[start:end].to(device),
                local_features[start:end].to(device),
                memory[start:end].to(device),
            ), dim=-1)
            recurrent, _hidden = model.recurrent(features)
            logits = model.policy_head(recurrent)
            selected = mask[start:end].to(device)
            labels = actions[start:end].to(device)
            correct += int((logits.argmax(-1)[selected] == labels[selected]).sum().item())
            total += int(selected.sum().item())
    return correct / total if total else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, default=Path("artifacts/policy_canonical_best.pt"))
    parser.add_argument("--train", type=Path, default=Path("artifacts/canonical_train_dagger_v1.npz"))
    parser.add_argument("--heldout", type=Path, default=Path("artifacts/canonical_heldout_merged.npz"))
    parser.add_argument("--output-prefix", type=Path, default=Path("artifacts/policy_stream_transfer"))
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    seed = 20260730
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    direct_train = Demonstrations.load(args.train)
    direct_heldout = Demonstrations.load(args.heldout)
    train = to_recurrent_sequences(direct_train, sequence_length=64)
    heldout = to_recurrent_sequences(direct_heldout, sequence_length=64)
    train_path = args.report.parent / "canonical_train_dagger_v1_recurrent64.npz"
    heldout_path = args.report.parent / "canonical_heldout_recurrent64.npz"
    train.save(train_path)
    heldout.save(heldout_path)

    template, parent_checkpoint = load_checkpoint(args.parent, device=args.device)
    parent_config = dict(parent_checkpoint["policy_config"])
    stream_config = PolicyConfig(**{**parent_config, "inference_mode": "stream"})
    template.config = stream_config
    for parameter in template.global_encoder.parameters():
        parameter.requires_grad = False
    for parameter in template.local_encoder.parameters():
        parameter.requires_grad = False
    for parameter in template.risk_head.parameters():
        parameter.requires_grad = False

    global_train, local_train = encode(template, train, args.device)
    train_memory = torch.from_numpy(train.memory).float()
    train_actions = torch.from_numpy(train.actions).long()
    train_mask = torch.from_numpy(train.supervision_mask).bool()
    candidate_reports = []

    for learning_rate in (1e-5, 5e-5):
        model, _ = load_checkpoint(args.parent, device=args.device)
        model.config = stream_config
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.recurrent.parameters():
            parameter.requires_grad = True
        for parameter in model.policy_head.parameters():
            parameter.requires_grad = True
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=learning_rate,
            weight_decay=1e-5,
        )
        generator = torch.Generator().manual_seed(seed)
        history = []
        for epoch in range(1, 11):
            model.train()
            order = torch.randperm(len(train_actions), generator=generator)
            loss_sum = 0.0
            labels_seen = 0
            for start in range(0, len(order), 8):
                indices = order[start:start + 8]
                features = torch.cat((
                    global_train[indices].to(args.device),
                    local_train[indices].to(args.device),
                    train_memory[indices].to(args.device),
                ), dim=-1)
                actions = train_actions[indices].to(args.device)
                mask = train_mask[indices].to(args.device)
                optimizer.zero_grad(set_to_none=True)
                recurrent, _hidden = model.recurrent(features)
                logits = model.policy_head(recurrent)
                loss = F.cross_entropy(logits[mask], actions[mask])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad], 5.0,
                )
                optimizer.step()
                count = int(mask.sum().item())
                loss_sum += float(loss.detach()) * count
                labels_seen += count
            history.append({"epoch": epoch, "train_loss": loss_sum / labels_seen})
            if epoch not in (5, 10):
                continue
            suffix = f"_lr{learning_rate:.0e}_e{epoch}.pt".replace("-", "m")
            output = Path(str(args.output_prefix) + suffix)
            metadata = {
                "parent_checkpoint": str(args.parent),
                "parent_checkpoint_sha256": sha256(args.parent),
                "dataset": str(train_path),
                "dataset_sha256": sha256(train_path),
                "source_dataset": str(args.train),
                "source_dataset_sha256": sha256(args.train),
                "sequence_length": 64,
                "sequence_semantics": "latest visible frame per decision; development baseline",
                "trainable_scope": "recurrent+policy_head",
                "encoder_frozen": True,
                "risk_head_frozen": True,
                "learning_rate": learning_rate,
                "cumulative_epochs": epoch,
                "seed": seed,
            }
            torch.save({
                "version": 1,
                "policy_config": asdict(stream_config),
                "state_dict": model.state_dict(),
                "history": history.copy(),
                "fine_tune_metadata": metadata,
            }, output)
            model.eval()
            direct_b3 = subset(direct_heldout, np.isin(direct_heldout.episode_ids, (0, 1)))
            direct_b4 = subset(direct_heldout, np.isin(direct_heldout.episode_ids, (2, 3)))
            recurrent_b3 = subset(heldout, np.isin(heldout.episode_ids, (0, 1)))
            recurrent_b4 = subset(heldout, np.isin(heldout.episode_ids, (2, 3)))
            candidate_reports.append({
                "checkpoint": str(output),
                "checkpoint_sha256": sha256(output),
                "fine_tune_metadata": metadata,
                "final_training": history[-1],
                "direct_heldout": {
                    "stage5_boss3": teacher_action_agreement(model, direct_b3, device=args.device),
                    "stage5_boss4": teacher_action_agreement(model, direct_b4, device=args.device),
                },
                "recurrent_heldout": {
                    "stage5_boss3": recurrent_agreement(model, recurrent_b3, args.device),
                    "stage5_boss4": recurrent_agreement(model, recurrent_b4, args.device),
                },
            })

    report = {
        "run_kind": "latest_only_stream_transfer_baseline",
        "acceptance_claim": False,
        "known_limitation": "decision_interval frames between latest observations are not consumed",
        "parent_checkpoint": str(args.parent),
        "parent_checkpoint_sha256": sha256(args.parent),
        "train_source": str(args.train),
        "train_source_sha256": sha256(args.train),
        "train_sequences": str(train_path),
        "train_sequences_sha256": sha256(train_path),
        "train_shape": list(train.global_frames.shape),
        "heldout_source": str(args.heldout),
        "heldout_source_sha256": sha256(args.heldout),
        "heldout_sequences": str(heldout_path),
        "heldout_sequences_sha256": sha256(heldout_path),
        "heldout_shape": list(heldout.global_frames.shape),
        "candidates": candidate_reports,
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
