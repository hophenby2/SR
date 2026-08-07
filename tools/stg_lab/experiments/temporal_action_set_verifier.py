"""Temporal, set-valued verification of a frozen selector candidate."""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

try:
    from .temporal_candidate_verifier import hard_negative_pairwise_loss
except ImportError:
    from temporal_candidate_verifier import hard_negative_pairwise_loss


EARLY_LEAD_MINIMUM = 4
EARLY_LEAD_MAXIMUM = 10

SUPERVISION_SEMANTICS: Mapping[str, Any] = {
    "schema_version": 3,
    "dense_action_set_rows": (
        "fit_only_gate_valid_positive_anticipatory_lead_4_through_10"
    ),
    "selected_candidate_rows": (
        "fit_only_all_decisions_where_frozen_candidate_differs_from_parent"
    ),
    "selected_candidate_target": (
        "positive_correction_required_and_preferred_equivalent_and_"
        "evaluation_safe_at_frozen_candidate"
    ),
    "no_correction_changed_candidate": "negative",
    "gate_invalid_changed_candidate": "negative",
    "no_correction_parent_candidate": "excluded_from_selected_loss",
}


@dataclass(frozen=True, slots=True)
class TemporalActionSetVerifierConfig:
    latent_size: int = 128
    action_count: int = 18
    hidden_size: int = 96
    bottleneck_size: int = 48
    ensemble_size: int = 3

    def __post_init__(self) -> None:
        for name in (
            "latent_size",
            "action_count",
            "hidden_size",
            "bottleneck_size",
            "ensemble_size",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.action_count != 18:
            raise ValueError("the action-set verifier requires 18 actions")

    @property
    def input_size(self) -> int:
        # Policy latent, selector probabilities, onset pair, mean/max physical
        # danger for all actions, and parent-action token.
        return self.latent_size + self.action_count * 4 + 2


@dataclass(frozen=True, slots=True)
class TemporalActionSetTrainingConfig:
    epochs: int = 24
    learning_rate: float = 1e-3
    weight_decay: float = 1e-3
    dense_bce_weight: float = 1.0
    selected_bce_weight: float = 1.0
    selected_pairwise_weight: float = 2.0
    action_set_rank_weight: float = 1.0
    pairwise_margin: float = 0.25
    pairwise_temperature: float = 0.5
    hard_negative_fraction: float = 0.25
    gradient_clip_max_norm: float = 5.0

    def __post_init__(self) -> None:
        if isinstance(self.epochs, bool) or not isinstance(self.epochs, int):
            raise TypeError("epochs must be an integer")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        positive = (
            self.learning_rate,
            self.dense_bce_weight,
            self.selected_bce_weight,
            self.selected_pairwise_weight,
            self.action_set_rank_weight,
            self.pairwise_temperature,
            self.hard_negative_fraction,
            self.gradient_clip_max_norm,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("positive training controls must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if not math.isfinite(self.pairwise_margin):
            raise ValueError("pairwise margin must be finite")
        if self.hard_negative_fraction > 1.0:
            raise ValueError("hard-negative fraction must not exceed one")


@dataclass(frozen=True, slots=True)
class TemporalActionSetEpisode:
    seed: int
    inputs: Tensor
    label_mask: Tensor
    action_labels: Tensor
    selected_candidates: Tensor
    selected_label_mask: Tensor
    selected_labels: Tensor

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("episode seed must be an integer")
        if self.inputs.ndim != 3 or self.inputs.shape[0] != 1:
            raise ValueError("action-set inputs must have shape [1, decisions, features]")
        decisions = self.inputs.shape[1]
        if self.label_mask.shape != (decisions,) or self.label_mask.dtype != torch.bool:
            raise ValueError("action-set label mask does not align")
        if self.action_labels.shape != (decisions, 18):
            raise ValueError("action-set labels do not align")
        if self.action_labels.dtype != torch.bool:
            raise ValueError("action-set labels must be Boolean")
        if self.selected_candidates.shape != (decisions,):
            raise ValueError("selected candidates do not align")
        if self.selected_candidates.dtype != torch.int64:
            raise ValueError("selected candidates must be int64")
        if bool(
            (
                (self.selected_candidates < 0)
                | (self.selected_candidates >= 18)
            ).any()
        ):
            raise ValueError("selected candidate is outside the vocabulary")
        if (
            self.selected_label_mask.shape != (decisions,)
            or self.selected_label_mask.dtype != torch.bool
        ):
            raise ValueError("selected-candidate label mask does not align")
        if self.selected_labels.shape != (decisions,):
            raise ValueError("selected-candidate labels do not align")
        if self.selected_labels.dtype != torch.bool:
            raise ValueError("selected-candidate labels must be Boolean")
        if bool((self.selected_labels & ~self.selected_label_mask).any()):
            raise ValueError("positive selected labels must be on supported rows")
        if not torch.is_floating_point(self.inputs):
            raise ValueError("action-set inputs must be floating point")
        if not bool(torch.isfinite(self.inputs).all()):
            raise ValueError("action-set inputs must be finite")


class _TemporalActionSetMember(nn.Module):
    def __init__(self, config: TemporalActionSetVerifierConfig) -> None:
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(config.input_size, config.hidden_size),
            nn.SiLU(),
            nn.LayerNorm(config.hidden_size),
        )
        self.recurrent = nn.GRU(
            config.hidden_size,
            config.hidden_size,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(config.hidden_size, config.bottleneck_size),
            nn.SiLU(),
            nn.Linear(config.bottleneck_size, config.action_count),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        encoded = self.input_projection(inputs)
        recurrent, _hidden = self.recurrent(encoded)
        return self.head(recurrent)


class TemporalActionSetVerifier(nn.Module):
    def __init__(self, config: TemporalActionSetVerifierConfig) -> None:
        super().__init__()
        self.config = config
        self.members = nn.ModuleList([
            _TemporalActionSetMember(config)
            for _ in range(config.ensemble_size)
        ])

    def forward_logits(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 3 or inputs.shape[0] != 1:
            raise ValueError("action-set inputs must have shape [1, decisions, features]")
        if inputs.shape[-1] != self.config.input_size:
            raise ValueError("action-set input width differs from configuration")
        if not torch.is_floating_point(inputs):
            raise ValueError("action-set inputs must be floating point")
        return torch.stack([member(inputs)[0] for member in self.members], dim=0)


def build_temporal_action_set_inputs(
    *,
    policy_latents: Tensor,
    mean_action_probabilities: Tensor,
    mean_gate: Tensor,
    minimum_gate: Tensor,
    physical_danger_probabilities: Tensor,
    parent_actions: Tensor,
    config: TemporalActionSetVerifierConfig,
) -> Tensor:
    """Construct inputs from frozen live-policy tensors, with no teacher fields."""

    if policy_latents.ndim != 3:
        raise ValueError("policy latents must have shape [members, decisions, latent]")
    members, decisions, latent_size = policy_latents.shape
    if members <= 0 or latent_size != config.latent_size:
        raise ValueError("policy latent inventory differs from configuration")
    if mean_action_probabilities.shape != (decisions, config.action_count):
        raise ValueError("selector probabilities do not align")
    if mean_gate.shape != (decisions,) or minimum_gate.shape != (decisions,):
        raise ValueError("onset probabilities do not align")
    if physical_danger_probabilities.ndim != 3 or (
        physical_danger_probabilities.shape[1:]
        != (decisions, config.action_count)
    ):
        raise ValueError("physical probabilities do not align")
    if parent_actions.shape != (decisions,) or parent_actions.dtype != torch.int64:
        raise ValueError("parent actions do not align")
    if bool(((parent_actions < 0) | (parent_actions >= config.action_count)).any()):
        raise ValueError("parent action is outside the vocabulary")
    floating = (
        policy_latents,
        mean_action_probabilities,
        mean_gate,
        minimum_gate,
        physical_danger_probabilities,
    )
    dtype = policy_latents.dtype
    device = policy_latents.device
    if not all(torch.is_floating_point(value) for value in floating):
        raise ValueError("action-set inputs must be floating point")
    if any(value.dtype != dtype or value.device != device for value in floating):
        raise ValueError("action-set tensors must share dtype and device")
    if parent_actions.device != device:
        raise ValueError("parent actions must share the input device")
    if not all(bool(torch.isfinite(value).all()) for value in floating):
        raise ValueError("action-set inputs must be finite")
    if bool(
        (mean_action_probabilities < 0.0).any()
        or (mean_action_probabilities > 1.0).any()
        or (physical_danger_probabilities < 0.0).any()
        or (physical_danger_probabilities > 1.0).any()
    ):
        raise ValueError("probabilities must be in [0, 1]")
    physical_mean = physical_danger_probabilities.mean(dim=0)
    physical_maximum = physical_danger_probabilities.amax(dim=0)
    parent_tokens = F.one_hot(
        parent_actions,
        num_classes=config.action_count,
    ).to(dtype=dtype)
    result = torch.cat(
        (
            policy_latents.mean(dim=0),
            mean_action_probabilities,
            mean_gate.unsqueeze(-1),
            minimum_gate.unsqueeze(-1),
            physical_mean,
            physical_maximum,
            parent_tokens,
        ),
        dim=-1,
    ).unsqueeze(0)
    if result.shape != (1, decisions, config.input_size):
        raise RuntimeError("action-set input construction is misaligned")
    return result


def temporal_action_set_targets(
    episode: Any,
    selected_candidates: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    decisions = int(episode.parent_actions.numel())
    if selected_candidates.shape != (decisions,) or (
        selected_candidates.dtype != torch.int64
    ):
        raise ValueError("selected candidates do not align")
    if bool(((selected_candidates < 0) | (selected_candidates >= 18)).any()):
        raise ValueError("selected candidate is outside the vocabulary")
    parent_actions = episode.parent_actions
    if parent_actions.shape != (decisions,) or parent_actions.dtype != torch.int64:
        raise ValueError("parent actions do not align")
    if bool(((parent_actions < 0) | (parent_actions >= 18)).any()):
        raise ValueError("parent action is outside the vocabulary")
    labels = episode.preferred_equivalent_actions
    required = episode.preferred_correction_required
    if labels.shape != (decisions, 18) or labels.dtype != torch.bool:
        raise ValueError("preferred equivalent actions do not align")
    if required.shape != (decisions,) or required.dtype != torch.bool:
        raise ValueError("correction-required rows do not align")
    has_positive = labels.any(dim=-1)
    if bool((has_positive & ~required).any()):
        raise ValueError("no-correction rows cannot contain equivalent actions")
    if bool((required & ~has_positive).any()):
        raise ValueError("correction-required rows need an equivalent action")
    parent_positive = labels.gather(
        -1, parent_actions.unsqueeze(-1)
    ).squeeze(-1)
    if bool(parent_positive.any()):
        raise ValueError("parent actions cannot be equivalent corrections")
    dense_mask = (
        episode.gate_valid
        & (episode.gate_targets > 0.0)
        & episode.anticipatory
        & (episode.anticipatory_lead_decisions >= EARLY_LEAD_MINIMUM)
        & (episode.anticipatory_lead_decisions <= EARLY_LEAD_MAXIMUM)
    )
    if dense_mask.shape != (decisions,) or dense_mask.dtype != torch.bool:
        raise ValueError("early action-set label mask does not align")
    evaluation_safe = episode.evaluation_safe_actions
    if evaluation_safe.shape != (decisions, 18):
        raise ValueError("evaluation-safe actions do not align")
    if evaluation_safe.dtype != torch.bool:
        raise ValueError("evaluation-safe actions must be Boolean")
    if bool((labels & ~evaluation_safe).any()):
        raise ValueError("preferred equivalent actions must be evaluation-safe")
    positive = episode.gate_valid & (episode.gate_targets > 0.0)
    if positive.shape != (decisions,) or positive.dtype != torch.bool:
        raise ValueError("positive gate rows do not align")
    selected_equivalent = labels.gather(
        -1, selected_candidates.unsqueeze(-1)
    ).squeeze(-1)
    selected_evaluation_safe = evaluation_safe.gather(
        -1, selected_candidates.unsqueeze(-1)
    ).squeeze(-1)
    selected_label_mask = selected_candidates != parent_actions
    selected_labels = (
        positive
        & required
        & selected_equivalent
        & selected_evaluation_safe
    )
    if bool((selected_labels & ~selected_label_mask).any()):
        raise ValueError("beneficial selected candidate is outside runtime support")
    return (
        dense_mask,
        labels.clone(),
        selected_label_mask,
        selected_labels,
    )


def _action_set_rank_loss(logits: Tensor, labels: Tensor) -> Tensor:
    positive_rows = labels.any(dim=-1)
    if not bool(positive_rows.any()):
        raise ValueError("action-set rank loss requires positive rows")
    row_logits = logits[positive_rows]
    row_labels = labels[positive_rows]
    positive = row_logits.masked_fill(~row_labels, -torch.inf).amax(dim=-1)
    negative = row_logits.masked_fill(row_labels, -torch.inf).amax(dim=-1)
    return F.softplus(negative - positive + 0.25).mean()


def _state_snapshot(module: nn.Module) -> dict[str, Tensor]:
    return {
        name: value.detach().clone()
        for name, value in module.state_dict().items()
    }


def train_temporal_action_set_verifier(
    verifier: TemporalActionSetVerifier,
    episodes: Sequence[TemporalActionSetEpisode],
    *,
    seed: int,
    config: TemporalActionSetTrainingConfig | None = None,
) -> list[dict[str, float]]:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("training seed must be an integer")
    if config is None:
        config = TemporalActionSetTrainingConfig()
    if not episodes or len({episode.seed for episode in episodes}) != len(episodes):
        raise ValueError("action-set fit episodes must be nonempty and unique")
    if any(parameter.device.type != "cpu" for parameter in verifier.parameters()):
        raise ValueError("formal action-set training requires CPU parameters")
    for episode in episodes:
        if episode.inputs.device.type != "cpu" or episode.inputs.dtype != torch.float32:
            raise ValueError("formal action-set inputs must be CPU float32")
        if episode.inputs.shape[-1] != verifier.config.input_size:
            raise ValueError("action-set episode width differs from verifier")

    state_before = _state_snapshot(verifier)
    modes = [(module, module.training) for module in verifier.modules()]
    runtime = [
        (
            parameter,
            parameter.requires_grad,
            None if parameter.grad is None else parameter.grad.detach().clone(),
        )
        for parameter in verifier.parameters()
    ]
    rng_state = torch.random.get_rng_state()

    def restore(*, rollback: bool) -> None:
        if rollback:
            verifier.load_state_dict(state_before, strict=True)
        for parameter, requires_grad, gradient in runtime:
            parameter.requires_grad_(requires_grad)
            parameter.grad = None if gradient is None else gradient.clone()
        for module, training in modes:
            module.training = training
        torch.random.set_rng_state(rng_state)

    try:
        torch.manual_seed(seed)
        verifier.train()
        for parameter in verifier.parameters():
            parameter.requires_grad_(True)
            parameter.grad = None
        optimizer = torch.optim.AdamW(
            verifier.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        order_rng = random.Random(seed)
        history: list[dict[str, float]] = []
        for epoch in range(1, config.epochs + 1):
            order = list(range(len(episodes)))
            order_rng.shuffle(order)
            dense_logits_by_member: list[list[Tensor]] = [
                [] for _ in range(verifier.config.ensemble_size)
            ]
            selected_logits_by_member: list[list[Tensor]] = [
                [] for _ in range(verifier.config.ensemble_size)
            ]
            dense_labels_parts: list[Tensor] = []
            selected_labels_parts: list[Tensor] = []
            for index in order:
                episode = episodes[index]
                logits = verifier.forward_logits(episode.inputs)
                if not bool(torch.isfinite(logits).all()):
                    raise ValueError("action-set verifier produced nonfinite logits")
                for member_index in range(verifier.config.ensemble_size):
                    member_logits = logits[member_index]
                    dense_logits_by_member[member_index].append(
                        logits[member_index, episode.label_mask]
                    )
                    selected_for_episode = member_logits.gather(
                        -1,
                        episode.selected_candidates.unsqueeze(-1),
                    ).squeeze(-1)
                    selected_logits_by_member[member_index].append(
                        selected_for_episode[episode.selected_label_mask]
                    )
                dense_labels_parts.append(
                    episode.action_labels[episode.label_mask]
                )
                selected_labels_parts.append(
                    episode.selected_labels[episode.selected_label_mask]
                )
            dense_labels = torch.cat(dense_labels_parts)
            selected_labels = torch.cat(selected_labels_parts)
            if dense_labels.numel() == 0 or not bool(dense_labels.any()):
                raise ValueError("action-set fit has no positive labelled cells")
            if selected_labels.numel() == 0:
                raise ValueError("selected-candidate runtime support is empty")
            if not bool(selected_labels.any()) or not bool((~selected_labels).any()):
                raise ValueError("selected-candidate fit needs both classes")
            member_losses = []
            dense_losses = []
            selected_losses = []
            pairwise_losses = []
            rank_losses = []
            for dense_parts, selected_parts in zip(
                dense_logits_by_member,
                selected_logits_by_member,
                strict=True,
            ):
                dense_logits = torch.cat(dense_parts)
                selected_logits = torch.cat(selected_parts)
                dense = F.binary_cross_entropy_with_logits(
                    dense_logits,
                    dense_labels.to(dense_logits.dtype),
                )
                selected = F.binary_cross_entropy_with_logits(
                    selected_logits,
                    selected_labels.to(selected_logits.dtype),
                )
                pairwise = hard_negative_pairwise_loss(
                    selected_logits,
                    selected_labels,
                    margin=config.pairwise_margin,
                    temperature=config.pairwise_temperature,
                    hard_negative_fraction=config.hard_negative_fraction,
                )
                rank = _action_set_rank_loss(dense_logits, dense_labels)
                member_losses.append(
                    config.dense_bce_weight * dense
                    + config.selected_bce_weight * selected
                    + config.selected_pairwise_weight * pairwise
                    + config.action_set_rank_weight * rank
                )
                dense_losses.append(dense)
                selected_losses.append(selected)
                pairwise_losses.append(pairwise)
                rank_losses.append(rank)
            loss = torch.stack(member_losses).mean()
            if not bool(torch.isfinite(loss)):
                raise ValueError("action-set verifier loss is nonfinite")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradients = [parameter.grad for parameter in verifier.parameters()]
            if any(gradient is None for gradient in gradients) or not all(
                bool(torch.isfinite(gradient).all())
                for gradient in gradients
                if gradient is not None
            ):
                raise ValueError("action-set verifier gradient is missing or nonfinite")
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                tuple(verifier.parameters()), config.gradient_clip_max_norm
            )
            if not bool(torch.isfinite(gradient_norm)):
                raise ValueError("action-set verifier gradient norm is nonfinite")
            optimizer.step()
            if not all(
                bool(torch.isfinite(parameter).all())
                for parameter in verifier.parameters()
            ):
                raise ValueError("action-set verifier parameter became nonfinite")
            history.append({
                "epoch": float(epoch),
                "loss": float(loss.detach()),
                "dense_bce": float(torch.stack(dense_losses).mean().detach()),
                "selected_bce": float(
                    torch.stack(selected_losses).mean().detach()
                ),
                "selected_pairwise": float(
                    torch.stack(pairwise_losses).mean().detach()
                ),
                "action_set_rank": float(
                    torch.stack(rank_losses).mean().detach()
                ),
                "labelled_rows": float(dense_labels.shape[0]),
                "dense_labelled_rows": float(dense_labels.shape[0]),
                "selected_labelled_rows": float(selected_labels.shape[0]),
                "positive_action_cells": float(dense_labels.sum()),
                "selected_positive_rows": float(selected_labels.sum()),
                "selected_negative_rows": float((~selected_labels).sum()),
                "gradient_norm_before_clip": float(gradient_norm.detach()),
            })
    except Exception:
        restore(rollback=True)
        raise
    restore(rollback=False)
    return history


def predict_temporal_action_set_verifier(
    verifier: TemporalActionSetVerifier,
    inputs: Tensor,
    candidates: Tensor,
) -> dict[str, Tensor]:
    if inputs.device.type != "cpu" or inputs.dtype != torch.float32:
        raise ValueError("formal action-set prediction requires CPU float32")
    if candidates.shape != (inputs.shape[1],) or candidates.dtype != torch.int64:
        raise ValueError("prediction candidates do not align")
    modes = [(module, module.training) for module in verifier.modules()]
    try:
        verifier.eval()
        with torch.no_grad():
            logits = verifier.forward_logits(inputs)
            probabilities = torch.sigmoid(logits)
        finite = torch.isfinite(logits) & torch.isfinite(probabilities)
        all_action_finite = finite.all(dim=0)
        mean_membership = probabilities.mean(dim=0)
        candidate_indices = candidates.view(1, -1, 1).expand(
            verifier.config.ensemble_size, -1, 1
        )
        selected_member = probabilities.gather(
            -1, candidate_indices
        ).squeeze(-1)
        selected_finite = finite.gather(-1, candidate_indices).squeeze(-1)
        all_selected_finite = selected_finite.all(dim=0)
        confidence = selected_member.mean(dim=0)
        confidence = torch.where(
            all_selected_finite,
            confidence,
            torch.full_like(confidence, float("nan")),
        )
        return {
            "member_logits": logits,
            "member_probabilities": probabilities,
            "all_action_cells_finite": all_action_finite,
            "mean_membership_probabilities": mean_membership,
            "selected_member_probabilities": selected_member,
            "all_selected_members_finite": all_selected_finite,
            "confidence": confidence,
        }
    finally:
        for module, training in modes:
            module.training = training


def action_set_verifier_state_sha256(
    verifier: TemporalActionSetVerifier,
) -> str:
    digest = hashlib.sha256(repr(asdict(verifier.config)).encode("utf-8"))
    for name, value in sorted(verifier.state_dict().items()):
        contiguous = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(repr(tuple(contiguous.shape)).encode("ascii"))
        digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


INFERENCE_INPUT_SEMANTICS: Mapping[str, Any] = {
    "schema_version": 1,
    "source": "frozen_delayed_vision_policy_outputs_only",
    "features": [
        "mean_frozen_shared_and_action_recurrent_latent",
        "frozen_mean_selector_probabilities",
        "frozen_mean_and_minimum_onset_probability",
        "frozen_per_action_physical_danger_mean_and_maximum",
        "frozen_parent_action_one_hot",
    ],
    "candidate_role": "gather_only_after_all_action_membership_prediction",
    "forbidden": [
        "preferred_equivalent_actions",
        "evaluation_safe_actions",
        "teacher_action",
        "teacher_regret",
        "lua_timer",
        "rng_state",
        "recorded_route",
    ],
}
