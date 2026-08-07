"""Temporal verification for a candidate selected by a frozen policy.

The verifier cannot choose an action.  It receives only tensors available from
the frozen delayed-vision policy and estimates whether that policy's selected
candidate belongs to the independently certified equivalent-action set.
Teacher evidence is used only by :func:`selected_candidate_targets` and is
never accepted by the inference-input builder.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
import random
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


EARLY_LEAD_MINIMUM = 4
EARLY_LEAD_MAXIMUM = 10


@dataclass(frozen=True, slots=True)
class TemporalCandidateVerifierConfig:
    latent_size: int = 64
    action_count: int = 18
    hidden_size: int = 64
    bottleneck_size: int = 32
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
            raise ValueError("the temporal verifier requires the 18-action vocabulary")

    @property
    def input_size(self) -> int:
        # Mean frozen action latent, candidate one-hot, selector distribution,
        # two onset scores, four physical summaries, and candidate-changed bit.
        return self.latent_size + self.action_count * 2 + 7


@dataclass(frozen=True, slots=True)
class TemporalVerifierTrainingConfig:
    epochs: int = 12
    learning_rate: float = 3e-4
    weight_decay: float = 1e-3
    pairwise_loss_weight: float = 1.0
    pairwise_margin: float = 0.25
    pairwise_temperature: float = 0.5
    hard_negative_fraction: float = 0.25
    gradient_clip_max_norm: float = 5.0

    def __post_init__(self) -> None:
        if isinstance(self.epochs, bool) or not isinstance(self.epochs, int):
            raise ValueError("epochs must be an integer")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        positive = (
            self.learning_rate,
            self.pairwise_loss_weight,
            self.pairwise_temperature,
            self.hard_negative_fraction,
            self.gradient_clip_max_norm,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("positive training controls must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if not math.isfinite(self.pairwise_margin):
            raise ValueError("pairwise_margin must be finite")
        if self.hard_negative_fraction > 1.0:
            raise ValueError("hard_negative_fraction must not exceed one")


@dataclass(frozen=True, slots=True)
class TemporalVerifierEpisode:
    seed: int
    inputs: Tensor
    label_mask: Tensor
    labels: Tensor

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("episode seed must be an integer")
        if self.inputs.ndim != 3 or self.inputs.shape[0] != 1:
            raise ValueError("verifier inputs must have shape [1, decisions, features]")
        decisions = self.inputs.shape[1]
        if self.label_mask.shape != (decisions,) or self.label_mask.dtype != torch.bool:
            raise ValueError("verifier label mask does not align")
        if self.labels.shape != (decisions,) or self.labels.dtype != torch.bool:
            raise ValueError("verifier labels do not align")
        if not torch.is_floating_point(self.inputs):
            raise ValueError("verifier inputs must be floating point")
        if not bool(torch.isfinite(self.inputs).all()):
            raise ValueError("verifier inputs must be finite")


class _TemporalCandidateVerifierMember(nn.Module):
    def __init__(self, config: TemporalCandidateVerifierConfig) -> None:
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
            nn.Linear(config.bottleneck_size, 1),
        )

    def forward(
        self,
        inputs: Tensor,
        hidden: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        encoded = self.input_projection(inputs)
        recurrent, hidden = self.recurrent(encoded, hidden)
        return self.head(recurrent).squeeze(-1), hidden


class TemporalCandidateVerifier(nn.Module):
    """Three independent temporal correctness estimates for one frozen candidate."""

    def __init__(self, config: TemporalCandidateVerifierConfig) -> None:
        super().__init__()
        self.config = config
        self.members = nn.ModuleList([
            _TemporalCandidateVerifierMember(config)
            for _ in range(config.ensemble_size)
        ])

    def forward_logits(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 3 or inputs.shape[0] != 1:
            raise ValueError("verifier inputs must have shape [1, decisions, features]")
        if inputs.shape[-1] != self.config.input_size:
            raise ValueError("verifier input width differs from its configuration")
        if not torch.is_floating_point(inputs):
            raise ValueError("verifier inputs must be floating point")
        return torch.stack(
            [member(inputs)[0][0] for member in self.members],
            dim=0,
        )


def _require_vector(
    value: Tensor,
    *,
    decisions: int,
    name: str,
    dtype: torch.dtype | None = None,
) -> None:
    if value.shape != (decisions,):
        raise ValueError(f"{name} must have shape [decisions]")
    if dtype is not None and value.dtype != dtype:
        raise ValueError(f"{name} has the wrong dtype")


def build_temporal_verifier_inputs(
    *,
    action_latents: Tensor,
    candidates: Tensor,
    mean_action_probabilities: Tensor,
    mean_gate: Tensor,
    minimum_gate: Tensor,
    physical_danger_probabilities: Tensor,
    parent_actions: Tensor,
    config: TemporalCandidateVerifierConfig,
) -> Tensor:
    """Build inference features exclusively from frozen live-policy outputs."""

    if action_latents.ndim != 3:
        raise ValueError("action latents must have shape [members, decisions, latent]")
    members, decisions, latent_size = action_latents.shape
    if members <= 0 or latent_size != config.latent_size:
        raise ValueError("action latent inventory differs from verifier configuration")
    _require_vector(candidates, decisions=decisions, name="candidates", dtype=torch.int64)
    _require_vector(
        parent_actions,
        decisions=decisions,
        name="parent_actions",
        dtype=torch.int64,
    )
    _require_vector(mean_gate, decisions=decisions, name="mean_gate")
    _require_vector(minimum_gate, decisions=decisions, name="minimum_gate")
    if mean_action_probabilities.shape != (decisions, config.action_count):
        raise ValueError("mean action probabilities do not align")
    if physical_danger_probabilities.ndim != 3 or (
        physical_danger_probabilities.shape[1:]
        != (decisions, config.action_count)
    ):
        raise ValueError("physical danger probabilities do not align")
    tensors = (
        action_latents,
        mean_action_probabilities,
        mean_gate,
        minimum_gate,
        physical_danger_probabilities,
    )
    if not all(torch.is_floating_point(value) for value in tensors):
        raise ValueError("all frozen verifier features must be floating point")
    if not all(bool(torch.isfinite(value).all()) for value in tensors):
        raise ValueError("all frozen verifier features must be finite")
    if bool(((candidates < 0) | (candidates >= config.action_count)).any()):
        raise ValueError("candidate is outside the action vocabulary")
    if bool(((parent_actions < 0) | (parent_actions >= config.action_count)).any()):
        raise ValueError("parent action is outside the action vocabulary")
    if bool((mean_action_probabilities < 0.0).any()):
        raise ValueError("selector probabilities cannot be negative")

    dtype = action_latents.dtype
    device = action_latents.device
    aligned = (
        mean_action_probabilities.device == device
        and mean_gate.device == device
        and minimum_gate.device == device
        and physical_danger_probabilities.device == device
        and candidates.device == device
        and parent_actions.device == device
    )
    if not aligned:
        raise ValueError("frozen verifier features must share one device")
    if any(value.dtype != dtype for value in tensors[1:]):
        raise ValueError("frozen verifier floating tensors must share one dtype")

    candidate_indices = candidates.view(1, decisions, 1).expand(
        physical_danger_probabilities.shape[0], decisions, 1
    )
    parent_indices = parent_actions.view(1, decisions, 1).expand_as(candidate_indices)
    candidate_danger = physical_danger_probabilities.gather(
        -1, candidate_indices
    ).squeeze(-1)
    parent_danger = physical_danger_probabilities.gather(
        -1, parent_indices
    ).squeeze(-1)
    physical_summary = torch.stack(
        (
            candidate_danger.mean(dim=0),
            candidate_danger.amax(dim=0),
            parent_danger.mean(dim=0),
            parent_danger.amax(dim=0),
        ),
        dim=-1,
    )
    candidate_tokens = F.one_hot(
        candidates,
        num_classes=config.action_count,
    ).to(dtype=dtype)
    changed = (candidates != parent_actions).to(dtype=dtype).unsqueeze(-1)
    result = torch.cat(
        (
            action_latents.mean(dim=0),
            candidate_tokens,
            mean_action_probabilities,
            mean_gate.unsqueeze(-1),
            minimum_gate.unsqueeze(-1),
            physical_summary,
            changed,
        ),
        dim=-1,
    ).unsqueeze(0)
    if result.shape != (1, decisions, config.input_size):
        raise RuntimeError("temporal verifier input construction is misaligned")
    if not bool(torch.isfinite(result).all()):
        raise ValueError("temporal verifier inputs became nonfinite")
    return result


def selected_candidate_targets(
    episode: Any,
    candidates: Tensor,
) -> tuple[Tensor, Tensor]:
    """Build labels without permitting teacher evidence into inference inputs."""

    parent_actions = episode.parent_actions
    decisions = int(parent_actions.numel())
    _require_vector(candidates, decisions=decisions, name="candidates", dtype=torch.int64)
    _require_vector(
        parent_actions,
        decisions=decisions,
        name="parent_actions",
        dtype=torch.int64,
    )
    equivalent = episode.preferred_equivalent_actions
    required = episode.preferred_correction_required
    if equivalent.shape != (decisions, 18) or equivalent.dtype != torch.bool:
        raise ValueError("preferred equivalent actions do not align")
    _require_vector(
        required,
        decisions=decisions,
        name="preferred_correction_required",
        dtype=torch.bool,
    )
    has_equivalent = equivalent.any(dim=-1)
    if bool((has_equivalent & ~required).any()):
        raise ValueError("a no-correction row cannot contain equivalent corrections")
    if bool((required & ~has_equivalent).any()):
        raise ValueError("a correction-required row must contain an equivalent action")
    parent_is_equivalent = equivalent.gather(
        -1, parent_actions.unsqueeze(-1)
    ).squeeze(-1)
    if bool(parent_is_equivalent.any()):
        raise ValueError("the parent action cannot be an equivalent correction")
    mask = (
        episode.gate_valid
        & (episode.gate_targets > 0.0)
        & episode.anticipatory
        & (episode.anticipatory_lead_decisions >= EARLY_LEAD_MINIMUM)
        & (episode.anticipatory_lead_decisions <= EARLY_LEAD_MAXIMUM)
    )
    _require_vector(mask, decisions=decisions, name="early label mask", dtype=torch.bool)
    targets = (
        equivalent.gather(-1, candidates.unsqueeze(-1)).squeeze(-1)
        & required
    )
    return mask, targets


def hard_negative_pairwise_loss(
    logits: Tensor,
    labels: Tensor,
    *,
    margin: float,
    temperature: float,
    hard_negative_fraction: float,
) -> Tensor:
    if logits.ndim != 1 or labels.shape != logits.shape or labels.dtype != torch.bool:
        raise ValueError("pairwise logits and labels do not align")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("pairwise logits must be finite")
    positives = logits[labels]
    negatives = logits[~labels]
    if positives.numel() == 0 or negatives.numel() == 0:
        raise ValueError("pairwise training requires positive and negative rows")
    count = max(1, math.ceil(int(negatives.numel()) * hard_negative_fraction))
    hard_negatives = torch.topk(negatives, k=count, sorted=False).values
    differences = (
        hard_negatives.unsqueeze(0)
        - positives.unsqueeze(1)
        + margin
    ) / temperature
    return F.softplus(differences).mean()


def _snapshot_state(module: nn.Module) -> dict[str, Tensor]:
    return {
        name: value.detach().clone()
        for name, value in module.state_dict().items()
    }


def train_temporal_candidate_verifier(
    verifier: TemporalCandidateVerifier,
    episodes: Sequence[TemporalVerifierEpisode],
    *,
    seed: int,
    config: TemporalVerifierTrainingConfig = TemporalVerifierTrainingConfig(),
) -> list[dict[str, float]]:
    """Fit on labelled fit episodes and restore all caller runtime state."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("training seed must be an integer")
    if not episodes:
        raise ValueError("temporal verifier training requires episodes")
    if len({episode.seed for episode in episodes}) != len(episodes):
        raise ValueError("temporal verifier training episode seeds must be unique")
    if any(parameter.device.type != "cpu" for parameter in verifier.parameters()):
        raise ValueError("temporal verifier formal training requires CPU parameters")
    for episode in episodes:
        if episode.inputs.device.type != "cpu" or episode.inputs.dtype != torch.float32:
            raise ValueError("formal verifier inputs must be CPU float32")
        if episode.inputs.shape[-1] != verifier.config.input_size:
            raise ValueError("episode input width differs from verifier")

    state_before = _snapshot_state(verifier)
    module_modes = [(module, module.training) for module in verifier.modules()]
    parameter_runtime = [
        (
            parameter,
            parameter.requires_grad,
            None if parameter.grad is None else parameter.grad.detach().clone(),
        )
        for parameter in verifier.parameters()
    ]
    rng_state = torch.random.get_rng_state()

    def restore_runtime(*, rollback: bool) -> None:
        if rollback:
            verifier.load_state_dict(state_before, strict=True)
        for parameter, requires_grad, gradient in parameter_runtime:
            parameter.requires_grad_(requires_grad)
            parameter.grad = None if gradient is None else gradient.clone()
        for module, training in module_modes:
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
            member_parts: list[list[Tensor]] = [
                [] for _ in range(verifier.config.ensemble_size)
            ]
            label_parts: list[Tensor] = []
            for index in order:
                episode = episodes[index]
                logits = verifier.forward_logits(episode.inputs)
                if not bool(torch.isfinite(logits).all()):
                    raise ValueError("temporal verifier produced nonfinite logits")
                for member_index in range(verifier.config.ensemble_size):
                    member_parts[member_index].append(
                        logits[member_index, episode.label_mask]
                    )
                label_parts.append(episode.labels[episode.label_mask])
            labels = torch.cat(label_parts)
            if labels.numel() == 0:
                raise ValueError("temporal verifier fit has no labelled early rows")
            if not bool(labels.any()) or not bool((~labels).any()):
                raise ValueError("temporal verifier fit needs both label classes")
            member_losses: list[Tensor] = []
            bce_values: list[Tensor] = []
            rank_values: list[Tensor] = []
            for parts in member_parts:
                member_logits = torch.cat(parts)
                bce = F.binary_cross_entropy_with_logits(
                    member_logits,
                    labels.to(member_logits.dtype),
                )
                ranking = hard_negative_pairwise_loss(
                    member_logits,
                    labels,
                    margin=config.pairwise_margin,
                    temperature=config.pairwise_temperature,
                    hard_negative_fraction=config.hard_negative_fraction,
                )
                bce_values.append(bce)
                rank_values.append(ranking)
                member_losses.append(
                    bce + config.pairwise_loss_weight * ranking
                )
            loss = torch.stack(member_losses).mean()
            if not bool(torch.isfinite(loss)):
                raise ValueError("temporal verifier loss is nonfinite")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradients = [
                parameter.grad
                for parameter in verifier.parameters()
                if parameter.requires_grad
            ]
            if not gradients or any(gradient is None for gradient in gradients):
                raise ValueError("temporal verifier gradient is missing")
            if not all(bool(torch.isfinite(gradient).all()) for gradient in gradients):
                raise ValueError("temporal verifier gradient is nonfinite")
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                tuple(verifier.parameters()),
                config.gradient_clip_max_norm,
            )
            if not bool(torch.isfinite(gradient_norm)):
                raise ValueError("temporal verifier gradient norm is nonfinite")
            optimizer.step()
            if not all(
                bool(torch.isfinite(parameter).all())
                for parameter in verifier.parameters()
            ):
                raise ValueError("temporal verifier parameter became nonfinite")
            history.append({
                "epoch": float(epoch),
                "loss": float(loss.detach()),
                "bce": float(torch.stack(bce_values).mean().detach()),
                "hard_negative_pairwise": float(
                    torch.stack(rank_values).mean().detach()
                ),
                "labelled_rows": float(labels.numel()),
                "positive_rows": float(labels.sum()),
                "negative_rows": float((~labels).sum()),
                "gradient_norm_before_clip": float(gradient_norm.detach()),
            })
    except Exception:
        restore_runtime(rollback=True)
        raise
    restore_runtime(rollback=False)
    return history


def predict_temporal_candidate_verifier(
    verifier: TemporalCandidateVerifier,
    inputs: Tensor,
) -> dict[str, Tensor]:
    if inputs.device.type != "cpu" or inputs.dtype != torch.float32:
        raise ValueError("formal verifier prediction requires CPU float32 inputs")
    modes = [(module, module.training) for module in verifier.modules()]
    try:
        verifier.eval()
        with torch.no_grad():
            logits = verifier.forward_logits(inputs)
            probabilities = torch.sigmoid(logits)
        member_finite = torch.isfinite(logits) & torch.isfinite(probabilities)
        all_finite = member_finite.all(dim=0)
        confidence = probabilities.mean(dim=0)
        confidence = torch.where(
            all_finite,
            confidence,
            torch.full_like(confidence, float("nan")),
        )
        return {
            "member_logits": logits,
            "member_probabilities": probabilities,
            "member_finite": member_finite,
            "all_members_finite": all_finite,
            "confidence": confidence,
        }
    finally:
        for module, training in modes:
            module.training = training


def verifier_state_sha256(verifier: TemporalCandidateVerifier) -> str:
    digest = hashlib.sha256()
    digest.update(repr(asdict(verifier.config)).encode("utf-8"))
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
        "mean_frozen_action_recurrent_latent",
        "frozen_candidate_one_hot",
        "frozen_mean_selector_probabilities",
        "frozen_mean_and_minimum_onset_probability",
        "frozen_candidate_and_parent_physical_danger_mean_and_maximum",
        "frozen_candidate_differs_from_parent",
    ],
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
