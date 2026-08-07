"""Learned, conservative action corrections for a frozen streaming policy.

The adapter never replaces the parent policy recurrent state.  It observes the
parent's current GRU output and action logits, then either leaves the logits
bit-for-bit unchanged or raises one learned correction above the parent top-1.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any, Mapping

try:
    import torch
    from torch import Tensor, nn
except ImportError:  # pragma: no cover - base install intentionally omits torch
    torch = None
    Tensor = object  # type: ignore[assignment,misc]
    nn = None


FUTURE_ONSET_GATE_SEMANTICS = {
    "name": "binary_future_correctable_onset",
    "version": 2,
    "horizon_decisions": 10,
    "target_values": [0, 1],
    "negative_tail_censoring": "right_censor_incomplete_horizon",
    "requires_current_candidate_physical_safety": True,
    "continuation_policy": "learned_without_unproven_previous_action_substitution",
}

ACTION_LOGIT_MODES = (
    "absolute",
    "parent_residual_joint",
    "parent_residual_factorized",
    "certified_membership",
)
SEMANTIC_PLAYER_POSITION_SIZE = 2


def _action_logit_semantics(mode: str) -> dict[str, Any]:
    if mode == "certified_membership":
        return {
            "name": "independent_certified_action_membership",
            "version": 1,
            "mode": mode,
            "decode": "raw_membership_logits",
            "probability": "finite_per_action_sigmoid",
            "ensemble_candidate": "argmax_mean_membership_probability",
            "parent_logits_added_during_decode": False,
            "parent_context_remains_model_input": True,
        }
    return {
        "name": "learned_parent_logit_residual",
        "version": 1,
        "mode": mode,
        "zero_delta": "parent_logits",
    }


def _selector_logit_semantics(mode: str) -> dict[str, Any]:
    if mode not in {"parent_residual_joint", "parent_residual_factorized"}:
        raise ValueError("dual-head selectors require parent-logit residuals")
    return {
        "name": "learned_parent_logit_residual_selector",
        "version": 1,
        "mode": mode,
        "zero_delta": "parent_logits",
        "probability": "finite_per_action_softmax",
        "ensemble_candidate": "argmax_mean_selector_probability",
        "ensemble_agreement": "mean_member_selector_argmax_matches_candidate",
        "candidate_source": "selector_head_only",
    }


MEMBERSHIP_CONFIDENCE_SEMANTICS = {
    "name": "independent_certified_action_membership_confidence",
    "version": 1,
    "probability": "finite_per_action_sigmoid",
    "ensemble_probability": "mean_member_membership_probability",
    "correction_confidence": (
        "mean_membership_probability_at_selector_selected_action"
    ),
    "candidate_source": "selector_head_only",
    "agreement_source": "selector_head_only",
    "membership_head_input": "detached_action_recurrent",
    "finite_requirement": "all_selector_and_membership_members_finite",
}


@dataclass(frozen=True, slots=True)
class ResidualAdapterConfig:
    recurrent_size: int
    action_count: int = 18
    hidden_size: int = 128
    ensemble_size: int = 3
    executed_action_context: bool = False
    per_action_safety_critic: bool = False
    visual_latent_size: int = 0
    per_action_physical_danger: bool = False
    action_logit_mode: str = "absolute"
    semantic_player_position: bool = False
    separate_action_recurrent: bool = False
    per_action_membership_confidence: bool = False

    def __post_init__(self) -> None:
        for name in ("recurrent_size", "action_count", "hidden_size", "ensemble_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.action_count != 18:
            raise ValueError("residual adapters require the 18-action vocabulary")
        if not isinstance(self.executed_action_context, bool):
            raise ValueError("executed_action_context must be a Boolean")
        if not isinstance(self.per_action_safety_critic, bool):
            raise ValueError("per_action_safety_critic must be a Boolean")
        if (
            isinstance(self.visual_latent_size, bool)
            or not isinstance(self.visual_latent_size, int)
            or self.visual_latent_size < 0
        ):
            raise ValueError("visual_latent_size must be a nonnegative integer")
        if not isinstance(self.per_action_physical_danger, bool):
            raise ValueError("per_action_physical_danger must be a Boolean")
        if self.per_action_physical_danger and self.visual_latent_size <= 0:
            raise ValueError("physical danger heads require a visual latent")
        if self.per_action_physical_danger and not self.per_action_safety_critic:
            raise ValueError("physical danger heads require dense safety heads")
        if self.action_logit_mode not in ACTION_LOGIT_MODES:
            choices = ", ".join(ACTION_LOGIT_MODES)
            raise ValueError(f"action_logit_mode must be one of: {choices}")
        if not isinstance(self.semantic_player_position, bool):
            raise ValueError("semantic_player_position must be a Boolean")
        if self.semantic_player_position and self.visual_latent_size <= 0:
            raise ValueError("semantic player position requires a visual latent")
        if not isinstance(self.separate_action_recurrent, bool):
            raise ValueError("separate_action_recurrent must be a Boolean")
        if not isinstance(self.per_action_membership_confidence, bool):
            raise ValueError("per_action_membership_confidence must be a Boolean")
        if (
            self.per_action_membership_confidence
            and not self.separate_action_recurrent
        ):
            raise ValueError(
                "per-action membership confidence requires a separate action "
                "recurrent"
            )
        if self.per_action_membership_confidence and self.action_logit_mode not in {
            "parent_residual_joint",
            "parent_residual_factorized",
        }:
            raise ValueError(
                "per-action membership confidence requires parent-logit "
                "residual selector logits"
            )
        if self.separate_action_recurrent and self.action_logit_mode == "absolute":
            raise ValueError(
                "a separate action recurrent requires parent-logit residuals"
            )

    @property
    def executed_action_context_size(self) -> int:
        # One-hot action, validity, and log1p(consecutive held decisions).
        return self.action_count + 2 if self.executed_action_context else 0

    @property
    def feature_size(self) -> int:
        # Parent recurrent state, centered parent logits, and parent top-1 token.
        return (
            self.recurrent_size
            + self.action_count * 2
            + self.executed_action_context_size
            + self.visual_latent_size
            + (
                SEMANTIC_PLAYER_POSITION_SIZE
                if self.semantic_player_position else
                0
            )
        )


@dataclass(frozen=True, slots=True)
class ResidualRuntimeConfig:
    gate_probability_threshold: float = 0.95
    minimum_member_gate_probability: float = 0.5
    action_probability_threshold: float = 0.5
    ensemble_agreement_threshold: float = 1.0
    override_logit_margin: float = 1.0
    legacy_gate_enabled: bool = True
    critic_enabled: bool = False
    current_critic_request_enabled: bool = True
    prefer_safe_previous_action: bool = False
    parent_collision_probability_threshold: float = 0.9
    candidate_collision_probability_threshold: float = 0.1
    parent_minimum_margin_threshold: float = 8.0
    candidate_minimum_margin_threshold: float = 8.0
    parent_danger_agreement_threshold: float = 2.0 / 3.0
    candidate_safety_agreement_threshold: float = 1.0
    critic_signal: str = "collision_margin"
    parent_physical_danger_probability_threshold: float = 0.5
    candidate_physical_danger_probability_threshold: float = 0.5
    future_onset_gate_enabled: bool = False

    def __post_init__(self) -> None:
        probabilities = (
            self.gate_probability_threshold,
            self.minimum_member_gate_probability,
            self.action_probability_threshold,
            self.ensemble_agreement_threshold,
            self.parent_collision_probability_threshold,
            self.candidate_collision_probability_threshold,
            self.parent_danger_agreement_threshold,
            self.candidate_safety_agreement_threshold,
            self.parent_physical_danger_probability_threshold,
            self.candidate_physical_danger_probability_threshold,
        )
        if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in probabilities):
            raise ValueError("residual probability thresholds must be finite and in [0, 1]")
        if not math.isfinite(self.override_logit_margin) or self.override_logit_margin <= 0.0:
            raise ValueError("override_logit_margin must be finite and positive")
        if not isinstance(self.legacy_gate_enabled, bool):
            raise ValueError("legacy_gate_enabled must be a Boolean")
        if not isinstance(self.critic_enabled, bool):
            raise ValueError("critic_enabled must be a Boolean")
        if not isinstance(self.current_critic_request_enabled, bool):
            raise ValueError("current_critic_request_enabled must be a Boolean")
        if not isinstance(self.prefer_safe_previous_action, bool):
            raise ValueError("prefer_safe_previous_action must be a Boolean")
        if not isinstance(self.future_onset_gate_enabled, bool):
            raise ValueError("future_onset_gate_enabled must be a Boolean")
        if self.critic_signal not in {"collision_margin", "physical_danger"}:
            raise ValueError("critic_signal must be collision_margin or physical_danger")
        if self.prefer_safe_previous_action and not self.critic_enabled:
            raise ValueError("safe-previous selection requires an enabled critic")
        if self.future_onset_gate_enabled and (
            not self.critic_enabled or self.critic_signal != "physical_danger"
        ):
            raise ValueError(
                "future-onset gates require an enabled physical danger critic"
            )
        if self.future_onset_gate_enabled and self.legacy_gate_enabled:
            raise ValueError("future-onset and legacy gates cannot both be enabled")
        if self.future_onset_gate_enabled and self.prefer_safe_previous_action:
            raise ValueError(
                "future-onset continuation must be learned; unproven previous "
                "actions cannot be substituted"
            )
        if (
            self.parent_danger_agreement_threshold <= 0.0
            or self.candidate_safety_agreement_threshold <= 0.0
        ):
            raise ValueError("critic agreement thresholds must be positive")
        margins = (
            self.parent_minimum_margin_threshold,
            self.candidate_minimum_margin_threshold,
        )
        if not all(math.isfinite(value) for value in margins):
            raise ValueError("critic margin thresholds must be finite")


if nn is not None:

    def decode_residual_action_logits(
        action_values: Tensor,
        parent_logits: Tensor,
        mode: str,
    ) -> Tensor:
        """Decode absolute scores or learned deltas against frozen parent logits."""

        if mode not in ACTION_LOGIT_MODES:
            choices = ", ".join(ACTION_LOGIT_MODES)
            raise ValueError(f"action logit mode must be one of: {choices}")
        if action_values.ndim < 1 or parent_logits.ndim < 1:
            raise ValueError("action values and parent logits must have an action axis")
        if action_values.shape[-1] != 18 or parent_logits.shape[-1] != 18:
            raise ValueError("residual action decoding requires 18 action logits")
        if action_values.ndim < parent_logits.ndim:
            raise ValueError("action values cannot have lower rank than parent logits")
        if not torch.is_floating_point(action_values) or not torch.is_floating_point(
            parent_logits
        ):
            raise ValueError("action values and parent logits must be floating point")

        leading_dimensions = action_values.ndim - parent_logits.ndim
        expanded_parent = parent_logits
        for _ in range(leading_dimensions):
            expanded_parent = expanded_parent.unsqueeze(0)
        if action_values.shape[leading_dimensions:] != parent_logits.shape:
            raise ValueError("action values and parent logits do not align")
        if mode in {"absolute", "certified_membership"}:
            return action_values

        expanded_parent = expanded_parent.to(action_values)
        if mode == "parent_residual_joint":
            return expanded_parent + action_values

        joint_deltas = action_values.reshape(*action_values.shape[:-1], 2, 9)
        direction_deltas = (
            torch.logsumexp(joint_deltas, dim=-2) - math.log(2.0)
        )
        speed_deltas = (
            torch.logsumexp(joint_deltas, dim=-1) - math.log(9.0)
        )
        factorized_deltas = (
            speed_deltas.unsqueeze(-1) + direction_deltas.unsqueeze(-2)
        ).reshape_as(action_values)
        return expanded_parent + factorized_deltas

    def semantic_player_position_features(global_frames: Tensor) -> Tensor:
        """Decode normalized player coordinates from semantic channel four."""

        if global_frames.ndim < 3 or global_frames.shape[-3] <= 4:
            raise ValueError(
                "global semantic frames must include player channel four"
            )
        if not torch.is_floating_point(global_frames):
            raise ValueError("global semantic frames must be floating point")
        player = global_frames[..., 4, :, :]
        if not bool(torch.isfinite(player).all()):
            raise ValueError("global semantic player channel must be finite")
        if bool((player < 0.0).any()):
            raise ValueError("global semantic player channel cannot be negative")
        mass = player.sum(dim=(-2, -1))
        if not bool(torch.isfinite(mass).all()) or bool((mass <= 0.0).any()):
            raise ValueError("global semantic player marker is missing")
        xs = torch.linspace(
            -1.0,
            1.0,
            player.shape[-1],
            dtype=player.dtype,
            device=player.device,
        )
        ys = torch.linspace(
            -1.0,
            1.0,
            player.shape[-2],
            dtype=player.dtype,
            device=player.device,
        )
        x = (player.sum(dim=-2) * xs).sum(dim=-1) / mass
        y = (player.sum(dim=-1) * ys).sum(dim=-1) / mass
        return torch.stack((x, y), dim=-1)

    def finite_sigmoid(logits: Tensor) -> Tensor:
        """Return sigmoid probabilities while preserving nonfinite failures."""

        probabilities = torch.sigmoid(logits)
        return torch.where(
            torch.isfinite(logits) & torch.isfinite(probabilities),
            probabilities,
            torch.full_like(probabilities, float("nan")),
        )

    def finite_action_probabilities(
        logits: Tensor,
        action_logit_mode: str = "absolute",
    ) -> tuple[Tensor, Tensor]:
        """Return stable probabilities and a per-member raw-logit finite mask."""

        if action_logit_mode not in ACTION_LOGIT_MODES:
            choices = ", ".join(ACTION_LOGIT_MODES)
            raise ValueError(f"action logit mode must be one of: {choices}")
        if action_logit_mode == "certified_membership":
            action_finite = torch.isfinite(logits)
            safe_logits = torch.where(
                action_finite,
                logits,
                torch.zeros_like(logits),
            )
            probabilities = torch.sigmoid(safe_logits)
            action_finite &= torch.isfinite(probabilities)
            probabilities = torch.where(
                action_finite,
                probabilities,
                torch.zeros_like(probabilities),
            )
            return probabilities, action_finite.all(dim=-1)

        member_finite = torch.isfinite(logits).all(dim=-1)
        safe_logits = torch.where(
            member_finite.unsqueeze(-1),
            logits,
            torch.zeros_like(logits),
        )
        probabilities = torch.softmax(safe_logits, dim=-1)
        probability_finite = torch.isfinite(probabilities).all(dim=-1)
        member_finite = member_finite & probability_finite
        probabilities = torch.where(
            member_finite.unsqueeze(-1),
            probabilities,
            torch.zeros_like(probabilities),
        )
        return probabilities, member_finite

    def ensemble_action_summary(
        action_probabilities: Tensor,
        action_member_finite: Tensor,
        membership_probabilities: Tensor | None = None,
        membership_member_finite: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Summarize an action ensemble identically for offline and live use."""

        if action_probabilities.ndim < 2:
            raise ValueError("ensemble action probabilities require an action axis")
        if action_member_finite.shape != action_probabilities.shape[:-1]:
            raise ValueError("action member finite mask does not align")
        if action_member_finite.dtype != torch.bool:
            raise ValueError("action member finite mask must be Boolean")
        member_finite = action_member_finite & torch.isfinite(
            action_probabilities,
        ).all(dim=-1)
        safe_probabilities = torch.where(
            member_finite.unsqueeze(-1),
            action_probabilities,
            torch.zeros_like(action_probabilities),
        )
        mean_probabilities = safe_probabilities.mean(dim=0)
        candidates = mean_probabilities.argmax(dim=-1)
        member_actions = safe_probabilities.argmax(dim=-1)
        summary = {
            "action_all_members_finite": member_finite.all(dim=0),
            "mean_action_probabilities": mean_probabilities,
            "candidates": candidates,
            "action_confidence": mean_probabilities.amax(dim=-1),
            "agreement": (
                member_actions == candidates.unsqueeze(0)
            ).to(mean_probabilities.dtype).mean(dim=0),
        }
        if (membership_probabilities is None) != (
            membership_member_finite is None
        ):
            raise ValueError(
                "membership probabilities and finite mask must be provided together"
            )
        if membership_probabilities is None:
            return summary

        assert membership_member_finite is not None
        if membership_probabilities.shape != action_probabilities.shape:
            raise ValueError("membership probabilities do not align with selectors")
        if membership_member_finite.shape != action_member_finite.shape:
            raise ValueError("membership finite mask does not align with selectors")
        if membership_member_finite.dtype != torch.bool:
            raise ValueError("membership member finite mask must be Boolean")
        membership_finite = membership_member_finite & torch.isfinite(
            membership_probabilities,
        ).all(dim=-1)
        safe_membership = torch.where(
            membership_finite.unsqueeze(-1),
            membership_probabilities,
            torch.zeros_like(membership_probabilities),
        )
        mean_membership = safe_membership.mean(dim=0)
        selected_membership = mean_membership.gather(
            -1,
            candidates.unsqueeze(-1),
        ).squeeze(-1)
        summary.update({
            "action_all_members_finite": (
                member_finite & membership_finite
            ).all(dim=0),
            "selector_all_members_finite": member_finite.all(dim=0),
            "membership_all_members_finite": membership_finite.all(dim=0),
            "mean_membership_probabilities": mean_membership,
            "action_confidence": selected_membership,
        })
        return summary

    def residual_future_onset_mask(
        mean_gate: Tensor,
        minimum_gate: Tensor,
        runtime_config: ResidualRuntimeConfig,
    ) -> Tensor:
        """Decode the v5 temporal onset gate with finite, ensemble-safe checks."""

        if mean_gate.shape != minimum_gate.shape:
            raise ValueError("residual gate probabilities do not align")
        return (
            runtime_config.future_onset_gate_enabled
            & torch.isfinite(mean_gate)
            & torch.isfinite(minimum_gate)
            & (mean_gate >= runtime_config.gate_probability_threshold)
            & (
                minimum_gate
                >= runtime_config.minimum_member_gate_probability
            )
        )

    def residual_candidate_selection(
        *,
        correction_actions: Tensor,
        correction_confidence: Tensor,
        agreement: Tensor,
        previous_actions: Tensor | None,
        runtime_config: ResidualRuntimeConfig,
        collision_probabilities: Tensor | None = None,
        minimum_margins: Tensor | None = None,
        physical_danger_probabilities: Tensor | None = None,
        parent_actions: Tensor | None = None,
        future_onset: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Apply learned safety inertia without searching for a new route."""

        selected = correction_actions
        selected_confidence = correction_confidence
        selected_agreement = agreement
        used_previous = torch.zeros_like(correction_actions, dtype=torch.bool)
        if not runtime_config.prefer_safe_previous_action:
            return {
                "correction_actions": selected,
                "correction_confidence": selected_confidence,
                "agreement": selected_agreement,
                "used_previous": used_previous,
            }
        if previous_actions is None:
            raise ValueError("safe-previous selection requires executed actions")
        if previous_actions.shape != correction_actions.shape:
            raise ValueError("previous actions do not align with correction actions")
        if runtime_config.critic_signal == "physical_danger":
            if physical_danger_probabilities is None:
                raise ValueError(
                    "physical safe-previous selection requires danger predictions"
                )
            dense_shape = physical_danger_probabilities.shape
        else:
            if collision_probabilities is None or minimum_margins is None:
                raise ValueError("safe-previous selection requires dense predictions")
            if collision_probabilities.shape != minimum_margins.shape:
                raise ValueError("critic collision and margin predictions do not align")
            dense_shape = collision_probabilities.shape
        if dense_shape[1:-1] != correction_actions.shape:
            raise ValueError("critic predictions do not align with correction actions")

        valid_previous = (previous_actions >= 0) & (
            previous_actions < dense_shape[-1]
        )
        indices = previous_actions.clamp_min(0).unsqueeze(0).unsqueeze(-1).expand(
            dense_shape[0],
            *previous_actions.shape,
            1,
        )
        if runtime_config.critic_signal == "physical_danger":
            assert physical_danger_probabilities is not None
            danger = physical_danger_probabilities.gather(-1, indices).squeeze(-1)
            member_safe = (
                torch.isfinite(danger)
                & (
                    danger
                    <= runtime_config.candidate_physical_danger_probability_threshold
                )
            )
        else:
            assert collision_probabilities is not None
            assert minimum_margins is not None
            collision = collision_probabilities.gather(-1, indices).squeeze(-1)
            margin = minimum_margins.gather(-1, indices).squeeze(-1)
            member_safe = (
                torch.isfinite(collision)
                & torch.isfinite(margin)
                & (
                    collision
                    <= runtime_config.candidate_collision_probability_threshold
                )
                & (margin >= runtime_config.candidate_minimum_margin_threshold)
            )
        safety_agreement = member_safe.to(correction_confidence.dtype).mean(dim=0)
        used_previous = valid_previous & (
            safety_agreement
            >= runtime_config.candidate_safety_agreement_threshold
        )
        used_previous &= (
            torch.isfinite(correction_confidence)
            & torch.isfinite(agreement)
        )
        if runtime_config.critic_signal == "physical_danger":
            assert physical_danger_probabilities is not None
            used_previous &= torch.isfinite(
                physical_danger_probabilities.gather(-1, indices).squeeze(-1)
            ).all(dim=0)
        else:
            assert collision_probabilities is not None
            assert minimum_margins is not None
            used_previous &= (
                torch.isfinite(
                    collision_probabilities.gather(-1, indices).squeeze(-1)
                ).all(dim=0)
                & torch.isfinite(
                    minimum_margins.gather(-1, indices).squeeze(-1)
                ).all(dim=0)
            )
        selected = torch.where(used_previous, previous_actions, correction_actions)
        # A unanimously safety-certified hold is a deliberate inertia candidate;
        # action-head confidence remains authoritative for escape candidates.
        selected_confidence = torch.where(
            used_previous,
            torch.ones_like(correction_confidence),
            correction_confidence,
        )
        selected_agreement = torch.where(
            used_previous,
            safety_agreement,
            agreement,
        )
        return {
            "correction_actions": selected,
            "correction_confidence": selected_confidence,
            "agreement": selected_agreement,
            "used_previous": used_previous,
        }

    def residual_override_masks(
        *,
        mean_gate: Tensor,
        minimum_gate: Tensor,
        action_all_members_finite: Tensor,
        correction_confidence: Tensor,
        agreement: Tensor,
        correction_actions: Tensor,
        parent_actions: Tensor,
        runtime_config: ResidualRuntimeConfig,
        collision_probabilities: Tensor | None = None,
        minimum_margins: Tensor | None = None,
        physical_danger_probabilities: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Return the exact fail-closed masks shared by offline and live use."""

        if action_all_members_finite.shape != correction_actions.shape:
            raise ValueError("action finite mask does not align with corrections")
        if action_all_members_finite.dtype != torch.bool:
            raise ValueError("action finite mask must be Boolean")
        quality = (
            action_all_members_finite
            & torch.isfinite(mean_gate)
            & torch.isfinite(minimum_gate)
            & torch.isfinite(correction_confidence)
            & torch.isfinite(agreement)
            & (correction_confidence >= runtime_config.action_probability_threshold)
            & (agreement >= runtime_config.ensemble_agreement_threshold)
            & (correction_actions != parent_actions)
        )
        legacy_request = (
            runtime_config.legacy_gate_enabled
            & quality
            & (mean_gate >= runtime_config.gate_probability_threshold)
            & (
                minimum_gate
                >= runtime_config.minimum_member_gate_probability
            )
        )
        future_onset = residual_future_onset_mask(
            mean_gate,
            minimum_gate,
            runtime_config,
        )
        future_onset_request = quality & future_onset
        empty = torch.zeros_like(legacy_request)
        if not runtime_config.critic_enabled:
            return {
                "action_all_members_finite": action_all_members_finite,
                "quality": quality,
                "legacy_request": legacy_request,
                "legacy_accepted": legacy_request,
                "critic_request": empty,
                "critic_accepted": empty,
                "future_onset": future_onset,
                "future_onset_request": future_onset_request,
                "future_onset_accepted": empty,
                "parent_danger": empty,
                "candidate_safe": empty,
                "candidate_safety_veto": empty,
                "override": legacy_request,
            }
        if runtime_config.critic_signal == "physical_danger":
            if physical_danger_probabilities is None:
                raise ValueError(
                    "enabled physical critic requires danger predictions"
                )
            dense_shape = physical_danger_probabilities.shape
        else:
            if collision_probabilities is None or minimum_margins is None:
                raise ValueError("enabled safety critic requires dense action predictions")
            if collision_probabilities.shape != minimum_margins.shape:
                raise ValueError("critic collision and margin predictions do not align")
            dense_shape = collision_probabilities.shape
        if len(dense_shape) != correction_actions.ndim + 2:
            raise ValueError("critic predictions have an invalid rank")
        if dense_shape[1:-1] != correction_actions.shape:
            raise ValueError("critic predictions do not align with actions")

        ensemble = dense_shape[0]
        parent_indices = parent_actions.unsqueeze(0).unsqueeze(-1).expand(
            ensemble,
            *parent_actions.shape,
            1,
        )
        candidate_indices = correction_actions.unsqueeze(0).unsqueeze(-1).expand(
            ensemble,
            *correction_actions.shape,
            1,
        )
        if runtime_config.critic_signal == "physical_danger":
            assert physical_danger_probabilities is not None
            parent_danger_score = physical_danger_probabilities.gather(
                -1,
                parent_indices,
            ).squeeze(-1)
            candidate_danger_score = physical_danger_probabilities.gather(
                -1,
                candidate_indices,
            ).squeeze(-1)
            parent_member_danger = (
                torch.isfinite(parent_danger_score)
                & (
                    parent_danger_score
                    >= runtime_config.parent_physical_danger_probability_threshold
                )
            )
            candidate_member_safe = (
                torch.isfinite(candidate_danger_score)
                & (
                    candidate_danger_score
                    <= runtime_config.candidate_physical_danger_probability_threshold
                )
            )
        else:
            assert collision_probabilities is not None
            assert minimum_margins is not None
            parent_collision = collision_probabilities.gather(
                -1,
                parent_indices,
            ).squeeze(-1)
            parent_margin = minimum_margins.gather(
                -1,
                parent_indices,
            ).squeeze(-1)
            candidate_collision = collision_probabilities.gather(
                -1,
                candidate_indices,
            ).squeeze(-1)
            candidate_margin = minimum_margins.gather(
                -1,
                candidate_indices,
            ).squeeze(-1)
            parent_member_danger = (
                torch.isfinite(parent_collision)
                & torch.isfinite(parent_margin)
                & (
                    (
                        parent_collision
                        >= runtime_config.parent_collision_probability_threshold
                    )
                    | (
                        parent_margin
                        <= runtime_config.parent_minimum_margin_threshold
                    )
                )
            )
            candidate_member_safe = (
                torch.isfinite(candidate_collision)
                & torch.isfinite(candidate_margin)
                & (
                    candidate_collision
                    <= runtime_config.candidate_collision_probability_threshold
                )
                & (
                    candidate_margin
                    >= runtime_config.candidate_minimum_margin_threshold
                )
            )
        parent_danger = (
            parent_member_danger.to(mean_gate.dtype).mean(dim=0)
            >= runtime_config.parent_danger_agreement_threshold
        )
        candidate_safe = (
            candidate_member_safe.to(mean_gate.dtype).mean(dim=0)
            >= runtime_config.candidate_safety_agreement_threshold
        )
        if runtime_config.critic_signal == "physical_danger":
            parent_danger &= torch.isfinite(parent_danger_score).all(dim=0)
            candidate_safe &= torch.isfinite(candidate_danger_score).all(dim=0)
        else:
            parent_danger &= (
                torch.isfinite(parent_collision).all(dim=0)
                & torch.isfinite(parent_margin).all(dim=0)
            )
            candidate_safe &= (
                torch.isfinite(candidate_collision).all(dim=0)
                & torch.isfinite(candidate_margin).all(dim=0)
            )
        critic_request = (
            runtime_config.current_critic_request_enabled
            & quality
            & parent_danger
        )
        legacy_accepted = legacy_request & candidate_safe
        critic_accepted = critic_request & candidate_safe
        future_onset_accepted = future_onset_request & candidate_safe
        requested = legacy_request | critic_request | future_onset_request
        return {
            "action_all_members_finite": action_all_members_finite,
            "quality": quality,
            "legacy_request": legacy_request,
            "legacy_accepted": legacy_accepted,
            "critic_request": critic_request,
            "critic_accepted": critic_accepted,
            "future_onset": future_onset,
            "future_onset_request": future_onset_request,
            "future_onset_accepted": future_onset_accepted,
            "parent_danger": parent_danger,
            "candidate_safe": candidate_safe,
            "candidate_safety_veto": requested & ~candidate_safe,
            "override": legacy_accepted | critic_accepted | future_onset_accepted,
        }

    class _ResidualCorrectionMember(nn.Module):
        def __init__(
            self,
            feature_size: int,
            hidden_size: int,
            action_count: int,
            *,
            per_action_safety_critic: bool,
            per_action_physical_danger: bool,
            separate_action_recurrent: bool,
        ) -> None:
            super().__init__()
            self.input_projection = nn.Sequential(
                nn.Linear(feature_size, hidden_size),
                nn.SiLU(),
                nn.LayerNorm(hidden_size),
            )
            self.recurrent = nn.GRU(hidden_size, hidden_size, batch_first=True)
            self.action_input_projection = (
                nn.Sequential(
                    nn.Linear(feature_size, hidden_size),
                    nn.SiLU(),
                    nn.LayerNorm(hidden_size),
                )
                if separate_action_recurrent else
                None
            )
            self.action_recurrent = (
                nn.GRU(hidden_size, hidden_size, batch_first=True)
                if separate_action_recurrent else
                None
            )
            self.gate_head = nn.Linear(hidden_size, 1)
            self.action_head = nn.Linear(hidden_size, action_count)
            self.collision_head = (
                nn.Linear(hidden_size, action_count)
                if per_action_safety_critic else None
            )
            self.minimum_margin_head = (
                nn.Linear(hidden_size, action_count)
                if per_action_safety_critic else None
            )
            self.physical_danger_head = (
                nn.Linear(hidden_size, action_count)
                if per_action_physical_danger else None
            )
            # The adapter attaches this only after every base ensemble member
            # has been initialized, preserving the original ensemble RNG stream.
            self.membership_head: nn.Linear | None = None

        def forward_with_all_safety_and_membership(
            self,
            features: Tensor,
            hidden: Tensor | None = None,
        ) -> tuple[
            Tensor,
            Tensor,
            Tensor | None,
            Tensor | None,
            Tensor | None,
            Tensor | None,
            Tensor,
        ]:
            if self.action_recurrent is None:
                shared_hidden = hidden
                action_hidden = None
            elif hidden is None:
                shared_hidden = None
                action_hidden = None
            else:
                if hidden.ndim != 3 or hidden.shape[0] != 2:
                    raise ValueError(
                        "separate action recurrent hidden state must contain "
                        "shared and action layers"
                    )
                shared_hidden = hidden[:1]
                action_hidden = hidden[1:]
            encoded = self.input_projection(features)
            recurrent, shared_hidden = self.recurrent(encoded, shared_hidden)
            if self.action_recurrent is None:
                action_recurrent = recurrent
                hidden = shared_hidden
            else:
                assert self.action_input_projection is not None
                action_encoded = self.action_input_projection(features)
                action_recurrent, action_hidden = self.action_recurrent(
                    action_encoded,
                    action_hidden,
                )
                hidden = torch.cat((shared_hidden, action_hidden), dim=0)
            return (
                self.gate_head(recurrent).squeeze(-1),
                self.action_head(action_recurrent),
                None if self.collision_head is None else self.collision_head(recurrent),
                (
                    None
                    if self.minimum_margin_head is None else
                    self.minimum_margin_head(recurrent)
                ),
                (
                    None
                    if self.physical_danger_head is None else
                    self.physical_danger_head(recurrent)
                ),
                (
                    None
                    if self.membership_head is None else
                    self.membership_head(action_recurrent.detach())
                ),
                hidden,
            )

        def forward_with_all_safety(
            self,
            features: Tensor,
            hidden: Tensor | None = None,
        ) -> tuple[
            Tensor,
            Tensor,
            Tensor | None,
            Tensor | None,
            Tensor | None,
            Tensor,
        ]:
            gate, actions, collision, margin, physical, _membership, hidden = (
                self.forward_with_all_safety_and_membership(features, hidden)
            )
            return gate, actions, collision, margin, physical, hidden

        def forward_with_safety(
            self,
            features: Tensor,
            hidden: Tensor | None = None,
        ) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None, Tensor]:
            gate, actions, collision, margin, _physical, hidden = (
                self.forward_with_all_safety(features, hidden)
            )
            return gate, actions, collision, margin, hidden

        def forward(
            self,
            features: Tensor,
            hidden: Tensor | None = None,
        ) -> tuple[Tensor, Tensor, Tensor]:
            gate, actions, _collision, _margin, hidden = self.forward_with_safety(
                features,
                hidden,
            )
            return gate, actions, hidden


    class ResidualCorrectionAdapter(nn.Module):
        """An ensemble that predicts whether and how to override a frozen parent."""

        def __init__(self, config: ResidualAdapterConfig) -> None:
            super().__init__()
            self.config = config
            self.members = nn.ModuleList([
                _ResidualCorrectionMember(
                    config.feature_size,
                    config.hidden_size,
                    config.action_count,
                    per_action_safety_critic=config.per_action_safety_critic,
                    per_action_physical_danger=config.per_action_physical_danger,
                    separate_action_recurrent=config.separate_action_recurrent,
                )
                for _ in range(config.ensemble_size)
            ])
            if config.per_action_membership_confidence:
                for member in self.members:
                    member.membership_head = nn.Linear(
                        config.hidden_size,
                        config.action_count,
                    )
            if config.action_logit_mode != "absolute":
                for member in self.members:
                    nn.init.zeros_(member.action_head.weight)
                    nn.init.zeros_(member.action_head.bias)
            self.register_buffer("feature_mean", torch.zeros(config.feature_size))
            self.register_buffer("feature_scale", torch.ones(config.feature_size))

        def raw_features(
            self,
            recurrent: Tensor,
            parent_logits: Tensor,
            executed_action_context: Tensor | None = None,
            visual_features: Tensor | None = None,
            player_position_features: Tensor | None = None,
        ) -> Tensor:
            if recurrent.shape[:-1] != parent_logits.shape[:-1]:
                raise ValueError("parent recurrent state and logits do not align")
            if recurrent.shape[-1] != self.config.recurrent_size:
                raise ValueError("parent recurrent width does not match adapter")
            if parent_logits.shape[-1] != self.config.action_count:
                raise ValueError("parent action width does not match adapter")
            centered_logits = parent_logits - parent_logits.mean(dim=-1, keepdim=True)
            parent_actions = parent_logits.argmax(dim=-1)
            parent_tokens = torch.nn.functional.one_hot(
                parent_actions,
                num_classes=self.config.action_count,
            ).to(dtype=parent_logits.dtype)
            values = [recurrent, centered_logits, parent_tokens]
            if self.config.executed_action_context:
                if executed_action_context is None:
                    raise ValueError(
                        "executed action context is required by this residual adapter"
                    )
                expected = (
                    *parent_logits.shape[:-1],
                    self.config.executed_action_context_size,
                )
                if executed_action_context.shape != expected:
                    raise ValueError("executed action context does not align with logits")
                if not torch.isfinite(executed_action_context).all():
                    raise ValueError("executed action context must be finite")
                values.append(executed_action_context.to(parent_logits))
            elif executed_action_context is not None:
                raise ValueError(
                    "executed action context was provided to a context-free adapter"
                )
            if self.config.visual_latent_size:
                if visual_features is None:
                    raise ValueError("visual features are required by this residual adapter")
                expected_visual = (
                    *parent_logits.shape[:-1],
                    self.config.visual_latent_size,
                )
                if visual_features.shape != expected_visual:
                    raise ValueError("visual features do not align with parent logits")
                if not torch.isfinite(visual_features).all():
                    raise ValueError("visual features must be finite")
                values.append(visual_features.to(parent_logits))
            elif visual_features is not None:
                raise ValueError(
                    "visual features were provided to a state-only residual adapter"
                )
            if self.config.semantic_player_position:
                if player_position_features is None:
                    raise ValueError(
                        "semantic player position is required by this residual adapter"
                    )
                expected_position = (
                    *parent_logits.shape[:-1],
                    SEMANTIC_PLAYER_POSITION_SIZE,
                )
                if player_position_features.shape != expected_position:
                    raise ValueError(
                        "semantic player position does not align with parent logits"
                    )
                if not torch.isfinite(player_position_features).all():
                    raise ValueError("semantic player position must be finite")
                values.append(player_position_features.to(parent_logits))
            elif player_position_features is not None:
                raise ValueError(
                    "semantic player position was provided to an unconditioned adapter"
                )
            return torch.cat(values, dim=-1)

        def set_feature_normalization(self, mean: Tensor, scale: Tensor) -> None:
            if mean.shape != self.feature_mean.shape or scale.shape != self.feature_scale.shape:
                raise ValueError("adapter feature normalization has an invalid shape")
            if not torch.isfinite(mean).all() or not torch.isfinite(scale).all():
                raise ValueError("adapter feature normalization must be finite")
            if not torch.all(scale > 0.0):
                raise ValueError("adapter feature scales must be positive")
            self.feature_mean.copy_(mean.detach().to(self.feature_mean))
            self.feature_scale.copy_(scale.detach().to(self.feature_scale))

        def normalized_features(
            self,
            recurrent: Tensor,
            parent_logits: Tensor,
            executed_action_context: Tensor | None = None,
            visual_features: Tensor | None = None,
            player_position_features: Tensor | None = None,
        ) -> Tensor:
            raw = self.raw_features(
                recurrent,
                parent_logits,
                executed_action_context,
                visual_features,
                player_position_features,
            )
            return (raw - self.feature_mean) / self.feature_scale

        def decode_action_logits(
            self,
            action_values: Tensor,
            parent_logits: Tensor,
        ) -> Tensor:
            return decode_residual_action_logits(
                action_values,
                parent_logits,
                self.config.action_logit_mode,
            )

        def forward(
            self,
            recurrent: Tensor,
            parent_logits: Tensor,
            hidden: tuple[Tensor, ...] | None = None,
            *,
            executed_action_context: Tensor | None = None,
            visual_features: Tensor | None = None,
            player_position_features: Tensor | None = None,
        ) -> tuple[Tensor, Tensor, tuple[Tensor, ...]]:
            gates, actions, _collision, _margin, next_hidden = (
                self.forward_with_safety(
                    recurrent,
                    parent_logits,
                    hidden,
                    executed_action_context=executed_action_context,
                    visual_features=visual_features,
                    player_position_features=player_position_features,
                )
            )
            return gates, actions, next_hidden

        def forward_with_safety(
            self,
            recurrent: Tensor,
            parent_logits: Tensor,
            hidden: tuple[Tensor, ...] | None = None,
            *,
            executed_action_context: Tensor | None = None,
            visual_features: Tensor | None = None,
            player_position_features: Tensor | None = None,
        ) -> tuple[
            Tensor,
            Tensor,
            Tensor | None,
            Tensor | None,
            tuple[Tensor, ...],
        ]:
            gates, actions, collision, margins, _physical, next_hidden = (
                self.forward_with_all_safety(
                    recurrent,
                    parent_logits,
                    hidden,
                    executed_action_context=executed_action_context,
                    visual_features=visual_features,
                    player_position_features=player_position_features,
                )
            )
            return gates, actions, collision, margins, next_hidden

        def forward_with_all_safety(
            self,
            recurrent: Tensor,
            parent_logits: Tensor,
            hidden: tuple[Tensor, ...] | None = None,
            *,
            executed_action_context: Tensor | None = None,
            visual_features: Tensor | None = None,
            player_position_features: Tensor | None = None,
        ) -> tuple[
            Tensor,
            Tensor,
            Tensor | None,
            Tensor | None,
            Tensor | None,
            tuple[Tensor, ...],
        ]:
            (
                gates,
                actions,
                collision,
                margins,
                physical,
                _membership,
                next_hidden,
            ) = self.forward_with_all_safety_and_membership(
                recurrent,
                parent_logits,
                hidden,
                executed_action_context=executed_action_context,
                visual_features=visual_features,
                player_position_features=player_position_features,
            )
            return gates, actions, collision, margins, physical, next_hidden

        def forward_with_all_safety_and_membership(
            self,
            recurrent: Tensor,
            parent_logits: Tensor,
            hidden: tuple[Tensor, ...] | None = None,
            *,
            executed_action_context: Tensor | None = None,
            visual_features: Tensor | None = None,
            player_position_features: Tensor | None = None,
        ) -> tuple[
            Tensor,
            Tensor,
            Tensor | None,
            Tensor | None,
            Tensor | None,
            Tensor | None,
            tuple[Tensor, ...],
        ]:
            features = self.normalized_features(
                recurrent,
                parent_logits,
                executed_action_context,
                visual_features,
                player_position_features,
            )
            if hidden is not None and len(hidden) != len(self.members):
                raise ValueError("residual hidden state does not match ensemble size")
            outputs = [
                member.forward_with_all_safety_and_membership(
                    features,
                    None if hidden is None else hidden[index],
                )
                for index, member in enumerate(self.members)
            ]
            collision = [
                value
                for _gate, _actions, value, _margin, _physical, _membership,
                _hidden in outputs
            ]
            margins = [
                value
                for _gate, _actions, _collision, value, _physical, _membership,
                _hidden in outputs
            ]
            physical = [
                value
                for _gate, _actions, _collision, _margin, value, _membership,
                _hidden in outputs
            ]
            membership = [
                value
                for _gate, _actions, _collision, _margin, _physical, value,
                _hidden in outputs
            ]
            action_values = torch.stack([
                actions
                for _gate, actions, _collision, _margin, _physical, _membership,
                _hidden
                in outputs
            ], dim=0)
            action_logits = self.decode_action_logits(action_values, parent_logits)
            if any(value is None for value in collision) != all(
                value is None for value in collision
            ):
                raise RuntimeError("residual collision heads are inconsistent")
            if any(value is None for value in margins) != all(
                value is None for value in margins
            ):
                raise RuntimeError("residual margin heads are inconsistent")
            if any(value is None for value in physical) != all(
                value is None for value in physical
            ):
                raise RuntimeError("residual physical danger heads are inconsistent")
            if any(value is None for value in membership) != all(
                value is None for value in membership
            ):
                raise RuntimeError("residual membership heads are inconsistent")
            if self.config.per_action_membership_confidence != (
                membership[0] is not None
            ):
                raise RuntimeError(
                    "residual membership heads do not match adapter config"
                )
            return (
                torch.stack([
                    gate
                    for gate, _actions, _collision, _margin, _physical,
                    _membership, _hidden
                    in outputs
                ], dim=0),
                action_logits,
                (
                    None
                    if collision[0] is None else
                    torch.stack(collision, dim=0)  # type: ignore[arg-type]
                ),
                (
                    None
                    if margins[0] is None else
                    torch.stack(margins, dim=0)  # type: ignore[arg-type]
                ),
                (
                    None
                    if physical[0] is None else
                    torch.stack(physical, dim=0)  # type: ignore[arg-type]
                ),
                (
                    None
                    if membership[0] is None else
                    torch.stack(membership, dim=0)  # type: ignore[arg-type]
                ),
                tuple(
                    next_hidden
                    for _gate, _actions, _collision, _margin, _physical,
                    _membership, next_hidden
                    in outputs
                ),
            )


    class ResidualPolicyWrapper(nn.Module):
        """Apply a learned correction only when every configured gate agrees."""

        def __init__(
            self,
            parent: nn.Module,
            adapter: ResidualCorrectionAdapter,
            runtime_config: ResidualRuntimeConfig,
        ) -> None:
            super().__init__()
            parent_config = getattr(parent, "config", None)
            if parent_config is None:
                raise ValueError("residual parent must declare a policy config")
            if int(getattr(parent_config, "recurrent_size", 0)) != adapter.config.recurrent_size:
                raise ValueError("residual parent recurrent size does not match adapter")
            if int(getattr(parent_config, "action_count", 0)) != adapter.config.action_count:
                raise ValueError("residual parent action count does not match adapter")
            expected_visual_size = int(getattr(parent_config, "feature_size", 0)) * 2
            if adapter.config.visual_latent_size not in {0, expected_visual_size}:
                raise ValueError("residual visual latent size does not match parent")
            self.parent = parent
            self.adapter = adapter
            self.runtime_config = runtime_config
            if runtime_config.critic_enabled:
                if (
                    runtime_config.critic_signal == "physical_danger"
                    and not adapter.config.per_action_physical_danger
                ):
                    raise ValueError(
                        "physical critic runtime requires physical danger heads"
                    )
                if (
                    runtime_config.critic_signal == "collision_margin"
                    and not adapter.config.per_action_safety_critic
                ):
                    raise ValueError("critic runtime requires per-action safety heads")
            self.config = parent_config
            for name in (
                "scenario_vocabulary",
                "previous_action_size",
                "previous_action_offset",
            ):
                if hasattr(parent, name):
                    setattr(self, name, getattr(parent, name))
            self._previous_executed_action: int | None = None
            self._consecutive_executed_decisions = 0
            self.reset_runtime_stats()

        def reset_runtime_stats(self) -> None:
            self._residual_decisions = 0
            self._residual_overrides = 0
            self._residual_override_actions = [0] * self.adapter.config.action_count
            self._residual_gate_probability_sum = 0.0
            self._residual_gate_probability_maximum = 0.0
            self._residual_legacy_overrides = 0
            self._residual_critic_overrides = 0
            self._residual_override_overlap = 0
            self._residual_legacy_requests = 0
            self._residual_critic_requests = 0
            self._residual_future_onset_requests = 0
            self._residual_future_onset_overrides = 0
            self._residual_critic_future_overlap = 0
            self._residual_candidate_safety_vetoes = 0
            self._residual_previous_candidates = 0
            self._residual_previous_candidate_overrides = 0
            self._residual_critic_parent_not_dangerous = 0
            self._residual_critic_candidate_not_safe = 0

        def reset_runtime_state(self) -> None:
            self.reset_runtime_stats()
            self._previous_executed_action: int | None = None
            self._consecutive_executed_decisions = 0

        def commit_executed_action(self, action: Any, *, frames: int) -> None:
            if isinstance(frames, bool) or not isinstance(frames, int) or frames <= 0:
                raise ValueError("executed action frames must be a positive integer")
            action_id = getattr(action, "discrete", None)
            if (
                isinstance(action_id, bool)
                or not isinstance(action_id, int)
                or not 0 <= action_id < self.adapter.config.action_count
            ):
                raise ValueError("executed action must use the residual action vocabulary")
            if action_id == self._previous_executed_action:
                self._consecutive_executed_decisions += 1
            else:
                self._previous_executed_action = action_id
                self._consecutive_executed_decisions = 1

        def _executed_action_context(self, reference: Tensor) -> Tensor | None:
            if not self.adapter.config.executed_action_context:
                return None
            context = reference.new_zeros((
                *reference.shape[:-1],
                self.adapter.config.executed_action_context_size,
            ))
            if self._previous_executed_action is not None:
                context[..., self._previous_executed_action] = 1.0
                context[..., -2] = 1.0
                context[..., -1] = math.log1p(
                    self._consecutive_executed_decisions,
                )
            return context

        def residual_runtime_stats(self) -> dict[str, Any]:
            decisions = self._residual_decisions
            result = {
                "decisions": decisions,
                "overrides": self._residual_overrides,
                "override_rate": self._residual_overrides / decisions if decisions else 0.0,
                "override_action_counts": {
                    str(index): count
                    for index, count in enumerate(self._residual_override_actions)
                    if count
                },
                "mean_gate_probability": (
                    self._residual_gate_probability_sum / decisions if decisions else 0.0
                ),
                "maximum_gate_probability": self._residual_gate_probability_maximum,
                "runtime_config": asdict(self.runtime_config),
            }
            if (
                self.adapter.config.per_action_safety_critic
                or self.adapter.config.per_action_physical_danger
            ):
                result["safety_critic"] = {
                    "legacy_overrides": self._residual_legacy_overrides,
                    "critic_overrides": self._residual_critic_overrides,
                    "override_overlap": self._residual_override_overlap,
                    "legacy_requests": self._residual_legacy_requests,
                    "critic_requests": self._residual_critic_requests,
                    "future_onset_requests": (
                        self._residual_future_onset_requests
                    ),
                    "future_onset_overrides": (
                        self._residual_future_onset_overrides
                    ),
                    "critic_future_overlap": (
                        self._residual_critic_future_overlap
                    ),
                    "candidate_safety_vetoes": (
                        self._residual_candidate_safety_vetoes
                    ),
                    "safe_previous_candidates": self._residual_previous_candidates,
                    "safe_previous_overrides": (
                        self._residual_previous_candidate_overrides
                    ),
                    "parent_not_dangerous": (
                        self._residual_critic_parent_not_dangerous
                    ),
                    "candidate_not_safe": self._residual_critic_candidate_not_safe,
                }
            return result

        def _apply_residual(
            self,
            parent_logits: Tensor,
            recurrent: Tensor,
            visual_features: Tensor | None = None,
            hidden: tuple[Tensor, ...] | None = None,
            *,
            player_position_features: Tensor | None = None,
        ) -> tuple[Tensor, tuple[Tensor, ...]]:
            (
                gate_logits,
                action_logits,
                collision_logits,
                normalized_minimum_margins,
                physical_danger_logits,
                membership_logits,
                next_hidden,
            ) = self.adapter.forward_with_all_safety_and_membership(
                recurrent,
                parent_logits,
                hidden,
                executed_action_context=self._executed_action_context(parent_logits),
                visual_features=visual_features,
                player_position_features=player_position_features,
            )
            gate_probabilities = finite_sigmoid(gate_logits)
            if membership_logits is None:
                action_probabilities, action_member_finite = (
                    finite_action_probabilities(
                        action_logits,
                        self.adapter.config.action_logit_mode,
                    )
                )
                action_summary = ensemble_action_summary(
                    action_probabilities,
                    action_member_finite,
                )
            else:
                action_probabilities, action_member_finite = (
                    finite_action_probabilities(action_logits)
                )
                membership_probabilities, membership_member_finite = (
                    finite_action_probabilities(
                        membership_logits,
                        "certified_membership",
                    )
                )
                action_summary = ensemble_action_summary(
                    action_probabilities,
                    action_member_finite,
                    membership_probabilities,
                    membership_member_finite,
                )
            action_all_members_finite = action_summary[
                "action_all_members_finite"
            ]
            mean_gate = gate_probabilities.mean(dim=0)
            minimum_gate = gate_probabilities.amin(dim=0)
            correction_actions = action_summary["candidates"]
            correction_confidence = action_summary["action_confidence"]
            agreement = action_summary["agreement"].to(mean_gate.dtype)
            parent_actions = parent_logits.argmax(dim=-1)
            collision_probabilities = (
                None
                if collision_logits is None else
                finite_sigmoid(collision_logits)
            )
            minimum_margins = (
                None
                if normalized_minimum_margins is None else
                normalized_minimum_margins * 16.0
            )
            physical_danger_probabilities = (
                None
                if physical_danger_logits is None else
                finite_sigmoid(physical_danger_logits)
            )
            previous_actions = parent_actions.new_full(
                parent_actions.shape,
                -1 if self._previous_executed_action is None else
                self._previous_executed_action,
            )
            future_onset = residual_future_onset_mask(
                mean_gate,
                minimum_gate,
                self.runtime_config,
            )
            selection = residual_candidate_selection(
                correction_actions=correction_actions,
                correction_confidence=correction_confidence,
                agreement=agreement,
                previous_actions=previous_actions,
                runtime_config=self.runtime_config,
                collision_probabilities=collision_probabilities,
                minimum_margins=minimum_margins,
                physical_danger_probabilities=physical_danger_probabilities,
                parent_actions=parent_actions,
                future_onset=future_onset,
            )
            correction_actions = selection["correction_actions"]
            correction_confidence = selection["correction_confidence"]
            agreement = selection["agreement"]
            masks = residual_override_masks(
                mean_gate=mean_gate,
                minimum_gate=minimum_gate,
                action_all_members_finite=action_all_members_finite,
                correction_confidence=correction_confidence,
                agreement=agreement,
                correction_actions=correction_actions,
                parent_actions=parent_actions,
                runtime_config=self.runtime_config,
                collision_probabilities=collision_probabilities,
                minimum_margins=minimum_margins,
                physical_danger_probabilities=physical_danger_probabilities,
            )
            override = masks["override"]

            effective = parent_logits.clone()
            selected_values = parent_logits.amax(dim=-1) + self.runtime_config.override_logit_margin
            current_values = effective.gather(
                -1, correction_actions.unsqueeze(-1),
            ).squeeze(-1)
            replacement = torch.where(override, selected_values, current_values)
            effective.scatter_(
                -1,
                correction_actions.unsqueeze(-1),
                replacement.unsqueeze(-1),
            )

            with torch.no_grad():
                flat_gate = mean_gate.detach().reshape(-1).cpu()
                flat_override = override.detach().reshape(-1).cpu()
                flat_actions = correction_actions.detach().reshape(-1).cpu()
                self._residual_decisions += int(flat_gate.numel())
                self._residual_overrides += int(flat_override.sum().item())
                self._residual_gate_probability_sum += float(flat_gate.sum().item())
                if flat_gate.numel():
                    self._residual_gate_probability_maximum = max(
                        self._residual_gate_probability_maximum,
                        float(flat_gate.max().item()),
                    )
                for action in flat_actions[flat_override].tolist():
                    self._residual_override_actions[int(action)] += 1
                if self.runtime_config.critic_enabled:
                    flat_legacy = masks["legacy_accepted"].detach().reshape(-1).cpu()
                    flat_critic = masks["critic_accepted"].detach().reshape(-1).cpu()
                    flat_legacy_request = masks["legacy_request"].detach().reshape(-1).cpu()
                    flat_critic_request = masks["critic_request"].detach().reshape(-1).cpu()
                    flat_future_request = masks["future_onset_request"].detach().reshape(
                        -1
                    ).cpu()
                    flat_future_accepted = masks["future_onset_accepted"].detach().reshape(
                        -1
                    ).cpu()
                    flat_veto = masks["candidate_safety_veto"].detach().reshape(-1).cpu()
                    flat_previous = selection["used_previous"].detach().reshape(-1).cpu()
                    flat_parent_danger = masks["parent_danger"].detach().reshape(-1).cpu()
                    flat_candidate_safe = masks["candidate_safe"].detach().reshape(-1).cpu()
                    self._residual_legacy_overrides += int(flat_legacy.sum().item())
                    self._residual_critic_overrides += int(flat_critic.sum().item())
                    self._residual_override_overlap += int(
                        (flat_legacy & flat_critic).sum().item()
                    )
                    self._residual_legacy_requests += int(
                        flat_legacy_request.sum().item()
                    )
                    self._residual_critic_requests += int(
                        flat_critic_request.sum().item()
                    )
                    self._residual_future_onset_requests += int(
                        flat_future_request.sum().item()
                    )
                    self._residual_future_onset_overrides += int(
                        flat_future_accepted.sum().item()
                    )
                    self._residual_critic_future_overlap += int(
                        (flat_critic & flat_future_accepted).sum().item()
                    )
                    self._residual_candidate_safety_vetoes += int(
                        flat_veto.sum().item()
                    )
                    self._residual_previous_candidates += int(
                        flat_previous.sum().item()
                    )
                    self._residual_previous_candidate_overrides += int(
                        (flat_previous & flat_override).sum().item()
                    )
                    self._residual_critic_parent_not_dangerous += int(
                        (~flat_parent_danger).sum().item()
                    )
                    self._residual_critic_candidate_not_safe += int(
                        (~flat_candidate_safe).sum().item()
                    )
            return effective, next_hidden

        def forward_with_recurrent(
            self,
            global_frames: Tensor,
            local_frames: Tensor,
            memory: Tensor | None = None,
            proficiency: Tensor | None = None,
            hidden: Tensor | None = None,
        ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
            if hidden is None:
                parent_hidden = None
                adapter_hidden = None
            elif (
                isinstance(hidden, tuple)
                and len(hidden) == 2
                and isinstance(hidden[1], tuple)
            ):
                parent_hidden, adapter_hidden = hidden
            else:
                raise ValueError("residual policy hidden state is invalid")
            if self.adapter.config.visual_latent_size:
                (
                    parent_logits,
                    risk,
                    parent_hidden,
                    recurrent,
                    visual_features,
                ) = self.parent.forward_with_visual_features(
                    global_frames,
                    local_frames,
                    memory,
                    proficiency,
                    parent_hidden,
                )
            else:
                parent_logits, risk, parent_hidden, recurrent = (
                    self.parent.forward_with_recurrent(
                        global_frames,
                        local_frames,
                        memory,
                        proficiency,
                        parent_hidden,
                    )
                )
                visual_features = None
            player_position_features = (
                semantic_player_position_features(global_frames)
                if self.adapter.config.semantic_player_position else
                None
            )
            logits, adapter_hidden = self._apply_residual(
                parent_logits,
                recurrent,
                visual_features,
                adapter_hidden,
                player_position_features=player_position_features,
            )
            return logits, risk, (parent_hidden, adapter_hidden), recurrent

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

    class ResidualCorrectionAdapter:  # type: ignore[no-redef]
        def __init__(self, config: ResidualAdapterConfig) -> None:
            raise RuntimeError("PyTorch is required for residual adapters")


    class ResidualPolicyWrapper:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("PyTorch is required for residual adapters")


def save_residual_adapter(
    adapter: ResidualCorrectionAdapter,
    path: str | Path,
    *,
    parent_checkpoint: str | Path,
    parent_policy_config: Mapping[str, Any],
    runtime_config: ResidualRuntimeConfig,
    training_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    if torch is None:
        raise RuntimeError("PyTorch is required for residual adapters")
    from .provenance import file_sha256

    parent_path = Path(parent_checkpoint)
    if not parent_path.is_file():
        raise FileNotFoundError(f"parent checkpoint does not exist: {parent_path}")
    output = Path(path)
    if output.resolve() == parent_path.resolve():
        raise ValueError("residual adapter cannot overwrite its parent checkpoint")
    if runtime_config.critic_enabled:
        if (
            runtime_config.critic_signal == "physical_danger"
            and not adapter.config.per_action_physical_danger
        ):
            raise ValueError("physical critic runtime requires physical danger heads")
        if (
            runtime_config.critic_signal == "collision_margin"
            and not adapter.config.per_action_safety_critic
        ):
            raise ValueError("critic runtime requires per-action safety heads")
    if runtime_config.future_onset_gate_enabled and not (
        adapter.config.per_action_physical_danger
        and adapter.config.visual_latent_size > 0
    ):
        raise ValueError("future-onset artifacts require visual physical safety heads")
    if (
        adapter.config.per_action_physical_danger
        and runtime_config.critic_signal != "physical_danger"
    ):
        raise ValueError("physical danger artifacts require physical danger runtime")
    semantic_action_logits = adapter.config.action_logit_mode != "absolute"
    if semantic_action_logits and not (
        runtime_config.future_onset_gate_enabled
        and runtime_config.critic_enabled
        and runtime_config.critic_signal == "physical_danger"
        and adapter.config.per_action_safety_critic
        and adapter.config.per_action_physical_danger
        and adapter.config.visual_latent_size > 0
    ):
        raise ValueError(
            "non-absolute action artifacts require a future-onset direct "
            "visual physical runtime"
        )
    dual_head_confidence = adapter.config.per_action_membership_confidence
    payload = {
        "version": (
            7
            if dual_head_confidence else
            6
            if semantic_action_logits else
            5
            if runtime_config.future_onset_gate_enabled else
            4
            if adapter.config.per_action_physical_danger else
            3
            if adapter.config.per_action_safety_critic else
            2
            if adapter.config.executed_action_context else
            1
        ),
        "kind": "frozen_parent_residual_correction_adapter",
        "parent_checkpoint": str(parent_path),
        "parent_checkpoint_sha256": file_sha256(parent_path),
        "parent_policy_config": dict(parent_policy_config),
        "adapter_config": asdict(adapter.config),
        "runtime_config": asdict(runtime_config),
        "state_dict": adapter.state_dict(),
        "training_metadata": dict(training_metadata),
    }
    if runtime_config.future_onset_gate_enabled:
        payload["gate_semantics"] = dict(FUTURE_ONSET_GATE_SEMANTICS)
    if dual_head_confidence:
        payload["selector_logit_semantics"] = _selector_logit_semantics(
            adapter.config.action_logit_mode
        )
        payload["membership_confidence_semantics"] = dict(
            MEMBERSHIP_CONFIDENCE_SEMANTICS
        )
    elif semantic_action_logits:
        payload["action_logit_semantics"] = _action_logit_semantics(
            adapter.config.action_logit_mode
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    return {
        "adapter": str(output),
        "adapter_sha256": file_sha256(output),
        "parent_checkpoint": str(parent_path),
        "parent_checkpoint_sha256": payload["parent_checkpoint_sha256"],
        "adapter_config": payload["adapter_config"],
        "runtime_config": payload["runtime_config"],
    }


def load_residual_adapter(
    parent: Any,
    adapter_path: str | Path,
    *,
    parent_checkpoint: str | Path,
    device: str = "cpu",
) -> tuple[ResidualPolicyWrapper, dict[str, Any]]:
    if torch is None:
        raise RuntimeError("PyTorch is required for residual adapters")
    from .provenance import file_sha256

    path = Path(adapter_path)
    if not path.is_file():
        raise FileNotFoundError(f"residual adapter does not exist: {path}")
    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("residual adapter artifact must be a mapping")
    version = payload.get("version")
    if version not in {1, 2, 3, 4, 5, 6, 7} or payload.get("kind") != (
        "frozen_parent_residual_correction_adapter"
    ):
        raise ValueError("unsupported residual adapter artifact")
    actual_parent_sha256 = file_sha256(parent_checkpoint)
    if payload.get("parent_checkpoint_sha256") != actual_parent_sha256:
        raise ValueError("residual adapter parent checkpoint hash does not match")
    expected_policy_config = payload.get("parent_policy_config")
    actual_policy_config = asdict(parent.config)
    if expected_policy_config != actual_policy_config:
        raise ValueError("residual adapter parent policy config does not match")
    adapter_values = payload.get("adapter_config")
    runtime_values = payload.get("runtime_config")
    state_dict = payload.get("state_dict")
    if not isinstance(adapter_values, Mapping) or not isinstance(runtime_values, Mapping):
        raise ValueError("residual adapter config metadata is invalid")
    if not isinstance(state_dict, Mapping):
        raise ValueError("residual adapter state dictionary is invalid")
    if payload.get("version") == 1 and adapter_values.get(
        "executed_action_context", False,
    ):
        raise ValueError("version 1 residual adapters cannot declare action context")
    critic_declared = bool(adapter_values.get("per_action_safety_critic", False))
    physical_declared = bool(
        adapter_values.get("per_action_physical_danger", False)
    )
    future_onset_declared = bool(
        runtime_values.get("future_onset_gate_enabled", False)
    )
    visual_latent_size = adapter_values.get("visual_latent_size", 0)
    action_logit_mode = adapter_values.get("action_logit_mode", "absolute")
    membership_confidence_declared = adapter_values.get(
        "per_action_membership_confidence",
        False,
    )
    if version in {1, 2, 3, 4, 5, 6} and (
        membership_confidence_declared is not False
    ):
        raise ValueError(
            "version 1-6 residual adapters cannot declare membership confidence"
        )
    if version == 7 and membership_confidence_declared is not True:
        raise ValueError(
            "version 7 residual adapters require membership confidence"
        )
    if version in {1, 2, 3, 4, 5} and action_logit_mode != "absolute":
        raise ValueError("version 1-5 residual adapters require absolute action logits")
    if version == 6 and action_logit_mode not in ACTION_LOGIT_MODES[1:]:
        raise ValueError(
            "version 6 residual adapters require explicit non-absolute action semantics"
        )
    if version in {1, 2, 3, 4, 5} and "action_logit_semantics" in payload:
        raise ValueError("version 1-5 residual adapters cannot declare action semantics")
    if version == 6 and payload.get("action_logit_semantics") != (
        _action_logit_semantics(action_logit_mode)
    ):
        raise ValueError("version 6 residual adapter action semantics do not match")
    if version in {1, 2, 3, 4, 5, 6} and (
        "selector_logit_semantics" in payload
        or "membership_confidence_semantics" in payload
    ):
        raise ValueError(
            "version 1-6 residual adapters cannot declare dual-head semantics"
        )
    if version == 7:
        if "action_logit_semantics" in payload:
            raise ValueError(
                "version 7 residual adapters require separate selector semantics"
            )
        if action_logit_mode not in {
            "parent_residual_joint",
            "parent_residual_factorized",
        }:
            raise ValueError(
                "version 7 residual adapters require parent-residual selectors"
            )
        if payload.get("selector_logit_semantics") != (
            _selector_logit_semantics(action_logit_mode)
        ):
            raise ValueError(
                "version 7 residual adapter selector semantics do not match"
            )
        if payload.get("membership_confidence_semantics") != (
            MEMBERSHIP_CONFIDENCE_SEMANTICS
        ):
            raise ValueError(
                "version 7 residual adapter membership semantics do not match"
            )
    if version in {1, 2} and critic_declared:
        raise ValueError("legacy residual adapters cannot declare a safety critic")
    if version == 3 and not critic_declared:
        raise ValueError("version 3 residual adapters must declare a safety critic")
    if version in {1, 2, 3} and physical_declared:
        raise ValueError("legacy residual adapters cannot declare physical danger heads")
    if version in {1, 2, 3} and visual_latent_size != 0:
        raise ValueError("legacy residual adapters cannot declare a visual latent")
    if version in {4, 5, 6, 7} and not (
        critic_declared
        and physical_declared
        and isinstance(visual_latent_size, int)
        and not isinstance(visual_latent_size, bool)
        and visual_latent_size > 0
    ):
        raise ValueError(
            "version 4/5/6/7 residual adapters require visual physical safety heads"
        )
    if version in {1, 2, 3, 4} and future_onset_declared:
        raise ValueError("legacy residual adapters cannot enable future-onset gates")
    if version in {5, 6, 7} and not future_onset_declared:
        raise ValueError("version 5/6/7 residual adapters require future-onset gates")
    if version in {5, 6, 7} and payload.get("gate_semantics") != (
        FUTURE_ONSET_GATE_SEMANTICS
    ):
        raise ValueError(
            "version 5/6/7 residual adapter gate semantics do not match"
        )
    if version in {1, 2, 3, 4} and payload.get(
        "gate_semantics"
    ) == FUTURE_ONSET_GATE_SEMANTICS:
        raise ValueError("legacy residual adapters cannot declare v5 gate semantics")
    if version in {1, 2} and runtime_values.get(
        "critic_enabled", False,
    ):
        raise ValueError("legacy residual adapters cannot enable a safety critic")
    if version in {1, 2, 3} and runtime_values.get(
        "critic_signal", "collision_margin",
    ) != "collision_margin":
        raise ValueError("legacy residual adapters require collision-margin critics")
    if version in {4, 5, 6, 7} and runtime_values.get(
        "critic_signal", "collision_margin",
    ) != "physical_danger":
        raise ValueError(
            "version 4/5/6/7 residual adapters require physical danger runtime"
        )
    if version in {6, 7} and not runtime_values.get("critic_enabled", False):
        raise ValueError("version 6/7 residual adapters require an enabled critic")
    if not all(
        isinstance(value, Tensor) and bool(torch.isfinite(value).all())
        for value in state_dict.values()
    ):
        raise ValueError("residual adapter state dictionary must be finite")
    adapter = ResidualCorrectionAdapter(
        ResidualAdapterConfig(**dict(adapter_values)),
    ).to(device)
    adapter.load_state_dict(state_dict, strict=True)
    adapter.eval()
    wrapper = ResidualPolicyWrapper(
        parent,
        adapter,
        ResidualRuntimeConfig(**dict(runtime_values)),
    ).to(device)
    wrapper.eval()
    metadata = dict(payload)
    metadata.pop("state_dict", None)
    metadata.update({
        "adapter": str(path),
        "adapter_sha256": file_sha256(path),
        "verified_parent_checkpoint": str(parent_checkpoint),
        "verified_parent_checkpoint_sha256": actual_parent_sha256,
    })
    return wrapper, metadata


__all__ = [
    "ACTION_LOGIT_MODES",
    "SEMANTIC_PLAYER_POSITION_SIZE",
    "ResidualAdapterConfig",
    "ResidualCorrectionAdapter",
    "ResidualPolicyWrapper",
    "ResidualRuntimeConfig",
    "decode_residual_action_logits",
    "ensemble_action_summary",
    "finite_action_probabilities",
    "finite_sigmoid",
    "load_residual_adapter",
    "residual_candidate_selection",
    "residual_future_onset_mask",
    "residual_override_masks",
    "save_residual_adapter",
    "semantic_player_position_features",
    "FUTURE_ONSET_GATE_SEMANTICS",
]
