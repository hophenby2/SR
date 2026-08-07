"""Training-only paired CV for preferred-action objective variants.

The experiment compares exact, independently certified equivalence-set, rank,
and conditional certified previous-action continuity objectives. All source
selection happens from a fixed training whitelist before any path or hash field
is inspected. No model or deployment artifact is written.
"""

from __future__ import annotations

import argparse
import copy
from collections import Counter
from dataclasses import asdict, dataclass, fields
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import torch

from stg_lab.provenance import file_sha256
from stg_lab.residual_adapter import ResidualAdapterConfig, ResidualCorrectionAdapter
from stg_lab.training import load_checkpoint

if __package__:
    from .train_temporal_residual_adapter import (
        ACTION_BRANCH_MODULE_NAMES,
        EARLY_ONSET_MINIMUM_LEAD_DECISIONS,
        FUTURE_ONSET_HORIZON_DECISIONS,
        GRADIENT_CLIP_MAX_NORM,
        GLOBAL_GRADIENT_CLIP_SEMANTICS,
        PREFERRED_ACTION_TIEBREAK_SEMANTICS,
        PREFERRED_ACTION_UNIFORM_SOFT_TARGET_SEMANTICS,
        SEPARATE_ACTION_GRADIENT_CLIP_SEMANTICS,
        EpisodeFeatures,
        _calibrate,
        _collision_positive_weights,
        _future_onset_calibration_diagnostics,
        _gradient_clip_parameter_groups,
        _load_episode,
        _metrics,
        _normalize,
        _offline_deployment_eligible,
        _physical_danger_positive_weights,
        _predict_episode,
        _preferred_action_tiebreak_mask,
        _train_member,
    )
else:  # pragma: no cover - exercised by the real script invocation
    from train_temporal_residual_adapter import (
        ACTION_BRANCH_MODULE_NAMES,
        EARLY_ONSET_MINIMUM_LEAD_DECISIONS,
        FUTURE_ONSET_HORIZON_DECISIONS,
        GRADIENT_CLIP_MAX_NORM,
        GLOBAL_GRADIENT_CLIP_SEMANTICS,
        PREFERRED_ACTION_TIEBREAK_SEMANTICS,
        PREFERRED_ACTION_UNIFORM_SOFT_TARGET_SEMANTICS,
        SEPARATE_ACTION_GRADIENT_CLIP_SEMANTICS,
        EpisodeFeatures,
        _calibrate,
        _collision_positive_weights,
        _future_onset_calibration_diagnostics,
        _gradient_clip_parameter_groups,
        _load_episode,
        _metrics,
        _normalize,
        _offline_deployment_eligible,
        _physical_danger_positive_weights,
        _predict_episode,
        _preferred_action_tiebreak_mask,
        _train_member,
    )


TRAINING_SEEDS = (
    20260730,
    20260731,
    20260732,
    10294,
    10295,
    20260733,
    10296,
    10298,
    10292,
    10297,
    10301,
    10302,
    10303,
    10304,
    10305,
)
PROHIBITED_SOURCE_SEEDS = frozenset((10306, 10307, 10308, 10309, 10310))
TRAINING_SEED_SET = frozenset(TRAINING_SEEDS)

DEFAULT_FAILURE = Path(
    "artifacts/policy-humanlike-highres-okuu3-v81-internal-joint-pos-separate-"
    "calibration-failure.json"
)
DEFAULT_PARENT = Path(
    "artifacts/policy-humanlike-highres-okuu3-v54-v37-onpolicy-gain025-"
    "minedit20-top1w2-kl20-ft60.pt"
)
DEFAULT_OUTPUT = Path(
    "artifacts/policy-humanlike-highres-okuu3-preferred-objective-"
    "uniform-soft-target-cv-e6.json"
)

EXPECTED_V81_CONFIG = {
    "recurrent_size": 512,
    "action_count": 18,
    "hidden_size": 64,
    "ensemble_size": 3,
    "executed_action_context": True,
    "per_action_safety_critic": True,
    "visual_latent_size": 256,
    "per_action_physical_danger": True,
    "action_logit_mode": "parent_residual_joint",
    "semantic_player_position": True,
    "separate_action_recurrent": True,
}

LABEL_CONFIG = {
    "safe_regret": 1.0,
    "minimum_parent_margin": 8.0,
    "minimum_margin_gain": 1.0,
    "predecessor_decisions": FUTURE_ONSET_HORIZON_DECISIONS,
    "future_onset_gate": True,
}

TRAINING_CONFIG = {
    "learning_rate": 3e-4,
    "weight_decay": 1e-3,
    "chunk_length": 128,
    "gate_positive_weight": 8.0,
    "action_loss_weight": 0.0,
    "preferred_action_loss_weight": 12.0,
    "safety_candidate_loss_weight": 0.0,
    "parent_copy_weight": 0.1,
    "collision_loss_weight": 0.0,
    "minimum_margin_loss_weight": 0.0,
    "physical_danger_loss_weight": 8.0,
    "maximum_collision_positive_weight": 24.0,
    "maximum_physical_danger_positive_weight": 24.0,
    "all_collision_row_weight": 0.25,
    "episode_bootstrap": False,
}

ARM_NAMES = (
    "exact",
    "equivalence",
    "equivalence_top1_rank",
    "equivalence_weak_tiebreak",
    "equivalence_uniform_soft_target",
)
RANK_OBJECTIVE_CONFIG = {
    "preferred_action_rank_loss_weight": 12.0,
    "preferred_action_rank_margin": 1.0,
}
TIEBREAK_OBJECTIVE_CONFIG = {
    "schema": PREFERRED_ACTION_TIEBREAK_SEMANTICS,
    "preferred_action_tiebreak_loss_weight": 3.0,
}
UNIFORM_SOFT_TARGET_OBJECTIVE_CONFIG = {
    "schema": PREFERRED_ACTION_UNIFORM_SOFT_TARGET_SEMANTICS,
    "preferred_action_uniform_loss_weight": 3.0,
}
UNIFORM_SOFT_TARGET_SCREEN = {
    "screening_epochs": 6,
    "expected_targets": 359,
    "expected_baseline_equivalent_top1": 60,
    "expected_baseline_direction_correct": 79,
    "expected_baseline_speed_correct": 194,
    "minimum_candidate_equivalent_top1": 68,
    "minimum_improved_audit_folds": 2,
    "minimum_candidate_direction_correct": 72,
    "minimum_candidate_speed_correct": 187,
}

RAW_RATE_FIELDS = (
    "exact_top1_rate",
    "equivalent_top1_rate",
    "candidate_changed_parent_rate",
    "direction_accuracy",
    "speed_accuracy",
    "tiebreak_eligible_previous_top1_rate",
    "tiebreak_eligible_exact_top1_rate",
    "tiebreak_eligible_equivalent_top1_rate",
    "tiebreak_eligible_direction_accuracy",
    "tiebreak_eligible_speed_accuracy",
)


@dataclass(frozen=True, slots=True)
class Fold:
    index: int
    fit_seeds: tuple[int, ...]
    calibration_seeds: tuple[int, ...]
    audit_seeds: tuple[int, ...]


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            mode="w",
            encoding="utf-8",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(dict(payload), handle, indent=2, allow_nan=False)
            handle.write("\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _validate_output_path(output: Path, protected_inputs: Sequence[Path]) -> None:
    resolved = output.resolve()
    collisions = [path for path in protected_inputs if path.resolve() == resolved]
    if collisions:
        raise ValueError(f"output must not overwrite protected input: {collisions[0]}")


def _required_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _required_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a nonempty string")
    return value


def _select_training_inventory(
    failure: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    """Filter solely by seed/role before touching path-bearing fields."""

    raw_inventory = failure.get("source_inventory")
    if not isinstance(raw_inventory, list):
        raise ValueError("failure diagnostics have no source inventory")
    selected: dict[int, Mapping[str, Any]] = {}
    for index, raw in enumerate(raw_inventory):
        if not isinstance(raw, Mapping):
            raise ValueError(f"source_inventory[{index}] must be an object")
        seed = _required_int(raw.get("seed"), field=f"source_inventory[{index}].seed")
        role = raw.get("role")
        if not isinstance(role, str):
            raise ValueError(f"source_inventory[{index}].role must be a string")
        if seed in PROHIBITED_SOURCE_SEEDS and role == "training":
            raise ValueError(f"prohibited source seed {seed} is marked training")
        if seed in TRAINING_SEED_SET:
            if role != "training":
                raise ValueError(f"allowed seed {seed} must have role='training'")
            if seed in selected:
                raise ValueError(f"duplicate allowed source seed: {seed}")
            # Keep the mapping opaque until the complete whitelist is verified.
            selected[seed] = raw
        elif role == "training":
            raise ValueError(f"unexpected training source seed: {seed}")
    missing = TRAINING_SEED_SET - selected.keys()
    if missing:
        raise ValueError(f"training source whitelist is incomplete: {sorted(missing)}")
    return [selected[seed] for seed in TRAINING_SEEDS]


def _verify_training_sources(
    inventory: Sequence[Mapping[str, Any]],
) -> tuple[list[tuple[Path, Path, Path]], list[dict[str, Any]]]:
    triplets: list[tuple[Path, Path, Path]] = []
    provenance: list[dict[str, Any]] = []
    for raw in inventory:
        seed = _required_int(raw.get("seed"), field="source seed")
        if seed not in TRAINING_SEED_SET:
            raise AssertionError("path resolution received a non-training seed")
        paths: dict[str, Path] = {}
        hashes: dict[str, str] = {}
        for kind in ("dataset", "report", "manifest"):
            path = Path(_required_string(raw.get(kind), field=f"seed {seed} {kind}"))
            declared = _required_string(
                raw.get(f"{kind}_sha256"),
                field=f"seed {seed} {kind}_sha256",
            )
            actual = file_sha256(path)
            if actual != declared:
                raise ValueError(
                    f"seed {seed} {kind} hash mismatch: declared={declared}, "
                    f"actual={actual}"
                )
            paths[kind] = path
            hashes[kind] = actual
        triplets.append((paths["dataset"], paths["report"], paths["manifest"]))
        provenance.append({
            "seed": seed,
            "role": "training",
            **{
                name: value
                for kind in ("dataset", "report", "manifest")
                for name, value in (
                    (kind, str(paths[kind])),
                    (f"{kind}_sha256", hashes[kind]),
                )
            },
            "declared_hashes_verified": True,
        })
    return triplets, provenance


def _fixed_folds() -> tuple[Fold, ...]:
    groups = (
        TRAINING_SEEDS[0:5],
        TRAINING_SEEDS[5:10],
        TRAINING_SEEDS[10:15],
    )
    folds = tuple(
        Fold(
            index=index,
            audit_seeds=tuple(groups[index]),
            calibration_seeds=tuple(groups[(index + 1) % 3][:2]),
            fit_seeds=tuple(
                (*groups[(index + 1) % 3][2:], *groups[(index + 2) % 3])
            ),
        )
        for index in range(3)
    )
    for fold in folds:
        roles = (set(fold.fit_seeds), set(fold.calibration_seeds), set(fold.audit_seeds))
        if tuple(map(len, roles)) != (8, 2, 5):
            raise AssertionError("fold role sizes are invalid")
        if roles[0] & roles[1] or roles[0] & roles[2] or roles[1] & roles[2]:
            raise AssertionError("fold roles overlap")
        if set.union(*roles) != TRAINING_SEED_SET:
            raise AssertionError("fold roles do not cover the training whitelist")
    if Counter(seed for fold in folds for seed in fold.audit_seeds) != Counter(
        {seed: 1 for seed in TRAINING_SEEDS}
    ):
        raise AssertionError("outer audit folds must cover each training seed once")
    return folds


def _clone_episode(episode: EpisodeFeatures) -> EpisodeFeatures:
    values: dict[str, Any] = {}
    for field in fields(EpisodeFeatures):
        value = getattr(episode, field.name)
        values[field.name] = value.clone() if isinstance(value, torch.Tensor) else value
    return EpisodeFeatures(**values)


def _episodes_by_seed(
    episodes: Sequence[EpisodeFeatures], seeds: Sequence[int]
) -> list[EpisodeFeatures]:
    by_seed = {episode.seed: episode for episode in episodes}
    if len(by_seed) != len(episodes):
        raise ValueError("episode seeds must be unique")
    try:
        return [by_seed[seed] for seed in seeds]
    except KeyError as error:
        raise ValueError(f"missing episode seed {error.args[0]}") from error


def _state_digest(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _states_equal(
    left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]
) -> bool:
    return left.keys() == right.keys() and all(
        torch.equal(left[name], right[name]) for name in left
    )


def _partition_action_branch_state(
    state: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    action: dict[str, torch.Tensor] = {}
    non_action: dict[str, torch.Tensor] = {}
    seen_modules: set[str] = set()
    for name, tensor in state.items():
        components = name.split(".")
        matches = set(components) & set(ACTION_BRANCH_MODULE_NAMES)
        if len(matches) > 1:
            raise AssertionError(f"ambiguous action branch state key: {name}")
        if matches:
            seen_modules.update(matches)
            action[name] = tensor
        else:
            non_action[name] = tensor
    missing = set(ACTION_BRANCH_MODULE_NAMES) - seen_modules
    if missing:
        raise AssertionError(
            f"trained state is missing action branch modules: {sorted(missing)}"
        )
    if not non_action:
        raise AssertionError("trained state has no non-action branch tensors")
    if set(action) & set(non_action) or set(action) | set(non_action) != set(state):
        raise AssertionError("action branch state partition overlaps or omits tensors")
    return action, non_action


def _assert_paired_non_action_states_equal(
    adapters: Mapping[str, ResidualCorrectionAdapter],
) -> str:
    if tuple(adapters) != ARM_NAMES:
        raise AssertionError("paired adapters are not in the declared arm order")
    states = {
        arm: _partition_action_branch_state(adapter.state_dict())[1]
        for arm, adapter in adapters.items()
    }
    baseline = states[ARM_NAMES[0]]
    baseline_digest = _state_digest(baseline)
    for arm in ARM_NAMES[1:]:
        candidate = states[arm]
        candidate_digest = _state_digest(candidate)
        if candidate_digest != baseline_digest:
            raise AssertionError(
                f"paired non-action branch digest differs for {arm}: "
                f"{candidate_digest} != {baseline_digest}"
            )
        if not _states_equal(baseline, candidate):
            raise AssertionError(
                f"paired non-action branch tensors differ for {arm}"
            )
    return baseline_digest


def _gradient_clip_group_signature(
    adapter: ResidualCorrectionAdapter,
) -> dict[str, tuple[str, ...]]:
    member = adapter.members[0]
    names = {id(parameter): name for name, parameter in member.named_parameters()}
    signature: dict[str, tuple[str, ...]] = {}
    for group, parameters in _gradient_clip_parameter_groups(member).items():
        try:
            signature[group] = tuple(
                sorted(names[id(parameter)] for parameter in parameters)
            )
        except KeyError as error:
            raise AssertionError(
                "gradient clip group contains an unnamed parameter"
            ) from error
    return signature


def _assert_paired_gradient_clip_groups(
    adapters: Mapping[str, ResidualCorrectionAdapter],
) -> dict[str, tuple[str, ...]]:
    if tuple(adapters) != ARM_NAMES:
        raise AssertionError("paired adapters are not in the declared arm order")
    signatures = {
        arm: _gradient_clip_group_signature(adapter)
        for arm, adapter in adapters.items()
    }
    baseline = signatures[ARM_NAMES[0]]
    for arm in ARM_NAMES[1:]:
        if signatures[arm] != baseline:
            raise AssertionError(f"paired gradient clip groups differ for {arm}")
    return baseline


def _exact_target_episode(episode: EpisodeFeatures) -> EpisodeFeatures:
    result = _clone_episode(episode)
    positive = result.gate_valid & (result.gate_targets > 0.0)
    preferred = result.preferred_actions
    if bool(positive.any()) and not bool(
        ((preferred[positive] >= 0) & (preferred[positive] < 18)).all()
    ):
        raise ValueError(f"seed {result.seed} has a positive row without a preferred action")
    exact = result.preferred_action_set.clone()
    exact[positive] = False
    if bool(positive.any()):
        rows = torch.nonzero(positive, as_tuple=False).flatten()
        exact[rows, preferred[rows]] = True
    required = positive & result.preferred_correction_required
    if bool(required.any()):
        certified = result.preferred_equivalent_actions.gather(
            -1, preferred.clamp_min(0).unsqueeze(-1)
        ).squeeze(-1)
        if not bool(certified[required].all()):
            raise ValueError(
                f"seed {result.seed} has an exact preferred correction outside its "
                "certified equivalence set"
            )
    no_correction = positive & ~result.preferred_correction_required
    if bool(no_correction.any()) and not bool(
        (preferred[no_correction] == result.parent_actions[no_correction]).all()
    ):
        raise ValueError(
            f"seed {result.seed} has a non-correction positive target that does not "
            "copy the parent"
        )
    if bool(positive.any()) and not bool((exact[positive].sum(dim=-1) == 1).all()):
        raise AssertionError("exact preferred targets must be one-hot")
    result.preferred_action_set = exact
    return result


def _assert_only_preferred_sets_differ(
    exact: Sequence[EpisodeFeatures], equivalent: Sequence[EpisodeFeatures]
) -> None:
    if [item.seed for item in exact] != [item.seed for item in equivalent]:
        raise AssertionError("paired episode orders differ")
    for left, right in zip(exact, equivalent, strict=True):
        for field in fields(EpisodeFeatures):
            if field.name == "preferred_action_set":
                continue
            left_value = getattr(left, field.name)
            right_value = getattr(right, field.name)
            if isinstance(left_value, torch.Tensor):
                if not torch.equal(left_value, right_value):
                    raise AssertionError(f"paired field differs: {field.name}")
            elif left_value != right_value:
                raise AssertionError(f"paired field differs: {field.name}")


def _assert_episodes_identical(
    left: Sequence[EpisodeFeatures], right: Sequence[EpisodeFeatures]
) -> None:
    if [item.seed for item in left] != [item.seed for item in right]:
        raise AssertionError("paired episode orders differ")
    for left_episode, right_episode in zip(left, right, strict=True):
        for field in fields(EpisodeFeatures):
            left_value = getattr(left_episode, field.name)
            right_value = getattr(right_episode, field.name)
            if isinstance(left_value, torch.Tensor):
                if not torch.equal(left_value, right_value):
                    raise AssertionError(f"paired tensor differs: {field.name}")
            elif left_value != right_value:
                raise AssertionError(f"paired field differs: {field.name}")


def _target_intervention_stats(
    exact: Sequence[EpisodeFeatures], equivalent: Sequence[EpisodeFeatures]
) -> dict[str, Any]:
    _assert_only_preferred_sets_differ(exact, equivalent)
    original_cardinality: Counter[int] = Counter()
    exact_cardinality: Counter[int] = Counter()
    positive_rows = 0
    correction_required_rows = 0
    differing_rows = 0
    for left, right in zip(exact, equivalent, strict=True):
        positive = right.gate_valid & (right.gate_targets > 0.0)
        required = positive & right.preferred_correction_required
        changed = (left.preferred_action_set != right.preferred_action_set).any(dim=-1)
        if bool((changed & ~positive).any()):
            raise AssertionError("preferred target intervention changed a non-positive row")
        positive_rows += int(positive.sum())
        correction_required_rows += int(required.sum())
        differing_rows += int(changed.sum())
        original_cardinality.update(
            int(value) for value in right.preferred_action_set[positive].sum(-1).tolist()
        )
        exact_cardinality.update(
            int(value) for value in left.preferred_action_set[positive].sum(-1).tolist()
        )
    return {
        "positive_target_rows": positive_rows,
        "correction_required_rows": correction_required_rows,
        "target_set_differing_rows": differing_rows,
        "equivalence_cardinality_distribution": {
            str(key): value for key, value in sorted(original_cardinality.items())
        },
        "exact_cardinality_distribution": {
            str(key): value for key, value in sorted(exact_cardinality.items())
        },
    }


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _raw_action_metrics(
    episodes: Sequence[EpisodeFeatures],
    predictions: Mapping[int, Mapping[str, torch.Tensor]],
) -> dict[str, Any]:
    totals = Counter()
    for episode in episodes:
        values = predictions[episode.seed]
        candidates = values["candidates"]
        early_required = (
            episode.gate_valid
            & (episode.gate_targets > 0.0)
            & episode.preferred_correction_required
            & (episode.anticipatory_lead_decisions >= EARLY_ONSET_MINIMUM_LEAD_DECISIONS)
            & (episode.anticipatory_lead_decisions <= FUTURE_ONSET_HORIZON_DECISIONS)
        )
        preferred = episode.preferred_actions
        equivalent = episode.preferred_equivalent_actions.gather(
            -1, candidates.unsqueeze(-1)
        ).squeeze(-1)
        finite = values["action_all_members_finite"]
        scored = early_required & finite
        tiebreak_eligible = _preferred_action_tiebreak_mask(
            episode.preferred_equivalent_actions,
            episode.previous_actions,
            early_required,
        )
        eligible_scored = tiebreak_eligible & finite
        previous = episode.previous_actions
        previous_exact = candidates == previous
        exact = candidates == preferred
        changed_parent = candidates != episode.parent_actions
        direction_correct = (candidates % 9) == (preferred % 9)
        speed_correct = (candidates >= 9) == (preferred >= 9)
        totals["targets"] += int(early_required.sum())
        totals["finite_top1"] += int(scored.sum())
        totals["exact_top1"] += int((scored & exact).sum())
        totals["equivalent_top1"] += int((scored & equivalent).sum())
        totals["candidate_changed_parent"] += int((scored & changed_parent).sum())
        totals["direction_correct"] += int((scored & direction_correct).sum())
        totals["speed_correct"] += int((scored & speed_correct).sum())
        totals["tiebreak_eligible_targets"] += int(tiebreak_eligible.sum())
        totals["tiebreak_eligible_finite_top1"] += int(eligible_scored.sum())
        totals["tiebreak_eligible_previous_top1"] += int(
            (eligible_scored & previous_exact).sum()
        )
        totals["tiebreak_eligible_exact_top1"] += int(
            (eligible_scored & exact).sum()
        )
        totals["tiebreak_eligible_equivalent_top1"] += int(
            (eligible_scored & equivalent).sum()
        )
        totals["tiebreak_eligible_direction_correct"] += int(
            (eligible_scored & direction_correct).sum()
        )
        totals["tiebreak_eligible_speed_correct"] += int(
            (eligible_scored & speed_correct).sum()
        )
    denominator = totals["targets"]
    eligible_denominator = totals["tiebreak_eligible_targets"]
    return {
        "targets": denominator,
        "finite_top1": totals["finite_top1"],
        "exact_top1": totals["exact_top1"],
        "exact_top1_rate": _ratio(totals["exact_top1"], denominator),
        "equivalent_top1": totals["equivalent_top1"],
        "equivalent_top1_rate": _ratio(totals["equivalent_top1"], denominator),
        "candidate_changed_parent": totals["candidate_changed_parent"],
        "candidate_changed_parent_rate": _ratio(
            totals["candidate_changed_parent"], denominator
        ),
        "direction_correct": totals["direction_correct"],
        "direction_accuracy": _ratio(totals["direction_correct"], denominator),
        "speed_correct": totals["speed_correct"],
        "speed_accuracy": _ratio(totals["speed_correct"], denominator),
        "tiebreak_eligible_targets": eligible_denominator,
        "tiebreak_eligible_finite_top1": totals[
            "tiebreak_eligible_finite_top1"
        ],
        "tiebreak_eligible_previous_top1": totals[
            "tiebreak_eligible_previous_top1"
        ],
        "tiebreak_eligible_previous_top1_rate": _ratio(
            totals["tiebreak_eligible_previous_top1"], eligible_denominator
        ),
        "tiebreak_eligible_exact_top1": totals["tiebreak_eligible_exact_top1"],
        "tiebreak_eligible_exact_top1_rate": _ratio(
            totals["tiebreak_eligible_exact_top1"], eligible_denominator
        ),
        "tiebreak_eligible_equivalent_top1": totals[
            "tiebreak_eligible_equivalent_top1"
        ],
        "tiebreak_eligible_equivalent_top1_rate": _ratio(
            totals["tiebreak_eligible_equivalent_top1"], eligible_denominator
        ),
        "tiebreak_eligible_direction_correct": totals[
            "tiebreak_eligible_direction_correct"
        ],
        "tiebreak_eligible_direction_accuracy": _ratio(
            totals["tiebreak_eligible_direction_correct"], eligible_denominator
        ),
        "tiebreak_eligible_speed_correct": totals[
            "tiebreak_eligible_speed_correct"
        ],
        "tiebreak_eligible_speed_accuracy": _ratio(
            totals["tiebreak_eligible_speed_correct"], eligible_denominator
        ),
    }


def _runtime_metrics(
    predictions: dict[int, dict[str, torch.Tensor]],
    episodes: list[EpisodeFeatures],
    runtime: Any,
) -> dict[str, Any]:
    metrics = _metrics(predictions, episodes, runtime)
    return {
        "total": metrics["total"],
        "offline_deployment_eligible": _offline_deployment_eligible(metrics),
    }


def _prediction_map(
    adapter: ResidualCorrectionAdapter,
    episodes: Sequence[EpisodeFeatures],
) -> dict[int, dict[str, torch.Tensor]]:
    adapter.cpu().eval()
    return {
        episode.seed: _predict_episode(adapter, episode, device="cpu")
        for episode in episodes
    }


def _run_arm(
    name: str,
    adapter: ResidualCorrectionAdapter,
    episodes: list[EpisodeFeatures],
    fold: Fold,
    *,
    epochs: int,
    member_seed: int,
    collision_weights: torch.Tensor,
    physical_weights: torch.Tensor,
    preferred_action_uniform_loss_weight: float = 0.0,
    preferred_action_tiebreak_loss_weight: float = 0.0,
    preferred_action_rank_loss_weight: float = 0.0,
    preferred_action_rank_margin: float = 1.0,
) -> dict[str, Any]:
    fit = _episodes_by_seed(episodes, fold.fit_seeds)
    calibration = _episodes_by_seed(episodes, fold.calibration_seeds)
    audit = _episodes_by_seed(episodes, fold.audit_seeds)
    torch.manual_seed(member_seed)
    history = _train_member(
        adapter,
        0,
        fit,
        seed=member_seed,
        epochs=epochs,
        learning_rate=TRAINING_CONFIG["learning_rate"],
        weight_decay=TRAINING_CONFIG["weight_decay"],
        chunk_length=TRAINING_CONFIG["chunk_length"],
        gate_positive_weight=TRAINING_CONFIG["gate_positive_weight"],
        action_loss_weight=TRAINING_CONFIG["action_loss_weight"],
        preferred_action_loss_weight=TRAINING_CONFIG[
            "preferred_action_loss_weight"
        ],
        preferred_action_uniform_loss_weight=(
            preferred_action_uniform_loss_weight
        ),
        preferred_action_tiebreak_loss_weight=(
            preferred_action_tiebreak_loss_weight
        ),
        preferred_action_rank_loss_weight=preferred_action_rank_loss_weight,
        preferred_action_rank_margin=preferred_action_rank_margin,
        safety_candidate_loss_weight=TRAINING_CONFIG[
            "safety_candidate_loss_weight"
        ],
        parent_copy_weight=TRAINING_CONFIG["parent_copy_weight"],
        collision_loss_weight=TRAINING_CONFIG["collision_loss_weight"],
        minimum_margin_loss_weight=TRAINING_CONFIG["minimum_margin_loss_weight"],
        physical_danger_loss_weight=TRAINING_CONFIG[
            "physical_danger_loss_weight"
        ],
        collision_positive_weights=collision_weights,
        physical_danger_positive_weights=physical_weights,
        all_collision_row_weight=TRAINING_CONFIG["all_collision_row_weight"],
        episode_bootstrap=TRAINING_CONFIG["episode_bootstrap"],
        device="cpu",
    )
    fit_cal_predictions = _prediction_map(adapter, [*fit, *calibration])
    runtime = None
    calibration_error = None
    diagnostics = None
    try:
        runtime = _calibrate(
            fit_cal_predictions,
            fit,
            calibration,
            ensemble_size=adapter.config.ensemble_size,
            per_action_safety_critic=adapter.config.per_action_safety_critic,
            per_action_physical_danger=adapter.config.per_action_physical_danger,
            future_onset_gate=True,
        )
    except ValueError as error:
        expected = "no fail-closed future-onset calibration covers early events"
        if expected not in str(error):
            raise
        calibration_error = str(error)
        diagnostics = _future_onset_calibration_diagnostics(
            fit_cal_predictions,
            fit,
            calibration,
            ensemble_size=adapter.config.ensemble_size,
        )

    # The audit prediction is deliberately delayed until calibration has
    # terminated. It is evaluated exactly once and cannot affect any choice.
    audit_predictions = _prediction_map(adapter, audit)
    predictions = {**fit_cal_predictions, **audit_predictions}
    raw = {
        "fit": _raw_action_metrics(fit, predictions),
        "calibration": _raw_action_metrics(calibration, predictions),
        "audit": _raw_action_metrics(audit, predictions),
    }
    runtime_metrics = None
    if runtime is not None:
        runtime_metrics = {
            "fit": _runtime_metrics(predictions, fit, runtime),
            "calibration": _runtime_metrics(predictions, calibration, runtime),
            "audit": _runtime_metrics(predictions, audit, runtime),
        }
    trained_state = adapter.state_dict()
    action_state, non_action_state = _partition_action_branch_state(trained_state)
    return {
        "name": name,
        "objective_controls": {
            "preferred_action_loss_weight": TRAINING_CONFIG[
                "preferred_action_loss_weight"
            ],
            "preferred_action_uniform_loss_weight": (
                preferred_action_uniform_loss_weight
            ),
            "preferred_action_tiebreak_loss_weight": (
                preferred_action_tiebreak_loss_weight
            ),
            "preferred_action_rank_loss_weight": (
                preferred_action_rank_loss_weight
            ),
            "preferred_action_rank_margin": preferred_action_rank_margin,
        },
        "member_seed": member_seed,
        "history": history,
        "trained_state_sha256": _state_digest(trained_state),
        "action_branch_state_sha256": _state_digest(action_state),
        "non_action_branch_state_sha256": _state_digest(non_action_state),
        "calibration": {
            "success": runtime is not None,
            "error": calibration_error,
            "runtime_config": None if runtime is None else asdict(runtime),
            "failure_diagnostics": diagnostics,
        },
        "raw_action_metrics": raw,
        "runtime_metrics": runtime_metrics,
    }


def _sum_raw(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = Counter()
    for row in rows:
        for key in (
            "targets",
            "finite_top1",
            "exact_top1",
            "equivalent_top1",
            "candidate_changed_parent",
            "direction_correct",
            "speed_correct",
            "tiebreak_eligible_targets",
            "tiebreak_eligible_finite_top1",
            "tiebreak_eligible_previous_top1",
            "tiebreak_eligible_exact_top1",
            "tiebreak_eligible_equivalent_top1",
            "tiebreak_eligible_direction_correct",
            "tiebreak_eligible_speed_correct",
        ):
            counts[key] += int(row[key])
    denominator = counts["targets"]
    eligible_denominator = counts["tiebreak_eligible_targets"]
    return {
        **{key: counts[key] for key in counts},
        "exact_top1_rate": _ratio(counts["exact_top1"], denominator),
        "equivalent_top1_rate": _ratio(counts["equivalent_top1"], denominator),
        "candidate_changed_parent_rate": _ratio(
            counts["candidate_changed_parent"], denominator
        ),
        "direction_accuracy": _ratio(counts["direction_correct"], denominator),
        "speed_accuracy": _ratio(counts["speed_correct"], denominator),
        "tiebreak_eligible_previous_top1_rate": _ratio(
            counts["tiebreak_eligible_previous_top1"], eligible_denominator
        ),
        "tiebreak_eligible_exact_top1_rate": _ratio(
            counts["tiebreak_eligible_exact_top1"], eligible_denominator
        ),
        "tiebreak_eligible_equivalent_top1_rate": _ratio(
            counts["tiebreak_eligible_equivalent_top1"], eligible_denominator
        ),
        "tiebreak_eligible_direction_accuracy": _ratio(
            counts["tiebreak_eligible_direction_correct"], eligible_denominator
        ),
        "tiebreak_eligible_speed_accuracy": _ratio(
            counts["tiebreak_eligible_speed_correct"], eligible_denominator
        ),
    }


def _comparison_summary(
    folds: Sequence[Mapping[str, Any]],
    *,
    baseline: str,
    candidate: str,
    audit_micro: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    fold_deltas: list[dict[str, Any]] = []
    for fold in folds:
        split_deltas = {}
        for split in ("fit", "calibration", "audit"):
            baseline_values = fold["arms"][baseline]["raw_action_metrics"][split]
            candidate_values = fold["arms"][candidate]["raw_action_metrics"][split]
            split_deltas[split] = {
                field: float(candidate_values[field]) - float(baseline_values[field])
                for field in RAW_RATE_FIELDS
            }
        fold_deltas.append({
            "fold": fold["fold"],
            "raw_action_rate_deltas": split_deltas,
        })
    audit_micro_delta = {
        field: float(audit_micro[candidate][field])
        - float(audit_micro[baseline][field])
        for field in RAW_RATE_FIELDS
    }
    direction = {}
    for field in RAW_RATE_FIELDS:
        values = [
            fold["raw_action_rate_deltas"]["audit"][field]
            for fold in fold_deltas
        ]
        direction[field] = {
            "candidate_better_folds": sum(value > 0.0 for value in values),
            "tied_folds": sum(value == 0.0 for value in values),
            "baseline_better_folds": sum(value < 0.0 for value in values),
        }
    return {
        "baseline": baseline,
        "candidate": candidate,
        "delta_semantics": f"{candidate}_minus_{baseline}",
        "fold_deltas": fold_deltas,
        "outer_audit_micro_rate_delta": audit_micro_delta,
        "outer_audit_fold_direction": direction,
    }


def _paired_summary(folds: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    audit_micro = {
        arm: _sum_raw([
            fold["arms"][arm]["raw_action_metrics"]["audit"] for fold in folds
        ])
        for arm in ARM_NAMES
    }
    comparisons = {
        f"{candidate}_minus_{baseline}": _comparison_summary(
            folds,
            baseline=baseline,
            candidate=candidate,
            audit_micro=audit_micro,
        )
        for baseline, candidate in (
            ("exact", "equivalence"),
            ("exact", "equivalence_top1_rank"),
            ("equivalence", "equivalence_top1_rank"),
            ("exact", "equivalence_weak_tiebreak"),
            ("equivalence", "equivalence_weak_tiebreak"),
            ("equivalence_top1_rank", "equivalence_weak_tiebreak"),
            ("exact", "equivalence_uniform_soft_target"),
            ("equivalence", "equivalence_uniform_soft_target"),
            ("equivalence_top1_rank", "equivalence_uniform_soft_target"),
            ("equivalence_weak_tiebreak", "equivalence_uniform_soft_target"),
        )
    }
    return {
        "outer_audit_micro": audit_micro,
        "comparisons": comparisons,
        "calibration_successful_folds": {
            arm: sum(
                bool(fold["arms"][arm]["calibration"]["success"])
                for fold in folds
            )
            for arm in ARM_NAMES
        },
    }


def _uniform_soft_target_screen(
    paired_summary: Mapping[str, Any],
    *,
    epochs: int,
) -> dict[str, Any]:
    baseline = paired_summary["outer_audit_micro"]["equivalence"]
    candidate = paired_summary["outer_audit_micro"][
        "equivalence_uniform_soft_target"
    ]
    comparison = paired_summary["comparisons"][
        "equivalence_uniform_soft_target_minus_equivalence"
    ]
    expected = UNIFORM_SOFT_TARGET_SCREEN
    applicable = epochs == expected["screening_epochs"]
    baseline_reproduced = all((
        int(baseline["targets"]) == expected["expected_targets"],
        int(baseline["finite_top1"]) == expected["expected_targets"],
        int(baseline["equivalent_top1"])
        == expected["expected_baseline_equivalent_top1"],
        int(baseline["direction_correct"])
        == expected["expected_baseline_direction_correct"],
        int(baseline["speed_correct"])
        == expected["expected_baseline_speed_correct"],
    ))
    improved_folds = comparison["outer_audit_fold_direction"][
        "equivalent_top1_rate"
    ]["candidate_better_folds"]
    checks = {
        "deterministic_reference_metrics_reproduced": baseline_reproduced,
        "candidate_finite_top1_complete": (
            int(candidate["targets"]) == expected["expected_targets"]
            and int(candidate["finite_top1"]) == expected["expected_targets"]
        ),
        "candidate_equivalent_top1_at_least_68": (
            int(candidate["equivalent_top1"])
            >= expected["minimum_candidate_equivalent_top1"]
        ),
        "candidate_improves_at_least_two_audit_folds": (
            int(improved_folds) >= expected["minimum_improved_audit_folds"]
        ),
        "candidate_direction_regression_within_two_percentage_points": (
            int(candidate["direction_correct"])
            >= expected["minimum_candidate_direction_correct"]
        ),
        "candidate_speed_regression_within_two_percentage_points": (
            int(candidate["speed_correct"])
            >= expected["minimum_candidate_speed_correct"]
        ),
    }
    return {
        "preregistered_before_audit": True,
        "applicable": applicable,
        "inapplicable_reason": (
            None if applicable else
            "the preregistered screen applies only to exactly 6 training epochs"
        ),
        "criteria": dict(expected),
        "checks": checks,
        "passed": applicable and all(checks.values()),
        "eligible_for_fixed_e20_followup": applicable and all(checks.values()),
        "deployment_eligible": False,
        "acceptance_claim": False,
    }


def _adapter_config(failure: Mapping[str, Any]) -> ResidualAdapterConfig:
    fit_record = failure.get("fit_checkpoint")
    if not isinstance(fit_record, Mapping):
        raise ValueError("failure diagnostics have no fit checkpoint record")
    raw = fit_record.get("adapter_config")
    if not isinstance(raw, Mapping) or dict(raw) != EXPECTED_V81_CONFIG:
        raise ValueError("failure diagnostics do not describe the fixed v81 architecture")
    values = dict(raw)
    values["ensemble_size"] = 1
    return ResidualAdapterConfig(**values)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired training-only CV for preferred-action objectives."
    )
    parser.add_argument("--failure", type=Path, default=DEFAULT_FAILURE)
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--cpu-threads", type=int, default=1)
    args = parser.parse_args()
    if args.epochs <= 0 or args.cpu_threads <= 0:
        raise ValueError("epochs and cpu threads must be positive")
    script_path = Path(__file__)
    helper_path = script_path.with_name("train_temporal_residual_adapter.py")
    _validate_output_path(
        args.output,
        (args.failure, args.parent, script_path, helper_path),
    )

    torch.set_num_threads(args.cpu_threads)
    torch.use_deterministic_algorithms(True)
    failure = _read_json(args.failure)
    selected = _select_training_inventory(failure)
    triplets, source_provenance = _verify_training_sources(selected)
    _validate_output_path(
        args.output,
        [path for triplet in triplets for path in triplet],
    )
    config = _adapter_config(failure)
    parent_sha256 = file_sha256(args.parent)
    if parent_sha256 != failure.get("parent_checkpoint_sha256"):
        raise ValueError("parent checkpoint hash does not match failure diagnostics")
    parent, _metadata = load_checkpoint(args.parent, device="cpu")
    parent.cpu().eval()
    for parameter in parent.parameters():
        parameter.requires_grad_(False)
    if parent.config.recurrent_size != config.recurrent_size:
        raise ValueError("parent recurrent size does not match the v81 adapter")

    feature_adapter = ResidualCorrectionAdapter(config)
    raw_episodes = [
        _load_episode(
            parent,
            feature_adapter,
            dataset,
            report,
            manifest,
            parent_checkpoint_sha256=parent_sha256,
            device="cpu",
            chunk_length=256,
            **LABEL_CONFIG,
        )
        for dataset, report, manifest in triplets
    ]
    if tuple(episode.seed for episode in raw_episodes) != TRAINING_SEEDS:
        raise ValueError("loaded episodes do not match the ordered training whitelist")

    fold_reports: list[dict[str, Any]] = []
    for fold in _fixed_folds():
        fold_seed = args.seed + fold.index * 100_003
        normalized = [_clone_episode(episode) for episode in raw_episodes]
        torch.manual_seed(fold_seed)
        base_adapter = ResidualCorrectionAdapter(config)
        fit = _episodes_by_seed(normalized, fold.fit_seeds)
        _normalize(base_adapter, fit, normalized)
        collision_weights = _collision_positive_weights(
            fit,
            maximum_weight=TRAINING_CONFIG["maximum_collision_positive_weight"],
        )
        physical_weights = _physical_danger_positive_weights(
            fit,
            maximum_weight=TRAINING_CONFIG[
                "maximum_physical_danger_positive_weight"
            ],
        )
        initial_state = copy.deepcopy(base_adapter.state_dict())
        initial_sha256 = _state_digest(initial_state)
        adapters = {
            arm: ResidualCorrectionAdapter(config) for arm in ARM_NAMES
        }
        for adapter in adapters.values():
            adapter.load_state_dict(initial_state)
        if not all(
            _states_equal(initial_state, adapter.state_dict())
            for adapter in adapters.values()
        ):
            raise AssertionError("paired arms do not share bit-identical initialization")
        gradient_clip_groups = _assert_paired_gradient_clip_groups(adapters)

        equivalent_episodes = [_clone_episode(episode) for episode in normalized]
        ranked_episodes = [_clone_episode(episode) for episode in normalized]
        tiebreak_episodes = [_clone_episode(episode) for episode in normalized]
        uniform_episodes = [_clone_episode(episode) for episode in normalized]
        exact_episodes = [_exact_target_episode(episode) for episode in normalized]
        _assert_only_preferred_sets_differ(exact_episodes, equivalent_episodes)
        _assert_only_preferred_sets_differ(exact_episodes, ranked_episodes)
        _assert_only_preferred_sets_differ(exact_episodes, tiebreak_episodes)
        _assert_only_preferred_sets_differ(exact_episodes, uniform_episodes)
        _assert_episodes_identical(equivalent_episodes, ranked_episodes)
        _assert_episodes_identical(equivalent_episodes, tiebreak_episodes)
        _assert_episodes_identical(equivalent_episodes, uniform_episodes)
        intervention = {
            split: _target_intervention_stats(
                _episodes_by_seed(exact_episodes, seeds),
                _episodes_by_seed(equivalent_episodes, seeds),
            )
            for split, seeds in (
                ("fit", fold.fit_seeds),
                ("calibration", fold.calibration_seeds),
                ("audit", fold.audit_seeds),
            )
        }
        member_seed = fold_seed + 1_009
        arms = {
            "exact": _run_arm(
                "preferred_exact_one_hot",
                adapters["exact"],
                exact_episodes,
                fold,
                epochs=args.epochs,
                member_seed=member_seed,
                collision_weights=collision_weights,
                physical_weights=physical_weights,
            ),
            "equivalence": _run_arm(
                "preferred_certified_equivalence_set",
                adapters["equivalence"],
                equivalent_episodes,
                fold,
                epochs=args.epochs,
                member_seed=member_seed,
                collision_weights=collision_weights,
                physical_weights=physical_weights,
            ),
            "equivalence_top1_rank": _run_arm(
                "preferred_certified_equivalence_set_with_top1_rank",
                adapters["equivalence_top1_rank"],
                ranked_episodes,
                fold,
                epochs=args.epochs,
                member_seed=member_seed,
                collision_weights=collision_weights,
                physical_weights=physical_weights,
                preferred_action_rank_loss_weight=RANK_OBJECTIVE_CONFIG[
                    "preferred_action_rank_loss_weight"
                ],
                preferred_action_rank_margin=RANK_OBJECTIVE_CONFIG[
                    "preferred_action_rank_margin"
                ],
            ),
            "equivalence_weak_tiebreak": _run_arm(
                "preferred_certified_equivalence_set_with_previous_action_tiebreak",
                adapters["equivalence_weak_tiebreak"],
                tiebreak_episodes,
                fold,
                epochs=args.epochs,
                member_seed=member_seed,
                collision_weights=collision_weights,
                physical_weights=physical_weights,
                preferred_action_tiebreak_loss_weight=TIEBREAK_OBJECTIVE_CONFIG[
                    "preferred_action_tiebreak_loss_weight"
                ],
            ),
            "equivalence_uniform_soft_target": _run_arm(
                "preferred_certified_equivalence_set_with_uniform_soft_target",
                adapters["equivalence_uniform_soft_target"],
                uniform_episodes,
                fold,
                epochs=args.epochs,
                member_seed=member_seed,
                collision_weights=collision_weights,
                physical_weights=physical_weights,
                preferred_action_uniform_loss_weight=(
                    UNIFORM_SOFT_TARGET_OBJECTIVE_CONFIG[
                        "preferred_action_uniform_loss_weight"
                    ]
                ),
            ),
        }
        paired_non_action_sha256 = _assert_paired_non_action_states_equal(adapters)
        if any(
            arm["non_action_branch_state_sha256"] != paired_non_action_sha256
            for arm in arms.values()
        ):
            raise AssertionError(
                "reported non-action branch digest differs from paired state check"
            )
        fold_reports.append({
            "fold": fold.index,
            "fit_seeds": list(fold.fit_seeds),
            "calibration_seeds": list(fold.calibration_seeds),
            "audit_seeds": list(fold.audit_seeds),
            "normalization_fit_seeds": list(fold.fit_seeds),
            "positive_weight_fit_seeds": list(fold.fit_seeds),
            "fold_seed": fold_seed,
            "member_seed": member_seed,
            "initial_state_sha256": initial_sha256,
            "paired_initial_states_equal": True,
            "paired_episode_order_equal": True,
            "paired_equivalence_label_tensors_equal": True,
            "paired_member_seed_equal": True,
            "paired_grouped_gradient_clipping_equal": True,
            "paired_gradient_clip_groups": {
                group: list(names) for group, names in gradient_clip_groups.items()
            },
            "paired_non_action_states_equal": True,
            "paired_non_action_state_sha256": paired_non_action_sha256,
            "normalization_sha256": _state_digest({
                "feature_mean": base_adapter.feature_mean,
                "feature_scale": base_adapter.feature_scale,
            }),
            "collision_positive_weights": collision_weights.tolist(),
            "physical_danger_positive_weights": physical_weights.tolist(),
            "target_intervention": intervention,
            "arms": arms,
        })

    paired_summary = _paired_summary(fold_reports)
    report = {
        "schema_version": 3,
        "kind": "preferred_action_objective_paired_training_only_cv",
        "training_only": True,
        "deployment_artifact_written": False,
        "formal_deployment_artifact_written": False,
        "deployment_eligible": False,
        "acceptance_claim": False,
        "audit_used_during_fit_or_calibration": False,
        "audit_used_for_threshold_epoch_fold_or_retry_selection": False,
        "audit_used_for_after_freeze_objective_comparison": True,
        "audit_prediction_policy": (
            "predicted exactly once only after each arm's calibration attempt; "
            "never used for thresholds, retries, epochs, or fold construction; "
            "used only for the declared after-freeze paired objective comparison"
        ),
        "teacher_forced_previous_action_limitation": True,
        "raw_action_metric_semantics": {
            "nonfinite_action_outputs_count_as_hits": False,
            "nonfinite_action_outputs_remain_in_denominators": True,
            "tiebreak_eligible_subset": (
                "early correction-required and multi-member certified set with "
                "a valid row-local previous action in that set"
            ),
            "tiebreak_eligible_previous_top1": (
                "finite candidate equals the row-local previous action"
            ),
        },
        "objective_scope": (
            "preferred-action objective only: exact one-hot, certified-set NLL, "
            "the same set NLL plus best-accepted top1 rank hinge, or the same "
            "set NLL plus conditional row-local previous-action continuity "
            "supervision, or the same set NLL plus a symmetric uniform soft "
            "target within multi-member certified sets; "
            "all gate, copy, safety, and physical objectives are identical"
        ),
        "rank_objective": {
            **RANK_OBJECTIVE_CONFIG,
            "formula": (
                "relu(best_rejected_logit - best_accepted_logit + margin)"
            ),
            "labels_identical_to_equivalence_arm": True,
            "original_arms_rank_loss_weight": 0.0,
        },
        "conditional_tiebreak_objective": {
            **TIEBREAK_OBJECTIVE_CONFIG,
            "preferred_action_set_loss_weight": TRAINING_CONFIG[
                "preferred_action_loss_weight"
            ],
            "formula": (
                "12 * -log(sum(softmax(action_logits)[certified_set])) + "
                "3 * -log(softmax(action_logits restricted to certified_set)"
                "[previous_action]) on valid positive rows with certified_set "
                "cardinality > 1 and row-local previous_action in certified_set"
            ),
            "rejected_action_logit_gradient_from_tiebreak": 0.0,
            "continuity_target": (
                "row-local recorded previous action only when it belongs to the "
                "current row's certified set; no teacher or propagated-preferred "
                "fallback"
            ),
            "labels_identical_to_equivalence_arm": True,
            "non_tiebreak_arms_tiebreak_loss_weight": 0.0,
            "singleton_rows_receive_tiebreak_loss": False,
            "uncertified_previous_action_rows_receive_tiebreak_loss": False,
            "tiebreak_eligible_metric_denominator": (
                "early correction-required rows with certified-set cardinality > 1 "
                "and a valid row-local previous action in that certified set"
            ),
            "runtime_action_hold_logic_added": False,
            "runtime_selection_logic_changed": False,
            "training_only": True,
        },
        "uniform_soft_target_objective": {
            **UNIFORM_SOFT_TARGET_OBJECTIVE_CONFIG,
            "preferred_action_set_loss_weight": TRAINING_CONFIG[
                "preferred_action_loss_weight"
            ],
            "formula": (
                "12 * -log(sum(softmax(action_logits)[certified_set])) + "
                "3 * KL(uniform(certified_set) || softmax(action_logits "
                "restricted to certified_set)) on valid positive rows with "
                "certified_set cardinality > 1"
            ),
            "gradient": (
                "inside the certified set: conditional_probability - "
                "1/cardinality; outside the certified set: exactly zero"
            ),
            "labels_identical_to_equivalence_arm": True,
            "non_uniform_arms_uniform_loss_weight": 0.0,
            "singleton_rows_receive_uniform_loss": False,
            "unique_teacher_target_used": False,
            "previous_action_target_used": False,
            "runtime_selection_logic_changed": False,
            "training_only": True,
        },
        "data_isolation": {
            "allowed_training_seeds": list(TRAINING_SEEDS),
            "prohibited_source_seeds": sorted(PROHIBITED_SOURCE_SEEDS),
            "selection_before_path_access": True,
            "nontraining_path_fields_accessed": False,
            "source_inventory": source_provenance,
        },
        "input_provenance": {
            "failure": str(args.failure),
            "failure_sha256": file_sha256(args.failure),
            "parent": str(args.parent),
            "parent_sha256": parent_sha256,
            "experiment_script": str(script_path),
            "experiment_script_sha256": file_sha256(script_path),
            "training_helper": str(helper_path),
            "training_helper_sha256": file_sha256(helper_path),
        },
        "experiment_config": {
            "epochs": args.epochs,
            "base_seed": args.seed,
            "device": "cpu",
            "cpu_threads": args.cpu_threads,
            "deterministic_algorithms": True,
            "adapter_config": asdict(config),
            "label_config": LABEL_CONFIG,
            "training_config": TRAINING_CONFIG,
            "ensemble_size_screening_override": 1,
            "gradient_clipping": {
                "max_norm": GRADIENT_CLIP_MAX_NORM,
                "separate_action_recurrent_semantics": (
                    SEPARATE_ACTION_GRADIENT_CLIP_SEMANTICS
                ),
                "action_group_modules": list(ACTION_BRANCH_MODULE_NAMES),
                "shared_safety_group": "all_other_trainable_member_parameters",
                "non_separate_architecture_semantics": (
                    GLOBAL_GRADIENT_CLIP_SEMANTICS
                ),
            },
        },
        "folds": fold_reports,
        "paired_summary": paired_summary,
        "uniform_soft_target_preregistered_screen": (
            _uniform_soft_target_screen(paired_summary, epochs=args.epochs)
        ),
    }
    _write_json_atomic(args.output, report)
    output_sha256 = file_sha256(args.output)
    print(json.dumps({
        "artifact": str(args.output),
        "sha256": output_sha256,
        "training_only": True,
        "deployment_artifact_written": False,
    }))


if __name__ == "__main__":
    main()
