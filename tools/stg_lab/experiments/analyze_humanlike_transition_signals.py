"""Compare human motion boundaries with streaming visual-policy decisions.

This is a development-only, read-only analysis.  Decision indices and replay
metadata are used only to align diagnostics; the candidate supervision signals
come exclusively from the semantic raster and its recent changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from stg_lab.training import load_checkpoint


DEFAULT_DATASET = Path("artifacts/context-human-slot3-exact-hold-v3.npz")
DEFAULT_CHECKPOINTS = (
    (
        "v25",
        Path(
            "artifacts/policy-humanlike-highres-okuu3-v25-onset500-"
            "rank200-speedrank200-ft30.pt"
        ),
    ),
    (
        "v29",
        Path(
            "artifacts/policy-humanlike-highres-okuu3-v29-"
            "onsetrank200-soft-ft20.pt"
        ),
    ),
)
DEFAULT_NATIVE_TRACES = (
    (
        "v25",
        Path(
            "artifacts/traces-humanlike-highres-v25-onset500-rank200-"
            "speedrank200-ft30-native-precheck/attack_okuu_Lunatic_3/"
            "expert/seed-20260730.json"
        ),
    ),
    (
        "v29",
        Path(
            "artifacts/policy-humanlike-highres-okuu3-v29-"
            "onsetrank200-soft-ft20-native-precheck.json"
        ),
    ),
)
DEFAULT_DAGGER_REPORTS = (
    (
        "v25_teacher_assisted",
        Path(
            "artifacts/dagger-humanlike-highres-v25-seed20260730-"
            "beta0-margin10-regret6-report.json"
        ),
    ),
    (
        "v29_disagreement",
        Path(
            "artifacts/dagger-humanlike-highres-v29-seed20260731-"
            "disagreement-report.json"
        ),
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _named_path(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    name, path = raw.split("=", 1)
    if not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("expected non-empty NAME=PATH")
    return name.strip(), Path(path)


def _direction(actions: np.ndarray) -> np.ndarray:
    return np.asarray(actions, dtype=np.int64) % 9


def _action_name(value: int) -> str:
    value = int(value)
    direction = value % 9
    move_x = direction % 3 - 1
    move_y = direction // 3 - 1
    return f"{'S' if value >= 9 else 'F'}({move_x:+d},{move_y:+d})"


def _event_masks(
    actions: np.ndarray,
    supervised: np.ndarray,
) -> dict[str, np.ndarray]:
    """Find reliable hard-label transitions, bridging one mixed window."""

    count = len(actions)
    result = {
        name: np.zeros(count, dtype=np.bool_)
        for name in ("onset", "stop", "turn", "speed_change")
    }
    indices = np.flatnonzero(supervised)
    if len(indices) < 2:
        return result
    previous_indices = indices[:-1]
    current_indices = indices[1:]
    reliable = current_indices - previous_indices <= 2
    previous_actions = actions[previous_indices]
    current_actions = actions[current_indices]
    previous_direction = _direction(previous_actions)
    current_direction = _direction(current_actions)
    previous_moving = previous_direction != 4
    current_moving = current_direction != 4
    result["onset"][current_indices] = (
        reliable & current_moving & ~previous_moving
    )
    result["stop"][current_indices] = (
        reliable & ~current_moving & previous_moving
    )
    result["turn"][current_indices] = (
        reliable
        & current_moving
        & previous_moving
        & (current_direction != previous_direction)
    )
    result["speed_change"][current_indices] = (
        reliable
        & current_moving
        & previous_moving
        & (current_direction == previous_direction)
        & (current_actions // 9 != previous_actions // 9)
    )
    return result


def _run_metrics(actions: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(actions, dtype=np.int64)
    if not len(values):
        return {"runs": 0, "mean": 0.0, "median": 0.0, "p90": 0.0, "max": 0}
    boundaries = np.flatnonzero(values[1:] != values[:-1]) + 1
    lengths = np.diff(np.concatenate(((0,), boundaries, (len(values),))))
    return {
        "runs": int(len(lengths)),
        "mean": float(np.mean(lengths)),
        "median": float(np.median(lengths)),
        "p90": float(np.percentile(lengths, 90)),
        "max": int(np.max(lengths)),
    }


def _motion_activity(actions: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(actions, dtype=np.int64)
    if not len(values):
        return {
            "decisions": 0,
            "action_changes": 0,
            "movement_onsets": 0,
            "movement_stops": 0,
            "moving_turns": 0,
            "speed_changes": 0,
            "action_runs": _run_metrics(values),
        }
    previous = values[:-1]
    current = values[1:]
    previous_direction = _direction(previous)
    current_direction = _direction(current)
    previous_moving = previous_direction != 4
    current_moving = current_direction != 4
    return {
        "decisions": int(len(values)),
        "action_changes": int(np.count_nonzero(current != previous)),
        "movement_onsets": int(np.count_nonzero(current_moving & ~previous_moving)),
        "movement_stops": int(np.count_nonzero(~current_moving & previous_moving)),
        "moving_turns": int(np.count_nonzero(
            current_moving & previous_moving & (current_direction != previous_direction)
        )),
        "speed_changes": int(np.count_nonzero(
            current_moving
            & previous_moving
            & (current_direction == previous_direction)
            & (current // 9 != previous // 9)
        )),
        "action_runs": _run_metrics(values),
    }


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=-1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / np.sum(exponent, axis=-1, keepdims=True)


def _checkpoint_predictions(
    checkpoint_path: Path,
    data: Mapping[str, np.ndarray],
    *,
    device: str,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    model, checkpoint = load_checkpoint(checkpoint_path, device=device)
    global_frames = data["global_frames"][:, -1]
    local_frames = data["local_frames"][:, -1]
    memory = data.get("memory")
    proficiency = data.get("proficiency")
    if memory is None:
        memory = np.zeros((len(global_frames), model.config.memory_size), dtype=np.float32)
    else:
        memory = memory[:, -1]
    if memory.shape != (len(global_frames), model.config.memory_size):
        raise ValueError(
            f"{checkpoint_path}: dataset memory {memory.shape} does not match "
            f"policy memory_size={model.config.memory_size}"
        )
    if proficiency is not None:
        proficiency = proficiency[:, -1]

    logits_chunks: list[np.ndarray] = []
    risk_chunks: list[np.ndarray] = []
    hidden = None
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(global_frames), chunk_size):
            stop = min(start + chunk_size, len(global_frames))
            global_tensor = torch.as_tensor(
                global_frames[start:stop], dtype=torch.float32, device=device,
            )[None]
            local_tensor = torch.as_tensor(
                local_frames[start:stop], dtype=torch.float32, device=device,
            )[None]
            memory_tensor = torch.as_tensor(
                memory[start:stop], dtype=torch.float32, device=device,
            )[None]
            proficiency_tensor = (
                None
                if proficiency is None else
                torch.as_tensor(
                    proficiency[start:stop], dtype=torch.float32, device=device,
                )[None]
            )
            logits, risk, hidden = model(
                global_tensor,
                local_tensor,
                memory_tensor,
                proficiency_tensor,
                hidden=hidden,
            )
            hidden = hidden.detach()
            logits_chunks.append(logits[0].cpu().numpy())
            risk_chunks.append(risk[0].cpu().numpy())

    training_config = checkpoint.get("training_config", {})
    transition_controls = {
        key: value
        for key, value in training_config.items()
        if key.startswith("movement_") or key.startswith("transition_action_rank")
    }
    metadata = {
        "path": str(checkpoint_path),
        "sha256": _sha256(checkpoint_path),
        "policy_config": checkpoint["policy_config"],
        "transition_training_controls": transition_controls,
    }
    return np.concatenate(logits_chunks), np.concatenate(risk_chunks), metadata


def _predicted_onsets(actions: np.ndarray) -> np.ndarray:
    directions = _direction(actions)
    return np.flatnonzero((directions[1:] != 4) & (directions[:-1] == 4)) + 1


def _match_onsets(
    human_actions: np.ndarray,
    human_onsets: np.ndarray,
    predicted_actions: np.ndarray,
    radius: int,
) -> dict[str, Any]:
    human_indices = np.flatnonzero(human_onsets)
    predicted_indices = _predicted_onsets(predicted_actions)
    unused = set(int(value) for value in predicted_indices)
    offsets: list[int] = []
    matches: dict[int, int] = {}
    for human_index in human_indices:
        target_direction = int(human_actions[human_index] % 9)
        candidates = [
            predicted_index
            for predicted_index in unused
            if abs(predicted_index - human_index) <= radius
            and int(predicted_actions[predicted_index] % 9) == target_direction
        ]
        if not candidates:
            continue
        selected = min(candidates, key=lambda value: (abs(value - human_index), value))
        unused.remove(selected)
        matches[int(human_index)] = int(selected)
        offsets.append(int(selected - human_index))
    histogram = {
        str(offset): int(offsets.count(offset))
        for offset in range(-radius, radius + 1)
    }
    labels = int(len(human_indices))
    return {
        "labels": labels,
        "matched": int(len(offsets)),
        "missing": int(labels - len(offsets)),
        "within_one_fraction_of_all_labels": (
            float(sum(abs(value) <= 1 for value in offsets) / labels)
            if labels else 0.0
        ),
        "early_by_two_or_more": int(sum(value <= -2 for value in offsets)),
        "late_by_two_or_more": int(sum(value >= 2 for value in offsets)),
        "offset_histogram": histogram,
        "mean_matched_offset": float(np.mean(offsets)) if offsets else None,
        "median_matched_offset": float(np.median(offsets)) if offsets else None,
        "matches": matches,
    }


def _masked_fraction(values: np.ndarray, mask: np.ndarray) -> float:
    return float(np.mean(values[mask])) if np.any(mask) else 0.0


def _model_metrics(
    logits: np.ndarray,
    human_actions: np.ndarray,
    supervised: np.ndarray,
    events: Mapping[str, np.ndarray],
    *,
    match_radius: int,
) -> dict[str, Any]:
    predicted = np.argmax(logits, axis=-1).astype(np.int64)
    probabilities = _softmax(logits)
    moving_columns = np.asarray([value % 9 != 4 for value in range(18)])
    moving_probability = np.sum(probabilities[:, moving_columns], axis=-1)
    human_direction = _direction(human_actions)
    predicted_direction = _direction(predicted)
    human_moving = human_direction != 4
    predicted_moving = predicted_direction != 4
    result: dict[str, Any] = {
        "exact_accuracy": _masked_fraction(predicted == human_actions, supervised),
        "direction_accuracy": _masked_fraction(
            predicted_direction == human_direction, supervised,
        ),
        "speed_accuracy": _masked_fraction(
            predicted // 9 == human_actions // 9, supervised,
        ),
        "stationary_false_move_rate": _masked_fraction(
            predicted_moving, supervised & ~human_moving,
        ),
        "moving_false_stop_rate": _masked_fraction(
            ~predicted_moving, supervised & human_moving,
        ),
        "mean_move_probability_at_human_onset": float(
            np.mean(moving_probability[events["onset"]])
        ),
        "predicted_action_activity": _motion_activity(predicted),
    }
    for name, mask in events.items():
        result[name] = {
            "labels": int(np.count_nonzero(mask)),
            "exact": _masked_fraction(predicted == human_actions, mask),
            "direction": _masked_fraction(
                predicted_direction == human_direction, mask,
            ),
            "speed": _masked_fraction(
                predicted // 9 == human_actions // 9, mask,
            ),
        }
    result["onset"]["temporal_same_direction"] = _match_onsets(
        human_actions,
        events["onset"],
        predicted,
        match_radius,
    )
    return result


def _finite_summary(values: np.ndarray) -> dict[str, float | int | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p10": None,
            "p90": None,
        }
    return {
        "count": int(len(finite)),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p10": float(np.percentile(finite, 10)),
        "p90": float(np.percentile(finite, 90)),
    }


def _auc_greater(positive: np.ndarray, negative: np.ndarray) -> float | None:
    positive = positive[np.isfinite(positive)]
    negative = negative[np.isfinite(negative)]
    if not len(positive) or not len(negative):
        return None
    differences = positive[:, None] - negative[None, :]
    return float(np.mean((differences > 0) + 0.5 * (differences == 0)))


def _feature_comparison(
    values: np.ndarray,
    onset: np.ndarray,
    stationary_hold: np.ndarray,
) -> dict[str, Any]:
    onset_values = np.asarray(values[onset], dtype=np.float64)
    hold_values = np.asarray(values[stationary_hold], dtype=np.float64)
    auc = _auc_greater(onset_values, hold_values)
    direction = None
    separability = None
    if auc is not None:
        direction = "higher_at_onset" if auc >= 0.5 else "lower_at_onset"
        separability = max(auc, 1.0 - auc)
    return {
        "human_onset": _finite_summary(onset_values),
        "stationary_hold": _finite_summary(hold_values),
        "pairwise_auc": auc,
        "onset_direction": direction,
        "pairwise_separability": separability,
    }


def _projected_clearance(
    local_frames: np.ndarray,
    *,
    horizon: int,
    local_extent: float = 72.0,
    velocity_scale: float = 8.0,
) -> np.ndarray:
    height, width = local_frames.shape[-2:]
    xs = np.linspace(-local_extent, local_extent, width, dtype=np.float64)
    ys = np.linspace(-local_extent, local_extent, height, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(xs, ys)
    result = np.full(len(local_frames), np.inf, dtype=np.float64)
    for index, frame in enumerate(local_frames):
        occupied = frame[0] > 0.0
        if not np.any(occupied):
            continue
        velocity_x = frame[1].astype(np.float64) * velocity_scale
        velocity_y = frame[2].astype(np.float64) * velocity_scale
        result[index] = min(
            float(np.min(np.hypot(
                grid_x[occupied] + velocity_x[occupied] * step,
                grid_y[occupied] + velocity_y[occupied] * step,
            )))
            for step in range(horizon + 1)
        )
    return result


def _lag_delta(values: np.ndarray, lag: int) -> np.ndarray:
    result = np.full(len(values), np.nan, dtype=np.float64)
    current = np.asarray(values[lag:], dtype=np.float64)
    previous = np.asarray(values[:-lag], dtype=np.float64)
    finite = np.isfinite(current) & np.isfinite(previous)
    result[lag:][finite] = current[finite] - previous[finite]
    return result


def _visual_signals(
    data: Mapping[str, np.ndarray],
    events: Mapping[str, np.ndarray],
    human_actions: np.ndarray,
    supervised: np.ndarray,
    *,
    projection_horizon: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    local = data["local_frames"][:, -1].astype(np.float32)
    global_frames = data["global_frames"][:, -1].astype(np.float32)
    local_mass = np.sum(local[:, 0], axis=(1, 2), dtype=np.float64)
    global_mass = np.sum(global_frames[:, 0], axis=(1, 2), dtype=np.float64)
    midpoint = local.shape[-1] // 2
    left_mass = np.sum(local[:, 0, :, :midpoint], axis=(1, 2), dtype=np.float64)
    right_mass = np.sum(local[:, 0, :, midpoint:], axis=(1, 2), dtype=np.float64)
    left_minus_right = left_mass - right_mass
    clearance = _projected_clearance(local, horizon=projection_horizon)
    stationary_hold = (
        supervised
        & (_direction(human_actions) == 4)
        & ~events["stop"]
    )
    onset = events["onset"]
    feature_values = {
        "local_occupancy_mass": local_mass,
        "local_occupancy_growth_1": _lag_delta(local_mass, 1),
        "local_occupancy_growth_3": _lag_delta(local_mass, 3),
        "global_occupancy_mass": global_mass,
        "left_minus_right_occupancy": left_minus_right,
        "left_minus_right_change_1": _lag_delta(left_minus_right, 1),
        "projected_minimum_clearance": clearance,
        "projected_clearance_change_1": _lag_delta(clearance, 1),
    }
    comparisons = {
        name: _feature_comparison(values, onset, stationary_hold)
        for name, values in feature_values.items()
    }
    comparisons["sample_counts"] = {
        "human_onset": int(np.count_nonzero(onset)),
        "stationary_hold": int(np.count_nonzero(stationary_hold)),
    }
    comparisons["proactive_onset_fraction"] = {
        "clearance_over_24": float(np.mean(clearance[onset] > 24.0)),
        "definition": (
            "human movement onset while the visible 0..12-frame projected "
            "minimum cell-center clearance remains above 24 world units"
        ),
    }
    return comparisons, feature_values


def _boundary_examples(
    human_actions: np.ndarray,
    events: Mapping[str, np.ndarray],
    model_logits: Mapping[str, np.ndarray],
    model_metrics: Mapping[str, Mapping[str, Any]],
    features: Mapping[str, np.ndarray],
    *,
    limit: int = 12,
) -> list[dict[str, Any]]:
    predictions = {
        name: np.argmax(logits, axis=-1).astype(np.int64)
        for name, logits in model_logits.items()
    }
    onset_indices = np.flatnonzero(events["onset"])
    rows: list[dict[str, Any]] = []
    for index in onset_indices:
        per_model: dict[str, Any] = {}
        interesting = False
        for name, predicted in predictions.items():
            temporal = model_metrics[name]["onset"]["temporal_same_direction"]
            matched = temporal["matches"].get(int(index))
            offset = None if matched is None else int(matched - index)
            exact = bool(predicted[index] == human_actions[index])
            interesting |= not exact or offset is None or abs(offset) >= 2
            per_model[name] = {
                "predicted_action": _action_name(int(predicted[index])),
                "exact_at_boundary": exact,
                "same_direction_onset_offset": offset,
            }
        if not interesting:
            continue
        rows.append({
            "diagnostic_decision_index": int(index),
            "human_action": _action_name(int(human_actions[index])),
            "local_occupancy_mass": float(features["local_occupancy_mass"][index]),
            "local_occupancy_growth_1": float(
                features["local_occupancy_growth_1"][index]
            ),
            "left_minus_right_change_1": float(
                features["left_minus_right_change_1"][index]
            ),
            "projected_minimum_clearance": float(
                features["projected_minimum_clearance"][index]
            ),
            "models": per_model,
        })
        if len(rows) >= limit:
            break
    return rows


def _json_action(value: Mapping[str, Any]) -> int:
    move_x = int(value["move_x"])
    move_y = int(value["move_y"])
    return (move_y + 1) * 3 + move_x + 1 + (9 if bool(value["slow"]) else 0)


def _native_trace(path: Path) -> tuple[dict[str, Any], np.ndarray]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    actions = np.asarray(
        [_json_action(step["action"]) for step in raw.get("action_steps", ())],
        dtype=np.int64,
    )
    outcome = raw.get("outcome_evidence", {})
    result = {
        "path": str(path),
        "sha256": _sha256(path),
        "seed": raw.get("seed"),
        "success": bool(raw.get("success", False)),
        "termination_reason": raw.get("termination_reason"),
        "frames": int(raw.get("frames", 0)),
        "player_path_distance": outcome.get("player_path_distance"),
        "motion_activity": _motion_activity(actions),
    }
    return result, actions


def _dagger_summary(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    decisions = int(raw.get("decision_count", len(raw.get("decisions", ()))))
    interventions = int(raw.get("teacher_interventions", 0))
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "seed": raw.get("seed"),
        "teacher_assisted_success": bool(raw.get("teacher_assisted_success", False)),
        "decisions": decisions,
        "teacher_interventions": interventions,
        "teacher_intervention_rate": (
            float(interventions / decisions) if decisions else 0.0
        ),
        "safety_teacher_interventions": int(
            raw.get("safety_teacher_interventions", 0)
        ),
        "policy_disagreement_interventions": int(
            raw.get("policy_disagreement_interventions", 0)
        ),
        "scheduled_teacher_interventions": int(
            raw.get("scheduled_teacher_interventions", 0)
        ),
    }


def _pairwise_model_comparison(
    logits: Mapping[str, np.ndarray],
    human_actions: np.ndarray,
    supervised: np.ndarray,
    events: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    names = tuple(logits)
    if len(names) != 2:
        return {}
    first_name, second_name = names
    first = np.argmax(logits[first_name], axis=-1)
    second = np.argmax(logits[second_name], axis=-1)
    changed = supervised & (first != second)
    first_correct = first == human_actions
    second_correct = second == human_actions
    return {
        "models": [first_name, second_name],
        "same_prediction_fraction_all_decisions": float(np.mean(first == second)),
        "supervised_prediction_changes": int(np.count_nonzero(changed)),
        f"{second_name}_changes_improved": int(np.count_nonzero(
            changed & second_correct & ~first_correct
        )),
        f"{second_name}_changes_regressed": int(np.count_nonzero(
            changed & first_correct & ~second_correct
        )),
        "onset_prediction_changes": int(np.count_nonzero(
            events["onset"] & (first != second)
        )),
    }


def _native_pairwise(actions: Mapping[str, np.ndarray]) -> dict[str, Any]:
    names = tuple(actions)
    if len(names) != 2:
        return {}
    first_name, second_name = names
    first = actions[first_name]
    second = actions[second_name]
    count = min(len(first), len(second))
    return {
        "models": [first_name, second_name],
        "compared_prefix_decisions": count,
        "exact_agreement": float(np.mean(first[:count] == second[:count])) if count else 0.0,
        "different_actions": int(np.count_nonzero(first[:count] != second[:count])),
    }


def _pct(value: Any) -> str:
    return "n/a" if value is None else f"{float(value) * 100.0:.2f}%"


def _number(value: Any, digits: int = 2) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def _markdown(report: Mapping[str, Any]) -> str:
    model_names = tuple(report["models"])
    lines = [
        "# Human-like transition signal analysis",
        "",
        "This report uses delayed semantic vision and short visual history only for "
        "candidate supervision signals. Decision indices, checkpoint metadata, and "
        "native outcomes are diagnostics, not policy inputs.",
        "",
        "## Held-out human replay",
        "",
        "| Metric | " + " | ".join(model_names) + " |",
        "|---" + "|---:" * len(model_names) + "|",
    ]
    metric_rows = (
        ("Exact action", "exact_accuracy"),
        ("Direction", "direction_accuracy"),
        ("Speed", "speed_accuracy"),
        ("Stationary false-move", "stationary_false_move_rate"),
        ("Moving false-stop", "moving_false_stop_rate"),
    )
    for label, key in metric_rows:
        lines.append(
            f"| {label} | "
            + " | ".join(_pct(report["models"][name][key]) for name in model_names)
            + " |"
        )
    event_rows = (
        ("Onset exact", "onset"),
        ("Stop exact", "stop"),
        ("Turn exact", "turn"),
        ("Speed-change exact", "speed_change"),
    )
    for label, key in event_rows:
        lines.append(
            f"| {label} | "
            + " | ".join(
                _pct(report["models"][name][key]["exact"])
                for name in model_names
            )
            + " |"
        )
    lines.append(
        "| Onset within +/-1 decision | "
        + " | ".join(
            _pct(
                report["models"][name]["onset"]["temporal_same_direction"]
                ["within_one_fraction_of_all_labels"]
            )
            for name in model_names
        )
        + " |"
    )

    comparison = report["model_comparison"]
    if comparison:
        lines.extend((
            "",
            "## Policy delta",
            "",
            f"The two checkpoints select the same action on "
            f"{_pct(comparison['same_prediction_fraction_all_decisions'])} of all "
            f"human visual decisions. Only {comparison['supervised_prediction_changes']} "
            f"hard labels change, including {comparison['onset_prediction_changes']} "
            "movement-onset labels.",
        ))

    lines.extend((
        "",
        "## Visual boundary signals",
        "",
        "| Visual-only feature | Human onset mean | Stationary hold mean | Direction | Pairwise separability |",
        "|---|---:|---:|---|---:|",
    ))
    visual = report["visual_signals"]
    for name in (
        "local_occupancy_mass",
        "local_occupancy_growth_1",
        "local_occupancy_growth_3",
        "left_minus_right_change_1",
        "projected_minimum_clearance",
        "projected_clearance_change_1",
    ):
        item = visual[name]
        lines.append(
            f"| `{name}` | {_number(item['human_onset']['mean'])} | "
            f"{_number(item['stationary_hold']['mean'])} | "
            f"{item['onset_direction']} | "
            f"{_pct(item['pairwise_separability'])} |"
        )
    lines.extend((
        "",
        f"{_pct(visual['proactive_onset_fraction']['clearance_over_24'])} of "
        "human onsets happen while 12-frame projected cell-center clearance "
        "is still above 24 world units. This is direct evidence that a single-frame "
        "imminent-risk trigger is insufficient.",
        "",
        "## Native closed loop",
        "",
        "| Policy | Result | Frames | Path distance | Action changes | Mean action run |",
        "|---|---|---:|---:|---:|---:|",
    ))
    human_prefix = report["native_human_prefix"]
    lines.append(
        f"| human slot3 prefix | reference | n/a | n/a | "
        f"{human_prefix['action_changes']} | "
        f"{_number(human_prefix['action_runs']['mean'])} |"
    )
    for name, item in report["native_traces"].items():
        activity = item["motion_activity"]
        lines.append(
            f"| {name} | {item['termination_reason']} | {item['frames']} | "
            f"{_number(item['player_path_distance'])} | "
            f"{activity['action_changes']} | "
            f"{_number(activity['action_runs']['mean'])} |"
        )

    dagger = report["dagger_reports"]
    if dagger:
        lines.extend(("", "## DAgger diagnostic", ""))
        for name, item in dagger.items():
            lines.append(
                f"- `{name}` succeeds only with teacher assistance and intervenes "
                f"on {_pct(item['teacher_intervention_rate'])} of decisions "
                f"({item['safety_teacher_interventions']} safety, "
                f"{item['policy_disagreement_interventions']} disagreement)."
            )
        lines.extend((
            "",
            "The high-volume disagreement collection does not produce a comparable "
            "held-out or closed-loop change. Ordinary disagreement therefore needs "
            "to be separated from reliable safety and motion-boundary labels.",
        ))

    lines.extend((
        "",
        "## Conclusion",
        "",
        "The onset-only ranking change does not materially alter the controller. "
        "The remaining human-likeness gap is a two-sided motion-boundary problem: "
        "the policy must learn when to start, when to keep holding, when to stop, "
        "and when to turn or change speed. The useful causal cues are recent "
        "occupancy growth, flow asymmetry, and projected clearance, all already "
        "present in semantic vision or recoverable by the GRU.",
        "",
        "Recommended next experiment: pair each reliable onset/stop/turn label with "
        "the preceding 1-3 reliable hard states from the same strictly successful "
        "episode. Apply a per-event, per-episode-normalized two-sided contrastive "
        "rank loss. Do not add absolute frame, script phase, route, or external "
        "memory. Accept it only if onset offsets and stop/turn accuracy improve "
        "without raising stationary false-move, and native seed validation achieves "
        "strict attack completion or timeout survival.",
        "",
    ))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--checkpoint",
        action="append",
        type=_named_path,
        help="NAME=PATH; defaults to the v25 and v29 checkpoints",
    )
    parser.add_argument(
        "--native-trace",
        action="append",
        type=_named_path,
        help="NAME=PATH; defaults to the matching seed-20260730 native traces",
    )
    parser.add_argument(
        "--dagger-report",
        action="append",
        type=_named_path,
        help="NAME=PATH; defaults to the v25 and v29 DAgger reports",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--match-radius", type=int, default=5)
    parser.add_argument("--projection-horizon", type=int, default=12)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("artifacts/humanlike-transition-signals-v25-v29-slot3.json"),
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=Path("artifacts/humanlike-transition-signals-v25-v29-slot3.md"),
    )
    args = parser.parse_args()
    if args.chunk_size <= 0:
        parser.error("--chunk-size must be positive")
    if args.match_radius < 0:
        parser.error("--match-radius cannot be negative")
    if args.projection_horizon < 0:
        parser.error("--projection-horizon cannot be negative")

    checkpoints = tuple(args.checkpoint or DEFAULT_CHECKPOINTS)
    native_paths = tuple(args.native_trace or DEFAULT_NATIVE_TRACES)
    dagger_paths = tuple(args.dagger_report or DEFAULT_DAGGER_REPORTS)
    names = [name for name, _path in checkpoints]
    if len(set(names)) != len(names):
        parser.error("checkpoint names must be unique")

    with np.load(args.dataset, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    human_actions = np.asarray(data["actions"][:, -1], dtype=np.int64)
    supervised = (
        np.ones(len(human_actions), dtype=np.bool_)
        if "supervision_mask" not in data else
        np.asarray(data["supervision_mask"][:, -1], dtype=np.bool_)
    )
    events = _event_masks(human_actions, supervised)

    logits: dict[str, np.ndarray] = {}
    model_metadata: dict[str, Any] = {}
    model_metrics: dict[str, Any] = {}
    for name, path in checkpoints:
        model_logits, _model_risks, metadata = _checkpoint_predictions(
            path,
            data,
            device=args.device,
            chunk_size=args.chunk_size,
        )
        logits[name] = model_logits
        model_metadata[name] = metadata
        model_metrics[name] = _model_metrics(
            model_logits,
            human_actions,
            supervised,
            events,
            match_radius=args.match_radius,
        )

    visual_signals, feature_values = _visual_signals(
        data,
        events,
        human_actions,
        supervised,
        projection_horizon=args.projection_horizon,
    )
    native_traces: dict[str, Any] = {}
    native_actions: dict[str, np.ndarray] = {}
    for name, path in native_paths:
        summary, actions = _native_trace(path)
        native_traces[name] = summary
        native_actions[name] = actions
    native_prefix_decisions = (
        max((len(actions) for actions in native_actions.values()), default=0)
    )

    report: dict[str, Any] = {
        "schema_version": 1,
        "run_kind": "humanlike_visual_transition_signal_analysis",
        "acceptance_claim": False,
        "causal_scope": {
            "candidate_model_inputs": [
                "delayed_semantic_vision",
                "recent_semantic_vision_changes",
                "streaming_recurrent_state",
            ],
            "excluded_model_inputs": [
                "absolute_frame",
                "script_phase",
                "fixed_route",
                "waypoint",
                "external_episode_memory",
            ],
            "diagnostic_only": [
                "decision_index",
                "checkpoint_metadata",
                "native_outcome",
            ],
        },
        "dataset": {
            "path": str(args.dataset),
            "sha256": _sha256(args.dataset),
            "decisions": int(len(human_actions)),
            "supervised_hard_labels": int(np.count_nonzero(supervised)),
            "event_counts": {
                name: int(np.count_nonzero(mask))
                for name, mask in events.items()
            },
            "human_action_activity": _motion_activity(human_actions),
        },
        "analysis_config": {
            "match_radius_decisions": args.match_radius,
            "projected_clearance_horizon": args.projection_horizon,
            "semantic_velocity_scale": 8.0,
            "local_extent": 72.0,
            "streaming_checkpoint_chunk_size": args.chunk_size,
        },
        "checkpoint_metadata": model_metadata,
        "models": model_metrics,
        "model_comparison": _pairwise_model_comparison(
            logits,
            human_actions,
            supervised,
            events,
        ),
        "visual_signals": visual_signals,
        "representative_onset_errors": _boundary_examples(
            human_actions,
            events,
            logits,
            model_metrics,
            feature_values,
        ),
        "native_traces": native_traces,
        "native_human_prefix": _motion_activity(
            human_actions[:native_prefix_decisions]
        ),
        "native_trace_comparison": _native_pairwise(native_actions),
        "dagger_reports": {
            name: _dagger_summary(path)
            for name, path in dagger_paths
        },
        "finding": (
            "The onset-only rank experiment changes too few decisions to alter "
            "closed-loop behavior. Train two-sided visual motion boundaries over "
            "onset, hold, stop, turn, and speed changes, normalized per event and "
            "per successful episode."
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    args.output_markdown.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({
        "output_json": str(args.output_json),
        "output_markdown": str(args.output_markdown),
        "model_comparison": report["model_comparison"],
    }, indent=2))


if __name__ == "__main__":
    main()
