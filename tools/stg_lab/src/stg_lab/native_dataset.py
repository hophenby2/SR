"""Native-engine streaming demonstration collection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .policy import proficiency_vector
from .protocol import Action
from .provenance import file_sha256
from .training import (
    TEACHER_ACTION_EVALUATION_FIELDS,
    Demonstrations,
    previous_actions_from_targets,
)
from .vision import VisionObservation


UNKNOWN_SCENARIO_CONTEXT = "<unknown>"
SAFETY_INTERVENTION_REASONS = frozenset({
    "clearance_regret",
    "minimum_margin",
    "predicted_collision",
})
_KNOWN_INTERVENTION_REASONS = SAFETY_INTERVENTION_REASONS | {
    "policy_disagreement",
    "scheduled_teacher",
}


def _report_action_id(value: Any, *, field: str, decision: int) -> int:
    if not isinstance(value, Mapping):
        raise ValueError(f"decision {decision} {field} must be a JSON object")
    derived: int | None = None
    if all(name in value for name in ("move_x", "move_y", "slow")):
        move_x = value["move_x"]
        move_y = value["move_y"]
        slow = value["slow"]
        if (
            isinstance(move_x, bool)
            or not isinstance(move_x, int)
            or isinstance(move_y, bool)
            or not isinstance(move_y, int)
            or not isinstance(slow, bool)
        ):
            raise ValueError(
                f"decision {decision} {field} has invalid movement fields"
            )
        try:
            derived = Action(move_x=move_x, move_y=move_y, slow=slow).discrete
        except ValueError as error:
            raise ValueError(
                f"decision {decision} {field} has invalid movement fields"
            ) from error
    raw_discrete = value.get("discrete")
    if raw_discrete is None:
        if derived is None:
            raise ValueError(
                f"decision {decision} {field} has no discrete or movement label"
            )
        return derived
    if (
        isinstance(raw_discrete, bool)
        or not isinstance(raw_discrete, int)
        or not 0 <= raw_discrete < 18
    ):
        raise ValueError(
            f"decision {decision} {field}.discrete must be an integer in [0, 18)"
        )
    if derived is not None and raw_discrete != derived:
        raise ValueError(
            f"decision {decision} {field}.discrete disagrees with its movement fields"
        )
    return raw_discrete


def _report_margin(value: Any, *, field: str, decision: int) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or math.isnan(float(value))
    ):
        raise ValueError(f"decision {decision} {field} must be a non-NaN number")
    return float(value)


def _strict_dagger_completion(report: Mapping[str, Any]) -> tuple[str, str, float]:
    if report.get("run_kind") != "live_luastg_native_dagger":
        raise ValueError("source report is not a native DAgger report")
    if report.get("success") is not True or report.get("passed") is not True:
        raise ValueError("source DAgger report does not claim strict success")
    if report.get("episode_completed") is not None and (
        report.get("episode_completed") is not True
    ):
        raise ValueError("source DAgger report does not claim episode completion")
    episode_kind = report.get("episode_kind")
    if episode_kind not in ("attack", "stage"):
        raise ValueError("source DAgger report has an invalid episode_kind")
    completion_reason = (
        "attack_complete" if episode_kind == "attack" else "stage_complete"
    )
    if report.get("terminated") is not True:
        raise ValueError("source DAgger episode was not terminated")
    if report.get("termination_reason") != completion_reason:
        raise ValueError(
            "source DAgger episode did not reach " + completion_reason
        )
    engine_reason = report.get("engine_termination_reason")
    if engine_reason is not None and engine_reason != completion_reason:
        raise ValueError("source DAgger engine termination evidence disagrees")
    outcome = report.get("outcome_evidence")
    final_player = outcome.get("final_player") if isinstance(outcome, Mapping) else None
    death = final_player.get("death") if isinstance(final_player, Mapping) else None
    if (
        isinstance(death, bool)
        or not isinstance(death, (int, float))
        or not math.isfinite(float(death))
        or float(death) != 0.0
    ):
        raise ValueError(
            "source DAgger outcome_evidence.final_player.death is not zero"
        )
    return episode_kind, completion_reason, float(death)


def relabel_dagger_demonstration_archive(
    demonstrations_path: str | Path,
    dagger_report_path: str | Path,
    output_path: str | Path,
    manifest_path: str | Path,
    *,
    interventions_only: bool = False,
    safety_interventions_only: bool = False,
    hard_intervention_labels: bool = False,
    teacher_evaluation_context: int | None = None,
    minimum_safety_margin_gain: float | None = None,
) -> dict[str, Any]:
    """Keep safe student executions and teacher corrections as DAgger labels."""

    source = Path(demonstrations_path)
    report_source = Path(dagger_report_path)
    output = Path(output_path)
    manifest_output = Path(manifest_path)
    if output.suffix.lower() != ".npz":
        raise ValueError("corrective DAgger output must use the .npz extension")
    if teacher_evaluation_context is not None and (
        isinstance(teacher_evaluation_context, bool)
        or not isinstance(teacher_evaluation_context, int)
        or teacher_evaluation_context < 0
    ):
        raise ValueError("teacher_evaluation_context must be a nonnegative integer")
    if hard_intervention_labels and not interventions_only:
        raise ValueError(
            "hard_intervention_labels requires interventions_only"
        )
    if safety_interventions_only and not interventions_only:
        raise ValueError(
            "safety_interventions_only requires interventions_only"
        )
    if minimum_safety_margin_gain is not None:
        if (
            isinstance(minimum_safety_margin_gain, bool)
            or not isinstance(minimum_safety_margin_gain, (int, float))
            or not math.isfinite(float(minimum_safety_margin_gain))
            or float(minimum_safety_margin_gain) < 0.0
        ):
            raise ValueError(
                "minimum_safety_margin_gain must be finite and nonnegative"
            )
        if not safety_interventions_only:
            raise ValueError(
                "minimum_safety_margin_gain requires safety_interventions_only"
            )
        minimum_safety_margin_gain = float(minimum_safety_margin_gain)
    resolved = {
        "demonstrations": source.resolve(),
        "report": report_source.resolve(),
        "output": output.resolve(),
        "manifest": manifest_output.resolve(),
    }
    if len(set(resolved.values())) != len(resolved):
        raise ValueError("corrective DAgger inputs and outputs must be distinct files")

    try:
        report = json.loads(report_source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"source DAgger report is not valid JSON: {error}") from error
    if not isinstance(report, Mapping):
        raise ValueError("source DAgger report must contain a JSON object")
    episode_kind, completion_reason, death = _strict_dagger_completion(report)
    implementation_sha256 = report.get("implementation_sha256")
    if (
        not isinstance(implementation_sha256, str)
        or len(implementation_sha256) != 64
        or any(character not in "0123456789abcdef" for character in implementation_sha256)
    ):
        raise ValueError(
            "source DAgger report implementation_sha256 must be 64 lowercase hex characters"
        )
    scenario = report.get("scenario")
    if not isinstance(scenario, str) or not scenario.strip():
        raise ValueError("source DAgger report has an invalid scenario identity")
    seed = report.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("source DAgger report has an invalid seed identity")
    attack = report.get("attack")
    if episode_kind == "attack":
        if (
            isinstance(attack, bool)
            or not isinstance(attack, int)
            or attack <= 0
        ):
            raise ValueError("source DAgger attack identity must be a positive integer")
    elif attack is not None:
        raise ValueError("source DAgger stage identity cannot include an attack")

    # Validate through the public schema, then preserve every stored array so
    # relabeling remains lossless as optional dataset fields evolve.
    demonstrations = Demonstrations.load(source)
    with np.load(source, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    actions = demonstrations.actions
    if actions.shape[1] != 1:
        raise ValueError(
            "corrective DAgger requires one streaming action label per sample"
        )
    if (
        not np.issubdtype(actions.dtype, np.integer)
        or np.issubdtype(actions.dtype, np.bool_)
    ):
        raise ValueError("source DAgger action labels must use an integer dtype")

    decisions = report.get("decisions")
    if not isinstance(decisions, list):
        raise ValueError("source DAgger report has no decisions list")
    sample_count = int(actions.shape[0])
    if sample_count <= 0:
        raise ValueError("source DAgger archive contains no decision samples")
    if len(decisions) != sample_count:
        raise ValueError(
            "source DAgger decision count does not match demonstration samples "
            f"({len(decisions)} decisions for {sample_count} samples)"
        )
    declared_count = report.get("decision_count")
    if (
        isinstance(declared_count, bool)
        or not isinstance(declared_count, int)
        or declared_count != sample_count
    ):
        raise ValueError("source DAgger decision_count does not match its decisions")

    teacher_ids = np.empty(sample_count, dtype=np.int64)
    executed_ids = np.empty(sample_count, dtype=np.int64)
    student_ids = np.empty(sample_count, dtype=np.int64)
    supervised_ids = np.empty(sample_count, dtype=np.int64)
    intervention_flags = np.empty(sample_count, dtype=bool)
    student_margins = np.empty(sample_count, dtype=np.float64)
    teacher_margins = np.empty(sample_count, dtype=np.float64)
    intervention_reasons: list[str | None] = []
    supervision = report.get("demonstration_supervision")
    source_supervision_mode = (
        supervision.get("mode") if isinstance(supervision, Mapping) else "teacher"
    )
    if source_supervision_mode not in {"teacher", "corrective"}:
        raise ValueError("source DAgger report has an invalid supervision mode")
    for index, decision in enumerate(decisions):
        if not isinstance(decision, Mapping):
            raise ValueError(f"decision {index} must be a JSON object")
        decision_index = decision.get("decision")
        if (
            isinstance(decision_index, bool)
            or not isinstance(decision_index, int)
            or decision_index != index
        ):
            raise ValueError(f"decision {index} has a non-contiguous decision index")
        teacher_ids[index] = _report_action_id(
            decision.get("teacher_action"), field="teacher_action", decision=index,
        )
        executed_ids[index] = _report_action_id(
            decision.get("executed_action"), field="executed_action", decision=index,
        )
        student_ids[index] = _report_action_id(
            decision.get("student_action"), field="student_action", decision=index,
        )
        if source_supervision_mode == "corrective":
            supervised_ids[index] = _report_action_id(
                decision.get("supervised_action"),
                field="supervised_action",
                decision=index,
            )
        else:
            supervised_ids[index] = teacher_ids[index]
        intervened = decision.get("teacher_intervened")
        if not isinstance(intervened, bool):
            raise ValueError(f"decision {index} teacher_intervened must be boolean")
        intervention_flags[index] = intervened
        if minimum_safety_margin_gain is not None:
            student_margins[index] = _report_margin(
                decision.get("student_predicted_minimum_margin"),
                field="student_predicted_minimum_margin",
                decision=index,
            )
            teacher_margins[index] = _report_margin(
                decision.get("teacher_predicted_minimum_margin"),
                field="teacher_predicted_minimum_margin",
                decision=index,
            )
        if safety_interventions_only:
            reason = decision.get("intervention_reason")
            if intervened:
                if not isinstance(reason, str) or reason not in (
                    _KNOWN_INTERVENTION_REASONS
                ):
                    raise ValueError(
                        f"decision {index} has an invalid intervention_reason"
                    )
            elif reason is not None:
                raise ValueError(
                    f"decision {index} has intervention_reason without "
                    "teacher_intervened=true"
                )
            intervention_reasons.append(reason)
        expected_executed = teacher_ids[index] if intervened else student_ids[index]
        if executed_ids[index] != expected_executed:
            source_name = "teacher" if intervened else "student"
            raise ValueError(
                f"decision {index} executed action disagrees with the {source_name} action"
            )
        expected_supervised = (
            expected_executed
            if source_supervision_mode == "corrective" else
            teacher_ids[index]
        )
        if supervised_ids[index] != expected_supervised:
            raise ValueError(
                f"decision {index} supervised action disagrees with "
                f"{source_supervision_mode} supervision"
            )
        agreement = decision.get("student_teacher_agreement")
        expected_agreement = student_ids[index] == teacher_ids[index]
        if not isinstance(agreement, bool) or agreement != expected_agreement:
            raise ValueError(
                f"decision {index} has inconsistent student_teacher_agreement"
            )

    if not np.array_equal(actions[:, 0], supervised_ids):
        mismatches = np.flatnonzero(actions[:, 0] != supervised_ids)
        source_label_field = (
            "supervised_action"
            if source_supervision_mode == "corrective" else
            "teacher_action"
        )
        raise ValueError(
            f"source demonstration labels do not match {source_label_field} at decision "
            f"{int(mismatches[0])}"
        )
    intervention_count = int(np.count_nonzero(intervention_flags))
    declared_interventions = report.get("teacher_interventions")
    if (
        isinstance(declared_interventions, bool)
        or not isinstance(declared_interventions, int)
        or declared_interventions != intervention_count
    ):
        raise ValueError("source DAgger teacher_interventions count is inconsistent")
    selected_intervention_flags = intervention_flags
    safety_intervention_count: int | None = None
    policy_disagreement_count: int | None = None
    scheduled_intervention_count: int | None = None
    excluded_same_action_count: int | None = None
    excluded_insufficient_gain_count: int | None = None
    if safety_interventions_only:
        safety_flags = np.fromiter(
            (
                reason in SAFETY_INTERVENTION_REASONS
                for reason in intervention_reasons
            ),
            dtype=np.bool_,
            count=sample_count,
        )
        policy_disagreement_flags = np.fromiter(
            (reason == "policy_disagreement" for reason in intervention_reasons),
            dtype=np.bool_,
            count=sample_count,
        )
        scheduled_intervention_flags = np.fromiter(
            (reason == "scheduled_teacher" for reason in intervention_reasons),
            dtype=np.bool_,
            count=sample_count,
        )
        safety_intervention_count = int(np.count_nonzero(safety_flags))
        policy_disagreement_count = int(np.count_nonzero(
            policy_disagreement_flags,
        ))
        scheduled_intervention_count = int(np.count_nonzero(
            scheduled_intervention_flags,
        ))
        for field, actual in (
            ("safety_teacher_interventions", safety_intervention_count),
            (
                "policy_disagreement_interventions",
                policy_disagreement_count,
            ),
            ("scheduled_teacher_interventions", scheduled_intervention_count),
        ):
            declared = report.get(field)
            if (
                isinstance(declared, bool)
                or not isinstance(declared, int)
                or declared != actual
            ):
                raise ValueError(
                    f"source DAgger {field} count is inconsistent"
                )
        if (
            safety_intervention_count
            + policy_disagreement_count
            + scheduled_intervention_count
            != intervention_count
        ):
            raise ValueError(
                "source DAgger intervention reason counts do not sum to "
                "teacher_interventions"
            )
        selected_intervention_flags = safety_flags
        if minimum_safety_margin_gain is not None:
            action_change_flags = student_ids != teacher_ids
            with np.errstate(invalid="ignore"):
                margin_gains = teacher_margins - student_margins
                sufficient_gain_flags = (
                    margin_gains >= minimum_safety_margin_gain
                )
            selected_intervention_flags = (
                safety_flags & action_change_flags & sufficient_gain_flags
            )
            excluded_same_action_count = int(np.count_nonzero(
                safety_flags & ~action_change_flags,
            ))
            excluded_insufficient_gain_count = int(np.count_nonzero(
                safety_flags & action_change_flags & ~sufficient_gain_flags,
            ))
    selected_intervention_count = int(np.count_nonzero(
        selected_intervention_flags,
    ))
    agreement_count = int(np.count_nonzero(student_ids == teacher_ids))
    declared_agreements = report.get("student_teacher_agreements")
    if (
        isinstance(declared_agreements, bool)
        or not isinstance(declared_agreements, int)
        or declared_agreements != agreement_count
    ):
        raise ValueError("source DAgger student_teacher_agreements count is inconsistent")

    relabeled_actions = actions.copy()
    relabeled_actions[:, 0] = executed_ids
    arrays["actions"] = relabeled_actions
    correction_flags = np.zeros(sample_count, dtype=np.bool_)
    if (
        safety_interventions_only
        and minimum_safety_margin_gain is not None
    ):
        correction_flags = selected_intervention_flags.copy()
    arrays["correction_mask"] = correction_flags.reshape(-1, 1).astype(
        np.uint8,
    )
    if interventions_only:
        arrays["supervision_mask"] = selected_intervention_flags.reshape(
            -1, 1,
        ).astype(np.uint8)
    if teacher_evaluation_context is not None:
        if "teacher_action_evaluation_mask" not in arrays:
            raise ValueError(
                "teacher evaluation context requires recorded action evaluations"
            )
        context_mask = np.zeros(sample_count, dtype=np.bool_)
        for index in np.flatnonzero(intervention_flags):
            start = max(0, int(index) - teacher_evaluation_context)
            stop = min(sample_count, int(index) + teacher_evaluation_context + 1)
            context_mask[start:stop] = True
        original_evaluation_mask = np.asarray(
            arrays["teacher_action_evaluation_mask"][:, 0], dtype=np.bool_,
        )
        context_mask &= original_evaluation_mask
        arrays["teacher_action_evaluation_mask"] = context_mask.reshape(
            -1, 1,
        ).astype(np.uint8)
    hard_intervention_evaluations_cleared = 0
    if (
        hard_intervention_labels
        and "teacher_action_evaluation_mask" in arrays
    ):
        evaluation_mask = np.asarray(
            arrays["teacher_action_evaluation_mask"][:, 0], dtype=np.bool_,
        ).copy()
        hard_intervention_evaluations_cleared = int(np.count_nonzero(
            evaluation_mask & selected_intervention_flags
        ))
        evaluation_mask[selected_intervention_flags] = False
        arrays["teacher_action_evaluation_mask"] = evaluation_mask.reshape(
            -1, 1,
        ).astype(np.uint8)
    soft_evaluation_count = (
        int(np.count_nonzero(arrays["teacher_action_evaluation_mask"]))
        if "teacher_action_evaluation_mask" in arrays else
        0
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    # Re-open through the public schema before publishing provenance.
    Demonstrations.load(output)

    replaced_count = int(np.count_nonzero(actions[:, 0] != executed_ids))
    manifest = {
        "schema_version": 1,
        "run_kind": "corrective_dagger_relabel",
        "acceptance_claim": False,
        "training_only": True,
        "strict_inclusion_criterion": (
            "terminated with attack_complete or stage_complete and no player death"
        ),
        "source_dataset": str(source),
        "source_dataset_sha256": file_sha256(source),
        "source_dagger_report": str(report_source),
        "source_dagger_report_sha256": file_sha256(report_source),
        "source_implementation_sha256": implementation_sha256,
        "dataset": str(output),
        "dataset_sha256": file_sha256(output),
        "samples": sample_count,
        "history": 1,
        "decisions": sample_count,
        "teacher_interventions": intervention_count,
        **(
            {
                "total_teacher_interventions": intervention_count,
                "selected_safety_interventions": selected_intervention_count,
                "excluded_policy_disagreement_interventions": (
                    policy_disagreement_count
                ),
                "excluded_scheduled_teacher_interventions": (
                    scheduled_intervention_count
                ),
                **(
                    {
                        "candidate_safety_interventions": (
                            safety_intervention_count
                        ),
                        "minimum_safety_margin_gain": (
                            minimum_safety_margin_gain
                        ),
                        "excluded_same_action_interventions": (
                            excluded_same_action_count
                        ),
                        "excluded_insufficient_margin_gain_interventions": (
                            excluded_insufficient_gain_count
                        ),
                    }
                    if minimum_safety_margin_gain is not None else
                    {}
                ),
            }
            if safety_interventions_only else
            {}
        ),
        "student_executions": sample_count - intervention_count,
        "source_supervision_mode": source_supervision_mode,
        "replaced_labels": replaced_count,
        "unchanged_labels": sample_count - replaced_count,
        "preserved_arrays": sorted(
            name
            for name in arrays
            if name != "actions"
            and name != "correction_mask"
            and not (interventions_only and name == "supervision_mask")
            and not (
                (
                    teacher_evaluation_context is not None
                    or hard_intervention_labels
                )
                and name == "teacher_action_evaluation_mask"
            )
        ),
        "teacher_action_evaluations_preserved": (
            demonstrations.teacher_action_evaluations is not None
        ),
        "teacher_action_evaluation_fields": (
            list(TEACHER_ACTION_EVALUATION_FIELDS)
            if demonstrations.teacher_action_evaluations is not None else
            None
        ),
        "teacher_action_evaluation_supervision": {
            "mode": (
                (
                    "teacher_intervention_context_except_hard_interventions"
                    if hard_intervention_labels else
                    "teacher_intervention_context"
                )
                if teacher_evaluation_context is not None else
                (
                    "all_recorded_evaluations_except_hard_interventions"
                    if hard_intervention_labels else
                    "all_recorded_evaluations"
                )
            ),
            "context_radius_decisions": teacher_evaluation_context,
            "available_decisions": soft_evaluation_count,
            "model_input": False,
        },
        "action_supervision": {
            "mode": (
                (
                    "effective_safety_interventions_only"
                    if minimum_safety_margin_gain is not None else
                    "safety_teacher_interventions_only"
                ) if safety_interventions_only else
                (
                    "teacher_interventions_only"
                    if interventions_only else
                    "all_executed_actions"
                )
            ),
            "mask": "supervision_mask",
            "supervised_decisions": (
                selected_intervention_count if interventions_only else sample_count
            ),
            "unsupervised_context_decisions": (
                sample_count - selected_intervention_count
                if interventions_only else
                0
            ),
            "recurrent_context_decisions": sample_count,
            "risk_targets_available": sample_count,
        },
        **(
            {
                "safety_intervention_selection": {
                    "enabled": True,
                    "accepted_reasons": sorted(SAFETY_INTERVENTION_REASONS),
                    "candidate_safety_interventions": safety_intervention_count,
                    "total_teacher_interventions": intervention_count,
                    "selected_safety_interventions": selected_intervention_count,
                    "excluded_policy_disagreement_interventions": (
                        policy_disagreement_count
                    ),
                    "excluded_scheduled_teacher_interventions": (
                        scheduled_intervention_count
                    ),
                    "requires_action_change": (
                        minimum_safety_margin_gain is not None
                    ),
                    "minimum_teacher_margin_gain": minimum_safety_margin_gain,
                    **(
                        {
                            "excluded_same_action_interventions": (
                                excluded_same_action_count
                            ),
                            "excluded_insufficient_margin_gain_interventions": (
                                excluded_insufficient_gain_count
                            ),
                        }
                        if minimum_safety_margin_gain is not None else
                        {}
                    ),
                    "semantics": (
                        "hard action supervision selects safety interventions "
                        "where the teacher changes the student action and improves "
                        "predicted minimum margin by at least the configured gain; "
                        "all rejected interventions remain recurrent context"
                        if minimum_safety_margin_gain is not None else
                        "hard action supervision selects only teacher_intervened "
                        "rows caused by predicted collision, minimum margin, or "
                        "clearance regret; policy disagreement and scheduled "
                        "teacher rows remain recurrent context"
                    ),
                },
            }
            if safety_interventions_only else
            {}
        ),
        **(
            {
                "hard_intervention_label_supervision": {
                    "enabled": True,
                    "requires_action_supervision": (
                        (
                            "effective_safety_interventions_only"
                            if minimum_safety_margin_gain is not None else
                            "safety_teacher_interventions_only"
                        ) if safety_interventions_only else
                        "teacher_interventions_only"
                    ),
                    "hard_label_decisions": selected_intervention_count,
                    "soft_evaluation_rows_cleared": (
                        hard_intervention_evaluations_cleared
                    ),
                    "remaining_soft_evaluation_decisions": soft_evaluation_count,
                    "semantics": (
                        "selected teacher intervention rows use the executed "
                        "teacher action as an exclusive hard label; all other "
                        "retained teacher evaluation rows, including policy "
                        "disagreements, remain set-valued soft targets"
                        if safety_interventions_only else
                        "teacher_intervened rows use the executed teacher action "
                        "as an exclusive hard label; all other retained teacher "
                        "evaluation rows remain set-valued soft targets"
                    ),
                },
            }
            if hard_intervention_labels else
            {}
        ),
        "accepted_episodes": [{
            "episode_kind": episode_kind,
            "scenario": scenario,
            "attack": attack,
            "seed": seed,
            "profile": str(report.get("profile", "corrective_dagger")),
            "decisions": sample_count,
            "strict_success": True,
            "termination_reason": completion_reason,
        }],
        "source_outcome_evidence": {
            "terminated": True,
            "termination_reason": completion_reason,
            "final_player_death": death,
        },
        "label_semantics": {
            "source": (
                "executed_action.discrete from corrective DAgger collection"
                if source_supervision_mode == "corrective" else
                "teacher_action.discrete for every visited state"
            ),
            "output": (
                "executed_action.discrete: retain the student's executed label "
                "when unassisted and the teacher correction when intervened"
            ),
            "supervision_mask": (
                "true only for safety interventions whose teacher action differs "
                "and improves predicted minimum margin by the configured amount"
                if minimum_safety_margin_gain is not None else
                "true only where teacher_intervened=true and intervention_reason "
                "is predicted_collision, minimum_margin, or clearance_regret"
                if safety_interventions_only else
                (
                    "true only where teacher_intervened=true"
                    if interventions_only else
                    "true for every executed action"
                )
            ),
            "correction_mask": (
                "true only for selected safety interventions whose teacher "
                "action differs from the student action and improves predicted "
                "minimum margin by at least the configured threshold; false "
                "for every other decision"
                if (
                    safety_interventions_only
                    and minimum_safety_margin_gain is not None
                ) else
                "false for every decision because this relabel operation did "
                "not select effective safety corrections by margin gain"
            ),
            **(
                {
                    "teacher_action_evaluation_mask": (
                        "false at selected safety intervention rows so the "
                        "executed teacher action is an exclusive hard label; "
                        "unchanged at other retained rows, including "
                        "policy_disagreement interventions"
                        if safety_interventions_only else
                        "false at teacher_intervened rows so the executed teacher "
                        "action is an exclusive hard label; unchanged at every "
                        "other retained teacher evaluation row"
                    ),
                }
                if hard_intervention_labels else
                {}
            ),
        },
    }
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def risk_from_clearance(clearance: float, *, scale: float = 8.0) -> float:
    """Map teacher clearance to a bounded danger target without exposing it."""

    if not math.isfinite(clearance):
        return 0.0 if clearance > 0.0 else 1.0
    margin = max(0.0, float(clearance))
    return float(scale / (scale + margin))


@dataclass(frozen=True, slots=True)
class NativeEpisodeIdentity:
    episode_kind: str
    scenario: str
    attack: int | None
    seed: int
    profile: str = "current"


def episode_context_key(
    episode_kind: str,
    scenario: str,
    attack: int | None,
) -> str:
    """Return an identity-only context key with no route or phase information."""

    kind = str(episode_kind).strip().lower()
    name = str(scenario).strip()
    if not name:
        raise ValueError("episode scenario cannot be empty")
    if kind == "stage":
        if attack is not None:
            raise ValueError("stage context cannot include an attack number")
        return f"stage:{name}"
    if kind != "attack":
        raise ValueError("episode_kind must be 'attack' or 'stage'")
    if isinstance(attack, bool) or not isinstance(attack, int) or attack <= 0:
        raise ValueError("attack context requires a positive attack number")
    return f"attack:{name}#{attack}"


def _manifest_archive_path(value: Any, manifest_path: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"manifest {manifest_path} contains an invalid archive path")
    requested = Path(value)
    candidates = (
        requested,
        manifest_path.parent / requested,
        manifest_path.parent / requested.name,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise ValueError(
        f"manifest {manifest_path} references missing archive {value!r}"
    )


def _episode_identities_from_manifest(
    manifest_path: str | Path,
    *,
    _visited: set[Path] | None = None,
) -> tuple[NativeEpisodeIdentity, ...]:
    """Resolve leaf episode identities through nested merge manifests."""

    path = Path(manifest_path).resolve()
    visited = set() if _visited is None else _visited
    if path in visited:
        raise ValueError(f"demonstration manifest cycle at {path}")
    visited.add(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError(f"manifest {path} must contain a JSON object")
        accepted = payload.get("accepted_episodes")
        if isinstance(accepted, list):
            identities: list[NativeEpisodeIdentity] = []
            for index, item in enumerate(accepted):
                if not isinstance(item, Mapping):
                    raise ValueError(
                        f"manifest {path} accepted_episodes[{index}] is invalid"
                    )
                attack = item.get("attack")
                if attack is not None:
                    if isinstance(attack, bool) or not isinstance(attack, int):
                        raise ValueError(
                            f"manifest {path} has a non-integer attack identity"
                        )
                seed = item.get("seed")
                if isinstance(seed, bool) or not isinstance(seed, int):
                    raise ValueError(f"manifest {path} has an invalid episode seed")
                identities.append(NativeEpisodeIdentity(
                    episode_kind=str(item.get("episode_kind", "")),
                    scenario=str(item.get("scenario", "")),
                    attack=attack,
                    seed=seed,
                    profile=str(item.get("profile", "current")),
                ))
            if not identities:
                raise ValueError(f"manifest {path} contains no accepted episodes")
            return tuple(identities)

        inputs = payload.get("inputs")
        if not isinstance(inputs, list) or not inputs:
            raise ValueError(
                f"manifest {path} has neither accepted_episodes nor merge inputs"
            )
        merged: list[NativeEpisodeIdentity] = []
        for index, item in enumerate(inputs):
            if not isinstance(item, Mapping):
                raise ValueError(f"manifest {path} inputs[{index}] is invalid")
            archive = _manifest_archive_path(item.get("path"), path)
            child_manifest = archive.with_suffix(".manifest.json")
            if not child_manifest.is_file():
                raise ValueError(
                    f"archive {archive} has no companion .manifest.json"
                )
            merged.extend(_episode_identities_from_manifest(
                child_manifest,
                _visited=visited,
            ))
        return tuple(merged)
    finally:
        visited.remove(path)


def contextualize_demonstrations(
    demonstrations: Demonstrations,
    identities: Sequence[NativeEpisodeIdentity],
    *,
    include_previous_action: bool = False,
    include_proficiency: bool = False,
) -> tuple[Demonstrations, tuple[str, ...], tuple[str, ...]]:
    """Attach a deterministic one-hot identity token to every episode.

    The identity token contains only the registered stage or attack. Optional
    prior-motor context comes from recorded execution, and optional proficiency
    is constant within an episode. Neither includes positions, timing, future
    actions, or routes.
    """

    demonstrations.validate()
    if demonstrations.episode_ids is None:
        raise ValueError("episode_ids are required for scenario conditioning")
    ordered_ids: list[int] = []
    for value in demonstrations.episode_ids:
        episode_id = int(value)
        if not ordered_ids or ordered_ids[-1] != episode_id:
            if episode_id in ordered_ids:
                raise ValueError("episode samples must form contiguous blocks")
            ordered_ids.append(episode_id)
    if len(ordered_ids) != len(identities):
        raise ValueError(
            "manifest episode count does not match the demonstration archive "
            f"({len(identities)} identities for {len(ordered_ids)} groups)"
        )
    contexts = tuple(
        episode_context_key(
            identity.episode_kind,
            identity.scenario,
            identity.attack,
        )
        for identity in identities
    )
    vocabulary = (
        UNKNOWN_SCENARIO_CONTEXT,
        *sorted(set(contexts)),
    )
    context_indices = {value: index for index, value in enumerate(vocabulary)}
    previous_action_size = 18 if include_previous_action else 0
    memory = np.zeros(
        (*demonstrations.actions.shape, len(vocabulary) + previous_action_size),
        dtype=np.float32,
    )
    for episode_id, context in zip(ordered_ids, contexts, strict=True):
        memory[demonstrations.episode_ids == episode_id, :, context_indices[context]] = 1.0
    if include_previous_action:
        previous_actions = (
            demonstrations.previous_actions
            if demonstrations.previous_actions is not None else
            previous_actions_from_targets(demonstrations)
        )
        for action_id in range(18):
            memory[:, :, len(vocabulary) + action_id] = (
                previous_actions == action_id
            )
    proficiency = demonstrations.proficiency
    if include_proficiency:
        annotated = np.zeros(
            (*demonstrations.actions.shape, len(proficiency_vector("expert"))),
            dtype=np.float32,
        )
        for episode_id, identity in zip(ordered_ids, identities, strict=True):
            annotated[demonstrations.episode_ids == episode_id, :] = (
                proficiency_vector(identity.profile)
            )
        if proficiency is not None and not np.allclose(
            proficiency,
            annotated,
            rtol=0.0,
            atol=1e-6,
        ):
            raise ValueError(
                "demonstration proficiency does not match manifest episode profiles"
            )
        proficiency = annotated
    result = Demonstrations(
        global_frames=demonstrations.global_frames,
        local_frames=demonstrations.local_frames,
        actions=demonstrations.actions,
        risks=demonstrations.risks,
        previous_actions=demonstrations.previous_actions,
        memory=memory,
        proficiency=proficiency,
        episode_ids=demonstrations.episode_ids,
        supervision_mask=demonstrations.supervision_mask,
        teacher_action_evaluations=demonstrations.teacher_action_evaluations,
        teacher_action_regrets=demonstrations.teacher_action_regrets,
        teacher_action_evaluation_mask=(
            demonstrations.teacher_action_evaluation_mask
        ),
        correction_mask=demonstrations.correction_mask,
    )
    result.validate()
    return result, tuple(vocabulary), contexts


def contextualize_demonstration_archive(
    demonstrations_path: str | Path,
    manifest_path: str | Path,
    output_path: str | Path,
    *,
    include_previous_action: bool = False,
    include_proficiency: bool = False,
) -> dict[str, Any]:
    """Build a context-conditioned archive from strict native provenance."""

    source = Path(demonstrations_path)
    manifest = Path(manifest_path)
    identities = _episode_identities_from_manifest(manifest)
    contextualized, vocabulary, contexts = contextualize_demonstrations(
        Demonstrations.load(source),
        identities,
        include_previous_action=include_previous_action,
        include_proficiency=include_proficiency,
    )
    output = Path(output_path)
    contextualized.save(output)
    return {
        "schema_version": 1,
        "run_kind": "scenario_context_annotation",
        "acceptance_claim": False,
        "source": str(source),
        "source_sha256": file_sha256(source),
        "source_manifest": str(manifest),
        "source_manifest_sha256": file_sha256(manifest),
        "output": str(output),
        "output_sha256": file_sha256(output),
        "samples": int(contextualized.actions.shape[0]),
        "episode_groups": len(identities),
        "scenario_vocabulary": list(vocabulary),
        "scenario_context_size": len(vocabulary),
        "previous_action_size": 18 if include_previous_action else 0,
        "previous_action_offset": len(vocabulary),
        "proficiency_size": (
            0
            if contextualized.proficiency is None else
            int(contextualized.proficiency.shape[-1])
        ),
        "proficiency_profiles": (
            sorted({identity.profile for identity in identities})
            if include_proficiency else
            None
        ),
        "teacher_action_evaluations_preserved": (
            contextualized.teacher_action_evaluations is not None
        ),
        "episode_contexts": [
            {
                **asdict(identity),
                "context_key": context,
            }
            for identity, context in zip(identities, contexts, strict=True)
        ],
        "context_semantics": (
            "identity-only one-hot token plus optional prior executed motor "
            "action and episode-level proficiency; excludes coordinates, frames, "
            "phases, waypoints, and routes"
        ),
    }


class NativeEpisodeBuffer:
    """One pending episode, committed only after strict native completion."""

    def __init__(self, identity: NativeEpisodeIdentity) -> None:
        self.identity = identity
        self.global_frames: list[np.ndarray] = []
        self.local_frames: list[np.ndarray] = []
        self.actions: list[int] = []
        self.previous_actions: list[int] = []
        self.risks: list[float] = []
        self.teacher_action_evaluations: list[np.ndarray] = []
        self.teacher_action_regrets: list[np.ndarray] = []
        self.teacher_action_evaluation_mask: list[bool] = []

    def record(
        self,
        visible: VisionObservation,
        action: Action,
        risk: float,
        previous_action: Action | None = None,
        teacher_action_evaluations: np.ndarray | None = None,
        teacher_action_regrets: np.ndarray | None = None,
    ) -> None:
        if visible.global_frames.shape[0] != 1 or visible.local_frames.shape[0] != 1:
            raise ValueError("native stream demonstrations must contain one latest frame")
        if (teacher_action_evaluations is None) != (teacher_action_regrets is None):
            raise ValueError(
                "teacher action evaluations and regrets must be recorded together"
            )
        if teacher_action_evaluations is None:
            evaluations = np.zeros(
                (18, len(TEACHER_ACTION_EVALUATION_FIELDS)), dtype=np.float32,
            )
            regrets = np.zeros(18, dtype=np.float32)
            evaluation_available = False
        else:
            evaluations = np.asarray(teacher_action_evaluations, dtype=np.float32)
            regrets = np.asarray(teacher_action_regrets, dtype=np.float32)
            if evaluations.shape != (
                18,
                len(TEACHER_ACTION_EVALUATION_FIELDS),
            ):
                raise ValueError("teacher action evaluation shape is invalid")
            if regrets.shape != (18,):
                raise ValueError("teacher action regret shape is invalid")
            if np.isnan(evaluations).any():
                raise ValueError("teacher action evaluations cannot contain NaN")
            if not np.isfinite(regrets).all() or np.any(regrets < 0.0):
                raise ValueError(
                    "teacher action regrets must be finite and nonnegative"
                )
            evaluation_available = True
        self.global_frames.append(visible.global_frames.copy())
        self.local_frames.append(visible.local_frames.copy())
        self.previous_actions.append(
            previous_action.discrete
            if previous_action is not None else
            (self.actions[-1] if self.actions else -1)
        )
        self.actions.append(action.discrete)
        self.risks.append(float(np.clip(risk, 0.0, 1.0)))
        self.teacher_action_evaluations.append(evaluations.copy())
        self.teacher_action_regrets.append(regrets.copy())
        self.teacher_action_evaluation_mask.append(evaluation_available)

    @property
    def decisions(self) -> int:
        return len(self.actions)


class NativeDemonstrationBuilder:
    """Merge strictly successful native episodes into one recurrent archive."""

    def __init__(self) -> None:
        self._accepted: list[NativeEpisodeBuffer] = []
        self._rejected: list[dict[str, Any]] = []

    def begin(self, identity: NativeEpisodeIdentity) -> NativeEpisodeBuffer:
        return NativeEpisodeBuffer(identity)

    def finish(
        self,
        episode: NativeEpisodeBuffer,
        *,
        strict_success: bool,
        termination_reason: str,
    ) -> None:
        evidence = {
            **asdict(episode.identity),
            "decisions": episode.decisions,
            "strict_success": bool(strict_success),
            "termination_reason": str(termination_reason),
        }
        if strict_success:
            if episode.decisions <= 0:
                raise ValueError("a successful native episode contains no decisions")
            self._accepted.append(episode)
        else:
            self._rejected.append(evidence)

    @property
    def accepted_count(self) -> int:
        return len(self._accepted)

    def build(self) -> Demonstrations:
        if not self._accepted:
            raise ValueError("no strictly successful native episodes were collected")
        global_frames: list[np.ndarray] = []
        local_frames: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        previous_actions: list[np.ndarray] = []
        risks: list[np.ndarray] = []
        teacher_action_evaluations: list[np.ndarray] = []
        teacher_action_regrets: list[np.ndarray] = []
        teacher_action_evaluation_mask: list[np.ndarray] = []
        episode_ids: list[int] = []
        for episode_id, episode in enumerate(self._accepted):
            global_frames.extend(episode.global_frames)
            local_frames.extend(episode.local_frames)
            actions.extend(
                np.asarray([value], dtype=np.int64) for value in episode.actions
            )
            previous_actions.extend(
                np.asarray([value], dtype=np.int64)
                for value in episode.previous_actions
            )
            risks.extend(
                np.asarray([value], dtype=np.float32) for value in episode.risks
            )
            teacher_action_evaluations.extend(
                np.asarray([value], dtype=np.float32)
                for value in episode.teacher_action_evaluations
            )
            teacher_action_regrets.extend(
                np.asarray([value], dtype=np.float32)
                for value in episode.teacher_action_regrets
            )
            teacher_action_evaluation_mask.extend(
                np.asarray([value], dtype=bool)
                for value in episode.teacher_action_evaluation_mask
            )
            episode_ids.extend([episode_id] * episode.decisions)
        has_teacher_evaluations = any(
            value.item() for value in teacher_action_evaluation_mask
        )
        demonstrations = Demonstrations(
            global_frames=np.stack(global_frames).astype(np.float32, copy=False),
            local_frames=np.stack(local_frames).astype(np.float32, copy=False),
            actions=np.stack(actions),
            risks=np.stack(risks),
            previous_actions=np.stack(previous_actions),
            memory=None,
            proficiency=None,
            episode_ids=np.asarray(episode_ids, dtype=np.int64),
            supervision_mask=np.ones((len(actions), 1), dtype=bool),
            teacher_action_evaluations=(
                np.stack(teacher_action_evaluations)
                if has_teacher_evaluations else
                None
            ),
            teacher_action_regrets=(
                np.stack(teacher_action_regrets)
                if has_teacher_evaluations else
                None
            ),
            teacher_action_evaluation_mask=(
                np.stack(teacher_action_evaluation_mask)
                if has_teacher_evaluations else
                None
            ),
        )
        demonstrations.validate()
        return demonstrations

    def save(
        self,
        path: str | Path,
        *,
        manifest_path: str | Path | None = None,
    ) -> dict[str, Any]:
        output = Path(path)
        demonstrations = self.build()
        demonstrations.save(output)
        manifest = {
            "schema_version": 1,
            "run_kind": "strict_native_streaming_demonstrations",
            "acceptance_claim": False,
            "training_only": True,
            "strict_inclusion_criterion": (
                "terminated with attack_complete or stage_complete and no player death"
            ),
            "model_inputs": [
                "delayed_visible_semantic_geometry",
                "current_visible_player_pose",
                "visible_displacement_motion",
            ],
            "recorded_fields": [
                "previous_executed_motor_action",
                *(
                    [
                        "teacher_action_evaluations",
                        "teacher_action_regrets",
                        "teacher_action_evaluation_mask",
                    ]
                    if demonstrations.teacher_action_evaluations is not None else
                    []
                ),
            ],
            "teacher_action_evaluation_schema": (
                None
                if demonstrations.teacher_action_evaluations is None else
                {
                    "action_count": 18,
                    "fields": list(TEACHER_ACTION_EVALUATION_FIELDS),
                    "regret": (
                        "max(0, selected_teacher_minimum_margin - "
                        "candidate_minimum_margin)"
                    ),
                    "model_input": False,
                }
            ),
            "excluded_model_inputs": [
                "scenario_identity",
                "attack_identity",
                "absolute_frame",
                "script_phase",
                "recorded_route",
                "waypoints",
            ],
            "dataset": str(output),
            "dataset_sha256": file_sha256(output),
            "samples": int(demonstrations.actions.shape[0]),
            "history": int(demonstrations.actions.shape[1]),
            "accepted_episodes": [
                {
                    **asdict(episode.identity),
                    "decisions": episode.decisions,
                    "strict_success": True,
                }
                for episode in self._accepted
            ],
            "rejected_episodes": list(self._rejected),
        }
        if manifest_path is not None:
            manifest_output = Path(manifest_path)
            manifest_output.parent.mkdir(parents=True, exist_ok=True)
            manifest_output.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        return manifest


__all__ = [
    "NativeDemonstrationBuilder",
    "NativeEpisodeBuffer",
    "NativeEpisodeIdentity",
    "UNKNOWN_SCENARIO_CONTEXT",
    "contextualize_demonstration_archive",
    "contextualize_demonstrations",
    "episode_context_key",
    "relabel_dagger_demonstration_archive",
    "risk_from_clearance",
]
