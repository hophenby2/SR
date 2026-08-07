from collections.abc import Iterator, Mapping

import pytest

torch = pytest.importorskip("torch")

from stg_lab.residual_adapter import ResidualAdapterConfig, ResidualCorrectionAdapter
from experiments.compare_preferred_objectives_cv import (
    ARM_NAMES,
    PROHIBITED_SOURCE_SEEDS,
    TRAINING_SEEDS,
    _assert_only_preferred_sets_differ,
    _assert_paired_gradient_clip_groups,
    _assert_paired_non_action_states_equal,
    _exact_target_episode,
    _fixed_folds,
    _paired_summary,
    _raw_action_metrics,
    _select_training_inventory,
    _sum_raw,
    _state_digest,
    _partition_action_branch_state,
    _target_intervention_stats,
    _uniform_soft_target_screen,
    _validate_output_path,
)
from experiments.train_temporal_residual_adapter import (
    EpisodeFeatures,
    _preferred_action_conditional_tiebreak_loss,
    _preferred_action_set_loss,
    _preferred_action_set_rank_loss,
    _preferred_action_tiebreak_mask,
    _preferred_action_uniform_conditional_loss,
    _train_member,
)


class SeedRoleOnly(Mapping[str, object]):
    """Inventory record that fails if code touches a path-bearing field."""

    def __init__(self, seed: int, role: str) -> None:
        self.seed = seed
        self.role = role

    def __getitem__(self, key: str) -> object:
        if key == "seed":
            return self.seed
        if key == "role":
            return self.role
        raise AssertionError(f"path-bearing field was accessed: {key}")

    def __iter__(self) -> Iterator[str]:
        yield "seed"
        yield "role"

    def __len__(self) -> int:
        return 2


def make_episode(seed: int = 1) -> EpisodeFeatures:
    decisions = 5
    action_count = 18
    parent_actions = torch.tensor([0, 9, 2, 3, 4])
    preferred_actions = torch.tensor([1, 10, 2, 3, 4])
    gate_targets = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0])
    gate_valid = torch.ones(decisions, dtype=torch.bool)
    required = torch.tensor([True, True, False, False, False])
    preferred_set = torch.zeros(decisions, action_count, dtype=torch.bool)
    preferred_set[0, [1, 10]] = True
    preferred_set[1, [1, 10]] = True
    preferred_set[2, 2] = True
    equivalent = torch.zeros_like(preferred_set)
    equivalent[0, [1, 10]] = True
    equivalent[1, [1, 10]] = True
    safe = torch.ones_like(preferred_set)
    collided = torch.zeros_like(preferred_set)
    margins = torch.full((decisions, action_count), 32.0)
    return EpisodeFeatures(
        seed=seed,
        dataset=f"dataset-{seed}",
        report=f"report-{seed}",
        manifest=f"manifest-{seed}",
        features=torch.arange(decisions * 3, dtype=torch.float32).reshape(1, decisions, 3),
        parent_logits=torch.zeros(1, decisions, action_count),
        parent_actions=parent_actions,
        previous_actions=torch.tensor([1, 1, 2, 3, 4]),
        gate_targets=gate_targets,
        gate_valid=gate_valid,
        hard_positive=torch.tensor([False, False, True, False, False]),
        correctable_hard_positive=torch.tensor([False, False, True, False, False]),
        anticipatory=torch.tensor([True, True, True, False, False]),
        future_onset_valid=torch.ones(decisions, dtype=torch.bool),
        anticipatory_lead_decisions=torch.tensor([4, 6, 8, 0, 0]),
        preferred_actions=preferred_actions,
        preferred_action_set=preferred_set,
        preferred_equivalent_actions=equivalent,
        preferred_correction_required=required,
        safety_candidate_actions=preferred_actions.clone(),
        safety_candidate_valid=torch.ones(decisions, dtype=torch.bool),
        safe_actions=safe,
        evaluation_safe_actions=safe.clone(),
        parent_evaluation_danger=torch.tensor([True, True, True, False, False]),
        collided_actions=collided,
        minimum_margins=margins,
        minimum_margin_mask=torch.ones_like(collided),
        teacher_selected_collision=torch.zeros(decisions, dtype=torch.bool),
        global_frame_dtype="float16",
        local_frame_dtype="float16",
    )


def test_inventory_filters_before_touching_nontraining_paths() -> None:
    records: list[Mapping[str, object]] = [
        {"seed": seed, "role": "training"} for seed in TRAINING_SEEDS
    ]
    records.extend(
        SeedRoleOnly(seed, "validation") for seed in PROHIBITED_SOURCE_SEEDS
    )
    selected = _select_training_inventory({"source_inventory": records})
    assert [record["seed"] for record in selected] == list(TRAINING_SEEDS)


def test_inventory_rejects_prohibited_or_unlisted_training_seed() -> None:
    base = [{"seed": seed, "role": "training"} for seed in TRAINING_SEEDS]
    with pytest.raises(ValueError, match="prohibited source seed 10306"):
        _select_training_inventory({
            "source_inventory": [*base, SeedRoleOnly(10306, "training")]
        })
    with pytest.raises(ValueError, match="unexpected training source seed"):
        _select_training_inventory({
            "source_inventory": [*base, SeedRoleOnly(55555, "training")]
        })


def test_fixed_folds_are_disjoint_and_cover_outer_audit_once() -> None:
    folds = _fixed_folds()
    assert len(folds) == 3
    audit = []
    for fold in folds:
        assert len(fold.fit_seeds) == 8
        assert len(fold.calibration_seeds) == 2
        assert len(fold.audit_seeds) == 5
        assert not set(fold.fit_seeds) & set(fold.calibration_seeds)
        assert not set(fold.fit_seeds) & set(fold.audit_seeds)
        assert not set(fold.calibration_seeds) & set(fold.audit_seeds)
        audit.extend(fold.audit_seeds)
    assert sorted(audit) == sorted(TRAINING_SEEDS)


def test_exact_arm_changes_only_positive_preferred_target_sets() -> None:
    equivalent = make_episode()
    exact = _exact_target_episode(equivalent)
    _assert_only_preferred_sets_differ([exact], [equivalent])
    assert torch.equal(exact.preferred_action_set[0], torch.nn.functional.one_hot(
        torch.tensor(1), num_classes=18
    ).bool())
    assert torch.equal(exact.preferred_action_set[1], torch.nn.functional.one_hot(
        torch.tensor(10), num_classes=18
    ).bool())
    assert torch.equal(exact.preferred_action_set[2:], equivalent.preferred_action_set[2:])
    assert equivalent.preferred_action_set[0].sum() == 2
    stats = _target_intervention_stats([exact], [equivalent])
    assert stats == {
        "positive_target_rows": 3,
        "correction_required_rows": 2,
        "target_set_differing_rows": 2,
        "equivalence_cardinality_distribution": {"1": 1, "2": 2},
        "exact_cardinality_distribution": {"1": 3},
    }


def test_raw_metrics_use_early_correction_required_denominator() -> None:
    episode = make_episode()
    # First target is exact; the second is the other certified speed sibling.
    candidates = torch.tensor([1, 1, 17, 3, 4])
    predictions = {
        episode.seed: {
            "candidates": candidates,
            "action_all_members_finite": torch.ones(5, dtype=torch.bool),
        }
    }
    metrics = _raw_action_metrics([episode], predictions)
    assert metrics["targets"] == 2
    assert metrics["exact_top1"] == 1
    assert metrics["equivalent_top1"] == 2
    assert metrics["candidate_changed_parent"] == 2
    assert metrics["direction_correct"] == 2
    assert metrics["speed_correct"] == 1
    assert metrics["exact_top1_rate"] == 0.5
    assert metrics["equivalent_top1_rate"] == 1.0
    assert metrics["tiebreak_eligible_targets"] == 2
    assert metrics["tiebreak_eligible_previous_top1"] == 2
    assert metrics["tiebreak_eligible_exact_top1"] == 1
    assert metrics["tiebreak_eligible_equivalent_top1"] == 2
    assert metrics["tiebreak_eligible_direction_correct"] == 2
    assert metrics["tiebreak_eligible_speed_correct"] == 1
    assert metrics["tiebreak_eligible_exact_top1_rate"] == 0.5
    assert metrics["tiebreak_eligible_equivalent_top1_rate"] == 1.0
    assert metrics["tiebreak_eligible_previous_top1_rate"] == 1.0


def test_raw_metrics_fail_closed_on_nonfinite_action_zero_fallback() -> None:
    episode = make_episode()
    episode.parent_actions[0] = 4
    episode.preferred_actions[0] = 0
    episode.preferred_action_set[0] = False
    episode.preferred_action_set[0, [0, 9]] = True
    episode.preferred_equivalent_actions[0] = False
    episode.preferred_equivalent_actions[0, [0, 9]] = True
    episode.previous_actions[0] = 0
    predictions = {
        episode.seed: {
            "candidates": torch.tensor([0, 10, 2, 3, 4]),
            "action_all_members_finite": torch.tensor([
                False, True, True, True, True,
            ]),
        }
    }

    metrics = _raw_action_metrics([episode], predictions)

    assert metrics["targets"] == 2
    assert metrics["finite_top1"] == 1
    assert metrics["exact_top1"] == 1
    assert metrics["equivalent_top1"] == 1
    assert metrics["candidate_changed_parent"] == 1
    assert metrics["tiebreak_eligible_targets"] == 2
    assert metrics["tiebreak_eligible_finite_top1"] == 1
    assert metrics["tiebreak_eligible_previous_top1"] == 0
    assert metrics["tiebreak_eligible_exact_top1"] == 1
    assert metrics["tiebreak_eligible_equivalent_top1"] == 1


def test_raw_metric_micro_summary_uses_tiebreak_eligible_denominator() -> None:
    first = make_episode(seed=11)
    second = make_episode(seed=12)
    first_metrics = _raw_action_metrics([first], {
        11: {
            "candidates": torch.tensor([1, 1, 2, 3, 4]),
            "action_all_members_finite": torch.ones(5, dtype=torch.bool),
        }
    })
    second_metrics = _raw_action_metrics([second], {
        12: {
            "candidates": torch.tensor([17, 10, 2, 3, 4]),
            "action_all_members_finite": torch.tensor([
                False, True, True, True, True,
            ]),
        }
    })

    summary = _sum_raw([first_metrics, second_metrics])

    assert summary["targets"] == 4
    assert summary["tiebreak_eligible_targets"] == 4
    assert summary["tiebreak_eligible_previous_top1"] == 2
    assert summary["tiebreak_eligible_exact_top1"] == 2
    assert summary["tiebreak_eligible_equivalent_top1"] == 3
    assert summary["tiebreak_eligible_exact_top1_rate"] == 0.5
    assert summary["tiebreak_eligible_equivalent_top1_rate"] == 0.75
    assert summary["tiebreak_eligible_previous_top1_rate"] == 0.5


def test_paired_summary_compares_rank_with_conditional_tiebreak() -> None:
    episode = make_episode()
    metrics = _raw_action_metrics([episode], {
        episode.seed: {
            "candidates": torch.tensor([1, 1, 2, 3, 4]),
            "action_all_members_finite": torch.ones(5, dtype=torch.bool),
        }
    })
    fold = {
        "fold": 0,
        "arms": {
            arm: {
                "raw_action_metrics": {
                    split: dict(metrics)
                    for split in ("fit", "calibration", "audit")
                },
                "calibration": {"success": True},
            }
            for arm in ARM_NAMES
        },
    }

    summary = _paired_summary([fold])

    assert summary["outer_audit_micro"]["equivalence_weak_tiebreak"][
        "tiebreak_eligible_previous_top1_rate"
    ] == 1.0
    assert (
        "equivalence_weak_tiebreak_minus_equivalence_top1_rank"
        in summary["comparisons"]
    )


def test_uniform_soft_target_screen_is_fixed_and_fail_closed() -> None:
    baseline = {
        "targets": 359,
        "finite_top1": 359,
        "equivalent_top1": 60,
        "direction_correct": 79,
        "speed_correct": 194,
    }
    candidate = {
        "targets": 359,
        "finite_top1": 359,
        "equivalent_top1": 68,
        "direction_correct": 72,
        "speed_correct": 187,
    }
    summary = {
        "outer_audit_micro": {
            "equivalence": baseline,
            "equivalence_uniform_soft_target": candidate,
        },
        "comparisons": {
            "equivalence_uniform_soft_target_minus_equivalence": {
                "outer_audit_fold_direction": {
                    "equivalent_top1_rate": {"candidate_better_folds": 2}
                }
            }
        },
    }

    passed = _uniform_soft_target_screen(summary, epochs=6)
    assert passed["passed"] is True
    assert passed["eligible_for_fixed_e20_followup"] is True
    assert passed["deployment_eligible"] is False

    summary["outer_audit_micro"]["equivalence"] = {
        **baseline,
        "equivalent_top1": 61,
    }
    drifted = _uniform_soft_target_screen(summary, epochs=6)
    assert drifted["passed"] is False
    assert drifted["checks"][
        "deterministic_reference_metrics_reproduced"
    ] is False

    inapplicable = _uniform_soft_target_screen(summary, epochs=1)
    assert inapplicable["applicable"] is False
    assert inapplicable["passed"] is False


def test_state_digest_is_order_independent_and_value_sensitive() -> None:
    first = {"b": torch.tensor([2.0]), "a": torch.tensor([1.0])}
    reordered = {"a": torch.tensor([1.0]), "b": torch.tensor([2.0])}
    changed = {"a": torch.tensor([1.0]), "b": torch.tensor([3.0])}
    assert _state_digest(first) == _state_digest(reordered)
    assert _state_digest(first) != _state_digest(changed)


def test_action_branch_digest_partition_and_paired_non_action_invariant() -> None:
    assert ARM_NAMES == (
        "exact",
        "equivalence",
        "equivalence_top1_rank",
        "equivalence_weak_tiebreak",
        "equivalence_uniform_soft_target",
    )
    config = ResidualAdapterConfig(
        recurrent_size=4,
        hidden_size=6,
        ensemble_size=1,
        action_logit_mode="parent_residual_joint",
        separate_action_recurrent=True,
    )
    torch.manual_seed(31)
    baseline = ResidualCorrectionAdapter(config)
    adapters = {
        arm: ResidualCorrectionAdapter(config) for arm in ARM_NAMES
    }
    for adapter in adapters.values():
        adapter.load_state_dict(baseline.state_dict())
    gradient_groups = _assert_paired_gradient_clip_groups(adapters)
    assert set(gradient_groups) == {"shared_safety", "action"}
    with torch.no_grad():
        adapters["equivalence"].members[0].action_head.bias.add_(1.0)
        adapters["equivalence_top1_rank"].members[0].action_recurrent.bias_ih_l0.add_(
            2.0
        )
        adapters["equivalence_weak_tiebreak"].members[0].action_head.bias.add_(3.0)
        adapters["equivalence_uniform_soft_target"].members[0].action_head.bias.add_(
            4.0
        )

    action_digests = {}
    non_action_digests = {}
    for arm, adapter in adapters.items():
        action, non_action = _partition_action_branch_state(adapter.state_dict())
        action_digests[arm] = _state_digest(action)
        non_action_digests[arm] = _state_digest(non_action)
    assert len(set(action_digests.values())) == len(ARM_NAMES)
    assert len(set(non_action_digests.values())) == 1
    assert _assert_paired_non_action_states_equal(adapters) == next(
        iter(non_action_digests.values())
    )

    with torch.no_grad():
        adapters["equivalence"].members[0].gate_head.bias.add_(1.0)
    with pytest.raises(AssertionError, match="non-action branch digest differs"):
        _assert_paired_non_action_states_equal(adapters)


def test_output_cannot_overwrite_any_protected_input(tmp_path) -> None:
    source = tmp_path / "dataset.npz"
    report = tmp_path / "report.json"
    _validate_output_path(tmp_path / "result.json", [source, report])
    with pytest.raises(ValueError, match="protected input"):
        _validate_output_path(source, [source, report])


def test_rank_loss_detects_external_top1_despite_high_set_probability() -> None:
    logits = torch.tensor([
        [2.0, 2.0, 2.5, -5.0],
        [float("nan"), float("nan"), float("nan"), float("nan")],
    ])
    accepted = torch.tensor([
        [True, True, False, False],
        [False, False, False, False],
    ])
    mask = torch.tensor([True, True])
    probabilities = torch.softmax(logits[0], dim=-1)
    assert probabilities[accepted[0]].sum() > probabilities[~accepted[0]].max()
    assert int(logits[0].argmax()) == 2
    assert _preferred_action_set_loss(logits[:1], accepted[:1], mask[:1]) < 1.0
    rank = _preferred_action_set_rank_loss(logits, accepted, mask, margin=1.0)
    assert torch.isfinite(rank)
    assert rank.item() == pytest.approx(1.5)


def test_rank_loss_is_zero_when_accepted_top1_satisfies_margin() -> None:
    logits = torch.tensor([[3.0, 1.0, 2.0, -1.0]])
    accepted = torch.tensor([[True, True, False, False]])
    rank = _preferred_action_set_rank_loss(
        logits,
        accepted,
        torch.tensor([True]),
        margin=1.0,
    )
    assert rank.item() == 0.0
    for invalid_margin in (float("nan"), float("inf"), -0.1):
        with pytest.raises(ValueError, match="rank margin"):
            _preferred_action_set_rank_loss(
                logits,
                accepted,
                torch.tensor([True]),
                margin=invalid_margin,
            )


def test_conditional_tiebreak_ignores_rejected_logits_and_gradients() -> None:
    logits = torch.tensor(
        [[1.0, 2.0, -100.0, 7.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    accepted = torch.tensor([[True, True, False, False]])
    preferred = torch.tensor([0])
    mask = torch.tensor([True])
    loss = _preferred_action_conditional_tiebreak_loss(
        logits,
        accepted,
        preferred,
        mask,
    )
    changed = logits.detach().clone()
    changed[0, 2:] = torch.tensor([1e100, -1e100], dtype=changed.dtype)
    changed_loss = _preferred_action_conditional_tiebreak_loss(
        changed,
        accepted,
        preferred,
        mask,
    )

    loss.backward()

    assert loss.item() == pytest.approx(changed_loss.item())
    assert logits.grad is not None
    assert torch.equal(logits.grad[0, 2:], torch.zeros(2, dtype=logits.dtype))
    assert bool((logits.grad[0, :2] != 0.0).all())


def test_uniform_soft_target_is_symmetric_and_ignores_rejected_logits() -> None:
    logits = torch.tensor(
        [[1.0, 2.0, -100.0, 7.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    accepted = torch.tensor([[True, True, False, False]])
    mask = torch.tensor([True])
    loss = _preferred_action_uniform_conditional_loss(logits, accepted, mask)
    changed = logits.detach().clone()
    changed[0, 2:] = torch.tensor([1e100, -1e100], dtype=changed.dtype)
    shifted = logits.detach() + 1e6

    changed_loss = _preferred_action_uniform_conditional_loss(
        changed,
        accepted,
        mask,
    )
    shifted_loss = _preferred_action_uniform_conditional_loss(
        shifted,
        accepted,
        mask,
    )
    loss.backward()

    assert loss.item() > 0.0
    assert loss.item() == pytest.approx(changed_loss.item())
    assert loss.item() == pytest.approx(shifted_loss.item())
    assert logits.grad is not None
    assert torch.equal(logits.grad[0, 2:], torch.zeros(2, dtype=logits.dtype))
    assert logits.grad[0, 0].item() == pytest.approx(-logits.grad[0, 1].item())
    assert bool((logits.grad[0, :2] != 0.0).all())


def test_uniform_soft_target_excludes_singleton_and_masked_rows() -> None:
    logits = torch.tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ], requires_grad=True)
    accepted = torch.tensor([
        [False, True, False],
        [True, False, True],
    ])
    loss = _preferred_action_uniform_conditional_loss(
        logits,
        accepted,
        torch.tensor([True, False]),
    )
    loss.backward()

    assert loss.item() == 0.0
    assert logits.grad is not None
    assert torch.equal(logits.grad, torch.zeros_like(logits))


def test_conditional_tiebreak_rejects_target_outside_certified_set() -> None:
    with pytest.raises(ValueError, match="outside its certified set"):
        _preferred_action_conditional_tiebreak_loss(
            torch.zeros((1, 4)),
            torch.tensor([[True, True, False, False]]),
            torch.tensor([2]),
            torch.tensor([True]),
        )


def test_previous_action_tiebreak_mask_is_fail_closed() -> None:
    accepted = torch.tensor([
        [True, True, False],
        [False, True, False],
        [True, False, True],
        [True, True, False],
    ])
    result = _preferred_action_tiebreak_mask(
        accepted,
        torch.tensor([0, 1, 1, -1]),
        torch.ones(4, dtype=torch.bool),
    )

    assert result.tolist() == [True, False, False, False]


def test_train_member_rejects_invalid_auxiliary_weights_before_model_access() -> None:
    for invalid_weight in (float("nan"), float("inf"), -0.1):
        with pytest.raises(ValueError, match="rank loss weight"):
            _train_member(
                None,  # type: ignore[arg-type]
                0,
                [],
                seed=1,
                epochs=1,
                learning_rate=1e-3,
                weight_decay=0.0,
                chunk_length=1,
                gate_positive_weight=1.0,
                action_loss_weight=0.0,
                parent_copy_weight=0.0,
                device="cpu",
                preferred_action_rank_loss_weight=invalid_weight,
            )
        with pytest.raises(ValueError, match="tiebreak loss weight"):
            _train_member(
                None,  # type: ignore[arg-type]
                0,
                [],
                seed=1,
                epochs=1,
                learning_rate=1e-3,
                weight_decay=0.0,
                chunk_length=1,
                gate_positive_weight=1.0,
                action_loss_weight=0.0,
                parent_copy_weight=0.0,
                device="cpu",
                preferred_action_tiebreak_loss_weight=invalid_weight,
            )
        with pytest.raises(ValueError, match="uniform loss weight"):
            _train_member(
                None,  # type: ignore[arg-type]
                0,
                [],
                seed=1,
                epochs=1,
                learning_rate=1e-3,
                weight_decay=0.0,
                chunk_length=1,
                gate_positive_weight=1.0,
                action_loss_weight=0.0,
                parent_copy_weight=0.0,
                device="cpu",
                preferred_action_uniform_loss_weight=invalid_weight,
            )


def test_conditional_tiebreak_excludes_singleton_and_masked_rows() -> None:
    logits = torch.tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ], requires_grad=True)
    accepted = torch.tensor([
        [True, True, False],
        [False, True, False],
        [True, False, True],
    ])
    loss = _preferred_action_conditional_tiebreak_loss(
        logits,
        accepted,
        torch.tensor([0, 1, 2]),
        torch.tensor([True, True, False]),
    )

    loss.backward()

    assert logits.grad is not None
    assert bool((logits.grad[0, :2] != 0.0).all())
    assert torch.equal(logits.grad[0, 2:], torch.zeros(1))
    assert torch.equal(logits.grad[1:], torch.zeros((2, 3)))
