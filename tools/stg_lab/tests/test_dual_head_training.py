import hashlib
from dataclasses import asdict

import pytest

torch = pytest.importorskip("torch")

import experiments.train_temporal_residual_adapter as training
from experiments.train_temporal_residual_adapter import (
    FIT_CHECKPOINT_VERSION,
    MEMBERSHIP_CONFIDENCE_FIT_CHECKPOINT_VERSION,
    EpisodeFeatures,
    _clip_member_gradients,
    _clip_membership_confidence_gradients,
    _load_fit_checkpoint,
    _membership_confidence_loss,
    _membership_confidence_training_metadata,
    _predict_episode,
    _save_fit_checkpoint,
    _train_member,
    _training_optimizer_parameter_groups,
    _validate_membership_confidence_training_semantics,
    _validate_restored_membership_confidence_training,
)
from stg_lab.residual_adapter import (
    ResidualAdapterConfig,
    ResidualCorrectionAdapter,
)


def _config(*, membership: bool) -> ResidualAdapterConfig:
    return ResidualAdapterConfig(
        recurrent_size=4,
        hidden_size=7,
        ensemble_size=1,
        action_logit_mode="parent_residual_joint",
        separate_action_recurrent=True,
        per_action_membership_confidence=membership,
    )


def _episode(feature_size: int, seed: int = 17) -> EpisodeFeatures:
    decisions = 6
    actions = 18
    preferred = torch.zeros((decisions, actions), dtype=torch.bool)
    preferred[0, [1, 10]] = True
    preferred[1, [2, 11]] = True
    preferred[2, [3, 12]] = True
    preferred[4, [4, 13]] = True
    safe = torch.ones_like(preferred)
    collided = torch.zeros_like(preferred)
    parent_logits = torch.linspace(
        -0.4,
        0.4,
        decisions * actions,
    ).reshape(1, decisions, actions)
    parent_actions = parent_logits[0].argmax(dim=-1)
    return EpisodeFeatures(
        seed=seed,
        dataset=f"dataset-{seed}",
        report=f"report-{seed}",
        manifest=f"manifest-{seed}",
        features=torch.linspace(
            -1.0,
            1.0,
            decisions * feature_size,
        ).reshape(1, decisions, feature_size),
        parent_logits=parent_logits,
        parent_actions=parent_actions,
        previous_actions=torch.tensor([0, 1, 2, 3, 4, 5]),
        gate_targets=torch.tensor([1.0, 1.0, 1.0, 0.0, 1.0, 0.0]),
        gate_valid=torch.ones(decisions, dtype=torch.bool),
        hard_positive=torch.tensor([False, False, True, False, True, False]),
        correctable_hard_positive=torch.tensor(
            [False, False, True, False, True, False]
        ),
        anticipatory=torch.tensor([True, True, True, False, True, False]),
        future_onset_valid=torch.ones(decisions, dtype=torch.bool),
        anticipatory_lead_decisions=torch.tensor([6, 5, 4, 0, 4, 0]),
        preferred_actions=torch.tensor([1, 2, 3, -1, 4, -1]),
        preferred_action_set=preferred,
        preferred_equivalent_actions=preferred.clone(),
        preferred_correction_required=torch.tensor(
            [True, True, True, False, True, False]
        ),
        safety_candidate_actions=torch.tensor([1, 2, 3, 3, 4, 5]),
        safety_candidate_valid=torch.ones(decisions, dtype=torch.bool),
        safe_actions=safe,
        evaluation_safe_actions=safe.clone(),
        parent_evaluation_danger=torch.tensor(
            [True, True, True, False, True, False]
        ),
        collided_actions=collided,
        minimum_margins=torch.full((decisions, actions), 32.0),
        minimum_margin_mask=torch.ones_like(collided),
        teacher_selected_collision=torch.zeros(decisions, dtype=torch.bool),
    )


def _base_state(adapter: ResidualCorrectionAdapter) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in adapter.state_dict().items()
        if ".membership_head." not in name
    }


def _state_digest(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in state.items():
        contiguous = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(str(tuple(contiguous.shape)).encode("ascii"))
        digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


def _fit_once(
    adapter: ResidualCorrectionAdapter,
    episode: EpisodeFeatures,
    *,
    membership_weight: float,
) -> list[dict[str, float]]:
    return _train_member(
        adapter,
        0,
        [episode],
        seed=90210,
        epochs=2,
        learning_rate=3e-4,
        weight_decay=1e-4,
        chunk_length=3,
        gate_positive_weight=4.0,
        action_loss_weight=2.0,
        parent_copy_weight=0.05,
        device="cpu",
        membership_confidence_loss_weight=membership_weight,
        membership_confidence_loss_mode="unweighted",
    )


def test_membership_confidence_loss_matches_all_cell_bce_and_balancing() -> None:
    logits = torch.linspace(-2.0, 2.0, 54).reshape(3, 18).requires_grad_()
    accepted = torch.zeros((3, 18), dtype=torch.bool)
    accepted[0, [1, 4, 17]] = True
    accepted[1, [2, 3]] = True
    mask = torch.tensor([True, False, True])

    unweighted = _membership_confidence_loss(
        logits,
        accepted,
        mask,
        mode="unweighted",
    )
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        logits[0],
        accepted[0].to(logits.dtype),
    )
    assert torch.equal(unweighted, expected)

    balanced = _membership_confidence_loss(
        torch.zeros_like(logits),
        accepted,
        mask,
        mode="balanced",
    )
    assert balanced.item() == pytest.approx(0.6931471805599453)


def test_membership_confidence_loss_is_fail_closed_only_on_labelled_rows() -> None:
    logits = torch.zeros((2, 18), requires_grad=True)
    accepted = torch.zeros((2, 18), dtype=torch.bool)
    accepted[0, 2] = True
    logits.data[1, 7] = float("nan")
    loss = _membership_confidence_loss(
        logits,
        accepted,
        torch.tensor([True, False]),
    )
    assert torch.isfinite(loss)

    logits.data[0, 7] = float("nan")
    with pytest.raises(ValueError, match="finite on labelled rows"):
        _membership_confidence_loss(
            logits,
            accepted,
            torch.tensor([True, False]),
        )
    with pytest.raises(ValueError, match="loss mode"):
        _membership_confidence_loss(
            torch.zeros((1, 18)),
            torch.ones((1, 18), dtype=torch.bool),
            torch.ones(1, dtype=torch.bool),
            mode="softmax",
        )


@pytest.mark.parametrize("invalid", (float("nan"), float("inf"), -1.0))
def test_membership_confidence_training_rejects_invalid_weight(invalid: float) -> None:
    with pytest.raises(ValueError, match="finite and nonnegative"):
        _validate_membership_confidence_training_semantics(
            enabled=True,
            action_logit_mode="parent_residual_joint",
            loss_weight=invalid,
            loss_mode="unweighted",
        )


def test_membership_confidence_training_requires_explicit_coherent_enablement() -> None:
    with pytest.raises(ValueError, match="must be zero"):
        _validate_membership_confidence_training_semantics(
            enabled=False,
            action_logit_mode="parent_residual_joint",
            loss_weight=12.0,
            loss_mode="unweighted",
        )
    with pytest.raises(ValueError, match="must be positive"):
        _validate_membership_confidence_training_semantics(
            enabled=True,
            action_logit_mode="parent_residual_joint",
            loss_weight=0.0,
            loss_mode="unweighted",
        )
    with pytest.raises(ValueError, match="selector action head"):
        _validate_membership_confidence_training_semantics(
            enabled=True,
            action_logit_mode="certified_membership",
            loss_weight=12.0,
            loss_mode="unweighted",
        )


def test_optimizer_and_gradient_clip_groups_are_complete_and_disjoint() -> None:
    adapter = ResidualCorrectionAdapter(_config(membership=True))
    member = adapter.members[0]
    groups = _training_optimizer_parameter_groups(
        member,
        per_action_membership_confidence=True,
    )
    base_ids = [id(parameter) for parameter in groups["base"]]
    membership_ids = [
        id(parameter) for parameter in groups["membership_confidence"]
    ]
    all_ids = [
        id(parameter)
        for parameter in member.parameters()
        if parameter.requires_grad
    ]
    assert not set(base_ids) & set(membership_ids)
    assert set(base_ids) | set(membership_ids) == set(all_ids)
    assert base_ids == [
        parameter_id
        for parameter_id in all_ids
        if parameter_id not in set(membership_ids)
    ]

    for parameter in groups["base"]:
        parameter.grad = torch.ones_like(parameter) * 20.0
    for parameter in groups["membership_confidence"]:
        parameter.grad = torch.ones_like(parameter) * 1000.0
    membership_before = [
        parameter.grad.clone()
        for parameter in groups["membership_confidence"]
        if parameter.grad is not None
    ]
    base_clip_groups = _clip_member_gradients(
        member,
        max_norm=1.0,
        excluded_parameters=groups["membership_confidence"],
    )
    assert set(base_clip_groups) == {"shared_safety", "action"}
    assert all(
        torch.equal(parameter.grad, before)
        for parameter, before in zip(
            groups["membership_confidence"],
            membership_before,
            strict=True,
        )
    )
    _clip_membership_confidence_gradients(
        groups["membership_confidence"],
        max_norm=1.0,
    )
    membership_norm = torch.linalg.vector_norm(torch.cat([
        parameter.grad.flatten()
        for parameter in groups["membership_confidence"]
        if parameter.grad is not None
    ]))
    assert membership_norm.item() == pytest.approx(1.0, rel=1e-5)


def test_training_uses_two_adamw_optimizers_with_identical_hyperparameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = ResidualCorrectionAdapter(_config(membership=True))
    expected = _training_optimizer_parameter_groups(
        adapter.members[0],
        per_action_membership_confidence=True,
    )
    calls: list[tuple[tuple[torch.nn.Parameter, ...], float, float]] = []

    class SpyAdamW:
        def __init__(self, parameters, *, lr: float, weight_decay: float) -> None:
            calls.append((tuple(parameters), lr, weight_decay))

    monkeypatch.setattr(training.torch.optim, "AdamW", SpyAdamW)
    history = _train_member(
        adapter,
        0,
        [],
        seed=1,
        epochs=1,
        learning_rate=7e-4,
        weight_decay=3e-5,
        chunk_length=3,
        gate_positive_weight=4.0,
        action_loss_weight=2.0,
        parent_copy_weight=0.05,
        device="cpu",
        membership_confidence_loss_weight=12.0,
    )

    assert history == [{
        "epoch": 1.0,
        "mean_chunk_loss": 0.0,
        "mean_membership_confidence_loss": 0.0,
        "membership_confidence_loss_weight": 12.0,
    }]
    assert len(calls) == 2
    assert [id(value) for value in calls[0][0]] == [
        id(value) for value in expected["base"]
    ]
    assert [id(value) for value in calls[1][0]] == [
        id(value) for value in expected["membership_confidence"]
    ]
    assert calls[0][1:] == calls[1][1:] == (7e-4, 3e-5)


def test_dual_head_training_preserves_base_initialization_updates_and_history() -> None:
    torch.manual_seed(8181)
    base = ResidualCorrectionAdapter(_config(membership=False))
    torch.manual_seed(8181)
    dual = ResidualCorrectionAdapter(_config(membership=True))
    assert _state_digest(_base_state(base)) == _state_digest(_base_state(dual))
    assert tuple(_base_state(base)) == tuple(_base_state(dual))

    episode = _episode(base.config.feature_size)
    membership_before = {
        name: value.detach().clone()
        for name, value in dual.state_dict().items()
        if ".membership_head." in name
    }
    base_history = _fit_once(base, episode, membership_weight=0.0)
    dual_history = _fit_once(dual, episode, membership_weight=12.0)

    assert [row["mean_chunk_loss"] for row in dual_history] == [
        row["mean_chunk_loss"] for row in base_history
    ]
    assert _state_digest(_base_state(base)) == _state_digest(_base_state(dual))
    assert all(
        torch.equal(value, _base_state(dual)[name])
        for name, value in _base_state(base).items()
    )
    assert any(
        not torch.equal(value, dual.state_dict()[name])
        for name, value in membership_before.items()
    )


def test_prediction_keeps_selector_candidate_and_uses_membership_confidence() -> None:
    adapter = ResidualCorrectionAdapter(_config(membership=True)).eval()
    with torch.no_grad():
        for parameter in adapter.parameters():
            parameter.zero_()
        member = adapter.members[0]
        member.action_head.bias[6] = 10.0
        assert member.membership_head is not None
        member.membership_head.bias.fill_(-10.0)
        member.membership_head.bias[6] = -2.0
        member.membership_head.bias[7] = 10.0
    episode = _episode(adapter.config.feature_size)

    values = _predict_episode(adapter, episode, device="cpu")

    assert values["candidates"].tolist() == [6] * episode.decisions
    assert values["mean_membership_probabilities"].argmax(dim=-1).tolist() == (
        [7] * episode.decisions
    )
    assert torch.allclose(
        values["action_confidence"],
        torch.full(
            (episode.decisions,),
            torch.sigmoid(torch.tensor(-2.0)).item(),
        ),
    )
    assert values["membership_probabilities"].shape == (1, 6, 18)
    assert values["membership_member_finite"].shape == (1, 6)
    assert torch.equal(
        values["action_all_members_finite"],
        values["selector_all_members_finite"]
        & values["membership_all_members_finite"],
    )


def test_disabled_prediction_contract_has_no_auxiliary_fields() -> None:
    adapter = ResidualCorrectionAdapter(_config(membership=False)).eval()
    values = _predict_episode(
        adapter,
        _episode(adapter.config.feature_size),
        device="cpu",
    )
    assert not {
        "membership_probabilities",
        "mean_membership_probabilities",
        "membership_member_finite",
        "selector_all_members_finite",
        "membership_all_members_finite",
    } & values.keys()


def test_restored_membership_confidence_provenance_is_exact_and_transitive() -> None:
    controls = _membership_confidence_training_metadata(
        enabled=True,
        action_logit_mode="parent_residual_joint",
        loss_weight=12.0,
        loss_mode="unweighted",
    )
    leaf = {"training_controls": controls}
    outer = {
        "training_controls": dict(controls),
        "fit_checkpoint_weight_source": {"training_metadata": leaf},
    }
    assert _validate_restored_membership_confidence_training(
        outer,
        enabled=True,
        action_logit_mode="parent_residual_joint",
        requested_weight=12.0,
        requested_mode="unweighted",
    ) == controls

    with pytest.raises(ValueError, match="mode and weight must match"):
        _validate_restored_membership_confidence_training(
            outer,
            enabled=True,
            action_logit_mode="parent_residual_joint",
            requested_weight=11.0,
            requested_mode="unweighted",
        )
    conflicting = {
        "training_controls": controls,
        "frozen_adapter_weight_source": {
            "training_metadata": {
                "training_controls": _membership_confidence_training_metadata(
                    enabled=True,
                    action_logit_mode="parent_residual_joint",
                    loss_weight=12.0,
                    loss_mode="balanced",
                ),
            },
        },
    }
    with pytest.raises(ValueError, match="inconsistent"):
        _validate_restored_membership_confidence_training(
            conflicting,
            enabled=True,
            action_logit_mode="parent_residual_joint",
            requested_weight=12.0,
            requested_mode="unweighted",
        )


def test_legacy_provenance_is_only_valid_for_a_disabled_head() -> None:
    legacy = {"training_controls": {"epochs": 6}}
    assert _validate_restored_membership_confidence_training(
        legacy,
        enabled=False,
        action_logit_mode="parent_residual_joint",
        requested_weight=0.0,
        requested_mode="unweighted",
    ) is None
    with pytest.raises(ValueError, match="legacy adapter has no"):
        _validate_restored_membership_confidence_training(
            legacy,
            enabled=True,
            action_logit_mode="parent_residual_joint",
            requested_weight=12.0,
            requested_mode="unweighted",
        )


def test_fit_checkpoint_bumps_only_for_dual_head_and_loads_old_v2(tmp_path) -> None:
    parent = tmp_path / "parent.pt"
    torch.save({"identity": "parent"}, parent)
    base_path = tmp_path / "base-fit.pt"
    dual_path = tmp_path / "dual-fit.pt"
    base = ResidualCorrectionAdapter(_config(membership=False))
    dual = ResidualCorrectionAdapter(_config(membership=True))
    policy_config = {"recurrent_size": 4}

    _save_fit_checkpoint(
        base,
        base_path,
        parent_checkpoint=parent,
        parent_policy_config=policy_config,
        training_metadata={},
    )
    _save_fit_checkpoint(
        dual,
        dual_path,
        parent_checkpoint=parent,
        parent_policy_config=policy_config,
        training_metadata={
            "training_controls": _membership_confidence_training_metadata(
                enabled=True,
                action_logit_mode="parent_residual_joint",
                loss_weight=12.0,
                loss_mode="unweighted",
            ),
        },
    )
    base_payload = torch.load(base_path, weights_only=False)
    dual_payload = torch.load(dual_path, weights_only=False)
    assert base_payload["version"] == FIT_CHECKPOINT_VERSION == 2
    assert dual_payload["version"] == MEMBERSHIP_CONFIDENCE_FIT_CHECKPOINT_VERSION == 3

    base_payload["adapter_config"].pop("per_action_membership_confidence")
    torch.save(base_payload, base_path)
    restored_base, base_metadata = _load_fit_checkpoint(
        base_path,
        parent_checkpoint=parent,
        parent_policy_config=policy_config,
        expected_adapter_config=base.config,
    )
    restored_dual, dual_metadata = _load_fit_checkpoint(
        dual_path,
        parent_checkpoint=parent,
        parent_policy_config=policy_config,
        expected_adapter_config=dual.config,
    )
    assert base_metadata["version"] == 2
    assert dual_metadata["version"] == 3
    assert restored_base.config == base.config
    assert restored_dual.config == dual.config
    assert asdict(restored_dual.config)["per_action_membership_confidence"] is True
