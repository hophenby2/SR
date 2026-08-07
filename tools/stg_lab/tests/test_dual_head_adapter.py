from copy import deepcopy
from dataclasses import asdict

import pytest

torch = pytest.importorskip("torch")

from stg_lab.policy import PolicyConfig
from stg_lab.residual_adapter import (
    ResidualAdapterConfig,
    ResidualCorrectionAdapter,
    ResidualPolicyWrapper,
    ResidualRuntimeConfig,
    ensemble_action_summary,
    finite_action_probabilities,
    load_residual_adapter,
    save_residual_adapter,
)


class StubParent(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = PolicyConfig(
            feature_size=8,
            recurrent_size=4,
            memory_size=0,
            proficiency_size=0,
            inference_mode="stream",
            local_feature_grid_size=2,
            local_downsample_stages=2,
        )
        self.anchor = torch.nn.Parameter(torch.zeros(()))


class StreamingStubParent(StubParent):
    def forward_with_recurrent(
        self,
        global_frames,
        local_frames,
        memory=None,
        proficiency=None,
        hidden=None,
    ):
        batch, steps = global_frames.shape[:2]
        logits = global_frames.new_zeros((batch, steps, 18))
        logits[..., 5] = 1.0
        risk = global_frames.new_zeros((batch, steps))
        recurrent = global_frames.new_zeros((batch, steps, 4))
        next_hidden = global_frames.new_zeros((1, batch, 4))
        return logits, risk, next_hidden, recurrent


def _dual_config(**overrides) -> ResidualAdapterConfig:
    values = {
        "recurrent_size": 4,
        "hidden_size": 6,
        "ensemble_size": 2,
        "action_logit_mode": "parent_residual_joint",
        "separate_action_recurrent": True,
        "per_action_membership_confidence": True,
    }
    values.update(overrides)
    return ResidualAdapterConfig(**values)


def _deployable_dual_adapter() -> ResidualCorrectionAdapter:
    return ResidualCorrectionAdapter(_dual_config(
        per_action_safety_critic=True,
        visual_latent_size=16,
        per_action_physical_danger=True,
    ))


def _future_onset_runtime() -> ResidualRuntimeConfig:
    return ResidualRuntimeConfig(
        gate_probability_threshold=0.5,
        minimum_member_gate_probability=0.5,
        action_probability_threshold=0.5,
        ensemble_agreement_threshold=1.0,
        legacy_gate_enabled=False,
        critic_enabled=True,
        critic_signal="physical_danger",
        parent_physical_danger_probability_threshold=0.7,
        candidate_physical_danger_probability_threshold=0.2,
        future_onset_gate_enabled=True,
    )


def test_dual_head_config_requires_a_separate_parent_residual_selector() -> None:
    assert ResidualAdapterConfig(
        recurrent_size=4,
    ).per_action_membership_confidence is False
    with pytest.raises(ValueError, match="must be a Boolean"):
        ResidualAdapterConfig(
            recurrent_size=4,
            per_action_membership_confidence=1,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="requires a separate action recurrent"):
        ResidualAdapterConfig(
            recurrent_size=4,
            action_logit_mode="parent_residual_joint",
            per_action_membership_confidence=True,
        )
    with pytest.raises(ValueError, match="residual selector logits"):
        ResidualAdapterConfig(
            recurrent_size=4,
            action_logit_mode="certified_membership",
            separate_action_recurrent=True,
            per_action_membership_confidence=True,
        )
    with pytest.raises(ValueError, match="residual selector logits"):
        ResidualAdapterConfig(
            recurrent_size=4,
            action_logit_mode="absolute",
            separate_action_recurrent=True,
            per_action_membership_confidence=True,
        )

    for mode in ("parent_residual_joint", "parent_residual_factorized"):
        assert ResidualAdapterConfig(
            recurrent_size=4,
            action_logit_mode=mode,
            separate_action_recurrent=True,
            per_action_membership_confidence=True,
        ).per_action_membership_confidence is True


def test_optional_head_preserves_base_parameter_initialization() -> None:
    base = {
        "recurrent_size": 4,
        "hidden_size": 6,
        "ensemble_size": 3,
        "action_logit_mode": "parent_residual_joint",
        "separate_action_recurrent": True,
    }
    torch.manual_seed(1234)
    single = ResidualCorrectionAdapter(ResidualAdapterConfig(**base)).eval()
    torch.manual_seed(1234)
    dual = ResidualCorrectionAdapter(ResidualAdapterConfig(
        **base,
        per_action_membership_confidence=True,
    )).eval()

    # The auxiliary heads are initialized after every original ensemble member.
    # This preserves base tensors, but intentionally advances the global RNG state.
    dual_base_state = {
        name: value
        for name, value in dual.state_dict().items()
        if ".membership_head." not in name
    }
    assert tuple(single.state_dict()) == tuple(dual_base_state)
    assert all(
        torch.equal(value, dual_base_state[name])
        for name, value in single.state_dict().items()
    )


def test_dual_head_preserves_old_forward_api_without_exposing_membership() -> None:
    torch.manual_seed(1234)
    adapter = ResidualCorrectionAdapter(_dual_config()).eval()
    recurrent = torch.randn((2, 4, 4))
    parent_logits = torch.randn((2, 4, 18))
    legacy = adapter.forward_with_all_safety(recurrent, parent_logits)
    extended = adapter.forward_with_all_safety_and_membership(
        recurrent,
        parent_logits,
    )
    assert len(legacy) == 6
    assert len(extended) == 7
    assert extended[5] is not None
    for legacy_value, extended_value in zip(
        legacy[:5],
        extended[:5],
        strict=True,
    ):
        if legacy_value is None:
            assert extended_value is None
        else:
            assert torch.equal(legacy_value, extended_value)
    assert all(torch.equal(left, right) for left, right in zip(
        legacy[-1],
        extended[-1],
        strict=True,
    ))
    assert len(adapter(recurrent, parent_logits)) == 3
    assert len(adapter.forward_with_safety(recurrent, parent_logits)) == 5


def test_dual_membership_and_two_layer_hidden_match_chunked_inference() -> None:
    torch.manual_seed(19)
    adapter = ResidualCorrectionAdapter(_dual_config(ensemble_size=3)).eval()
    recurrent = torch.randn((2, 7, 4))
    parent_logits = torch.randn((2, 7, 18))

    full = adapter.forward_with_all_safety_and_membership(
        recurrent,
        parent_logits,
    )
    first = adapter.forward_with_all_safety_and_membership(
        recurrent[:, :3],
        parent_logits[:, :3],
    )
    second = adapter.forward_with_all_safety_and_membership(
        recurrent[:, 3:],
        parent_logits[:, 3:],
        first[-1],
    )

    assert full[5] is not None and first[5] is not None and second[5] is not None
    assert torch.allclose(
        full[5],
        torch.cat((first[5], second[5]), dim=2),
        atol=1e-6,
    )
    assert all(value.shape == (2, 2, 6) for value in full[-1])
    assert all(torch.allclose(left, right, atol=1e-6) for left, right in zip(
        full[-1],
        second[-1],
        strict=True,
    ))


def test_membership_loss_updates_only_the_detached_membership_head() -> None:
    torch.manual_seed(7)
    adapter = ResidualCorrectionAdapter(_dual_config(ensemble_size=1))
    member = adapter.members[0]
    membership = adapter.forward_with_all_safety_and_membership(
        torch.randn((2, 5, 4)),
        torch.randn((2, 5, 18)),
    )[5]
    assert membership is not None

    membership.square().mean().backward()

    assert member.membership_head is not None
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in member.membership_head.parameters()
    )
    assert all(
        parameter.grad is None
        for name, parameter in member.named_parameters()
        if not name.startswith("membership_head.")
    )


def test_selector_chooses_candidate_and_membership_only_supplies_confidence() -> None:
    selector_logits = torch.zeros((2, 1, 18))
    selector_logits[:, 0, 6] = 5.0
    selector_logits[0, 0, 7] = 4.0
    selector_probabilities, selector_finite = finite_action_probabilities(
        selector_logits,
    )
    membership_logits = torch.full((2, 1, 18), -8.0)
    membership_logits[:, 0, 6] = -2.0
    membership_logits[:, 0, 7] = 8.0
    membership_probabilities, membership_finite = finite_action_probabilities(
        membership_logits,
        "certified_membership",
    )

    summary = ensemble_action_summary(
        selector_probabilities,
        selector_finite,
        membership_probabilities,
        membership_finite,
    )

    assert summary["candidates"].tolist() == [6]
    assert summary["mean_membership_probabilities"].argmax(dim=-1).tolist() == [7]
    assert summary["action_confidence"].item() == pytest.approx(
        torch.sigmoid(torch.tensor(-2.0)).item(),
    )
    assert summary["agreement"].tolist() == [1.0]
    assert summary["action_all_members_finite"].tolist() == [True]


def test_live_uses_selected_membership_confidence_without_membership_argmax() -> None:
    adapter = ResidualCorrectionAdapter(_dual_config())
    with torch.no_grad():
        for parameter in adapter.parameters():
            parameter.zero_()
        for member in adapter.members:
            member.gate_head.bias.fill_(20.0)
            member.action_head.bias[6] = 10.0
            assert member.membership_head is not None
            member.membership_head.bias.fill_(-20.0)
            member.membership_head.bias[6] = -2.0
            member.membership_head.bias[7] = 20.0
    frames = torch.zeros((1, 1, 1))

    rejected = ResidualPolicyWrapper(
        StreamingStubParent(),
        adapter,
        ResidualRuntimeConfig(
            gate_probability_threshold=0.5,
            minimum_member_gate_probability=0.5,
            action_probability_threshold=0.2,
            ensemble_agreement_threshold=1.0,
        ),
    )
    rejected_logits = rejected.forward_with_recurrent(frames, frames)[0]
    assert rejected_logits.argmax(dim=-1).tolist() == [[5]]

    accepted = ResidualPolicyWrapper(
        StreamingStubParent(),
        adapter,
        ResidualRuntimeConfig(
            gate_probability_threshold=0.5,
            minimum_member_gate_probability=0.5,
            action_probability_threshold=0.1,
            ensemble_agreement_threshold=1.0,
        ),
    )
    accepted_logits = accepted.forward_with_recurrent(frames, frames)[0]
    assert accepted_logits.argmax(dim=-1).tolist() == [[6]]
    assert accepted.residual_runtime_stats()["overrides"] == 1


def test_nonfinite_membership_vetoes_live_override_without_changing_selector() -> None:
    adapter = ResidualCorrectionAdapter(_dual_config())
    with torch.no_grad():
        for parameter in adapter.parameters():
            parameter.zero_()
        for member in adapter.members:
            member.gate_head.bias.fill_(20.0)
            member.action_head.bias[6] = 10.0
            assert member.membership_head is not None
            member.membership_head.bias.fill_(20.0)
        adapter.members[0].membership_head.bias[17] = float("nan")
    wrapper = ResidualPolicyWrapper(
        StreamingStubParent(),
        adapter,
        ResidualRuntimeConfig(
            gate_probability_threshold=0.5,
            minimum_member_gate_probability=0.5,
            action_probability_threshold=0.0,
            ensemble_agreement_threshold=0.0,
        ),
    )
    frames = torch.zeros((1, 1, 1))

    logits, _risk, _hidden, _recurrent = wrapper.forward_with_recurrent(
        frames,
        frames,
    )

    assert logits.argmax(dim=-1).tolist() == [[5]]
    assert wrapper.residual_runtime_stats()["overrides"] == 0


def test_nonfinite_selector_vetoes_dual_head_override_with_finite_membership() -> None:
    adapter = ResidualCorrectionAdapter(_dual_config())
    with torch.no_grad():
        for parameter in adapter.parameters():
            parameter.zero_()
        for member in adapter.members:
            member.gate_head.bias.fill_(20.0)
            member.action_head.bias[6] = 10.0
            assert member.membership_head is not None
            member.membership_head.bias.fill_(20.0)
        adapter.members[0].action_head.bias[17] = float("nan")
    wrapper = ResidualPolicyWrapper(
        StreamingStubParent(),
        adapter,
        ResidualRuntimeConfig(
            gate_probability_threshold=0.5,
            minimum_member_gate_probability=0.5,
            action_probability_threshold=0.0,
            ensemble_agreement_threshold=0.0,
        ),
    )
    frames = torch.zeros((1, 1, 1))

    logits, _risk, _hidden, _recurrent = wrapper.forward_with_recurrent(
        frames,
        frames,
    )

    assert logits.argmax(dim=-1).tolist() == [[5]]
    assert wrapper.residual_runtime_stats()["overrides"] == 0


def test_version_seven_round_trip_and_semantic_tamper_rejection(tmp_path) -> None:
    parent = StubParent()
    parent_checkpoint = tmp_path / "parent.pt"
    artifact = tmp_path / "dual-v7.pt"
    torch.save({"identity": "parent"}, parent_checkpoint)
    torch.manual_seed(83)
    adapter = _deployable_dual_adapter().eval()
    runtime = _future_onset_runtime()
    recurrent = torch.randn((2, 5, 4))
    parent_logits = torch.randn((2, 5, 18))
    visual_features = torch.randn((2, 5, 16))
    expected_outputs = adapter.forward_with_all_safety_and_membership(
        recurrent,
        parent_logits,
        visual_features=visual_features,
    )

    save_residual_adapter(
        adapter,
        artifact,
        parent_checkpoint=parent_checkpoint,
        parent_policy_config=asdict(parent.config),
        runtime_config=runtime,
        training_metadata={"action_training": "dual-head"},
    )
    wrapper, metadata = load_residual_adapter(
        parent,
        artifact,
        parent_checkpoint=parent_checkpoint,
    )

    assert metadata["version"] == 7
    assert metadata["selector_logit_semantics"]["probability"] == (
        "finite_per_action_softmax"
    )
    assert metadata["membership_confidence_semantics"][
        "correction_confidence"
    ] == "mean_membership_probability_at_selector_selected_action"
    assert wrapper.adapter.config.per_action_membership_confidence is True
    assert all(member.membership_head is not None for member in wrapper.adapter.members)
    assert all(
        torch.equal(value, wrapper.adapter.state_dict()[name])
        for name, value in adapter.state_dict().items()
    )
    actual_outputs = wrapper.adapter.forward_with_all_safety_and_membership(
        recurrent,
        parent_logits,
        visual_features=visual_features,
    )
    for expected, actual in zip(
        expected_outputs[:-1],
        actual_outputs[:-1],
        strict=True,
    ):
        assert expected is not None and actual is not None
        assert torch.equal(expected, actual)
    assert all(torch.equal(expected, actual) for expected, actual in zip(
        expected_outputs[-1],
        actual_outputs[-1],
        strict=True,
    ))

    original = torch.load(artifact, map_location="cpu", weights_only=False)
    for field, message in (
        ("selector_logit_semantics", "selector semantics do not match"),
        ("membership_confidence_semantics", "membership semantics do not match"),
    ):
        tampered = deepcopy(original)
        tampered[field]["version"] = 99
        torch.save(tampered, artifact)
        with pytest.raises(ValueError, match=message):
            load_residual_adapter(
                StubParent(),
                artifact,
                parent_checkpoint=parent_checkpoint,
            )

    disguised = deepcopy(original)
    disguised["version"] = 6
    torch.save(disguised, artifact)
    with pytest.raises(ValueError, match="cannot declare membership confidence"):
        load_residual_adapter(
            StubParent(),
            artifact,
            parent_checkpoint=parent_checkpoint,
        )


def test_version_six_mapping_without_new_config_field_remains_compatible(
    tmp_path,
) -> None:
    parent = StubParent()
    parent_checkpoint = tmp_path / "parent.pt"
    artifact = tmp_path / "single-v6.pt"
    torch.save({"identity": "parent"}, parent_checkpoint)
    adapter = ResidualCorrectionAdapter(ResidualAdapterConfig(
        recurrent_size=4,
        hidden_size=6,
        ensemble_size=2,
        per_action_safety_critic=True,
        visual_latent_size=16,
        per_action_physical_danger=True,
        action_logit_mode="parent_residual_joint",
        separate_action_recurrent=True,
    ))
    save_residual_adapter(
        adapter,
        artifact,
        parent_checkpoint=parent_checkpoint,
        parent_policy_config=asdict(parent.config),
        runtime_config=_future_onset_runtime(),
        training_metadata={"source_version": 6},
    )
    payload = torch.load(artifact, map_location="cpu", weights_only=False)
    assert payload["version"] == 6
    payload["adapter_config"].pop("per_action_membership_confidence")
    torch.save(payload, artifact)

    wrapper, metadata = load_residual_adapter(
        parent,
        artifact,
        parent_checkpoint=parent_checkpoint,
    )

    assert metadata["version"] == 6
    assert wrapper.adapter.config.per_action_membership_confidence is False
    assert all(member.membership_head is None for member in wrapper.adapter.members)
