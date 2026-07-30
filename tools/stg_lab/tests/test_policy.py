import pytest
import torch

from stg_lab.policy import HumanVisionPolicy, PolicyConfig


def test_policy_forward_and_gradient() -> None:
    config = PolicyConfig(feature_size=24, recurrent_size=32)
    model = HumanVisionPolicy(config)
    global_frames = torch.rand(2, 4, 6, 56, 48)
    local_frames = torch.rand(2, 4, 6, 40, 40)
    memory = torch.rand(2, 4)
    logits, risk, hidden = model(global_frames, local_frames, memory)
    assert logits.shape == (2, 4, 18)
    assert risk.shape == (2, 4)
    assert hidden.shape == (1, 2, 32)
    loss = logits.square().mean() + risk.square().mean()
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable")
def test_policy_forward_supports_real_vision_shapes_on_mps() -> None:
    model = HumanVisionPolicy(PolicyConfig(feature_size=16, recurrent_size=24)).to("mps")
    global_frames = torch.rand(1, 4, 6, 56, 48, device="mps")
    local_frames = torch.rand(1, 4, 6, 40, 40, device="mps")
    logits, risk, _ = model(global_frames, local_frames)
    assert logits.shape == (1, 4, 18)
    assert risk.shape == (1, 4)
