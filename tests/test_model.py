import math

import pytest
import torch

pytest.importorskip("mamba_ssm")

from when2what.config import load_config
from when2what.models import build_model, count_parameters, use_reference_mamba


@pytest.fixture(scope="module")
def model():
    use_reference_mamba()
    torch.manual_seed(0)
    return build_model(load_config()).eval()


def test_parameter_count(model):
    assert count_parameters(model) == 24_879_348


def test_forward_shapes_and_zero_map(model):
    x = torch.randn(1, 60000)
    with torch.no_grad():
        out = model(x)
        assert out["timing"].shape == (1, 30000) and out["wave"].shape == (1, 30000)
        assert torch.equal(out["wave"], model.generate(x, torch.zeros(1, 30000)))


def test_spectral_gate_is_a_sigmoid_gate(model):
    gate = torch.sigmoid(model.spectral.gate)
    assert gate.shape == (1201,)
    assert torch.allclose(gate, torch.full_like(gate, math.log(2.0)))


def test_tcn_backbone_builds():
    assert count_parameters(build_model(load_config(overrides={"backbone": "tcn"}))) == 19_668_212
