import torch

from when2what.training.losses import soft_f1, split_mse, timing_loss, waveform_loss


def test_waveform_loss_is_zero_for_a_perfect_forecast():
    y = torch.randn(2, 100)
    assert waveform_loss(y, y).item() == 0.0


def test_split_mse_weights_the_inner_band():
    y = torch.tensor([[0.0, 3.0]])
    pred = torch.tensor([[1.0, 4.0]])
    assert split_mse(pred, y).item() == 5.0 * 1.0 + 1.0 * 1.0


def test_timing_loss_prefers_the_true_map():
    true = torch.zeros(1, 1000)
    true[0, 100:110] = 1.0
    assert timing_loss(true, true).item() < timing_loss(torch.zeros_like(true), true).item()


def test_soft_f1_is_the_overlap_of_eq_4():
    true = torch.tensor([[0.0, 1.0, 0.5, 0.0]])
    pred = torch.tensor([[0.2, 0.8, 1.5, -0.3]])
    p = pred.clamp(0.0, 1.0)
    expected = 2.0 * (p * true).sum() / (p.sum() + true.sum())
    assert torch.allclose(soft_f1(pred, true), expected.reshape(1), atol=1e-6)
