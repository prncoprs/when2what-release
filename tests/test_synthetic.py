import numpy as np
import torch

from when2what.synthetic import generator as G
from when2what.synthetic import models as M


def test_records_are_deterministic():
    a = G.make_split(2, 3, "test", n=2)
    b = G.make_split(2, 3, "test", n=2)
    assert all(np.array_equal(a[k], b[k], equal_nan=True) for k in a)
    assert a["hist"].shape == (2, 2000) and a["fut"].shape == (2, 1000)


def test_synth_net_shapes():
    net = M.SynthNet()
    hist, resp_h, resp_f = torch.zeros(2, 2000), torch.zeros(2, 2000), torch.zeros(2, 1000)
    wave, logits = net(hist, resp_h, resp_f, torch.zeros(2, 1000))
    assert wave.shape == (2, 1000) and logits.shape == (2, 1000)
    loss, _ = M.losses(net, "when2what", (hist, resp_h, resp_f, torch.zeros(2, 1000), torch.zeros(2, 1000)))
    assert torch.isfinite(loss)
