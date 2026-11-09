import numpy as np

from when2what.data.filters import ecg_sos, history_input, normalize_history, periodic_right_pad


def _ecg_like(n, fs=500, seed=0):
    t = np.arange(n) / fs
    x = np.zeros(n)
    for b in np.arange(0.3, t[-1], 0.8):
        x += np.exp(-0.5 * ((t - b) / 0.01) ** 2)
    return x + 0.01 * np.random.default_rng(seed).normal(size=n)


def test_history_ignores_samples_after_the_origin():
    raw = _ecg_like(80000)
    start = 5000
    a = history_input(raw, 0, start, ecg_sos(), 60000, 5000, 2500, 500)
    changed = raw.copy()
    changed[start + 60000:] = 50.0 * np.random.default_rng(1).normal(size=changed.size - start - 60000)
    b = history_input(changed, 0, start, ecg_sos(), 60000, 5000, 2500, 500)
    assert a.shape == (60000,) and np.array_equal(a, b)


def test_periodic_pad_continues_the_rhythm():
    x = _ecg_like(10000)                          # beats every 400 samples
    pad = periodic_right_pad(x, 2000, 500)        # five whole periods, shifted to start at x[-1]'s level
    assert pad.shape == (2000,) and np.allclose(pad, x[-2000:] + (x[-1] - x[-2001]))


def test_normalisation():
    z, mu, sd = normalize_history(np.array([1.0, 2.0, 3.0], np.float32))
    assert np.isclose(mu, 2.0) and np.allclose(z.mean(), 0) and np.isclose(z.std(), 1)
