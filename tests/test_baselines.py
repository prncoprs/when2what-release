import numpy as np

from when2what.baselines import simple_rr_extrapolation, template_forecast
from when2what.baselines.cadence import correct_rr


def test_regular_rhythm_is_continued():
    hist = np.arange(100, 60000, 400)
    ev = simple_rr_extrapolation(hist)
    assert np.array_equal(ev, np.arange(hist[-1] + 400 - 60000, 30000, 400))


def test_missed_beat_is_split():
    rr, dropped, _ = correct_rr(np.array([0.8] * 6 + [1.6] + [0.8] * 6))
    assert np.allclose(rr, 0.8) and rr.size == 14 and not dropped


def test_template_is_placed_at_the_events():
    x = np.zeros(60000)
    beats = np.arange(500, 59500, 400)
    x[beats] = 1.0
    w = template_forecast(x, beats, np.array([1000, 5000]))
    assert w[1000] == 1.0 and w[5000] == 1.0 and w[3000] == 0.0
