import numpy as np

from when2what.evaluation import (PatientBootstrap, beat_correspondence, chance_f1, event_f1, expression_rate, pair_events,
                                  skill_difference)
from when2what.evaluation.matching import bin_counts
from when2what.evaluation.skill import primary_skill


def test_pairing_is_one_to_one_within_tolerance():
    pred, ref, pi, ri = pair_events([100, 110, 400], [105, 1000], tol=25)
    assert pi.tolist() == [0] and ri.tolist() == [0]


def test_bins_attribute_hits_to_reference_and_false_alarms_to_prediction():
    ref = [500, 1500, 3000]                       # 1 s, 3 s and 6 s
    pred = [510, 2000, 3000]                      # hit, false alarm at 4 s, hit
    c = bin_counts(pred, ref)
    assert c[0].tolist() == [1, 0, 0]             # [0, 2) s
    assert c[1].tolist() == [0, 1, 1]             # [2, 5) s: missed 1500, false alarm 2000
    assert c[2].tolist() == [1, 0, 0]             # [5, 10) s


def test_perfect_forecast_has_f1_one_and_skill_one():
    ref = np.arange(200, 30000, 400)
    f1 = event_f1(ref, ref)
    assert np.allclose(f1, 1.0)
    C = np.stack([chance_f1(ref, rng=np.random.default_rng(k)) for k in range(4)])
    F = np.stack([f1] * 4)
    assert np.isclose(primary_skill(F, C), 1.0)


def test_chance_level_matches_tolerance_fraction():
    ref = np.arange(250, 30000, 500)              # RR 1 s: a random phase lands within +-25 samples w.p. 51 / 500
    c = chance_f1(ref, n_draws=2000, rng=np.random.default_rng(0))
    assert np.all(np.abs(c - 51 / 500) < 0.02)


def test_paired_difference_of_identical_forecasts_is_zero():
    rng = np.random.default_rng(0)
    F, C = rng.random((60, 3)), 0.2 * rng.random((60, 3))
    boot = PatientBootstrap(np.repeat(np.arange(20), 3), n_boot=200)
    d = skill_difference(F, F, C, boot)
    assert d["difference"] == 0.0 and d["lo"] == 0.0 and d["hi"] == 0.0


def test_beat_correspondence_counts():
    ref = [500, 1000, 1500, 6000]                 # last beat outside the first 10 s
    head = [500, 1000, 1500]
    wave = [500, 1700]
    c = beat_correspondence(head, wave, ref)
    assert c == {"n11": 1, "n10": 2, "n01": 0, "n00": 0, "n11_strict": 1}


def test_prediction_takes_the_closest_unmatched_reference():
    _, _, pi, ri = pair_events([100, 104], [102, 140], tol=40)
    assert pi.tolist() == [0, 1] and ri.tolist() == [0, 1]


def test_beat_expression_counts():
    c = beat_correspondence(head_events=[100, 600, 1100], wave_events=[110, 1500], ref=[100, 600, 1100, 1600])
    assert (c["n11"], c["n10"], c["n01"], c["n00"], c["n11_strict"]) == (1, 2, 0, 1, 1)
    rate = expression_rate(np.array([[1.0, 2.0, 1.0], [3.0, 0.0, 2.0]]))
    assert rate["expression"] == 4 / 6 and rate["strict_expression"] == 3 / 6
