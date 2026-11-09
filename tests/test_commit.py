import numpy as np

from when2what.data.labels import timing_map
from when2what.inference import commit_events, committed_map, history_rr, pick_peaks


def test_history_rr_uses_last_five_valid_intervals():
    beats = np.cumsum([400] * 10 + [50, 2000] + [420] * 5)
    assert history_rr(beats, 500) == 420.0
    assert history_rr([], 500) == 400.0           # 0.8 s default


def test_pick_peaks_respects_spacing():
    x = np.zeros(1000)
    x[[100, 140, 500]] = [1.0, 0.9, 0.8]
    assert pick_peaks(x, 500, 0.1, 0.5).tolist() == [100, 500]


def test_commit_recovers_decaying_beats_and_redraws_unit_bumps():
    beats = np.arange(300, 30000, 400)
    decay = np.exp(-np.arange(30000) / 15000.0)
    m_hat = timing_map(beats, 30000, 500, 0.03) * decay + 0.02
    events = commit_events(m_hat, history_events=np.arange(0, 60000, 400))
    assert np.array_equal(events, beats)
    m = committed_map(events, 30000)
    assert np.isclose(m.max(), 1.0) and np.allclose(m[beats], 1.0)


def test_event_perturbations():
    from when2what.inference import perturb_events, window_rng
    e = np.arange(10) * 400 + 200
    deleted = perturb_events(e, "delete", window_rng(3))
    inserted = perturb_events(e, "insert", window_rng(3))
    assert deleted.size == 7 and set(deleted) <= set(e)
    assert inserted.size == 13 and set(e) <= set(inserted) and inserted.min() >= e.min() and inserted.max() <= e.max()
