from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from measurement import clean_laps, match_laps, predict, fit_huber, objective
from run_experiment import latest_pre_q, select


def sample():
    drivers = list('ABCDEFGH')
    frame = pd.DataFrame({'driver_id': drivers, 'team_id': ['T' + str(i // 2) for i in range(8)],
                          'raw_anchor': 90 + np.arange(8) * .1, 'Q2_ridge_rank_residual': np.arange(1, 9)})
    laps = pd.DataFrame([{'driver_id': d, 'lap_seconds': 90 + i * .1, 'clock': 1000 + j * 100 + i,
                          'age': j + 2, 'compound': 'SOFT', 'stint': 1}
                         for j in range(3) for i, d in enumerate(drivers)])
    return frame, laps


def test_filter_rejects_inaccurate_pit_unknown_age_and_non_green_laps():
    raw = pd.DataFrame({'Driver': list('ABCDEF'), 'LapTime': [90.] * 6, 'Time': [1000.] * 6,
                        'Compound': ['SOFT'] * 6, 'TyreLife': [2, 2, np.nan, 2, 2, 2],
                        'IsAccurate': [True, False, True, True, True, True],
                        'TrackStatus': [1, 1, 1, 2, 1, 1],
                        'PitInTime': [np.nan] * 5 + [1100.],
                        'PitOutTime': [np.nan] * 4 + [900., np.nan]})
    assert clean_laps(raw).driver_id.tolist() == ['A']


def test_matching_respects_compound_clock_age_and_independent_drivers():
    _, laps = sample()
    edge, support = match_laps(laps)
    assert all(v['supported'] for v in support.values())
    for a, b in edge:
        x, y = laps.iloc[a], laps.iloc[b]
        assert x.driver_id != y.driver_id and x.compound == y.compound
        assert abs(x.clock - y.clock) <= 180 and abs(x.age - y.age) <= 2
    isolated = laps.copy(); isolated.loc[isolated.driver_id.eq('A'), 'compound'] = 'WET'
    _, support = match_laps(isolated)
    assert not support['A']['supported']


def test_robust_normal_equations_match_independent_convex_optimizer():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(50, 4)); y = x @ np.array([.3, -.4, .2, 1.])
    y[[2, 7, 10]] += [20, -10, 8]
    w = rng.uniform(.5, 2, 50); penalty = np.array([1., 2., 3., 4.])
    beta, _, diagnostic = fit_huber(x, y, w, penalty)
    other = minimize(lambda b: objective(b, x, y, w, penalty), np.zeros(4), method='BFGS',
                     jac=lambda b: x.T @ (w * np.clip(x @ b - y, -3, 3)) + penalty * b,
                     options={'gtol': 1e-9})
    np.testing.assert_allclose(beta, other.x, rtol=0, atol=1e-6)
    assert diagnostic['gradient_max_abs'] < 1e-5


def test_prediction_is_immutable_and_invariant_to_common_clock_translation():
    frame, laps = sample(); original = frame.copy(deep=True)
    first, diagnostics = predict(frame, laps, .5, .5)
    shifted = laps.copy(); shifted.clock += 100000
    second, later = predict(frame, shifted, .5, .5)
    assert first.tolist() == second.tolist() and sorted(first) == list(range(1, 9))
    np.testing.assert_allclose(diagnostics['applied_adjustment_equivalent_positions'],
                               later['applied_adjustment_equivalent_positions'], atol=1e-10)
    pd.testing.assert_frame_equal(frame, original)


def test_unsupported_event_preserves_exact_Q2_and_target_labels_are_rejected():
    frame, laps = sample(); laps.clock = np.arange(len(laps)) * 1000
    ranks, diagnostics = predict(frame, laps, .5, 1)
    assert diagnostics['status'] == 'whole_event_Q2_fallback'
    assert ranks.tolist() == frame.Q2_ridge_rank_residual.tolist()
    with pytest.raises(ValueError, match='target'):
        predict(frame.assign(qualy_position=np.arange(8)), laps, .5, 1)


def test_post_qualifying_practice_cannot_be_selected():
    sessions = [{'session_name': 'Qualifying', 'session_type': 'qualifying', 'session_order': 2},
                {'session_name': 'Practice 3', 'session_type': 'free_practice', 'session_order': 3}]
    with pytest.raises(ValueError, match='causal'):
        latest_pre_q({'sessions': sessions})


def test_selection_uses_only_2023_scores():
    candidates = {'A': {'penalty_strength': .5, 'blend_strength': 1},
                  'B': {'penalty_strength': 4, 'blend_strength': .5}}
    events = [{'year': 2023, 'event_key': 202301 + i, 'models': {'A': {'mae': 2}, 'B': {'mae': 3}}}
              for i in range(16)]
    events.append({'year': 2024, 'event_key': 202401, 'models': {'A': {'mae': 99}, 'B': {'mae': 0}}})
    assert select(events, candidates)['selected'] == 'A'
