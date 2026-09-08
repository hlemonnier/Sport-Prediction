"""New gap identity, full-cohort and reference-bit contracts; synthetic only."""
import copy

import numpy as np
import pandas as pd
import pytest

from . import evaluate as e
from . import models as m
from .test_models import frame, references, target_free


@pytest.fixture(scope='module')
def selection():
    primary = frame(e.SELECTION_ROWS, year=2023)
    primary['lap_time_seconds'] = 90.
    unmatched = np.arange(len(primary)) % 48 == 0
    unmatched[np.flatnonzero(unmatched)[e.SELECTION_UNMATCHED:]] = False
    primary['outcome_status'] = np.where(unmatched, 'unmatched', 'matched')
    primary['target_id'] = [f'{identifier}:target' for identifier in primary.issuance_id]
    primary['target_at_ns'] = pd.Series(primary.issued_at_ns+90_000_000_000, dtype=object)
    primary['target_same_stint'] = primary.target_same_stint.astype(object)
    for column in e.original.TARGET_FIELDS:
        primary.loc[unmatched, column] = None
    baseline = np.full(len(primary), 90.+e.original.EXPECTED_SELECTION_BASE_MAE, dtype=np.float64)
    support = primary.gap_supported.to_numpy()
    predictions = {'base_hgb': baseline, 'gap_hgb': np.where(support, 90.1, baseline),
                   'quality_hgb': np.where(support, 90.4, baseline)}
    zero = primary.copy(deep=True);zero['gap_cutoff_ns'] = zero.issued_at_ns
    expected = primary.loc[~unmatched].copy()
    issued = target_free(primary)
    return (primary, predictions, zero, copy.deepcopy(predictions), expected, issued,
            references(primary, baseline), references(zero, baseline))


def report(values):
    primary, points, zero, zpoints, matched, issued, refs, zrefs = values
    return e.evaluate_selection(primary, points, zero, zpoints, expected_matched=matched,
        expected_issuances=issued, primary_references=refs, sensitivity_references=zrefs)


@pytest.fixture(scope='module')
def passed_report(selection):
    return report(selection)


def test_only_gap_advances_against_both_controls_on_complete_original_inventory(passed_report):
    result = passed_report
    assert result['candidate_eligible_for_advancement'] == 'gap_hgb'
    assert result['advances_to_later_evaluation'] is True and result['promotion'] is False
    assert result['sensitivity_refitted'] is False
    assert result['expected_events'] == list(e.SELECTION_EVENTS)
    for lag in ('2', '0'):
        assert set(result['metrics'][lag]) == set(m.MODEL_NAMES)
        assert set(result['comparisons'][lag]) == set(e.REFERENCES)
        assert set(result['gate_checks'][lag]) == set(e.REFERENCES)
        assert set(result['contributing_events_and_drivers'][lag]) == set(e.REFERENCES)
        assert result['coverage'][lag]['rows_all'] == 20432
        assert result['coverage'][lag]['rows_matched'] == 20007
        assert result['coverage'][lag]['rows_unmatched_retained'] == 425
        for name in m.MODEL_NAMES:
            assert len(result['metrics'][lag][name]['per_event']) == 22
        for reference in e.REFERENCES:
            diagnostic = result['contributing_events_and_drivers'][lag][reference]
            assert diagnostic['event_balanced_delta_seconds'] == pytest.approx(result['comparisons'][lag][reference]['delta'], abs=1e-15)
            assert sum(row['contribution_to_event_mae_delta_seconds'] for row in diagnostic['per_driver']) == pytest.approx(diagnostic['event_balanced_delta_seconds'], abs=1e-15)


def test_adapter_preserves_every_original_numeric_metric_interval_gate_and_decision(selection, passed_report):
    primary, points, zero, zpoints, matched, issued, _, _ = selection
    pf = primary.assign(telemetry_supported=primary.gap_supported)
    zf = zero.assign(telemetry_supported=zero.gap_supported)
    renamed = lambda p: {('telemetry_hgb' if name == 'gap_hgb' else name): value for name, value in p.items()}
    original = e.original.evaluate_selection(pf, renamed(points), zf, renamed(zpoints),
        expected_matched=matched, expected_issuances=issued)
    for key in ('comparisons', 'gate_checks', 'coverage', 'uncertainty', 'baseline_reproduction',
                'advances_to_later_evaluation', 'decision'):
        assert passed_report[key] == original[key]
    for lag in ('2', '0'):
        for name in m.MODEL_NAMES:
            old_name = 'telemetry_hgb' if name == 'gap_hgb' else name
            for key, value in original['metrics'][lag][old_name].items():
                assert passed_report['metrics'][lag][name][key] == value


@pytest.mark.parametrize('reference', e.REFERENCES)
def test_candidate_cannot_bypass_either_primary_reference(selection, monkeypatch, reference):
    original = e.original.comparison;calls = []

    def comparison(frame, candidate, baseline):
        index = len(calls);calls.append(index)
        value = original(frame, candidate, baseline)
        if index < 2 and e.REFERENCES[index] == reference:
            value['block3_ci95'][1] = 0.
        return value

    monkeypatch.setattr(e.original, 'comparison', comparison)
    result = report(selection)
    assert result['advances_to_later_evaluation'] is False
    assert result['gate_checks']['2'][reference]['block3_ci_upper_negative'] is False


@pytest.mark.parametrize('reference', e.REFERENCES)
def test_zero_lag_must_beat_both_references_without_reselection(selection, reference):
    values = list(selection);values[3] = copy.deepcopy(values[3])
    values[3]['gap_hgb'] = values[3][reference].copy()
    result = report(values)
    assert result['advances_to_later_evaluation'] is False
    assert result['gate_checks']['0'][reference]['positive_relative_gain'] is False


@pytest.mark.parametrize('fault', ['base_bit', 'gap_fallback_bit', 'quality_fallback_bit', 'omitted_quality', 'changed_id', 'float32'])
def test_saved_base_identity_and_both_unsupported_controls_are_exact(selection, fault):
    primary, points, *_rest, reference, _zero_reference = selection
    points = copy.deepcopy(points)
    if fault == 'base_bit': points['base_hgb'][1] = np.nextafter(points['base_hgb'][1], np.inf)
    elif fault == 'gap_fallback_bit': points['gap_hgb'][0] = np.nextafter(points['base_hgb'][0], np.inf)
    elif fault == 'quality_fallback_bit': points['quality_hgb'][0] = np.nextafter(points['base_hgb'][0], np.inf)
    elif fault == 'omitted_quality': del points['quality_hgb']
    elif fault == 'changed_id':
        primary = primary.copy();primary.loc[0, 'issuance_id'] = 'different'
    else: points['gap_hgb'] = points['gap_hgb'].astype(np.float32)
    with pytest.raises(ValueError): e._predictions(primary, points, reference)


@pytest.mark.parametrize('fault', ['full_rows', 'matched_rows', 'unmatched_status', 'original_order', 'changed_target_id', 'changed_ns', 'wrong_lag'])
def test_counts_order_labels_and_clocks_fail_before_scoring(selection, fault):
    primary, _points, _zero, _zpoints, matched, issued, *_ = selection
    primary = primary.copy();matched = matched.copy();issued = issued.copy()
    if fault == 'full_rows': primary = primary.iloc[:-1]
    elif fault == 'matched_rows': matched = matched.iloc[:-1]
    elif fault == 'unmatched_status': primary.loc[0, 'outcome_status'] = 'matched'
    elif fault == 'original_order': issued = issued.iloc[::-1]
    elif fault == 'changed_target_id': primary.loc[1, 'target_id'] = 'altered'
    elif fault == 'changed_ns': primary.loc[1, 'issued_at_ns'] += 1
    else: primary['gap_cutoff_ns'] += 1
    with pytest.raises(ValueError):
        e._population(primary, matched, issued, latency_seconds=2)


def test_event_weighted_driver_contributions_do_not_revert_to_row_weighting():
    value = pd.DataFrame({'event_key': [202301]+[202302]*100,
                          'driver_id': ['44']+['1']*100, 'lap_time_seconds': np.full(101, 90.)})
    reference = np.full(101, 91.)
    candidate = np.array([92.]+[90.5]*100)
    result = e.contributions(value, candidate, reference)
    assert result['event_balanced_delta_seconds'] == pytest.approx(.25)
    assert result['row_mae_delta_seconds'] == pytest.approx(-49/101)
    drivers = {r['driver_id']: r['contribution_to_event_mae_delta_seconds'] for r in result['per_driver']}
    assert drivers == pytest.approx({'44': .5, '1': -.25})
    assert sum(drivers.values()) == pytest.approx(.25)


def test_ordinary_block_bootstrap_and_loo_agree_with_independent_manual_event_oracle(selection, passed_report):
    primary, points, *_ = selection
    mask = primary.outcome_status.eq('matched').to_numpy()
    y = primary.loc[mask, 'lap_time_seconds'].to_numpy()
    row_delta = np.abs(points['gap_hgb'][mask]-y)-np.abs(points['quality_hgb'][mask]-y)
    by_event = pd.DataFrame({'event': primary.loc[mask, 'event_key'].to_numpy(), 'delta': row_delta}).groupby('event').delta.mean().to_numpy()
    rng = np.random.default_rng(20260907);n = len(by_event)
    ordinary = by_event[rng.integers(n, size=(20000, n))].mean(axis=1)
    starts = rng.integers(n, size=(20000, int(np.ceil(n/3))))
    indices = ((starts[:, :, None]+np.arange(3))%n).reshape(20000, -1)[:, :n]
    block = by_event[indices].mean(axis=1)
    result = passed_report['comparisons']['2']['quality_hgb']
    np.testing.assert_allclose(result['event_ci95'], np.quantile(ordinary, [.025, .975]), rtol=0, atol=1e-15)
    np.testing.assert_allclose(result['block3_ci95'], np.quantile(block, [.025, .975]), rtol=0, atol=1e-15)
    assert result['loo_max_delta'] == pytest.approx(max(np.delete(by_event, i).mean() for i in range(n)), abs=1e-15)
