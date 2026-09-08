"""Synthetic cohort, uncertainty and advancement checks; no historical scores."""
import copy

import numpy as np
import pandas as pd
import pytest

from research.experiments.boundary_20260908.telemetry import evaluate as e


def population():
    rows = []
    for event in e.SELECTION_EVENTS:
        for lap in [1, 2, 3]:
            matched = lap < 3
            row = dict.fromkeys(e.BASE_FEATURES, 0.0)
            row.update(event_key=event, driver_id='44', issued_after_lap_number=lap,
                       issued_at_timestamp=100. * lap,
                       issuance_id=f'{event}:44:{lap}',
                       forecast_naive_seconds=90., year=2023,
                       outcome_status='matched' if matched else 'unmatched',
                       telemetry_supported=lap == 2,
                       telemetry_values=[0.] * 90,
                       issued_at_ns=100_000_000_000 * lap,
                       telemetry_cutoff_ns=100_000_000_000 * lap - 2_000_000_000,
                       target_id=f'{event}:44:{lap+1}' if matched else None,
                       target_at_ns=100_000_000_000 * lap + 90_000_000_000 if matched else None,
                       target_lap_number=lap+1 if matched else None,
                       target_timestamp=100.*lap+90. if matched else None,
                       lap_time_seconds=90. if matched else None,
                       target_same_stint=True if matched else None)
            rows.append(row)
    primary = pd.DataFrame(rows)
    base = np.full(len(primary), 90. + e.EXPECTED_SELECTION_BASE_MAE)
    supported = primary.telemetry_supported.to_numpy()
    predictions = {'base_hgb': base, 'telemetry_hgb': np.where(supported, 90.49, base),
                   'quality_hgb': np.where(supported, 90.51, base)}
    sensitivity = primary.copy(deep=True)
    sensitivity['telemetry_cutoff_ns'] = sensitivity.issued_at_ns
    expected = primary.loc[primary.outcome_status.eq('matched')].copy()
    original_issuances = primary[[*e.KEYS, 'forecast_naive_seconds', *e.BASE_FEATURES]].copy()
    return primary, predictions, sensitivity, copy.deepcopy(predictions), expected, original_issuances


def report(values):
    f, p, z, zp, expected, original_issuances = values
    return e.evaluate_selection(f, p, z, zp, expected_matched=expected,
                                expected_issuances=original_issuances)


def test_complete_all_event_gate_includes_unmatched_and_fallbacks():
    r = report(population())
    assert r['advances_to_later_evaluation'] is True
    assert r['coverage']['2']['rows_all'] == 66
    assert r['coverage']['2']['rows_matched'] == 44
    assert r['coverage']['2']['rows_unmatched_retained'] == 22
    assert r['coverage']['2']['fallback_rows_matched'] == 22
    assert r['coverage']['2']['fallback_rows_all'] == 44
    assert len(r['coverage']['2']['per_event']) == 22
    assert r['sensitivity_refitted'] is False
    assert set(r['metrics']['2']) == set(e.MODEL_NAMES)


def test_original_event_uncertainty_rng_and_loo_are_exact():
    rng = np.random.default_rng(888)
    keys = np.repeat([202303, 202301, 202302], [13, 2, 7])
    frame = pd.DataFrame({'event_key': keys, 'lap_time_seconds': 90 + rng.normal(size=len(keys))})
    base = np.full(len(keys), 90.)
    candidate = 90 + rng.normal(size=len(keys))
    expected = e.original.diagnostics(frame, candidate, reference=base)
    assert e.comparison(frame, candidate, base) == expected
    deltas = np.array([x['candidate_mae'] - x['baseline_mae'] for x in expected['per_event']])
    assert expected['delta'] == pytest.approx(deltas.mean(), abs=1e-15)
    assert expected['loo_max_delta'] == pytest.approx(max(np.delete(deltas, i).mean() for i in range(3)))
    row_delta = (np.abs(candidate - frame.lap_time_seconds) - np.abs(base - frame.lap_time_seconds)).mean()
    assert abs(expected['delta'] - row_delta) > .01  # equal event weights, not equal row weights


@pytest.mark.parametrize('field,bad', [
    ('relative_reduction', .009999999), ('event_ci95', [-.1, 0.]),
    ('block3_ci95', [-.1, 0.]), ('loo_max_delta', 0.),
    ('loo_max_delta', None),
])
def test_each_fixed_primary_gate_is_required(field, bad):
    good = dict(relative_reduction=.01, event_ci95=[-.2, -.1],
                block3_ci95=[-.2, -.1], loo_max_delta=-.1)
    assert all(e._checks(good, .01).values())
    good[field] = bad
    assert not all(e._checks(good, .01).values())


@pytest.mark.parametrize('reference', e.REFERENCES)
def test_strong_reference_cannot_be_omitted_or_bypassed(reference, monkeypatch):
    values = population()
    original = e.comparison
    reference_values = values[1][reference][values[0].outcome_status.eq('matched')]

    def fail_one(frame, candidate, baseline):
        result = original(frame, candidate, baseline)
        if np.array_equal(baseline, reference_values):
            result['block3_ci95'][1] = 0.
        return result

    monkeypatch.setattr(e, 'comparison', fail_one)
    assert report(values)['advances_to_later_evaluation'] is False


@pytest.mark.parametrize('reference', e.REFERENCES)
def test_zero_second_sensitivity_must_improve_both_corresponding_references(reference):
    values = population()
    values[3]['telemetry_hgb'] = values[3][reference].copy()
    r = report(values)
    assert r['gate_checks']['0'][reference]['positive_relative_gain'] is False
    assert r['advances_to_later_evaluation'] is False


def test_fallback_bit_difference_rejected_even_below_metric_precision():
    values = population()
    values[1]['telemetry_hgb'][0] = np.nextafter(values[1]['base_hgb'][0], np.inf)
    with pytest.raises(ValueError, match='bitwise'):
        report(values)


def test_missing_whole_event_or_matched_row_cannot_improve_population():
    f, p, z, zp, expected, original_issuances = population()
    for keep in [f.event_key.ne(202301).to_numpy(), np.arange(len(f)) != 0]:
        with pytest.raises(ValueError, match='events|cohort|inventory'):
            e.evaluate_selection(f.loc[keep], {k: v[keep] for k, v in p.items()},
                z.loc[keep], {k: v[keep] for k, v in zp.items()}, expected_matched=expected,
                expected_issuances=original_issuances)


def test_original_target_and_base_features_cannot_change():
    for column in ['lap_time_seconds', 'target_timestamp', 'forecast_naive_seconds', e.BASE_FEATURES[0]]:
        values = population()
        values[0].loc[0, column] += .01
        with pytest.raises(ValueError, match='original (matched|issuance) value'):
            report(values)


@pytest.mark.parametrize('mutation', ['unmatched_label', 'matched_nan', 'missing_status', 'duplicate', 'target_identity'])
def test_explicit_outcome_and_identity_failures(mutation):
    values = population()
    f = values[0]
    if mutation == 'unmatched_label':
        f.loc[2, 'lap_time_seconds'] = 1.
    elif mutation == 'matched_nan':
        f.loc[0, 'lap_time_seconds'] = np.nan
    elif mutation == 'missing_status':
        f.loc[0, 'outcome_status'] = 'pending'
    elif mutation == 'duplicate':
        f.loc[1, 'issuance_id'] = f.loc[0, 'issuance_id']
    else:
        f.loc[1, 'target_id'] = f.loc[0, 'target_id']
    with pytest.raises(ValueError):
        report(values)


def test_sensitivity_population_must_be_identical_and_ordered():
    f, p, z, zp, expected, original_issuances = population()
    order = np.arange(len(z))[::-1]
    with pytest.raises(ValueError, match='sensitivity changed'):
        e.evaluate_selection(f, p, z.iloc[order], {k: v[order] for k, v in zp.items()}, expected_matched=expected,
                             expected_issuances=original_issuances)


def test_shipped_2022_23_base_cannot_silently_replace_discovery_base():
    values = population()
    for p in [values[1], values[3]]:
        for name in e.MODEL_NAMES:
            p[name] += .1
    with pytest.raises(ValueError, match='2022-only base HGB'):
        report(values)


def test_unmatched_forecasts_are_required_even_though_not_scored():
    values = population()
    values[1]['base_hgb'][2] = np.nan
    with pytest.raises(ValueError, match='every issuance'):
        report(values)


def test_full_original_issuance_inventory_is_mandatory():
    f, p, z, zp, expected, _ = population()
    with pytest.raises(TypeError, match='expected_issuances'):
        e.evaluate_selection(f, p, z, zp, expected_matched=expected)


@pytest.mark.parametrize('drop_all_unmatched', [False, True])
def test_deleting_unmatched_issues_fails_even_with_all_matches_and_events(drop_all_unmatched):
    f, p, z, zp, expected, original_issuances = population()
    keep = (f.outcome_status.eq('matched').to_numpy() if drop_all_unmatched
            else np.arange(len(f)) != 2)
    assert set(f.loc[keep].event_key) == set(e.SELECTION_EVENTS)
    assert f.loc[keep].outcome_status.eq('matched').sum() == len(expected)
    with pytest.raises(ValueError, match='including unmatched'):
        e.evaluate_selection(f.loc[keep], {k: v[keep] for k, v in p.items()},
            z.loc[keep], {k: v[keep] for k, v in zp.items()}, expected_matched=expected,
            expected_issuances=original_issuances)


@pytest.mark.parametrize('column', ['forecast_naive_seconds', e.BASE_FEATURES[0], e.BASE_FEATURES[-1]])
def test_unmatched_issuances_preserve_original_anchor_and_base_features(column):
    values = population()
    values[0].loc[2, column] += .01
    with pytest.raises(ValueError, match='original issuance value'):
        report(values)


def test_extra_unmatched_issuance_cannot_be_added_to_coverage():
    f, p, z, zp, expected, original_issuances = population()
    extra = f.iloc[[-1]].copy()
    extra['issued_after_lap_number'] = 99
    extra['issuance_id'] = 'extra'
    f = pd.concat([f, extra], ignore_index=True)
    z = pd.concat([z, extra], ignore_index=True)
    p = {k: np.append(v, v[-1]) for k, v in p.items()}
    zp = {k: np.append(v, v[-1]) for k, v in zp.items()}
    with pytest.raises(ValueError, match='original inventory'):
        e.evaluate_selection(f, p, z, zp, expected_matched=expected,
                             expected_issuances=original_issuances)


def test_original_inventory_order_can_differ_without_changing_issuance_alignment():
    values = list(population())
    values[5] = values[5].iloc[::-1]
    assert report(values)['advances_to_later_evaluation'] is True


@pytest.mark.parametrize('difference', [0., .1])
def test_perfect_reference_has_null_relative_gain_and_fails_gate(difference):
    import json
    frame = pd.DataFrame({'event_key': [202301, 202302], 'lap_time_seconds': [90., 91.]})
    reference = frame.lap_time_seconds.to_numpy()
    result = e.comparison(frame, reference + difference, reference)
    assert result['relative_reduction'] is None
    assert result['baseline_mae'] == 0.
    assert e._checks(result, .01)['minimum_relative_gain'] is False
    json.dumps(result, allow_nan=False)
