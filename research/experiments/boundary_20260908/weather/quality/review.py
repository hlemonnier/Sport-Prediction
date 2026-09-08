"""Read-only weather source/parser/clock QA; no model fitting or error scoring."""
from pathlib import Path
from collections import Counter
import hashlib
import importlib.util
import json
import pickle
import re
import sys
import numpy as np
import pandas as pd
import fastf1
from fastf1.utils import to_timedelta

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
ACQUIRE = HERE.parent / 'acquire.py'
OUT = ROOT / 'artifacts/research/boundary_20260908/weather/quality'
MANIFEST = OUT.parent / 'input_manifest.json'
spec = importlib.util.spec_from_file_location('reviewed_acquisition', ACQUIRE)
acquisition = importlib.util.module_from_spec(spec); spec.loader.exec_module(acquisition)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def native_examples():
    examples = [
        ('2026-03-08_Australian_Grand_Prix', '2026-03-08_Race', 'round_01_australian_grand_prix'),
        ('2026-03-29_Japanese_Grand_Prix', '2026-03-29_Race', 'round_03_japanese_grand_prix'),
        ('2026-05-03_Miami_Grand_Prix', '2026-05-03_Race', 'round_04_miami_grand_prix'),
    ]
    evidence = []
    for meeting, race, event in examples:
        native = ROOT / 'data/f1/cache/2026' / meeting / race
        exported = ROOT / 'data/f1/raw/weekends/2026' / event
        wp, tp = native / 'weather_data.ff1pkl', native / '_extended_timing_data.ff1pkl'
        wc, lc = exported / '05_race_weather.csv', exported / '05_race_laps.csv'
        with wp.open('rb') as handle: weather = pd.DataFrame(pickle.load(handle)['data'])
        weather.Time = pd.to_timedelta(weather.Time).dt.total_seconds()
        pd.testing.assert_frame_equal(weather, pd.read_csv(wc), check_dtype=False, check_exact=False, atol=1e-9, rtol=0)
        with tp.open('rb') as handle: timing = pickle.load(handle)['data'][0].copy()
        timing['driver_number'] = pd.to_numeric(timing.Driver)
        timing['lap'] = pd.to_numeric(timing.NumberOfLaps)
        timing['native_time'] = pd.to_timedelta(timing.Time).dt.total_seconds()
        laps = pd.read_csv(lc); laps['driver_number'] = pd.to_numeric(laps.DriverNumber); laps['lap'] = pd.to_numeric(laps.LapNumber)
        matched = laps.merge(timing[['driver_number', 'lap', 'native_time']], on=['driver_number', 'lap'], how='left', validate='one_to_one')
        gap = (matched.Time - matched.native_time).abs()
        assert gap.max() == 0
        evidence.append({'event': event, 'native_weather_equals_exported_rows': len(weather),
                         'comparable_lap_timestamps': int(gap.notna().sum()), 'maximum_timestamp_difference_seconds': float(gap.max()),
                         'noncomparable_missing_native_lap_times': int(gap.isna().sum()),
                         'source_hashes': {str(p.relative_to(ROOT)): sha(p) for p in [wp, tp, wc, lc]}})
    return evidence


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = load(MANIFEST)
    assert len(manifest['events']) == 44 and not manifest['failed_events']
    assert sha(ACQUIRE) == manifest['source_sha256']
    indices = {}
    for index in manifest['index_sources']:
        path = ROOT / index['path']; assert sha(path) == index['sha256']; indices[index['year']] = load(path)['Meetings']
    discovery = pd.read_pickle(ROOT / 'artifacts/research/frontier_20260907/live/discovery_data.pkl')
    issued = discovery[['event_key', 'issued_at_timestamp']]
    rainfall_values, field_presence = Counter(), Counter()
    all_intervals, all_ages, event_reports, numeric_values = [], [], [], []
    partial = []; mismatched_numbers = []; receipt_hashes = {}
    for event in manifest['events']:
        key = event['event_key']; raw_path, csv_path = ROOT / event['raw_path'], ROOT / event['csv_path']
        assert sha(raw_path) == event['raw_sha256'] and sha(csv_path) == event['csv_sha256']
        assert sha(ROOT / event['original_laps_path']) == event['original_laps_sha256']
        receipt_path = raw_path.with_name(raw_path.name + '.source.json')
        receipt = load(receipt_path); assert receipt['sha256'] == event['raw_sha256']
        assert receipt['url'] == event['actual_download_url']; receipt_hashes[str(receipt_path.relative_to(ROOT))] = sha(receipt_path)
        meetings = [m for m in indices[key // 100] if m['Key'] == event['meeting_key']]
        assert len(meetings) == 1 and meetings[0]['Name'] == event['meeting_name']
        session = [s for s in meetings[0]['Sessions'] if s.get('Name') == 'Race']
        assert len(session) == 1 and session[0]['Key'] == event['session_key'] and session[0]['Path'] == event['session_path']
        if event['provider_meeting_number'] != key % 100:
            mismatched_numbers.append({'event_key': key, 'provider_meeting_number': event['provider_meeting_number'], 'meeting_name': event['meeting_name']})
        raw = raw_path.read_bytes(); frame = acquisition.parse(raw)
        csv = pd.read_csv(csv_path)
        pd.testing.assert_frame_equal(frame, csv, check_exact=False, atol=1e-12, rtol=0)
        response = []
        for line in raw.decode('utf-8-sig').splitlines():
            if not line.strip(): continue
            clock, entry = line[:12], json.loads(line[12:])
            assert re.fullmatch(r'\d{2}:\d{2}:\d{2}\.\d{3}', clock)
            field_presence[len(entry)] += 1
            rainfall_values[repr(entry.get('Rainfall'))] += 1
            missing = sorted(set(acquisition.CHANNELS) - set(entry))
            if missing: partial.append({'event_key': key, 'clock': clock, 'missing': missing})
            response.append(to_timedelta(clock).total_seconds())
        np.testing.assert_allclose(response, frame.Time, rtol=0, atol=1e-9)
        assert not frame[acquisition.CHANNELS].isna().any().any()
        assert frame.Rainfall.isin([0., 1.]).all()
        intervals = np.diff(frame.Time); assert (intervals > 0).all()
        checkpoints = issued.loc[issued.event_key.eq(key), 'issued_at_timestamp'].to_numpy()
        weather_clock = frame.Time.to_numpy()
        positions = np.searchsorted(weather_clock, checkpoints - 60, side='left') - 1
        assert (positions >= 0).all()
        ages = checkpoints - weather_clock[positions]
        assert (ages > 60).all()
        all_intervals.extend(intervals); all_ages.extend(ages); numeric_values.append(frame[acquisition.CHANNELS])
        event_reports.append({'event_key': key, 'weather_rows': len(frame), 'eligible_matched_checkpoints': len(checkpoints),
                              'rain_records': int(frame.Rainfall.sum()), 'maximum_update_gap_seconds': float(intervals.max()),
                              'maximum_weather_age_at_strict60s_checkpoint_seconds': float(ages.max()),
                              'checkpoint_weather_age_above180_seconds': int((ages > 180).sum()),
                              'checkpoint_weather_age_above300_seconds': int((ages > 300).sum())})
    numerical = pd.concat(numeric_values)
    native = native_examples()
    report = {'status': 'PASS_WITH_RECORDED_RECEIPT_LATENCY_LIMITATION', 'manifest_sha256': sha(MANIFEST),
              'reviewer_source_sha256': sha(__file__), 'acquisition_source_sha256': sha(ACQUIRE),
              'installed_fastf1_version': fastf1.__version__, 'events': len(event_reports), 'weather_rows': len(numerical),
              'field_count_histogram': dict(field_presence), 'partial_updates': partial,
              'raw_rainfall_values': dict(rainfall_values), 'missing_numeric_fields': int(numerical.isna().sum().sum()),
              'numeric_ranges': {c: [float(numerical[c].min()), float(numerical[c].max())] for c in acquisition.CHANNELS},
              'update_interval_seconds_quantiles_0_50_99_100': np.quantile(all_intervals, [0, .5, .99, 1]).tolist(),
              'strict60s_join_weather_age_seconds_quantiles_0_50_99_100': np.quantile(all_ages, [0, .5, .99, 1]).tolist(),
              'strict60s_join_missing_weather_checkpoints': 0,
              'strict60s_join_age_above180_seconds': int((np.array(all_ages) > 180).sum()),
              'strict60s_join_age_above300_seconds': int((np.array(all_ages) > 300).sum()),
              'canonical_round_vs_provider_calendar_number_differences': mismatched_numbers,
              'numbering_interpretation': 'Expected2023 calendar numbering after cancelled meeting; identity is bound by unique provider meeting name/key and Race session key/path.',
              'event_quality': event_reports, 'native_cache_alignment': native, 'source_receipt_hashes': receipt_hashes,
              'time_contract': 'Use weather stream seconds directly against lap completion/session Time; do not subtract actual/scheduled race start or apply GMT offset. Both use stream-zero origin.',
              'latency_limit': 'Archived stream clocks are not historical consumer receipt times; current downloaded_at only dates acquisition. FastF1 postprocesses lap-end clocks. A60s lag is a declared research availability assumption, not proven receipt latency; zero-lag results are sensitivity evidence.',
              'partial_update_contract': 'No partial updates occurred in this sample. For future partial updates, retain channel-specific observation ages when carrying last-known values; never backfill or treat omission as observed zero.',
              'performance_scored': False}
    (OUT / 'review.json').write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps({k: report[k] for k in ['status', 'events', 'weather_rows', 'missing_numeric_fields', 'update_interval_seconds_quantiles_0_50_99_100',
                                           'strict60s_join_weather_age_seconds_quantiles_0_50_99_100', 'strict60s_join_age_above180_seconds', 'strict60s_join_age_above300_seconds']}, indent=2))


if __name__ == '__main__': main()
