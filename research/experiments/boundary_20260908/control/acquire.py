"""Acquire only the44 discovery global track-status streams; preserve raw changes."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd
from fastf1 import _api

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
DATA = ROOT / 'data/f1/boundary_20260908/control'
OUT = ROOT / 'artifacts/research/boundary_20260908/control'
WEATHER_SOURCE = HERE.parent / 'weather/acquire.py'
WEATHER_MANIFEST = OUT.parent / 'weather/input_manifest.json'
module_spec = importlib.util.spec_from_file_location('audited_weather_fetch', WEATHER_SOURCE)
weather = importlib.util.module_from_spec(module_spec); module_spec.loader.exec_module(weather)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path, obj): Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + '\n')


def parse(raw):
    records = []
    for sequence, line in enumerate(raw.decode('utf-8-sig').splitlines()):
        if not line.strip(): continue
        time = pd.to_timedelta(line[:12]).total_seconds(); payload = json.loads(line[12:])
        if not np.isfinite(time) or not isinstance(payload, dict): raise ValueError('invalid track-status record')
        status = payload.get('Status')
        records.append({'Time': float(time), 'sequence': sequence,
                        'Status': '' if status is None else str(status),
                        'Message': '' if payload.get('Message') is None else str(payload['Message']),
                        'status_present': 'Status' in payload, 'message_present': 'Message' in payload})
    frame = pd.DataFrame(records)
    if frame.empty or not frame.Time.is_monotonic_increasing: raise ValueError('status stream must be nonempty and chronological')
    return frame


def one(event):
    key = event['event_key']; assert key // 100 in [2022, 2023]
    path = DATA / f'{key}_TrackStatus.jsonStream'; csv = DATA / f'{key}_track_status.csv'
    page = _api.pages['track_status']; assert page == 'TrackStatus.jsonStream'
    upstream = _api.base_url + '/static/' + event['session_path'] + page
    mirror = _api.base_url_mirror + '/static/' + event['session_path'] + page
    try:
        raw, status, actual = weather.fetch(upstream, path, mirror)
        frame = parse(raw)
        if csv.exists():
            prior = pd.read_csv(csv, dtype={'Status': str, 'Message': str}, keep_default_na=False)
            pd.testing.assert_frame_equal(prior, frame, check_exact=False, atol=1e-12, rtol=0)
        else: frame.to_csv(csv, index=False)
        receipt = path.with_name(path.name + '.source.json')
        print('control', key, len(frame), frame.Status.value_counts().to_dict(), flush=True)
        return {'event_key': key, 'meeting_name': event['meeting_name'], 'meeting_key': event['meeting_key'],
                'session_key': event['session_key'], 'session_path': event['session_path'], 'status': status,
                'actual_download_url': actual, 'raw_path': str(path.relative_to(ROOT)), 'raw_sha256': sha(path),
                'csv_path': str(csv.relative_to(ROOT)), 'csv_sha256': sha(csv),
                'source_receipt_path': str(receipt.relative_to(ROOT)), 'source_receipt_sha256': sha(receipt),
                'rows': len(frame), 'status_counts': {str(k): int(v) for k, v in frame.Status.value_counts().items()},
                'unknown_or_absent_status_rows': int((~frame.Status.isin(['1', '2', '4', '5', '6', '7'])).sum()),
                'partial_status_rows': int((~frame.status_present).sum()),
                'same_timestamp_adjacent_changes': int(frame.Time.duplicated().sum()),
                'first_time': float(frame.Time.iloc[0]), 'last_time': float(frame.Time.iloc[-1])}
    except Exception as exc:
        print('control-failed', key, repr(exc), flush=True)
        return {'event_key': key, 'status': 'failed', 'error': repr(exc), 'session_path': event['session_path']}


def main():
    OUT.mkdir(parents=True, exist_ok=True); DATA.mkdir(parents=True, exist_ok=True)
    if (OUT / 'input_manifest.json').exists(): raise FileExistsError('immutable acquisition already exists')
    parent = json.loads(WEATHER_MANIFEST.read_text()); assert len(parent['events']) == 44 and not parent['failed_events']
    attempt = {'started_at': datetime.now(timezone.utc).isoformat(), 'source_sha256': sha(__file__),
               'audited_fetch_source_sha256': sha(WEATHER_SOURCE), 'weather_manifest_sha256': sha(WEATHER_MANIFEST),
               'scope': 'Only44discovery2022-2023Race TrackStatus streams; no model fitting or error scoring',
               'time_contract': 'Recorded session clock at12-character stream prefix; no UTC/start offset; ordered repeated timestamps retained. Historical receipt latency unknown.',
               'unknown_contract': 'Unknown status remains unknown; missing Status is distinguished from an explicit unknown update. No future backfill.',
               'events': [int(e['event_key']) for e in parent['events']]}
    write(OUT / 'acquisition_attempt.json', attempt)
    with ThreadPoolExecutor(max_workers=2) as pool: results = list(pool.map(one, parent['events']))
    assert sha(__file__) == attempt['source_sha256'] and sha(WEATHER_SOURCE) == attempt['audited_fetch_source_sha256']
    write(OUT / 'input_manifest.json', {**attempt, 'finished_at': datetime.now(timezone.utc).isoformat(),
                                       'events': results, 'failed_events': [e['event_key'] for e in results if e['status'] == 'failed']})
    print(json.dumps({'events': len(results), 'failed_events': [e['event_key'] for e in results if e['status'] == 'failed']}), flush=True)


if __name__ == '__main__': main()
