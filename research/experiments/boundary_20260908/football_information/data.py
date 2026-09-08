"""Strict cached-odds joins and separate availability-filtered label access."""
from copy import deepcopy
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from . import model

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
INTERNAL = 'shot_strength_90d_ridge0.1'
INTERNAL_REFERENCES = ('production_default_dc_auto', 'dc365_elo50', 'dc_180', INTERNAL)
METADATA = ('match_id', 'league', 'season', 'home', 'away', 'day', 'forecast_cutoff_utc',
            'result_available_at', 'fit_id', 'fit_cutoff_utc')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def utc(value):
    """Old artifacts explicitly encode UTC as naive ISO; offsets are normalized."""
    if not isinstance(value, str): raise ValueError('Expected an ISO timestamp')
    result = datetime.fromisoformat(value)
    return result.replace(tzinfo=timezone.utc) if result.tzinfo is None else result.astimezone(timezone.utc)


def identity(row):
    league, year, home, away = (row[k] for k in ('league', 'season', 'home', 'away'))
    if league not in ('E0', 'SP1', 'I1') or type(year) is not int or not home or not away:
        raise ValueError('Invalid fixture metadata')
    expected = f'{league}:{year}:{home}:{away}'
    if row['match_id'] != expected: raise ValueError('Fixture identity disagrees with metadata')
    if utc(row['fit_cutoff_utc']) > utc(row['forecast_cutoff_utc']):
        raise ValueError('Base fit occurs after issuance')
    if utc(row['result_available_at']) <= utc(row['forecast_cutoff_utc']):
        raise ValueError('Result is not later than the forecast')
    return expected


def unique(rows):
    result = {}
    for row in rows:
        key = identity(row)
        if key in result: raise ValueError('Duplicate fixture identity')
        result[key] = row
    return result


def load_feature_cache(spec):
    rows = []
    for league in spec['leagues']:
        name = f'artifacts/research/boundary_20260908/football/features_selection_{league}.json'
        path = ROOT/name
        if sha(path) != spec['inputs'][name]: raise ValueError('Frozen feature cache changed')
        payload = json.loads(path.read_text())
        for input_path, digest in payload['input_hashes'].items():
            if sha(ROOT/input_path) != digest: raise ValueError('Cached feature input changed')
        if any(r['league'] != league for r in payload['rows']): raise ValueError('Mixed cache league')
        rows.extend(payload['rows'])
    unique(rows)
    return sorted(rows, key=lambda r: (utc(r['forecast_cutoff_utc']), r['match_id']))


def load_frozen_incumbents(spec, phase='selection'):
    if phase not in ('selection', 'evaluation'): raise ValueError('Unknown phase')
    name = f'artifacts/research/boundary_20260908/football/{phase}.json'
    path = ROOT/name
    if sha(path) != spec['inputs'][name]: raise ValueError('Frozen incumbent evidence changed')
    payload = json.loads(path.read_text())
    if payload['selected_model'] != INTERNAL: raise ValueError('Unexpected frozen incumbent')
    return unique(payload['predictions'])


def reuse_incumbent(row, frozen):
    """Reuse vectors verbatim; do not recalibrate, renormalize or refit them."""
    if any(row[k] != frozen[k] for k in METADATA): raise ValueError('Incumbent metadata mismatch')
    for name in INTERNAL_REFERENCES:
        if name in row['probabilities'] and row['probabilities'][name] != frozen['probabilities'][name]:
            raise ValueError('Existing incumbent vector mismatch')
    return {**row, 'probabilities': {**row['probabilities'],
            **{name: deepcopy(frozen['probabilities'][name]) for name in INTERNAL_REFERENCES}}}


def load_quotes(spec, years):
    """Read prices/fixture metadata only; do not consume result/statistic values."""
    base = ROOT/'data/football/performance_20260907'; result = {}; bindings = {}
    for manifest in ('source_manifest.json', 'transfer_source_manifest.json'):
        relative = str((base/manifest).relative_to(ROOT))
        if sha(base/manifest) != spec['inputs'][relative]: raise ValueError('Acquisition manifest changed')
        for entry in json.loads((base/manifest).read_text())['files']:
            path = base/entry['name']
            if path.suffix != '.csv': continue
            year = int(path.stem.split('_')[-2])
            if year not in years: continue
            league = 'E0' if path.name.startswith('E0_') else path.name.split('_')[1]
            if sha(path) != entry['sha256']: raise ValueError('Odds CSV changed')
            bindings[str(path.relative_to(ROOT))] = entry['sha256']
            with path.open(encoding='utf-8-sig', newline='') as stream:
                reader = csv.DictReader(stream)
                prefix = 'Avg' if 'AvgH' in reader.fieldnames else 'BbAv'
                for row in reader:
                    home, away = row['HomeTeam'].strip(), row['AwayTeam'].strip()
                    key = f'{league}:{year}:{home}:{away}'
                    if key in result: raise ValueError('Duplicate odds fixture')
                    day = None
                    for fmt in ('%d/%m/%Y', '%d/%m/%y'):
                        try: day = datetime.strptime(row['Date'], fmt).date().isoformat(); break
                        except ValueError: pass
                    if day is None: raise ValueError('Unparseable fixture date')
                    result[key] = {'match_id': key, 'day': day,
                        'avg': [row.get(prefix+c) for c in 'HDA'],
                        'b365': [row.get('B365'+c) for c in 'HDA'],
                        'fields': [prefix+c for c in 'HDA']+['B365'+c for c in 'HDA'],
                        'source_path': str(path.relative_to(ROOT)), 'source_sha256': entry['sha256'],
                        'quote_observed_at': None, 'receipt_certified': False,
                        'product': 'provider_preclosing_snapshot'}
    return result, bindings


def numeric_odds(raw):
    try:
        odds = [float(value) for value in raw]
        if len(odds) != 3 or any(not math.isfinite(x) or x <= 1 for x in odds): return None
        return odds
    except (ValueError, TypeError):
        return None


def issue_row(row, quote):
    key = identity(row)
    if quote['match_id'] != key or quote['day'] != row['day']:
        raise ValueError('Odds identity/date differs from incumbent fixture')
    vectors = {name: deepcopy(row['probabilities'][name]) for name in INTERNAL_REFERENCES}
    for vector in vectors.values(): model.probabilities([vector])
    average, bet365 = numeric_odds(quote['avg']), numeric_odds(quote['b365'])
    ready = average is not None and bet365 is not None
    powers = {}
    if ready:
        for book, odds in (('avg', average), ('b365', bet365)):
            for method in ('normalized', 'power'):
                values, power = model.devig(odds, method)
                vectors[f'{book}_{method}'] = values.tolist()
                if power is not None: powers[book] = power
        vectors['arithmetic_half'] = ((np.array(vectors[INTERNAL])+vectors['avg_power'])/2).tolist()
    else:
        for name in ('avg_normalized', 'avg_power', 'b365_normalized', 'b365_power', 'arithmetic_half'):
            vectors[name] = deepcopy(vectors[INTERNAL])
    return {**{name: deepcopy(row[name]) for name in METADATA}, 'probabilities': vectors,
            'market_ready': ready, 'fallback_reason': None if ready else 'incomplete_or_invalid_preclosing_snapshot',
            'quote': deepcopy(quote), 'power_exponents': powers,
            'labels_attached': False, 'internal_vector_reused_without_transformation': True}


def build_issuance(rows, quotes):
    unique(rows)
    result = []
    for row in rows:
        if row['match_id'] not in quotes: raise ValueError('Missing fixture in odds source')
        result.append(issue_row(row, quotes[row['match_id']]))
    return result


def label_for(row, sources):
    original = sources[row['match_id']]
    if any(row[k] != original[k] for k in METADATA): raise ValueError('Outcome identity/clock mismatch')
    label = original['label']
    if type(label) is not int or label not in (0, 1, 2): raise ValueError('Invalid H/D/A outcome label')
    return label


def training_rows(issued, sources, cutoff):
    """Filter clocks before inspecting odds, probabilities or outcome labels."""
    cutoff = utc(cutoff); lower = cutoff-timedelta(days=1460)
    accepted = []; unavailable = []; labels = []
    for row in issued:
        stamp, resolved = utc(row['forecast_cutoff_utc']), utc(row['result_available_at'])
        if not lower <= stamp < cutoff or not resolved < cutoff:
            continue
        if not row['market_ready']:
            unavailable.append(row['match_id']); continue
        label = label_for(row, sources)
        accepted.append(row);labels.append(label)
    ids = [r['match_id'] for r in accepted]
    if len(ids) != len(set(ids)): raise ValueError('Duplicate training identity')
    return accepted, labels, unavailable
