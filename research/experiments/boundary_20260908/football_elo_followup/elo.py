"""Fixed cumulative odds/result Elo using the frozen delayed quote contract.

The odds branch delegates unchanged. Result Elo differs only in the admitted
observed score and K=14; its ordered-logit readout belongs to the runner.
Suggested commit: research(football): add causal matched result-Elo reference
"""
from __future__ import annotations

from datetime import timedelta
import math

from research.experiments.boundary_20260908.football_past_market import data as parent

KINDS = ('odds', 'result')
RESULT_K = 14.0


def _kind(kind):
    if not isinstance(kind, str) or kind not in KINDS:
        raise ValueError('Only the frozen odds or result Elo processes are allowed')
    return kind


class EloCursor(parent.EloCursor):
    """Nondecreasing UTC queries; quote-valid equal-clock updates are atomic.

    History accumulates without expiration. The caller applies the separate
    rolling support mask and exact incumbent fallback. Merely querying an
    unknown team never adds that team to the observed rating state.
    """

    def __init__(self, archive, extra_days, kind='odds'):
        self.kind = _kind(kind)
        super().__init__(archive, extra_days)

    def query(self, fixtures, cutoff):
        if self.kind == 'odds':
            # Do not wrap, rename, add provenance, or recompute this output.
            return super().query(fixtures, cutoff)
        cutoff = parent.utc(cutoff)
        if self.last_cutoff is not None and cutoff < self.last_cutoff:
            raise ValueError('Elo queries cannot move backward')
        self.last_cutoff = cutoff
        while self.index < len(self.events) and self.events[self.index][0] < cutoff:
            clock = self.events[self.index][0]
            end = self.index
            while end < len(self.events) and self.events[end][0] == clock:
                end += 1
            changes, accepted, missing = {}, [], []
            # Interpret and validate the entire batch before applying any delta.
            for _, _, raw in self.events[self.index:end]:
                a0 = parent.midnight(raw.day + timedelta(days=1), raw.league)
                if a0 >= cutoff:
                    raise ValueError('Result availability must strictly precede the query cutoff')
                quote = parent._quote(self.archive, raw)
                if quote is None:
                    missing.append(raw.match_id)
                    continue
                outcome, goals = parent._outcome(raw)
                observed = (1.0, 0.5, 0.0)[outcome]
                home, away = (raw.league, raw.home), (raw.league, raw.away)
                rh, ra = self.ratings.get(home, 1000.0), self.ratings.get(away, 1000.0)
                z = math.log(10.0) * (rh - ra + 80.0) / 400.0
                expected = 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))
                delta = RESULT_K * (observed - expected)
                changes.setdefault(home, []).append(delta)
                changes.setdefault(away, []).append(-delta)
                accepted.append({
                    'match_id': raw.match_id, 'source_sha256': raw.source_sha256,
                    'source_row_hash': raw.source_row_hash, 'q_normalized': quote['q_normalized'],
                    'rating_home_before': rh, 'rating_away_before': ra, 'delta': delta,
                    'outcome': outcome, 'goals': goals, 'observed_score': observed,
                    'a0_utc': a0.isoformat(),
                })
            for team, deltas in changes.items():
                self.ratings[team] = self.ratings.get(team, 1000.0) + math.fsum(deltas)
                self.last_updates[team] = clock.isoformat()
            self.index = end
            self.missing.extend(missing)
            self.updates.append({'quote_available_at_utc': clock.isoformat(), 'rows': accepted})
            self._updates_digest = None
        if self._updates_digest is None:
            self._updates_digest = parent.digest(self.updates)
        x, queried = [], []
        for fixture in fixtures:
            league = fixture['league']
            if league not in parent.LEAGUES:
                raise ValueError('Unknown league')
            home, away = (league, fixture['home']), (league, fixture['away'])
            rh, ra = self.ratings.get(home, 1000.0), self.ratings.get(away, 1000.0)
            x.append((rh - ra) / 400.0)
            queried.append({
                'match_id': fixture['match_id'], 'rating_home': rh, 'rating_away': ra,
                'home_last_update_utc': self.last_updates.get(home),
                'away_last_update_utc': self.last_updates.get(away),
            })
        return {'x': x, 'provenance': {
            'cutoff_utc': cutoff.isoformat(), 'extra_days': self.extra_days,
            'processed_raw_rows': self.index, 'update_batches': len(self.updates),
            'updates_sha256': self._updates_digest, 'missing_quote_ids': list(self.missing),
            'fixtures': queried, 'kind': 'result', 'K': RESULT_K,
        }}


def calibration_rows(archive, extra_days, cutoff, kind='odds'):
    """2019–2021 resolved labels, with inputs captured at original issuance.

    The current calibration fixture's quote and outcome may be read only after
    it has passed admission at the later calibration cutoff. Those values do
    not enter its earlier explanatory variable. Both processes share exactly
    the same admitted quote-valid row identities.
    """
    if _kind(kind) == 'odds':
        return parent.calibration_rows(archive, extra_days, cutoff)
    cutoff, extra_days = parent.utc(cutoff), parent._delay(extra_days)
    cursor = EloCursor(archive, extra_days, kind='result')
    candidates = sorted((r for r in archive.rows if r.season in parent.CALIBRATION_SEASONS
                         and parent.availability(r.day, r.league, extra_days) < cutoff
                         and parent.midnight(r.day + timedelta(days=1), r.league) < cutoff),
                        key=lambda r: (parent.midnight(r.day, r.league), r.match_id))
    x, y, records, skipped = [], [], [], []
    for raw in candidates:
        snapshot = cursor.query([parent._metadata(raw)], parent.midnight(raw.day, raw.league))
        quote = parent._quote(archive, raw)
        if quote is None:
            skipped.append(raw.match_id)
            continue
        outcome, _ = parent._outcome(raw)
        value = snapshot['x'][0]
        x.append(value)
        y.append(outcome)
        records.append({**parent._metadata(raw), 'x': value, 'y': outcome,
                        'quote_available_at_utc': parent.availability(raw.day, raw.league, extra_days).isoformat(),
                        'rating_provenance': snapshot['provenance']})
    return x, y, {'extra_days': extra_days, 'calibration_cutoff_utc': cutoff.isoformat(),
                 'season_start_years': list(parent.CALIBRATION_SEASONS), 'rows': records,
                 'excluded_missing_quote_ids': skipped, 'elo_updates': cursor.updates,
                 'kind': 'result', 'K': RESULT_K}
