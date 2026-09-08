"""Independent gap-history numerics over the frozen pilot packet semantics.

Suggested commit: research(f1-live): independently replay rival-gap prediction evidence
"""
from decimal import Decimal, localcontext
import math

from research.experiments.boundary_20260908.rival_gap_pilot_v2.pilot import GapCursor

FIELDS = ('Position', 'GapToLeader', 'IntervalToPositionAhead')
LEADERS = {'leader', 'leader_lap_counter'}
VALID_GAPS = {'seconds', 'leader', 'leader_lap_counter', 'lap_deficit'}
FLAGS = ('unknown_or_invalid_rank', 'duplicate_reported_rank',
         'interval_predates_last_rank_change', 'leader_rank_category_inconsistent',
         'latest_field_timestamp_regressed', 'source_parse_gap_or_unknown_clock',
         'missing_clear_invalid_or_stale_gap')


def limited(value, lo, hi):
    return float(max(Decimal(str(lo)), min(Decimal(str(hi)), value)))


def history_content(history, current, rank, cutoff_ns):
    """Decimal OLS and physical-time lag matching; no feature-builder imports."""
    h = [item for item in history if cutoff_ns-30_000_000_000 <= item[1]*1_000_000 < cutoff_ns]
    values = [math.nan]*4
    available = [False]*3
    valid = (bool(h) and rank is not None and rank > 1
             and current.get('category') == 'seconds'
             and not current.get('source_regressed', False)
             and current.get('age_seconds') is not None and current['age_seconds'] <= 10
             and (current.get('sequence'), current.get('available_ms'), current.get('value')) == h[-1])
    if valid:
        with localcontext() as context:
            context.prec = 60
            for j, delay in enumerate((5000, 15000)):
                predecessors = [item for item in h if item[1] <= h[-1][1]-delay]
                if predecessors:
                    values[j] = limited(Decimal(str(h[-1][2]))-Decimal(str(predecessors[-1][2])), -30, 30)
                    available[j] = True
            if len(h) >= 3 and h[-1][1]-h[0][1] >= 15000:
                x = [Decimal(item[1]-h[0][1])/1000 for item in h]
                y = [Decimal(str(item[2])) for item in h]
                xm, ym = sum(x)/len(x), sum(y)/len(y)
                slope = sum((a-xm)*(b-ym) for a,b in zip(x,y))/sum((a-xm)**2 for a in x)
                values[2] = limited(slope, -2, 2)
                available[2] = True
            values[3] = limited(max(Decimal(str(t[2])) for t in h)-min(Decimal(str(t[2])) for t in h), 0, 60)
    span = (h[-1][1]-h[0][1])/1000 if h else 0.0
    return values, [*map(float, available), float(min(len(h),128)), float(min(span,30))], h


class PrefixReplay(GapCursor):
    def __init__(self, path):
        super().__init__(path)
        self.histories = {}

    def _apply(self, record):
        def rank_pair(driver):
            field = self.states.get(driver, {}).get('fields', {}).get('Position', {})
            return field.get('category'), field.get('value')
        previous = {driver: rank_pair(driver) for driver in self.states}
        super()._apply(record)
        if record['error']:
            return
        lines = record['payload'].get('Lines', {})
        if not isinstance(lines, dict):
            return
        for driver, patch in lines.items():
            if not isinstance(driver, str) or not isinstance(patch, dict):
                continue
            fields = self.states.get(driver, {}).get('fields', {})
            rank = fields.get('Position', {})
            interval = fields.get('IntervalToPositionAhead', {})
            h = self.histories.setdefault(driver, [])
            if previous.get(driver, (None,None)) != rank_pair(driver):
                h.clear()
            if interval.get('sequence') != record['sequence']:
                continue
            if (interval.get('category') != 'seconds' or record['regressed']
                    or rank.get('category') != 'rank' or rank['value'] <= 1):
                h.clear()
                continue
            item = (record['sequence'], record['available_ms'], interval['value'])
            if h and h[-1][1] == item[1]:
                h[-1] = item
            else:
                h.append(item)
            self.histories[driver] = [v for v in h if v[1] >= item[1]-30000]

    def reconstruct(self, driver, *, cutoff_ns):
        snapshot = super().query(driver, cutoff_ns=cutoff_ns)
        fields = snapshot['fields']
        position, leader, interval = [fields[name] for name in FIELDS]
        rank = position['value'] if position['category'] == 'rank' else None
        values = [math.nan]*12
        if rank is not None:
            values[0], values[4] = float(rank), float(rank == 1)
        def fresh(field):
            return ('sequence' in field and not field['source_regressed']
                    and field['age_seconds'] is not None and field['age_seconds'] <= 10)
        consistent_leader = rank == 1 and not snapshot['flags']['leader_rank_category_inconsistent']
        if fresh(leader):
            if leader['category'] == 'seconds':
                values[1], values[3] = min(float(leader['value']),180.0), 0.0
            elif leader['category'] in LEADERS and consistent_leader:
                values[1], values[3] = 0.0, 0.0
            elif leader['category'] == 'lap_deficit':
                values[3] = float(min(leader['value'],20))
        if fresh(interval) and interval['category'] == 'seconds':
            values[2] = min(float(interval['value']),60.0)
            values[5:8] = [float(interval['value'] < threshold) for threshold in (1,2,5)]
        qualities = []
        for name, field in zip(FIELDS, (position,leader,interval)):
            valid = field['category'] == 'rank' if name == 'Position' else field['category'] in VALID_GAPS
            age = field['age_seconds']
            qualities.extend((float('sequence' in field), float(valid), float(fresh(field)),
                              min(max(float(age),0.0),180.0) if age is not None else 180.0))
        for field in (leader, interval):
            category = field['category']
            qualities.extend(map(float, (category == 'seconds', category in LEADERS,
                                         category == 'lap_deficit', category not in VALID_GAPS)))
        qualities.extend(float(snapshot['flags'][key]) for key in FLAGS)
        qualities.append(float(snapshot['race_order_gap_ready']))
        values[8:12], history_quality, h = history_content(self.histories.get(driver, []), interval, rank, cutoff_ns)
        qualities.extend(history_quality)
        assert len(values) == 12 and len(qualities) == 33 and all(math.isfinite(v) for v in qualities)
        return {'values': values+qualities, 'supported': snapshot['race_order_gap_ready'],
                'cutoff_ns': cutoff_ns, 'history': h, 'snapshot': snapshot}
