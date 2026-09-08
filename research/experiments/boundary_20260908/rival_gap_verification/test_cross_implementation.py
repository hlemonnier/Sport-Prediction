"""Separate arithmetic and feature builders on identical synthetic raw streams."""
import json
import random

import numpy as np

from research.experiments.boundary_20260908.rival_gap_forecast.features import GapFeatureCursor
from .prefix import PrefixReplay


def test_mixed_prefixes_match_independent_decimal_implementation(tmp_path):
    rng = random.Random(20260908)
    queries_checked = 0
    for trial in range(12):
        path = tmp_path/f'{trial}.stream'
        packets = []
        for driver in ('2','3','4'):
            packets.append((0,driver,{'Position':driver,'GapToLeader':'5','IntervalToPositionAhead':{'Value':'2'}}))
        clock = 0
        for i in range(100):
            clock += rng.choice((0,100,1000,2500,5000))
            stamp = max(0,clock-rng.choice((0,0,0,1000)))
            driver = rng.choice(('2','3','4'))
            value = rng.choice((None,'','0','0.2','1','2.5','14','1e307','LAP 4','+1 LAP',True))
            patch = {'IntervalToPositionAhead':{'Value':value}}
            if rng.random() < .15:
                patch = {'IntervalToPositionAhead':{'Catching':True}}
            if rng.random() < .2:
                patch['Position'] = rng.choice(('1','2','3','4','',None))
            if rng.random() < .3:
                patch['GapToLeader'] = rng.choice(('0','3','120','LAP 7','+1 LAP',''))
            packets.append((stamp,driver,patch))
        data = bytearray(b'\xef\xbb\xbf')
        for stamp,driver,patch in packets:
            hours,remainder = divmod(stamp,3600000)
            minutes,remainder = divmod(remainder,60000)
            seconds,millis = divmod(remainder,1000)
            header = f'{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}'
            data.extend((header+json.dumps({'Lines':{driver:patch}})+'\n').encode())
        path.write_bytes(data)
        actual,independent = GapFeatureCursor(path),PrefixReplay(path)
        try:
            cutoffs = sorted({rng.randrange(max(clock,1))*1_000_000+rng.choice((0,1)) for _ in range(30)})
            for cutoff in cutoffs:
                for driver in ('2','3','4'):
                    a = actual.query(driver,cutoff_ns=cutoff)
                    b = independent.reconstruct(driver,cutoff_ns=cutoff)
                    np.testing.assert_allclose(a.gap_values,b['values'],rtol=1e-9,atol=1e-9,equal_nan=True,
                                               err_msg=f'trial={trial} cutoff={cutoff} driver={driver}')
                    assert a.gap_supported is b['supported']
                    observed = [(r['sequence'],r['available_ns']//1_000_000,r['seconds'])
                                for r in a.provenance['retained_interval_observations']]
                    assert observed == b['history']
                    queries_checked += 1
        finally:
            actual.close()
            independent.close()
    assert queries_checked == 1080
