"""Restore already cached observed fields omitted by the earlier narrow export."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import fastf1
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
ORIGINAL=ROOT/'data/f1/performance_20260907/live_recent_final'
OUT=ROOT/'data/f1/frontier_20260907/recent_full_observations'
ART=ROOT/'artifacts/research/frontier_20260907/input_recovery'
EXTRA=['Sector1Time','Sector2Time','Sector3Time','SpeedI1','SpeedI2','SpeedFL','SpeedST','Position','FreshTyre']


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def merge_observed_fields(old, native):
    if not set(EXTRA).issubset(native):raise ValueError('Complete observed field schema required')
    new=pd.DataFrame(native).copy()
    for col in ['Time','LapTime','PitInTime','PitOutTime','Sector1Time','Sector2Time','Sector3Time']:
        if not pd.api.types.is_timedelta64_dtype(new[col]):raise TypeError(f'Native {col} must have declared timedelta units')
        new[col]=new[col].dt.total_seconds()
    for data in [old,new]:data['DriverNumber']=data.DriverNumber.astype(str)
    keys=['DriverNumber','LapNumber']
    check=old.merge(new[keys+['Time','LapTime']],on=keys,validate='one_to_one',suffixes=('','_cached'))
    if len(check)!=len(old) or len(check)!=len(new):raise ValueError('Recovered native lap population changed')
    for col in ['Time','LapTime']:np.testing.assert_allclose(check[col],check[col+'_cached'],rtol=0,atol=1e-9,equal_nan=True)
    enriched=old.merge(new[keys+EXTRA],on=keys,validate='one_to_one',sort=False)
    pd.testing.assert_frame_equal(enriched[old.columns],old.reset_index(drop=True))
    return enriched


def main():
    OUT.mkdir(parents=True,exist_ok=True);ART.mkdir(parents=True,exist_ok=True)
    if (ART/'manifest.json').exists():raise FileExistsError('Existing recovery is immutable')
    base=ROOT/'artifacts/research/frontier_20260907/live'
    cache=(ORIGINAL/'isolated_fastf1_cache').resolve()
    frozen={'before_recovery_utc':datetime.now(timezone.utc).isoformat(),'mode':'offline_cache_only_no_new_network',
            'model_sha256':sha(base/'fitted_models.pkl'),'selection_sha256':sha(base/'selection.json'),
            'original_transfer_sha256':sha(base/'results.json'),'runner_sha256':sha(__file__),
            'reason':'Earlier nine-feature exports omit sector/speed/position/fresh-tyre fields required by the new model. Preserve original failed result; no model or selection changes.',
            'cache_payloads':{str(p.relative_to(ROOT)):sha(p) for p in sorted(cache.rglob('*.ff1pkl'))}}
    (ART/'before_recovery.json').write_text(json.dumps(frozen,indent=2)+'\n')
    fastf1.Cache.enable_cache(str(cache));fastf1.Cache.offline_mode(True)
    schedule=fastf1.get_event_schedule(2026,include_testing=False,backend='f1timing')
    records=[]
    for path in sorted(ORIGINAL.glob('*_race_laps.csv')):
        rnd=int(path.name.split('_')[2]);session=schedule.get_event_by_round(rnd).get_session('Race')
        session.load(laps=True,telemetry=False,weather=False,messages=False)
        original=pd.read_csv(path,dtype={'DriverNumber':str});enriched=merge_observed_fields(original,session.laps)
        out=OUT/path.name
        if out.exists():raise FileExistsError(out)
        enriched.to_csv(out,index=False)
        record={'event_key':202600+rnd,'source_path':str(path.relative_to(ROOT)),'source_sha256':sha(path),'path':str(out.relative_to(ROOT)),
                'sha256':sha(out),'rows':len(enriched),'baseline_columns_identical':True,'added_columns':EXTRA,
                'nonmissing_counts':{c:int(enriched[c].notna().sum()) for c in EXTRA}}
        records.append(record);print(json.dumps(record),flush=True)
    assert frozen['model_sha256']==sha(base/'fitted_models.pkl') and frozen['selection_sha256']==sha(base/'selection.json')
    for p,h in frozen['cache_payloads'].items():assert sha(ROOT/p)==h,p
    (ART/'manifest.json').write_text(json.dumps({'freeze':frozen,'events':records,'model_unchanged':True,'cached_payloads_unchanged':True},indent=2)+'\n')


if __name__=='__main__':main()
