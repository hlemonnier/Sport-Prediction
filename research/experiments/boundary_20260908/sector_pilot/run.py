"""Bounded raw sector feasibility: archive prefixes first, outcomes only after hash lock."""
from bisect import bisect_right
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
from pathlib import Path

from .ledger import build

ROOT=Path(__file__).resolve().parents[4]
SOURCE=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/research/boundary_20260908/sector_pilot'
ACQUISITION=OUT/'network_retry/acquisition.json'


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def now():return datetime.now(timezone.utc).isoformat()
def relative(path):return str(Path(path).resolve().relative_to(ROOT))
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
def read(path):return json.loads(Path(path).read_text())
def rows(path):
    with Path(path).open() as f:
        for line in f:yield json.loads(line)


def input_paths(event,acquisition):
    paths={r['stream']:ROOT/r['decoded_body_path'] for r in acquisition['streams'] if r['event_key']==event}
    paths['TrackStatus']=ROOT/f'data/f1/boundary_20260908/control/{event}_TrackStatus.jsonStream'
    return paths


def check_inputs():
    design=read(OUT/'design_lock.json');spec=read(SOURCE/'specification.json');acquisition=read(ACQUISITION)
    assert sha(SOURCE/'specification.json')==design['spec_sha256']==acquisition['spec_sha256']
    checks={**design['input_files'],**design['installed_primary_source_files']}
    checks[relative(SOURCE/'acquire.py')]=acquisition['source_sha256']
    for stream in acquisition['streams']:
        assert stream['status']=='downloaded'
        checks[stream['decoded_body_path']]=stream['decoded_body_sha256']
        for attempt in stream['attempts']:
            if 'wire_body_path' in attempt:checks[attempt['wire_body_path']]=attempt['wire_body_sha256']
    for path,expected in checks.items():assert sha(ROOT/path)==expected,path
    return spec,acquisition,checks


def numeric(value):
    try:return float(value)
    except (ValueError,TypeError):return float('nan')


def eligible(row):
    y=numeric(row.get('LapTime'))
    return (math.isfinite(y) and y>0 and str(row.get('IsAccurate','')).lower() in ['true','1']
            and not math.isfinite(numeric(row.get('PitInTime'))) and not math.isfinite(numeric(row.get('PitOutTime')))
            and not any(c in str(row.get('TrackStatus','')) for c in '4567'))


def target_index(csv_path):
    by_driver=defaultdict(list);stats=Counter()
    with Path(csv_path).open() as f:
        for ordinal,row in enumerate(csv.DictReader(f)):
            stats['canonical_completed_rows']+=1
            if not eligible(row):continue
            stamp=numeric(row.get('Time'));number=numeric(row.get('DriverNumber'))
            if not math.isfinite(stamp) or not math.isfinite(number):stats['eligible_missing_key_or_time']+=1;continue
            stats['recorded_eligible_rows']+=1
            by_driver[str(int(number))].append({'csv_row':ordinal,'recorded_time_seconds':stamp,'lap_number':numeric(row['LapNumber']),'lap_time_seconds':numeric(row['LapTime'])})
    for values in by_driver.values():values.sort(key=lambda r:(r['recorded_time_seconds'],r['csv_row']))
    return dict(by_driver),dict(stats)


def resolve_target(row,index,terminal):
    values=index.get(row['driver'],[]);time=row['checkpoint_ms']/1000
    pos=bisect_right([r['recorded_time_seconds'] for r in values],time)
    target=values[pos] if pos<len(values) else None
    return {'event_key':row['event_key'],'ledger_id':row['ledger_id'],'candidate_checkpoint':row['candidate_checkpoint'],
            'status':'matched_recorded_future_eligible_lap' if target else 'unmatched_in_terminal_recorded_archive' if terminal else 'unresolved_in_incomplete_archive',
            'target':target,'interpretation':'Separate completed-CSV diagnostic; neither target availability at issuance nor canonical current-lap identity is established.'}


def main():
    spec,acquisition,inputs=check_inputs()
    assert not (OUT/'results.json').exists(),'Immutable completed run; use a separate execution directory for another run.'
    execution={'created_at_utc':now(),'spec_sha256':sha(SOURCE/'specification.json'),'design_lock_sha256':sha(OUT/'design_lock.json'),
               'acquisition_sha256':sha(ACQUISITION),'source_files':{relative(p):sha(p) for p in sorted(SOURCE.glob('*.py'))},
               'verified_input_files':inputs,'raw_outcome_diagnostics_started':False,'models_fitted':0}
    write(OUT/'execution_lock.json',execution)
    summaries={};locked={}
    for session in spec['sessions']:
        event=session['event_key'];path=OUT/f'{event}_ledger.jsonl'
        with path.open('x') as f:
            summaries[str(event)]=build(input_paths(event,acquisition),event,lambda row:f.write(json.dumps(row,sort_keys=True,allow_nan=False,separators=(',',':'))+'\n'))
        locked[relative(path)]={'sha256':sha(path),'bytes':path.stat().st_size,'rows':summaries[str(event)]['ledger_rows']}
    # This lock is persisted before any canonical CSV is parsed for diagnostics.
    write(OUT/'ledger_lock.json',{'created_at_utc':now(),'execution_lock_sha256':sha(OUT/'execution_lock.json'),'ledgers':locked,'canonical_outcome_diagnostics_started':False})
    results={'created_at_utc':now(),'status':'causal_archive_feasibility_only_no_model_fitting_or_scores','sessions':{},
             'ledger_lock_sha256':sha(OUT/'ledger_lock.json'),'execution_lock_sha256':sha(OUT/'execution_lock.json')}
    for session in spec['sessions']:
        event=session['event_key'];stats=summaries[str(event)];ledger_path=OUT/f'{event}_ledger.jsonl';target_path=OUT/f'{event}_outcome_diagnostics.jsonl'
        index,csv_stats=target_index(ROOT/session['original_laps_path'])
        terminal=stats['auxiliary']['SessionStatus']['final_status'] in ['Finished','Finalised','Ends']
        outcomes=Counter();candidate_outcomes=Counter();candidate_sectors=Counter();all_sectors=Counter();exclusions=Counter();candidate_fields=Counter();field_names=set();ambiguous_rows=0;candidate_first=None;candidate_last=None
        with target_path.open('x') as f:
            for row in rows(ledger_path):
                result=resolve_target(row,index,terminal);f.write(json.dumps(result,sort_keys=True,allow_nan=False,separators=(',',':'))+'\n')
                outcomes[result['status']]+=1;all_sectors[str(row['sector'])]+=1;exclusions.update(row['exclusion_reasons']);ambiguous_rows+=bool(row['ambiguity_reasons'])
                field_names.update(row['raw_field_presence'])
                if row['candidate_checkpoint']:
                    candidate_outcomes[result['status']]+=1;candidate_sectors[str(row['sector'])]+=1
                    candidate_fields.update(k for k,v in row['raw_field_presence'].items() if v)
                    candidate_first=row['checkpoint_ms'] if candidate_first is None else min(candidate_first,row['checkpoint_ms']);candidate_last=row['checkpoint_ms'] if candidate_last is None else max(candidate_last,row['checkpoint_ms'])
        assert sha(ledger_path)==locked[relative(ledger_path)]['sha256']
        candidate_count=sum(candidate_sectors.values())
        results['sessions'][str(event)]={**stats,'sector_rows':dict(all_sectors),'candidate_sector_rows':dict(candidate_sectors),
            'rows_with_any_epoch_ambiguity':ambiguous_rows,'exclusion_reason_rows':dict(exclusions),
            'candidate_raw_field_present_rows':{k:candidate_fields[k] for k in sorted(field_names)},'candidate_raw_field_missing_rows':{k:candidate_count-candidate_fields[k] for k in sorted(field_names)},
            'candidate_first_clock_ms':candidate_first,'candidate_last_clock_ms':candidate_last,
            'canonical_hgb_exact_reconstruction_certified_rows':0,'canonical_hgb_unresolved_candidate_rows':candidate_count,
            'completed_csv_diagnostics':{**csv_stats,'all_ledger_outcomes':dict(outcomes),'candidate_outcomes':dict(candidate_outcomes),'terminal_status_observed':terminal,
               'csv_path':session['original_laps_path'],'csv_sha256':session['original_laps_sha256'],'output':relative(target_path),'output_sha256':sha(target_path)}}
    write(OUT/'results.json',results)
    print(json.dumps({'results':relative(OUT/'results.json'),'sha256':sha(OUT/'results.json'),'sessions':{k:{'rows':v['ledger_rows'],'candidate_rows':v['timing']['candidate_checkpoints']} for k,v in results['sessions'].items()}}))


if __name__=='__main__':main()
