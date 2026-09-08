"""Independent raw coverage, prefix invariance, availability and outcome checks."""
from collections import Counter
import csv
import json
import math
from pathlib import Path
import tempfile

from . import ledger
from .run import ROOT,SOURCE,OUT,ACQUISITION,sha,read,rows,write,now,input_paths,check_inputs,relative


def positive(raw):
    try:
        parts=str(raw).split(':');value=0.
        for part in parts:value=60*value+float(part)
        return value if math.isfinite(value) and value>0 else None
    except (ValueError,TypeError):return None


def independent_updates(path):
    values=[]
    with Path(path).open(encoding='utf-8-sig') as f:
        for seq,line in enumerate(f):
            if not line.strip():continue
            try:body=json.loads(line[12:])
            except json.JSONDecodeError:continue
            patches=body.get('Lines',{})
            for driver,patch in patches.items():
                sectors=patch.get('Sectors',{})
                sectors=enumerate(sectors) if isinstance(sectors,list) else sectors.items()
                for key,update in sectors:
                    if str(key) not in ['0','1'] or not isinstance(update,dict):continue
                    value=positive(update.get('Value'))
                    if value is not None:values.append((seq,str(driver),int(key)+1,value))
    return Counter(values)


def verify_outcomes(session,issued,diagnostics):
    # Separate direct scalar predicate and brute-force first-later matching; no
    # production or experiment target-index helper is called.
    def number(x):
        try:return float(x)
        except (ValueError,TypeError):return float('nan')
    with (ROOT/session['original_laps_path']).open() as f:completed=list(csv.DictReader(f))
    eligible=[]
    for ordinal,r in enumerate(completed):
        y=number(r['LapTime']);stamp=number(r['Time'])
        if not (math.isfinite(y) and y>0 and math.isfinite(stamp) and r['IsAccurate'].lower() in ['true','1']):continue
        if any(math.isfinite(number(r[k])) for k in ['PitInTime','PitOutTime']):continue
        if set(r['TrackStatus']) & set('4567'):continue
        eligible.append((str(int(number(r['DriverNumber']))),stamp,ordinal,number(r['LapNumber']),y))
    assert len(issued)==len(diagnostics)
    for row,diagnostic in zip(issued,diagnostics):
        assert (row['event_key'],row['ledger_id'])==(diagnostic['event_key'],diagnostic['ledger_id'])
        later=[r for r in eligible if r[0]==row['driver'] and r[1]>row['checkpoint_ms']/1000]
        if not later:assert diagnostic['target'] is None;continue
        selected=min(later,key=lambda r:(r[1],r[2]));target=diagnostic['target']
        assert target=={'recorded_time_seconds':selected[1],'csv_row':selected[2],'lap_number':selected[3],'lap_time_seconds':selected[4]}
    return len(issued)


def main():
    spec,acquisition,inputs=check_inputs();execution=read(OUT/'execution_lock.json');locked=read(OUT/'ledger_lock.json');results=read(OUT/'results.json')
    assert sha(OUT/'execution_lock.json')==locked['execution_lock_sha256']==results['execution_lock_sha256']
    assert sha(OUT/'ledger_lock.json')==results['ledger_lock_sha256']
    for path,expected in execution['source_files'].items():assert sha(ROOT/path)==expected,path
    verified={}
    for session in spec['sessions']:
        event=session['event_key'];paths=input_paths(event,acquisition);path=OUT/f'{event}_ledger.jsonl';issued=list(rows(path));stats=results['sessions'][str(event)]
        assert sha(path)==locked['ledgers'][relative(path)]['sha256']
        assert len(issued)==locked['ledgers'][relative(path)]['rows']==stats['ledger_rows']
        independently_counted=independent_updates(paths['TimingData'])
        assert independently_counted==Counter((r['packet_sequence'],r['driver'],r['sector'],r['sector_seconds']) for r in issued)
        for i,r in enumerate(issued):
            assert r['ledger_id']==i and r['checkpoint_ms']>=r['recorded_ms']
            assert r['candidate_checkpoint']==(not r['exclusion_reasons'])
            assert not r['canonical_hgb_exactly_reconstructible'] and not r['historical_client_receipt_available']
            assert all(v['available_ms']<=r['checkpoint_ms'] and v['sequence']<=r['packet_sequence'] for v in r['received_driver_fields'].values())
            assert all(v['available_ms']<r['checkpoint_ms'] for v in r['received_tyre_candidate']['fields'].values())
            for key in ['session_status_available_ms','track_status_available_ms']:
                assert r[key] is None or r[key]<r['checkpoint_ms']
            if r['candidate_checkpoint']:assert r['update_kind']=='first' and r['session_status']=='Started' and r['received_driver_fields']['InPit']['value'] is False and not r['ambiguity_reasons']
        with paths['TimingData'].open(encoding='utf-8-sig') as f:raw=f.readlines()
        prefix_checks=[]
        for cut in [len(raw)//3,2*len(raw)//3,len(raw)-1]:
            # Auxiliary files are not arbitrarily truncated: full files provide
            # the frozen cross-stream timestamp proxy; only records strictly
            # before the checkpoint may enter state. This is not a watermark or
            # historical network-receipt guarantee.
            with tempfile.TemporaryDirectory(prefix='sector-prefix-') as directory:
                truncated=Path(directory)/'TimingData.jsonStream';truncated.write_text(''.join(raw[:cut]));subset={**paths,'TimingData':truncated}
                replay=[];ledger.build(subset,event,replay.append)
                expected=[r for r in issued if r['packet_sequence']<cut]
                assert replay==expected
                prefix_checks.append({'physical_lines':cut,'immutable_rows':len(replay)})
        outcome_path=OUT/f'{event}_outcome_diagnostics.jsonl'
        assert sha(outcome_path)==stats['completed_csv_diagnostics']['output_sha256']
        outcome_rows=verify_outcomes(session,issued,list(rows(outcome_path)))
        verified[str(event)]={'all_positive_s1_s2_rows':sum(independently_counted.values()),'prefix_checks':prefix_checks,'source_time_checked_rows':len(issued),'independent_outcome_matches_checked_rows':outcome_rows}
    result={'verified_at_utc':now(),'status':'passed','errors':[],'verified_input_hash_count':len(inputs),'results_sha256':sha(OUT/'results.json'),'execution_lock_sha256':sha(OUT/'execution_lock.json'),'ledger_lock_sha256':sha(OUT/'ledger_lock.json'),'sessions':verified,'models_fitted_or_scored':0}
    write(OUT/'verification.json',result);print(json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
