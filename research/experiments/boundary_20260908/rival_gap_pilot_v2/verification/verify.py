"""Read-only closure, population and snapshot arithmetic checks; no raw feed read."""
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import hashlib,json,math

ROOT=Path(__file__).resolve().parents[5]
OUT=ROOT/'artifacts/research/boundary_20260908/rival_gap_pilot_v2'
IDENTITY=('event_key','driver_id','issued_after_lap_number','issued_at_timestamp','issued_at_ns','issuance_id')
TARGETS={'outcome_status','target_id','target_at_ns','target_lap_number','target_timestamp','lap_time_seconds','target_same_stint','y','y_true'}


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def bound(p):return {'path':str(Path(p).relative_to(ROOT)),'sha256':sha(p),'bytes':Path(p).stat().st_size}
def jsonl(p):return [json.loads(line) for line in Path(p).read_text().splitlines() if line]
def close(a,b):assert math.isclose(a,b,rel_tol=0,abs_tol=1e-12),(a,b)


def main():
    lock=read(OUT/'design_lock.json');result=read(OUT/'result.json');checked=[]
    for source,digest in lock['source_files'].items():
        assert sha(ROOT/source)==digest;checked.append(bound(ROOT/source))
    for item in [*lock['inputs'],lock['review'],lock['specification'],result['design_lock']]:
        path=ROOT/item['path'];assert sha(path)==item['sha256']
        if 'bytes' in item:assert path.stat().st_size==item['bytes']
        checked.append(bound(path))
    review=read(ROOT/lock['review']['path'])
    assert review['approved_for_execution_lock'] is True and review['source_files']==lock['source_files']
    spec=read(ROOT/lock['specification']['path'])
    by_event={e['event_key']:e for e in spec['events']}
    assert result['rows_both_lags']==3454 and result['model_fits']==result['prediction_scores']==result['new_downloads']==0
    summary=[];identities={};entries=result['entries']
    assert [(e['event_key'],e['lag_seconds']) for e in entries]==[(202201,2),(202201,0),(202301,2),(202301,0)]
    for entry in entries:
        event=entry['event_key'];lag=entry['lag_seconds'];rows=jsonl(ROOT/entry['snapshot_file']['path'])
        assert sha(ROOT/entry['snapshot_file']['path'])==entry['snapshot_file']['sha256']
        checked.append(bound(ROOT/entry['snapshot_file']['path']))
        original=jsonl(ROOT/by_event[event]['issuances']['path'])
        assert len(rows)==len(original)==entry['rows']==by_event[event]['issuances']['rows']
        assert len({r['issuance_id'] for r in rows})==len(rows)
        identities[event,lag]=[r['issuance_id'] for r in rows]
        counts=Counter();categories=Counter();missing=Counter();leader_ready=0
        for row,old in zip(rows,original):
            assert not TARGETS.intersection(row) and not TARGETS.intersection(old)
            assert {k:row[k] for k in IDENTITY}=={k:old[k] for k in IDENTITY}
            assert row['lag_seconds']==lag and row['cutoff_ns']==row['issued_at_ns']-lag*10**9
            for name,field in row['fields'].items():
                categories[name+'/'+field['category']]+=1;missing[name]+=int(field['age_seconds'] is None)
                if field['age_seconds'] is not None:
                    delta=row['cutoff_ns']-field['available_ms']*10**6
                    assert delta>0 and field['sequence']<=row['last_consumed_packet_sequence']
                    assert field['recorded_ms']<=field['available_ms']
                    close(field['age_seconds'],delta/10**9)
                    close(field['age_seconds_at_issuance'],delta/10**9+lag)
            rank=row['fields']['Position']['value'];gap=row['fields']['GapToLeader'];interval=row['fields']['IntervalToPositionAhead']
            flags=row['flags'];leaders={'leader','leader_lap_counter'}
            blockers=any(flags[k] for k in ('unknown_or_invalid_rank','duplicate_reported_rank','leader_rank_category_inconsistent','latest_field_timestamp_regressed','source_parse_gap_or_unknown_clock'))
            for age in (2,5,10,30):
                field=gap if rank==1 else interval
                fresh=field['age_seconds'] is not None and field['age_seconds']<=age and not field.get('source_regressed',False)
                category=(field['category'] in leaders or (field['category']=='seconds' and field['value']==0)) if rank==1 else field['category']=='seconds'
                prior_ok=rank==1 or (rank is not None and rank>1 and not flags['interval_predates_last_rank_change'])
                expected=bool(fresh and category and prior_ok and not blockers)
                assert row['readiness_by_max_age_seconds'][str(age)] is expected
                counts['ready_'+str(age)+'s']+=int(expected)
            assert row['race_order_gap_ready'] is row['readiness_by_max_age_seconds']['10']
            for name,value in flags.items():counts[name]+=int(value)
            leader_ready+=int(row['leader_gap_category_ready'])
            assert row['unplaceable_timestamp_boundary'] is None
        assert dict(counts)==entry['coverage'] and dict(categories)==entry['field_categories'] and dict(missing)==entry['field_age_missing']
        summary.append({'event_key':event,'lag_seconds':lag,'rows':len(rows),'ready_10s':counts['ready_10s'],
            'ready_10s_percent':100*counts['ready_10s']/len(rows),'coverage':dict(counts),
            'leader_gap_category_ready':leader_ready,'clock_identity_count_checks_passed':True})
    for event in by_event:assert identities[event,2]==identities[event,0]
    for event in by_event:
        parts=[e for e in entries if e['event_key']==event]
        assert parts[0]['full_stream_inventory']==parts[1]['full_stream_inventory']
    receipt={'status':'passed','checked_at_utc':datetime.now(timezone.utc).isoformat(),
        'method':'Independent JSONL identity, hash, strict-clock, age arithmetic, readiness and aggregation checks. Raw streams were hashed only, never parsed by this verifier.',
        'source':bound(Path(__file__)),'selection_or_model_evaluation':False,'raw_feed_payload_reads':0,
        'result':bound(OUT/'result.json'),'design_lock':bound(OUT/'design_lock.json'),
        'source_and_input_bindings_checked':checked,'rows_checked':3454,'unique_original_issuances':1727,'summaries':summary}
    with (OUT/'verification.json').open('x') as f:json.dump(receipt,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps({'status':'passed','rows_checked':3454,'verification':bound(OUT/'verification.json'),'summaries':summary},indent=2))


if __name__=='__main__':main()
