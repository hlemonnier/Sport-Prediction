"""Target-free race-order gap feasibility; no model or label operations.

Suggested commit: research(f1-live): prepare causal rival-gap feasibility pilot
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal, DecimalException
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import traceback

ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/research/boundary_20260908/rival_gap_pilot'
FIELDS=('Position','GapToLeader','IntervalToPositionAhead')
GAPS=FIELDS[1:]
THRESHOLDS=(2,5,10,30)
MAX_PAYLOAD_BYTES=16*1024*1024
HEADER=re.compile(rb'[0-9]{2}:[0-5][0-9]:[0-5][0-9]\.[0-9]{3}')
TARGETS={'outcome_status','target_id','target_at_ns','target_lap_number','target_timestamp','lap_time_seconds','target_same_stint','y','y_true'}
IDENTITY=('event_key','driver_id','issued_after_lap_number','issued_at_timestamp','issued_at_ns','issuance_id')


def now():return datetime.now(timezone.utc).isoformat()
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(path):return json.loads(Path(path).read_text(),parse_constant=lambda v:(_ for _ in ()).throw(ValueError('Nonfinite JSON '+v)))
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def binding(path):return {'path':str(Path(path).resolve()),'sha256':sha(path),'bytes':Path(path).stat().st_size}
def check(item):
    path=ROOT/item['path']
    if sha(path)!=item['sha256'] or ('bytes' in item and path.stat().st_size!=item['bytes']):raise ValueError('Changed binding: '+str(path))
    return path
def raw_type(value):
    return ('null' if value is None else 'boolean' if isinstance(value,bool) else 'object' if isinstance(value,dict)
            else 'array' if isinstance(value,list) else 'string' if isinstance(value,str) else 'number' if isinstance(value,(int,float)) else 'other')


def parse_value(value,*,position=False):
    """Presence-sensitive parsing. No implicit seconds for categorical gaps."""
    outer=raw_type(value)
    if isinstance(value,dict):
        if 'Value' not in value:return {'category':'metadata_only','outer_type':outer,'leaf_type':None,'updates':False,'value':None}
        value=value['Value']
    leaf=raw_type(value)
    result={'category':'invalid','outer_type':outer,'leaf_type':leaf,'updates':True,'value':None}
    if value is None:return {**result,'category':'clear_null'}
    if isinstance(value,bool) or not isinstance(value,(str,int,float)):return result
    text=str(value).strip()
    if not text:return {**result,'category':'clear_blank'}
    if text.upper() in {'-','--','N/A','NONE','NULL'}:return {**result,'category':'unavailable'}
    if not position:
        if text.upper()=='LEADER':return {**result,'category':'leader'}
        match=re.fullmatch(r'LAP\s+(\d+)',text,re.I)
        if match:return {**result,'category':'leader_lap_counter','value':int(match[1])}
        match=re.fullmatch(r'\+?\s*(\d+)\s*LAPS?',text,re.I)
        if match and int(match[1])>0:return {**result,'category':'lap_deficit','value':int(match[1])}
    try:
        if ':' in text and not position:
            if not re.fullmatch(r'\+?\s*[0-9]+(?::[0-9]+){0,1}:[0-9]+(?:\.[0-9]+)?',text):return result
            parts=text.removeprefix('+').strip().split(':')
            if len(parts) not in (2,3):return result
            parsed=[Decimal(p) for p in parts]
            if any(not p.is_finite() or p<0 for p in parsed) or any(p>=60 for p in parsed[1:]):return result
            if any(p!=p.to_integral_value() for p in parsed[:-1]):return result
            number=sum(p*(Decimal(60)**i) for i,p in enumerate(reversed(parsed)))
        else:
            if not re.fullmatch(r'\+?\s*(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?',text):return result
            number=Decimal(text.replace(' ',''))
        if not number.is_finite() or number<0:return result
        numeric=float(number)
        if not math.isfinite(numeric):return result
        if position:
            if number!=number.to_integral_value() or not 1<=number<=20:return result
            return {**result,'category':'rank','value':int(number)}
        return {**result,'category':'seconds','value':numeric}
    except (DecimalException,ValueError,OverflowError):return result


class HeaderCursor:
    """Read only a bounded binary header until a packet is admitted.

    An invalid header has no availability. Forecast consumers must stop there;
    only the separate full-file inventory may discard it and continue.
    """
    def __init__(self,path):
        self.file=Path(path).open('rb');self.pending=None;self.highwater=-1
        self.sequence=0;self.finished=False

    def peek(self):
        while self.pending is None and not self.finished:
            prefix=self.file.readline(12)
            if not prefix:self.finished=True;break
            sequence=self.sequence;self.sequence+=1
            if prefix in (b'\n',b'\r\n'):continue
            valid=HEADER.fullmatch(prefix) is not None
            stamp=(int(prefix[0:2])*3600+int(prefix[3:5])*60+int(prefix[6:8]))*1000+int(prefix[9:12]) if valid else None
            regressed=stamp<self.highwater if valid else None
            if valid:self.highwater=max(self.highwater,stamp)
            self.pending={'sequence':sequence,'recorded_ms':stamp,
                'available_ms':self.highwater if valid else None,'regressed':regressed,
                'header_valid':valid,'prefix_hex':prefix.hex(),'prefix_ended_line':prefix.endswith(b'\n')}
        return self.pending

    def _remainder(self):
        payload=self.file.readline(MAX_PAYLOAD_BYTES+1)
        if len(payload)<=MAX_PAYLOAD_BYTES:return payload,None
        while not payload.endswith(b'\n'):
            payload=self.file.readline(MAX_PAYLOAD_BYTES+1)
            if not payload:break
        return None,'OversizePayload'

    def consume(self):
        record=self.peek()
        if record is None:raise ValueError('No pending packet')
        if not record['header_valid']:raise ValueError('Unplaceable timestamp cannot be consumed for forecasting')
        payload,error=self._remainder();self.pending=None;body={}
        if error is None:
            try:
                body=json.loads(payload.decode('utf-8'))
                if not isinstance(body,dict):raise ValueError('Expected dictionary packet')
            except (UnicodeDecodeError,ValueError) as exc:error=type(exc).__name__;body={}
        return {**record,'payload':body,'error':error}

    def discard_unplaceable(self):
        """Inventory-only recovery; no guessed clock and no forecast state."""
        record=self.peek()
        if record is None or record['header_valid']:raise ValueError('Expected an unplaceable header')
        if not record['prefix_ended_line']:self._remainder()
        self.pending=None
        return record

    def close(self):self.file.close()


def inventory_stream(path):
    """Separate complete-file diagnostics, never consulted by snapshots."""
    cursor=HeaderCursor(path);counts=Counter();updates=Counter();errors=Counter()
    try:
        while (header:=cursor.peek()) is not None:
            counts['packets']+=1
            if not header['header_valid']:
                counts['invalid_timestamp_headers']+=1;cursor.discard_unplaceable();continue
            record=cursor.consume();counts['timestamp_regressions']+=int(record['regressed'])
            if record['error']:errors[record['error']]+=1;continue
            lines=record['payload'].get('Lines',{})
            if not isinstance(lines,dict):errors['InvalidLines']+=1;continue
            for driver,patch in lines.items():
                if not isinstance(driver,str) or not isinstance(patch,dict):errors['InvalidDriverPatch']+=1;continue
                for name in FIELDS:
                    if name in patch:
                        value=parse_value(patch[name],position=name=='Position')
                        updates[f'{name}/{value["outer_type"]}/{value["leaf_type"]}/{value["category"]}']+=1
    finally:cursor.close()
    return {'counts':dict(sorted(counts.items())),'payload_errors':dict(sorted(errors.items())),
        'raw_update_inventory':dict(sorted(updates.items()))}


class GapCursor:
    def __init__(self,path):
        self.header_cursor=HeaderCursor(path)
        self.states={};self.inventory=Counter();self.processed=Counter();self.last_sequence=None
        self.last_cutoff=None;self.source_ambiguous=False;self.blocked_header=None

    def _apply(self,record):
        self.processed['packets']+=1;self.last_sequence=record['sequence']
        if record['regressed']:self.processed['timestamp_regressions']+=1
        if record['error']:
            self.source_ambiguous=True;self.processed['parse_gaps']+=1;return
        lines=record['payload'].get('Lines',{})
        if not isinstance(lines,dict):self.source_ambiguous=True;self.processed['invalid_lines']+=1;return
        # Apply every driver in this atomic packet before answering any query.
        for driver,patch in lines.items():
            if not isinstance(driver,str) or not isinstance(patch,dict):
                self.source_ambiguous=True;self.processed['invalid_driver_patch']+=1;continue
            state=self.states.setdefault(driver,{'fields':{},'rank_change_sequence':None})
            updates={field:parse_value(patch[field],position=field=='Position') for field in FIELDS if field in patch}
            for field,value in updates.items():
                self.inventory[f'{field}/{value["outer_type"]}/{value["leaf_type"]}/{value["category"]}']+=1
                if not value['updates']:continue
                prior=state['fields'].get(field)
                if field=='Position' and (prior is None or (prior['category'],prior['value'])!=(value['category'],value['value'])):
                    state['rank_change_sequence']=record['sequence']
                state['fields'][field]={**value,'available_ms':record['available_ms'],'recorded_ms':record['recorded_ms'],
                    'sequence':record['sequence'],'source_regressed':bool(record['regressed'])}

    def before(self,cutoff_ns):
        if type(cutoff_ns) is not int or (self.last_cutoff is not None and cutoff_ns<self.last_cutoff):
            raise ValueError('Exact nondecreasing nanosecond cutoffs required')
        while True:
            header=self.header_cursor.peek()
            if header is None:break
            if not header['header_valid']:
                self.source_ambiguous=True;self.blocked_header=dict(header);break
            if header['available_ms']*1_000_000>=cutoff_ns:break
            self._apply(self.header_cursor.consume())
        self.last_cutoff=cutoff_ns

    def query(self,driver,*,cutoff_ns):
        self.before(cutoff_ns)
        state=self.states.get(driver,{'fields':{},'rank_change_sequence':None})
        fields={}
        for name in FIELDS:
            current=state['fields'].get(name)
            if current is None:fields[name]={'category':'never_observed','value':None,'age_seconds':None}
            else:
                fields[name]={**current,'age_seconds':(cutoff_ns-current['available_ms']*1_000_000)/1e9}
                if fields[name]['age_seconds']<=0:raise AssertionError('Future/equal-time update reached snapshot')
        position=fields['Position'];rank=position['value'] if position['category']=='rank' else None
        duplicate=rank is not None and sum(s['fields'].get('Position',{}).get('category')=='rank'
            and s['fields']['Position']['value']==rank for s in self.states.values())>1
        gap,interval=fields['GapToLeader'],fields['IntervalToPositionAhead']
        leaders={'leader','leader_lap_counter'}
        inconsistent=(rank!=1 and (gap['category'] in leaders or interval['category'] in leaders)) if rank is not None else False
        if rank==1:
            inconsistent=inconsistent or gap['category']=='lap_deficit' or (gap['category']=='seconds' and gap['value']>0)
            inconsistent=inconsistent or (interval['category']=='seconds' and interval['value']>0) or interval['category']=='lap_deficit'
        rank_changed=state['rank_change_sequence']
        old_interval=rank_changed is not None and interval.get('sequence',-1)<rank_changed
        regressed=any(field.get('source_regressed',False) for field in fields.values())
        flags={'unknown_or_invalid_rank':rank is None,'duplicate_reported_rank':duplicate,
            'interval_predates_last_rank_change':old_interval,'leader_rank_category_inconsistent':bool(inconsistent),
            'latest_field_timestamp_regressed':regressed,'source_parse_gap_or_unknown_clock':self.source_ambiguous}
        readiness={}
        for age in THRESHOLDS:
            def fresh(field):return field['age_seconds'] is not None and field['age_seconds']<=age and not field.get('source_regressed',False)
            leader_ok=rank==1 and fresh(gap) and (gap['category'] in leaders or (gap['category']=='seconds' and gap['value']==0))
            follower_ok=rank is not None and rank>1 and fresh(interval) and interval['category']=='seconds' and not old_interval
            readiness[str(age)]=bool((leader_ok or follower_ok) and not any(flags[k] for k in flags if k!='interval_predates_last_rank_change'))
        relevant=gap if rank==1 else interval
        relevant_category=(relevant['category'] in leaders or (relevant['category']=='seconds' and relevant['value']==0)) if rank==1 else relevant['category']=='seconds'
        flags['missing_clear_invalid_or_stale_gap']=bool(not relevant_category or relevant['age_seconds'] is None or relevant['age_seconds']>10)
        leader_gap_ready=gap['category'] in {'seconds','leader','leader_lap_counter','lap_deficit'} and gap['age_seconds'] is not None and gap['age_seconds']<=10 and not gap.get('source_regressed',False)
        return {'fields':fields,'rank_change_sequence':rank_changed,'flags':flags,
            'race_order_gap_ready':readiness['10'],'readiness_by_max_age_seconds':readiness,
            'leader_gap_category_ready':bool(leader_gap_ready and not self.source_ambiguous and not inconsistent),
            'last_consumed_packet_sequence':self.last_sequence,'processed_packet_count':self.processed['packets'],
            'unplaceable_timestamp_boundary':self.blocked_header}

    def close(self):self.header_cursor.close()

    def diagnostics(self):
        return {'processed':dict(self.processed),'source_ambiguous':self.source_ambiguous,
            'unplaceable_timestamp_boundary':self.blocked_header}


def issuance_metadata(path,event,expected_rows):
    rows=[]
    with Path(path).open() as f:
        for line in f:
            value=json.loads(line)
            if set(value)&TARGETS:raise ValueError('Target fields are forbidden')
            row={key:value[key] for key in IDENTITY}
            if row['event_key']!=event or type(row['issued_at_ns']) is not int or not isinstance(row['driver_id'],str):
                raise ValueError('Original issuance metadata invalid')
            expected=f"{event}/{row['driver_id']}/{row['issued_after_lap_number']}/{row['issued_at_ns']}"
            if row['issuance_id']!=expected:raise ValueError('Original issuance identity changed')
            rows.append(row)
    if len(rows)!=expected_rows or len({r['issuance_id'] for r in rows})!=len(rows):raise ValueError('Original issuance population changed')
    if [r['issued_at_ns'] for r in rows]!=sorted(r['issued_at_ns'] for r in rows):raise ValueError('Original issuance clock order changed')
    return rows


def source_files():
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(HERE.iterdir()) if p.suffix in {'.py','.json','.md'}}


def inputs(spec):
    return [*spec['parent_bindings'],*(v for e in spec['events'] for v in (e['stream'],e['issuances']))]


def validate_spec(spec):
    if [v['event_key'] for v in spec['events']]!=[202201,202301] or spec['lags_seconds']!=[2,0]:raise ValueError('Pilot scope changed')
    if [v['issuances']['rows'] for v in spec['events']]!=[853,874] or spec['expected_original_issuances']!=1727:raise ValueError('Pilot population changed')
    if spec['fields']!=list(FIELDS) or spec['readiness']['maximum_gap_age_seconds']!=10 or spec['readiness']['coverage_age_thresholds_seconds']!=list(THRESHOLDS):raise ValueError('Declared semantics changed')


def freeze(out,review_path):
    out=Path(out);spec=read(HERE/'specification.json');validate_spec(spec)
    sources=source_files();review=read(review_path)
    if review.get('approved_for_execution_lock') is not True or review.get('source_files')!=sources:raise ValueError('Exact-source independent approval required')
    for item in inputs(spec):check(item)
    command=[sys.executable,'-m','pytest','-q','--import-mode=importlib','-p','no:cacheprovider',str(HERE)]
    test=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
    save(out/'pre_fit_tests.json',{'command':command,'exit_code':test.returncode,'stdout':test.stdout,'stderr':test.stderr,'source_files':sources})
    if test.returncode or source_files()!=sources:raise ValueError('Tests failed or source changed before closure')
    save(out/'design_lock.json',{'closed_at_utc':now(),'source_files':sources,'inputs':inputs(spec),
        'specification':binding(HERE/'specification.json'),'review':binding(review_path),
        'historical_payload_analysis_before_lock':False,'model_fits':0,'prediction_scores':0})


def run(out):
    out=Path(out);lock=read(out/'design_lock.json')
    if source_files()!=lock['source_files']:raise ValueError('Frozen source changed')
    spec=read(check(lock['specification']));validate_spec(spec)
    for item in lock['inputs']:check(item)
    save(out/'attempt.json',{'started_at_utc':now(),'design_lock':binding(out/'design_lock.json')})
    try:
        entries=[];total=0
        for event in spec['events']:
            original=issuance_metadata(check(event['issuances']),event['event_key'],event['issuances']['rows'])
            event_entries=[]
            for lag in spec['lags_seconds']:
                cursor=GapCursor(check(event['stream']));path=out/f"{event['event_key']}_lag{lag}.jsonl"
                coverage=Counter();categories=Counter();age_missing=Counter()
                try:
                    with path.open('x') as f:
                        for row in original:
                            cutoff=row['issued_at_ns']-lag*10**9;snapshot=cursor.query(row['driver_id'],cutoff_ns=cutoff)
                            for field in snapshot['fields'].values():
                                field['age_seconds_at_issuance']=None if field['age_seconds'] is None else field['age_seconds']+lag
                            value={**row,'lag_seconds':lag,'cutoff_ns':cutoff,**snapshot}
                            f.write(json.dumps(value,sort_keys=True,allow_nan=False)+'\n')
                            for age,flag in value['readiness_by_max_age_seconds'].items():coverage['ready_'+age+'s']+=int(flag)
                            for name,flag in value['flags'].items():coverage[name]+=int(flag)
                            for name,field in value['fields'].items():
                                categories[name+'/'+field['category']]+=1
                                age_missing[name]+=int(field['age_seconds'] is None)
                finally:cursor.close()
                event_entries.append({'event_key':event['event_key'],'lag_seconds':lag,'rows':len(original),
                    'snapshot_file':binding(path),'coverage':dict(coverage),'field_categories':dict(categories),
                    'field_age_missing':dict(age_missing),'forecast_consumption':cursor.diagnostics()})
                total+=len(original)
                print(json.dumps({'stage':'pilot_event_lag_closed','event_key':event['event_key'],'lag':lag,'rows':len(original)}),flush=True)
            # Only after both snapshot files close may the full payload inventory
            # read remaining future or temporally unplaceable packets.
            full_inventory=inventory_stream(check(event['stream']))
            for entry in event_entries:entry['full_stream_inventory']=full_inventory
            entries.extend(event_entries)
        if total!=spec['expected_rows_both_lags']:raise ValueError('Population changed')
        for item in lock['inputs']:check(item)
        if source_files()!=lock['source_files']:raise ValueError('Source changed during pilot')
        save(out/'result.json',{'closed_at_utc':now(),'status':'closed_target_free_feasibility_not_model_validation',
            'design_lock':binding(out/'design_lock.json'),'rows_both_lags':total,'entries':entries,
            'model_fits':0,'prediction_scores':0,'new_downloads':0,'physical_nearest_car_claim':False})
    except Exception as exc:
        save(out/'failure.json',{'failed_at_utc':now(),'error':str(exc),'traceback':traceback.format_exc()});raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['freeze','run'])
    parser.add_argument('--out',type=Path,default=OUT);parser.add_argument('--review',type=Path);args=parser.parse_args()
    if args.stage=='freeze':
        if args.review is None:parser.error('--review is required')
        freeze(args.out,args.review)
    else:run(args.out)


if __name__=='__main__':main()
