"""Locked 44-race discovery input acquisition; never load sessions or fit models."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import time
import zlib

import requests
from fastf1 import _api

ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/research/boundary_20260908/sector_forecast'
DATA=ROOT/'data/f1/boundary_20260908/sector_forecast'
CONTRACT=HERE/'acquisition_contract.json'


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def now():return datetime.now(timezone.utc).isoformat()
def rel(path):return str(Path(path).resolve().relative_to(ROOT))
def read(path):return json.loads(Path(path).read_text())
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


def check_binding(binding):
    path=ROOT/binding['path'];assert sha(path)==binding['sha256'],str(path)
    if 'bytes' in binding:assert path.stat().st_size==binding['bytes']


def validate():
    contract=read(CONTRACT)
    assert contract['max_parallel_requests']==2
    assert contract['streams']==['TimingData','SessionStatus','TimingAppData']
    assert contract['base_urls']==[_api.base_url_mirror,_api.base_url]
    for field in ['manifest','pilot_acquisition','pilot_evidence']:check_binding(contract[field])
    sessions=contract['sessions'];mapping={s['event_key']:s for s in read(ROOT/contract['manifest']['path'])['mappings']}
    assert len(sessions)==44 and len({s['event_key'] for s in sessions})==44
    assert {s['event_key']//100 for s in sessions}=={2022,2023}
    assert sorted(contract['reuse_pilot_event_keys'])==[202201,202301]
    for session in sessions:
        assert session['session_path'].endswith('_Race/')
        assert all(mapping[session['event_key']][k]==v for k,v in session.items())
        assert sha(ROOT/session['original_laps_path'])==session['original_laps_sha256']
    track=contract['track_status_reuse'];assert {s['event_key'] for s in track}==set(mapping)
    for row in track:
        check_binding(row['stream']);check_binding(row['receipt'])
    pilot=read(ROOT/contract['pilot_acquisition']['path'])
    assert len(pilot['streams'])==6
    for row in pilot['streams']:
        assert row['status']=='downloaded' and row['event_key'] in contract['reuse_pilot_event_keys']
        assert sha(ROOT/row['decoded_body_path'])==row['decoded_body_sha256']
        for attempt in row['attempts']:
            if 'wire_body_path' in attempt:assert sha(ROOT/attempt['wire_body_path'])==attempt['wire_body_sha256']
    free=shutil.disk_usage(ROOT).free
    assert free>=contract['minimum_free_disk_bytes'],'Insufficient disk headroom'
    return contract,pilot,free


def validate_stream_start(path):
    with Path(path).open(encoding='utf-8-sig') as f:
        line=next((x for x in f if x.strip()),None)
    if line is None:raise ValueError('Empty stream')
    stamp=line[:12];parts=stamp.split(':')
    if len(parts)!=3 or not all(part.replace('.','',1).isdigit() for part in parts):raise ValueError('Invalid initial recorded timestamp')
    if not isinstance(json.loads(line[12:]),dict):raise ValueError('Expected dictionary initial payload')


def decoded_body(path,encoding,cap):
    """Bound decompression as well as the wire body; identity avoids a duplicate."""
    if encoding in ['', 'identity']:return path
    destination=path.with_suffix('.decoded.jsonStream')
    if encoding=='gzip':
        with gzip.open(path,'rb') as f:body=f.read(cap+1)
    elif encoding=='deflate':
        decoder=zlib.decompressobj();body=decoder.decompress(path.read_bytes(),cap+1)
        if decoder.unconsumed_tail or not decoder.eof:raise ValueError('Incomplete or oversized deflate body')
    else:raise ValueError('Unsupported content encoding: '+encoding)
    if len(body)>cap:raise ValueError('Decoded stream exceeds cap')
    destination.write_bytes(body);return destination


def verify_stream_result(row):
    for attempt in row.get('attempts',[]):
        if 'wire_body_path' in attempt:assert sha(ROOT/attempt['wire_body_path'])==attempt['wire_body_sha256']
    if row['status']=='downloaded':assert sha(ROOT/row['decoded_body_path'])==row['decoded_body_sha256']


def fetch(session,stream,contract):
    event=session['event_key'];meta=DATA/f'{event}_{stream}.source.json'
    if meta.exists():
        old=read(meta);assert old['contract_sha256']==sha(CONTRACT);verify_stream_result(old);return old
    result={'event_key':event,'stream':stream,'session_key':session['session_key'],'session_path':session['session_path'],
            'contract_sha256':sha(CONTRACT),'status':'unavailable','attempts':[],
            'receipt_limit':'These UTC timestamps describe current acquisition, not historical race-day client receipts.'}
    for ordinal,base in enumerate(contract['base_urls'],start=1):
        url=base+'/static/'+session['session_path']+stream+'.jsonStream';response=None
        wire_path=DATA/f'{event}_{stream}.attempt{ordinal}.httpbody';assert not wire_path.exists(),'Unexpected unreceipted body; inspect before resuming.'
        attempt={'requested_url':url,'request_started_at_utc':now(),'wire_body_complete':False};started=time.monotonic()
        try:
            response=requests.get(url,headers={**_api.headers,'Accept-Encoding':'identity'},timeout=(contract['request_timeout_connect_seconds'],contract['request_timeout_read_seconds']),stream=True)
            attempt.update(actual_url=response.url,http_status=response.status_code,headers_received_at_utc=now(),response_headers={k:v for k,v in response.headers.items() if k.lower() in ['date','etag','last-modified','content-type','content-encoding','content-length']})
            count=0
            with wire_path.open('xb') as f:
                while True:
                    chunk=response.raw.read(min(65536,contract['max_http_body_bytes']+1-count),decode_content=False)
                    if not chunk:break
                    f.write(chunk);count+=len(chunk)
                    if count>contract['max_http_body_bytes']:raise ValueError('HTTP body exceeds cap; saved prefix is partial')
            expected=response.headers.get('Content-Length')
            if expected is not None and int(expected)!=count:raise ValueError('HTTP content length mismatch')
            attempt['wire_body_complete']=True
            if response.status_code!=200:continue
            decoded=decoded_body(wire_path,response.headers.get('Content-Encoding','').lower(),contract['max_decoded_body_bytes'])
            validate_stream_start(decoded)
            result.update(status='downloaded',decoded_body_path=rel(decoded),decoded_body_sha256=sha(decoded),decoded_body_bytes=decoded.stat().st_size)
        except Exception as exc:
            attempt.update(error_type=type(exc).__name__,error=str(exc))
        finally:
            attempt.update(body_finished_at_utc=now(),wall_seconds=time.monotonic()-started)
            if wire_path.exists():attempt.update(wire_body_path=rel(wire_path),wire_body_bytes=wire_path.stat().st_size,wire_body_sha256=sha(wire_path))
            result['attempts'].append(attempt)
            if response is not None:response.close()
        if result['status']=='downloaded':
            result['successful_request']=attempt;break
    result['completed_at_utc']=now();write(meta,result)
    print(event,stream,result['status'],result.get('decoded_body_bytes',0),flush=True)
    return result


def main(dry_run=False):
    contract,pilot,free=validate();OUT.mkdir(parents=True,exist_ok=True);DATA.mkdir(parents=True,exist_ok=True)
    source_files={rel(Path(__file__)):sha(__file__),rel(CONTRACT):sha(CONTRACT)}
    summary={'sessions':44,'existing_pilot_streams':6,'existing_track_status_streams':44,'new_stream_jobs':126,
             'max_simultaneous_http_requests':2,'free_disk_bytes':free,'estimated_raw_disk_bytes':contract['estimated_raw_disk_bytes'],'source_files':source_files}
    if dry_run:print(json.dumps(summary,sort_keys=True));return summary
    if (OUT/'acquisition.json').exists():raise FileExistsError('Completed acquisition is immutable')
    lock_path=OUT/'acquisition_lock.json'
    if lock_path.exists():
        assert read(lock_path)['source_files']==source_files,'Source drift during resume'
    else:write(lock_path,{'frozen_at_utc':now(),'contract_sha256':sha(CONTRACT),'source_files':source_files,'preflight':summary,'scope':'Acquisition only; zero fits and zero predictive scores'})
    streams=[]
    for row in pilot['streams']:
        streams.append({**row,'status':'reused_verified_pilot','source_status':row['status'],'pilot_manifest':contract['pilot_acquisition'],'no_copy_or_request':True})
    jobs=[(session,stream) for session in contract['sessions'] if session['event_key'] not in contract['reuse_pilot_event_keys'] for stream in contract['streams']]
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures={executor.submit(fetch,session,stream,contract):(session['event_key'],stream) for session,stream in jobs}
        for future in as_completed(futures):
            row=future.result();streams.append(row)
            write(OUT/'acquisition_progress.json',{'updated_at_utc':now(),'completed_streams':len(streams),'expected_streams':132,'downloaded':sum(r['status']=='downloaded' for r in streams),'unavailable':sum(r['status']=='unavailable' for r in streams)})
    streams.sort(key=lambda r:(r['event_key'],r['stream']))
    for row in streams:
        if row['status']=='reused_verified_pilot':assert sha(ROOT/row['decoded_body_path'])==row['decoded_body_sha256']
        else:verify_stream_result(row)
    assert len(streams)==132
    paths={ROOT/a['wire_body_path'] for r in streams if r['status']!='reused_verified_pilot' for a in r['attempts'] if 'wire_body_path' in a}
    paths|={ROOT/r['decoded_body_path'] for r in streams if r['status']=='downloaded'}
    result={'completed_at_utc':now(),'status':'complete_all_stream_jobs_recorded','contract_sha256':sha(CONTRACT),'acquisition_lock_sha256':sha(lock_path),
            'source_files':source_files,'sessions':contract['sessions'],'streams':streams,'existing_track_status':contract['track_status_reuse'],
            'counts':{'expected_streams':132,'downloaded':sum(r['status']=='downloaded' for r in streams),'reused_verified_pilot':6,'unavailable':sum(r['status']=='unavailable' for r in streams),'reused_track_status':44,'new_saved_body_bytes':sum(p.stat().st_size for p in paths),'new_http_attempts':sum(len(r['attempts']) for r in streams if r['status']!='reused_verified_pilot')},
            'models_fitted':0,'predictive_scores_computed':0,'free_disk_bytes_after':shutil.disk_usage(ROOT).free}
    write(OUT/'acquisition.json',result);print(json.dumps({'manifest':rel(OUT/'acquisition.json'),'sha256':sha(OUT/'acquisition.json'),'counts':result['counts']},sort_keys=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--dry-run',action='store_true');args=parser.parse_args();main(args.dry_run)
