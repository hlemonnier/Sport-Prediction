"""Two named discovery sessions; exact raw bytes and acquisition provenance."""
from datetime import datetime,timezone
import gzip,hashlib,json,time,zlib
from pathlib import Path
import requests
from fastf1 import _api

ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/research/boundary_20260908/sector_pilot'
DATA=ROOT/'data/f1/boundary_20260908/sector_pilot'
STREAMS=['TimingData','SessionStatus','TimingAppData']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,obj):Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def now():return datetime.now(timezone.utc).isoformat()


def fetch(session,stream):
    event=session['event_key'];path=DATA/f'{event}_{stream}.jsonStream';meta=path.with_name(path.name+'.source.json')
    if path.exists():
        existing=json.loads(meta.read_text());assert sha(path)==existing['decoded_body_sha256'];return existing
    attempts=[]
    for base in [_api.base_url_mirror,_api.base_url]:
        url=base+'/static/'+session['session_path']+stream+'.jsonStream';start=now();clock=time.monotonic();response=None
        try:
            response=requests.get(url,headers={**_api.headers,'Accept-Encoding':'identity'},timeout=(10,45),stream=True)
            headers_time=now();wire=response.raw.read(30_000_001,decode_content=False)
            if len(wire)>30_000_000:raise ValueError('Response exceeds frozen30MBsingle-stream safety cap')
            ended=now();wire_path=DATA/f'{event}_{stream}.attempt{len(attempts)+1}.httpbody';wire_path.write_bytes(wire)
            record={'requested_url':url,'actual_url':response.url,'http_status':response.status_code,'request_started_at_utc':start,'headers_received_at_utc':headers_time,'body_received_at_utc':ended,'wall_seconds':time.monotonic()-clock,'response_headers':{k:v for k,v in response.headers.items() if k.lower() in ['date','etag','last-modified','content-type','content-encoding','content-length']},'wire_body_path':str(wire_path.relative_to(ROOT)),'wire_body_bytes':len(wire),'wire_body_sha256':sha(wire_path)}
            attempts.append(record)
            if response.status_code!=200:continue
            encoding=response.headers.get('Content-Encoding','').lower()
            body=gzip.decompress(wire) if encoding=='gzip' else zlib.decompress(wire) if encoding=='deflate' else wire
            if encoding not in ['', 'identity','gzip','deflate']:raise ValueError('Unsupported content encoding')
            text=body.decode('utf-8-sig');first=next(x for x in text.splitlines() if x.strip());json.loads(first[12:])
            path.write_bytes(body)
            result={'event_key':event,'stream':stream,'session_path':session['session_path'],'status':'downloaded','successful_request':record,'decoded_body_path':str(path.relative_to(ROOT)),'decoded_body_bytes':len(body),'decoded_body_sha256':sha(path),'attempts':attempts,'provenance_limit':'UTCacquisition receipts are currentdownload receipts, not historicalliveclient packetreceipt times.'}
            write(meta,result);print(event,stream,len(body),response.status_code,flush=True);return result
        except Exception as exc:
            attempts.append({'requested_url':url,'request_started_at_utc':start,'failed_at_utc':now(),'error_type':type(exc).__name__,'error':str(exc)})
        finally:
            if response is not None:response.close()
    result={'event_key':event,'stream':stream,'session_path':session['session_path'],'status':'unavailable','attempts':attempts};write(meta,result);print('unavailable',event,stream,flush=True);return result


def main():
    if (OUT/'acquisition.json').exists():raise FileExistsError('Preserve completed acquisition')
    protocol=json.loads((HERE/'specification.json').read_text());lock=json.loads((OUT/'design_lock.json').read_text());assert sha(HERE/'specification.json')==lock['spec_sha256']
    for path,value in lock['input_files'].items():assert sha(ROOT/path)==value
    assert len(protocol['sessions'])==2 and {s['event_key'] for s in protocol['sessions']}=={202201,202301}
    write(OUT/'acquisition_attempt.json',{'started_at_utc':now(),'source_sha256':sha(__file__),'spec_sha256':sha(HERE/'specification.json'),'streams':STREAMS,'sessions':protocol['sessions'],'no_fitting_or_scoring':True})
    results=[fetch(session,stream) for session in protocol['sessions'] for stream in STREAMS]
    write(OUT/'acquisition.json',{'completed_at_utc':now(),'source_sha256':sha(__file__),'spec_sha256':sha(HERE/'specification.json'),'sessions':protocol['sessions'],'streams':results,'no_fitting_or_scoring':True})


if __name__=='__main__':main()
