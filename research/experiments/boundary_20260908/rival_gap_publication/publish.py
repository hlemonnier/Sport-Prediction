"""Publish verified compact gap evidence without reading raw or row payloads.

Suggested commit: research(f1-live): publish verified rival-gap prediction evidence
"""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent
DEFAULT_EXECUTION=ROOT/'artifacts/research/boundary_20260908/rival_gap_forecast'
DEFAULT_DESTINATION=ROOT/'docs/research/evidence/boundary_rival_gap_forecast_20260908'
DEFAULT_DESIGN='cf656765fad43c8c95e831f2abb3c6b4edf621b8a92b3605ed4fdd3f26d4d49b'
EXECUTION_FILES=('freeze_attempt.json','pre_fit_tests.json','design_lock.json','prepare_attempt.json',
    'features_2022/feature_closure.json','features_2023/feature_closure.json','data_lock.json',
    'selection_attempt.json','fit_lock.json','selection_issuance_lock.json','selection.json','selection_lock.json')
MAX_FILE_BYTES=4*1024**2
MAX_TOTAL_BYTES=16*1024**2


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read(path):
    def bad(value):raise ValueError('Nonstandard JSON: '+value)
    return json.loads(Path(path).read_text(encoding='utf-8'),parse_constant=bad)


def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:
        json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')


def local(path,root=ROOT):
    root=Path(root).resolve();path=(root/Path(path)).resolve();path.relative_to(root)
    return path


def record(path,root=ROOT):
    path=local(path,root)
    return {'path':str(path.relative_to(Path(root).resolve())),'sha256':sha(path),'bytes':path.stat().st_size}


def check(item,root=ROOT):
    path=local(item['path'],root)
    if sha(path)!=item['sha256']:raise ValueError('Changed binding: '+str(path))
    if 'bytes' in item and path.stat().st_size!=item['bytes']:raise ValueError('Changed byte count: '+str(path))
    return path


def stamp(value):
    result=datetime.fromisoformat(value)
    if result.utcoffset() is None:raise ValueError('Evidence chronology requires timezone-aware timestamps')
    return result


def ordered(items):
    clocks=[stamp(value) for _,value in items]
    if clocks!=sorted(clocks):raise ValueError('Evidence chronology changed')
    return [{'milestone':name,'timestamp':value} for name,value in items]


def copy_compact(items,destination,*,root=ROOT):
    """Copy fixed JSON metadata only, after preflighting the complete package."""
    destination=local(destination,root)
    if destination.exists():raise FileExistsError('Publication directory is immutable')
    prepared=[];names=set();total=0
    for source,name in items:
        source=local(source,root);relative=Path(name)
        if (relative.is_absolute() or '..' in relative.parts or relative.suffix!='.json'
                or source.suffix!='.json' or str(relative) in names):raise ValueError('Only unique compact JSON receipt paths are permitted')
        size=source.stat().st_size
        if size>MAX_FILE_BYTES:raise ValueError('Receipt exceeds compact size limit')
        if not isinstance(read(source),dict):raise ValueError('Receipts must be JSON objects, not row arrays')
        total+=size;names.add(str(relative));prepared.append((source,relative,sha(source),size))
    if total>MAX_TOTAL_BYTES:raise ValueError('Evidence package exceeds compact size limit')
    destination.mkdir(parents=True,exist_ok=False);copies=[]
    for source,name,digest,size in prepared:
        target=destination/name;target.parent.mkdir(parents=True,exist_ok=True)
        with source.open('rb') as original,target.open('xb') as published:shutil.copyfileobj(original,published)
        if sha(source)!=digest or sha(target)!=digest or target.stat().st_size!=size:
            raise ValueError('Source changed during byte-exact copy')
        copies.append({'source':str(source.relative_to(Path(root).resolve())),
            'published':str(target.relative_to(Path(root).resolve())),'sha256':digest,'bytes':size})
    return copies


def verify_copies(copies,*,root=ROOT):
    total=0
    for item in copies:
        source=local(item['source'],root);published=local(item['published'],root)
        if (source.read_bytes()!=published.read_bytes() or sha(source)!=item['sha256']
                or source.stat().st_size!=item['bytes']):raise ValueError('Published bytes differ from preserved originals')
        total+=item['bytes']
    return {'all_copies_byte_identical':True,'files_checked':len(copies),'bytes_checked':total,
        'hash_algorithm':'SHA256','original_payloads_rewritten':False}


def publish(execution,verification,destination,*,design_sha256,selection_sha256,verification_sha256,
            ci_receipt=None,ci_sha256=None,root=ROOT):
    root=Path(root).resolve();execution=local(execution,root);verification=local(verification,root);destination=local(destination,root)
    if destination.exists():raise FileExistsError('Publication directory is immutable')
    # Do not open selection metrics until a hash-bound successful verifier exists.
    if not (execution/'selection_lock.json').is_file() or not verification.is_file():
        raise ValueError('Closed selection and independent verification are required')
    if any(execution.glob('*failure*.json')):raise ValueError('Failed execution cannot be published as complete')
    check({'path':str(verification),'sha256':verification_sha256},root)
    verified=read(verification)
    if (verified.get('status')!='passed' or verified.get('design_sha256')!=design_sha256
            or verified.get('selection_sha256')!=selection_sha256 or verified.get('model_fits')!=0):
        raise ValueError('Successful verification must match this exact design and selection')
    check({'path':str(execution/'design_lock.json'),'sha256':design_sha256},root)
    check({'path':str(execution/'selection.json'),'sha256':selection_sha256},root)
    design=read(execution/'design_lock.json');selection=read(execution/'selection.json');selected=read(execution/'selection_lock.json')
    if (check(selected['selection'],root)!=execution/'selection.json'
            or check(selected['design_lock'],root)!=execution/'design_lock.json'
            or selected['selected_candidate']!='gap_hgb' or selected['promotion'] is not False):raise ValueError('Selection closure mismatch')
    decision=selected['advances_to_later_evaluation']
    if (type(decision) is not bool or selection['summary']['advances_to_later_evaluation'] is not decision
            or verified['recomputed']['advances_to_later_evaluation'] is not decision
            or selection['summary']['promotion'] is not False):raise ValueError('Reported and verified gate decisions differ')
    if verified['saved_hgb_forecast_values_replayed']!=20432*3*2 or len(verified['raw_prefix_snapshots'])!=24:
        raise ValueError('Declared complete forecast and fixed prefix verification is missing')
    spec_path=check(design['specification'],root);protocol_path=check(design['verification_protocol'],root)
    review_path=check(design['independent_review'],root);review=read(review_path)
    if review.get('approved_for_execution_lock') is not True or review['source_files']!=design['sources']:
        raise ValueError('Execution source review mismatch')
    if sha(protocol_path)!=verified['protocol_sha256']:raise ValueError('Verification protocol changed')
    if design['sources'].get(str(protocol_path.relative_to(root)))!=sha(protocol_path):raise ValueError('Protocol was not in the reviewed source lock')
    pre_fit_path=check(design['pre_fit_tests'],root);tests=read(pre_fit_path)
    if tests['exit_code']!=0 or tests['source_files']!=design['sources']:raise ValueError('Pre-fit tests did not cover reviewed sources')
    data=read(execution/'data_lock.json');fit=read(execution/'fit_lock.json');issued=read(execution/'selection_issuance_lock.json')
    if (check(data['design_lock'],root)!=execution/'design_lock.json' or data['external_labels_read'] is not False
            or data['model_fits']!=0 or data['original_issuances_per_lag']!=39220):raise ValueError('Target-free data closure mismatch')
    if (check(fit['data_lock'],root)!=execution/'data_lock.json' or fit['external_selection_labels_read'] is not False
            or fit['fit_summary']['fits']!=2 or fit['fit_summary']['base_refits']!=0
            or fit['fit_summary']['rows_per_fit']!=18363 or fit['fit_summary']['fit_year']!=2022):raise ValueError('Fixed two-fit training closure mismatch')
    if (check(issued['fit_lock'],root)!=execution/'fit_lock.json' or check(issued['data_lock'],root)!=execution/'data_lock.json'
            or issued['external_selection_labels_read'] is not False or set(issued['forecasts'])!={'2','0'}
            or any(v['rows']!=20432 for v in issued['forecasts'].values())):raise ValueError('Both full target-free forecast closures are required')
    for key,path in (('design_lock','design_lock.json'),('data_lock','data_lock.json'),('fit_lock','fit_lock.json'),('issuance_lock','selection_issuance_lock.json')):
        if check(selection[key],root)!=execution/path:raise ValueError('Selection is not bound to this execution')
    year_closures={year:read(check(data['years'][str(year)]['feature_closure'],root)) for year in (2022,2023)}
    for year,total in ((2022,18788),(2023,20432)):
        closure=year_closures[year]
        if (check(data['years'][str(year)]['feature_closure'],root)!=execution/f'features_{year}/feature_closure.json'
                or closure['design_lock']!=data['design_lock'] or closure['external_labels_read'] is not False
                or closure['model_fits']!=0 or closure['issuances_per_lag']!=total
                or [v['event_key'] for v in closure['events']]!=list(range(year*100+1,year*100+23))):raise ValueError('Full year feature closure mismatch')
    attempts={name:read(execution/(name+'_attempt.json')) for name in ('freeze','prepare','selection')}
    chronology=ordered([
        ('independent_execution_review',review['completed_at_utc']),('freeze_started',attempts['freeze']['started_at_utc']),
        ('pre_fit_tests_completed',tests['completed_at_utc']),('design_closed',design['closed_at_utc']),
        ('prepare_started',attempts['prepare']['started_at_utc']),('2022_features_closed',year_closures[2022]['closed_at_utc']),
        ('2023_features_closed',year_closures[2023]['closed_at_utc']),('all_features_closed',data['closed_at_utc']),
        ('selection_started',attempts['selection']['started_at_utc']),('models_closed',fit['closed_at_utc']),
        ('both_forecast_ledgers_closed',issued['closed_at_utc']),('selection_completed',selection['completed_at_utc']),
        ('selection_lock_closed',selected['closed_at_utc']),('verification_completed',verified['completed_at_utc'])])
    inputs=[(execution/name,'execution/'+name) for name in EXECUTION_FILES]
    inputs.extend([(spec_path,'protocol/specification.json'),(protocol_path,'protocol/verification_protocol.json'),
        (review_path,'reviews/execution_review.json'),(verification,'verification/result.json')])
    maps=[design['sources'],design['inputs'],verified['source_files']]
    ci=None
    if (ci_receipt is None)!=(ci_sha256 is None):raise ValueError('CI path and exact SHA must be provided together')
    if ci_receipt is not None:
        ci_path=check({'path':str(ci_receipt),'sha256':ci_sha256},root);ci=read(ci_path)
        if (ci['status']!='passed' or ci['exit_code']!=0 or any(ci.get(k) is not True for k in (
            'no_historical_data_or_artifacts_present','no_provider_downloads_or_historical_fits',
            'source_copy_and_original_hashes_unchanged_after_tests'))):raise ValueError('Successful source-only CI receipt required')
        table=ci.get('source_files',ci.get('gap_and_verifier_sources'))
        if not isinstance(table,dict) or not table:raise ValueError('CI declared source bindings are required')
        maps.append(table)
        if 'workflow' in ci:check(ci['workflow'],root)
        inputs.append((ci_path,'reviews/ci_source_only.json'))
    evidence_bindings={str(local(path,root)):sha(local(path,root)) for path,_ in inputs}
    def guard():
        if any(execution.glob('*failure*.json')):raise ValueError('Late failed execution cannot be published')
        for table in maps:
            for path,digest in table.items():check({'path':path,'sha256':digest},root)
        for path,item in verified['bindings'].items():check({'path':path,**item},root)
        for path,digest in evidence_bindings.items():check({'path':path,'sha256':digest},root)
    guard()
    # All copied receipts that the verifier saw must still match its bindings.
    for path,_ in inputs:
        key=str(local(path,root).relative_to(root))
        if key in verified['bindings']:check({'path':key,**verified['bindings'][key]},root)
    try:
        copies=copy_compact(inputs,destination,root=root);guard()
        checks=verify_copies(copies,root=root)
        checks.update(checked_at_utc=datetime.now(timezone.utc).isoformat(),design_sha256=design_sha256,
            selection_sha256=selection_sha256,verification_sha256=verification_sha256,
            stage_chronology_checked=True,failed_execution_markers_absent=True)
        write(destination/'publication_check.json',checks)
        publisher_dir=root/'research/experiments/boundary_20260908/rival_gap_publication'
        publisher_sources={str(p.relative_to(root)):sha(p) for p in sorted(publisher_dir.iterdir()) if p.suffix in ('.py','.md')}
        manifest={'schema_version':1,'status':'closed_verified_retrospective_gap_research',
            'published_at_utc':datetime.now(timezone.utc).isoformat(),'execution_directory':str(execution.relative_to(root)),
            'design_sha256':design_sha256,'selection_sha256':selection_sha256,'verification_sha256':verification_sha256,
            'verification_protocol_sha256':verified['protocol_sha256'],'advances_to_later_evaluation':decision,'promotion':False,
            'evidence_copies':copies,'publication_check':record(destination/'publication_check.json',root),
            'chronology':chronology,'sources':{**design['sources'],**verified['source_files'],**publisher_sources},
            'raw_data_published':False,'serialized_models_published':False,'source_only_ci_included':ci is not None,
            'copy_policy':'Every included receipt is copied byte for byte; original paths and hashes remain unchanged.',
            'omitted_artifacts':'TimingData bodies, original and derived JSONL features/labels/forecasts, and pickle model assets remain local. Their hashes remain in the closed metadata.',
            'reproduction_limit':'Complete replay requires the omitted hash-matched local inputs and fitted models. This compact package preserves structured findings and provenance.',
            'verification_limit':'Full saved-forecast and metric replay is verified; raw-feature numerical replay uses the fixed 24-prefix sample. No additional fitting occurs during verification or publication.',
            'ci_limit':'An included local source-only test receipt does not prove a remote GitHub workflow run. Only its declared source paths are checked.',
            'suggested_commit':'research(f1-live): publish verified rival-gap prediction evidence'}
        write(destination/'publication_manifest.json',manifest)
        guard();verify_copies(copies,root=root)
        return record(destination/'publication_manifest.json',root)
    except Exception as exc:
        if destination.exists():write(destination/'publication_failure.json',{'status':'failed','error':str(exc),
            'failed_at_utc':datetime.now(timezone.utc).isoformat(),'partial_outputs_preserved':True})
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execution',type=Path,default=DEFAULT_EXECUTION)
    parser.add_argument('--verification',type=Path,required=True);parser.add_argument('--out',type=Path,default=DEFAULT_DESTINATION)
    parser.add_argument('--design-sha256',default=DEFAULT_DESIGN);parser.add_argument('--selection-sha256',required=True)
    parser.add_argument('--verification-sha256',required=True);parser.add_argument('--ci-receipt',type=Path);parser.add_argument('--ci-sha256')
    args=parser.parse_args()
    print(json.dumps(publish(args.execution,args.verification,args.out,design_sha256=args.design_sha256,
        selection_sha256=args.selection_sha256,verification_sha256=args.verification_sha256,
        ci_receipt=args.ci_receipt,ci_sha256=args.ci_sha256),allow_nan=False),flush=True)


if __name__=='__main__':main()
