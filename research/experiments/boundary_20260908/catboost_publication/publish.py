"""Publish the closed CatBoost screen as byte-exact compact evidence.

Suggested commit: research(f1-live): publish verified CatBoost selection evidence
"""
from datetime import datetime, timezone
import json
from pathlib import Path

from research.experiments.boundary_20260908.rival_gap_publication.publish import (
    copy_compact, verify_copies, read, write, sha, record, check)

ROOT = Path(__file__).resolve().parents[4]
EXECUTION = ROOT/'artifacts/research/boundary_20260908/catboost_frontier'
OUT = ROOT/'docs/research/evidence/boundary_catboost_frontier_20260908'
DESIGN = '5bca91308a982c84e9343ef99eb0f5340ce53236da3a274db6f93515ce2e222f'
SELECTION = 'd7c2a837956b2108537af0810313e949a5bcb52cb7a40a49ef96908acb364b57'
VERIFIED = '4981fb09a74920564e49974b4a3a15464a17c8c9283ba2f1470d805597d4d0a1'
CI = '1576a3d17502771d4b3e65ef606ffbd203b5076026d86de41882a8093591f1ca'


def main():
    if OUT.exists(): raise FileExistsError('Evidence publication is immutable')
    verification_path = check({'path':str(EXECUTION/'verification/result.json'),'sha256':VERIFIED})
    v = read(verification_path)
    if (v['status'] != 'passed' or v['design_sha256'] != DESIGN or v['selection_sha256'] != SELECTION
            or v['model_fits'] != 0 or v['catboost_imported'] is not False):
        raise ValueError('An exact successful independent replay is required')
    design = read(check({'path':str(EXECUTION/'design_lock.json'),'sha256':DESIGN}))
    selection = read(check({'path':str(EXECUTION/'selection.json'),'sha256':SELECTION}))
    selected = read(EXECUTION/'selection_lock.json')
    if selected['selection']['sha256'] != SELECTION or selected['winner'] != selection['summary']['winner']:
        raise ValueError('Selection closure changed')
    if selection['summary']['winner'] != v['recomputed']['winner'] or selection['summary']['advances_to_later_evaluation'] is not False:
        raise ValueError('This publication records the closed failed screen only')
    ci_path = check({'path':str(ROOT/'artifacts/research/boundary_20260908/catboost_publication/ci_source_only.json'),'sha256':CI})
    ci = read(ci_path)
    if ci['status'] != 'passed' or ci['exit_code'] != 0: raise ValueError('CI receipt did not pass')
    runtime_path = check(design['parent']['runtime_manifest']);runtime = read(runtime_path)
    receipts = [(EXECUTION/name,'execution/'+name) for name in (
        'freeze_attempt.json','pre_fit_tests.json','design_lock.json','selection_attempt.json',
        'fit_lock.json','forecast_lock.json','selection.json','selection_lock.json')]
    receipts += [(verification_path,'verification/result.json'),(check(design['specification']),'protocol/specification.json'),
        (check(design['review']),'reviews/execution_review.json'),(runtime_path,'runtime/manifest.json'),
        (check(runtime['synthetic_feasibility']),'runtime/synthetic_feasibility.json'),(ci_path,'reviews/ci_source_only.json')]
    evidence = {str(p):sha(p) for p,_ in receipts}
    def guard():
        if any(EXECUTION.glob('*_failure.json')): raise ValueError('Failed execution cannot be published')
        for path,item in v['bindings'].items(): check({'path':path,**item})
        for table in (design['sources'],design['inputs'],ci['source_files'],evidence):
            for path,digest in table.items(): check({'path':path,'sha256':digest})
    guard()
    try:
        copies = copy_compact(receipts,OUT);guard();checked = verify_copies(copies)
        write(OUT/'publication_check.json',checked)
        manifest = {'published_at_utc':datetime.now(timezone.utc).isoformat(),'status':'closed_verified_selection_failed',
            'design_sha256':DESIGN,'selection_sha256':SELECTION,'verification_sha256':VERIFIED,
            'ci_receipt_sha256':CI,'winner':'plain','advances_to_later_evaluation':False,'promotion':False,
            'evidence_copies':copies,'publication_check':record(OUT/'publication_check.json'),
            'sources':design['sources'],'publisher':record(Path(__file__)),
            'copy_helper':record(ROOT/'research/experiments/boundary_20260908/rival_gap_publication/publish.py'),
            'omitted':'Native/JSON model assets, original row ledgers, labels, forecast JSONL, package wheels and runtime binaries remain local and hash-bound.',
            'reproduction_limit':'Full independent prediction replay needs the omitted exported models and original row ledgers. No optimizer is rerun.',
            'ci_limit':'Included receipt records local source-only tests; remote workflow status must be checked separately.',
            'suggested_commit':'research(f1-live): publish verified CatBoost selection evidence'}
        write(OUT/'publication_manifest.json',manifest);guard();verify_copies(copies)
        print(json.dumps(record(OUT/'publication_manifest.json')),flush=True)
    except Exception as exc:
        if OUT.exists(): write(OUT/'publication_failure.json',{'error':str(exc),'partial_outputs_preserved':True})
        raise


if __name__ == '__main__': main()
