"""Synthetic compact publication tests; no actual research result is opened."""
import json
from pathlib import Path

import pytest

from research.experiments.boundary_20260908.rival_gap_publication import publish as p


def compact(tmp_path):
    source=tmp_path/'source.json';source.write_bytes(b'{ "exact": "caf\xc3\xa9", "nested": {"x":1} }\r\n')
    return source


def test_byte_exact_copy_preserves_original_encoding_whitespace_and_line_endings(tmp_path):
    source=compact(tmp_path);out=tmp_path/'out'
    copies=p.copy_compact([(source,'execution/result.json')],out,root=tmp_path)
    assert source.read_bytes()==(out/'execution/result.json').read_bytes()
    assert p.verify_copies(copies,root=tmp_path)['all_copies_byte_identical']
    with pytest.raises(FileExistsError):p.copy_compact([],out,root=tmp_path)


@pytest.mark.parametrize('extension',['jsonl','npz','npy','pkl','pt','bin','httpbody','jsonStream','csv','py'])
def test_nonreceipt_assets_are_rejected_before_creating_output(tmp_path,extension):
    path=tmp_path/('raw.'+extension);path.write_text('{}')
    with pytest.raises(ValueError):p.copy_compact([(path,'artifact.'+extension)],tmp_path/'out',root=tmp_path)
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('content',['{"x":NaN}','{"x":Infinity}','[1,2,3]'])
def test_nonstandard_json_and_row_arrays_are_rejected(tmp_path,content):
    path=tmp_path/'x.json';path.write_text(content)
    with pytest.raises(ValueError):p.copy_compact([(path,'x.json')],tmp_path/'out',root=tmp_path)
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('relative',['../escape.json','/absolute.json'])
def test_destination_path_escape_is_rejected(tmp_path,relative):
    with pytest.raises(ValueError):p.copy_compact([(compact(tmp_path),relative)],tmp_path/'out',root=tmp_path)
    assert not (tmp_path/'out').exists()


def test_duplicate_names_and_size_limits_preflight_entire_package(tmp_path,monkeypatch):
    source=compact(tmp_path)
    with pytest.raises(ValueError):p.copy_compact([(source,'same.json'),(source,'same.json')],tmp_path/'duplicate',root=tmp_path)
    monkeypatch.setattr(p,'MAX_FILE_BYTES',1)
    with pytest.raises(ValueError,match='size'):p.copy_compact([(source,'one.json')],tmp_path/'large',root=tmp_path)
    monkeypatch.setattr(p,'MAX_FILE_BYTES',1000);monkeypatch.setattr(p,'MAX_TOTAL_BYTES',source.stat().st_size)
    with pytest.raises(ValueError,match='size'):p.copy_compact([(source,'one.json'),(source,'two.json')],tmp_path/'total',root=tmp_path)
    assert not any((tmp_path/name).exists() for name in ('duplicate','large','total'))


@pytest.mark.parametrize('mutate_original',[True,False])
def test_copy_or_original_mutation_is_detected(tmp_path,mutate_original):
    source=compact(tmp_path);out=tmp_path/'out';copies=p.copy_compact([(source,'copy.json')],out,root=tmp_path)
    (source if mutate_original else out/'copy.json').write_text('{}')
    with pytest.raises(ValueError):p.verify_copies(copies,root=tmp_path)


def fixture(tmp_path):
    execution=tmp_path/'execution';execution.mkdir();written=[]
    def save(path,value):
        path=tmp_path/path;path.parent.mkdir(parents=True,exist_ok=True);p.write(path,value);written.append(path)
        return p.record(path,tmp_path)
    def at(n):return f'2026-09-08T00:00:{n:02d}+00:00'
    source=tmp_path/'source.py';source.write_text('# frozen source\n');written.append(source)
    protocol=save('protocol.json',{'sample':24});spec=save('spec.json',{'fixed':True})
    source_map={str(path.relative_to(tmp_path)):p.sha(path) for path in (source,tmp_path/'protocol.json')}
    review=save('review.json',{'completed_at_utc':at(0),'approved_for_execution_lock':True,'source_files':source_map})
    save('execution/freeze_attempt.json',{'started_at_utc':at(1)})
    tests=save('execution/pre_fit_tests.json',{'completed_at_utc':at(2),'exit_code':0,'source_files':source_map})
    design=save('execution/design_lock.json',{'closed_at_utc':at(3),'sources':source_map,'inputs':{},
        'specification':spec,'verification_protocol':protocol,'independent_review':review,'pre_fit_tests':tests})
    save('execution/prepare_attempt.json',{'started_at_utc':at(4)})
    years={}
    for year,clock,rows in ((2022,5,18788),(2023,6,20432)):
        closure=save(f'execution/features_{year}/feature_closure.json',{'closed_at_utc':at(clock),'design_lock':design,
            'external_labels_read':False,'model_fits':0,'issuances_per_lag':rows,
            'events':[{'event_key':k} for k in range(year*100+1,year*100+23)]})
        years[str(year)]={'feature_closure':closure}
    data=save('execution/data_lock.json',{'closed_at_utc':at(7),'design_lock':design,'external_labels_read':False,
        'model_fits':0,'original_issuances_per_lag':39220,'years':years})
    save('execution/selection_attempt.json',{'started_at_utc':at(8)})
    fit=save('execution/fit_lock.json',{'closed_at_utc':at(9),'data_lock':data,'external_selection_labels_read':False,
        'fit_summary':{'fits':2,'base_refits':0,'rows_per_fit':18363,'fit_year':2022}})
    issued=save('execution/selection_issuance_lock.json',{'closed_at_utc':at(10),'data_lock':data,'fit_lock':fit,
        'external_selection_labels_read':False,'forecasts':{'0':{'rows':20432},'2':{'rows':20432}}})
    selected=save('execution/selection.json',{'completed_at_utc':at(11),'design_lock':design,'data_lock':data,'fit_lock':fit,
        'issuance_lock':issued,'summary':{'advances_to_later_evaluation':False,'promotion':False}})
    save('execution/selection_lock.json',{'closed_at_utc':at(12),'selection':selected,'design_lock':design,
        'selected_candidate':'gap_hgb','advances_to_later_evaluation':False,'promotion':False})
    # These can be hashed as provenance; the publisher must never parse them.
    for name in ('raw.jsonStream','features.jsonl','labels.jsonl','forecasts.jsonl','models.pkl'):
        path=tmp_path/name;path.write_bytes(b'not parseable as JSON or pickle');written.append(path)
    verified=save('verification.json',{'completed_at_utc':at(13),'status':'passed','model_fits':0,
        'design_sha256':design['sha256'],'selection_sha256':selected['sha256'],'protocol_sha256':protocol['sha256'],
        'saved_hgb_forecast_values_replayed':122592,'raw_prefix_snapshots':[{} for _ in range(24)],
        'recomputed':{'advances_to_later_evaluation':False},'source_files':source_map,
        'bindings':{str(path.relative_to(tmp_path)):p.record(path,tmp_path) for path in written}})
    publisher=tmp_path/'research/experiments/boundary_20260908/rival_gap_publication';publisher.mkdir(parents=True)
    (publisher/'publish.py').write_text('# synthetic publisher source fixture\n')
    return {'execution':execution,'verification':tmp_path/'verification.json','destination':tmp_path/'publication',
        'design_sha256':design['sha256'],'selection_sha256':selected['sha256'],'verification_sha256':verified['sha256'],'root':tmp_path}


def test_complete_synthetic_publication_has_exact_receipts_and_omits_all_row_and_model_assets(tmp_path):
    args=fixture(tmp_path);got=p.publish(**args);manifest=p.read(tmp_path/got['path'])
    assert manifest['advances_to_later_evaluation'] is False and manifest['promotion'] is False
    assert len(manifest['evidence_copies'])==16 and not manifest['source_only_ci_included']
    assert p.verify_copies(manifest['evidence_copies'],root=tmp_path)['files_checked']==16
    assert not any(path.suffix!='.json' for path in args['destination'].rglob('*') if path.is_file())
    assert not any(Path(row['source']).suffix!='.json' for row in manifest['evidence_copies'])
    assert len(manifest['chronology'])==14
    with pytest.raises(FileExistsError):p.publish(**args)


def test_optional_ci_validates_only_declared_source_files_not_new_publication_source_inventory(tmp_path):
    args=fixture(tmp_path)
    ci=tmp_path/'ci.json';p.write(ci,{'status':'passed','exit_code':0,'no_historical_data_or_artifacts_present':True,
        'no_provider_downloads_or_historical_fits':True,'source_copy_and_original_hashes_unchanged_after_tests':True,
        'source_files':{'source.py':p.sha(tmp_path/'source.py')}})
    got=p.publish(**args,ci_receipt=ci,ci_sha256=p.sha(ci));manifest=p.read(tmp_path/got['path'])
    assert manifest['source_only_ci_included'] and len(manifest['evidence_copies'])==17


@pytest.mark.parametrize('fault',['missing_selection_lock','missing_verifier','failed_verifier','wrong_verification_sha',
    'wrong_design_sha','wrong_selection_sha','failed_execution','wrong_decision','missing_forecast_replay','missing_prefix_sample'])
def test_unclosed_or_mismatched_evidence_cannot_create_publication(tmp_path,fault):
    args=fixture(tmp_path);v=p.read(args['verification'])
    if fault=='missing_selection_lock':(args['execution']/'selection_lock.json').unlink()
    elif fault=='missing_verifier':args['verification'].unlink()
    elif fault=='failed_verifier':v['status']='failed'
    elif fault=='wrong_verification_sha':args['verification_sha256']='0'*64
    elif fault=='wrong_design_sha':args['design_sha256']='0'*64
    elif fault=='wrong_selection_sha':args['selection_sha256']='0'*64
    elif fault=='failed_execution':p.write(args['execution']/'selection_failure.json',{'error':'synthetic'})
    elif fault=='wrong_decision':v['recomputed']['advances_to_later_evaluation']=True
    elif fault=='missing_forecast_replay':v['saved_hgb_forecast_values_replayed']=122591
    else:v['raw_prefix_snapshots'].pop()
    if fault in ('failed_verifier','wrong_decision','missing_forecast_replay','missing_prefix_sample'):
        args['verification'].write_text(json.dumps(v));args['verification_sha256']=p.sha(args['verification'])
    with pytest.raises((ValueError,FileNotFoundError)):p.publish(**args)
    assert not args['destination'].exists()


@pytest.mark.parametrize('timestamp',['2026-09-08T00:00:10','2026-09-08T00:00:01+00:00'])
def test_naive_or_out_of_order_verification_clock_rejected(tmp_path,timestamp):
    args=fixture(tmp_path);v=p.read(args['verification']);v['completed_at_utc']=timestamp
    args['verification'].write_text(json.dumps(v));args['verification_sha256']=p.sha(args['verification'])
    with pytest.raises(ValueError,match='chronology'):p.publish(**args)
    assert not args['destination'].exists()


@pytest.mark.parametrize('fault',['source','failure_marker'])
def test_concurrent_change_preserves_partial_publication_and_marks_failure(tmp_path,monkeypatch,fault):
    args=fixture(tmp_path);original=p.copy_compact
    def altered(*a,**kw):
        result=original(*a,**kw)
        if fault=='source':(tmp_path/'source.py').write_text('# changed\n')
        else:p.write(args['execution']/'selection_failure.json',{'error':'late failure'})
        return result
    monkeypatch.setattr(p,'copy_compact',altered)
    with pytest.raises(ValueError):p.publish(**args)
    assert args['destination'].exists() and (args['destination']/'publication_failure.json').exists()
    assert not (args['destination']/'publication_manifest.json').exists()
    with pytest.raises(FileExistsError):p.publish(**args)
