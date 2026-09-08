"""Publish compact closed CPC evidence without rewriting any original payload.

Suggested commit: research(f1-live): publish verified causal sequence evidence
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
DEFAULT = ROOT/'docs/research/evidence/boundary_telemetry_sequence_20260908'
EXECUTION_FILES = (
    'design_lock.json', 'pre_fit_tests.json', 'corpus_attempt.json', 'corpus_lock.json',
    'corpus/corpus.json', 'schedule/schedule.json', 'pretrain_attempt.json',
    'pretrain_lock.json', 'encoders/ordered_training.json', 'encoders/permuted_training.json',
    'embed_attempt.json', 'embedding_lock.json', 'selection_attempt.json', 'fit_lock.json',
    'selection_issuance_lock.json', 'selection.json', 'selection_lock.json')
PARENT_FILES = ('pre_fit_tests.json', 'freeze_failure.json', 'freeze_source_before_guard/amendment.json',
                'freeze_source_before_guard/run.py', 'freeze_source_before_guard/test_run.py',
                'verification_review.json', 'ci_source_only.json')
MAX_FILE_BYTES = 10*1024**2
MAX_TOTAL_BYTES = 32*1024**2


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    def invalid(value): raise ValueError('Nonstandard JSON: '+value)
    return json.loads(Path(path).read_text(encoding='utf-8'), parse_constant=invalid)


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False); stream.write('\n')


def local(path, root=ROOT):
    path = (Path(root)/Path(path)).resolve()
    path.relative_to(Path(root).resolve())
    return path


def record(path, root=ROOT):
    path = local(path, root)
    return {'path': str(path.relative_to(root)), 'sha256': sha(path), 'bytes': path.stat().st_size}


def check(item, root=ROOT):
    path = local(item['path'], root)
    assert sha(path) == item['sha256'], 'Changed binding: '+str(path)
    if 'bytes' in item: assert path.stat().st_size == item['bytes']
    return path


def copy_compact(items, destination, *, root=ROOT):
    """Fixed relative names only; never copy serialized arrays, models or rows."""
    destination = local(destination, root)
    if destination.exists(): raise FileExistsError('Publication directory is immutable')
    prepared = []; seen = set(); total = 0
    for source, name in items:
        source = local(source, root); relative = Path(name)
        assert not relative.is_absolute() and '..' not in relative.parts
        assert relative.suffix in ('.json', '.py'), 'Only compact JSON receipts or preserved source snapshots are publishable'
        assert source.suffix == relative.suffix and str(relative) not in seen
        if relative.suffix == '.json': read(source)  # reject nonstandard JSON before any output
        else:
            assert relative.parts[:2] == ('prior_freeze', 'freeze_source_before_guard')
            assert relative.name in ('run.py', 'test_run.py')
        size = source.stat().st_size
        assert size <= MAX_FILE_BYTES, 'Receipt exceeds compact publication limit'
        total += size; seen.add(str(relative)); prepared.append((source, relative, sha(source), size))
    assert total <= MAX_TOTAL_BYTES, 'Evidence package exceeds compact publication limit'
    destination.mkdir(parents=True, exist_ok=False)
    output = []
    for source, relative, digest, size in prepared:
        target = destination/relative; target.parent.mkdir(parents=True, exist_ok=True)
        with source.open('rb') as original, target.open('xb') as published:
            shutil.copyfileobj(original, published)
        assert sha(source) == sha(target) == digest and target.stat().st_size == size
        output.append({'source': str(source.relative_to(root)), 'published': str(target.relative_to(root)),
                       'sha256': digest, 'bytes': size})
    return output


def verify_copies(copies, *, root=ROOT):
    total = 0
    for item in copies:
        original = local(item['source'], root); published = local(item['published'], root)
        assert original.read_bytes() == published.read_bytes(), 'Publication bytes differ'
        assert sha(original) == sha(published) == item['sha256']
        assert original.stat().st_size == published.stat().st_size == item['bytes']
        total += item['bytes']
    return {'all_copies_byte_identical': True, 'files_checked': len(copies), 'bytes_checked': total,
            'hash_algorithm': 'SHA256', 'original_payloads_rewritten': False}


def publish(execution, verification, destination, *, design_sha256, selection_sha256):
    execution = local(execution); verification = local(verification); destination = local(destination)
    if destination.exists(): raise FileExistsError('Publication directory is immutable')
    # A completed evidence result is required before reading any scores or
    # building a publication. This helper never reads target or forecast rows.
    if not (execution/'selection_lock.json').is_file() or not verification.is_file():
        raise ValueError('Closed selection and independent verification are required')
    if any(execution.glob('*_failure.json')): raise ValueError('A failed execution cannot be published as completed')
    selected = read(execution/'selection_lock.json'); verified = read(verification)
    assert selected['design_lock']['sha256'] == design_sha256
    assert selected['selection']['sha256'] == selection_sha256
    assert verified['status'] == 'passed' and verified['design_sha256'] == design_sha256
    assert verified['selection_sha256'] == selection_sha256
    assert check(selected['design_lock']) == execution/'design_lock.json'
    assert check(selected['selection']) == execution/'selection.json'
    design = read(execution/'design_lock.json'); result = read(execution/'selection.json')
    execution_review_path = check(design['independent_review'])
    execution_review = read(execution_review_path)
    assert result['summary']['advances_to_later_evaluation'] is selected['advances_to_later_evaluation']
    assert result['summary']['promotion'] is False and selected['promotion'] is False
    assert verified['recomputed']['advances_to_later_evaluation'] is selected['advances_to_later_evaluation']
    parent = execution.parent
    failure = read(parent/'freeze_failure.json'); amendment = read(parent/'freeze_source_before_guard/amendment.json')
    assert failure['design_lock_created'] is False and failure['new_historical_construction_or_fits'] == 0
    assert failure['scientific_settings_changed'] is False and amendment['scientific_settings_changed'] is False
    assert amendment['historical_stages_executed'] == 0 and not (parent/'design_lock.json').exists()
    assert local(failure['next_execution_directory']) == execution
    for item in amendment['prior_freeze_attempt_bindings']: check(item)
    for item in amendment['sources']:
        assert sha(local(item['preserved_before_path'])) == item['before_sha256']
        assert sha(local(item['source_path'])) == item['current_sha256']
    ci = read(parent/'ci_source_only.json'); review = read(parent/'verification_review.json')
    assert ci['status'] == 'passed' and ci['exit_code'] == 0
    assert ci['no_historical_data_or_artifacts_present'] is True
    assert ci['no_provider_downloads_or_historical_fits'] is True
    assert ci['source_copy_and_original_hashes_unchanged_after_tests'] is True
    assert review['approved_for_post_selection_replay'] is True and review['design_sha256'] == design_sha256
    for table in (design['sources'], ci['sequence_and_verifier_sources'], review['source_files']):
        for path, expected in table.items(): assert sha(local(path)) == expected, 'Source snapshot changed'
    if 'workflow' in ci: check(ci['workflow'])
    stages = [read(execution/name) for name in ('design_lock.json', 'corpus_lock.json', 'pretrain_lock.json',
              'embedding_lock.json', 'fit_lock.json', 'selection_issuance_lock.json', 'selection_lock.json')]
    clocks = [datetime.fromisoformat(stage['closed_at_utc']) for stage in stages]
    assert all(t.utcoffset() is not None for t in clocks) and clocks == sorted(clocks)
    items = [(execution/name, 'execution_v2/'+name) for name in EXECUTION_FILES]
    items += [(parent/name, ('prior_freeze/' if name.startswith(('freeze_', 'pre_fit_')) else 'reviews/')+name)
              for name in PARENT_FILES]
    items += [(verification, 'verification/'+verification.name),
              (check(design['independent_review']), 'reviews/execution_review.json'),
              (check(design['specification']), 'protocol/specification.json')]
    verifier_protocol = ROOT/'research/experiments/boundary_20260908/telemetry_sequence_verification/protocol.json'
    assert sha(verifier_protocol) == verified['protocol_sha256']
    items += [(verifier_protocol, 'protocol/verification_protocol.json')]
    if 'supersedes_review' in execution_review:
        items += [(check(execution_review['supersedes_review']), 'reviews/execution_review_before_guard.json')]
    protocol_binding = execution_review['independent_numeric_verification']['protocol']
    assert check(protocol_binding) == verifier_protocol
    corpus_attempt = read(execution/'corpus_attempt.json')
    pretrain_attempt = read(execution/'pretrain_attempt.json')
    embedding_attempt = read(execution/'embed_attempt.json')
    selection_attempt = read(execution/'selection_attempt.json')
    temporal = [execution_review['reviewed_at_utc'], design['closed_at_utc'],
                corpus_attempt['started_at_utc'], pretrain_attempt['started_at_utc']]
    assert [datetime.fromisoformat(t) for t in temporal] == sorted(datetime.fromisoformat(t) for t in temporal)
    assert datetime.fromisoformat(review['reviewed_at_utc']) <= datetime.fromisoformat(selection_attempt['started_at_utc'])
    # All files also consumed by the verifier must match its original receipt.
    for path, _ in items:
        relative = str(local(path).relative_to(ROOT))
        if relative in verified['bindings']: check({'path': relative, **verified['bindings'][relative]})
    copies = copy_compact(items, destination)
    checks = verify_copies(copies)
    checks.update({'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'selection_sha256': selection_sha256,
                   'design_sha256': design_sha256, 'verification_sha256': sha(verification)})
    write(destination/'publication_check.json', checks)
    manifest = {
        'schema_version': 1, 'status': 'closed_verified_retrospective_research',
        'published_at_utc': datetime.now(timezone.utc).isoformat(), 'execution_directory': str(execution.relative_to(ROOT)),
        'selection_sha256': selection_sha256, 'design_sha256': design_sha256,
        'independent_verification_sha256': sha(verification),
        'advances_to_later_evaluation': selected['advances_to_later_evaluation'], 'promotion': False,
        'evidence_copies': copies, 'publication_check': record(destination/'publication_check.json'),
        'copy_policy': 'Every included original is copied byte for byte. Internal historical paths and hashes are preserved.',
        'prior_attempt': {'failed_before_design_and_historical_execution': True, 'scientific_settings_changed': False,
            'description': 'Initial freeze rejected concurrent source drift. The failed-state guard amendment and exact prior source snapshots are preserved.'},
        'execution_v2_chronology': [{'closure': name, 'closed_at_utc': stage['closed_at_utc']} for name, stage in zip(
            ('design', 'corpus', 'pretraining', 'embeddings', 'supervised_fit', 'selection_forecasts', 'selection'), stages)],
        'verification_declaration_chronology': {
            'protocol_sha256': protocol_binding['sha256'],
            'protocol_bound_by_execution_review_at_utc': execution_review['reviewed_at_utc'],
            'design_closed_at_utc': design['closed_at_utc'],
            'corpus_started_at_utc': corpus_attempt['started_at_utc'],
            'pretraining_started_at_utc': pretrain_attempt['started_at_utc'],
            'embedding_started_at_utc': embedding_attempt['started_at_utc'],
            'full_verifier_reviewed_at_utc': review['reviewed_at_utc'],
            'selection_started_at_utc': selection_attempt['started_at_utc'],
            'interpretation': 'The sample/statistical verification protocol was hash-bound before corpus construction and SSL. The complete verifier implementation was reviewed later, before selection; these are distinct milestones.'},
        'raw_data_published': False, 'serialized_models_published': False,
        'omitted_artifacts': 'Raw streams, per-driver binary tokens, sample schedule arrays, NPZ embeddings, PT/pickle models, and JSONL features/forecasts/labels remain local; their closed hashes remain in the included metadata.',
        'reproduction_limit': 'This compact package preserves results and provenance. Full replay additionally requires the omitted hash-matched local inputs and models.',
        'source_only_test_limit': 'The included CI receipt is an executed local source-only test, not proof of a remote GitHub run.',
        'sources': {**ci['sequence_and_verifier_sources'],
                    **{str(p.relative_to(ROOT)): sha(p) for p in sorted(HERE.glob('*.py'))}},
        'suggested_commit': 'research(f1-live): publish verified causal sequence evidence'}
    write(destination/'publication_manifest.json', manifest)
    # Exclusive output and final byte check also preserve partial publication if
    # an input is changed concurrently; no failed copy is silently overwritten.
    final_check = verify_copies(copies)
    assert final_check == {key: checks[key] for key in final_check}
    return record(destination/'publication_manifest.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execution', type=Path, required=True)
    parser.add_argument('--verification', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=DEFAULT)
    parser.add_argument('--design-sha256', required=True); parser.add_argument('--selection-sha256', required=True)
    args = parser.parse_args()
    print(json.dumps(publish(args.execution, args.verification, args.out,
        design_sha256=args.design_sha256, selection_sha256=args.selection_sha256), allow_nan=False), flush=True)


if __name__ == '__main__': main()
