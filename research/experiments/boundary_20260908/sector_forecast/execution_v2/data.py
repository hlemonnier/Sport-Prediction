"""Target-free raw checkpoint assembly and a separate post-lock label pass."""
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import math
from numbers import Real
from pathlib import Path

from research.experiments.boundary_20260908.sector_pilot import ledger
from research.experiments.boundary_20260908.sector_forecast import baselines, completed, context

ROOT=Path(__file__).resolve().parents[5]
HERE=Path(__file__).resolve().parent
STREAMS=('TimingData','SessionStatus','TrackStatus','TimingAppData')
ISSUED_STATUS='issued'


def _sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def _event(value):
    if type(value) is not int or value<=0:raise ValueError('event_key must be a positive integer')
    return value


def load_event_paths(event_key,manifest_path=None):
    """Read only the frozen 44-event manifest and verify one event's raw bodies."""
    _event(event_key);contract=json.loads((HERE/'data_contract.json').read_text())
    path=ROOT/contract['acquisition']['path'] if manifest_path is None else Path(manifest_path)
    if _sha(path)!=contract['acquisition']['sha256']:raise ValueError('Acquisition manifest hash differs from the frozen contract')
    if event_key not in contract['acquisition']['event_keys']:raise ValueError('Event is outside the frozen discovery acquisition')
    manifest=json.loads(path.read_text());paths={}
    for record in manifest['streams']:
        if record['event_key']!=event_key:continue
        if record['stream'] in paths:raise ValueError('Duplicate acquired stream')
        if record['status'] not in {'downloaded','reused_verified_pilot'}:raise ValueError('Required raw stream is unavailable')
        raw=ROOT/record['decoded_body_path']
        if _sha(raw)!=record['decoded_body_sha256']:raise ValueError('Raw stream hash mismatch')
        paths[record['stream']]=raw
    track=[r for r in manifest['existing_track_status'] if r['event_key']==event_key]
    if len(track)!=1:raise ValueError('Exactly one existing TrackStatus binding is required')
    raw=ROOT/track[0]['stream']['path']
    if _sha(raw)!=track[0]['stream']['sha256']:raise ValueError('TrackStatus hash mismatch')
    paths['TrackStatus']=raw
    if set(paths)!=set(STREAMS):raise ValueError('All four input streams are required')
    return paths


def _key(row):
    key=(row['event_key'],row['driver'],row['packet_sequence'],row['sector'])
    if (type(key[0]) is not int or key[0]<=0 or not isinstance(key[1],str) or not key[1]
            or type(key[2]) is not int or key[2]<0 or key[3] not in (1,2)):
        raise ValueError('Invalid checkpoint identity')
    return key


def _finite(value,name,positive=False):
    if isinstance(value,bool) or not isinstance(value,Real) or not math.isfinite(value):raise ValueError(name+' must be finite numeric data')
    if positive and value<=0:raise ValueError(name+' must be positive')
    return float(value)


def _feature_value(value,name):
    if isinstance(value,bool) or not isinstance(value,Real) or math.isinf(value):raise ValueError(name+' must be numeric, finite or explicitly NaN')
    return float(value)


def _feature_encoder():
    # A separate owned module is loaded lazily; importing this assembly module
    # never reads canonical outcomes or constructs historical features.
    from .features import FEATURES, encode
    return tuple(FEATURES),encode


def _validate_context(checkpoint,ctx):
    fields=('event_key','driver','packet_sequence','sector','checkpoint_ms','recorded_ms','counter_seen','local_epoch','update_kind')
    if any(checkpoint[k]!=ctx[k] for k in fields):raise ValueError('Checkpoint/context identity or attribution mismatch')
    if checkpoint['ambiguity_reasons']!=ctx['epoch_attribution_flags']:raise ValueError('Checkpoint/context ambiguity mismatch')
    if ctx['context_key']!=list(_key(checkpoint)):raise ValueError('Context key does not match its fields')
    if type(ctx['known_asof_contamination']) is not bool:raise ValueError('Known contamination must be an explicit boolean')
    for control in ctx['strictly_prior_control_snapshot'].values():
        time=control['available_ms']
        if time is not None and _finite(time,'control source time')>=checkpoint['checkpoint_ms']:raise ValueError('Control snapshot is not strictly prior')


def _prefix(checkpoint,prior_s1=None):
    current=_finite(checkpoint['sector_seconds'],'Current sector',True)
    if checkpoint['sector']==1:return [current]
    s1=checkpoint['received_driver_fields'].get('Sector1')
    if s1 is None:raise ValueError('Supported S2 checkpoint lacks S1')
    # Prior S1 existence establishes the same-epoch prerequisite. Its duration
    # need not equal the latest value: an atomic S2 packet may correct S1.
    if (prior_s1 is None or prior_s1['sector']!=1
            or any(prior_s1[k]!=checkpoint[k] for k in ('event_key','driver','local_epoch'))
            or prior_s1['packet_sequence']>=checkpoint['packet_sequence']
            or prior_s1['ambiguity_reasons']):
        raise ValueError('S2 lacks unambiguous immutable prior S1 support in its epoch')
    _finite(prior_s1['sector_seconds'],'Prior S1 support',True)
    if _finite(prior_s1['checkpoint_ms'],'Prior S1 support clock')>checkpoint['checkpoint_ms']:
        raise ValueError('Future prior S1 support')
    if (type(s1['sequence']) is not int or s1['sequence']<0
            or s1['sequence']>checkpoint['packet_sequence']
            or _finite(s1['available_ms'],'S1 value source clock')>checkpoint['checkpoint_ms']):
        raise ValueError('Future or invalid S1 value source')
    return [_finite(s1['value'],'Latest observed S1',True),current]


def build_event(paths,event_key):
    """Return every raw checkpoint with status; this function never reads CSV.

    Complete input integrity is checked before any records are returned. The
    caller must persist and hash this target-free return before attach_targets.
    """
    _event(event_key)
    if not all(k in paths for k in STREAMS):raise ValueError('Four raw stream paths are required')
    paths={k:Path(paths[k]) for k in STREAMS};before={k:_sha(p) for k,p in paths.items()}
    completion_stats={};all_completed=list(completed.iter_completed(paths,event_key,completion_stats))
    if completion_stats.get('stream_input_valid') is not True:raise ValueError('Full raw stream input validation failed')
    checkpoint_rows=[];pilot_stats=ledger.build(paths,event_key,checkpoint_rows.append)
    context_stats={};context_rows=list(context.iter_contexts(paths,event_key,context_stats));contexts={}
    for ctx in context_rows:
        key=_key(ctx)
        if key in contexts:raise ValueError('Duplicate context key')
        contexts[key]=ctx
    keys=[_key(r) for r in checkpoint_rows]
    if len(set(keys))!=len(keys) or set(keys)!=set(contexts):raise ValueError('Missing, extra or duplicate checkpoint/context key')
    valid=[];completion_ids=set()
    for item in all_completed:
        if item['event_key']!=event_key:raise ValueError('Completion belongs to another event')
        if item['completion_id'] in completion_ids:raise ValueError('Duplicate completion identity')
        completion_ids.add(item['completion_id'])
        if item['observed_valid_completed'] is True:
            _finite(item['available_ms'],'Completion availability')
            valid.append(item)
    valid.sort(key=lambda r:(r['available_ms'],r['packet_sequence'],r['completion_id']))
    feature_names,encode=_feature_encoder()
    if not feature_names or len(set(feature_names))!=len(feature_names):raise ValueError('Fixed feature schema must be nonempty and unique')
    own_history=defaultdict(list);cursor=0;records=[];status_counts=Counter();previous_clock=-math.inf
    prior_s1=defaultdict(list)
    for checkpoint in checkpoint_rows:
        key=_key(checkpoint);ctx=contexts[key];_validate_context(checkpoint,ctx)
        clock=_finite(checkpoint['checkpoint_ms'],'Checkpoint clock')
        if clock<previous_clock:raise ValueError('Checkpoint replay clock is not monotone')
        previous_clock=clock
        while cursor<len(valid) and valid[cursor]['available_ms']<clock:
            item=valid[cursor];own_history[item['driver']].append(item);cursor+=1
        history=own_history[checkpoint['driver']]
        retired=any(checkpoint['received_driver_fields'].get(k,{}).get('value') is True for k in ['Retired','Stopped'])
        reasons=list(checkpoint['exclusion_reasons'])
        if retired:reasons.append('known_retired_or_stopped_at_issuance')
        first_gate=checkpoint['candidate_checkpoint'] is True and not retired
        status='pilot_excluded' if not checkpoint['candidate_checkpoint'] else 'retired_or_stopped' if retired else 'insufficient_history' if len(history)<baselines.MINIMUM_HISTORY else ISSUED_STATUS
        if first_gate and len(history)<baselines.MINIMUM_HISTORY:reasons.append('fewer_than_three_strictly_prior_valid_own_completions')
        last=history[-1] if history else None
        row={'event_key':event_key,'ledger_id':checkpoint['ledger_id'],'driver':checkpoint['driver'],
             'packet_sequence':checkpoint['packet_sequence'],'sector':checkpoint['sector'],
             'issuance_id':f'{event_key}:{checkpoint["driver"]}:{checkpoint["packet_sequence"]}:S{checkpoint["sector"]}',
             'recorded_ms':checkpoint['recorded_ms'],'checkpoint_ms':checkpoint['checkpoint_ms'],
             'counter_seen':checkpoint['counter_seen'],'raw_epoch_start_ms':ctx['raw_epoch_start_ms'],
             'raw_epoch_start_packet_sequence':ctx['raw_epoch_start_packet_sequence'],
             'status':status,'status_reasons':sorted(set(reasons)),
             'pilot_candidate_checkpoint':checkpoint['candidate_checkpoint'],'candidate_checkpoint':first_gate,
             'known_asof_contamination':ctx['known_asof_contamination'],'unknown_context_coverage':ctx['unknown_coverage'],
             'context_quality_flags':list(ctx['quality_flags']),'history_count':len(history),
             'history_last_available_ms':None if last is None else last['available_ms'],
             'source_provenance':{'context_key':list(key),'timing_packet_sequence':checkpoint['packet_sequence'],
                 'history_last_completion_id':None if last is None else last['completion_id'],
                 'history_last_packet_sequence':None if last is None else last['packet_sequence'],
                 'canonical_alignment_certified':False,'historical_client_receipt_certified':False},
             'points':{},'features':{},'baseline_diagnostics':{}}
        # Local epoch is only a within-pilot key for already received S1. Never
        # join it with the completed parser's differently defined ordinal.
        prefix_key=(checkpoint['driver'],checkpoint['local_epoch'])
        if status==ISSUED_STATUS:
            earlier=None
            if checkpoint['sector']==2:
                # Match pilot.first[1], not a later revision's own checkpoint
                # status. The latest received value is validated separately.
                earlier=next((r for r in prior_s1[prefix_key] if r['packet_sequence']<checkpoint['packet_sequence']),None)
            prefix=_prefix(checkpoint,earlier)
            if earlier is not None:
                source=checkpoint['received_driver_fields']['Sector1']
                row['source_provenance']['s1_prefix']={
                    'prior_support':{'event_key':earlier['event_key'],'driver':earlier['driver'],
                        'local_epoch':earlier['local_epoch'],'packet_sequence':earlier['packet_sequence'],
                        'available_ms':earlier['checkpoint_ms'],'value':earlier['sector_seconds']},
                    'value_source':deepcopy(source),
                    'same_packet_value_update':source['sequence']==checkpoint['packet_sequence'],
                    'value_changed_from_prior_support':source['value']!=earlier['sector_seconds']}
            result=baselines.predict_baselines(history,prefix,known_asof_contamination=ctx['known_asof_contamination'])
            expected=set(baselines.REFERENCE_NAMES)|{baselines.GAUSSIAN_NAME}
            if set(result['points'])!=expected:raise ValueError('All fixed references and Gaussian must issue together')
            row['points']={name:_finite(value,'Model point',True) for name,value in result['points'].items()}
            # Preserve the actual atomic snapshot and source clock. Never
            # backdate a correction or mutate an earlier issued S1 forecast.
            features=encode(checkpoint,history,ctx,result)
            if not isinstance(features,dict) or set(features)!=set(feature_names):raise ValueError('Encoder returned a different feature schema')
            row['features']={name:_feature_value(features[name],'Feature '+name) for name in feature_names}
            row['baseline_diagnostics']=deepcopy(result['diagnostics'])
        if checkpoint['sector']==1:prior_s1[prefix_key].append(checkpoint)
        records.append(row);status_counts[status]+=1
    after={k:_sha(p) for k,p in paths.items()}
    if before!=after:raise ValueError('Raw inputs changed during assembly')
    if len(records)!=pilot_stats['ledger_rows']:raise ValueError('Not every positive sector update was retained')
    terminal_status=pilot_stats['auxiliary']['SessionStatus']['final_status']
    diagnostics={'event_key':event_key,'records':len(records),'status_counts':dict(status_counts),
                 'terminal_status':terminal_status,'terminal':terminal_status in {'Finished','Finalised','Ends'},
                 'stream_input_valid':True,'completion_parser':completion_stats,'pilot_parser':pilot_stats,'context_parser':context_stats,
                 'source_streams':{k:{'path':str(paths[k]),'sha256':before[k],'bytes':paths[k].stat().st_size} for k in STREAMS},
                 'feature_names':list(feature_names),'point_names':list(baselines.REFERENCE_NAMES)+[baselines.GAUSSIAN_NAME],
                 'target_free':True,'canonical_csv_read':False,'model_fits_or_scores':0}
    return records,diagnostics


def attach_targets(records,csv_path,terminal):
    """Label deep copies AFTER the caller closes and hashes target-free records.

    Target IDs never contain sector: S1/S2 sharing a future outcome share weight.
    No model, feature or population decision is made from these outcomes.
    """
    if type(terminal) is not bool:raise ValueError('terminal must be an explicit boolean')
    from research.experiments.boundary_20260908.sector_pilot.run import target_index,resolve_target
    records=list(records);ids=[r['issuance_id'] for r in records]
    if len(set(ids))!=len(ids):raise ValueError('Duplicate issuance identity')
    events={_event(r['event_key']) for r in records}
    if len(events)>1:raise ValueError('One completed-lap CSV may label only one event')
    forbidden={'target','target_id','target_group_id','y','outcome_status'}
    if any(forbidden & r.keys() for r in records):raise ValueError('Expected target-free records, not an already labeled table')
    index,_=target_index(csv_path);labeled=[];targets={}
    for original in records:
        resolved=resolve_target(original,index,terminal);target=resolved['target'];row=deepcopy(original)
        identity=None if target is None else f'{original["event_key"]}:{original["driver"]}:csvrow:{target["csv_row"]}'
        if target is not None:
            signature=(_finite(target['lap_time_seconds'],'Target duration',True),_finite(target['recorded_time_seconds'],'Target time'),_finite(target['lap_number'],'Target lap number',True))
            if signature[1]<=original['checkpoint_ms']/1000:raise ValueError('Target is not strictly later than issuance')
            if identity in targets and targets[identity]!=signature:raise ValueError('Conflicting values for one target identity')
            targets[identity]=signature
        row.update({'outcome_status':resolved['status'],'target':deepcopy(target),'target_id':identity,'target_group_id':identity,
                    'y':None if target is None else target['lap_time_seconds'],
                    'target_time_seconds':None if target is None else target['recorded_time_seconds'],
                    'target_lap_number':None if target is None else target['lap_number'],
                    'target_csv_row':None if target is None else target['csv_row']})
        labeled.append(row)
    return labeled
