"""Raw-prefix checkpoint contamination context; no canonical laps or outcomes.

The epoch counter and local reset rules match the frozen pilot ledger. This is
observed-stream context, not final IsAccurate or physical/canonical lap identity.
"""
from collections import Counter

from research.experiments.boundary_20260908.sector_pilot.ledger import (
    Aux, duration, items, new_driver, records,
)


class Control(Aux):
    """Observe every received transition, including brief between-driver changes."""
    def advance(self, clock):
        transitions=[]
        while self.next is not None and self.next['recorded_ms'] is not None and self.next['available_ms']<clock:
            record=self.next
            self.consumed_errors+=int(record['error'] is not None)
            if 'Status' in record['payload']:
                self.status=str(record['payload']['Status'])
                self.status_time=record['available_ms'];self.status_sequence=record['sequence']
                self.status_recorded_ms=record['recorded_ms'];self.status_regressed=record['regressed']
            transitions.append({**record,'kind':self.kind})
            self.next=next(self.iterator,None)
        return transitions

    def snapshot(self):
        return {'value':self.status,'available_ms':self.status_time,'recorded_ms':self.status_recorded_ms,
                'sequence':self.status_sequence,'timestamp_regressed':self.status_regressed,
                'processed_parse_gaps':self.consumed_errors}


def _state():
    return {'core':new_driver(),'pit':None,'pit_source':None,'pit_out':None,'pit_out_source':None,
            'retired':None,'stopped':None,'epoch_start_sequence':None,'local_epoch_start_ms':None,
            'pit_sources':{},'neutralization_sources':{},'unknown':set(),'quality':set()}


def _note_control(state, kind, snapshot, at_epoch_start=False):
    unknown=state['unknown'];quality=state['quality'];value=snapshot['value']
    if value is None:unknown.add(kind+'_coverage_unknown')
    if snapshot['processed_parse_gaps']:unknown.add(kind+'_parse_gap')
    if snapshot['timestamp_regressed']:unknown.add(kind+'_timestamp_regression')
    if kind=='SessionStatus' and value is not None:
        if value!='Started':quality.add('session_not_started_during_observed_epoch')
        if value not in {'Inactive','Started','Aborted','Finished','Finalised','Ends'}:unknown.add('SessionStatus_unrecognized_value')
    if kind=='TrackStatus':
        if value is not None and value not in {'1','2','4','5','6','7'}:unknown.add('TrackStatus_unrecognized_value')
        if value is not None and any(c in value for c in '4567'):
            state['neutralization_sources'].setdefault(value,{**snapshot,'active_at_epoch_start':at_epoch_start})


def _start_epoch(state, sequence, clock, controls):
    state['epoch_start_sequence']=sequence;state['local_epoch_start_ms']=clock
    state['pit_sources']={};state['neutralization_sources']={};state['unknown']=set();state['quality']=set()
    if state['pit'] is True:state['pit_sources']['InPit']={**state['pit_source'],'active_at_epoch_start':True}
    elif state['pit'] is not False:state['unknown'].add('InPit_coverage_unknown')
    for kind,control in controls.items():_note_control(state,kind,control.snapshot(),True)
    # Deliberately do not carry the last PitOut=True state into a new epoch.
    # Only fresh received true updates contaminate the epoch in _own_fields().


def _own_fields(state, patch, record, apply_contamination):
    source={'available_ms':record['available_ms'],'recorded_ms':record['recorded_ms'],'sequence':record['sequence']}
    if 'InPit' in patch:
        state['pit']=patch['InPit'];state['pit_source']={**source,'value':patch['InPit']}
    if 'PitOut' in patch:
        state['pit_out']=patch['PitOut'];state['pit_out_source']={**source,'value':patch['PitOut']}
    if 'Retired' in patch:state['retired']=patch['Retired']
    if 'Stopped' in patch:state['stopped']=patch['Stopped']
    if not apply_contamination:return
    if state['pit'] is True:state['pit_sources'].setdefault('InPit',{**state['pit_source'],'active_at_epoch_start':False})
    elif state['pit'] is not False:state['unknown'].add('InPit_coverage_unknown')
    if patch.get('PitOut') is True:state['pit_sources'].setdefault('PitOut_fresh_true',{**state['pit_out_source'],'active_at_epoch_start':False})
    if state['retired'] is True:state['quality'].add('known_retired')
    if state['stopped'] is True:state['quality'].add('known_stopped')


def iter_contexts(paths,event_key,stats=None):
    """Yield one immutable context for every positive S1/S2 packet update.

    Required paths: TimingData, SessionStatus, TrackStatus. TimingAppData may be
    supplied but is unused: this context makes no active-stint/tyre-age claim.
    All auxiliary transitions use the pilot's strict earlier archive-time rule.
    """
    controls={kind:Control(kind,paths[kind]) for kind in ['SessionStatus','TrackStatus']}
    packet_stats=Counter();states={};counts=Counter()
    for record in records(paths['TimingData'],packet_stats):
        clock=record['available_ms'];transitions=[]
        for control in controls.values():transitions.extend(control.advance(clock))
        # Track every transition for every open epoch, not merely latest state at
        # the next update for that driver. Same-clock cross-stream ordering has
        # no effect on the union of contamination/unknown flags.
        for transition in sorted(transitions,key=lambda x:(x['available_ms'],x['kind'],x['sequence'])):
            for state in states.values():
                snapshot={'value':str(transition['payload']['Status']) if 'Status' in transition['payload'] else None,
                          'available_ms':transition['available_ms'],'recorded_ms':transition['recorded_ms'],
                          'sequence':transition['sequence'],'timestamp_regressed':transition['regressed'],
                          'processed_parse_gaps':int(transition['error'] is not None)}
                if 'Status' in transition['payload'] or transition['error']:_note_control(state,transition['kind'],snapshot)
        for driver,patch in items(record['payload'].get('Lines',{})):
            if not isinstance(patch,dict):continue
            driver=str(driver);new=driver not in states
            state=states.setdefault(driver,_state());core=state['core']
            # Atomic packet pit/retirement snapshot is established before a new
            # counter epoch starts. Boundary InPit=False need not carry old pit.
            _own_fields(state,patch,record,False)
            previous=core['counter'];current=previous;changed=False
            if 'NumberOfLaps' in patch:
                try:
                    number=float(patch['NumberOfLaps'])
                    if not number.is_integer() or number<0:raise ValueError('counter')
                    current=int(number)
                except (ValueError,TypeError):core['epoch_reasons'].add('invalid_counter')
                else:
                    if previous is None or current!=previous:
                        changed=True;core['epoch']+=1;core['first']={};core['last']={};core['cleared']=set();core['boundary_ms']=clock;core['epoch_reasons']=set()
                        high=core['counter_highwater']
                        if previous is not None and current<previous:core['counter_quarantine']=True;core['epoch_reasons'].add('counter_reversal')
                        if previous is not None and current>previous+1:core['epoch_reasons'].add('counter_jump')
                        if core['counter_quarantine'] and high is not None and current>high:core['counter_quarantine']=False
                        if core['counter_quarantine']:core['epoch_reasons'].add('counter_quarantine')
                        core['counter_highwater']=max(current,high if high is not None else current)
                    core['counter']=current
            if changed or new:_start_epoch(state,record['sequence'],clock,controls)
            if record['regressed']:core['epoch_reasons'].add('raw_timestamp_regression')
            if packet_stats['malformed_packets']:state['unknown'].add('TimingData_parse_gap')
            _own_fields(state,patch,record,True)
            if current is None:state['unknown'].add('observed_counter_unknown')
            for kind,control in controls.items():_note_control(state,kind,control.snapshot())
            prepacket_s1=dict(core['fields'].get('Sector1',{}));sector_updates=[]
            for key,update in items(patch.get('Sectors',{})):
                if not isinstance(update,dict) or 'Value' not in update:continue
                try:sector=int(key)+1
                except (ValueError,TypeError):continue
                if sector not in [1,2,3]:continue
                value=duration(update['Value'])
                core['fields']['Sector'+str(sector)]={'value':value,'available_ms':clock,'sequence':record['sequence']}
                sector_updates.append((sector,update['Value'],value))
            for sector,raw_value,value in sorted(sector_updates):
                if raw_value in ['',None]:core['cleared'].add(sector);continue
                if value is None:continue
                if sector==1 and 1 in core['cleared'] and 2 in core['first'] and not changed:
                    core['epoch']+=1;core['first']={};core['last']={};core['cleared']=set()
                    core['epoch_reasons'].add('reset_cycle_without_counter_advance');state['local_epoch_start_ms']=clock
                    # Contamination stays: this reset is not an observed counter boundary.
                reasons=set(core['epoch_reasons'])
                if current is None:reasons.add('unknown_counter')
                if changed:reasons.add('counter_sector_same_packet')
                if core['boundary_ms'] is not None and clock-core['boundary_ms']<=5000:reasons.add('within_five_seconds_of_counter_boundary')
                if packet_stats['malformed_packets']:reasons.add('timing_stream_parse_gap')
                if sector==2:
                    s1=core['first'].get(1);s1field=core['fields'].get('Sector1',{})
                    if s1 is None or s1['sequence']>=record['sequence'] or prepacket_s1.get('value') is None or s1field.get('value') is None or 1 in core['cleared']:reasons.add('no_strictly_prior_s1_in_epoch')
                    elif s1['ambiguity_reasons']:reasons.add('prior_s1_attribution_ambiguous')
                previous_sector=core['first'].get(sector)
                kind='first' if previous_sector is None else 'repeat' if core['last'].get(sector)==value else 'revision'
                core['last'][sector]=value;core['cleared'].discard(sector)
                if previous_sector is None:core['first'][sector]={'sequence':record['sequence'],'value':value,'ambiguity_reasons':sorted(reasons)}
                if sector not in [1,2]:continue
                pit=bool(state['pit_sources']);neutralized=bool(state['neutralization_sources'])
                snapshot={kind:control.snapshot() for kind,control in controls.items()}
                row={'event_key':event_key,'driver':driver,'packet_sequence':record['sequence'],'sector':sector,
                     'context_key':[event_key,driver,record['sequence'],sector],
                     'recorded_ms':record['recorded_ms'],'checkpoint_ms':clock,'counter_seen':current,'local_epoch':core['epoch'],
                     'raw_epoch_start_ms':core['boundary_ms'],'raw_epoch_start_packet_sequence':state['epoch_start_sequence'] if core['boundary_ms'] is not None else None,
                     'local_epoch_start_ms':state['local_epoch_start_ms'],'update_kind':kind,
                     'known_pit_contamination':pit,'known_neutralization_contamination':neutralized,'known_asof_contamination':pit or neutralized,
                     'pit_contamination_sources':{k:dict(v) for k,v in state['pit_sources'].items()},
                     'neutralization_contamination_sources':{k:dict(v) for k,v in state['neutralization_sources'].items()},
                     'current_in_pit':state['pit'],'last_received_pit_out_value':state['pit_out'],
                     'fresh_pit_out_true_in_epoch':'PitOut_fresh_true' in state['pit_sources'],
                     'unknown_coverage':bool(state['unknown']),'unknown_flags':sorted(state['unknown']),
                     'quality_flags':sorted(state['quality']|state['unknown']|reasons),
                     'epoch_attribution_flags':sorted(reasons),'strictly_prior_control_snapshot':snapshot,
                     'context_contract':'Observed raw epoch contamination only; not future cleanliness, IsAccurate or canonical lap identity.',
                     'historical_client_receipt_available':False}
                assert all(v['available_ms'] is None or v['available_ms']<clock for v in snapshot.values())
                counts['rows']+=1;counts['known_asof_contamination_rows']+=int(row['known_asof_contamination']);counts['unknown_coverage_rows']+=int(row['unknown_coverage'])
                yield row
    if stats is not None:stats.update({'contexts':dict(counts),'timing':dict(packet_stats),'drivers':len(states)})


iter_context=iter_contexts
