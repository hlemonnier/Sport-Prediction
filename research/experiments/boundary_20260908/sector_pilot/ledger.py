"""Streaming archival-packet ledger. Does not reconstruct historical client receipt."""
from collections import Counter
import json,math
from pathlib import Path


def milliseconds(value):
    h,m,s=value.split(':');return (int(h)*3600+int(m)*60)*1000+int(round(float(s)*1000))


def duration(value):
    try:
        parts=str(value).split(':');number=sum(float(x)*60**i for i,x in enumerate(reversed(parts)))
        return number if math.isfinite(number) and number>0 else None
    except (ValueError,TypeError):return None


def items(value):
    return value.items() if isinstance(value,dict) else enumerate(value) if isinstance(value,list) else []


def records(path,stats):
    highwater=-1
    with Path(path).open(encoding='utf-8-sig') as file:
        for sequence,line in enumerate(file):
            if not line.strip():continue
            stats['packets']+=1
            try:stamp=milliseconds(line[:12])
            except Exception as exc:
                stats['malformed_packets']+=1
                yield {'sequence':sequence,'recorded_ms':None,'available_ms':highwater,'regressed':False,'error':type(exc).__name__,'payload':{}}
                continue
            late=stamp<highwater;stats['timestamp_regressions']+=int(late);highwater=max(highwater,stamp)
            try:
                payload=json.loads(line[12:])
                if not isinstance(payload,dict):raise ValueError('Expected dictionary packet')
            except Exception as exc:
                stats['malformed_packets']+=1
                yield {'sequence':sequence,'recorded_ms':stamp,'available_ms':highwater,'regressed':late,'error':type(exc).__name__,'payload':{}}
                continue
            yield {'sequence':sequence,'recorded_ms':stamp,'available_ms':highwater,'regressed':late,'error':None,'payload':payload}


class Aux:
    def __init__(self,kind,path):
        self.kind=kind;self.stats=Counter();self.consumed_errors=0;self.iterator=iter(records(path,self.stats));self.next=next(self.iterator,None);self.status=None;self.status_time=None;self.status_sequence=None;self.status_recorded_ms=None;self.status_regressed=False;self.tyres={}

    def before(self,time):
        # A malformed timestamp has no causal cross-stream placement. Stop this
        # auxiliary cursor; never move the unknown packet into an earlier prefix.
        while self.next is not None and self.next['recorded_ms'] is not None and self.next['available_ms']<time:
            record=self.next;body=record['payload'];self.consumed_errors+=int(record['error'] is not None)
            if self.kind in ['SessionStatus','TrackStatus']:
                if 'Status' in body:
                    self.status=str(body['Status']);self.status_time=record['available_ms'];self.status_sequence=record['sequence'];self.status_recorded_ms=record['recorded_ms'];self.status_regressed=record['regressed']
            elif self.kind=='TimingAppData':
                for driver,patch in items(body.get('Lines',{})):
                    if not isinstance(patch,dict):continue
                    stints=self.tyres.setdefault(str(driver),{})
                    for number,update in items(patch.get('Stints',{})):
                        if not isinstance(update,dict):continue
                        try:key=int(number)
                        except (ValueError,TypeError):continue
                        state=stints.setdefault(key,{})
                        for field,value in update.items():state[field]={'value':value,'available_ms':record['available_ms'],'sequence':record['sequence'],'recorded_ms':record['recorded_ms'],'timestamp_regressed':record['regressed']}
            self.next=next(self.iterator,None)

    def tyre(self,driver):
        stints=self.tyres.get(driver,{})
        if not stints:return {'highest_observed_stint_index':None,'fields':{},'active_stint_certified':False}
        key=max(stints)
        return {'highest_observed_stint_index':key,'fields':{k:dict(v) for k,v in stints[key].items() if k in ['Compound','New','StartLaps','TotalLaps','LapNumber']},'active_stint_certified':False}

    def finish(self):
        self.before(float('inf'))
        return {'packets':dict(self.stats),'final_status':self.status,'final_status_time_ms':self.status_time,'blocked_by_unknown_timestamp':self.next is not None and self.next['recorded_ms'] is None}


def new_driver():
    return {'counter':None,'counter_highwater':None,'epoch':0,'epoch_reasons':set(),'counter_quarantine':False,'boundary_ms':None,'first':{},'last':{},'cleared':set(),'fields':{}}


def build(paths,event,emit):
    aux={kind:Aux(kind,paths[kind]) for kind in ['SessionStatus','TimingAppData','TrackStatus']}
    stats=Counter();drivers={};update_kinds=Counter();ambiguity=Counter();known=Counter();raw_fields=Counter();running=Counter();ledger_count=0;positive_counts=Counter()
    for record in records(paths['TimingData'],stats):
        now=record['available_ms']
        for cursor in aux.values():cursor.before(now)
        for driver,patch in items(record['payload'].get('Lines',{})):
            if not isinstance(patch,dict):continue
            driver=str(driver);state=drivers.setdefault(driver,new_driver());raw_fields.update(patch.keys())
            previous=state['counter'];current=previous;changed=False
            if 'NumberOfLaps' in patch:
                try:
                    number=float(patch['NumberOfLaps'])
                    if not number.is_integer() or number<0:raise ValueError('counter')
                    current=int(number)
                except (ValueError,TypeError):
                    state['epoch_reasons'].add('invalid_counter');stats['invalid_counter_updates']+=1
                else:
                    if previous is None or current!=previous:
                        changed=True;state['epoch']+=1;state['first']={};state['last']={};state['cleared']=set();state['boundary_ms']=now;state['epoch_reasons']=set()
                        high=state['counter_highwater']
                        if previous is not None and current<previous:
                            state['counter_quarantine']=True;state['epoch_reasons'].add('counter_reversal');stats['counter_reversals']+=1
                        if previous is not None and current>previous+1:
                            state['epoch_reasons'].add('counter_jump');stats['counter_jumps']+=1
                        if state['counter_quarantine'] and high is not None and current>high:state['counter_quarantine']=False
                        if state['counter_quarantine']:state['epoch_reasons'].add('counter_quarantine')
                        state['counter_highwater']=max(current,high if high is not None else current)
                        stats['counter_boundaries']+=1
                    state['counter']=current
            if record['regressed']:state['epoch_reasons'].add('raw_timestamp_regression')
            for field in ['InPit','PitOut','Position','Retired','Stopped','IsAccurate']:
                if field in patch:state['fields'][field]={'value':patch[field],'available_ms':now,'sequence':record['sequence']}
            last=patch.get('LastLapTime',{})
            if isinstance(last,dict) and 'Value' in last:
                state['fields']['LastLapTime']={'value':duration(last['Value']),'available_ms':now,'sequence':record['sequence']}
            for key,update in items(patch.get('Speeds',{})):
                if isinstance(update,dict) and 'Value' in update:
                    state['fields']['Speed'+str(key)]={'value':update['Value'],'available_ms':now,'sequence':record['sequence']}
            prepacket_s1=dict(state['fields'].get('Sector1',{}));sector_updates=[]
            # Snapshot every sector field atomically before emitting either S1 or
            # S2. JSON key order cannot determine an issued packet's information.
            for sector,update in items(patch.get('Sectors',{})):
                if not isinstance(update,dict) or 'Value' not in update:continue
                try:sn=int(sector)+1
                except (ValueError,TypeError):continue
                if sn not in [1,2,3]:continue
                value=duration(update['Value'])
                state['fields']['Sector'+str(sn)]={'value':value,'available_ms':now,'sequence':record['sequence']}
                sector_updates.append((sn,update['Value'],value))
            for sn,raw_value,value in sorted(sector_updates):
                if raw_value in ['',None]:state['cleared'].add(sn);stats['sector_clear_updates']+=1;continue
                if value is None:stats['invalid_sector_values']+=1;continue
                positive_counts[str(sn)]+=1
                if sn==1 and 1 in state['cleared'] and 2 in state['first'] and not changed:
                    state['epoch']+=1;state['first']={};state['last']={};state['cleared']=set();state['epoch_reasons'].add('reset_cycle_without_counter_advance');stats['ambiguous_reset_cycles']+=1
                reasons=set(state['epoch_reasons'])
                if current is None:reasons.add('unknown_counter')
                if changed:reasons.add('counter_sector_same_packet')
                if state['boundary_ms'] is not None and now-state['boundary_ms']<=5000:reasons.add('within_five_seconds_of_counter_boundary')
                if record['regressed']:reasons.add('raw_timestamp_regression')
                if stats['malformed_packets']:reasons.add('timing_stream_parse_gap')
                if sn==2:
                    s1=state['first'].get(1);s1field=state['fields'].get('Sector1',{})
                    if s1 is None or s1['sequence']>=record['sequence'] or prepacket_s1.get('value') is None or s1field.get('value') is None or 1 in state['cleared']:
                        reasons.add('no_strictly_prior_s1_in_epoch')
                    elif s1['ambiguity_reasons']:reasons.add('prior_s1_attribution_ambiguous')
                previous_sector=state['first'].get(sn)
                kind='first' if previous_sector is None else 'repeat' if state['last'].get(sn)==value else 'revision'
                state['last'][sn]=value;state['cleared'].discard(sn)
                if previous_sector is None:state['first'][sn]={'value':value,'sequence':record['sequence'],'available_ms':now,'ambiguity_reasons':sorted(reasons)}
                if sn not in [1,2]:continue
                status=aux['SessionStatus'];track=aux['TrackStatus'];tyre=aux['TimingAppData'].tyre(driver)
                exclusion=set(reasons)
                if status.status_regressed or track.status_regressed or any(v['timestamp_regressed'] for v in tyre['fields'].values()):exclusion.add('auxiliary_latest_value_timestamp_regressed')
                if kind!='first':exclusion.add('not_first_sector_value_in_epoch')
                if status.status!='Started':exclusion.add('running_state_unknown_or_not_started')
                if state['fields'].get('InPit',{}).get('value') is not False:exclusion.add('not_known_outside_pit')
                if any(cursor.consumed_errors for cursor in aux.values()):exclusion.add('auxiliary_stream_parse_gap')
                flags={field:field in state['fields'] and state['fields'][field]['value'] not in [None,''] for field in ['LastLapTime','Position','InPit','Sector1','Sector2','Sector3','SpeedI1','SpeedI2','SpeedFL','SpeedST']}
                flags['TrackStatus']=track.status is not None
                flags['ObservedCounter']=current is not None
                for field in ['Compound','New','StartLaps','TotalLaps']:flags['Tyre'+field]=field in tyre['fields'] and tyre['fields'][field]['value'] not in [None,'']
                flags['IsAccurateRaw']=bool('IsAccurate' in state['fields'])
                row={'event_key':event,'ledger_id':ledger_count,'driver':driver,'packet_sequence':record['sequence'],'recorded_ms':record['recorded_ms'],'checkpoint_ms':now,'receipt_clock':'archival_file_order_cumulative_timestamp_proxy_not_historical_client_receipt','counter_seen':current,'counter_highwater':state['counter_highwater'],'local_epoch':state['epoch'],'sector':sn,'sector_seconds':value,'update_kind':kind,'first_sector_sequence':state['first'][sn]['sequence'],'ambiguity_reasons':sorted(reasons),'candidate_checkpoint':not exclusion,'exclusion_reasons':sorted(exclusion),'session_status':status.status,'session_status_available_ms':status.status_time,'track_status':track.status,'track_status_available_ms':track.status_time,'received_driver_fields':{k:dict(v) for k,v in state['fields'].items()},'received_tyre_candidate':tyre,'raw_field_presence':flags,'canonical_hgb_exactly_reconstructible':False,'canonical_hgb_unresolved_reasons':['No receipt-certified IsAccurate or equivalent completed-lap validation','Processed lap-end/lap-index alignment not reconstructed from raw prefixes','Highest observed tyre-stint record is not certified active/canonical TyreLife'],'historical_client_receipt_available':False}
                assert all(v['available_ms']<=now for v in row['received_driver_fields'].values())
                row['session_status_source']={'sequence':status.status_sequence,'recorded_ms':status.status_recorded_ms,'timestamp_regressed':status.status_regressed}
                row['track_status_source']={'sequence':track.status_sequence,'recorded_ms':track.status_recorded_ms,'timestamp_regressed':track.status_regressed}
                assert all(v['available_ms']<now for v in tyre['fields'].values())
                emit(row);ledger_count+=1;update_kinds[kind]+=1;ambiguity.update(reasons);known.update(k for k,v in flags.items() if v);running[str(status.status)]+=1
                stats['candidate_checkpoints']+=int(not exclusion)
    auxiliary={kind:cursor.finish() for kind,cursor in aux.items()}
    return {'timing':dict(stats),'drivers':len(drivers),'sector_positive_updates':dict(positive_counts),'ledger_rows':ledger_count,'update_kinds':dict(update_kinds),'ambiguity_counts':dict(ambiguity),'raw_asof_field_present_rows':dict(known),'session_states_at_updates':dict(running),'raw_driver_patch_keys':dict(raw_fields),'auxiliary':auxiliary}
