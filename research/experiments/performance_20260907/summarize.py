"""Build a source-bound scorecard for the first performance research cycle."""
from pathlib import Path
import csv
import hashlib
import json

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'docs/research/evidence'
SOURCES={
 'pre_event':'artifacts/research/performance_20260907/pre_event/cycle1/results.json',
 'race':'artifacts/research/performance_20260907/race/v3/results.json',
 'live':'artifacts/research/performance_20260907/live/corrected_cycle_1/results.json',
 'football':'artifacts/research/performance_20260907/football/evidence.json',
 'football_transfer':'artifacts/research/performance_20260907/football/transfer_evidence.json',
 'live_recent':'artifacts/research/performance_20260907/live/recent_extension_final/results.json',
}


def main():
    data={k:json.loads((ROOT/p).read_text()) for k,p in SOURCES.items()}
    rows=[]
    def add(lane,model,period,n,baseline,candidate,ci,units,role):
        assert baseline>0 and n>0
        rows.append({'lane':lane,'candidate':model,'period':period,'scored_events_or_matches':n,
          'baseline':baseline,'candidate_metric':candidate,'delta':candidate-baseline,
          'relative_reduction_percent':100*(baseline-candidate)/baseline,
          'ci95_delta_low':ci[0],'ci95_delta_high':ci[1],'units':units,'evidence_role':role})
    for block in ['primary_2024_2025','exposed_2026']:
        s=data['pre_event']['summaries'][block]
        for name,m in s['paired'].items():
            baseline='baseline_qualifying' if name.startswith('Q') else 'baseline_best_lap'
            add('qualifying' if name.startswith('Q') else 'best_lap',name,block,s['events'],
                s['point_metrics'][baseline],s['point_metrics'][name],m['three_event_circular_block_ci95'],
                'positions MAE' if name.startswith('Q') else 'seconds MAE','fixed mechanisms; historical diagnostic, 3-event block CI')
    for block in ['held_forward_2024_2025','exposed_2026']:
        for name,s in data['race']['summary'][block].items():
            if name=='baseline':continue
            add('race_order',name,block,s['events'],s['baseline_mae'],s['mae'],s['paired_event_ci95'],
                'positions MAE','2023-selected within family; amended retrospective cohort, event CI')
    for block in ['transfer_2024_2025','exposed_2026']:
        for name,s in data['live']['results'][block].items():
            add('live_next_eligible_lap',name,block,s['events'],s['baseline_event_mae_seconds'],s['candidate_event_mae_seconds'],
                s['ci95_delta_seconds'],'seconds MAE','2023-selected within family; historical transfer, event CI')
    for name in ['dc_365','dc_180','dc365_elo50']:
        s=data['football']['test']['all_matches'];m=data['football']['paired_test_uncertainty'][name]['blocks_28_days']
        add('football_EPL',name,'test_2024_25_2025_26',s[name]['n'],s['dc_equal']['log_loss'],s[name]['log_loss'],
            m['percentile_95_interval'],'natural-log multiclass log loss',
            ('EPL validation-selected winner' if name==data['football']['selection']['selected_model'] else 'fixed unselected comparator')+'; season-stratified 28-day CI')
    for league, detail in data['football_transfer']['leagues'].items():
        s=detail['overall']; m=s['models']
        add('football_'+league,'dc365_elo50','fixed_out_of_league_2024_25_2025_26',s['n'],m['dc_equal']['log_loss'],m['dc365_elo50']['log_loss'],
            s['paired_28_day_interval']['percentile_95_interval'],'natural-log multiclass log loss','unchanged EPL-selected model; season-stratified 28-day CI')
    s=data['football_transfer']['pooled_1520_match_transfer'];m=s['models']
    add('football_SP1_I1_pooled','dc365_elo50','fixed_out_of_league_2024_25_2025_26',s['n'],m['dc_equal']['log_loss'],m['dc365_elo50']['log_loss'],
        s['paired_28_day_interval']['percentile_95_interval'],'natural-log multiclass log loss','same-task descriptive pool; four league-season strata, 28-day CI')
    assert data['live_recent']['status']=='completed'
    s=data['live_recent']['aggregate']
    add('live_next_eligible_lap','ridge_correction','held_aside_2026_R10_R13',s['events'],s['baseline_event_mae_seconds'],s['candidate_event_mae_seconds'],
        s['ci95_delta_seconds'],'seconds MAE','unchanged 2022-2023 coefficients; four-event historical transfer, underpowered event CI')
    OUT.mkdir(parents=True,exist_ok=True)
    csv_path=OUT/'performance_research_20260907_scorecard.csv'
    with csv_path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    artifact={'schema_version':'performance_research_scorecard_v1','baseline_commit':'9fc781c5f607eb8bba2d5e951a6ff42552d7ebb8',
      'branch':'codex/performance-research-20260907','production_changes':False,'promotion':False,
      'rows':rows,'selected_mechanisms':{'live':data['live']['selected_models']['preferred_mechanism_by_2023_only'],
      'football':data['football']['selection']['selected_model'],'race':data['race']['preferred_mechanism'],
      'pre_event':'all four fixed mechanisms, no held-forward selection'},
      'sources':{k:{'path':p,'sha256':hashlib.sha256((ROOT/p).read_bytes()).hexdigest()} for k,p in SOURCES.items()},
      'scorecard_sha256':hashlib.sha256(csv_path.read_bytes()).hexdigest(),
      'interpretation':'Per-lane baselines, units and populations differ. Do not aggregate relative gains across tasks. Research intervals are conditional historical comparisons, not prospective or profit claims.'}
    (OUT/'performance_research_20260907_summary.json').write_text(json.dumps(artifact,indent=2,allow_nan=False)+'\n')
    print(f'Wrote {len(rows)} exact source-bound comparisons')


if __name__=='__main__':main()
