"""Build a hash-bound scorecard and research figure from frozen results."""
from pathlib import Path
import csv
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
ART=ROOT/'artifacts/research/frontier_20260907/live'
DOC=ROOT/'docs/research/evidence'
ASSETS=ROOT/'docs/research/assets'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    sources={'selection':ART/'selection.json','canonical_results':ART/'corrected_input_contract/results.json',
             'verification':ART/'corrected_input_contract/verification.json','attribution':ART/'attribution/results.json',
             'input_recovery':ART.parent/'input_recovery/manifest.json','candidate':ART/'candidate/manifest.json'}
    values={k:json.loads(p.read_text()) for k,p in sources.items()};result=values['canonical_results'];selection=values['selection']
    rows=[]
    def add(name,period,r,role):
        rows.append({'candidate':name,'period':period,'role':role,'events':r['events'],'matched_rows':r['rows'],
                     'baseline_mae_seconds':r['baseline_mae'],'candidate_mae_seconds':r['candidate_mae'],
                     'relative_reduction_percent':100*r['relative_reduction'],'delta_seconds':r['delta'],
                     'block3_ci95_low':r['block3_ci95'][0],'block3_ci95_high':r['block3_ci95'][1],
                     'events_improved':r['events_improved']})
    for name,r in selection['all_candidate_selection_metrics'].items():add(name,'2023',r,'candidate selection only')
    for period,models in result['results'].items():
        for name,r in models.items():add(name,period,r,'fixed preferred model' if name==result['preferred'] else 'fixed family comparator')
    for name,periods in values['attribution']['results'].items():
        for period,r in periods.items():add(name,period,r,'post-discovery attribution; no reselection')
    DOC.mkdir(exist_ok=True);ASSETS.mkdir(exist_ok=True)
    scorecard=DOC/'frontier_live_performance_20260907_scorecard.csv'
    with scorecard.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    payload={'model_id':values['candidate']['model_id'],'preferred':result['preferred'],'selection_candidates':len(selection['all_candidate_selection_metrics']),
             'substantial_research_gate_passed':result['substantial_research_gate_passed'],'gate':result['gate'],
             'candidate_model_sha256':values['candidate']['model_pickle_sha256'],'sources':{k:{'path':str(p.relative_to(ROOT)),'sha256':sha(p)} for k,p in sources.items()},
             'scorecard_sha256':sha(scorecard),'rows':len(rows),'production_changed':False,'promotion':False,
             'limits':'Amended retrospective evaluation after recovery of observed input fields; historical dates previously exposed. No prospective or strategy-value claim.'}
    (DOC/'frontier_live_performance_20260907_summary.json').write_text(json.dumps(payload,indent=2)+'\n')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,(left,right)=plt.subplots(1,2,figsize=(11.6,4.8),gridspec_kw={'width_ratios':[1.12,1]})
    for period,color,label in [('2024_2025','#3466ad','2024–25: 46/48 races improve'),('2026','#008573','2026: 13/13 races improve')]:
        ev=result['results'][period][result['preferred']]['per_event']
        left.scatter([e['baseline_mae'] for e in ev],[e['candidate_mae'] for e in ev],s=32,color=color,label=label,alpha=.8)
    all_events=[e for p in ['2024_2025','2026'] for e in result['results'][p][result['preferred']]['per_event']]
    limit=1.06*max(max(e['baseline_mae'],e['candidate_mae']) for e in all_events)
    left.plot([0,limit],[0,limit],color='#9b9b9b',ls='--',lw=1,label='Equal error')
    left.set_xlim(0,limit);left.set_ylim(0,limit)
    left.set(xlabel='Naive event MAE (seconds)',ylabel='Frozen challenger event MAE (seconds)',title='Paired results on 61 races')
    left.legend(loc='upper left',frameon=False,fontsize=8);left.grid(alpha=.12)
    periods=['2024','2025','2026'];y=np.arange(3)
    baseline=[result['results'][p][result['preferred']]['baseline_mae'] for p in periods]
    candidate=[result['results'][p][result['preferred']]['candidate_mae'] for p in periods]
    right.barh(y+.15,baseline,height=.28,color='#bdc5d0',label='Naive baseline')
    right.barh(y-.15,candidate,height=.28,color='#3466ad',label='Frozen boosting')
    for i,p in enumerate(periods):right.text(baseline[i]+.015,y[i],f"{100*result['results'][p][result['preferred']]['relative_reduction']:.2f}% lower",va='center',fontsize=10)
    right.set(yticks=y,yticklabels=['2024 · 24 races','2025 · 24 races','2026 · 13 races'],xlabel='Mean event MAE (seconds)',title='Improvement persists across seasons',xlim=(0,.8))
    right.legend(frameon=False,fontsize=8,loc='lower right');right.grid(axis='x',alpha=.12)
    fig.suptitle('A frozen nonlinear lap correction reduces error by about 10%',fontsize=15,fontweight='bold',y=1.01)
    fig.text(.5,-.01,'Next eligible clean lap • Identical target pairs • Parameters selected on 2023 and fitted on 2022–23 • Retrospective evidence',ha='center',fontsize=8,color='#444')
    fig.tight_layout();fig.savefig(ASSETS/'frontier_live_performance_20260907.png',dpi=180,bbox_inches='tight');fig.savefig(ASSETS/'frontier_live_performance_20260907.pdf',bbox_inches='tight');plt.close(fig)
    print(json.dumps({'scorecard_rows':len(rows),'selected_candidates':len(selection['all_candidate_selection_metrics']),'sources':len(sources)},indent=2))


if __name__=='__main__':main()
