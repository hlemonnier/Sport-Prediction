"""Export and use the verified frozen research challenger on observed prefixes."""
import argparse
import json
from pathlib import Path
import pickle

import pandas as pd

from research.experiments.frontier_20260907 import live_frontier as experiment
from research.experiments.frontier_20260907.feature_encoder import observed_features, REQUIRED

OUT=experiment.OUT/'candidate'
MODEL_ID='frontier_live_hgb_l15_i150_20260907'


def export():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'model.pkl').exists():raise FileExistsError('Refusing to replace frozen challenger')
    results=experiment.OUT/'corrected_input_contract/results.json'
    verification=experiment.OUT/'corrected_input_contract/verification.json'
    result=json.loads(results.read_text());checked=json.loads(verification.read_text())
    assert checked['results_sha256']==experiment.sha(results) and checked['status']=='passed'
    assert result['substantial_research_gate_passed'] and result['preferred']=='hgb_l15_i150'
    with (experiment.OUT/'fitted_models.pkl').open('rb') as f:models=pickle.load(f)
    bundle={'model_id':MODEL_ID,'model':models[result['preferred']],'required_observed_columns':REQUIRED,
            'feature_encoder_sha256':experiment.sha(Path(__file__).with_name('feature_encoder.py')),
            'frozen_experiment_source_sha256':experiment.sha(Path(__file__).with_name('live_frontier.py')),
            'results_sha256':experiment.sha(results),'verification_sha256':experiment.sha(verification),
            'target':'same driver next eligible clean completed lap; may skip numbered laps',
            'units':'seconds','fit_years':[2022,2023],'selection_year':2023,'promotion':False}
    with (OUT/'model.pkl').open('wb') as f:pickle.dump(bundle,f)
    experiment.write(OUT/'manifest.json',{k:v for k,v in bundle.items() if k!='model'}|{'model_pickle_sha256':experiment.sha(OUT/'model.pkl')})
    print(str(OUT/'manifest.json'))


def predict(raw, event_key, bundle):
    assert experiment.sha(Path(__file__).with_name('feature_encoder.py'))==bundle['feature_encoder_sha256']
    assert experiment.sha(Path(__file__).with_name('live_frontier.py'))==bundle['frozen_experiment_source_sha256']
    issued=observed_features(raw,event_key)
    columns=experiment.KEYS+['forecast_naive_seconds','forecast_challenger_seconds','model_id']
    if issued.empty:return pd.DataFrame(columns=columns)
    out=issued[experiment.KEYS+['forecast_naive_seconds']].copy()
    out['forecast_challenger_seconds']=experiment.predict_model(issued,bundle['model'])
    out['model_id']=bundle['model_id']
    return out


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['export','predict'])
    parser.add_argument('--laps',type=Path);parser.add_argument('--event-key',type=int);parser.add_argument('--cutoff',type=float);parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.command=='export':export()
    else:
        if args.laps is None or args.event_key is None or args.output is None:parser.error('predict needs --laps, --event-key and --output')
        if args.output.exists():raise FileExistsError(args.output)
        manifest=json.loads((OUT/'manifest.json').read_text());assert experiment.sha(OUT/'model.pkl')==manifest['model_pickle_sha256']
        with (OUT/'model.pkl').open('rb') as f:bundle=pickle.load(f)
        raw=pd.read_csv(args.laps)
        if args.cutoff is not None:raw=raw.loc[raw.Time<=args.cutoff]
        prediction=predict(raw,args.event_key,bundle);args.output.parent.mkdir(parents=True,exist_ok=True);prediction.to_csv(args.output,index=False)
        print(json.dumps({'rows':len(prediction),'output':str(args.output),'model_id':MODEL_ID}))
