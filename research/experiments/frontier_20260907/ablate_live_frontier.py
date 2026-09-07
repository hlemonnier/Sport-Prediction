"""Post-discovery attribution only: fixed hyperparameters, no candidate selection."""
import importlib.util
import json
from pathlib import Path
import sys

import pandas as pd

spec=importlib.util.spec_from_file_location('frontier_ablation_target',Path(__file__).with_name('live_frontier.py'))
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)


def main():
    out=m.OUT/'attribution';out.mkdir(parents=True,exist_ok=True)
    if (out/'results.json').exists():raise FileExistsError('Attribution run already completed')
    selection=json.loads((m.OUT/'selection.json').read_text());preferred=selection['preferred']
    config=selection['configurations'][preferred.removesuffix('_half')]
    all_features=selection['features']
    variants={'published_nine_features_only':m.base.FEATURES,
              'without_sector_or_speed_features':[c for c in all_features if not c.startswith(('x_Sector','x_Speed'))],
              'without_rich_peer_features':[c for c in all_features if not c.startswith('x_peer') and c!='x_relative_field_pace']}
    frozen={'purpose':'Post-discovery mechanism diagnosis only; not a new selection or independent test','selected_model':preferred,
            'configuration':config,'source_sha256':m.sha(__file__),'variants':variants,
            'canonical_results_sha256':m.sha(m.OUT/'corrected_input_contract/results.json')}
    m.write(out/'design.json',frozen)
    training=pd.read_pickle(m.OUT/'discovery_data.pkl')
    transfer=pd.read_pickle(m.OUT/'corrected_input_contract/transfer_data_and_forecasts.pkl')
    results={}
    for name,features in variants.items():
        model=m.fit_model(training,config,features);p=m.predict_model(transfer,model)
        results[name]={}
        for period,mask in [('2024_2025',transfer.year.isin([2024,2025])),('2026',transfer.year.eq(2026))]:
            results[name][period]=m.diagnostics(transfer.loc[mask],p[mask])
        print(name,{period:r['relative_reduction'] for period,r in results[name].items()},flush=True)
    m.write(out/'results.json',{'design':frozen,'results':results,'selection_changed':False,'model_changed':False})


if __name__=='__main__':main()
