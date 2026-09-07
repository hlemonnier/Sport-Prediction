import importlib.util
from pathlib import Path
import sys
from dataclasses import replace

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('race_performance_cycle', Path(__file__).with_name('run.py'))
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def synthetic_events():
    rng = np.random.default_rng(912)
    return [mod.Event(f'2023:{i+1:02d}',2023,i+1,'test_track','standard',
                      ['a','b','c','d','e','f'],['t1','t1','t2','t2','t3','t3'],
                      rng.permutation(np.arange(1,7)).astype(float),
                      rng.permutation(np.arange(1,7)).astype(float),
                      np.array([True,True,True,True,True,i%2==0])) for i in range(12)]


@pytest.mark.parametrize('kind,parameter',[('baseline',0),('shrunken_movement',1),('ridge_movement',1),('huber_movement',1),('boosted_absolute_movement',1)])
def test_predictions_do_not_use_current_or_future_outcomes(kind,parameter):
    events = synthetic_events()
    cut=8
    poisoned = [e if i<cut else replace(e,target=e.target[::-1].copy(),classified_finish=~e.classified_finish) for i,e in enumerate(events)]
    full=mod.build_panels(events); changed=mod.build_panels(poisoned); prefix=mod.build_panels(events[:cut+1])
    for i in range(cut+1):
        np.testing.assert_array_equal(full[i],changed[i])
        np.testing.assert_array_equal(full[i],prefix[i])
    x=np.vstack(full[:cut]);y=np.concatenate([(e.target-e.qualifying)/(len(e.ids)-1) for e in events[:cut]])
    w=np.concatenate([np.full(len(e.ids),1/len(e.ids)) for e in events[:cut]])
    a=mod.predict(kind,parameter,x,y,w,full[cut],events[cut].ids)
    b=mod.predict(kind,parameter,x,y,w,changed[cut],events[cut].ids)
    np.testing.assert_array_equal(a,b)
    assert sorted(a.tolist())==list(range(1,7))


def test_prediction_field_is_equivariant_to_input_order():
    values=np.array([.2,.4,.2,.1]);ids=['a','b','c','d'];p=np.array([2,0,3,1])
    expected=mod.ranks(values,ids)
    actual=mod.ranks(values[p],[ids[i] for i in p])
    np.testing.assert_array_equal(actual,expected[p])


def test_bootstrap_compares_identical_event_population():
    e=synthetic_events()[0];r=mod.event_score(e,e.qualifying)
    with pytest.raises(AssertionError):mod.summarize([r],[{**r,'event':'different'}])


def test_manifest_session_inventory_keeps_sprint_grand_prix_classifications():
    events,manifest,exclusions=mod.load_events()
    assert any(e.weekend_format=='sprint' for e in events)
    assert not any('_sprint_qualifying_results.csv' in p for p in manifest)
    assert not any('_sprint_race_results.csv' in p for p in manifest)
    assert len(events)+len(exclusions)==101
