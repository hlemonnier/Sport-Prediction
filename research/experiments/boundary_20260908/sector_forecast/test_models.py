"""Target multiplicity and conditional-outcome accounting contracts."""
import numpy as np
import pandas as pd
import pytest

from research.experiments.boundary_20260908.sector_forecast import models


def test_training_weights_preserve_event_and_target_totals():
    frame = pd.DataFrame({"event_key":[202201]*4+[202202]*2,"target_id":["a","a","a","b","c","d"]})
    frame["weight"] = models.training_weights(frame)
    totals=frame.groupby(["event_key","target_id"])["weight"].sum()
    assert len(set(totals.round(12))) == 1
    assert np.isclose(frame.weight.mean(),1)
    assert np.isclose(frame.groupby("event_key").weight.sum().diff().dropna(),0).all()


def test_repeated_target_does_not_dominate_reported_primary_error():
    frame = pd.DataFrame({"event_key":[202201]*4,"target_id":["a","a","a","b"],
                          "y_true":[90.]*4,"candidate":[91.,91.,91.,99.]})
    metrics=models.error_summary(frame,"candidate")
    assert metrics["checkpoint_event_mae"] == 3.
    assert metrics["target_balanced_event_mae"] == 5.


def test_unmatched_issued_prediction_remains_explicit_and_unscored():
    frame = pd.DataFrame({"event_key":[202201,202201],"target_id":["a",None],
                          "y_true":[90.,np.nan],"candidate":[91.,200.]})
    metrics=models.error_summary(frame,"candidate")
    assert metrics["issued"]==2 and metrics["resolved_issued"]==1 and metrics["unmatched_issued"]==1
    assert metrics["target_balanced_event_mae"]==1.
    with pytest.raises(ValueError):models.training_weights(frame)


def test_identical_models_have_exact_zero_paired_uncertainty():
    frame=pd.DataFrame({"event_key":[202201,202202,202301,202302],"target_id":["a","b","c","d"],
                        "y_true":[90.,91.,92.,93.],"c":[91.,90.,94.,92.],"r":[91.,90.,94.,92.]})
    value=models.paired_uncertainty(frame,"c","r",resamples=100)
    assert value["difference_candidate_minus_reference"]==0.
    assert value["event_percentile_95_interval"]==value["three_event_percentile_95_interval"]==[0.,0.]


def test_model_rejects_explicit_future_target_features_before_fitting():
    frame=pd.DataFrame({"event_key":[202201],"target_id":["a"],"y_true":[90.],"base":[91.]})
    with pytest.raises(ValueError,match="Future"):
        models.fit_residual(frame,["y_true"],"base",{"max_leaf_nodes":7})
