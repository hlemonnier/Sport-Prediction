"""Synthetic mathematical contracts; never score historical forecast targets."""
from copy import deepcopy

import numpy as np
import pytest

from research.experiments.boundary_20260908.sector_forecast import baselines as b


def record(sectors, residual=0.):
    return {"sectors_seconds": list(sectors), "full_lap_seconds": float(sum(sectors)+residual)}


def history():
    return [record([30+.2*i, 40+.1*i, 25+.15*i]) for i in range(12)]


def test_common_minimum_three_and_observed_prefix_only():
    with pytest.raises(ValueError, match="three"):
        b.predict_baselines(history()[:2], [30], known_asof_contamination=False)
    for prefix in ([], [30,40,25]):
        with pytest.raises(ValueError, match="one or two"):
            b.predict_baselines(history(), prefix, known_asof_contamination=False)
    for flag in (None, 0, 1, "False", np.bool_(True)):
        with pytest.raises(ValueError, match="boolean"):
            b.predict_baselines(history(), [30], known_asof_contamination=flag)


@pytest.mark.parametrize("prefix", [[31.], [31.,41.]])
def test_four_references_reproduce_complete_joint_template_equations(prefix):
    h = history();k=len(prefix);total=sum(prefix)
    remainder=np.array([r["full_lap_seconds"]-sum(r["sectors_seconds"][:k]) for r in h])
    points=b.predict_baselines(h,prefix,known_asof_contamination=False)["points"]
    assert set(points)==set(b.REFERENCE_NAMES)|{b.GAUSSIAN_NAME}
    assert points[b.REFERENCE_NAMES[0]]==total+remainder[-1]
    assert points[b.REFERENCE_NAMES[1]]==total+np.median(remainder[-5:])
    scaled=[np.clip(total/sum(r["sectors_seconds"][:k]),.97,1.03)*v for r,v in zip(h[-5:],remainder[-5:])]
    assert points[b.PACE_REFERENCE]==total+np.median(scaled)


def test_joint_remainder_median_is_not_sum_of_marginal_sector_medians():
    h=[record([30,1,101]),record([30,101,1]),record([30,101,101])]
    point=b.predict_baselines(h,[30],known_asof_contamination=False)["points"][b.REFERENCE_NAMES[1]]
    assert point==132.
    incorrect=30+np.median([1,101,101])+np.median([101,1,101])
    assert incorrect==232. and point!=incorrect


def test_alpha_half_weighted_median_degenerates_to_latest_template():
    values=np.array([70,20,80,30,90,40,100,50,110,60.])
    ages=np.arange(9,-1,-1)
    weights=.5*.5**ages
    assert weights[-1]/weights.sum()>.5
    assert b.weighted_median(values,weights)==values[-1]
    values[-1]=1000.
    assert b.weighted_median(values,weights)==1000.
    corrected=.2*.8**ages
    assert corrected[-1]/corrected.sum()<.5
    assert b.weighted_median(values,corrected)!=values[-1]
    assert b.RECENCY_ALPHA==.2


def test_weighted_median_minimizes_weighted_absolute_loss():
    values=np.array([11.,2.,8.,20.]);weights=np.array([.1,.2,.4,.3])
    median=b.weighted_median(values,weights)
    assert median==8.
    minimum=np.dot(weights,abs(values-median))
    assert all(np.dot(weights,abs(values-v))>=minimum-1e-12 for v in np.linspace(0,25,251))
    assert b.weighted_median([1,4],[1,1])==1.  # Frozen lower-median tie rule.
    assert b.weighted_median([1,4],[1e308,1e308])==1.


@pytest.mark.parametrize("values,weights", [([],[]),([1],[0]),([1],[-1]),([np.nan],[1]),([1],[np.inf]),([1,2],[1])])
def test_invalid_weighted_median_inputs_fail(values,weights):
    with pytest.raises(ValueError):b.weighted_median(values,weights)


def test_known_contamination_uses_whole_laps_without_observed_prefix():
    h=history();frozen=deepcopy(h)
    a=b.predict_baselines(h,[30],known_asof_contamination=True)
    z=b.predict_baselines(h,[200,300],known_asof_contamination=True)
    assert a["points"]==z["points"]
    assert a["points"][b.REFERENCE_NAMES[0]]==h[-1]["full_lap_seconds"]
    assert a["points"][b.PACE_REFERENCE]==np.median([r["full_lap_seconds"] for r in h[-5:]])
    assert a["points"][b.GAUSSIAN_NAME]==a["points"][b.PACE_REFERENCE]
    assert a["diagnostics"]["gaussian"]["status"]=="known_contamination_whole_lap_median5"
    assert h==frozen


@pytest.mark.parametrize("prefix", [[31.2],[31.2,40.7]])
def test_gaussian_matches_independent_schur_conditional_remainder(prefix):
    h=history();k=len(prefix)
    x=np.asarray([r["sectors_seconds"][:k] for r in h[-10:]])
    y=np.asarray([r["full_lap_seconds"] for r in h[-10:]])-x.sum(1)
    joint=np.column_stack((x,y));mean=joint.mean(0)
    raw=np.cov(joint,rowvar=False,ddof=1)
    sigma=.5*raw+.5*np.diag(np.diag(raw))+.01*np.eye(k+1)
    assert np.linalg.eigvalsh(sigma).min()>=.01-1e-14
    gain=sigma[-1,:k]@np.linalg.inv(sigma[:k,:k])
    expected=sum(prefix)+mean[-1]+gain@(np.array(prefix)-mean[:k])
    actual=b.predict_baselines(h,prefix,known_asof_contamination=False)
    assert actual["points"][b.GAUSSIAN_NAME]==pytest.approx(expected,abs=1e-12)
    assert actual["diagnostics"]["gaussian"]["status"]=="available"


def test_gaussian_keeps_known_prefix_coefficient_one_when_remainder_constant():
    h=[record([30+i,40,25]) for i in range(10)]
    result=b.predict_baselines(h,[39.5],known_asof_contamination=False)
    assert result["points"][b.GAUSSIAN_NAME]==39.5+65.
    # Shrinking the covariance of (S1, fullLapTime) would incorrectly halve
    # the deterministic known-prefix response despite zero remainder variance.


def test_gaussian_diagonal_floor_handles_constant_and_rank_deficient_history():
    h=[record([30,40,25]) for _ in range(3)]
    for prefix in ([31.],[31.,42.]):
        result=b.predict_baselines(h,prefix,known_asof_contamination=False)
        assert result["diagnostics"]["gaussian"]["status"]=="available"
        assert result["points"][b.GAUSSIAN_NAME]==sum(prefix)+(65 if len(prefix)==1 else 25)


def test_nonpositive_gaussian_remainder_falls_back_without_dropping_forecast():
    h=[record([float(i),21.-2*i,1.]) for i in range(1,11)]
    result=b.predict_baselines(h,[100.],known_asof_contamination=False)
    assert result["diagnostics"]["gaussian"]["status"]=="fallback_nonpositive_conditional_remainder"
    assert result["points"][b.GAUSSIAN_NAME]==result["points"][b.PACE_REFERENCE]


def test_gaussian_solve_failure_falls_back_with_explicit_reason(monkeypatch):
    def fail(*args):raise np.linalg.LinAlgError("synthetic failure")
    monkeypatch.setattr(np.linalg,"solve",fail)
    result=b.predict_baselines(history(),[31],known_asof_contamination=False)
    assert result["diagnostics"]["gaussian"]["status"]=="fallback_linear_solve_failed"
    assert result["points"][b.GAUSSIAN_NAME]==result["points"][b.PACE_REFERENCE]


def test_full_lap_remainder_respects_received_rounding_residual():
    h=[record([30,40,25],residual=.002) for _ in range(3)]
    result=b.predict_baselines(h,[31,41],known_asof_contamination=False)
    assert result["points"][b.REFERENCE_NAMES[1]]==pytest.approx(97.002)
    h[0]["full_lap_seconds"]+=.002
    with pytest.raises(ValueError,match="coherence"):
        b.predict_baselines(h,[31],known_asof_contamination=False)


@pytest.mark.parametrize("bad", [True,np.nan,np.inf,0.,-1.,"30"])
def test_nonpositive_nonfinite_or_coerced_durations_are_rejected(bad):
    with pytest.raises(ValueError):
        b.predict_baselines(history(),[bad],known_asof_contamination=False)
    h=history();h[-1]["sectors_seconds"][0]=bad
    with pytest.raises(ValueError):
        b.predict_baselines(h,[31],known_asof_contamination=False)


def test_later_sector_and_target_metadata_are_not_read_or_mutated():
    class Poison(dict):
        def __getitem__(self,key):
            if key not in ("sectors_seconds","full_lap_seconds"):
                raise AssertionError("Target or future metadata was read")
            return super().__getitem__(key)
    h=history();prefix=[31.]
    expected=b.predict_baselines(h,prefix,known_asof_contamination=False)
    augmented=[Poison({**r,"future_target":-999,"final_IsAccurate":False,"next_compound":"WET"}) for r in h]
    assert b.predict_baselines(augmented,prefix,known_asof_contamination=False)==expected
    assert prefix==[31.] and h==history()


def test_all_estimators_ignore_history_older_than_their_fixed_window():
    h=history();a=b.predict_baselines(h,[31],known_asof_contamination=False)
    h[0]=record([100,100,100]);h[1]=record([200,200,200])
    z=b.predict_baselines(h,[31],known_asof_contamination=False)
    assert a==z
