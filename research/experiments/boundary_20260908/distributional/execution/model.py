"""Finite quantile forecasts and proper scores, preserving the supplied median."""
import os
for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ[key]="1"
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from research.experiments.frontier_20260907 import live_frontier as original


def weighted_quantile(values, weights, levels):
    values, weights, levels = map(lambda x:np.asarray(x,dtype=float),(values,weights,levels))
    if values.ndim!=1 or values.shape!=weights.shape or not len(values) or not np.isfinite(values).all() or not np.isfinite(weights).all() or np.any(weights<0) or weights.sum()<=0:
        raise ValueError("Invalid empirical distribution")
    if levels.ndim!=1 or np.any((levels<0)|(levels>1)) or not np.isfinite(levels).all():
        raise ValueError("Invalid quantile levels")
    keep=weights>0;values,weights=values[keep],weights[keep]
    order=np.argsort(values,kind="stable");values,weights=values[order],weights[order]
    cdf=np.cumsum(weights)/weights.sum();cdf[-1]=1.
    return values[np.minimum(np.searchsorted(cdf,levels,side="left"),len(values)-1)]


def project_offsets(raw):
    raw=np.asarray(raw,dtype=float)
    if raw.ndim!=2 or raw.shape[1]!=9 or not np.isfinite(raw).all():
        raise ValueError("Expected finite nine-quantile matrix")
    result=raw.copy()
    result[:,:4]=np.sort(np.minimum(raw[:,:4],0.),axis=1)
    result[:,4]=0.
    result[:,5:]=np.sort(np.maximum(raw[:,5:],0.),axis=1)
    return result


def strata(frame,cuts):
    volatility=np.maximum(frame.x_mad_3.to_numpy(),frame.x_mad_8.to_numpy())
    quartile=np.searchsorted(np.asarray(cuts),volatility,side="right")
    return (frame.wet_compound.to_numpy()>.5).astype(int)*8+(frame.stint_clean_count.to_numpy()<=3).astype(int)*4+quartile


def fit_empirical(frame,point,spec):
    residual=frame.lap_time_seconds.to_numpy()-np.asarray(point)
    weights=original.weights(frame)
    levels=spec["quantile_levels"]
    global_raw=weighted_quantile(residual,weights,levels)
    cuts=weighted_quantile(np.maximum(frame.x_mad_3,frame.x_mad_8),weights,[.25,.5,.75])
    group=strata(frame,cuts)
    cells,diagnostics={},{}
    for key in np.unique(group):
        mask=group==key;w=weights[mask]
        effective_n=float(w.sum()**2/(w@w));mix=effective_n/(effective_n+200.)
        combined=(1-mix)*weights/weights.sum()+mix*mask*weights/w.sum()
        cells[str(key)]=project_offsets(weighted_quantile(residual,combined,levels)[None,:])[0]
        diagnostics[str(key)]={"rows":int(mask.sum()),"effective_n":effective_n,"cell_mixture_weight":mix}
    return {"global":project_offsets(global_raw[None,:])[0],"cuts":cuts,"cells":cells,"cell_diagnostics":diagnostics,"fit_rows":len(frame)}


def empirical_predict(frame,point,bundle):
    point=np.asarray(point,dtype=float)
    glob=np.tile(bundle["global"],(len(frame),1))
    groups=strata(frame,bundle["cuts"])
    conditional=np.asarray([bundle["cells"].get(str(key),bundle["global"]) for key in groups])
    return {"global_empirical":glob+point[:,None],"stratified_empirical":conditional+point[:,None]}


def fit_quantiles(frame,point,features,config,spec):
    residual=frame.lap_time_seconds.to_numpy()-np.asarray(point)
    if not np.isfinite(residual).all() or not np.isfinite(frame[features].to_numpy()).all():
        raise ValueError("Nonfinite training inputs")
    models={}
    for q in spec["quantile_levels"]:
        if q==.5:continue
        learner=HistGradientBoostingRegressor(loss="quantile",quantile=q,max_leaf_nodes=config["leaves"],max_iter=config["iterations"],**spec["hgb"])
        learner.fit(frame[features],residual,sample_weight=original.weights(frame))
        models[str(q)]=learner
    return {"models":models,"features":features,"config":config,"fit_rows":len(frame)}


def quantile_predict(frame,point,bundle,spec):
    raw=np.column_stack([np.zeros(len(frame)) if q==.5 else bundle["models"][str(q)].predict(frame[bundle["features"]]) for q in spec["quantile_levels"]])
    projected=project_offsets(raw)
    diagnostics={"raw_crossing_rows":int((np.diff(raw,axis=1)<0).any(axis=1).sum()),
                 "projected_rows":int((raw!=projected).any(axis=1).sum()),"rows":len(frame)}
    result=projected+np.asarray(point)[:,None]
    np.testing.assert_array_equal(result[:,4],point)
    return result,diagnostics


def pinball_rows(y,quantiles,levels):
    y=np.asarray(y,dtype=float);q=np.asarray(quantiles,dtype=float);levels=np.asarray(levels)
    if q.shape!=(len(y),len(levels)) or not np.isfinite(y).all() or not np.isfinite(q).all() or np.any(np.diff(q,axis=1)<0):
        raise ValueError("Nonfinite or incoherent forecast")
    if levels.ndim!=1 or not np.isfinite(levels).all() or np.any((levels<0)|(levels>1)) or np.any(np.diff(levels)<=0):
        raise ValueError("Invalid ordered quantile levels")
    error=y[:,None]-q
    return np.maximum(levels*error,(levels-1)*error)


def wis_rows(y,quantiles,spec):
    return 2*np.mean(pinball_rows(y,quantiles,spec["quantile_levels"]),axis=1)


def metrics(frame,quantiles,spec):
    y=frame.lap_time_seconds.to_numpy();q=np.asarray(quantiles)
    pin=pinball_rows(y,q,spec["quantile_levels"])
    data={"event_key":frame.event_key.to_numpy(),"wis":2*pin.mean(1),"point_mae":np.abs(y-q[:,4])}
    for index,nominal in enumerate(spec["central_interval_coverages"]):
        lo,hi=q[:,3-index],q[:,5+index]
        data[f"coverage_{nominal}"]=((y>=lo)&(y<=hi)).astype(float)
        data[f"width_{nominal}"]=hi-lo
    for index,level in enumerate(spec["quantile_levels"]):data[f"pinball_{level}"]=pin[:,index]
    events=pd.DataFrame(data).groupby("event_key").mean()
    return {"rows":len(frame),"events":len(events),"wis":float(events.wis.mean()),"point_mae":float(events.point_mae.mean()),
        "coverage":{str(v):float(events[f"coverage_{v}"].mean()) for v in spec["central_interval_coverages"]},
        "mean_width_seconds":{str(v):float(events[f"width_{v}"].mean()) for v in spec["central_interval_coverages"]},
        "pinball_by_quantile":{str(v):float(events[f"pinball_{v}"].mean()) for v in spec["quantile_levels"]},
        "per_event":[{"event_key":int(key),**{k:float(v) for k,v in row.items()}} for key,row in events.iterrows()]}


def paired(frame,candidate,reference,spec):
    ev=pd.DataFrame({"event":frame.event_key.to_numpy(),"delta":wis_rows(frame.lap_time_seconds,candidate,spec)-wis_rows(frame.lap_time_seconds,reference,spec)}).groupby("event").mean()
    rng=np.random.default_rng(spec["uncertainty"]["seed"]);count=spec["uncertainty"]["resamples"]
    boot=np.zeros(count);block=np.zeros(count);strata_counts={}
    for year in sorted(set(ev.index//100)):
        delta=ev.loc[ev.index//100==year,"delta"].to_numpy();n=len(delta);weight=n/len(ev)
        boot+=weight*delta[rng.integers(n,size=(count,n))].mean(1)
        starts=rng.integers(n,size=(count,int(np.ceil(n/3))))
        indices=((starts[:,:,None]+np.arange(3))%n).reshape(count,-1)[:,:n]
        block+=weight*delta[indices].mean(1);strata_counts[str(year)]=n
    delta=ev.delta.to_numpy()
    if len(delta)<2:raise ValueError("Paired uncertainty needs at least two events")
    return {"candidate_minus_reference":float(delta.mean()),"event_ci95":np.quantile(boot,[.025,.975]).tolist(),
        "block3_ci95":np.quantile(block,[.025,.975]).tolist(),"events_improved":int((delta<0).sum()),
        "loo_min_delta":float(min((delta.sum()-v)/(len(delta)-1) for v in delta)),
        "loo_max_delta":float(max((delta.sum()-v)/(len(delta)-1) for v in delta)),"year_strata":strata_counts}
