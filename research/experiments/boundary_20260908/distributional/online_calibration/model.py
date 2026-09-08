"""Delayed quantile-feedback offsets; no conformal coverage guarantee."""
import numpy as np
from research.experiments.boundary_20260908.distributional.execution.model import project_offsets

LEVELS=np.array([.025,.05,.1,.25,.5,.75,.9,.95,.975])


def replay(frame,static,config):
    frame=frame.reset_index(drop=True);static=np.asarray(static,dtype=float)
    if static.shape!=(len(frame),9) or not np.isfinite(static).all() or (np.diff(static,axis=1)<0).any():raise ValueError('Invalid static quantiles')
    if not (frame.target_timestamp>frame.issued_at_timestamp).all():raise ValueError('Target must be strictly later')
    if not np.isfinite(frame[['issued_at_timestamp','target_timestamp','lap_time_seconds']].to_numpy()).all():raise ValueError('Nonfinite timing or outcome')
    eta=float(config['eta_seconds']);lam=float(config['shrink_per_second'])
    if not np.isfinite(eta) or not np.isfinite(lam) or eta<=0 or lam<=0 or not 0<eta*lam<1:
        raise ValueError('Require finite positive eta and shrinkage with eta*lambda<1')
    context=config['context']
    if context not in ['race_global','race_compound']:raise ValueError('Unknown context')
    if frame.compound.isna().any():raise ValueError('Missing issuance compound')
    output=np.full_like(static,np.nan);latest=np.full(len(frame),-np.inf);seen=np.zeros(len(frame),int)
    projected=0;updates=0;max_offset=0.;resolved_count=0
    for event,rows in frame.groupby('event_key',sort=True):
        actions={};keys={}
        for i,row in rows.iterrows():
            keys[i]='all' if context=='race_global' else str(row.compound)
            actions.setdefault(float(row.issued_at_timestamp),[[],[]])[0].append(i)
            actions.setdefault(float(row.target_timestamp),[[],[]])[1].append(i)
        state={};last={};counts={}
        for timestamp,(issues,resolved) in sorted(actions.items()):
            # Every issuance in an equal-time batch precedes all label arrivals.
            for i in issues:
                key=keys[i];theta=state.get(key,np.zeros(9))
                raw=static[i]-static[i,4]+theta
                coherent=project_offsets(raw[None,:])[0]
                q=coherent+static[i,4]
                projected+=int(not np.array_equal(raw,coherent))
                output[i]=q;latest[i]=last.get(key,-np.inf);seen[i]=counts.get(key,0)
                assert latest[i]<timestamp
            gradients={}
            for i in resolved:
                if not np.isfinite(output[i]).all():raise ValueError('Label resolved before issuance')
                gradient=LEVELS-(float(frame.at[i,'lap_time_seconds'])<=output[i]);gradient[4]=0.
                gradients.setdefault(keys[i],[]).append(gradient)
                resolved_count+=1
            for key,grad in gradients.items():
                state[key]=(1-eta*lam)*state.get(key,np.zeros(9))+eta*np.mean(grad,axis=0)
                state[key][4]=0.;last[key]=timestamp;counts[key]=counts.get(key,0)+len(grad)
                max_offset=max(max_offset,float(np.abs(state[key]).max()));updates+=1
    assert resolved_count==len(frame) and np.isfinite(output).all() and not (np.diff(output,axis=1)<0).any()
    np.testing.assert_array_equal(output[:,4],static[:,4])
    return output,{'latest_feedback_timestamp':latest,'resolved_outcomes_seen':seen}, {'issuances':len(frame),'resolved_outcomes':resolved_count,'state_update_batches':updates,'projected_rows':projected,'max_absolute_offset_seconds':max_offset,'point_max_absolute_difference_seconds':0.}
