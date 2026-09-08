"""Label-free relative pace measurement from comparable pre-qualifying laps."""
import numpy as np
import pandas as pd

COMPOUNDS = ['SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE', 'WET']


def clean_laps(raw):
    required = {'Driver', 'LapTime', 'Time', 'Compound', 'TyreLife', 'IsAccurate', 'TrackStatus', 'PitInTime', 'PitOutTime'}
    if not required.issubset(raw.columns):
        raise ValueError(f'missing required lap columns: {sorted(required-set(raw.columns))}')
    x = pd.DataFrame({'driver_id': raw.Driver.astype(str),
                      'lap_seconds': pd.to_numeric(raw.LapTime, errors='coerce'),
                      'clock': pd.to_numeric(raw.Time, errors='coerce'),
                      'age': pd.to_numeric(raw.TyreLife, errors='coerce'),
                      'compound': raw.Compound.astype(str).str.upper().replace({'INTER': 'INTERMEDIATE'}),
                      'stint': raw.get('Stint', pd.Series(0, index=raw.index))})
    accurate = raw.IsAccurate.astype(str).str.lower().isin(['true', '1', '1.0'])
    green = raw.TrackStatus.astype(str).isin(['1', '1.0'])
    deleted = raw.get('Deleted', pd.Series(False, index=raw.index)).astype(str).str.lower().isin(['true', '1', '1.0'])
    valid = (accurate & green & ~deleted & raw.PitInTime.isna() & raw.PitOutTime.isna()
             & x.lap_seconds.between(30, 240) & x.clock.notna() & x.age.between(1, 6)
             & x.compound.isin(COMPOUNDS))
    x = x.loc[valid].copy()
    if x.empty:
        return x.reset_index(drop=True)
    best = x.groupby(['driver_id', 'compound']).lap_seconds.transform('min')
    x = x.loc[x.lap_seconds.le(1.02 * best)]
    # Several quick laps, with each driver/compound contributing at most four.
    return (x.sort_values(['lap_seconds', 'clock'], kind='stable')
            .groupby(['driver_id', 'compound'], sort=False).head(4)
            .sort_values(['clock', 'driver_id'], kind='stable').reset_index(drop=True))


def match_laps(laps):
    edges = set()
    for a, lap in laps.iterrows():
        peers = laps.loc[laps.driver_id.ne(lap.driver_id) & laps.compound.eq(lap.compound)
                         & laps.clock.sub(lap.clock).abs().le(180)
                         & laps.age.sub(lap.age).abs().le(2)].copy()
        peers['separation'] = peers.clock.sub(lap.clock).abs()
        peers = peers.sort_values(['separation', 'age', 'clock'], kind='stable').drop_duplicates('driver_id')
        if len(peers) >= 2:
            edges.update(tuple(sorted((int(a), int(b)))) for b in peers.index)
    edge = np.array(sorted(edges), dtype=int).reshape(-1, 2)
    graph = {str(d): set() for d in laps.driver_id.unique()}
    own = {str(d): set() for d in laps.driver_id.unique()}
    for a, b in edge:
        da, db = str(laps.at[a, 'driver_id']), str(laps.at[b, 'driver_id'])
        graph[da].add(db); graph[db].add(da)
        own[da].add(int(a)); own[db].add(int(b))
    component = {}
    for driver in graph:
        seen, todo = set(), [driver]
        while todo:
            item = todo.pop()
            if item not in seen:
                seen.add(item); todo.extend(graph[item] - seen)
        component[driver] = len(seen)
    support = {d: {'matched_laps': len(own[d]), 'peer_drivers': len(graph[d]),
                   'component_drivers': component[d],
                   'supported': len(own[d]) >= 2 and len(graph[d]) >= 3 and component[d] >= 6}
               for d in graph}
    return edge, support


def objective(beta, x, y, w, penalty):
    r = np.abs(y - x @ beta)
    return float(np.sum(w * np.where(r <= 3, .5 * r * r, 3 * (r - 1.5))) + .5 * np.dot(penalty * beta, beta))


def fit_huber(x, y, w, penalty):
    beta = np.zeros(x.shape[1]); previous = objective(beta, x, y, w, penalty)
    for iteration in range(60):
        r = np.abs(y - x @ beta)
        weight = w * np.minimum(1, 3 / np.maximum(r, 1e-12))
        hessian = x.T @ (weight[:, None] * x) + np.diag(penalty)
        rhs = x.T @ (weight * y)
        new = np.linalg.solve(hessian, rhs)
        value = objective(new, x, y, w, penalty)
        assert value <= previous + 1e-7
        if np.max(np.abs(new - beta)) < 1e-8:
            beta = new
            break
        beta = new; previous = value
    # Recompute weights at the final coefficient, also used by conditional deletion sensitivity.
    weight = w * np.minimum(1, 3 / np.maximum(np.abs(y - x @ beta), 1e-12))
    gradient = x.T @ (w * np.clip(x @ beta - y, -3, 3)) + penalty * beta
    return beta, weight, {'iterations': iteration + 1, 'gradient_max_abs': float(np.max(np.abs(gradient))),
                          'objective': objective(beta, x, y, w, penalty)}


def predict(frame, laps, penalty_strength, blend_strength):
    if 'qualy_position' in frame:
        raise ValueError('target labels in inference frame')
    frame = frame.reset_index(drop=True)
    n = len(frame); drivers = frame.driver_id.astype(str).tolist()
    teams = sorted(frame.team_id.astype(str).unique())
    team_of = dict(zip(drivers, frame.team_id.astype(str)))
    laps = laps.loc[laps.driver_id.isin(drivers)].reset_index(drop=True)
    edge, support = match_laps(laps)
    prior = (frame.Q2_ridge_rank_residual.to_numpy(dtype=float) - 1) * 19 / max(n - 1, 1)
    default = {'matched_laps': 0, 'peer_drivers': 0, 'component_drivers': 0, 'supported': False}
    support = {d: support.get(d, default.copy()) for d in drivers}
    diagnostic = {'clean_representative_laps': len(laps), 'matched_pairs': len(edge), 'support': support}
    if not len(edge) or not any(s['supported'] for s in support.values()):
        return frame.Q2_ridge_rank_residual.to_numpy(dtype=int), {**diagnostic, 'status': 'whole_event_Q2_fallback'}
    # Anchor IQR gives seconds per 20-car-equivalent rank unit, using only pre-Q anchors.
    anchor = pd.to_numeric(frame.raw_anchor, errors='coerce')
    scale = max(float(anchor.quantile(.75) - anchor.quantile(.25)) / 9.5, .02)
    if not np.isfinite(scale):
        return frame.Q2_ridge_rank_residual.to_numpy(dtype=int), {**diagnostic, 'status': 'whole_event_Q2_fallback_bad_scale'}
    z_driver = (laps.driver_id.to_numpy()[:, None] == np.array(drivers)[None, :]).astype(float)
    z_team = (laps.driver_id.map(team_of).to_numpy()[:, None] == np.array(teams)[None, :]).astype(float)
    clock = (laps.clock.to_numpy() - laps.clock.median()) / 600
    knots = np.quantile(clock, [.25, .5, .75])
    z_clock = np.column_stack([clock, *[np.maximum(clock - knot, 0) for knot in knots]])
    z_age = np.column_stack([laps.age.to_numpy() * laps.compound.eq(c).to_numpy() for c in COMPOUNDS])
    z = np.column_stack([z_team, z_driver, z_clock, z_age])
    a, b = edge.T
    x = z[a] - z[b]
    base_by_driver = dict(zip(drivers, prior))
    lap_prior = laps.driver_id.map(base_by_driver).to_numpy()
    y = (laps.lap_seconds.to_numpy()[a] - laps.lap_seconds.to_numpy()[b]) / scale - (lap_prior[a] - lap_prior[b])
    degree = np.bincount(edge.ravel(), minlength=len(laps))
    weights = .5 * (1 / degree[a] + 1 / degree[b])
    penalty = np.array([penalty_strength] * len(teams) + [4 * penalty_strength] * n
                       + [.1] * 4 + [2 * penalty_strength] * len(COMPOUNDS))
    beta, robust_weight, fit = fit_huber(x, y, weights, penalty)
    evaluate = np.zeros((n, len(penalty)))
    for i, d in enumerate(drivers):
        evaluate[i, teams.index(team_of[d])] = 1
        evaluate[i, len(teams) + i] = 1
    innovation = evaluate @ beta
    # Conditional deletion, with robust weights frozen. This is a sensitivity diagnostic,
    # not an independent-sample SE or calibrated posterior interval.
    h = x.T @ (robust_weight[:, None] * x) + np.diag(penalty)
    rhs = x.T @ (robust_weight * y)
    fixed_beta = np.linalg.solve(h, rhs)
    sensitivity = np.zeros(n)
    for lap in np.unique(edge):
        take = (a == lap) | (b == lap)
        reduced_h = h - x[take].T @ (robust_weight[take, None] * x[take])
        reduced_rhs = rhs - x[take].T @ (robust_weight[take] * y[take])
        deleted = np.linalg.solve(reduced_h, reduced_rhs)
        sensitivity = np.maximum(sensitivity, np.abs(evaluate @ (deleted - fixed_beta)))
    reliability = np.array([min(support[d]['matched_laps'] / 3, 1) * min(support[d]['peer_drivers'] / 5, 1)
                            if support[d]['supported'] else 0 for d in drivers])
    reliability /= 1 + (sensitivity / 2) ** 2
    adjustment = blend_strength * reliability * np.clip(innovation, -5, 5)
    score = prior + adjustment
    order = np.lexsort((frame.Q2_ridge_rank_residual.to_numpy(), score))
    ranks = np.empty(n, dtype=int); ranks[order] = np.arange(1, n + 1)
    diagnostic.update({'status': 'fitted', 'seconds_per_equivalent_position': scale,
                       'clock_knots': knots.tolist(), 'teams': teams, 'drivers': drivers,
                       'coefficients': beta.tolist(), **fit,
                       'innovation_equivalent_positions': innovation.tolist(),
                       'conditional_leave_lap_out_sensitivity': sensitivity.tolist(),
                       'reliability_multiplier': reliability.tolist(),
                       'applied_adjustment_equivalent_positions': adjustment.tolist()})
    return ranks, diagnostic
