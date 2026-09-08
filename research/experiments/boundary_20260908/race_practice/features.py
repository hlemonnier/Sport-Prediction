"""Completed pre-qualifying practice measurements, independent of race labels."""
import numpy as np
import pandas as pd

NAMES = ["longrun_delta_pct", "longrun_rank_gap", "longrun_log_support",
         "longrun_missing", "longrun_mad_pct", "raw_slope_pct_per_age",
         "longrun_session_count", "bestlap_rank_gap", "bestlap_missing",
         "team_longrun_rank_gap"]


def session_measurements(raw):
    required = {"Driver", "Time", "LapTime", "Compound", "Stint", "TyreLife",
                "IsAccurate", "TrackStatus", "PitInTime", "PitOutTime"}
    if not required.issubset(raw.columns):
        raise ValueError(f"missing columns: {sorted(required - set(raw.columns))}")
    x = pd.DataFrame({"driver": raw.Driver.astype(str), "time": pd.to_numeric(raw.Time, errors="coerce"),
                      "lap": pd.to_numeric(raw.LapTime, errors="coerce"), "compound": raw.Compound.astype(str).str.upper(),
                      "stint": pd.to_numeric(raw.Stint, errors="coerce"), "age": pd.to_numeric(raw.TyreLife, errors="coerce")})
    accurate = raw.IsAccurate.astype(str).str.lower().isin(["true", "1", "1.0"])
    clear = raw.TrackStatus.astype(str).isin(["1", "1.0"])
    good = (accurate & clear & raw.PitInTime.isna() & raw.PitOutTime.isna()
            & x.lap.between(30, 240) & np.isfinite(x.time) & np.isfinite(x.age)
            & x.age.ge(1) & np.isfinite(x.stint)
            & x.compound.isin(["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]))
    x = x.loc[good].sort_values(["time", "driver", "lap"], kind="stable").reset_index(drop=True)
    if x.empty:
        return {}, {"clean_laps": 0, "longrun_laps": 0, "matched_laps": 0}
    scale = float(x.lap.median())
    best = x.groupby("driver").lap.min()
    best_ranks = (best.rank(method="average") - 1) / max(len(best) - 1, 1)
    result = {d: {"delta": [], "slope": [], "best_rank": float(best_ranks[d])} for d in best.index}
    group = x.groupby(["driver", "stint", "compound"], sort=False).lap
    run = x.loc[group.transform("size").ge(5) & x.lap.le(group.transform("median") + 5)].copy()
    # A retained lap needs independent drivers, not multiple laps of one peer.
    for _, lap in run.iterrows():
        peers = run.loc[run.driver.ne(lap.driver) & run.compound.eq(lap.compound)
                        & run.time.sub(lap.time).abs().le(300) & run.age.sub(lap.age).abs().le(3)].copy()
        peers["separation"] = peers.time.sub(lap.time).abs()
        peers = peers.sort_values(["separation", "time", "driver", "lap"], kind="stable").drop_duplicates("driver")
        if len(peers) >= 3:
            result[lap.driver]["delta"].append(float(100 * (lap.lap - peers.lap.median()) / scale))
    for (driver, _, _), g in run.groupby(["driver", "stint", "compound"], sort=False):
        g = g.sort_values(["age", "time"]).drop_duplicates("age")
        if len(g) >= 5:
            a, b = np.triu_indices(len(g), 1)
            slopes = 100 * (g.lap.to_numpy()[b] - g.lap.to_numpy()[a]) / (g.age.to_numpy()[b] - g.age.to_numpy()[a]) / scale
            result[driver]["slope"].append(float(np.median(slopes)))
    return result, {"clean_laps": len(x), "longrun_laps": len(run),
                    "matched_laps": sum(len(v["delta"]) for v in result.values())}


def aggregate(ids, teams, qualifying, sessions):
    n = len(ids)
    q = (np.asarray(qualifying, float) - 1) / (n - 1)
    assert n >= 2 and len(set(ids)) == n
    pooled = {d: {"delta": [], "slope": [], "best": [], "sessions": 0} for d in ids}
    for session in sessions:
        for d in ids:
            if d not in session:
                continue
            v = session[d]
            pooled[d]["delta"].extend(v["delta"])
            pooled[d]["slope"].extend(v["slope"])
            pooled[d]["best"].append(v["best_rank"])
            pooled[d]["sessions"] += int(bool(v["delta"]))
    supported = {d: float(np.median(v["delta"])) for d, v in pooled.items() if len(v["delta"]) >= 2}
    sr = pd.Series(supported, dtype=float)
    percentiles = ((sr.rank(method="average") - 1) / max(len(sr) - 1, 1)).to_dict()
    rank_gap = np.array([percentiles[d] - q[i] if d in supported else 0. for i, d in enumerate(ids)])
    rows = []
    for i, (d, team) in enumerate(zip(ids, teams)):
        v = pooled[d]; delta = np.asarray(v["delta"], float)
        present = d in supported
        teammates = [j for j, t in enumerate(teams) if t == team and ids[j] in supported]
        rows.append([supported.get(d, 0.), rank_gap[i], np.log1p(len(delta)), float(not present),
                     float(np.median(np.abs(delta - np.median(delta)))) if present else 0.,
                     float(np.median(v["slope"])) if v["slope"] else 0., v["sessions"],
                     float(np.median(v["best"]) - q[i]) if v["best"] else 0., float(not v["best"]),
                     float(np.mean(rank_gap[teammates])) if teammates else 0.])
    output = np.asarray(rows, float)
    assert output.shape == (n, len(NAMES)) and np.isfinite(output).all()
    return output, {"drivers": n, "supported_drivers": len(supported),
                    "matched_laps": sum(len(v["delta"]) for v in pooled.values())}
