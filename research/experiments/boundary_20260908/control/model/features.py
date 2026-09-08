"""Causal prefix-only global flag features; timing uses recorded session seconds."""
from __future__ import annotations

import numpy as np
import pandas as pd

KNOWN = ("1", "2", "4", "5", "6", "7")
FAMILIES = {"yellow": ("2",), "sc": ("4",), "red": ("5",),
            "vsc": ("6", "7"), "neutralized": ("4", "5", "6", "7")}


def status_records(stream):
    """Ignore missing-field updates, preserving explicit unknown and source order."""
    if stream.empty:
        return np.array([], dtype=float), []
    assert stream.Time.is_monotonic_increasing and np.isfinite(stream.Time).all()
    present = stream.status_present.eq(True)
    rows = stream.loc[present].sort_values(["Time", "sequence"], kind="mergesort")
    codes = [str(s) if pd.notna(s) else "" for s in rows.Status]
    return rows.Time.to_numpy(dtype=float), codes


def one_point(times, codes, checkpoint_time, prior_time, lag_seconds):
    if lag_seconds < 0 or not np.isfinite([checkpoint_time, prior_time, lag_seconds]).all():
        raise ValueError("finite clocks and nonnegative lag required")
    if prior_time > checkpoint_time:
        raise ValueError("prior issuance cannot be later than checkpoint")
    cutoff, prior_cutoff = checkpoint_time-lag_seconds, prior_time-lag_seconds
    n = int(np.searchsorted(times, cutoff, side="left"))
    prior_n = int(np.searchsorted(times, prior_cutoff, side="left"))
    visible_times, visible_codes = times[:n], codes[:n]
    current = visible_codes[-1] if n else ""
    prior = codes[prior_n-1] if prior_n else ""
    result = {"g_known": float(current in KNOWN), "g_prior_known": float(prior in KNOWN)}
    for status in (*KNOWN, "unknown"):
        result["g_current_"+status] = float((current == status) if status != "unknown" else current not in KNOWN)
        result["g_prior_"+status] = float((prior == status) if status != "unknown" else prior not in KNOWN)
    result["g_last_update_age"] = min(cutoff-visible_times[-1], 7200.) if n else 0.
    result["g_last_update_missing"] = float(n == 0)
    result["g_changed_from_prior"] = float(current != prior and current in KNOWN and prior in KNOWN)
    changes = [(float(t), s) for i, (t, s) in enumerate(zip(visible_times, visible_codes))
               if i == 0 or s != visible_codes[i-1]]
    result["g_last_change_age"] = min(cutoff-changes[-1][0], 7200.) if changes else 0.
    result["g_changes_after_prior"] = float(sum(t >= prior_cutoff for t, _ in changes))
    window_start = cutoff-300.
    ends = list(visible_times[1:])+[cutoff] if n else []
    nonclear_window = 0.
    for t, end, code in zip(visible_times, ends, visible_codes):
        if code in KNOWN and code != "1":
            nonclear_window += max(0., end-max(t, window_start))
    result["g_nonclear_duration300"] = nonclear_window
    for name, family in FAMILIES.items():
        entry = exit_time = completed_duration = None
        entry_left_censored = True
        old = ""
        for t, code in changes:
            if code in family and old not in family:
                entry = t
                entry_left_censored = old not in KNOWN
            elif old in family and code in KNOWN and code not in family:
                exit_time = t
                if entry is not None and not entry_left_censored:
                    completed_duration = t-entry
            old = code
        active = current in family
        prefix = "g_"+name+"_"
        result[prefix+"active"] = float(active)
        result[prefix+"entry_age"] = min(cutoff-entry, 7200.) if entry is not None else 0.
        result[prefix+"entry_missing"] = float(entry is None)
        result[prefix+"entry_left_censored"] = float(entry_left_censored)
        result[prefix+"exit_age"] = min(cutoff-exit_time, 7200.) if exit_time is not None else 0.
        result[prefix+"exit_missing"] = float(exit_time is None)
        result[prefix+"current_duration_lower_bound"] = min(cutoff-entry, 7200.) if active and entry is not None else 0.
        result[prefix+"completed_duration"] = min(completed_duration, 7200.) if completed_duration is not None else 0.
        result[prefix+"completed_duration_missing"] = float(completed_duration is None)
        for suffix, low in (("duration300", window_start), ("duration_since_prior", prior_cutoff)):
            duration = sum(max(0., end-max(t, low)) for t, end, code in zip(visible_times, ends, visible_codes)
                           if code in family)
            result[prefix+suffix] = min(duration, 7200.)
    result["control_observed_time"] = float(visible_times[-1]) if n else np.nan
    assert all(np.isfinite(v) and v >= 0 for k, v in result.items() if k.startswith("g_"))
    return result


def event_features(stream, points, lag_seconds=15):
    times, codes = status_records(stream)
    records = [one_point(times, codes, r.checkpoint_time, r.prior_time, lag_seconds)
               for r in points.itertuples()]
    return pd.DataFrame(records, index=points.index)


def activation(frame, variant):
    gate = ~frame.eligible.to_numpy(dtype=bool) & frame.g_known.eq(1).to_numpy()
    if variant["scope"] == "flag_context":
        gate &= (frame.g_current_1.eq(0) | frame.g_nonclear_duration300.gt(0)).to_numpy()
    elif variant["scope"] != "all_known_ineligible":
        raise ValueError("unknown activation scope")
    return gate
