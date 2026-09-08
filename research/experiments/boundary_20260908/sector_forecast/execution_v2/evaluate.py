"""Strict common-cohort scoring for the frozen sector experiment."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

METRICS = ("checkpoint_event_mae", "target_balanced_event_mae")
MATCHED = "matched_recorded_future_eligible_lap"
UNMATCHED = {"unmatched_in_terminal_recorded_archive", "unresolved_in_incomplete_archive"}


def validate(frame, predictions):
    if frame.empty or frame["ledger_id"].isna().any() or frame["ledger_id"].duplicated().any():
        raise ValueError("Require a nonempty cohort with unique issuance identities")
    if not all(re.fullmatch(r"20\d{4}", str(v)) for v in frame["event_key"]):
        raise ValueError("Invalid event identity")
    if frame["driver"].isna().any() or not frame["sector"].isin([1, 2]).all():
        raise ValueError("Invalid driver or sector")
    clocks = frame["checkpoint_ms"].to_numpy(float)
    if not np.isfinite(clocks).all() or (clocks < 0).any():
        raise ValueError("Invalid issuance clock")
    y = frame["y_true"].to_numpy(float)
    matched = frame["outcome_status"].eq(MATCHED).to_numpy()
    if not frame["outcome_status"].isin({MATCHED, *UNMATCHED}).all():
        raise ValueError("Unknown issued-outcome status")
    if np.isinf(y).any() or not np.array_equal(np.isfinite(y), matched) or (y[matched] <= 0).any():
        raise ValueError("Finite positive labels only for explicit matched outcomes; NaN only for unmatched")
    if not matched.any():
        raise ValueError("No resolved targets")
    if frame.loc[matched, "target_id"].isna().any() or frame.loc[~matched, "target_id"].notna().any():
        raise ValueError("Target identity must agree with outcome status")
    times = frame.loc[matched, "target_time_seconds"].to_numpy(float)
    if not np.isfinite(times).all() or not (times > clocks[matched] / 1000).all():
        raise ValueError("Matched target must be strictly later than issuance")
    for _, group in frame.loc[matched].groupby(["event_key", "target_id"], sort=False):
        if any(group[name].nunique(dropna=False) != 1 for name in ["driver", "y_true", "target_time_seconds"]):
            raise ValueError("Inconsistent driver, value or time for a repeated target")
    # One recorded target time for one driver cannot acquire stage-dependent IDs.
    if (frame.loc[matched].groupby(["event_key", "driver", "target_time_seconds"])["target_id"].nunique() > 1).any():
        raise ValueError("The same target was split across different identities")
    for prediction in predictions:
        values = frame[prediction].to_numpy(float)
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError(f"Every issued row requires a finite positive forecast: {prediction}")
    return matched


def summarize(frame, predictions):
    matched = validate(frame, predictions)
    resolved = frame.loc[matched]
    result = {
        "issued": len(frame), "resolved_issued": int(matched.sum()),
        "unmatched_issued": int((~matched).sum()),
        "issued_events": sorted(frame["event_key"].unique().tolist()),
        "resolved_events": sorted(resolved["event_key"].unique().tolist()),
        "resolved_event_targets": len(resolved.groupby(["event_key", "target_id"])),
        "per_event_outcomes": {}, "models": {},
        "scope": "Point errors conditional on a later recorded eligible target existing; all issued unmatched outcomes are retained.",
    }
    for event, group in frame.groupby("event_key"):
        result["per_event_outcomes"][str(event)] = {
            "issued": len(group), "resolved": int(group["outcome_status"].eq(MATCHED).sum()),
            "unmatched": int(group["outcome_status"].ne(MATCHED).sum()),
            "resolved_targets": int(group["target_id"].nunique()),
        }
    for prediction in predictions:
        errors = resolved[["event_key", "target_id"]].copy()
        errors["error"] = np.abs(resolved[prediction].to_numpy(float) - resolved["y_true"].to_numpy(float))
        point = errors.groupby("event_key")["error"].mean()
        target = errors.groupby(["event_key", "target_id"])["error"].mean().groupby("event_key").mean()
        result["models"][prediction] = {
            METRICS[0]: float(point.mean()), METRICS[1]: float(target.mean()),
            "per_event": {
                METRICS[0]: {str(k): float(v) for k, v in point.items()},
                METRICS[1]: {str(k): float(v) for k, v in target.items()},
            },
        }
    return result


def uncertainty(differences, keys, resamples=20000, seed=20260908):
    differences = np.asarray(differences, dtype=float)
    if len(keys) < 2 or len(keys) != len(differences) or not np.isfinite(differences).all():
        raise ValueError("Require finite paired differences for at least two events")
    rng = np.random.default_rng(seed)
    ordinary = differences[rng.integers(0, len(keys), (resamples, len(keys)))].mean(1)
    blocks = np.zeros(resamples)
    for year in sorted({str(k)[:4] for k in keys}):
        local = differences[[i for i, k in enumerate(keys) if str(k).startswith(year)]]
        starts = rng.integers(0, len(local), (resamples, int(np.ceil(len(local) / 3))))
        draws = ((starts[..., None] + np.arange(3)) % len(local)).reshape(resamples, -1)[:, :len(local)]
        blocks += len(local) / len(keys) * local[draws].mean(1)
    omissions = (differences.sum() - differences) / (len(differences) - 1)
    return {
        "difference_candidate_minus_reference": float(differences.mean()),
        "event_percentile_95_interval": np.quantile(ordinary, [.025, .975]).tolist(),
        "three_event_percentile_95_interval": np.quantile(blocks, [.025, .975]).tolist(),
        "leave_one_event_out_max_difference": float(omissions.max()),
        "events_won": int((differences < 0).sum()), "events_lost": int((differences > 0).sum()),
        "events_tied": int((differences == 0).sum()), "resamples": resamples, "seed": seed,
    }


def compare(summary, candidate, references, resamples=20000, seed=20260908):
    result = {}
    for reference in references:
        result[reference] = {}
        for metric in METRICS:
            c, r = summary["models"][candidate], summary["models"][reference]
            keys = sorted(c["per_event"][metric])
            if keys != sorted(r["per_event"][metric]):
                raise ValueError("Candidate and reference event cohorts differ")
            differences = [c["per_event"][metric][k] - r["per_event"][metric][k] for k in keys]
            result[reference][metric] = {
                "candidate": c[metric], "reference": r[metric],
                "relative_gain_fraction": 1 - c[metric] / r[metric] if r[metric] > 0 else None,
                **uncertainty(differences, keys, resamples=resamples, seed=seed),
            }
    return result


def selection_report(frame, candidates, references, spec):
    frame = frame.copy()
    frame["event_key"] = frame["event_key"].map(str)
    expected = sorted(str(k) for k in spec["discovery_events"] if int(k) // 100 in spec["split"]["selection_years"])
    if sorted(frame["event_key"].unique().tolist()) != expected:
        raise ValueError("Selection cohort differs from the exact declared discovery events")
    names = references + candidates
    summary = summarize(frame, names)
    chosen = min(candidates, key=lambda name: (summary["models"][name][METRICS[1]], candidates.index(name)))
    settings = spec["evaluation"]["uncertainty"]
    comparisons = {c: compare(summary, c, references, settings["resamples"], settings["seed"]) for c in candidates}
    checks = {"complete_selection_event_coverage": summary["issued_events"] == summary["resolved_events"]
              and len(summary["resolved_events"]) == spec["split"]["required_selection_resolved_events"]}
    for reference, by_metric in comparisons[chosen].items():
        point, target = by_metric[METRICS[0]], by_metric[METRICS[1]]
        gain = target["relative_gain_fraction"]
        checks[reference] = (gain is not None and gain >= spec["selection"]["gain_fraction_vs_every_reference"]
                             and target["three_event_percentile_95_interval"][1] < 0
                             and point["candidate"] <= point["reference"])
    return {"summary": summary, "comparisons": comparisons, "selected": chosen,
            "selection_checks": checks, "selection_passed": all(checks.values()),
            "subgroups": {
                str(sector): summarize(frame.loc[frame["sector"] == sector], names)
                for sector in [1, 2]
            }, "new_substantial_gain_established": False}
