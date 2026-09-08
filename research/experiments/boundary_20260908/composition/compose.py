"""Join two frozen point models by observed checkpoint regime, with no fitting."""
import numpy as np
import pandas as pd

CP_KEYS = ["event_key", "driver_id", "checkpoint_lap", "checkpoint_time"]
TARGETS = ["target_lap", "target_time", "target_seconds"]
ANCHOR_RENAME = {
    "issued_after_lap_number": "checkpoint_lap",
    "issued_at_timestamp": "checkpoint_time",
    "target_lap_number": "target_lap",
    "target_timestamp": "target_time",
    "lap_time_seconds": "target_seconds",
}


def compose(checkpoints, checkpoint_column, anchor=None, anchor_column=None):
    """Replace eligible observations only; retain every ineligible expert point.

    Branch choices are made by the external frozen 2023 selection lock, not by
    this function or any evaluation labels. Labels enter only exact-pair checks.
    """
    frame = checkpoints.copy()
    frame["driver_id"] = frame.driver_id.astype(str)
    if frame.duplicated(CP_KEYS).any():
        raise ValueError("Duplicate checkpoint identity")
    if not frame.eligible.isin([True, False]).all():
        raise ValueError("Observed eligibility must be a boolean")
    eligible = frame.eligible.to_numpy(bool)
    values = frame[checkpoint_column].to_numpy(float).copy()
    reference = frame.reference.to_numpy(float)
    if (not np.isfinite(values).all() or not np.isfinite(reference).all()
            or (values <= 0).any() or (reference <= 0).any()):
        raise ValueError("Component forecast must be finite positive seconds")
    if not np.array_equal(values[eligible], reference[eligible]):
        raise ValueError("Checkpoint branch must retain original eligible baseline")
    if not (frame.target_time > frame.checkpoint_time).all():
        raise ValueError("Targets must be strictly later than checkpoint")
    if anchor is None:
        return values
    if not anchor_column:
        raise ValueError("Anchor column required")
    source = anchor.rename(columns=ANCHOR_RENAME).copy()
    source["driver_id"] = source.driver_id.astype(str)
    if source.duplicated(CP_KEYS).any():
        raise ValueError("Duplicate anchor identity")
    expected = frame.loc[eligible, CP_KEYS+TARGETS].copy()
    expected["_order"] = np.flatnonzero(eligible)
    merged = expected.merge(source[CP_KEYS+TARGETS+[anchor_column]], on=CP_KEYS,
        how="outer", validate="one_to_one", suffixes=("", "_anchor"), indicator=True)
    if not merged._merge.eq("both").all() or len(merged) != int(eligible.sum()):
        raise ValueError("Anchor and eligible populations differ")
    for target in TARGETS:
        if not np.array_equal(merged[target].to_numpy(), merged[target+"_anchor"].to_numpy()):
            raise ValueError("Anchor target pairing differs: "+target)
    anchor_values = merged[anchor_column].to_numpy(float)
    if not np.isfinite(anchor_values).all() or (anchor_values <= 0).any():
        raise ValueError("Anchor forecast must be finite positive seconds")
    values[merged._order.to_numpy(int)] = merged[anchor_column].to_numpy(float)
    np.testing.assert_array_equal(values[~eligible], frame.loc[~eligible, checkpoint_column].to_numpy())
    return values
