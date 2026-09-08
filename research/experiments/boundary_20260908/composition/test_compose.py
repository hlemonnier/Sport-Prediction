import numpy as np
import pandas as pd
import pytest

from research.experiments.boundary_20260908.composition.compose import compose


def inputs():
    cp = pd.DataFrame({"event_key":[1,1,1], "driver_id":["7","7","8"],
        "checkpoint_lap":[3,4,3], "checkpoint_time":[270.,420.,275.],
        "target_lap":[5,5,4], "target_time":[510.,510.,366.],
        "target_seconds":[90.,90.,91.], "eligible":[True,False,True],
        "reference":[94.,94.,92.], "expert":[94.,90.5,92.]})
    anchor = pd.DataFrame({"event_key":[1,1],"driver_id":[8,7],
        "issued_after_lap_number":[3,3],"issued_at_timestamp":[275.,270.],
        "target_lap_number":[4,5],"target_timestamp":[366.,510.],
        "lap_time_seconds":[91.,90.],"point":[91.5,91.]})
    return cp, anchor


def test_composition_preserves_asof_regime_and_repeated_targets():
    cp, anchor = inputs()
    np.testing.assert_array_equal(compose(cp, "expert", anchor, "point"), [91.,90.5,91.5])
    np.testing.assert_array_equal(compose(cp, "expert"), cp.expert.to_numpy())
    # Outcomes validate pairing; their magnitudes cannot optimize the gate.
    cp.target_seconds += 1000
    anchor.lap_time_seconds += 1000
    np.testing.assert_array_equal(compose(cp, "expert", anchor, "point"), [91.,90.5,91.5])


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "target", "future", "nonfinite", "nonpositive"])
def test_invalid_component_pairing_is_rejected(mutation):
    cp, anchor = inputs()
    if mutation == "missing":
        anchor = anchor.iloc[:1]
    elif mutation == "duplicate":
        anchor = pd.concat([anchor,anchor.iloc[:1]])
    elif mutation == "target":
        anchor.loc[0,"target_lap_number"] += 1
    elif mutation == "future":
        cp.loc[0,"target_time"] = cp.loc[0,"checkpoint_time"]
    elif mutation == "nonfinite":
        anchor.loc[0,"point"] = np.nan
    else:
        anchor.loc[0,"point"] = 0.
    with pytest.raises(ValueError):
        compose(cp, "expert", anchor, "point")


def test_checkpoint_expert_cannot_silently_modify_eligible_points():
    cp, anchor = inputs()
    cp.loc[0,"expert"] = 91.
    with pytest.raises(ValueError,match="original eligible"):
        compose(cp,"expert",anchor,"point")
