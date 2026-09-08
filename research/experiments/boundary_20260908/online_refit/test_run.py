import numpy as np
import pandas as pd

from research.experiments.boundary_20260908.online_refit.run import event_weights, plan


def test_refit_plan_never_uses_current_block_or_future_outcomes():
    data = pd.DataFrame({"event_key": [202201,202202]+list(range(202301,202310)),
                         "year": [2022,2022]+[2023]*9})
    blocks = plan(data, 2023)
    assert len(blocks) == 3
    assert blocks[0] == ([202201,202202], [202301,202302,202303,202304])
    assert blocks[1][0][-1] == 202304
    assert blocks[2][1] == [202309]
    seen = []
    for fitted, predicted in blocks:
        assert max(fitted) < min(predicted)
        seen += predicted
    assert seen == list(range(202301,202310))
    future = pd.concat([data, pd.DataFrame({"event_key": [202401], "year":[2024]})])
    assert plan(future, 2023) == blocks


def test_weights_balance_events_then_apply_declared_decay():
    data = pd.DataFrame({"event_key": [1,2,2,3,3,3]})
    uniform = pd.Series(event_weights(data, None)).groupby(data.event_key).sum().to_numpy()
    np.testing.assert_allclose(uniform, [2.,2.,2.])
    w = event_weights(data, 1)
    totals = pd.Series(w).groupby(data.event_key).sum().to_numpy()
    np.testing.assert_allclose(totals/totals[-1], [.25,.5,1.])
    np.testing.assert_allclose(w.mean(), 1., rtol=0, atol=1e-14)
