import numpy as np
import pytest
import torch

from research.experiments.boundary_20260908.telemetry_sequence import encoder, tokens
from research.experiments.boundary_20260908.telemetry_sequence_verification import numerics


def test_manual_tap_orientation_and_dilated_prefix_geometry():
    x = np.arange(1, 7, dtype=float)[:, None]
    w = np.array([[[100., 10., 1.]]])
    got = numerics.causal_convolution(x, w, np.array([.5]), 2)
    np.testing.assert_array_equal(got[:, 0], [1.5, 2.5, 13.5, 24.5, 135.5, 246.5])
    changed = x.copy();changed[3:] = 999
    np.testing.assert_array_equal(numerics.causal_convolution(changed, w, np.array([.5]), 2)[:3], got[:3])


@pytest.mark.parametrize('padding_count', [0, 1, 64, 127, 128])
@pytest.mark.parametrize('control', ['ordered', 'permuted'])
def test_separate_float64_forward_matches_frozen_torch(padding_count, control):
    torch.set_num_threads(1)
    rng = np.random.default_rng(45)
    context = rng.uniform(0, 5, size=(128, 34)).astype(np.float32)
    context[:, -1] = 0
    context[:padding_count] = 0;context[:padding_count, -1] = 1
    key = tokens.permutation_key(202201, '1', 411)
    if control == 'permuted': context = tokens.permute_context(context, key=key)
    model = encoder.new_model(frozen=True)
    parameters = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    expected = model.representation(torch.tensor(context[None])).numpy()[0]
    actual = numerics.representation(parameters, context)
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
    if padding_count == 128: np.testing.assert_array_equal(actual, np.zeros(16))


def test_forward_rejects_noncausal_padding_or_nonfinite_values():
    x = np.zeros((128, 34));x[1, -1] = 1
    with pytest.raises(ValueError, match='padding'): numerics.representation({}, x)
    x = np.zeros((128, 34));x[3, 1] = np.nan
    with pytest.raises(ValueError, match='finite'): numerics.representation({}, x)
