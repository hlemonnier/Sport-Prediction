"""Independent NumPy neural forward pass for the fixed CPC representation.

No Torch model methods, fitting, checkpoint selection or historical I/O.
Suggested commit: research(f1-live): independently replay causal encoder numerics
"""
from __future__ import annotations

import numpy as np
from scipy.special import erf


def causal_convolution(values, weights, bias, dilation):
    """Cross-correlation in timestamp/channel layout, evaluated in float64."""
    values, weights, bias = (np.asarray(x, dtype=np.float64) for x in (values, weights, bias))
    if (values.ndim != 2 or weights.ndim != 3 or weights.shape[1] != values.shape[1]
            or weights.shape[2] != 3 or bias.shape != (weights.shape[0],)
            or type(dilation) is not int or dilation <= 0):
        raise ValueError('Invalid causal convolution dimensions')
    padded = np.pad(values, ((2*dilation, 0), (0, 0)))
    result = np.broadcast_to(bias, (len(values), len(bias))).copy()
    for tap in range(3):
        result += padded[tap*dilation:tap*dilation+len(values)] @ weights[:, :, tap].T
    return result


def representation(parameters, context):
    """Recompute the endpoint vector from a plain mapping of numeric weights.

    Float64 arithmetic and SciPy's erf deliberately differ from Torch's float32
    kernels. The verification protocol fixes tolerances before historical use.
    The caller supplies the ordered or already-permuted admitted context.
    """
    context = np.asarray(context)
    if (context.shape != (128, 34) or not np.isfinite(context).all()
            or np.any(context < 0) or np.any(context > 5)):
        raise ValueError('Expected a finite bounded 128 by 34 context')
    padding = context[:, -1]
    if (not np.isin(padding, (0, 1)).all() or np.any(np.diff(padding) > 0)
            or np.any(context[padding == 1, :-1] != 0)):
        raise ValueError('Invalid left padding')
    values = context.astype(np.float64)
    for layer, dilation in enumerate((1, 2, 4, 8, 16, 32, 64)):
        prefix = f'layers.{layer}.'
        values = causal_convolution(values, parameters[prefix+'conv.weight'],
                                    parameters[prefix+'conv.bias'], dilation)
        centered = values - values.mean(axis=-1, keepdims=True)
        values = centered / np.sqrt(np.mean(centered**2, axis=-1, keepdims=True) + 1e-5)
        values = values*np.asarray(parameters[prefix+'norm.weight']) + np.asarray(parameters[prefix+'norm.bias'])
        values = .5*values*(1 + erf(values/np.sqrt(2.)))
    result = values[-1] @ np.asarray(parameters['projection.weight']).T + np.asarray(parameters['projection.bias'])
    result = result / max(float(np.linalg.norm(result)), 1e-12)
    if np.all(padding == 1): result = np.zeros(16, dtype=np.float64)
    if result.shape != (16,) or not np.isfinite(result).all():
        raise ValueError('Invalid recomputed representation')
    return result
