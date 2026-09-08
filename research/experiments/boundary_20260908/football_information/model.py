"""Small convex probability pools and outcome-free odds transformations."""
from dataclasses import dataclass
from numbers import Real

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.special import logsumexp, softmax

FLOOR = 1e-12
PENALTY = .01
# Orthonormal coordinates avoid double-counting the constrained third bias.
BIAS_BASIS = np.array([[1/np.sqrt(2), 1/np.sqrt(6)],
                       [-1/np.sqrt(2), 1/np.sqrt(6)], [0., -2/np.sqrt(6)]])


def devig(odds, method):
    if len(odds) != 3 or any(isinstance(x, (bool, np.bool_)) or not isinstance(x, Real) for x in odds):
        raise ValueError('Three numeric decimal odds are required')
    o = np.asarray(odds, dtype=float)
    if np.any(~np.isfinite(o)) or np.any(o <= 1):
        raise ValueError('Decimal odds must be finite and strictly greater than one')
    logs = -np.log(o)
    if method == 'normalized':
        return softmax(logs), None
    if method != 'power':
        raise ValueError('Unknown margin removal method')
    # logsumexp stays finite for underrounds, extreme prices and large powers.
    high = 1.
    while logsumexp(high * logs) > 0:
        high *= 2
    power = brentq(lambda k: logsumexp(k*logs), 0., high, xtol=1e-14, rtol=1e-12)
    return softmax(power*logs), float(power)


def probabilities(values):
    p = np.asarray(values, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or not len(p):
        raise ValueError('Expected a nonempty N by three probability matrix')
    if np.any(~np.isfinite(p)) or np.any(p < 0) or np.any(p > 1) or not np.allclose(p.sum(1), 1., atol=1e-10, rtol=0):
        raise ValueError('Invalid probability simplex')
    p = np.maximum(p, FLOOR)
    return p / p.sum(1, keepdims=True)


def dimensions(family, internal):
    if family not in ('shared', 'classwise') or type(internal) is not bool:
        raise ValueError('Unknown pool family or internal-component policy')
    width = 1 if family == 'shared' else 3
    return width, width*(1+internal)+2


def unpack(theta, family, internal):
    width, size = dimensions(family, internal)
    theta = np.asarray(theta, dtype=float)
    if theta.shape != (size,) or np.any(~np.isfinite(theta)):
        raise ValueError('Invalid parameter vector')
    a = theta[:width]
    c = theta[width:2*width] if internal else np.zeros(width)
    u = theta[-2:]
    return a, c, BIAS_BASIS @ u


def objective(theta, log_market, log_internal, labels, family, internal, penalty=PENALTY):
    a, c, b = unpack(theta, family, internal)
    logits = log_market*a + b
    if internal:
        logits = logits + log_internal*c
    n = len(labels)
    value = np.mean(logsumexp(logits, axis=1)-logits[np.arange(n), labels])
    value += penalty/2 * (np.sum((a-1)**2)+np.sum(c*c)+np.sum(b*b))
    residual = softmax(logits, axis=1)
    residual[np.arange(n), labels] -= 1
    residual /= n
    ga = np.sum(residual*log_market, axis=0)
    if family == 'shared':
        ga = np.array([ga.sum()])
    gradients = [ga + penalty*(a-1)]
    if internal:
        gc = np.sum(residual*log_internal, axis=0)
        if family == 'shared':
            gc = np.array([gc.sum()])
        gradients.append(gc+penalty*c)
    gradients.append(residual.sum(0) @ BIAS_BASIS + penalty*np.asarray(theta[-2:]))
    return float(value), np.concatenate(gradients)


def projected_gradient(theta, gradient, bounds):
    projected = np.asarray(gradient).copy()
    for i, (lower, upper) in enumerate(bounds):
        if lower is not None and theta[i] <= lower+1e-8:
            projected[i] = min(projected[i], 0.)
        if upper is not None and theta[i] >= upper-1e-8:
            projected[i] = max(projected[i], 0.)
    return float(np.max(np.abs(projected)))


@dataclass
class Pool:
    family: str
    internal: bool
    theta: np.ndarray
    diagnostics: dict

    def predict(self, market, prior=None):
        lm = np.log(probabilities(market))
        a, c, b = unpack(self.theta, self.family, self.internal)
        logits = lm*a+b
        if self.internal:
            lp = np.log(probabilities(prior))
            if lp.shape != lm.shape:
                raise ValueError('Market and internal populations differ')
            logits += lp*c
        output = softmax(logits, axis=1)
        if np.any(~np.isfinite(output)) or np.any(output <= 0):
            raise ValueError('Fitted pool produced invalid probabilities')
        return output

    def serialize(self):
        return {'family': self.family, 'internal': self.internal,
                'theta': self.theta.tolist(), 'diagnostics': self.diagnostics}


def fit(market, prior, labels, family, internal):
    lm = np.log(probabilities(market))
    lp = np.log(probabilities(prior)) if internal else None
    if internal and lp.shape != lm.shape:
        raise ValueError('Training populations differ')
    y = np.asarray(labels)
    if y.shape != (len(lm),) or y.dtype.kind not in 'iu' or np.any((y < 0) | (y > 2)):
        raise ValueError('Labels must be integer H/D/A indices')
    width, size = dimensions(family, internal)
    initial = np.zeros(size); initial[:width] = 1.
    bounds = [(0., 4.)]*(size-2)+[(None, None)]*2
    result = minimize(objective, initial, args=(lm, lp, y, family, internal), jac=True,
                      method='L-BFGS-B', bounds=bounds,
                      options={'ftol': 1e-13, 'gtol': 1e-8, 'maxiter': 2000})
    value, gradient = objective(result.x, lm, lp, y, family, internal)
    kkt = projected_gradient(result.x, gradient, bounds)
    if not result.success or not np.isfinite(value) or kkt > 1e-6:
        raise ValueError(f'Pool did not converge: {result.message}; KKT={kkt}')
    return Pool(family, internal, result.x, {'n': len(y), 'objective': value,
                'projected_gradient_max': kkt, 'iterations': int(result.nit),
                'solver_success': bool(result.success), 'message': str(result.message),
                'penalty': PENALTY, 'bias_sum': float(unpack(result.x, family, internal)[2].sum())})
