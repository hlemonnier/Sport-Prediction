"""Synthetic mathematical checks; no historical fitting or outcome scoring."""
import numpy as np
import pytest

from . import model


@pytest.mark.parametrize('odds', [[2., 3., 4.], [10., 10., 10.], [2., 4., 4.],
                                 [1.01, 1.02, 1.03], [1.0000001, 1e100, 1e200]])
def test_power_root_handles_overround_underround_and_extreme_prices(odds):
    p, power = model.devig(odds, 'power')
    assert power > 0 and np.all(p >= 0) and p.sum() == pytest.approx(1.)
    assert np.exp(-power*np.log(odds)).sum() == pytest.approx(1., abs=1e-10)
    raw_sum = np.reciprocal(np.asarray(odds)).sum()
    assert (power < 1) if raw_sum < 1 else power >= 1-1e-12


@pytest.mark.parametrize('odds', [[1., 2., 3.], [0., 2., 3.], [np.nan, 2., 3.],
                                 [np.inf, 2., 3.], [True, 2., 3.], [2., 3.]])
def test_invalid_odds_are_not_silently_imputed(odds):
    with pytest.raises(ValueError): model.devig(odds, 'power')


def matrices():
    rng = np.random.default_rng(57)
    return rng.dirichlet([1, 2, 3], 40), rng.dirichlet([3, 2, 1], 40), rng.integers(0, 3, 40)


@pytest.mark.parametrize('family', ['shared', 'classwise'])
@pytest.mark.parametrize('internal', [False, True])
def test_analytic_gradient_and_convexity(family, internal):
    m, p, y = matrices(); rng = np.random.default_rng(92)
    width, size = model.dimensions(family, internal)
    theta = rng.uniform(.2, 1.5, size);theta[-2:] -= .5
    f = lambda t: model.objective(t, np.log(m), np.log(p) if internal else None, y, family, internal)
    value, analytic = f(theta);h = 1e-6;eye = np.eye(size)
    numeric = np.array([(f(theta+h*d)[0]-f(theta-h*d)[0])/(2*h) for d in eye])
    np.testing.assert_allclose(analytic, numeric, rtol=1e-6, atol=1e-8)
    hessian = np.column_stack([(f(theta+h*d)[1]-f(theta-h*d)[1])/(2*h) for d in eye])
    np.testing.assert_allclose(hessian, hessian.T, atol=1e-8)
    assert np.linalg.eigvalsh(hessian).min() >= model.PENALTY-1e-7
    b = model.unpack(theta, family, internal)[2]
    assert b.sum() == pytest.approx(0., abs=1e-14)
    assert b @ b == pytest.approx(theta[-2:] @ theta[-2:])


@pytest.mark.parametrize('family', ['shared', 'classwise'])
@pytest.mark.parametrize('internal', [False, True])
def test_market_identity_and_synthetic_fit_replay(family, internal):
    m, p, y = matrices();width, size = model.dimensions(family, internal)
    initial = np.zeros(size);initial[:width] = 1.
    np.testing.assert_allclose(model.Pool(family, internal, initial, {}).predict(m, p), m, atol=1e-15)
    fitted = model.fit(m, p, y, family, internal)
    assert fitted.diagnostics['projected_gradient_max'] <= 1e-6
    frozen = fitted.serialize()
    replay = model.Pool(frozen['family'], frozen['internal'], np.array(frozen['theta']), frozen['diagnostics'])
    np.testing.assert_array_equal(fitted.predict(m, p), replay.predict(m, p))
    assert fitted.diagnostics['objective'] <= model.objective(initial, np.log(m), np.log(p), y, family, internal)[0]


def test_matched_market_only_reference_does_not_consume_internal_values():
    m, _, y = matrices()
    class Poison:
        def __array__(self, *_args, **_kwargs): raise AssertionError('Internal input read by market-only model')
    pool = model.fit(m, Poison(), y, 'classwise', False)
    assert pool.predict(m, Poison()).shape == (40, 3)


def test_projected_gradient_uses_correct_bound_signs():
    assert model.projected_gradient([0., 4., 0.], [1., -1., 0.], [(0., 4.), (0., 4.), (None, None)]) == 0.
    assert model.projected_gradient([0.], [-2.], [(0., 4.)]) == 2.


@pytest.mark.parametrize('family', ['shared', 'classwise'])
def test_synthetic_planted_internal_information_reaches_the_pool(family):
    # Functional information-path check, not historical predictive evidence.
    y = np.tile(np.arange(3), 20)
    m = np.full((len(y), 3), 1/3)
    p = np.full_like(m, .05);p[np.arange(len(y)), y] = .9
    pooled = model.fit(m, p, y, family, True)
    market = model.fit(m, None, y, family, False)
    q, reference = pooled.predict(m, p), market.predict(m)
    assert -np.log(q[np.arange(len(y)), y]).mean() < -np.log(reference[np.arange(len(y)), y]).mean()-.5


@pytest.mark.parametrize('invalid', [[[.5, .5, .1]], [[-1., 1., 1.]], [[np.inf, 0, 0]], [], [[True, False, False]]])
def test_probability_validation(invalid):
    if invalid == [[True, False, False]]:
        # Numeric 0/1 simplex vertices are supported; flooring preserves finite logs.
        assert np.all(model.probabilities(invalid) > 0)
    else:
        with pytest.raises(ValueError): model.probabilities(invalid)
