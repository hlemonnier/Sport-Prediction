"""One tail-controlled joint score distribution for every emitted forecast."""
from __future__ import annotations
from dataclasses import dataclass
import math
from .joint import dixon_coles_tau, validate_rho

DEFAULT_TAIL_TOLERANCE = 1e-12

@dataclass(frozen=True)
class ScoreDistribution:
    matrix: tuple[tuple[float, ...], ...]
    omitted_probability_mass: float
    reconciled: bool = False
    tail_probability_bound: float = 0.0
    source_parameters: tuple[float, float, float] | None = None

    @property
    def outcome_probabilities(self) -> tuple[float, float, float]:
        result = [0.0, 0.0, 0.0]
        for home, row in enumerate(self.matrix):
            for away, value in enumerate(row):
                result[0 if home > away else 1 if home == away else 2] += value
        return tuple(result)

    @property
    def expected_goals(self) -> tuple[float, float]:
        return (sum(h*p for h,row in enumerate(self.matrix) for p in row),
                sum(a*p for row in self.matrix for a,p in enumerate(row)))

    @property
    def most_likely_scoreline(self) -> tuple[int, int, float]:
        return max(((h,a,p) for h,row in enumerate(self.matrix) for a,p in enumerate(row)),
                   key=lambda item: (item[2], -item[0], -item[1]))

    def reconcile(self, selected: tuple[float, float, float]) -> "ScoreDistribution":
        if len(selected) != 3 or any(not math.isfinite(p) or p < 0 for p in selected):
            raise ValueError("Selected 1X2 probabilities must be finite and nonnegative.")
        total = math.fsum(selected)
        if total <= 0:
            raise ValueError("Selected 1X2 probabilities must have positive mass.")
        targets = [p / total for p in selected]
        regions = self.outcome_probabilities
        if any(mass <= 0 and target > 0 for mass,target in zip(regions,targets)):
            raise ValueError("Cannot assign selected probability to an empty score region.")
        amplification = max(target/mass for target,mass in zip(targets,regions) if mass > 0)
        bound = self.tail_probability_bound * amplification
        if bound > DEFAULT_TAIL_TOLERANCE and self.source_parameters is not None:
            tighter = DEFAULT_TAIL_TOLERANCE / amplification / 2
            return build_score_distribution(*self.source_parameters, tail_tolerance=tighter).reconcile(tuple(targets))
        matrix = tuple(tuple(p * targets[0 if h>a else 1 if h==a else 2]
                             / regions[0 if h>a else 1 if h==a else 2]
                             for a,p in enumerate(row)) for h,row in enumerate(self.matrix))
        return ScoreDistribution(matrix, self.omitted_probability_mass, True, bound, self.source_parameters)


def _poisson_support(rate: float, tail_tolerance: float, minimum: int) -> tuple[list[float], float]:
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("Poisson rates must be finite and positive.")
    if rate > 100:
        raise ValueError("Football score distribution does not support goal rates above 100.")
    probabilities = [math.exp(-rate)]
    while True:
        k = len(probabilities) - 1
        next_probability = probabilities[-1] * rate / (k+1)
        ratio_bound = rate / (k+2)
        # Subsequent Poisson ratios decrease, so a geometric series bounds
        # the omitted mass without cancellation in 1 - computed CDF.
        tail_bound = next_probability / (1-ratio_bound) if ratio_bound < 1 else math.inf
        if k >= max(1,minimum) and tail_bound <= tail_tolerance:
            return probabilities, tail_bound
        if k >= 1000:
            raise RuntimeError("Poisson support failed to meet tail tolerance.")
        probabilities.append(next_probability)


def build_score_distribution(lambda_home: float, lambda_away: float, rho: float,
                             *, tail_tolerance: float = DEFAULT_TAIL_TOLERANCE,
                             min_max_goals: int = 1) -> ScoreDistribution:
    if not math.isfinite(tail_tolerance) or not 1e-25 <= tail_tolerance < 1e-3:
        raise ValueError("tail_tolerance must lie in [1e-25, 1e-3).")
    validate_rho(lambda_home, lambda_away, rho)
    home, home_bound = _poisson_support(lambda_home, tail_tolerance/2, min_max_goals)
    away, away_bound = _poisson_support(lambda_away, tail_tolerance/2, min_max_goals)
    size = max(len(home), len(away))
    for values,rate in ((home,lambda_home),(away,lambda_away)):
        while len(values)<size:
            values.append(values[-1]*rate/len(values))
    matrix = [[hp*ap for ap in away] for hp in home]
    for h in (0,1):
        for a in (0,1):
            matrix[h][a] *= dixon_coles_tau(h,a,lambda_home,lambda_away,rho)
            if matrix[h][a] < 0:
                raise ValueError("Negative score mass; the fitted support is invalid.")
    total = math.fsum(p for row in matrix for p in row)
    omitted = max(0.0, 1-total)
    if omitted > tail_tolerance * 1.01 + 1e-15:
        raise RuntimeError("Score support did not satisfy the declared tail bound.")
    # Renormalization differs from the infinite distribution by at most the
    # reported tail bound. The same matrix supplies every downstream output.
    return ScoreDistribution(tuple(tuple(p/total for p in row) for row in matrix), omitted,
                             False, home_bound+away_bound, (lambda_home,lambda_away,rho))
