# Architecture choices after the rival-gap screen

Decision recorded before the new CatBoost historical fits. The completed gap
screen improved 2023 event MAE by0.3542% against the original-feature HGB but
failed its fixed advancement rule. This is evidence against promoting that
candidate, not proof that additional information is useless.

The next executed comparison is `catboost_frontier/`: two fixed symmetric-tree
boosting schemes on the unchanged80 inputs and full2022 training cohort.
The original frontier covered HGB, ExtraTrees and a row-wise MLP, but not
CatBoost Plain or Ordered. The [CatBoost paper](https://arxiv.org/abs/1706.09516)
motivates an alternative training scheme; empirical gains in that paper do not
predict an F1 gain. No categorical feature processing is invoked here.

CatBoost1.2.10 and graphviz0.21 were installed only into an isolated research
directory. Both full-shape synthetic10-tree fits completed: Plain0.293s and
Ordered0.939s, maximum cumulative process RSS312,852,480bytes. These timings
include initialization and are not exact forecasts for1,000-tree runs.
The runtime manifest binds91 installed files and both wheels. The official
[macOS CPython3.12 wheel](https://pypi.org/project/catboost/1.2.10/) is available
without a foundation-model checkpoint or paid compute.

Two distinct alternatives remain unexecuted:

- **Bayesian run-length residual adaptation.** Unlike the existing fixed EWMA
  and single-state Kalman models, this maintains a posterior across possible
  persistent-error segment lengths and uses the median of its predictive
  mixture. A concrete Normal–inverse-gamma protocol exists in
  `next_mechanisms.md`; its prior must be estimated from expanding2022
  out-of-fold errors, never in-sample residuals. The
  [run-length inference paper](https://arxiv.org/abs/0710.3742) supports the
  algorithm. The earlier residual study's0.1767% gain gives weak evidence for
  substantial upside, so this remains behind the direct architecture test.
- **Football strengths learned from past market forecasts.** The completed
  market pool consumed a current-fixture quote; a separate model could instead
  learn team strength from earlier matches' market probabilities and forecast
  at the existing local-midnight horizon. The
  [Elo-Odds study](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0198668)
  supports this information-transfer hypothesis without claiming superiority
  to current-market odds. Read-only inspection verified27 cached CSV hashes,
  with10,259/10,260 valid average-odds triples. Prior quote coverage exists for
  both teams in758/760 selection fixtures under a delayed availability proxy,
  or757/760 with seven additional days. No candidate fit or score was computed.
  This path needs a new fixed protocol, stronger internal/matched controls,
  explicit handling of promoted teams and missing quotes, and provenance for
  the unavailable original quote receipt timestamps.

TabPFN2.5 remains another architecture possibility, but was not executed.
The1,000-row CPU guard is configurable, not a mathematical input limit, and
large KV-cache estimates do not establish an uncached memory lower bound.
Full-cohort uncached inference has not been measured on this16GiB machine.
Model-version and checkpoint rights must also be explicit. No checkpoint was
downloaded or external training result treated as evidence about this repo.

Suggested commit: `research: prioritize untested learning algorithms after the gap screen`.
