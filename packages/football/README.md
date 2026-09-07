# Football research models

The executable package provides pre-match 1X2 forecasts and a coherent joint score
distribution. Its status is **research only**. Implemented methods and passing
synthetic regression tests establish neither real-data predictive accuracy nor a
betting edge. `maturity.json` records the supported and unfinished surfaces.

## Implemented mathematics

The goal model maximizes a joint Dixon–Coles likelihood:

`sum_m w_m [log Pois(H_m; lambda_h,m) + log Pois(A_m; lambda_a,m) + log tau_m]`

minus `0.002 / 2` times the sum of squared attack/defense parameters. Home and away
rates use separate log intercepts and the relevant team attack and opponent
concession parameters. Attack and defense parameters each sum to zero; positive
defense parameters mean a greater concession rate. The fit jointly estimates rho,
constrains rates to 0.05–6, and enforces nonnegative support for every pair of fitted
or neutral unseen teams. The implementation does not repair negative probabilities
by clipping them after fitting. SLSQP success, feasibility and a first-order KKT
stationarity residual are checked; failed fits raise an explicit error.

No temporal decay is used by default. `--goal-strength-half-life-days DAYS` enables
weights `2 ** (-age_days / DAYS)` and records their effective sample size. This is
an explicit modelling option, not a demonstrated improvement; choose it without
using the final test block.

A frequency baseline and optional gradient-boosting classifier provide comparisons.
The classifier uses past team points, goal differences, xG differences and recent
sample counts. Unversioned xG is replaced with the documented final-goal proxy;
xG is consumed only when `xg_available_at` establishes availability at the relevant
forecast cutoff. All training feature rows use results available before prediction.

Every forecast uses one adaptively extended Poisson score grid with omitted base
mass bounded by a geometric Poisson-tail bound. Calibrated/GBDT/hybrid 1X2 probabilities are reconciled with
that grid by preserving conditional score probabilities inside each home-win,
draw and away-win region. Support is extended further when region reweighting
would amplify the omitted-tail error above `1e-12` (separately from floating-point
rounding). The displayed most likely score, its probability, outcome
probabilities and expected goals all come from that final joint matrix. Legacy
`lambda_home_goals`/`lambda_away_goals` fields are aliases for the final means;
`dixon_base_lambda_*` fields retain the fitted Poisson rates before reconciliation.
Reconciliation is a coherent construction, not proof of improved score forecasting.

## Causal training and evaluation

A scored row requires a dated kickoff. Prefer an explicit `result_available_at`
(or `final_whistle_at`/`score_available_at`) timestamp. Without one, availability is
conservatively assigned to the beginning of the following UTC day. Merely starting
a match does not make its final score available. Undated training rows are excluded;
undated target fixtures are not forecast. An empty cutoff produces an explicit
prior-only forecast, never reuse of future outcomes. A missing requested round
produces no unrelated predictions.

For sufficient data, chronological whole-day blocks are reserved for:

1. Base parameter fitting: at least 45 matches.
2. Probability calibration: at least 30 matches, normally 15% of history.
3. Mixture selection: at least 30 matches, normally 15% of history.
4. Untouched outer evaluation: at least 30 matches, normally 20% of history.

Availability across each boundary is enforced; late-published rows are purged from
the earlier parameter/selection population and identified in diagnostics. Block
minimums are feasibility constraints, not a claim of adequate statistical power.
All reported models are evaluated on the same outer identities. No fitting or
calibration metric is reported as out-of-sample performance. Insufficient history
uses a fit/calibration split when possible, otherwise uncalibrated fitting, with
no claimed validation and no hybrid selection.

The deployed base parameters remain those from the fit prefix. Calibration and
selection never train base parameters; the outer outcomes never choose a policy.
Completed results may update rolling form for later predictions, including during
the outer block, matching a sequential pre-match forecast protocol. A new run
repeats this documented procedure on its updated available history. This approach
trades some parameter recency for a directly auditable separation of evidence.

Automatic calibration uses Platt scaling of class log-odds on smaller held-out
samples; isotonic becomes eligible at 200 calibration rows. Explicit `isotonic`
remains available above its sample/class support checks. Per-class calibration is
normalized onto the simplex and evaluated after normalization; exact multiclass
calibration is not assumed. The hybrid weight is selected only on its separate
selection block. The outer block reports log loss, multiclass Brier, top-label ECE,
ranking summaries and selected-model scoreline log loss. Clipped log scoring uses
a `1e-12` floor. Match identities, population/content SHA256 values, outer forecast
rows, solver diagnostics and fixture score matrices accompany the result.

## Runtime and remaining scope

Install `packages/football/requirements.txt` into an isolated environment. Joint
goal fitting requires NumPy and SciPy. If the optional sklearn import is unavailable,
GBDT and learned calibration are explicitly unavailable and the goal model remains
usable. Pandas/pyarrow are additionally needed to consume parquet input.

Only the local CSV/parquet provider is connected. Football-data.org, FBref and
StatsBomb adapters remain explicit stubs. Live-match and player-props models,
lineups, injuries, rest/travel and market inputs remain unimplemented. Weather is
output metadata only. Pre-match and scoreline wrappers share this implementation;
their separate folders do not imply independent trained models. The legacy
notebook is an experiment template.

No real football dataset or validated market-relative benchmark is bundled in the
canonical data/artifact directories. A real deployment decision still requires
versioned source snapshots, multiple chronological seasons/leagues, uncertainty
from matchday/season-aware resampling, comparison with a bookmaker baseline at the
same decision time, and costs/limits appropriate to the intended use. Those are
future empirical work, not hidden claims of this package.
