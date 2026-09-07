# Model mathematics remediation — 2026-09-07

This work implements all 16 findings in the September model audit and the additional defects found during cross-review. Mathematical correctness and empirical forecasting advantage remain separate questions. Existing baselines remain active; historical replays do not constitute prospective promotion evidence.

Baseline checkout: `d18202f4b68accb82ab939a198b6c8d9e3c8951c`. Validation completed before the requested commit and push; no application deployment was performed. Original frozen artifacts and user-owned duplicate copies remain in place. The current register, notebook and maturity report have been finalized and validated.

## Implementation and mathematical verification

| Audit | Correction | Regression evidence |
|---|---|---|
| 01 | Simulator and deterministic planner persist a distinct clean baseline through state adapters; one-time pit/transient losses do not become permanent pace penalties. | Multi-lap additive-cost conservation, state roundtrip, MPC and calibration tests. |
| 02 | Monte Carlo draws initial posterior uncertainty once, then transition noise only. | Gaussian one-step/cumulative moments and covariance tests; the constant-state H-lap posterior contribution is H²P, not the repeatedly injected sum of squared remaining horizons. |
| 03 | Cars are projected to a common declared race-distance target, capped by known scheduled distance. | Asynchronous snapshots, genuine lap deficits, scheduled/remaining caps and terminal no-rollout tests. This is a conditional distance-horizon forecast, not a claim that unknown race duration is observed. |
| 04 | Football result ingestion rejects future and ambiguous availability, excludes unknown dates, and never substitutes a different requested round. | Future-only cold starts, undated results, publication delays, timezones, duplicate identities and actual CLI execution. |
| 05 | Football uses separate chronological fit, calibration, selection and outer-test blocks; compared candidates share the same held-out population. | Outcome/availability perturbations and a synthetic 260-row workflow with 130/39/39/52 rows. Frozen fit parameters are reused for deployment; form may update only from already available outcomes. |
| 06 | Football validation grows with history and leaves an attainable calibration population. | Large/small history split boundaries and an actual held-out Platt fit for both components. |
| 07 | A new stint resets the filter even on an excluded pit/outlap; observed tyre wear is retained. | Pit-in/outlap/clean-lap traces, scrubbed tyres and provider stint attachment. |
| 08 | Dixon–Coles rho must preserve all four low-score cells for all supported team pairs, including neutral/unseen teams. | Unobserved-cell and extreme-rate tests; feasibility is checked independently after optimization. Negative mass is never repaired by clipping. |
| 09 | Scoreline, 1X2 and expected-goal outputs derive from one tail-controlled joint distribution. | Poisson tail bounds, probability conservation and marginal/mean consistency after mixture reconciliation. |
| 10 | Goal rates and rho fit one joint penalized Dixon–Coles likelihood with matching derivatives and constrained convergence checks. | Numerical gradient checks, optimizer comparisons, KKT stationarity and synthetic CLI validation. Optional time decay is explicit; no time-varying latent-strength model is claimed. |
| 11 | Untimed failures contribute an interval-censored observed-data likelihood. | The known 20% terminal-incidence population no longer collapses toward zero as sample size grows; finite-difference likelihood gradients and covariate regressions. |
| 12 | Disqualification is an exclusion mechanism separate from timed retirement; order uses completed distance. | Full-distance DSQ exclusion, integer completed-lap ordering and coherent six-category probabilities. |
| 13 | Head-to-head scores retain the full event’s feature context. | Pair probabilities agree with the corresponding full-field scores. |
| 14 | All best-lap components pass chronology checks before missing-target filtering; future training seasons/invalid event coordinates are rejected before providers; explicit target names never fall back silently. | Missing-target leakage, provider-not-called and named-target tests. |
| 15 | Practice comparisons use independent peers matched on known compound, tyre age and session clock. Unsupported comparisons remain missing with zero support. | Singleton/sparse/comparable-peer, native timedelta-unit and provider integration tests. Relative drift and dispersion are descriptive, not identified fuel/tyre physics or calibrated uncertainty. |
| 16 | Metrics require valid complete evidence; probabilities have strict identity/bounds/alias checks; promotion requires adequate, frozen, unexposed event evidence. | Empty/mismatched/invalid metric inputs, malformed evidence, tiny/exposed/reordered audits, hash changes, weakened family gates and textual false assertions all fail closed. |

The new regression suites are `test_live_race_math_regressions.py`, `test_live_tyre_transition_math.py`, `test_live_sparse_lap_transitions.py`, `test_pre_event_mathematical_repairs.py`, `test_betting_probability_contract.py`, `test_math_audit_evidence_contracts.py`, and football `test_probability_integrity.py`. Existing integration suites also exercise the changed paths.

## Additional corrections from cross-review

- Sparse live observations advance the latent mean and covariance by the actual elapsed lap count. For a gap Δ, the state uses A^Δ and covariance A^Δ P (A^Δ)ᵀ + Σ[j=0..Δ−1] A^j Q (A^j)ᵀ. New-stint resets use a known start boundary when available; scrubbed tyre age is not invented elapsed race time. Unknown boundaries are labeled explicitly.
- A simulated `PIT_NOW` traverses its first full new-compound lap and therefore ends at tyre age 1. `PIT_NEXT` first runs the old tyre and queues the next pit. Both engines reset the compound degradation prior.
- Betting de-vig requires a complete unique field and normalizes top-K markets to mass min(K,n). Quote fields cannot override model probabilities. Stakes are explicitly a capped independent binary-Kelly heuristic; no joint portfolio optimality is claimed.
- Registry evidence binds candidate and baseline model bytes, validates model/fallback identities and retains mandatory family requirements even when callers request custom gates. Learned strategy additionally needs model-bound OPE uncertainty/support and subsequent prospective shadow evidence.
- The live next-lap replay is always descriptive. Its final weight fit on all evaluated events is not the same fitted object as the causal sequence evaluated in the replay, so that refit cannot become a validated runtime default.
- Best-lap confirmatory evidence binds the actual model specification, implementation, fitted-role input bytes and partitions. Its point and interval products remain separately gated.
- Exported qualifying/race missing values are standard JSON `null`; infinities are rejected. Frozen forecast hashes remain meaningful. Canonical grid-capture discovery excludes OS duplicate filenames without deleting them.

## Likelihood and evidence semantics

For an unknown-time cause-c failure, the observed likelihood is Σ_b S(b−1) h_c(b), where S(b)=Π[s≤b](1−Σ_k h_k(s)). A known-time failure contributes its observed bin term; a censored row contributes survival over its observed exposure. The baseline EM and covariate objective use this same observation contract. DNS and exclusion conditional on starting are separate mechanisms.

Dixon–Coles feasibility requires max(−1/λ_home,−1/λ_away) ≤ rho ≤ min(1,1/(λ_home λ_away)). The implementation fits the joint Poisson-plus-log-tau likelihood, with consistent ridge scaling and explicit rate/support constraints. Solver success is supplemented by a KKT stationarity check; this establishes local first-order convergence, not a proof of global optimality.

The promotion minimum is eight independent audit events and at least two in each required weekend format. This is a design floor, not a power guarantee. R1–R9 of 2026 are recorded as development-exposed and cannot be redeclared pristine. Bootstrap improvement frequency describes empirical event resampling; it is not a posterior probability or a transport guarantee. P05–P90 is an 85% interval, and small calibration samples do not establish precise conditional tail coverage.

## Verification and current empirical results

Final verification: **1,037 F1/service tests passed**, **31 football tests passed**, and three optional-backend tests were skipped. The complete existing suites were run under Python 3.12, including service adapters and the new mathematical regressions. All nine notebook code cells executed with no cell errors. Source and documentation whitespace checks pass; frozen CSV inputs and captured test logs retain their original bytes, including CRLF or emitted whitespace. Commands and hashed logs are recorded in `evidence/model_math_verification_20260907.json`; runtime versions are in `evidence/model_math_runtime_20260907.json`.

Every one of the **11 registered F1 artifacts** passes recursive code/data provenance checks: **4,145 transitive file references**, **158 nested JSON references**, and **41 aggregate digests**. Eight affected artifacts were regenerated; three unchanged telemetry controls were reverified. The notebook now reads the single checked-in artifact register instead of keeping a second list of paths and hashes. Summary JSON, all exported comparison tables, chart, analytics snapshot and maturity metadata were refreshed.

| Historical comparison | Retained baseline | Corrected candidate | Outcome |
|---|---:|---:|---|
| Qualifying audit MAE, positions (3 events) | 1.363636 | 1.363636 | Tie; baseline retained |
| Race-order audit MAE, positions (3 events) | 3.090909 | 3.363636 | Candidate remains worse |
| Best-lap audit MAE, seconds (3 events) | 0.371380 | 0.377919 | Candidate remains worse |
| Live next-lap event-mean MAE, seconds (9 events / 8,144 forecasts) | 0.549184 | 0.540255 | Diagnostic CI crosses zero; baseline retained |

The race repair reduces the previous candidate's historical MAE from 3.787879 to 3.363636, but its status Brier remains 0.166040 versus baseline 0.164096, and log loss 0.511188 versus 0.508221. These are correctness-regression diagnostics on inspected data, not a new blind accuracy claim. Qualifying probability calibration slightly improves log loss but slightly worsens Brier. Best-lap candidate interval coverage is 90.91% at width 1.833781s, versus 86.36% at 1.395000s for the baseline. It remains unpromoted.

Live blend-minus-naive MAE is -0.008929s, with event-bootstrap 95% interval [-0.019131, 0.000498]s. The all-data refit weight is explicitly unvalidated and cannot replace the naïve runtime default. Strategy replay still has 9,506 partial-label behavior-cloning rows, zero eligible offline-Q rows and zero propensity-OPE rows; behavior cloning ties the trivial policy and pit F1 remains zero.

Football's reproducible **synthetic** workflow is preserved under `artifacts/validation/football/math_remediation_20260907/`, including its CSV inputs, output and source/data-hashed validation manifest. It verifies an actual 130/39/39/52 split, successful held-out Platt calibration, common outer evaluation population, feasible joint goal fitting and coherent score probabilities. It establishes execution and mathematical integrity only. Real-data football accuracy, market advantage and profitability remain unestablished.

## Current artifact pins

The authoritative machine-readable list is `evidence/f1_model_research_v2_artifact_register.csv`. Original frozen files were not overwritten. Intermediate reruns created while repairing dependencies are superseded and are absent from the current register.

The committed evidence includes the eight regenerated artifacts, validation logs, synthetic football inputs/output, and nine FIA grid snapshots referenced by the frozen race report under ` 2.json` filenames. Those nine snapshots are byte-identical to already tracked canonical copies; retaining their exact referenced paths preserves the immutable dependency chain. Unrelated duplicate source and document files remain untracked.

| Artifact | SHA256 |
|---|---|
| `best_lap` | `3377bbc04f49be113f28afcc45d3c6deb0d2e80f57c66623cc8a36d1d6f3f657` |
| `live_next_lap` | `7d857a91dd5546b799ccea1da7a95cadb41da35ab4899f1f005212f41f4fe565` |
| `live_strategy_replay` | `53514c61baccd9aa91e9855371101e2f1cfc7b22580da67386cbbd53a412ff78` |
| `qualifying` | `9c0384ed4a263df502faadd48fb04217aa243972aae84cef54d27fa55357fcbb` |
| `qualifying_probability` | `f5a84a304df101ee2a4bed8577cfdc48d14d62cf4b32ec7c1457eb9cd6a8cbc3` |
| `race_certified` | `4392de97b18731c5bbfca167ded9d3ad8393952ecebbbca385f6f7c2c453e34f` |
| `race_main` | `563c5f1360dfd8d3e85f6c3b497d46cf9f86e98d94f033661b15c70a9d7441fe` |
| `same_season_residual` | `ce22d3d254b299d3328cc43259a22bef40a13d6a54aa3ec8a5bad0e073228a23` |
| `telemetry_sequence` | `1743e0a5a492f05ec0507092b192332e33f22102a585af45a1a051472c87e4f8` |
| `telemetry_tcn` | `9c98244bd4c854b86850bda00a8dfd2454a845a2cbf83b1944c21e3bff09f999` |
| `telemetry_tcn_matrix` | `d148d512376bc7c14479ff48f07c73181fe7cf033b5a2eea8936d0067ef35197` |

## Reproduction

Use `.venv-f1/bin/python` with the repository on `PYTHONPATH` and BLAS/OpenMP threads set to 2. F1 tests additionally use the legacy F1 script directory and both service directories on `PYTHONPATH`; football tests run separately with the football script directory to avoid same-named entrypoint imports.

- Qualifying: `run_qualifying_pairwise_challenger_backtest.py --years 2026 --evaluation-years 2026 --bootstrap-samples 20000 --seed 20260713`.
- Qualifying probabilities: `run_qualifying_probability_calibration_audit.py --input <pinned qualifying artifact> --source-sha256 <its pinned digest>`.
- Race: `run_race_survival_order_backtest.py --years 2026 --evaluation-years 2026 --simulations 1000 --selection-simulations 300 --bootstrap-samples 20000 --seed 20260713`; the original 25-candidate grid is unchanged.
- Best lap: `run_best_estimated_lap_2026_backtest.py --year 2026 --rounds auto --bootstrap-samples 200000 --bootstrap-seed 20260711 --tcn-evidence <registered TCN artifact>`.
- Residual: `run_same_season_latent_residual_research.py --year 2026 --rounds auto --audit-events 3 --seed 20260713`.
- Live: `run_live_next_lap_2026_backtest.py --year 2026 --rounds 1,2,3,4,5,6,7,8,9 --weight-grid 0:1:0.05 --cold-start-ssm-weight 0 --warmup-laps 3 --live-seed 42 --bootstrap-samples 500000 --bootstrap-seed 20260711`, plus `run_live_strategy_replay_audit.py`.
- Certified grid: `run_race_certified_grid_prior_ablation.py --year 2026` with the current generation timestamp.

All runners use `--weekends-dir data/f1/raw/weekends` where supported and must receive a new `--output` path. Do not overwrite the registered frozen outputs. The evidence notebook consumes only the pinned register and fails if any artifact, code file or data dependency differs.


## Suggested focused commits

- `fix(f1-live): correct latent trajectories and persistent race state`
- `fix(f1-live): propagate sparse laps and correct pit tyre transitions`
- `fix(football): fit coherent goal probabilities with causal evaluation`
- `fix(f1): correct survival and pre-event feature contracts`
- `fix(evaluation): enforce model-bound prospective promotion evidence`
- `fix(f1-betting): validate probability fields and top-k margins`
- `evidence(models): regenerate mathematical remediation results`
