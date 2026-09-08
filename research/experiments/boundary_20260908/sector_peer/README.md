# Proposed experiment: shared raw-sector innovations

**Recommendation: test a small, leave-driver-out dynamic factor model of fresh sector surprises.** The new information is another driver's already received S1/S2, or raw-valid completed S3, before the recipient's next clean-lap outcome. This is a proposal and a metadata feasibility check: no filter, candidate, correlation, prediction score or new acquisition was run for this proposal. All frozen lanes remain unchanged.

## Why this is a different experiment

Checkpoint cycle2 already tested current clean **whole-lap** peer medians, same-compound cohorts, paired same-driver pace changes and own-minus-field offsets. All four residual variants failed its 2023 screen. Repeating these aggregates with more trees is not the proposed mechanism.

The sector execution now supplies an earlier, source-audited observation stream. Its strongest learned candidate, HGB15, reduced 2023 target-balanced error versus the last-template reference by 6.55%, but the block interval crossed zero and advancement failed. Its 75 features contain only own history and context. A shared sector filter could identify a coherent change across independently observed drivers, separate it from persistent driver-specific surprises, and quantify stale/sparse support. Peer S3 is particularly relevant when the recipient has reached S2 but has not yet observed its own S3. This is a hypothesis; these data do not identify fuel load, pure track evolution, causal weather effects or canonical tyre age.

## Data support checked without outcomes

I checked the first six predeclared 2022 events only. For each issued checkpoint, a peer must be a different driver and have an issued S1/S2 with known uncontaminated context. Its availability must satisfy `max(recipient_last_completion, checkpoint−180s) < available < checkpoint−15s`. Three distinct peers must support the same sector.

| Event | Issued checkpoints | At least one sector supported | Fraction |
| --- | ---: | ---: | ---: |
| 202201 | 2,075 | 1,762 | 84.92% |
| 202202 | 1,481 | 1,115 | 75.29% |
| 202203 | 1,756 | 1,169 | 66.57% |
| 202204 | 1,982 | 1,627 | 82.09% |
| 202205 | 1,928 | 1,458 | 75.62% |
| 202206 | 2,281 | 1,664 | 72.95% |
| Total | 11,503 | 8,795 | 76.46% |

The check reads only saved support/identity/clock metadata; it does not calculate predictive errors. S3 support has **not** been measured. Coverage is sufficient to justify a source-level prototype, not evidence of predictive signal.

## Fixed proposed model and units

For peer driver j, sector s and received time t, form a dimensionless log surprise

\[
z_{j,s,t}=\log S_{j,s,t}-\operatorname{median}_{h\in H_{j,t}^{(3)}}\log S_{j,s,h},
\]

where H contains three **strictly earlier raw-valid completions**, never final CSV eligibility. The observed history defines the normalization, not a physical driver-strength parameter. For each sector, maintain a linear state with shared component g and driver nuisance components b:

\[
z_{j,s,t}=g_s(t)+b_{j,s}(t)+\epsilon_{j,s,t},\qquad
x(t+\Delta)=F_\Delta x(t)+\eta_t.
\]

Use independent sector filters. Shared-state decay is exp(−Δ/τ), with τ in {60,180} seconds; nuisance decay is exp(−Δ/600). Initial means are zero. Stationary standard deviations are 0.01 for g and 0.02 for b, in log units; diagonal process variance is σ²(1−exp(−2Δ/τ)). These are fixed regularization assumptions, not fitted physical parameters. A known received compound change or pit-out resets only that driver's nuisance mean/cross-covariances and variance; it does not reset shared state or reconstruct tyre life.

Measurement variance is max((1.4826 MAD(log prior-sector values))², 0.005²), from the last up to five prior valid own completions, minimum three. The mean-reverting, zero-centered nuisance prior and fixed unit loadings resolve the statistical level convention. Common-versus-driver attribution remains prior-dependent; correlated pit cycles can still masquerade as shared changes.

For each recipient d, use a separate filter that has **never assimilated driver d's timing values**. Create driver states only when their first observation arrives; no final-event roster is needed. With measurement vector h selecting shared state plus peer j's nuisance state:

\[
\nu=z-h^T m^-,\quad V=h^TP^-h+R,\quad
w=\min(1,2.5\sqrt V/|\nu|),\quad R^*=R/w,
\]

with w=1 when ν=0. Then K=P⁻h/(hᵀP⁻h+R*), m=m⁻+Kν and

\[
P=(I-Kh^T)P^-(I-Kh^T)^T+KR^*K^T.
\]

This is a single-pass, bounded-influence variance-inflated Kalman update. It is a specified approximation, **not** an exact Student-t posterior. Its working covariance is a reliability feature, not a calibrated lap-time interval.

The methodological basis is the dynamic-factor representation with serial idiosyncratic states and nonsynchronous observations in [Bańbura and Modugno, ECB Working Paper 1189, sections 2.1–2.3](https://www.ecb.europa.eu/pub/pdf/scpwps/ecbwp1189.pdf). Their news decomposition motivates weighting unexpected releases by information content rather than treating raw levels equally. This proposal uses forward filtering only; their full-sample EM/smoothing and backdating are not permitted at historical checkpoints. Robust heavy-tail filtering is a separate established approach in [Roth, Özkan and Gustafsson, ICASSP 2013](https://www.2013.ieeeicassp.org/Papers/ViewPapers_MSe515.html?PaperNum=3185); the approximation above is our explicit design, not a claim to reproduce that paper's algorithm or empirical results.

## Online source-time contract

- Preserve the exact v2 issuance IDs, support population, 75 own features, five original points, terminal status and target associations. No model-specific row deletion. Unsupported peer evidence returns the exact frozen HGB15 point.
- Primary peer cutoff is u=t−15 seconds; only sources with `available_ms < u` enter. Filter updates run in source order up to u. Query the filtered state at u; do not smooth earlier states using later arrivals. Store the causal filtered snapshot at each own completion to form later state changes. Same-clock peers cannot inform each other. Source time and actual historical receipt time remain different; this lag is a sensitivity assumption, not receipt certification.
- S1/S2 measurements use first pilot-candidate sector observations, already known outside pits and without known neutralization or unknown context coverage, with three earlier raw-valid own completions. Read current atomic values at their actual source clock. Do not re-assimilate a later revision of a consumed sector as an independent sample or rewrite an earlier state/forecast. Record revision counts. A corrected S1 not previously consumed may enter once through its first supported observation; no backdating.
- Add S3 **only** from `observed_valid_completed is True` records at completion availability. Do not also ingest that record's S1/S2; they were separate measurements. Use same-driver history strictly before that completion to normalize S3. No canonical current lap number, final IsAccurate, future control state or active-stint reconstruction.
- At a recipient checkpoint, require at least three different fresh peer drivers in one sector, with sources strictly newer than both the recipient's last completion and t−180s. Count each driver's latest eligible observation once per sector for support. Other sectors without support have explicit missing features. Known recipient contamination also uses exact HGB15 fallback.

## Small family and evaluation, to freeze before implementation

Add 18 columns: for each sector, shared-state mean, working standard deviation, change from the stored state at the recipient's last completion, distinct fresh peer count, mean age and maximum age. Driver identities index online states only; they are not extra fitted HGB predictors.

Fit **four models total**, all using the unchanged HGB15 architecture, B4 anchor, clipping, event/target-balanced weights and original 2022 labeled training rows:

| Model | Role |
| --- | --- |
| Shared-state τ=60s | Selectable candidate |
| Shared-state τ=180s | Selectable candidate |
| Fresh peer-median control | Nonselectable mechanism control; replace state mean/uncertainty/change with causal median surprise, MAD/√n and stored-median change; same 18-column shape |
| Delayed τ=60s control | Nonselectable timing control; factor snapshots are delayed another 180s, retaining the same real support gate and actual older ages |

Fit augmented HGBs directly from causal features; do not stack in-sample HGB residual predictions as training features. The original HGB15 pickle stays fixed as comparator and fallback. The controls distinguish a latent-factor benefit from merely adding fresh peers, and fresh information from slowly varying historical context. All four results, failures and support distributions must be retained.

Choose only between the two state candidates by 2023 target-balanced event MAE, deterministic listed-order ties. This is another **retrospective selection on exposed 2023**, not pristine validation. Report both event-balanced checkpoint MAE and event-balanced target MAE, all 22 events, S1/S2 subgroups, unmatched coverage, 20,000 paired event resamples, within-year circular three-event block intervals and leave-one-event-out deltas. Reproduce every serialized forecast and independently re-resolve CSV targets after forecast closure.

Stop unless the selected model gains at least 2% on both metrics versus frozen HGB15, passes the original all-four-reference selection gate, has negative upper block endpoints versus HGB15 for both metrics, and improves target-balanced point error versus both controls without worsening checkpoint error. Recompute the selected model at peer lags 0 and 30 seconds without refitting or reselection; require improvement versus HGB15 on both metrics at 30s. Report the support lost to lag separately: a longer lag can mechanically remove all evidence newer than the recipient's latest completion. A gain confined to a single race or sector fails: both sectors and every leave-one-event-out mean must improve versus HGB15; also report per-driver contributions and the worst single-driver omission.

Only then consider new 2024–2025 sector acquisition. The substantial target remains ≥10% checkpoint and ≥5% target-balanced gains versus HGB15 and every original reference, with each year/sector improving and both block/LOO checks passing; later 2026 is separate historical stress. Previously exposed outcomes do not become prospective merely because raw sectors were newly acquired.

## Minimum independent test before any candidate fit

1. Rebuild the first six 2022 events from raw streams in a new directory and independently reconcile every proposed peer measurement with packet/driver/sector/epoch/source time. Reproduce the S1/S2 support counts above; measure S3 separately. Stop if S3 cannot supply three independent, newer peer observations for at least 25% of S2 checkpoints in at least four of those six events. This is a predeclared feasibility threshold, not a prediction-score threshold.
2. Synthetic latent-state tests must recover a planted shared shock with fixed driver offsets and irregular arrivals, distinguish one-driver contamination, and preserve positive-semidefinite working covariance. Appending future sources, changing the recipient's timing payload values while keeping source/query clock metadata fixed, adding equal-time peers, duplicating a revision or permuting atomic keys must not alter the earlier leave-driver-out state. A completed S3 cannot appear at its physical crossing time before its source availability.
3. Verify exact original 75-feature/reference/issuance parity and deterministic no-support fallback. Predeclare new source hashes, fit matrix hashes, all constants, train/selection dates and CPU limits before fitting. Cap a first-six-event feature extraction at five minutes on one numerical thread; optimize the same equations if needed, without inspecting errors or changing the candidate grid.

## Pinned local evidence

| Evidence | SHA256 |
| --- | --- |
| `checkpoint/cycle2/results.json` | `47b30f1b2d8b0fd185eca8160421c5c4b4f61462262c2159a9304b0cf5a769de` |
| `sector_forecast/acquisition.json` | `6e4f3b16c7667a749c07dfe160e737832069729430cb4cead1eb71234971f37d` |
| `sector_forecast/execution_v2/data_lock.json` | `256f866083846b955e8bc66771006797622d8fc0af67fba6e3ca868273062fb5` |
| `sector_forecast/execution_v2/selection.json` | `dfe4f6f4d15fc47750a49e8bf3d55eadb3c5f66d6feaa14a1a7ead172bab4adc` |
| `sector_forecast/execution_v2/models/sector_hgb_l15.pkl` | `08a226ace9c7437c73881c5c4206d6dec29389b07833395229c3679f02710734` |

Paths in this table are relative to `artifacts/research/boundary_20260908/`. The data lock pins all six metadata-check inputs and the full existing 44-event discovery population. No 2024+ sector acquisition is authorized by the failed previous experiment; this new proposal needs its own tested, locked execution.

Suggested commit: `research(f1): propose causal shared-sector innovation experiment`.
