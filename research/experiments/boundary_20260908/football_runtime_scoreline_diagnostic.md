# Football scoreline replay: missing runtime state

**The closed 2,280-match artifacts cannot support a paired full-history equal-DC versus controlled production-default scoreline NLL comparison with zero new fits.** The full-history comparator retains its goal-model parameters; the runtime comparator retains only its final home/draw/away probabilities and fit diagnostics. Those marginal probabilities do not determine the probabilities of individual scores within each outcome.

This is a post hoc operational artifact audit on the old England, Spain and Italy 2024/25–2025/26 population. Inspection completed at `2026-09-08T09:46:11.564557+00:00`, at HEAD `332de1cc52ad0acc13804def084a2e723a1857df`. It performs **zero new fits, zero new forecasts, zero new model scores, and zero downloads**. It changes no production policy, frozen source, earlier research decision, or current Elo advancement gate. The previously computed 1X2 comparison remains in [football_runtime_gap.md](football_runtime_gap.md).

## What is actually retained

| Closed feature artifact | Matches | Fit blocks | Complete full-history equal-DC parameters | Runtime goal-parameter records |
| --- | ---: | ---: | ---: | ---: |
| `features_evaluation_E0.json` | 760 | 12 | 12 | 0 |
| `features_evaluation_SP1.json` | 760 | 12 | 12 | 0 |
| `features_evaluation_I1.json` | 760 | 12 | 12 | 0 |
| Total | 2,280 | 36 | 36 | 0 |

The files are under `artifacts/research/boundary_20260908/football/`. Each full-history model at `fits[j].base.dixon_coles.dc_equal` contains `attack`, `defense`, `home_intercept`, `away_intercept`, and `rho`. These suffice to reconstruct its two goal rates for every saved fixture using `packages/football/mrp/joint.py:25–38` and its joint distribution using `score_distribution.py:76–101`.

Every corresponding `fits[j].production_default` has exactly three keys: `model_diagnostics`, `calibration`, and `populations`. None contains fitted goal coefficients, fixture goal rates, a joint probability matrix, or within-outcome conditional score probabilities. Its diagnostics retain training membership, objective and convergence information; these are not a serialized optimum from which forecasts can be recovered without optimization. The forecast rows retain final `probabilities.production_default_dc_auto`, a three-element H/D/A vector, plus observed goals and ordinary fixture metadata. They contain no additional runtime goal-distribution state.

The omission is explicit in the original writer: `research/experiments/boundary_20260908/football/run.py:93` fits the runtime comparator, line 105 stores its final 1X2 vector, and lines 108–110 serialize only its diagnostics, calibration method name, and partitions. A bounded inventory of both old football artifact directories—`boundary_20260908/football` and `performance_20260907/football`—found no `.pkl`, `.pickle`, `.joblib`, `.npz`, `.npy`, or `.parquet` assets providing an alternative saved estimator. This conclusion concerns those closed artifacts; it is not a claim that no external copy could exist.

All 19 source-file hashes embedded in the old evaluation still match current bytes. Its three feature-artifact hashes match, its prior verification is passed and binds the same evaluation, and the combined population contains 2,280 distinct match IDs. No rows were removed to obtain a more convenient comparison.

## Why the saved 1X2 gain is insufficient for this pair

The canonical runtime obtains raw DC probabilities and reconciles them to the selected 1X2 vector at `packages/football/mrp/prediction.py:281–287`. Write the raw score probability as \(r(h,a)\), the result class as \(c(h,a)\), its raw class mass as \(R_c\), and the selected normalized class probability as \(q_c\). The implementation at `score_distribution.py:35–53` gives

\[
p(h,a)=r(h,a)\frac{q_{c(h,a)}}{R_{c(h,a)}}
       =q_{c(h,a)}\,r(h,a\mid c(h,a)).
\]

The archived final \(q\) identifies the first factor, not the conditional distribution in the second. For the proposed comparison, full-history DC and the older runtime fit generally have different goal rates and dependence parameters, so their conditional score distributions also differ. The observed-score NLL difference contains **both** a 1X2 term and a conditional-score term. The previously measured 3.432267% 1X2 NLL improvement therefore supplies no scoreline NLL estimate. Scoring only the reconstructable full-history model would not fill in the missing paired reference, so no one-sided scoreline result is reported here.

The saved final 1X2 vectors mean that missing calibrator coefficients alone are **not** the essential obstacle. For retrospective score reconstruction, the missing ingredient is the runtime raw DC kernel: its actual fitted coefficients, or per-fixture `(lambda_home, lambda_away, rho)`, or the corresponding score matrix/conditional score distribution.

## A useful distinction: a pure 1X2 overlay on one unchanged kernel

For a different, explicitly defined future integration, suppose the old and new models retain the **same** conditional score kernel \(k(h,a\mid c)\), and only replace old 1X2 probabilities \(p_c\) with new probabilities \(q_c\). For an observed score \((h_i,a_i)\),

\[
\Delta_i^{\mathrm{score}}
=-\log\!\left(q_{c_i}k(h_i,a_i\mid c_i)\right)
 +\log\!\left(p_{c_i}k(h_i,a_i\mid c_i)\right)
=\log(p_{c_i}/q_{c_i})
=\Delta_i^{\mathrm{1X2}}.
\]

Consequently, the paired mean difference, each subgroup difference, and every paired bootstrap/leave-one-out difference are identical when the populations, weights and resamples also agree. This cancellation requires positive old/new class probabilities, positive conditional probability for each observed score, identical score support and conditional kernel, and the actual final normalized vectors used in both joint forecasts. It does not apply to the full-history versus truncated-history DC pair above.

There is a small implementation detail to freeze rather than assume: `ScoreDistribution.reconcile` can rebuild a tighter grid when the new class weights amplify the tail bound (`score_distribution.py:45–49`). Applying it twice with different class targets does not automatically guarantee bit-identical finite-support kernels. A future overlay must explicitly use a common sufficient grid/conditional kernel for this exact identity, or quantify the resulting numerical discrepancy. Its support must be chosen without observing the eventual score. This algebra could avoid unnecessary new goal-model fits for an eligible pure 1X2 overlay; **no new numerical Elo claim, promotion, or forecast is made here**. It also does not establish calibration of totals, expected goals, or other score-derived outputs.

## Minimal next assessment for the changed DC fitting policy

1. Preserve this failed zero-fit feasibility attempt and the old predictions. Recover any separately archived runtime parameters if available; otherwise predeclare a new reconstruction run under the original source, input hashes, 36 fit cutoffs, historical partitions and optimization settings. Reconstructing the raw runtime kernel requires its 36 DC fits; reproducing the whole original caller additionally includes its calibration behavior. Do not substitute full-history parameters for the missing reference.
2. Save the actual runtime DC parameters or per-fixture rates and rho, with fit IDs, admitted training IDs and source/input hashes. Preserve the old final 1X2 vectors for the retrospective reconciliation. If the full default forecast is regenerated, verify parity with those vectors and report any difference instead of silently replacing the reference. The full-history comparator's 36 saved parameter sets can be reused without refitting.
3. Close and hash target-independent joint-forecast state for all 2,280 original fixtures before the scoring step. Predeclare the tail/support rule and whether the estimand is the exact finite matrix emitted by the runtime or an analytic infinite-support distribution. Do not expand the matrix after seeing an out-of-support outcome or silently clip score likelihoods.
4. Independently reconstruct both distributions and compute paired exact-score NLL, with the same declared fixture population and country-season block uncertainty. This would still be a post hoc assessment on previously inspected dates. Keep any subsequent causal holdout and production decision separate, and do not change the current Elo experiment's frozen criteria.

## Checked artifact and source bindings

SHA256 values below pin the inspected state. The full original 19-file source map remains in `evaluation.json.source_files`.

```json
{
  "artifacts/research/boundary_20260908/football/evaluation.json": "1c1d037ffbdf00a693728254e630792315ee7db8d8f79cbdb3545d34df66a795",
  "artifacts/research/boundary_20260908/football/verification.json": "3a6328942dd4fe0708aa3539d2f89464106f11fc727abbbf7c15bfbbef89356a",
  "artifacts/research/boundary_20260908/football/features_evaluation_E0.json": "d499ff3b88e6c3c1e6d71de5f817cf115c28814a177d27474b932f61eed986e0",
  "artifacts/research/boundary_20260908/football/features_evaluation_SP1.json": "2ca1b7efa0b22d0d67a080b9f200a09555de1542b849e2706977d8496aba0437",
  "artifacts/research/boundary_20260908/football/features_evaluation_I1.json": "12513263b09f03e77d35edf0b8741fbb05789acd8f5c623d8e4a0457ef623090",
  "research/experiments/boundary_20260908/football_runtime_gap.md": "f163df31cfce7e535dfb5e59402ee8d609c05e55ae5c26964c6946584826e2f6",
  "research/experiments/boundary_20260908/football/run.py": "d735042fb251e324012c625ab5e51ce265c3498bf53fc175eca43f45eca81083",
  "research/experiments/boundary_20260908/football/models.py": "fe66fd4d399efe8249d1887820884cd576541e06769d3289109e11d64d01c971",
  "packages/football/mrp/prediction.py": "c5085b32c8a33efa485cac87ce2c2b279a23cff97b0fe04fdeb10f383cb90918",
  "packages/football/mrp/score_distribution.py": "904c747f36658050112600fa31ee2392f289fab96ec292a66a2552f9dcfb0bfa",
  "packages/football/mrp/joint.py": "cab06d09c0b827cf37de34b1ad89793a1f442aded2d72ca0d0eeb35027559b4b"
}
```

Suggested commit: `docs(football): record the missing joint-score replay state`.
