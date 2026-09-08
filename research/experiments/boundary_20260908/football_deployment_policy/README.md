# Fixed football deployment-policy assessment

This is an operational comparison of **one fixed policy**: use all causally
admitted history for equal-weight Dixon–Coles fixture forecasts, with
calibration off. The reference is the actual current default: Dixon–Coles
fitted to the oldest chronological fit prefix, followed by automatic held-out
calibration. There is no hyperparameter search, alternative candidate, blend,
or claim of a new best research model.

The comparison uses exactly the existing 36 fit blocks and 2,280 fixtures from
England, Spain and Italy, seasons 2024/25 and 2025/26. These dates and their
1X2 results have already been inspected. The previously observed 1X2 gap is
motivation, not prospective validation. The earlier shot, static-market and
Elo research decisions remain unchanged. At protocol creation there are zero
new fits, forecasts or scoreline scores.

## Fixed policies and faithful caller reconstruction

Both policies receive the same ordered, previously admitted history at each
frozen block cutoff. The old 1,095-local-calendar-day history window and
two-month refit blocks stay unchanged. “Full history” means this complete
admitted block history, not every cached match or a new history-window choice.
Only results available at or before the cutoff are admitted; match dates must
strictly precede it. Preserve the exact original IDs, UTC clocks, provider
completion-day convention and resumed Fiorentina–Inter fixture.

The reference must execute the canonical `packages.football.mrp.prediction`
path with `football_model=dixon`, `football_calibration=auto` and no decay,
using the exact block input records. Capture its actual goal parameters and
calibrator state. Compare its final 1X2 probabilities with the saved
`production_default_dc_auto` vector for every fixture. A mismatch stops the
attempt before new scoring; the old vector cannot silently replace a
different current caller.

The candidate uses the same canonical Dixon–Coles objective, priors,
constraints, optimizer settings, neutral-team handling, score builder and
output semantics. Its only differences are the full admitted fit population
and calibration off. Its final H/D/A vector must match saved `dc_equal` within
the predeclared absolute tolerance of 1e-10, with relative tolerance zero.
The saved full-history parameters are reusable if input, source and complete
caller/model parity are proved. Otherwise perform exactly one full-history
fit per block under unchanged settings, and require the same parity. A
parity failure does not authorize tuning, a third policy or choosing the
better reconstruction.

The fit budget is 36 restored prefix DC fits plus their 36 automatic
calibration-policy calls, and at most 36 full-history DC fits. The automatic
calibrator's existing internal algorithm choice is part of the frozen
reference, not a new search. Reuse captured prefix estimators when needed for
caller parity rather than fitting them twice. No shot, Elo, GBDT or hybrid
research fits are needed. An unused GBDT branch may be suppressed in the
harness only after synthetic differential tests prove that this leaves the
selected Dixon outputs unchanged; do not replace DC, calibration, population
splitting or reconciliation with a favorable approximation.

## Joint-score estimand and proof before labels

Save both complete target-free probability matrices, their dimensions, raw
goal rates and rho, final 1X2 probabilities, expected goals, mode score,
omitted-mass bound and reconciliation metadata before attaching scoring
labels. Use the actual canonical finite matrix: tail tolerance 1e-12,
initial `min_max_goals=1`, with only the existing parameter/selected-probability
driven reconciliation refinement. No observed score may choose support.

One shared scoring function reads each policy's saved matrix at the observed
integer score. The policies have different conditional score kernels; only
the scoring rule is shared. An out-of-support or zero-probability score has
infinite NLL and prevents advancement. Do not expand support after observing
the target, clip likelihoods, remove fixtures or substitute an analytic
infinite-support distribution. Undefined bootstrap results fail the gate and
must be represented explicitly in strict JSON, never as `NaN` or a finite
fabricated loss.

For raw score mass r, raw result-class mass R and selected class probabilities
q, reconciliation is p(h,a)=r(h,a)q[c(h,a)]/R[c(h,a)]. Both r and q must be
retained. A 1X2 gain does not identify a joint-score gain when r changes.
The canonical shadow-evaluation path currently expands support using observed
goals and floors its score probability; that historical diagnostic is not the
new scoreline estimand and must not supply this assessment's joint NLL.

## Acceptance, fixed before new joint scores

Use the same complete 2,280-row population for every comparison. Require all:

- At least 2% pooled 1X2 NLL reduction versus the actual reference and a
  strictly negative paired 28-calendar-day 95% upper endpoint.
- Positive 1X2 NLL improvement in each country; nonworse 1X2 NLL and
  class-summed Brier in every country-season.
- Nonworse pooled joint-score NLL, a paired 28-day 95% upper endpoint at most
  zero, and nonworse joint-score point loss in each country.
- Complete population, causal history, serialized-state replay, coherent
  score-output and canonical-caller parity checks all pass.

Use 4,000 paired fixed-calendar-block resamples, seed 20260908, separately
within each country-season, weighted by the original fixture counts. Report
1/7/28-day intervals with 28 days governing acceptance. Lower loss is better;
deltas are candidate minus reference. There is no retrospective change to
gates, multiple-comparison correction claim, future-season guarantee or
country exclusion.

## Historical assessment and future deployment remain separate

Keep the disjoint fit/calibration/selection/test assessment and its original
parameter lineage. A future-fixture model may use all outcomes causally
available at that future cutoff, including rows formerly reserved for a
historical assessment. It must not then score itself on those same rows and
label the result out-of-sample. Report evaluation-model fit IDs separately
from fixture-model fit IDs and state which produced every metric/output.

No production activation occurs during this protocol. If both numerical and
caller-parity gates pass, the already authorized integration can proceed with
canonical API/CLI/default-policy and regression verification; that decision
is scoped to the default Dixon policy, not untested GBDT/hybrid options.
Keep the reference source snapshot and historical reports intact when the
future implementation changes. A failed gate or execution leaves the current
policy active and stops the attempt.

`specification.json` binds the old evaluation, verification, three feature
artifacts, their 27 raw CSV hashes, original acquisition manifests, canonical
source files, and the two completed runtime diagnostics. It intentionally
does not bind unrelated publication documents still under construction.
Before execution, freeze the new reviewed implementation, runtime, tests and
these protocol files. Use exclusive attempts and closures, preserve failures,
one CPU thread, a 2 GiB RSS limit and one hour per stage. This document itself
implements and runs nothing.

Suggested commit: `research(football): specify full-history DC deployment proof`.
