# Pre-fit mathematical review of sector-checkpoint forecasts

Reviewed on 2026-09-08, before real feature assembly, candidate fitting or selection scoring in this execution cycle. This review covers the published `models.py` and `baselines.py`, the execution protocol, and the new evaluation implementation. It does not certify the full packet parser, feature implementation, final execution lock or historical delivery latency. No historical model was fitted or scored for this review; the checks below use synthetic inputs.

The proposed experiment is mathematically defensible for its stated product: predict the next recorded eligible clean lap at an S1 or S2 observation. All four references and all three candidates must issue at exactly those same checkpoints. This is a later-information, generally shorter-horizon product than the integrated HGB forecast issued after an earlier completed lap. Passing these gates would support a historical improvement over the four sector references, not an improvement at the original HGB issuance time or proof of production latency.

The essential pre-fit changes identified here concern evaluation and input contracts, not candidate tuning. The published uncertainty helper covers only one of the two requested errors. Its permissive outcome handling also permits malformed rows to disappear. The separate execution evaluator is the appropriate place to address both issues while preserving published source bytes. Exact event-set binding, feature missingness, target identity and the final source lock must be checked before execution.

## 1. Population, target and information contract

An issuance is an immutable event/driver/packet/sector checkpoint with its own archived availability clock. It requires the pilot's attribution gate, no already observed retirement/stop, and at least three strictly earlier valid own-driver completions. Equal-clock histories and auxiliary observations are unavailable. Unsupported checkpoints stay in the ledger, and support must not depend on which model performs well or whether a later clean target exists.

The target is the first same-driver canonical eligible completed row strictly after the issuance clock. It may be the current lap or a subsequent lap. Outcome attachment occurs only after the target-free ledger is closed and hashed. The target identity is event + driver + canonical row, excluding sector; S1 and S2 resolving to one lap share one target. A common target must have one driver, value and completion time. Every issued row must receive every reference and candidate, including when the eventual target is unavailable. Unmatched outcomes remain explicit and cannot be assigned invented errors.

Consequently, point errors estimate performance conditional on a subsequent recorded eligible target existing. They do not establish performance on terminal or missing outcomes, and the supporting history is a raw observed cleanliness proxy, not final `IsAccurate`. Historical canonical eligibility can be used to define the target after issuance; it cannot select the issuance or repair its inputs. The archived stream clock and cumulative maximum timestamp policy are an explicit replay proxy, not measured historical client receipt times. A prefix-invariance test establishes dependence on that proxy prefix, not its equivalence to a live client's information set.

The frozen transfer list was independently compared with the original canonical input manifest: all 61 entries are identical, comprising 24 events in 2024, 24 in 2025 and 13 in 2026. The old manifest's lap-end issuance counts are provenance only; they are not the expected counts for this new sector population. All declared events must be accounted for, including acquisition or support failures. A favorable available subset cannot inherit the full-scope gate.

## 2. Four fixed references

For stage k in {1,2}, let x be the observed sector vector and P its sum. In each already available historical completion h, let X_h contain the corresponding first k sectors, L_h the recorded full lap, and R_h = L_h − sum(X_h). Historical sector/full-lap coherence is checked within the frozen millisecond tolerance. The remainder keeps the joint relationship between sectors and the recorded rounding residual.

When no contamination is known as of issuance, the implementations are:

| Reference | Point forecast |
| --- | --- |
| B1, last template | P + most recent R_h |
| B2, joint median of last five | P + median of the last five R_h |
| B3, recency-weighted median of last ten | P + weighted median of the last ten R_h, with age-a weight 0.2 × 0.8^a |
| B4, pace-scaled median of last five | P + median of clip(P / sum(X_h), 0.97, 1.03) × R_h over the last five templates |

These are coherent scalar-remainder calculations. In general, the median of S2+S3 is not the sum of their marginal medians. The implementation correctly avoids that substitution. The weighted median is the lower inverse-CDF median, an absolute-loss minimizer; scaling all its weights leaves the point unchanged. With a finite normalized history, alpha=0.5 would put more than half the mass on the newest observation and collapse B3 to B1. The frozen alpha=0.2 avoids that particular degeneracy; it does not guarantee distinct predictions on every checkpoint.

If pit or neutralization contamination is already known in the current epoch, each reference uses its corresponding whole-lap template estimate; B4 and the Gaussian use the last-five full-lap median. This is a fixed causal branch. It must not inspect whether the current lap eventually passes canonical eligibility. Without known contamination, P+R remains a current-lap approximation to a potentially later target. That mismatch is a real source of model error, not an algebraic error or permission to filter those cases away.

## 3. Gaussian candidate

The candidate forms samples (X_h,R_h) from the last min(10,n) valid own completions. It uses the ordinary sample covariance S with divisor n−1 and

    Sigma = 0.5 S + 0.5 diag(diag(S)) + 0.01 I.
    predicted remainder = mu_R + Sigma_RX solve(Sigma_XX, x − mu_X).
    predicted lap = sum(x) + predicted remainder.

Covariance entries and the 0.01 diagonal floor have units seconds squared. For any nonzero vector v, the positive-semidefinite sample terms plus 0.01||v||² make Sigma positive definite in exact arithmetic. The linear solve therefore has a well-defined regularized system even with three histories and rank-deficient empirical covariance. Computing the covariance with the remainder, rather than shrinking a covariance with the full lap, preserves the coefficient of the already known prefix.

This is the conditional-mean formula for an assumed joint Gaussian, whose conditional mean also equals its median. The conditioning identity is given in equation A.6 of [Rasmussen and Williams, Appendix A](https://gaussianprocess.org/gpml/chapters/RWA.pdf). That reference supports the formula and numerical regularization principle; it does not establish Gaussian F1 sectors or validate the particular shrinkage/floor values.

Here, n is small, consecutive laps need not be independent or stationary, tyre and fuel regimes change, and raw-valid history need not have the same distribution as the future canonical eligible target. In particular, a current prefix does not condition a later lap as if it were that later lap's observed prefix. The candidate is therefore a regularized empirical predictor to test, not a proven conditional median for the scored target and not a calibrated predictive distribution. Known-contamination fallback, nonpositive/nonfinite remainder checks and solve-failure fallback to B4 are appropriate. They retain the common issued population; fallback reasons must remain visible. No additional clipping or covariance search is needed before scoring.

## 4. HGB objective and weighting

The two learned candidates use the same fixed features and B4 anchor, differing only in seven versus fifteen maximum leaf nodes. Published settings are absolute-error loss, learning rate 0.05, 150 iterations, minimum 60 samples per leaf, L2 setting 20, no early stopping, and seed 20260908. Inputs permit declared NaN missingness, but not infinity. The [official HistGradientBoostingRegressor documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html) confirms the absolute-error, sample-weight and native missing-value APIs. It does not guarantee statistical gains or make a feature list causal.

For E events and N resolved issuance rows, event e contains T_e distinct targets and target t contains m_et issuances. The implemented mean-one weights are

    w_eti = N / (E T_e m_et).

Thus each event has total weight N/E, and each target within that event has total weight N/(E T_e). This matches the declared event-balanced, within-event target-balanced objective. It does not give identical total weight to targets in different events with different target counts. The published docstring's phrase about each target having weight one should be understood in this conditional sense.

Duplicating a target's observation does not change that target's relative mass within its event. Mean-one normalization can nevertheless change the absolute total weight, and duplication can affect fixed regularization, histogram construction or minimum-sample constraints. Do not claim exact fitted-model invariance to duplicated rows. The important safeguard is to prevent accidental duplicate issuance IDs and to freeze the intended normalization.

Let r = y−B4. Training clips r to [−10,10] seconds and the final correction to [−5,5]. For every correction f in [−5,5],

    |r−f| − |clip(r,−10,10)−f| = max(|r|−10, 0).

The difference does not depend on f, so label clipping alone preserves the optimum of the absolute-loss problem constrained to those correction bounds. Finite boosted-tree fitting followed by output clipping is not exact constrained optimization, so this identity is not a convergence guarantee. It does show that these nested bounds are not an intrinsic contradiction in the intended loss.

The weighted loss targets the declared weighted evaluation population. Target multiplicities are computed from historical resolved labels during training and scoring; they must never become prediction inputs. The feature whitelist, fitted arrays and serialized forecasts must independently exclude target aliases and final eligibility, since the published helper's short denylist cannot establish causality by itself. Aligning the point functional and scoring rule before evaluating follows [Gneiting, Making and Evaluating Point Forecasts](https://arxiv.org/abs/0912.0902). That source does not justify this experiment's block length or confer prospective validity.

## 5. Errors, uncertainty and gates

With absolute error a_eti for issuance i of target t, the two event errors are

    checkpoint error C_e = sum_t sum_i a_eti / sum_t m_et;
    target-balanced error T_e = (1/T_e) sum_t [(1/m_et) sum_i a_eti].

Each reported overall MAE is the unweighted mean of its event errors. These answer different questions. C weights every issued checkpoint equally within an event; T gives targets equal weight within that event regardless of repeated S1/S2 updates or a long interval before the next eligible target. Both are useful, and neither may silently substitute for the other.

For each reference and each metric, form the paired per-event difference candidate−reference. The ordinary bootstrap samples whole paired events. The declared circular block bootstrap samples consecutive three-event blocks separately within each year, truncates each resample to the original year's event count, then combines years with their original event-count weights. Calculate the percentile interval and worst leave-one-event-out mean separately for C and T. Sharing the deterministic event draws across these series preserves comparison consistency. Sorting requires canonical chronological event IDs; a count alone is insufficient.

The circular wrap from the end to the start of a season is a dependence approximation. With only three events in a year, a three-event circular block contains the same full set and produces a degenerate mean; the planned complete 22-event selection and 24/24/13-event transfer groups avoid that particular small-sample case. These are descriptive uncertainty summaries conditional on the frozen candidate/cohort, not a posterior probability of improvement, a guarantee under arbitrary cross-event dependence, or a correction for the wider history of research experiments.

The selection rule is defensible: fit learned models on 2022 only; choose the lowest 2023 target-balanced event MAE among the three frozen candidates, using the listed order for exact ties; then require at least 2% gain against every reference, a negative target-balanced three-event interval upper bound, and no worsening of checkpoint MAE against any reference. Do not select another candidate after the chosen candidate fails a secondary criterion. Selection must contain the exact 22 declared 2023 events, each with supported resolved forecasts. Zero reference MAE cannot support a positive relative-gain claim and must fail that screen explicitly.

Only a passing selection permits the frozen refit on 2022–2023 and transfer acquisition/evaluation. The substantial 2024–2025 gate requires at least 10% checkpoint-MAE and 5% target-balanced-MAE gain against all four references, negative block intervals and worst leave-one-event-out differences for both metrics, and positive gains in each year and each sector. The 2026 stress gate requires at least 5% target-balanced gain, positive checkpoint gain, both negative intervals/leave-one-event-out checks, and positive sector gains. The complete declared event scopes are required. These thresholds are meaningful design choices, not mathematical constants. The existing later seasons have research exposure; a pass would remain historical transfer evidence, with prospective claims requiring a separate subsequent evaluation.

Beating all four fixed same-checkpoint references is a materially stronger comparison than beating only a carried earlier HGB point. It still does not establish superiority over every possible sector model. A carried HGB comparison can be secondary, with its different information age and historical receipt caveat made explicit. Neither the Gaussian identity nor the new information warrants promising a large gain before execution.

## 6. Reproduced helper defects and required execution safeguards

The following synthetic probes were run against the immutable published `models.py`:

```python
import numpy as np
import pandas as pd
from research.experiments.boundary_20260908.sector_forecast.models import error_summary

f = pd.DataFrame(dict(event_key=[202301, np.nan], target_id=['a', 'b'],
                      y_true=[90., 90.], p=[91., 190.]))
assert error_summary(f, 'p')['checkpoint_event_mae'] == 1.
# Both rows count as resolved, but pandas drops the null-event error of 100.

f = pd.DataFrame(dict(event_key=[202301, 202301], target_id=['a', 'b'],
                      y_true=[90., np.inf], p=[91., 100.]))
assert error_summary(f, 'p')['unmatched_issued'] == 1
# A corrupt infinite outcome is misclassified as an unavailable target.
```

Also, `models.paired_uncertainty` uses only `per_event_target_balanced_mae`. Its output cannot establish checkpoint-error interval or leave-one-out requirements. These defects do not invalidate the fixed prediction formulas. The new execution evaluator must replace the permissive scoring path, reject malformed event identities and infinite outcomes, accept missing targets only with an explicit unmatched status, require consistent target labels/time, and compute both metric series.

Before locking execution, validate exact declared event sets, unique issuance keys, sector-independent target IDs, common prediction coverage and strictly later target timestamps. The data contract must allow declared NaN features, consistently with the feature contract and HGB; only the five baseline/Gaussian points must all be positive and finite. No unsupported feature, missing source or invalid model output may quietly remove a model-specific row. The implementation must bind its ordered feature list, source files, canonical labels, train/selection partition and pre-fit synthetic tests before any historical assembly or scoring.

## 7. Verification boundary and source bindings

Fresh synthetic checks: 28 baseline tests passed in 0.08 seconds; five published model/weight/scoring tests passed in 1.10 seconds. The latter's only `fit_residual` call fails at its forbidden-feature guard before constructing or fitting an estimator. The synthetic malformed-input probes above reproduced exactly. Runtime for these checks: NumPy 2.5.3 and scikit-learn 1.9.0, one BLAS/OpenMP thread, Python bytecode and pytest cache disabled. No real candidate fitting, selection scores or transfer scores were produced or consulted for this review.

Published source SHA256 bindings:

| File relative to sector_forecast | SHA256 |
| --- | --- |
| models.py | 0886c52b2416b55382c359342056991d5492a210534507f3470d7d7942b8d24d |
| baselines.py | 2a8dfffb21ab4c098d5445bad6ee293d3917075dc1779f92e684d821486c4a47 |
| test_models.py | 72215f7a6be5fa344cd0f9007ec69c73504afeff970013241a91802b4cf166cb |
| test_baselines.py | 0627b45be94743fa1db8d114579c359c1e083ebf4e799d6adac78afbe1de6deb |

The execution specification, evaluator and data/feature contracts were still under pre-fit construction when this review began. The final pre-fit review below closes the identified implementation issues and records the reviewed snapshots. The execution lock must bind those sources and its own test receipt; the published-file bindings above do not substitute for that lock.

## 8. Final execution review and readiness

**Ready for the frozen discovery execution.** The reviewed implementation has no remaining critical pre-fit blocker within this review's scope. This conclusion authorizes neither a predictive-gain claim nor production activation: no historical sector feature build, real estimator fit or real selection score was performed by this reviewer.

The follow-up read covered `execution/run.py`, `evaluate.py`, `data.py`, `features.py` and their tests. The following earlier issues are closed:

- The new evaluator computes paired uncertainty and leave-one-event-out differences independently for both metrics; it does not use the published helper's one-metric uncertainty output for gates.
- Invalid/null event IDs, infinite targets, inconsistent repeated target labels/times, split S1/S2 target identities and incomplete model predictions fail validation. The exact declared 2023 event set is checked after canonical event-ID normalization. A zero reference MAE cannot pass a positive relative-gain screen.
- The data contract now consistently permits explicit NaN feature missingness while requiring positive finite point forecasts.
- The runner now uses the assembler's explicit `diagnostics['terminal']` boolean. It previously recognized only `Ends`, which incorrectly classified unmatched rows from `Finished` or `Finalised` archives as incomplete. Four synthetic status regressions now cover `Finished`, `Finalised`, `Ends` and nonterminal `Started`.

The assembled data path retains every positive pilot checkpoint, applies one shared source-defined support gate, and joins context by exact event/driver/packet/sector identity plus attribution metadata. Completed histories enter only at a strictly earlier global availability clock, and the full iterator's input-integrity result is checked before publishing an event. No ordinal epoch from the completion parser is joined to a pilot epoch. An unchanged S1 display repeated in an S2 packet is normalized only using matching immutable prior S1 support from that pilot epoch. A failed association aborts the input rather than removing a difficult model row.

The feature encoder's 75-column order matches its frozen contract. S1 cannot read current S2/S3; S2 cannot read current S3. Optional source metadata is checked before its value, prior own histories are filtered before their numeric payload, and tyre fields remain observed reports with their own availability information. Source identifiers serve pairing and provenance, not numeric model features. The numeric transforms, stage missingness, Gaussian fallback indicator and historical joint remainders match the declared definitions.

The runner verifies the source/input lock, builds all 44 target-free event ledgers, closes and hashes them, and only then attaches training labels. Each learned candidate receives only 2022 resolved training rows. The 2023 frame is constructed from target-free records; both learned prediction arrays are persisted and hash-bound before any 2023 target attachment. Global `issuance_id` replaces the event-local pilot ledger ordinal in the scoring frame. After attachment, exact issuance ordering and feature/reference/Gaussian equality are checked before joining learned predictions by that order. Unsupported and unmatched records remain in the saved labeled ledgers. Sector summaries use the same selected candidate, with no sector-specific candidate choice.

An additional independent synthetic runner replay exercised that control flow using dummy predictors, with no estimator fitting. The spy enforced this exact sequence:

    attach 2022 labels;
    fit-input check for HGB7 on 2022 only; predict from outcome-free 2023 inputs;
    fit-input check for HGB15 on 2022 only; predict from outcome-free 2023 inputs;
    verify closed prediction file and issuance-lock hash;
    attach 2023 labels; verify final common-row and target grouping.

That replay passed with 44 synthetic issued rows spanning all 22 selection events and 22 shared S1/S2 targets. Its disposable files were confined to a temporary directory. It produced no historical forecast or fitted model. A fresh run of the complete execution test directory, after the terminal and event-ID corrections, passed **117 synthetic tests in 1.47 seconds**:

```sh
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=. .venv-f1/bin/python -m pytest -q --import-mode=importlib -p no:cacheprovider \
research/experiments/boundary_20260908/sector_forecast/execution
```

Final reviewed execution snapshots, SHA256:

| File relative to sector_forecast/execution | SHA256 |
| --- | --- |
| run.py | 7f178839f4a677f49df24e014ca1b069e81c4862b160ae425984efffe55384dc |
| evaluate.py | 6be893a852495df15b4d3222d8e1ffcd27d73c171f8d95c4e4cf1b15ba9821e5 |
| data.py | 50d40552ef6d34e2d7891fbac7d4ea7c3e37df8c1cd057bab8ea612394c38936 |
| features.py | 4052f16e67e842286171ffe2df1124babf8bfca9cc442d253d8bd1e144593cef |
| test_run.py | 81dd79604ec978e1fe043bca98205fbf83a0e7d12678a40779c5b35cea960f98 |
| test_evaluate.py | ecbf1b401e8c37bd33fb2bedb39dffacb76972da68f0e1098dd1ae5fcbd22a65 |
| test_data.py | 0a4c184c195a8821f226fc12f3379f4b2f5c062d6dfd9b1cc81a6ebae70ed211 |
| test_features.py | 144d5496b6f2e695c1087a95603460c70cbb6af9db4b170e16162c2101460ea7 |
| specification.json | c5ca0a95315c06d21455b72f8b2b659b066268eee9fc93fa0566be6755d7ec71 |
| data_contract.json | 612818fe0fa362348e38367e1f93145c4444e3153365aaf533998b17c214dd02 |
| feature_contract.json | b3398db61b856cf2f564c43ab6ebc7713ffc4c870b0b5b6c4f49801c92b9e53f |

Freeze these reviewed sources and the final pre-fit test receipt before real assembly. The retained limits are the conditional observed-target population, archived receipt proxy, small empirical Gaussian history, historical research exposure and descriptive rather than prospective uncertainty. They are explicit scope limits, not unresolved arithmetic or wiring defects.

Suggested commit: `research(f1): review sector experiment mathematics and pre-fit gates`.
