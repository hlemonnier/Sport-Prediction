I reviewed the repository code and ran targeted checks on the uploaded project. I did not rely only on the circuit-card layer; the biggest issues are actually in evaluation, race target design, leakage, and probability/betting consistency.

Blunt verdict: **current overall project logic = FAIL for rigorous predictive claims and betting use; PASS only as an exploratory F1 ranking prototype.**

Blunt verdict on circuit-card implementation: **FAIL as a mathematically effective circuit-specific modeling layer; PASS as basic data plumbing/name mapping.** The card lookup, aliases, and São/Sao normalization are fine, but most circuit features are not coupled strongly enough to the race-generation logic, and several “interactions” are rank-invariant in the fallback path.

The most important dynamic check I ran was this: changing `track_qualy_importance` from 0.10 to 0.95 produced the exact same race fallback score ordering when the only change was the circuit-scaled qualifying features. That is because `_hierarchical_fallback()` re-ranks `qualy_position_circuit_importance_adj`, `qualy_position_track_adj`, and `qualy_position`; multiplying every driver’s qualifying position by the same event constant cannot change the within-event rank. This directly explains why Monaco did not materially change.

## Top 10 issues, ranked by severity

**1. The backtest/evaluation framework is not measuring what the project thinks it is measuring.**

In `run_experiment.py`, `_prediction_payload()` returns `rows` from `result.table`, and `result.table` is only the top 10 via `format_prediction_table(output, top_n=10)` in `rqp/prediction.py:1133-1134`. Then `_run_backtest_ablation_compare()` evaluates those top-10 rows in `run_experiment.py:719-725`. So the reported MAE is often **MAE on predicted top-10 rows only**, not full-field rank MAE.

That makes the ablation numbers hard to interpret. A model can improve “MAE” simply by changing which top-10 drivers are included, while ignoring the rest of the field. The full-field rows exist in `extras["all_prediction_rows"]`, but the main backtest path does not use them.

Expected impact of fixing this: reported MAE may move materially. This will not improve the model, but it will make the metrics honest. Failure mode: older reports will no longer be comparable.

Concrete fix: in `run_experiment.py`, evaluate `payload["all_prediction_rows"]` when present, not `rows`.

```python
eval_rows = payload.get("all_prediction_rows") or rows
evaluation_row = evaluate_prediction_rows(
    predicted_rows=eval_rows,
    actual_results=actual,
    actual_position_col="position",
)
```

Also prefer `driver_id` matching over `driver_name` in `rqp/evaluation.py`, because name matching is brittle.

---

**2. Race prediction target and race input are not well-posed enough.**

Race training in `rqp/data.py:961-984` merges qualifying `position` as `qualy_position`, then predicts race `position`. But the race should use the **starting grid**, not only qualifying classification. The local raw data has `GridPosition` in race results, but `LocalWeekendProvider.get_race_results()` only returns `Position` / `ClassifiedPosition` and drops `GridPosition` in `rqp/providers.py:1075-1106`.

This is a major issue for Monaco and for any race with grid penalties, sprint-weekend complications, pit-lane starts, parc fermé penalties, or DNS changes. For race prediction, “actual start position” is the dominant prior, not qualifying result.

Current race modeling is effectively:

```text
finish_rank ≈ f(qualifying_position, FP pace, static track features)
```

A better target is:

```text
finish_rank = starting_grid_rank + race_delta
```

where `race_delta` is modeled as track-dependent movement plus reliability/chaos:

```text
delta_i = finish_rank_i - grid_rank_i

E[delta_i] =
    α_circuit
  + β_overtake(circuit) * pace_residual_i
  + β_tyre(circuit) * race_sim_i
  + β_team * team_form_i
  + β_driver * driver_form_i

pace_residual_i = pace_rank_i - grid_rank_i
```

For Monaco, `β_overtake(circuit)` should be very small. For Monza/Spa/Baku, it can be larger.

Expected impact: likely improves race behavior at low-overtake circuits and reduces fake confidence from qualifying-only baselines. Failure mode: if `GridPosition` is missing or unreliable in some local files, fallback must be explicit and logged.

---

**3. Race training leaks in-sample qualifying predictions.**

In `_merge_predicted_qualifying_context()` in `rqp/prediction.py:927-936`, the qualifying model is trained on `qual_train`, then predicts `qual_train` itself, and those predictions are merged into `race_train`. That gives the race model in-sample qualifying signals that are too good. Current-event race features get out-of-sample current qualifying-context predictions, but training rows get in-sample predictions. That is train/live mismatch and leakage.

Concrete replacement:

For each historical race event `e`, generate qualifying predictions with a model trained only on events before `e`.

Algorithm:

```python
for event_key in sorted(qual_train.event_key.unique()):
    train_q = qual_train[qual_train.event_key < event_key]
    val_q = qual_train[qual_train.event_key == event_key]

    if len(train_q.event_key.unique()) < min_events:
        use_baseline_signal(val_q)
    else:
        q_model = train_model(train_q, qual_feature_cols, ...)
        q_pred[event_key] = predict_with_model(q_model, val_q, ...)
```

Then merge those out-of-fold qualifying signals into `race_train`.

Expected impact: race CV/backtest performance may initially drop, but the live/holdout performance estimate becomes more honest. Failure mode: early-season historical events have too little training data; use a simple FP baseline for those.

Test to add: for any historical event `e`, assert that its `qualy_pred_position` was generated by a model whose training `event_key < e`.

---

**4. Team rolling features leak teammate current-event information during training.**

In `_add_temporal_features_train()` in `rqp/data.py:614-649`, team rolling features group by team and do `s.shift(1)` row-by-row. Because two teammates appear in the same event, the second teammate can receive the first teammate’s current-event FP value as “history.” The current-event path `_attach_temporal_features_current()` maps only historical rows, so training and inference are inconsistent.

Problem pattern:

```python
team_group = out.groupby(team)["fp_mean_delta"]
out["team_ewma_fp_mean_delta"] = team_group.transform(lambda s: s.shift(1).ewm(...).mean())
```

Correct event-level approach:

```text
team_event_value(team, event) = mean fp value of all team drivers in that event

team_form(team, event_t) =
    rolling_mean(team_event_value(team, event_{t-1}), event_{t-2}, ...)
```

Then merge the same team-event historical value back to both drivers of the current event. Both teammates in the same event should receive the same pre-event team form.

Expected impact: model selection scores may drop, but holdout validity improves. Failure mode: small teams/driver changes require careful team-name normalization.

Test to add: construct two same-team rows in the same event with different FP deltas. Their `team_form_3_fp_weighted_delta` must not depend on either current-event row.

---

**5. Circuit-card interactions are mostly rank-invariant in fallback mode.**

In `rqp/data.py:154-223`, many circuit interactions are simple scalar multiplications:

```python
qualy_position_circuit_importance_adj = qualy_position * (0.35 + circuit_qualifying_importance)
fp_weighted_delta_downforce_adj = fp_weighted_delta * (0.45 + downforce)
fp_race_sim_delta_tyre_adj = fp_race_sim_delta * (0.45 + tyre)
```

Inside `_hierarchical_fallback()` in `rqp/prediction.py:149-159`, these columns are individually ranked again. For a given event, `circuit_qualifying_importance`, `downforce`, `power`, and `tyre` are identical for every driver. Multiplying all drivers by the same positive constant preserves the rank. Therefore, these features do not really change the fallback order.

This is the core reason Monaco showed no material change.

A real circuit interaction must change the **relative ordering between drivers/cars**, not just scale the whole event. For example:

```text
car_fit_i,c =
    w_downforce_c * team_downforce_strength_i
  + w_power_c     * team_power_strength_i
  + w_tyre_c      * team_tyre_strength_i
  + w_street_c    * driver_street_skill_i
```

Then use:

```text
score_i = base_score_i - λ * car_fit_i,c
```

Current code has circuit sensitivities, but it does not have reliable learned car-strength latent factors. `team_archetype_form_3_fp_weighted_delta` helps a little, but it is sparse and affected by the team rolling leakage above.

Expected impact: the circuit layer will start having measurable event-specific effects. Failure mode: with tiny data, learned team-circuit factors can overfit hard unless heavily shrunk.

---

**6. Track-stat design confounds overtaking, DNFs, penalties, and chaos.**

`LocalWeekendProvider._race_event_summary()` computes:

```python
overtake_propensity = mean(abs(qualy_position - race_position)) / (field_size - 1)
```

in `rqp/providers.py:843-846`.

That is not overtaking. It is total finish-order movement, which includes DNFs, penalties, pit strategy, safety cars, starting-grid penalties, and bad qualifying. It also uses qualifying position, not starting grid. For race modeling, this can misclassify chaotic races as “good overtaking tracks.”

There is also a concrete bug in `rqp/providers.py:847-848`:

```python
grid_stability = abs(spearman_corr)
```

A perfectly inverted grid would have Spearman correlation `-1`, and `abs(-1)` becomes `1`, falsely marking it as highly stable. It should not use `abs`.

Better:

```text
grid_stability = clip((spearman_corr + 1) / 2, 0, 1)
```

or simply:

```text
grid_stability = clip(spearman_corr, 0, 1)
```

depending on whether you want neutral random order to be 0.5 or 0.

Also, `track_chaos_index` in `rqp/data.py:132-133` includes overtake propensity as a positive chaos component. That conflates “overtaking is possible” with “race is random.” Monza can have overtaking without being Baku/Singapore-style chaotic.

Expected impact: cleaner Monaco/Singapore/Baku/Monza behavior. Failure mode: historical sample per circuit is tiny; use shrinkage.

---

**7. Probability outputs are not event-consistent enough for betting.**

There are two probability paths.

The optional `pl_gumbel` listwise path in `_pl_gumbel_listwise()` is structurally better: it samples full race/qualifying orders, so `sum(p_win) ≈ 1`, `sum(p_top3) ≈ 3`, and `sum(p_top10) ≈ 10`.

The default rank-based/calibrated path is not good enough. `_rank_based_probability()` in `rqp/prediction.py:307-324` clips probabilities after scaling. For a 20-driver field, I checked:

```text
sum p_win  = 1.000
sum p_top3 = 3.000
sum p_top10 = 6.995
```

So default top-10 probabilities do not even sum to 10. With learned calibrators, `win`, `top3`, and `top10` are fitted independently in `rqp/training.py:1576-1584`, so they can violate event totals and can also violate monotonicity such as `p_win <= p_top3` unless explicitly corrected.

`_predict_probabilities()` only enforces `top3 <= top10` in `rqp/prediction.py:351-356`. It does not enforce event totals or `win <= top3`.

Concrete fix: make listwise probabilities the default for all full-field outputs. Calibrate only the PL temperature on walk-forward out-of-fold data.

PL model:

```text
u_i = -score_i

P(π | u, τ) =
  ∏_{j=1}^{n}
    exp(u_{π_j}/τ) / ∑_{k=j}^{n} exp(u_{π_k}/τ)
```

Fit `τ` by minimizing OOF negative log likelihood:

```text
τ* = argmin_τ - ∑_events log P(actual_order_event | u_event, τ)
```

Then derive:

```text
p_win_i  = P(rank_i = 1)
p_top3_i = P(rank_i <= 3)
p_top10_i = P(rank_i <= 10)
```

Expected impact: rank MAE may not change, but betting math and probability credibility improve. Failure mode: if model scores are weak, calibrated PL probabilities will be flatter, which is correct.

---

**8. Betting EV math is formally correct in places, but operationally unsafe.**

`rqp/betting.py` computes:

```python
expected_roi = model_probability * decimal_odds - 1
kelly_fraction_raw = expected_roi / (decimal_odds - 1)
```

That formula is correct for decimal odds if `model_probability` is calibrated. The problem is that model probabilities are not sufficiently calibrated or event-consistent.

The engine also computes both:

```python
probability_edge = model_probability - implied_probability_raw
fair_probability_edge = model_probability - fair_market_probability
```

but `_reject_reason()` in `rqp/betting.py:143-160` uses `probability_edge`, not `fair_probability_edge`. If you have a complete market from one bookmaker, the fair/no-vig probability is the better comparison. If the market is incomplete, `market_overround` is not meaningful.

Concrete fix:

Use `fair_probability_edge` for filtering only when the full market is present and overround is plausible. Otherwise skip no-vig normalization and label the recommendation as “raw odds only.”

```text
q_i_raw  = 1 / odds_i
q_i_fair = q_i_raw / ∑_market q_j_raw

edge_i = p_i - q_i_fair
EV_i   = p_i * odds_i - 1
f*_i   = max(0, EV_i / (odds_i - 1))
stake_i = bankroll * min(max_bet, fractional_kelly * f*_i)
```

But betting should be gated behind calibration checks:

```text
Allow bets only if:
- OOF Brier score beats baseline
- OOF log loss beats market-free baseline
- event probability sums pass invariants
- calibration slope is within a defined range
- no in-sample calibration source is used
```

Expected impact: fewer recommendations, lower false edge. Failure mode: fewer or zero bets for many races, which is acceptable.

---

**9. Model selection is too flexible for the sample size.**

The project tries baselines, ridge, hist-gradient boosting, XGBoost ranking/regression, LightGBM optional, empirical-Bayes ranking, blends, and optional PyTorch MLP. That is a lot for F1 data. A season has roughly 20–24 events and about 20 drivers per event. Even with several seasons, the independent event count is tiny.

`_walk_forward_folds()` in `rqp/training.py:892-903` is directionally correct, but the model selection score is an arbitrary composite in `rqp/training.py:886-889`:

```python
0.35 * mae_score + 0.25 * spearman + 0.25 * ndcg10 + 0.15 * hit10
```

This can select noise. Candidate failures are often silently swallowed. Deep learning in `rqp/dl_models.py` is not defensible by default for this sample size unless it consistently loses to simple baselines or is heavily regularized and evaluated only as a shadow model.

Expected impact of simplification: more stable out-of-sample behavior. Failure mode: less apparent sophistication, but better scientific validity.

Recommended default hierarchy:

```text
1. race: grid/delta constrained model
2. qualifying: FP event-relative baseline + empirical-Bayes driver/team prior
3. optional: monotonic gradient boosting with strict walk-forward validation
4. PL probability layer on top
5. DL only as non-selected shadow candidate
```

---

**10. Live/state-space replay evaluation leaks future information.**

In `rqp/live_runner.py:833`, the live race runner builds the event lap baseline from the full observations frame before iterating laps:

```python
baseline = build_event_lap_baseline(observations, min_clean_obs_per_lap=8)
```

`build_event_lap_baseline()` in `rqp/live_state_space.py:182-260` uses clean laps across the event to estimate median baseline per lap. In a true live setting, future laps are not available. Therefore `evaluate_live_replay()` can report one-step metrics that benefit from future information.

Concrete fix: in replay mode, baseline at lap `t` must be built only from laps `<= t-1`, or use historical/event-prior baseline learned from previous races. A truncation invariance test should pass:

```text
Predictions through lap L from full replay
must equal predictions through lap L from data truncated at lap L.
```

Current implementation will fail this because the full-race baseline is known upfront.

Expected impact: live replay metrics will get worse but become honest. Failure mode: early-race predictions become noisy; solve with historical priors.

## Why qualifying improved while race degraded

The ablation result is consistent with the implementation.

Qualifying can improve slightly because circuit cards add weak prior/context features to FP-derived ranking. Even if many are event-constant, the model path can use historical archetype features such as `driver_archetype_form_3_fp_weighted_delta` and `team_archetype_form_3_fp_weighted_delta`. That can help a little.

Race can degrade because race prediction is dominated by qualifying position, and the new circuit layer adds noisy or duplicated race signals. In the fallback path, circuit-scaled qualifying columns are mostly duplicate rank signals. In the trained path, `qualy_context_position` blends actual qualifying with in-sample predicted qualifying during training, and with current predicted qualifying at inference. That creates leakage/training mismatch and can pull the race model away from the strongest known race prior: actual starting grid.

The reported ablation magnitudes are also tiny. Assuming the winner rates are over 24 rounds, qualifying winner hit rate improved from about 5/24 to 7/24, while race winner rate degraded from about 16/24 to 15/24. Top-10 differences look like roughly one driver-slot over ~240 top-10 decisions. Those are not reliable improvements without paired confidence intervals or permutation tests.

## How Monaco should be modeled

Monaco should not be “normal race model + static circuit features.” It should be a constrained perturbation around the starting grid.

For each driver:

```text
grid_i = official starting grid rank
pace_rank_i = event-relative race pace estimate from FP/race sim/team form
residual_i = pace_rank_i - grid_i
```

Define circuit parameters:

```text
q_c = qualifying_importance_c
o_c = overtake_propensity_c
s_c = safety_car_probability_c
v_c = strategy_variance_c
d_c = dnf_rate_c
```

Then:

```text
κ_c = clip(κ0 + κ1 * o_c + κ2 * DRS_c - κ3 * overtaking_difficulty_c, 0, 1)

μ_delta_i = κ_c * residual_i
σ_delta_c = softplus(σ0 + σ1*s_c + σ2*v_c + σ3*d_c)
```

For Monaco:

```text
κ_monaco ≈ 0.03 to 0.10
σ_delta_monaco = low in clean race, higher only under SC/rain/reliability
```

Sample race outcomes:

```text
dnf_i ~ Bernoulli(p_dnf_i)

latent_finish_i =
    grid_i
  + μ_delta_i
  + Normal(0, σ_delta_c)

if dnf_i:
    latent_finish_i += large_penalty
```

Then sort `latent_finish_i` to generate a full finishing order. This gives you winner/top3/top10/order probabilities naturally.

For Monaco specifically, the model should preserve the grid in clean simulations and only allow meaningful movement through:

```text
- pit strategy offset
- safety car / red flag
- reliability / DNF
- very large pace mismatch
- starting-grid penalties
```

This is exactly where the current implementation is weak: it does not use official grid, and it does not turn low-overtake tracks into a constrained race distribution.

## Static cards vs embeddings vs Bayesian priors

The best design here is a hybrid.

Static cards are useful as **priors**, especially for new or rare circuits. But they should not be treated as strong learned truth. Learned embeddings are dangerous with this sample size unless heavily regularized. A hierarchical Bayesian structure is more appropriate:

```text
circuit_trait_c ~ Normal(static_card_c, σ_static)

team_factor_team,season,k ~ Normal(previous_team_factor, σ_team)

performance_i,c =
    driver_skill_i
  + team_base_team
  + Σ_k circuit_trait_c,k * team_factor_team,k
  + noise
```

Then update with historical evidence:

```text
posterior_trait_c =
    w_c * observed_trait_c + (1 - w_c) * static_trait_c

w_c = n_same_circuit / (n_same_circuit + K)
```

This is already conceptually attempted in `circuit_card_from_event()`, but the downstream model does not use the priors in a way that changes race distributions enough.

## Smallest high-impact implementation plan

First, fix evaluation before touching the model. Use full-field `all_prediction_rows`, match on `driver_id`, and report per-round paired deltas. This is the highest priority because without it you cannot trust improvements.

Second, fix temporal leakage. Rewrite team rolling features at event level, and generate out-of-fold qualifying context for race training. Expect apparent validation scores to drop. That is a good sign if the previous scores were inflated.

Third, use `GridPosition` from race results and define `grid_position` explicitly. Race features should use `grid_position`, not just `qualy_position`. Keep `qualy_position` as a diagnostic feature, but the race prior should be grid.

Fourth, replace race target logic with a grid-delta model. The simplest version is:

```text
target_delta = race_position - grid_position
pred_finish_score = grid_position + predicted_delta
```

Use monotonic constraints or hard-coded slope bounds so that on Monaco-like circuits the model cannot overreact to FP pace.

Fifth, make PL/listwise probabilities the default. Calibrate temperature with walk-forward OOF predictions. Remove or clearly label independent binary calibrators as experimental.

Sixth, refactor circuit interactions. Do not add event-constant scaled duplicates as primary features. Instead create driver-varying fit features:

```text
team_power_strength_prior
team_downforce_strength_prior
team_tyre_strength_prior
driver_street_prior
driver_wet_or_chaos_prior, if real data exists
```

Then multiply those by circuit sensitivities.

Seventh, split track stats into separate concepts:

```text
overtake_potential
grid_stability
safety_car_risk
dnf_risk
strategy_variance
pit_loss
weather_risk
```

Do not put overtaking into chaos.

Eighth, betting should be disabled by default unless probability calibration passes OOF checks. Recommendations can still be produced, but label them as “research only” when calibration is not validated.

## Honest backtest design

Use only walk-forward or strict holdout evaluation. Do not mix 2025 holdout into model selection and then report 2025 as final proof.

A clean design:

```text
For each target event e:
    train_events = all events with event_key < e
    build all rolling features using only train_events
    build track stats using only train_events
    if race:
        generate qualifying context for train_events via OOF only
        generate current qualifying context from model trained on train_events
    predict full field for e
    save full predicted order and probabilities
```

Report per event:

```text
- full-field MAE
- Spearman correlation
- NDCG@10
- top10 hit count / 10
- winner hit
- podium hit count / 3
- Brier score for winner/top3/top10
- log loss or ranked probability score
- probability sum invariants
```

For circuit-card ablation, compare paired events:

```text
delta_metric_e = metric_with_cards_e - metric_without_cards_e
```

Then report:

```text
mean paired delta
median paired delta
bootstrap 95% CI over events
sign test: count of events improved vs worsened
per-archetype deltas: Monaco/Hungaroring/Singapore vs Monza/Spa/Baku/etc.
```

No fake precision. With ~20–24 races, report results like:

```text
MAE delta = -0.03 ranks, bootstrap 95% CI [-0.28, +0.22]
```

not as if `0.0293` is meaningful.

For betting, do not report invented P&L unless you have timestamped odds and a rule that existed before the race. Use:

```text
- closing-line value, if odds snapshots exist
- calibration curves
- Brier/log loss vs market implied probabilities
- paper P&L only for timestamped pre-race recommendations
```

## Tests that should be added

Add a test proving `run_experiment.py` evaluates `all_prediction_rows`, not only `rows`.

Add a leakage test for team rolling features: two teammates in the same event must receive the same pre-event team form, and it must not depend on either teammate’s current FP delta.

Add a leakage test for race qualifying context: historical `qualy_pred_position` merged into race training must be out-of-fold by event.

Add a provider test that `LocalWeekendProvider.get_race_results()` exposes `grid_position` from `GridPosition`, and race feature building uses it.

Add a circuit behavior test:

```text
Monaco:
    Driver A starts P1 with slightly worse race pace.
    Driver B starts P4 with better race pace.
Expected:
    Driver A remains more likely to win unless safety/reliability is high.

Monza:
    Same inputs.
Expected:
    Driver B gets materially more comeback probability than at Monaco.
```

Add an invariant test showing that changing only `circuit_qualifying_importance` must change the race distribution on Monaco-like tracks. Current fallback would fail this.

Add a track-stat test where Spearman correlation is `-1`; `grid_stability` must not become `1`.

Add probability tests:

```text
sum p_win per event ≈ 1
sum p_top3 per event ≈ min(3, field_size)
sum p_top10 per event ≈ min(10, field_size)
0 <= p_win <= p_top3 <= p_top10 <= 1
```

Run these for both default and PL modes. The current default top-10 rank-based probability fails the sum invariant.

Add a betting test where `probability_edge` is positive but `fair_probability_edge` is negative after overround normalization. The recommendation should be skipped.

Add a live replay truncation test:

```text
run replay on full race up to lap L
run replay on data truncated at lap L
predictions through lap L must match
```

Current live baseline design should fail because it sees future laps.

Add CI/package tests. Right now, running the selected tests directly from the Python folder without `PYTHONPATH=.` fails with `ModuleNotFoundError: No module named 'rqp'`. Add `pyproject.toml` or set up editable install in CI.

## Kill criteria

Kill or quarantine the current circuit-card layer if, after leakage fixes and full-field evaluation, it meets any of these:

```text
- Mean paired MAE improvement is between -0.05 and +0.05 ranks with a wide CI crossing zero.
- Race MAE degrades on low-overtake tracks.
- Monaco/Singapore/Hungaroring behavior remains unchanged under synthetic tests.
- Circuit coefficients or Bayesian posterior trait effects shrink effectively to zero.
- Probability calibration gets worse with cards: higher Brier/log loss or worse PL NLL.
- Card effects are driven by one or two rounds only.
```

Kill betting recommendations as user-facing outputs if:

```text
- OOF Brier/log loss does not beat a simple rank/market-free baseline.
- Probability sums fail event invariants.
- Calibration uses in-sample fallback.
- Paper P&L is reported without timestamped odds and pre-declared stake rules.
```

Kill the DL candidate as a selectable model if:

```text
- It does not beat the best simple baseline over at least two independent seasons or rolling blocks.
- Its selected wins disappear under bootstrap/permutation tests.
- It increases variance or produces unstable probabilities.
```

## Bottom line

The project has useful engineering scaffolding: local data ingestion, FP feature extraction, rolling features, model families, evidence outputs, listwise probability option, live replay machinery, and betting scaffolding. But the current mathematical stack is not yet rigorous enough to support claims like “calibrated probabilities,” “circuit-specific race modeling,” or “positive EV betting.”

The most damaging issue is not the Monaco card value. It is that the current race/fallback machinery mostly cannot express “Monaco should be grid-constrained” in a mathematically effective way. Fix the evaluation first, remove leakage, use starting grid, model race as a circuit-conditioned perturbation around grid, and make PL probabilities default. Only then will the circuit-card idea be testable rather than cosmetic.
