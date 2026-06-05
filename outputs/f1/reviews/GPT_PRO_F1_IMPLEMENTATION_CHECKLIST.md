# GPT Pro F1 Review Implementation Checklist

Source review: `outputs/f1/reviews/GPT_PRO_F1_PROJECT_REVIEW.md`

## Status

- [x] Preserve GPT Pro review as markdown.
- [x] Evaluate full-field `all_prediction_rows` in ablation backtests instead of top-10 display rows.
- [x] Prefer `driver_id` matching in evaluation.
- [x] Expose official `grid_position` from race results where available.
- [x] Use `grid_position` in race feature assembly and context interactions, with qualifying-position fallback when grid is missing.
- [x] Persist `grid_source` provenance (`official_grid`, `qualifying_fallback`, `missing`) for race features.
- [x] Fix `grid_stability` so negative Spearman correlation is not treated as stable.
- [x] Decouple `track_chaos_index` from overtaking potential.
- [x] Reduce overtaking-propensity contamination by excluding known DNF/classification failures where local race status allows it.
- [x] Replace row-wise same-team temporal shifts with event-level team history.
- [x] Generate historical race-training qualifying context out-of-fold by event.
- [x] Replace rank-invariant race fallback with grid-plus-circuit-delta fallback.
- [x] Add driver-varying `circuit_fit_index` so Monaco/Monza-style circuit cards can change relative ordering instead of only scaling event constants.
- [x] Constrain trained race-delta predictions by circuit mobility so low-overtake tracks shrink/clip grid movement.
- [x] Normalize rank-based probability allocations to event totals and enforce probability monotonicity.
- [x] Make PL/Gumbel listwise probabilities the default offline probability layer.
- [x] Fit OOF PL temperature and require baseline-relative Brier/log-loss/calibration-slope gates before betting.
- [x] Use explicit raw-vs-fair edge selection in betting recommendations.
- [x] Gate betting recommendations by default on probability sum/monotonicity invariants, with an explicit research-mode override.
- [x] Require fair-market odds completeness/overround bounds before using fair-market edge instead of raw implied edge.
- [x] Keep DL/high-capacity candidates shadow-only for selection until stronger holdout evidence exists.
- [x] Add local pytest pythonpath config for direct F1 package tests.

## Follow-up Items

- [x] Fit PL temperature from walk-forward out-of-fold likelihood instead of fixed config temperature.
- [x] Convert trained race target fully to `finish_position = grid_position + predicted_delta` with target-delta model selection.
- [x] Add historical priors for safety car, DNF, strategy variance, and weather as separate race-generation components.
- [x] Rework live replay lap baselines so replay predictions through lap `L` are truncation-invariant.
- [x] Run paired full-field circuit-card ablation with bootstrap confidence intervals after leakage fixes.
- [x] Replace invariant-only betting gate with full OOF Brier/log-loss/calibration-slope gates once OOF probability audit artifacts exist.

## Changed Or Quarantined Claims

- Circuit cards are implemented as data plumbing, driver-varying fit features, and circuit-mobility race constraints, but they are not marketed as proven alpha. The full-field paired baseline ablation generated `circuit_card_decision.state = quarantine`.
- DL candidates are kept as shadow/research candidates. They can be compared in the leaderboard but are excluded from production selection unless the candidate set contains only DL models.
- Betting output remains recommendation-only and is gated by probability invariants plus OOF probability audit. The latest Monaco OOF smoke has normalized event totals but fails the stricter `top10` calibration audit, so betting should be blocked for that artifact.
- Race generation was changed into deterministic grid-delta modeling plus separate priors and circuit mobility constraints. It is not claimed to be a full stochastic race simulator.

## Verification Evidence

- `python3 -m pytest -q` from `research/projects/F1/rising_qualification_prediction/Python`: 56 passed.
- Focused review tests: `python3 -m pytest -q tests/test_gpt_pro_review_fixes.py tests/test_circuit_cards.py tests/test_betting_system.py`: 27 passed.
- Monaco 2025 race fallback smoke: `outputs/f1/reviews/monaco_2025_race_smoke.json`.
  - 20 full-field rows, heuristic fallback, probability sums `win=1`, `top3=3`, `top10=10`.
- Monaco 2025 race OOF smoke: `outputs/f1/reviews/monaco_2025_race_smoke_oof.json`.
  - 20 full-field rows, model `qualifying_baseline`, OOF PL temperature `5.653`.
  - Probability sums `win=1`, `top3=3`, `top10=10`.
  - OOF audit available from `walk_forward_oof`, but `passed=false` because `top10_calibration_failed`.
- Circuit-card ablation artifact: `outputs/f1/reviews/circuit_card_ablation_2025_fullfield.json`.
  - 38 paired full-field events, no skipped events.
  - Forced-baseline selector result: paired MAE delta with-minus-without cards `0.0` with 95% bootstrap CI `[0.0, 0.0]`; paired top10 delta `0.0` with 95% bootstrap CI `[0.0, 0.0]`.
  - Generated decision: `quarantine` because paired full-field ablation showed zero/uncertain effect.
- Multi-agent audit artifact: `outputs/f1/reviews/GPT_PRO_F1_MULTI_AGENT_AUDIT.md`.

## Commit Name

Suggested commit name: `fix: harden f1 race modeling probabilities and circuit audits`
