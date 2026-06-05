# GPT Pro F1 Multi-Agent Audit

Date: 2026-06-05

Source review: `outputs/f1/reviews/GPT_PRO_F1_PROJECT_REVIEW.md`
Implementation checklist: `outputs/f1/reviews/GPT_PRO_F1_IMPLEMENTATION_CHECKLIST.md`

## Verdict

The GPT Pro review points are now either implemented, fixed, or deliberately changed/quarantined. The important distinction is that circuit cards and DL are not promoted as proven alpha. They are implemented as research/modeling layers, while the evidence gates prevent overstating them.

## Independent Review Coverage

- Copernicus reviewed evaluation, target design, leakage, and model selection.
- Linnaeus reviewed circuit-card behavior, live/replay logic, and Monaco-style race behavior.
- Hooke reviewed probability math, calibration, and betting logic.

## Implemented Or Fixed

- Full-field evaluation now uses `all_prediction_rows` when available instead of evaluating only the top-10 display table.
- Evaluation prefers `driver_id` matching over brittle display names.
- Local race results expose official `grid_position`, and race features carry `grid_source` provenance.
- Race training uses `race_delta_target = finish_position - grid_position`, with prediction reconstructed as `grid_position + predicted_delta`.
- Race-delta predictions are constrained by circuit mobility. Low-overtake/high-difficulty circuits shrink and clip movement more aggressively than high-overtake circuits.
- Historical qualifying context for race training is generated out-of-fold by event.
- Team temporal features are event-level history instead of row-wise teammate leakage.
- Negative grid-order correlation is no longer treated as stable grid behavior.
- `track_chaos_index` is decoupled from overtaking, and overtaking propensity excludes known DNF/classification failures where local status data supports it.
- Circuit features include driver-varying `circuit_fit_index`, so circuit cards can change relative ordering rather than only multiply all drivers by event constants.
- Rank-based and calibrated probabilities are normalized to event totals, with monotonicity enforced across win/top3/top10.
- PL/Gumbel listwise probabilities are the default offline full-field probability layer.
- PL temperature is fitted from walk-forward out-of-fold likelihood.
- Probability audit now checks Brier/log-loss against absolute thresholds and uniform event baselines, plus calibration slope availability/range.
- Betting recommendations are gated by probability invariants and OOF audit by default.
- Fair-market betting edge is used only when the market has enough selections and a plausible overround range; otherwise raw implied edge is used and weaker rows are rejected.
- Live replay baselines are truncation-invariant through lap `L`.

## Changed Or Quarantined

- Circuit-card alpha claim: quarantined. The regenerated paired full-field baseline ablation over 38 events produced zero MAE delta and zero top10 delta, with 95% bootstrap CIs exactly `[0.0, 0.0]`. The generated decision is `quarantine`.
- DL model selection: changed to shadow-only. DL candidates can appear in diagnostics, but production selection prefers non-DL candidates unless only DL candidates exist.
- Betting: changed to gated research/recommendation mode. The latest Monaco OOF smoke has normalized event probabilities but fails `top10` calibration, so default betting should block it.
- Full stochastic race simulation: changed to deterministic grid-delta modeling with separate safety-car/DNF/strategy/weather priors and circuit mobility constraints. This is not claimed as a full stochastic simulator.

## Verification

- `python3 -m pytest -q tests/test_gpt_pro_review_fixes.py tests/test_circuit_cards.py tests/test_betting_system.py`: 27 passed.
- `python3 -m pytest -q`: 56 passed.
- `outputs/f1/reviews/monaco_2025_race_smoke.json`: 20-row heuristic fallback smoke, event probability totals normalized.
- `outputs/f1/reviews/monaco_2025_race_smoke_oof.json`: 20-row OOF smoke, PL temperature `5.653333333333334`, probability totals normalized, audit available but failed by `top10_calibration_failed`.
- `outputs/f1/reviews/circuit_card_ablation_2025_fullfield.json`: 38 paired events, 0 skipped, generated decision `quarantine`.

## Suggested Commit Name

`fix: harden f1 race modeling probabilities and circuit audits`
