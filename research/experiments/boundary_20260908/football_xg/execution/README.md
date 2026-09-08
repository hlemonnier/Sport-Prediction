# Delayed historical xG — execution of the frozen proposal

**Completed and independently verified: selection failed.** The best candidate, `xg_add90`, improves natural-log loss by only **0.0367368%** against the previously selected shot model, below the frozen 0.5% advancement requirement. No 14-day candidate comparison, new transfer forecasts or production change followed.

| Selection model, same 760 matches | Log loss |
| --- | ---: |
| DC365/Elo50 | 0.9595989050 |
| DC180 | 0.9722014036 |
| Prior 90-day shot correction | 0.9581581185 |
| xG added, 90-day half-life — selected | **0.9578061216** |
| xG added, 365-day half-life | 0.9585532402 |
| xG replacing shots, 90-day half-life | 0.9595392507 |
| xG replacing shots, 365-day half-life | 0.9622401587 |

The selected-minus-shot difference is -0.0003519969, with paired 28-day block interval **[-0.0018885129, +0.0011117553]** and seven-day interval [-0.0020948624, +0.0014024434]. The gain against DC365/Elo50 is 0.186826%, also below the selection threshold. Beating DC180 by 1.480689% does not establish improvement against the stronger references.

Post-selection season diagnostics show the selected model improving the shot model in 2022/23 (0.987622 versus 0.989585) and worsening it in 2023/24 (0.927990 versus 0.926731). Average absolute class-probability change is 0.00557, so the weak gain is not an exact identity artifact. These descriptive checks did not change selection or open later evaluation.

All 5,700 historical feature rows are retained; 95 involve an unseen team in the xG training window. Seven of the 760 selection fixtures use that fallback and remain scored. Independent verification checked 76 locked hashes, 90 feature fits, 180 xG mean models, 60 correction/incumbent models, every scored forecast and six sets of 20,000 bootstrap draws. Serialized xG means replay exactly; the maximum probability replay difference is 2.22e-16. Maximum quasi-score and offset-gradient errors are 2.187e-12 and 5.841e-7. **105 pre-fit tests passed.**

Selection SHA256: `297891ba46c79c843ab40da9f7ec6df0d0df3a60f56b4d1a1a3c87c6f30a4c35`. Independent verification SHA256: `0b2e26e59d465695cfc00e7dab53affbfacf463314fa106b92a1a9b4285e99a9`. Artifacts are under `artifacts/research/boundary_20260908/football_xg/execution/`.

This execution directory implements the four configurations declared in the immutable parent specification. The parent draft is preserved. New source corrects its availability handling to match the finalized data contract: next local midnight after the validated completion date, converted to UTC, plus 168 or 336 elapsed hours. Publication and revision timestamps are unknown, so these delays remain explicit retrospective assumptions.

The model estimates team attack, opponent defense and home advantage from continuous nonnegative xG with a penalized log-link mean model. Poisson quasi-likelihood identifies a conditional mean; it does not assert that xG is an integer Poisson count. Two fixed half-lives, 90 and 365 days, feed an identity-shrunk multinomial correction either alongside or instead of the existing predicted shot/SOT means. Ridge penalties, history windows, fit cadence and validation gates are inherited unchanged.

Every historical xG feature is constructed at that row's original fit cutoff. Available exact-joined xG records are intersected with the corresponding canonical base-model training population. Date discrepancies exclude xG training observations only; the scored match population remains unchanged. Unknown teams use zero attack/defense coefficients plus the learned intercept/home effect, with support recorded. Correction training sees only earlier resolved matches and uses training-only scaling. Current-match targets are removed from prediction inputs.

The primary selection population is exactly 760 EPL matches in 2022/23 and 2023/24. A candidate must improve log loss by at least 0.5% against all three stronger references: DC365/Elo50, DC180 and the previously selected 90-day shot correction. Its fixed setting must then improve all three after rebuilding historical features and correction training with a 14-day delay. Every correction block also refits the unchanged shot model and requires bit-identical incumbent probabilities.

Only passing selection permits evaluation on the existing 2,280 matches from three leagues in 2024/25 and 2025/26. The substantial criterion requires at least 2% pooled log-loss improvement and a negative upper 28-day block-bootstrap bound against all three strong references, positive per-league and pooled per-season gains over the shot correction, no Brier deterioration, a bounded ECE increase, and the fixed 14-day sensitivity checks. Exposed historical dates and repeated research do not constitute prospective confirmation.

Source and inputs must be locked before any real-data fitting. The runner refuses to replace completed attempts. A separately written independent verifier lives in the nested `verification` directory, outside the immutable runtime source fingerprint. Large feature/fit tables are local replay dependencies. No production import or promotion is implied by executing this research.

Re-run the synthetic contracts:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$PWD" .venv-f1/bin/python -m pytest -q --import-mode=importlib research/experiments/boundary_20260908/football_xg/execution/test_model.py research/experiments/boundary_20260908/football_xg/execution/test_run.py
```

The one-time mathematical verification used `verification/verify.py` with the same environment. It refuses to overwrite its completed result. Its mathematical routines can be invoked read-only for a replay, or the output filename can be changed in a separate verifier copy kept in that same nested directory so relative input paths remain valid. The original frozen experiment is not restarted or retuned.

Primary motivation: [predicting match statistics](https://arxiv.org/abs/2001.09097) and [modeling shot success](https://arxiv.org/abs/2101.02104). This custom historical-xG mean model does not reproduce those papers or inherit their reported gains.

Suggested commit: `research(football): execute delayed xG strength against frozen incumbents`.
