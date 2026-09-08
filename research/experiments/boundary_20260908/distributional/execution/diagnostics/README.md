# Diagnosis of frozen quantile forecasts

This is post-score exploratory analysis of the stored forecasts. It fits no model, changes no candidate, and does not create a new selection or transfer result. The script reconstructs each WIS value from its interval components and checks it against the completed evidence. All component quantities below are event-balanced seconds; their deltas sum to the selected-minus-conditional-reference WIS difference.

| Population | Weighted interval width | Penalty when outcome exceeds upper bound | Penalty when outcome falls below lower bound | Median component | Total WIS difference |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2023 selection | -0.0088083670 | +0.0043191209 | -0.0087921065 | 0 | -0.0132813525 |
| 2024–2025 | -0.0020335769 | -0.0021887574 | -0.0104286250 | 0 | -0.0146509594 |
| 2026 | +0.0026609261 | -0.0028794322 | -0.0073730222 | 0 | -0.0075915283 |

The historical improvement is not chiefly a consequence of narrower intervals: 71.18% of the WIS reduction is smaller penalties when outcomes fall below the lower bounds, 14.94% smaller upper-tail penalties and 13.88% smaller weighted widths. In 2026 the weighted-width component becomes worse, while both tail penalties improve enough to offset it. These are algebraic contributions, not causal attributions or independent hypotheses.

Marginal calibration and proper scoring are different assessments. For the selected model, the empirical probabilities `P(Y <= q)` at the predicted median / 75th / 90th percentiles are:

| Population | At q50 | At q75 | At q90 |
| --- | ---: | ---: | ---: |
| 2023 selection | 0.527236 | 0.757704 | 0.900413 |
| 2024–2025 | 0.520290 | 0.742204 | 0.900808 |
| 2026 | 0.462636 | 0.697444 | 0.869087 |

The fixed point has the same q50 frequency for every model. Its changing frequency therefore cannot be repaired by these quantile models while the median remains fixed. The upper quantiles also show transport drift in 2026. Conversely, in 2024–2025 the selected q10 frequency is 0.111794, versus the conditional reference's 0.097538; the reference is closer to nominal 0.10 even though its lower-quantile losses are worse. Interval coverage or violation counts alone do not summarize conditional forecast quality or the severity of tail misses.

The comparative coverage failure is narrow and remains binding: 2025 selected 80% coverage is 79.82936%, compared with the reference's 82.96795%, exceeding the allowed three-percentage-point drop by 0.13859 points. Lower coverage than a conservative reference can accompany a legitimate proper-score improvement. This observation does not retrospectively amend the frozen gate.

Under the original global event weights, 2023 early-stint issuances carry 13.19% of weight but contribute -0.00520043 of the -0.01328135 WIS difference. Later historical early-stint issuances carry 12.03% and contribute -0.00365178. Improvements are present in all four past-volatility groups; the highest group contributes -0.00864927 of the historical total. These regime diagnostics overlap when viewed across different partitions and are not separately selected models. The 2026 eligible population contains **no wet-compound rows**, so its aggregate cannot establish wet-regime transfer.

A possible next, separately frozen mechanism is a past-only asymmetric quantile-calibration state driven by already-resolved forecast violations. The 2023 tradeoff already shows an increased upper-tail penalty, providing a selection-era reason to study it. For a non-median quantile level `tau`, a delayed stochastic pinball update could take the form `theta <- (1-eta*lambda)*theta + eta*(tau - 1{Y <= issued_quantile})`, with strict target-arrival ordering, shrinkage toward the fixed conditional model, and coherent quantile projection preserving the median. This is a research hypothesis, not an executed model or a guarantee of improvement. No learning rate, grouping or hyperparameter was selected from the transfer outcomes.

Run the diagnostic only in a fresh output directory if replaying; the default output refuses to overwrite its frozen result. The source is `diagnose.py`; the complete results are `artifacts/research/boundary_20260908/distributional/execution/diagnostics/results.json`, SHA256 `759af5b57e8063ad57a2ad2aadebc99bb66f0c6a7b440a67d55899a1b8d6f4be`.

Suggested commit: `research(f1-live): explain fixed-quantile score gains and calibration drift`.
