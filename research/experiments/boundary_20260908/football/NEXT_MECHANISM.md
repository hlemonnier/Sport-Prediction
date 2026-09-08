# Proposed next mechanism — not implemented or fitted

The completed six-candidate experiment remains frozen. Its substantial-gain target failed. This note records one materially different next hypothesis for review, not additional model selection or a performance claim.

Use shallow nonlinear multiclass residual boosting with the previous DC365/Elo50 log probabilities as a fixed offset and an identity initialization:

`p_new(x) = softmax(log(p_prior(x)) + f(x))`

`objective = mean[-log p_new(x_i)[y_i]] + lambda * sum_leaf ||w_leaf||²`

Here `f` is a sum of shallow vector-valued trees; leaf weights have three outcome components. Negative-log-loss gradients must be computed against the combined offset-plus-residual probability, not by training a generic classifier and multiplying its probabilities afterward. The offset fixes the reference model at zero residual; the penalty and bounded tree complexity control departures. Train and standardize any continuous features using strictly prior available prequential rows. Generate future correction predictions before their results become available. Retain the prior blend and the newly selected shot correction as strong references, along with the production comparator.

The concrete opportunity is a **context-dependent use of the shot forecast**. Selection-era rows show that the global affine correction helps uncertain fixtures and some balanced shot profiles, while hurting stronger favourites and some high-volume, imbalanced profiles. A global affine map has no conditional feature interactions, whereas depth-two trees can express these gates.

The following diagnostics use only the existing 760 EPL selection forecasts from 2022/23–2023/24. Positive differences mean the selected linear shot correction is worse than the prior blend. They are exploratory subgroup summaries, not independently validated effects or uncertainty estimates.

| Selection subgroup | Matches | Selected minus prior log loss |
| --- | ---: | ---: |
| Prior maximum probability below 0.45 | 200 | -0.0041224024 |
| Prior maximum probability 0.45–0.60 | 318 | -0.0023449968 |
| Prior maximum probability at least 0.60 | 242 | +0.0019636023 |
| Low forecast shot-target volume, low imbalance | 221 | -0.0067414361 |
| Low volume, high imbalance | 159 | -0.0028294466 |
| High volume, low imbalance | 159 | +0.0001035582 |
| High volume, high imbalance | 221 | +0.0037478548 |

For these descriptive bins, volume is the sum of predicted home and away shots on target, split at the selection median 8.99765787. Imbalance is the absolute log ratio of those means, split at its selection median 0.34162383. These median thresholds are diagnostics; they are not frozen production gates. No new evaluation forecasts were generated for this proposal.

Before fitting, freeze a small complexity set, stopping rule and identical prequential selection/transfer contract. Include a genuinely unused later cohort when available. Existing 2024/25–2025/26 dates remain reused retrospective diagnostics, even for this different model family. The observed subgroup pattern supports testing interactions but does not establish that another 2% improvement is attainable.

Suggested commit, only if a future implementation is authorized: `research: test nonlinear prequential football residuals`.
