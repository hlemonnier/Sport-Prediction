# Conditional proposal: learn ordered telemetry state at the original issuance

Proposed on 2026-09-08 while the frozen telemetry90 preparation was running. **Not activated:** no new historical join, representation fit, lap forecast or score was produced for this note, and no current telemetry result was inspected. Existing sources remain immutable.

My next choice, conditional on telemetry90 failing, is one small **self-supervised, causal packet-sequence encoder**, with a separately trained order-destroyed control. The precise hypothesis is that recent brake/coast/acceleration order contains predictive information about the next eligible lap beyond the current speed traps, lap-history features and packet-window marginals. This is a hypothesis about incremental information, not a claim to recover fuel, battery charge, tyre condition or driver intention.

## Why this adds a distinct test

The existing [next-mechanisms review](../next_mechanisms.md) already covers recent-error adaptation, Bayesian run length, refitting and the proposed raw telemetry summaries. The [sector-peer proposal](../sector_peer/README.md) already specifies asynchronous leave-driver-out shared-state filtering. Repeating that factor model would not be a new proposal. This experiment instead preserves the existing original lap-end horizon and asks whether telemetry aggregation itself discards useful information.

The frozen [telemetry encoder](../telemetry/features.py) calculates 30/90/180-second means, fractions, speed dispersion and one final-state repetition statistic. Those features preserve little temporal order. Permuting distinct packet payloads within one nested-window membership class preserves all their content marginals; with no adjacent repeated tuples it also preserves the repetition statistic. Ordered control transitions can therefore differ while all 90 summary values remain identical. The original 80 lap/sector/speed-trap predictors can also be held fixed in this synthetic construction. More trees on those 170 columns cannot distinguish the two cases.

Available information is real but its usefulness is unmeasured: the closed acquisition contains all 44 discovery `CarData.z` streams (42 downloads plus two verified pilots), with 355,191,352 decoded-body bytes. This note reads their manifest and source contracts, not their issuance-joined contents. The 39,220 original issuances and 38,370 matched targets remain the required population. Millions of raw samples do not create more than 22 independent training races.

## One representation, one fixed objective

Reuse the current validity rules and atomic per-driver packet reductions. Each token has 34 numeric coordinates: eight content measurements (the nine existing content fields excluding cross-packet speed dispersion), their eight validity indicators, the existing 15 per-packet quality fractions, log bundle size, the interval since the previous driver packet, and a padding indicator. Missing measurements are zero with their validity indicator unset. No DRS on/off reinterpretation and no invalid-code clipping into a physical value.

Use the latest at most 128 driver packets in the preceding 180 seconds, left padded. Keep recorded packet order. Divide speed, RPM, throttle and gear by fixed scales 350, 15,000, 100 and 8; fractions retain unit scale. Scale log bundle size by log(18) and the gap by five seconds. Clip scaled numeric magnitudes to five for numerical stability; these are feature normalizations, not reconstructed physical limits. No whole-session normalization, circuit/driver/year predictor or cross-driver input is added.

Let x_i denote a token and c_i = g_theta(x_{i-127:i}) its 16-dimensional representation. Fix a seven-layer causal dilated convolution, width 24, kernel three, dilations 1/2/4/8/16/32/64, per-timestamp LayerNorm, GELU, no dropout, and a final linear projection. Left padding only; no bidirectional convolution or batch normalization. Parameter count must be checked before lock and remain below 25,000, including prediction heads.

Learn solely from 2022 raw streams using contrastive predictive coding. For k in {4,16,32} packet observations, a pointwise target encoder z_j = phi_psi(x_j) and a bilinear score give

\[
L=-\frac{1}{3}\sum_k\log
\frac{\exp(c_i^T W_k z_{i+k}/0.1)}
{\sum_{j\in\{i+k\}\cup N_i}\exp(c_i^T W_k z_j/0.1)}.
\]

Normalize c and z to unit norm before scoring. The 31 negatives come from the same driver and race at packet availability at least 180 seconds from the context endpoint, limiting identity/circuit shortcuts. Do not use lap outcomes, lap boundaries, future stint or future cleanliness to sample positives or negatives. Forecast horizons here count received packets, not equal physical time steps; observed gaps are inputs. Sample races uniformly, then drivers and supported endpoints uniformly. Fix seed 20260908, batch 32, 800 Adam steps at 0.001, gradient norm limit one, and the final checkpoint; no score-selected epoch or architecture sweep. Raw future tokens are self-supervision targets only within the completed 2022 fitting period. No 2023 token enters pretraining or normalization.

[Van den Oord, Li and Vinyals (2018)](https://arxiv.org/abs/1807.03748) motivate predictive latent representations with autoregressive context and a contrastive objective. This is a small application-specific adaptation, not reproduction of their networks or empirical gains. [TS2Vec (Yue et al., 2022)](https://ojs.aaai.org/index.php/AAAI/article/download/20881/20640) provides another primary example of learned timestamp representations; its general benchmark results do not establish an advantage here. We do not adopt its general contextual encoder without the explicit causal restriction above.

## Matched controls and exact time boundary

Train exactly two encoders. The candidate sees ordered histories. Its matched control sees a deterministic, seed-fixed permutation of the preceding token measurement/quality bundles, keeping the newest token and chronological gap/padding slots fixed. Apply this transformation to every training and inference context; use exactly the same original future-token positives, negative identities, sampling schedule, initialization and optimizer. This removes most earlier order while retaining the current endpoint, marginal packet content, packet count and timing support. The transformation itself uses only the admitted prefix. A third, untrained encoder with the same initialization is a random-representation control; it requires no extra representation fit.

Freeze both encoders before supervised fitting. Fit three HGB15/150 models on identical original 2022 matched rows: original 80 plus telemetry90 plus the respective 16 ordered, permuted or random features. Preserve current event weights, naive anchor, training residual bound and prediction correction bound. No additional residual-stacking feature or representation fine-tuning with lap labels. Compare against the unchanged 2022-only base HGB, telemetry90 and quality90 models as separate named references; do not select an old reference by event or later score.

At issuance t, only packets with cumulative recorded-prefix availability a < t−2 seconds enter. Use exact integer-nanosecond clocks, header-only lookahead, atomic bundles and immutable snapshot bytes. No source UTC, full-session t0, interpolation, future packet payload, final CSV history backfill or sector-checkpoint substitution. Keep the existing six-packet/24-second-span/five-second-age support gate and exact base-HGB fallback; left padding handles a short sequence without a new cohort filter. Preserve every unmatched issuance. Apply the fixed models at zero lag only as the already declared sensitivity, never as a selectable timing rule. Archive availability remains a delivery proxy, not certified historical client receipt.

## A decisive bounded test

Before any fit, independently construct two synthetic histories with identical original 170 predictors and different packet order, then verify that the sequence input distinguishes them. Check strict/equal-clock boundaries, later-packet and UTC poisoning, prefix replay, frozen original row/anchor/target parity and exact fallback. This is the minimum information test; failure means the implementation has not added the proposed information. Do not inspect new historical target associations to choose the sequence length or masks.

For a separately frozen execution, close all target-free features, encoder hashes and 2023 forecast vectors before external selection-label attachment. Report all three candidates and all old references on the full 22-event 2023 population. Only the ordered encoder is eligible for advancement; the others are diagnostic controls. Stop unless it improves event-balanced MAE by at least 1% against every named reference/control, with negative paired event and circular three-event-block 95% upper bounds, all leave-one-event-out deltas negative, and positive gain against each at fixed zero lag. Report ordinary row MAE and largest event/driver contributions too. Improved contrastive loss without incremental lap accuracy is a failure. Winning against random features but not against the permuted control is a failure of the order hypothesis.

Only a closed pass could justify a separately frozen later-data stage. Retain the research program's predeclared substantial-gain thresholds: at least 10% full-event 2024–25 gain and 5% exposed-2026 gain against the strong relevant references, each year positive, negative paired block intervals and all event-omission deltas negative. These repeatedly observed years are retrospective evidence; they do not become pristine because the representation was newly trained.

Expected compute, not a benchmark: the two 800-step, roughly 15–25k-parameter pretraining jobs should be budgeted at 15–60 minutes total on one CPU thread, followed by three small HGB fits. Stop at a predeclared 60-minute pretraining cap rather than changing the grid. A batch's raw token tensor is about 0.56 MB; all 39,220 issuance sequences are about 0.68 GB per lag in float32. Stream or memory-map them; cap resident working memory at 3 GB and run one fit at a time. Do not launch this concurrently with the current fit.

The main failure risk is identifiability: the encoder may predict circuit phase, packet repetition or driver style rather than the transient state that matters for the next eligible lap. Same-driver negatives, the permuted control and strong original features reduce easy shortcuts; they cannot prove physical state recovery. Self-supervision can expose information lost by aggregation, but it cannot create signal or overcome the small number of independent races. I prefer this explicit order-information test to an unrestricted deep-model sweep.

Pinned reviewed inputs: telemetry acquisition SHA256 `9fd8b37c7905c0b9036f9b06a9d33595ecee0e42fd68d774baade5884902b220`; frozen telemetry feature source `7354b6675a4725d311c5bbf11f329e615118abf33fef5ef4bf4ee69474c1371f`. No execution source was modified.

Suggested commit: `research(f1-live): propose controlled causal telemetry sequence encoding`.
