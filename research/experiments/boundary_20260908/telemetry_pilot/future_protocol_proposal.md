# Proposed original-issuance telemetry experiment after the two-event pilot

This is a new measurement/model proposal, not permission to fit or acquire later data. No original issuance clocks, lap tables, targets or model scores were read by this pilot verification. The acquisition/parser/diagnostics sources and their closed artifacts remain unchanged.

## What the closed pilot establishes

Independent replay used installed FastF1's raw-DEFLATE decoder, without importing the pilot parser or diagnostics implementation. It verified **21 unique file/hash bindings**, all three HTTP attempts/receipts, both source records and complete raw count/channel/driver equality. The successful response bodies contain 8,476,579 and 8,107,479 bytes. The 2023 mirror 404 body is preserved; its official-source fallback succeeded. There were no additional requests.

| Raw whole-session quantity | 2022 Bahrain | 2023 Bahrain |
| --- | ---: | ---: |
| Archive records | 9,626 | 9,486 |
| Bundled records | 9,564 | 9,354 |
| Source entries | 36,656 | 36,065 |
| Car rows | 733,120 | 721,300 |
| Drivers | 20 | 20 |
| Maximum entries in one packet | 14 | 17 |
| Maximum archive availability gap | 3.200 s | 12.480 s |
| Missing values in six observed raw channels | 0 | 0 |
| Archive-clock / source-UTC regressions | 0 / 0 | 0 / 0 |
| Gear codes outside 0–8 | 0 | 63 |
| Throttle code 104 | 72,279 (9.859%) | 100,975 (13.999%) |

All raw channel values in these two files are integers. Brake has codes **0, 100 and 104**, not pressure measurements: their counts are 551,710 / 109,128 / 72,282 in 2022 and 519,485 / 100,757 / 101,058 in 2023. All 2022 throttle104 records and all but one in 2023 also have brake104. Most occur at zero speed, but 5,974 throttle104 rows in 2022 and 16 in 2023 report speed above 30 km/h. Thus neither silently clipping104 to100 nor assuming moving samples have valid percentage inputs is defensible. The 63 out-of-range 2023 gear values span9–49 and occur with speed at most30; no raw record is deleted on that basis.

This is observed code behavior, not proof that104 is an unavailable-data sentinel or an error. Source clocks may advance while a car's raw state repeats. Availability freshness alone does not establish that its physical state was refreshed. DRS also has several observed codes, including9/13; do not invent a binary interpretation. Installed FastF1 documents DRS's later absence, so missing DRS must not disable the whole future forecast.

The independent receipt is `artifacts/research/boundary_20260908/telemetry_pilot/independent_verification.json`, SHA256 `234df766302dd27b4d8e4b48f9491967b982d9bd718c96640c5946c5713e7e33`. The supplemental co-occurrence receipt is `code_diagnostics.json`, SHA256 `3cccc650cbc48768882ed246bdafe241a9ebbc9ef07eca7a6c49f9db4f78e066`. Its first new helper attempt failed before producing counts because its reader did not strip the initial UTF-8 BOM; that reader was corrected and the fix disclosed. The frozen parser and first independent verifier already handled the BOM, and raw bytes did not change.

## Exact causal measurement proposal

Use three windows, **30, 90 and 180 seconds**, ending at `c = original_issuance_time − 2 seconds`. A physical packet has availability `a = cumulative_max(recorded_prefix_time)` and belongs to a window iff `c−W <= a < c`. Preserve physical order and process whole packets atomically. All source entries from an admitted packet are available together, regardless of their `Utc` values; none can be used before the packet's availability. Do not align by a full-session `t0_date`, interpolate, reconstruct a completed-lap segment or read later packet contents. Convert clock text without floating-point rounding across a strict boundary; retain the original issuance identity unchanged.

For a driver, first aggregate entries **within each packet**, then weight eligible packets equally within the window. If a packet contains17 entries, it has the same window weight as a packet with3. Averages below are packet-weighted summaries, not time fractions or energy integrals. No forward fill between packets is used. This loses precise sample timing deliberately, rather than relying on a future-derived origin estimate.

Within a packet, retain raw values but treat measurement inputs as follows. Numeric measurement validity excludes booleans; the raw boolean and null values still remain in the ledger and presence indicators.

- Speed/RPM: finite nonnegative numeric values; zero is valid. Do not claim engine power, fuel or energy from RPM.
- Throttle: finite values from0 through100 are percentage measurements; others, including104, are unavailable to the physical summary and explicitly counted in quality columns.
- Gear: integer codes0 through8, with0 retained as a raw observed code. Other codes are unavailable to gear summaries and counted.
- Brake: retain raw codes. A `brake_code100_fraction` uses the fraction of available entries with code100 among entries whose code is0 or100. Code104 has its own quality fraction; neither the fraction nor code magnitude is analog pressure. A diagnostic `raw_brake_nonzero` flag, if retained, must explicitly include104 and never substitute for that recognized-code fraction.
- DRS: keep categorical raw code presence; a valid code is a finite nonnegative integer. The proposed content feature is named literally `drs_codes_10_12_14_fraction`; it is not labelled verified wing-open time. No inference from absent DRS to off.

For each window, propose exactly **nine content measurements**: packet-mean speed, dispersion of packet-mean speed, packet-mean RPM, packet-mean valid throttle, valid-throttle-at-least95 fraction, brake-code100 fraction, low-throttle-at-most5/zero-brake joint fraction among jointly valid entries, packet-mean valid gear, and the literal DRS-code-group fraction. Means/fractions are first computed within each packet over their valid entries and then averaged equally over packets with measurement support. The speed dispersion is population standard deviation across the packet means. An empty measurement is encoded as0 with an explicit zero-valid-support column; it is not an observed zero. No learned bins, speed threshold gates or target-dependent windows are introduced. Total: **27 content columns**.

The proposed **63 support/quality columns** are21 per window: packet count, availability span, age of newest packet, largest inter-packet-or-boundary gap, mean bundle size; for each of six raw channels, packet-weighted fraction of entries present and fraction valid under the above measurement convention; raw throttle104 and brake104 fractions; fraction of adjacent packets whose final six-channel raw tuple repeats exactly; and the packet-weighted fraction of entries with **jointly valid throttle and brake**. The joint support first divides the number of entries where both measurements are valid by all driver entries in that packet, then averages equally across packets in the window. Separate positive throttle/brake support does not imply any jointly valid pair. This extra column distinguishes an unsupported joint-fraction encoding of0 from an observed joint fraction of0. All are observed-only, and raw category/null distinctions are preserved when forming tuple fingerprints. An empty window uses count/span0, newest ageW, largest gapW, zero fractions and zero bundle size. The zero packet count distinguishes this encoding from an actual observation. A one-packet window has repetition fraction0. No semantic claim follows from repetition. Total: **90 new columns:27 content plus63 support/quality**.

The common forecast-support gate is **at least6 driver packets in the30-second window, an availability span of at least24 seconds and newest-packet age at most5 seconds**. It depends on raw support only. Do not require all channels or DRS to be present; individual content measurements may be missing. Other windows remain available with explicit partial support. Gate failure gives the **exact incumbent HGB point**, and every failed-support issuance remains in the ledger and full-population score. These thresholds are proposed before any issuance-joined support or target score has been examined; freeze them before the next build rather than adjust them to obtain favorable coverage.

## Fixed model and information ablation

To isolate information, use the existing HGB15/150 architecture and original target/loss/weights/clipping contract unchanged. Fit only **two models**: the original80 features plus all90 new columns, and an otherwise identical fit where the27 content columns are identically zero while all63 support/quality columns remain. This second model is a matched information ablation, not an untrained zeroing at inference. Fit both on the same original2022 rows with equal event total weight; freeze hyperparameters, random seed and column order. Distinct `telemetry_` prefixes must prevent feature/metadata collisions.

The primary delay is2 seconds for both fits and2023 evaluation. At fixed coefficients also evaluate the same policies with0 additional delay as a timing sensitivity, without refitting or choosing the better lag. The incumbent80-feature HGB must be fitted on2022 for2023 selection; the production2022–23 fit is unavailable for a fair2023 reference. All policies retain identical original issuance/target keys, target values and unmatched ledgers. New-model points replace the incumbent only under the common source-support gate. There is no S1/S2 issuance, later checkpoint, next-numbered-lap substitution or future eligibility-based telemetry filter.

The primary candidate must improve full-population event-balanced2023 MAE by at least1% against **both** incumbent HGB and the matched quality-only model, with negative paired event and three-event-block95% upper bounds and all leave-one-event-out mean deltas negative. Its0-second fixed-model sensitivity must retain positive gain against both corresponding references. All22 selection races must be accounted for; abstention or poor data availability is not a reason to remove an event. Report every fitted/control/sensitivity result. If this screen fails, stop without later acquisition, transfer or settings search. These repeated historical dates do not become untouched validation because a new source is added.

If the screen passes, freeze final2022–23 fitting and the complete declared61-event transfer input scope before later forecasts. Retain the user's substantial-gain target of at least10% historical2024–25 event-MAE improvement over current HGB, negative paired block intervals and all omission deltas, with positive improvement in each year and a separately frozen exposed2026 material threshold. Also require gain over the matched quality-only control: otherwise evidence concerns collection/quality state, not the proposed content measurements. Historical transfer is research evidence, not retrospective proof of live delivery or an automatic production change.

Before fitting, require synthetic and real-prefix invariance of the **full original issuance ledger**, all90 added columns, support gates and predictions under future telemetry/lap poisoning. Check both directions of a strict clock boundary, malformed future packets, source-UTC changes, packet bundling, out-of-range value masking/quality preservation, exact missing-support fallback and identical ablation column layout. Include a synthetic disjoint-validity case: one entry has valid throttle and invalid brake, another has invalid throttle and valid brake. Each marginal support must be positive, joint support must be0, and the joint content value must use the unsupported0 encoding. Contrast it with jointly valid entries whose observed low-throttle/zero-brake fraction is0: joint support must then be positive. Verify unequal bundle sizes still receive equal packet weights for joint support. Close/hash target-free features before attachment of2023 targets, and verify prediction vectors before scoring. Acquire2022–23 only under a separate bounded scope after the measurement contract is finalized.

The pilot establishes decoder compatibility and whole-session channel coverage in two Bahrain races. It does **not** establish availability at all original forecast checkpoints, coverage on44 discovery races, any predictive gain, or transfer across2026's different channel schema.

Suggested commit: `research(f1-live): verify raw telemetry pilot and propose controlled feature test`.
