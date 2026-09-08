# Sector forecasting: unfitted research prototype

This directory contains verified data acquisition and tested forecasting components. **It does not contain an executed forecasting experiment, measured predictive gains, a fitted sector model, or a serving integration.** The complete chronological orchestrator, immutable issuance ledger, causal target join, selection protocol and transfer evaluation remain pending. The components must not be registered as a validated improvement in the global prediction system.

The preceding [two-race sector pilot](../sector_pilot/README.md) completed a feasibility study: 4,312 positive S1/S2 updates were retained, 4,245 met its recorded-input checkpoint rule, 4,231 candidate checkpoints had a later recorded eligible lap, and 14 were unmatched. These are structural and outcome-availability counts, not predictive results. Canonical HGB feature parity and historical client receipt remain uncertified. An empty current S3 display slot does not imply that no earlier positive S3 was received.

Acquisition is complete for exactly **44 discovery races: 22 in 2022 and 22 in 2023**. The manifest records 126 downloaded TimingData/SessionStatus/TimingAppData streams, six verified pilot-stream reuses and 44 existing TrackStatus streams. No stream was unavailable. The acquisition verification checked 297 hash bindings. Historical feeds can later change; the recorded URLs, download receipts and hashes describe this captured snapshot, not original live-client receipt or publication latency. There are no newly acquired 2024–2026 sector inputs in this scope.

Implemented components:

- `acquire.py` and `acquisition_contract.json`: bounded public-feed acquisition, immutable manifests and saved failure history.
- `completed.py` and `completed_contract.json`: immutable raw completed-history records. Valid history requires an unambiguous unit counter advance, ordered earlier S1/S2, same-packet S3 plus LastLapTime, the fixed 0.003-second sum check, and the declared observed control/pit quality proxy. Late revisions are diagnostics and never upgrade an earlier completion.
- `context.py`: checkpoint context preserving the pilot's exact keys, counter/epoch semantics and as-of contamination. It records all strictly prior control transitions, persistent InPit state and fresh PitOut events. The final version passed 14 tests, including exact parity for all 4,312 pilot keys; source SHA256 is `8a717e68b2caff4870aef7349cd3832bdc73ce251728a56dff6eaacc2c461d2a`, test SHA256 is `49fdce0e58a2d1be6b748c2da8a7c0af80ac6d77994c26f66de98dac4292f488`.
- `baselines.py` and [BASELINE_REVIEW.md](BASELINE_REVIEW.md): four fixed same-checkpoint remainder references and a Gaussian conditional-remainder prototype. Their numerical tests do not establish empirical accuracy.
- `models.py`: residual-HGB fitting machinery and event/target-balanced accounting. It has not been fitted or scored on this sector cohort. It supplies no selected checkpoint model.

The completed parser passed 37 synthetic tests. Its two-pilot structural smoke check yielded 874 eligible completed-history records in 2022 and 888 in 2023, with both input-integrity flags true; the two late 2023 S3 updates remained ineligible diagnostic records. These smoke counts are not a full 44-race execution or a performance comparison.

Required orchestration boundaries:

1. Fully consume `iter_completed(paths, event_key, stats)` and require `stats["stream_input_valid"]` before using that source in fitting. This protects against unplaceable or malformed packets that cannot certify complete input integrity.
2. Admit only records with `observed_valid_completed is True` and `available_ms < checkpoint_ms`. Issue every forecast at a shared timestamp before admitting any completion at that timestamp, including other drivers' completions.
3. Do not join the numeric `local_epoch` counters from the completed parser and checkpoint context. The context also counts ambiguous resets without counter advances. Use explicit driver/counter/boundary provenance and reject ambiguous associations; never infer canonical LapNumber or TyreLife.
4. Freeze the full feature, issuance, history-support, selection and target-resolution protocol before fitting. Retain unsupported checkpoints and unmatched issued forecasts with explicit statuses. A same-checkpoint sector forecast has a different horizon from the original completed-lap HGB benchmark.
5. Close and hash issued forecasts before resolving later targets. Compare all candidates on the same population, with event and target balance and event-level uncertainty. No serving activation is justified until those checks and empirical gates succeed.

Compact evidence is published under `docs/research/evidence/boundary_sector_publication_20260908/`, with an original-to-copy SHA256 map in `docs/research/evidence/boundary_sector_publication_20260908.json`. The [publication note](../../../../docs/research/boundary_sector_publication_20260908.md) explains the local-data boundary. Raw HTTP bodies, decoded feeds, original lap CSVs and large ledger/outcome JSONLs remain local and ignored by Git. Consequently, a fresh source checkout supports code review and synthetic tests; it does not contain every input needed to replay the recorded pilot.

From the repository root, run the component tests with the configured Python environment:

```sh
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv-f1/bin/python -m pytest -q -p no:cacheprovider research/experiments/boundary_20260908/sector_pilot/test_ledger.py research/experiments/boundary_20260908/sector_forecast
```

The two real-input context parity cases skip when their optional local pilot data are absent. The frozen pilot README contains a slash in one module command; the corrected verifier command is:

```sh
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 .venv-f1/bin/python -m research.experiments.boundary_20260908.sector_pilot.verify
```

That verifier needs the exact original local artifact and input paths. Do not overwrite frozen completed artifacts merely to rerun it, and do not treat current redownloads as byte-identical without checking the recorded hashes.

Suggested commit: `research(f1): publish sector feasibility evidence and unfitted forecasting components`
