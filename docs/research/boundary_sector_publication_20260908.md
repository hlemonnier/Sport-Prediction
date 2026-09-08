# Sector research publication, 2026-09-08

This publication preserves the completed two-race raw-sector feasibility study, the completed 44-race discovery-feed acquisition, and the tested sector-forecasting components. **Sector forecasting remains an unfitted prototype: no predictive gains have been measured, no sector model has been selected, and the full orchestrator and causal target join remain unfinished. There is no sector serving integration or promotion.**

The publication manifest is [boundary_sector_publication_20260908.json](evidence/boundary_sector_publication_20260908.json). It maps each ignored original structured artifact to a byte-identical published copy, recording both paths, both SHA256 values and the byte count. The original frozen sources, manifests and research artifacts were preserved. These copies permit review of the recorded evidence without force-adding raw provider data to Git.

Verified scope:

| Work | Completed evidence | Interpretation |
|---|---|---|
| Two Bahrain races, 2022/2023 | 4,312 S1/S2 updates; 4,245 recorded-input candidates; 4,231 later recorded eligible outcomes and 14 unmatched candidates | Feasibility and conditional outcome availability; no prediction errors or gains |
| Pilot verification | All 4,312 updates preserved; six TimingData-prefix checks; source-time checks and 4,312 independent CSV target resolutions; 29 input hash checks | Recorded-stream proxy, not original client-receipt causality or canonical HGB parity |
| Discovery acquisition | 44 races, 22 per year; 126 downloads, six pilot-stream reuses, 44 existing TrackStatus streams; zero unavailable; 297 hash bindings checked | Completed acquisition only; no 2024+ sector acquisition or model scoring |
| Forecast components | Completed-history/context parsers, four fixed references, Gaussian prototype, residual-model machinery and tests | No completed forecasting experiment, selected model or serving activation |

The snapshot retains failed acquisition and verification attempts alongside their successful successors. In particular, the preserved acquisition-verification failure concerned comparison of standard percent-encoded São Paulo URLs; the final check compares the frozen session identities after standard URL decoding. It did not change the downloaded source payloads or produce model outcomes.

The current context implementation is the final 14-test version: `context.py` SHA256 `8a717e68b2caff4870aef7349cd3832bdc73ce251728a56dff6eaacc2c461d2a`, `test_context.py` SHA256 `49fdce0e58a2d1be6b748c2da8a7c0af80ac6d77994c26f66de98dac4292f488`. The completed-history parser's 37 synthetic tests and two-pilot structural smoke do not constitute a full 44-race model evaluation. Source hashes for every published component are in the manifest.

The following remain local: provider HTTP bodies and decoded streams, original canonical lap CSVs, large immutable ledger/outcome JSONLs, duplicate per-stream receipts and mutable progress logs. Their original hashes and paths remain in the copied manifests. Publication verification checks copy equality and source bindings; it is **not** a claim that all transitive raw inputs are present in a fresh clone. Replaying the recorded pilot requires restoring its exact original paths and matching every pinned hash. A new download can legitimately differ from the captured historical snapshot.

The frozen pilot README has an invalid slash in one module invocation. Without changing that hash-bound document, the corrected command is:

```sh
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 .venv-f1/bin/python -m research.experiments.boundary_20260908.sector_pilot.verify
```

Run it only with the required local pilot artifacts and data available. Synthetic component tests and the two optional real-input parity checks are documented in the [forecast README](../../research/experiments/boundary_20260908/sector_forecast/README.md). Missing optional pilot data cause those two parity checks to skip; a source checkout alone cannot establish the recorded raw-data replay result.

The next research execution must freeze the full chronological protocol and build the causal issuance/target join before fitting. Completed histories need source-integrity validation and `available_ms < checkpoint_ms`; ambiguous epoch ordinals must not be treated as canonical lap identities. The target remains the first later recorded eligible lap, which can skip the current lap. Every unsupported or unmatched checkpoint needs explicit accounting. Later-horizon sector gains, if eventually observed, cannot be represented as gains on the original HGB issuance population.

Suggested commit: `research(f1): publish sector feasibility evidence and unfitted forecasting components`
