# Original-issuance telemetry: discovery acquisition and execution lock

All **44 original 2022–2023 discovery races** have a downloaded `CarData.z`
stream. The two independently verified Bahrain pilots were reused; 42 new
downloads required 63 HTTP attempts: 42 successful responses and 21 preserved
404 responses before official-source fallback. There were no unavailable events.
The 44 decoded bodies total **355,191,352 bytes**. Acquisition closed on
8 September 2026 at 05:10:10 UTC.

An independent closure check verified 175 unique source, receipt and body files.
This proves byte identity and acquisition completeness; payload integrity is a
separate exhaustive check during feature preparation. The raw bodies remain
local. The compact [acquisition evidence](evidence/boundary_telemetry_discovery_20260908/acquisition.json)
retains the exact session mapping, paths, hashes, byte counts and attempt history.

The [experiment specification](../../research/experiments/boundary_20260908/telemetry/specification.json)
keeps the existing next-eligible-clean-lap issuance, all 80 original predictors
and the original anchor. It adds 27 driving measurements and 63 quality
indicators across 30-, 90- and 180-second windows. A matched control zeros only
the measurement coordinates. All three models use the same matched 2022 rows,
event weights and fixed HGB configuration. The third model reconstructs the
original 80-feature, 2022-only baseline.

All **39,220 original issuances** must remain, including **850 unmatched
outcomes**. All 47,997 source lap timestamps passed an exact decimal-to-nanosecond
metadata check without rounding. Telemetry packets must be strictly earlier than
the cutoff, with a two-second primary delivery lag. A zero-second sensitivity
reuses the same coefficients. Unsupported rows receive the exact baseline
forecast. Channel code 104 is recorded separately instead of being clipped into
a valid throttle or brake measurement.

The execution design closed at **05:19:54 UTC**, binding **24 source files and
248 input paths**, after **226 passing synthetic tests**. These include the new
46 acquisition-wrapper tests, 159 feature/data/model/evaluation tests and 21
execution tests. The older transport's 56 tests also passed separately; they are
not part of the 226 count. Reviews fixed full unmatched-issuance retention and
finite-value aggregation before the first historical telemetry feature build.

The runner closes both years' target-free feature ledgers, validates every
downloaded stream, attaches 2022 training labels, performs three fixed fits,
then closes every 2023 forecast at both lags before external evaluation-label
attachment and matched-cache access. The original issuance helper internally
constructs and discards matching information while replaying historical laps;
the claim is separation of external labels and fitted predictions, not that the
helper never processes any matching information internally.

Selection requires at least 1% event-MAE improvement over **both** the baseline
and quality-only control, negative upper endpoints for event and three-event
block intervals, improvement under every leave-one-event-out comparison, and
positive improvement in the zero-second sensitivity. A pass only permits later
evaluation. Substantial-gain gates remain 10% on all 48 races from 2024–2025 and
5% on the declared 13 races from 2026, against both references and with the
specified uncertainty checks. No later telemetry acquisition is permitted after
a failed selection gate.

This publication records completed acquisition and the pre-execution lock.
It does **not** report a telemetry forecasting gain, promotion or deployment.
Historical outcomes will be published separately after the closed execution.
Archive-prefix timing remains a source-availability proxy rather than certified
historical client receipt timing. The current global model is unchanged.

The [reproduction instructions](../../research/experiments/boundary_20260908/telemetry/README.md)
describe the separate `freeze`, `prepare` and `select` commands. Full replay
requires the hash-matched local raw inputs. The
[publication manifest](evidence/boundary_telemetry_discovery_20260908/publication_manifest.json)
binds the compact evidence and experiment sources.

Suggested commit: `research(f1-live): acquire and lock the complete telemetry discovery cohort`.
