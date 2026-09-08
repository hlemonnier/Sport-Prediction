# Football information experiment: pre-closing market snapshot pooling

**Protocol proposal, 8 September 2026. No new candidate has been scored.** The next useful information source is the cached pre-closing betting market. It may incorporate squad availability and other information absent from our team-history inputs. This is a different input source, not another shot/xG residual sweep. Its value remains unmeasured in this cycle.

This is a **separate retrospective snapshot product**. It does not upgrade the existing midnight forecast or establish availability at kickoff minus 60 minutes. Historical quote timestamps are absent. An execution lock must be written after independent review and mathematical tests, before selection scores.

## Verified data and current comparators

The actual `PredictionConfig` default remains equal-weight Dixon–Coles with automatic held-out calibration. `packages/football/features/{market,lineup,injuries}.py` are unimplemented contracts. The stronger research comparators remain DC365/Elo50, DC180 and the selected 90-day shot correction with ridge 0.1. The shot correction's previous 0.265% pooled improvement over DC365/Elo50 failed its material-gain gate; neither it nor the failed xG correction is a newly promoted default.

Read-only coverage inspection verified all 27 cached CSV hashes against the existing source manifests. Each league has nine 380-match seasons, 2017/18–2025/26. A complete odds triple means all three decimal prices are finite and strictly greater than one. No result-conditioned filtering or new probability scoring was performed.

| League | Fixtures | Pre-closing average 1X2 | Bet365 1X2 | Pinnacle 1X2 | Closing average 1X2 | Kickoff `Time` |
|---|---:|---:|---:|---:|---:|---:|
| EPL | 3,420 | 3,420 | 3,420 | 3,250 | 2,660 | 2,660 |
| Spain | 3,420 | 3,420 | 3,420 | 3,226 | 2,660 | 2,660 |
| Italy | 3,420 | 3,419 | 3,417 | 3,236 | 2,660 | 2,660 |
| Total | 10,260 | 10,259 | 10,257 | 9,712 | 7,980 | 7,980 |

`AvgH/AvgD/AvgA` applies from 2019/20; the earlier names are `BbAvH/BbAvD/BbAvA`. All **760 EPL selection fixtures** and **2,280 later fixtures across the three leagues** have complete Avg and Bet365 triples. No CSV header contains quote/publication timestamps or player, lineup, squad or injury fields. This is an absence in these inspected files, not a claim about every possible provider dataset.

In 2025/26, complete Pinnacle triples fall to **210/380 EPL, 189/380 Spain and 200/380 Italy**. Avoid Pinnacle-only inputs. The cached provider page reports unreliable, stale Pinnacle delivery from 23 July 2025 and removal of those quotes from its market aggregates. The provider also distinguishes pre-closing from closing columns and describes Friday-afternoon/weekend and Tuesday-afternoon/midweek collection. These are not literal opening quotes or per-fixture receipt times. Fresh requests for the provider's notes/data/England pages returned HTTP 503 during this review; the already acquired primary snapshots were read and hashed instead. [Provider schema](https://www.football-data.co.uk/notes.txt), [data description](https://www.football-data.co.uk/data.php).

## Exact information and fitting contract

The forecast information set is the provider's **first/pre-closing snapshot**, at an unknown original historical quote time. Store `quote_observed_at=null`, `receipt_certified=false`, `product=provider_preclosing_snapshot`. Never invent a local timestamp, call it T−60, or substitute a `C` column. The incumbent's midnight prediction is carried forward unchanged as an internal component; comparisons against it measure added information at a different product cutoff. Market-only references share the snapshot and provide the fair test of whether our internal model adds value.

Learning uses a conservative day clock independently of the unknown quote time: freeze coefficients on the first fixture day of each inherited two-month refit bucket, using only rows with forecast local-day midnight strictly before the cutoff and assumed result availability at next local midnight strictly before the cutoff. Keep the last 1,460 elapsed days, pooled across the three leagues; never update from another match on the current local date. Use the already frozen fixture-timezone and resumed-match handling. This remains retrospective because original result publication/revision times are also unavailable.

Generate internal training predictions for 2020/21–2021/22 with the **unchanged selected shot model**, using cached 2019-onward prequential feature rows and the existing `predict_corrections` implementation. No in-sample fitted probabilities may enter the pool. Continue the same rolling prequential rule after that. Reuse and verify exact incumbent vectors already frozen for selection/evaluation; any required additional country training vectors use the same fixed configuration, not a new selection. Historical meta-training includes all three leagues, so later country reports measure chronological stability rather than zero-shot geographic transfer.

Fit/select on EPL **2022/23–2023/24, 760 fixtures**, then lock exactly one candidate before later **2024/25–2025/26, 2,280 fixtures**. Those dates and earlier aggregate results are already exposed. No claim of a pristine or prospective test is permitted. Known resumed fixtures stay in the primary cohort with the inherited explicit sensitivity; they are not pre-original-kickoff forecasts.

## Fixed mathematical family and strong references

For decimal odds `o_j > 1`, set `r_j = 1/o_j`. Two required market-only references are multiplicative normalization, `m_j = r_j / sum(r)`, and the power method, `m_j = r_j^k`, where the unique positive `k` solves `sum(r_j^k)=1`. Existence follows from continuity and strict decrease from three to zero for `k` from zero to infinity; overround need not be positive. These are margin-removal models, not identified true probabilities. The reciprocal of an average price is not the average of bookmaker implied probabilities. Use both methods for **Avg and Bet365**, giving four raw market references.

For the fitted models, `m` is power-corrected Avg and `p` is the frozen selected shot forecast, in H/D/A order. Floor probability inputs at `1e-12` and renormalize before logs. There are exactly **two candidates**, with no penalty grid:

1. Shared-coefficient logarithmic pool: `q = softmax(a log(m) + c log(p) + b)`, where `a,c` are scalars.
2. Class-specific logarithmic pool: `q_j ∝ exp(a_j log(m_j) + c_j log(p_j) + b_j)`.

Both minimize mean natural-log loss plus `0.01/2 * (||a−1||² + ||c||² + ||b||²)`, with `0 ≤ a,c ≤ 4` coordinatewise and `sum(b)=0`. Thus the identity is the market forecast, not a weak uniform distribution. The logits are affine in the parameters, giving a convex regularized objective on convex constraints. The class-specific candidate has eight identifiable parameters; the shared candidate has four. No time, league, bookmaker-age or unavailable squad features are invented.

**Mandatory additional market-only references** fit each identical family with `c=0`, using the exact same past rows, penalty, constraints and cadence. This distinguishes information from the internal model from ordinary market calibration. Also retain a fixed 50/50 arithmetic mixture of the selected shot forecast and power-corrected Avg as a simple pooling reference.

All gates apply against **every reference**: the four raw market forecasts, the two matched calibrated market-only forecasts, the arithmetic mixture, controlled production DC+auto, DC365/Elo50, DC180 and the selected shot incumbent. Include `market_only` and `internal_only` as identity fallbacks; do not relabel a winning reference as a new mechanism.

## Population and advancement gates

Every model emits on the identical locked fixture population. The planned 760/2,280 cohorts currently have full Avg/Bet365 coverage. In training, missing either triple or the required causal internal vector excludes that row from every fitted family, with counts and IDs recorded; no model receives extra training examples. At issuance, an unexpected absent/invalid required snapshot makes **all market policies fall back to the same frozen shot incumbent** and remains an explicit row. Fail the planned experiment if the supposedly complete selection/evaluation cohort changes; inspect the input failure before any scoring. Never delete a difficult match or replace a missing quote with a closing price.

- **Selection:** choose minimum EPL pooled NLL among the two candidates, fixed candidate-ID tie-break. Advance only with at least **0.5% relative NLL reduction against every required reference**, and an upper endpoint below zero for every candidate-minus-reference paired 28-day 95% interval. Otherwise stop, keep all selection results, and do not score new transfer forecasts.
- **Substantial later result:** at least **2% pooled NLL reduction against every reference**, negative paired 28-day interval against each, positive improvement in each country, no deterioration in any of the six country-seasons, and nonworse class-sum Brier pooled and in every country-season against each reference. The unrounded computed values determine gates.
- Use 4,000 seeded paired block resamples (`seed=20260908`), stratified by league-season, with fixed-calendar 28-day blocks; report 1- and 7-day sensitivity. Also report accuracy, ten-bin top-label ECE and all denominator counts. These intervals do not correct for past research reuse or guarantee future gains.

Even clearing these research gates would not promote the midnight product. A separate timestamped-data validation is required for any claimed fixed lead-time deployment.

## Practical route and timestamped alternatives

Implementation can stay entirely in this new directory: a strict cached-column loader and target-free issuance table; numerical power root and pooling objective; mathematical/prefix tests; then a runner that locks sources/input hashes, materializes prequential training vectors, writes selection forecasts before outcome attachment, and only evaluates the later cohort after advancement. Reuse the existing paired evaluation code and frozen incumbent artifacts. The fits have at most eight parameters, run serially on one CPU thread, and require no new package or raw-data download. This is an executable route, not a claim that the runner has already been implemented or run.

For a later exact-clock product, [Betfair's primary specification](https://historicdata.betfair.com/files/Betfair-Historical-Data-Feed-Specification.pdf) exposes publication time `pt`, market status, `inPlay`, scheduled start and runner changes. BASIC includes last-traded price, not executable back/lay ladders. Replay physical updates only through the cutoff, apply image/reset semantics, retain runner-specific last price clocks, and reject missing/ambiguous/in-play states. An unchanged last-traded price is not necessarily refreshed every minute. [Betfair update semantics](https://support.developer.betfair.com/hc/en-us/articles/25918481395740-How-is-the-BASIC-Historical-Data-traded-volume-updated). Its [official education page](https://www.betfair.com.au/hub/education/how-to-model/historical-data-sources/) describes BASIC as free, but [registered account access is required](https://support.developer.betfair.com/hc/en-us/articles/360002407712-How-can-I-access-the-Betfair-Historical-Data-service). No account or access was assumed here. A later pilot should request only two predetermined matches and audit source-clock/runner completeness before any larger collection.

[The Odds API](https://the-odds-api.com/historical-odds-data/) offers explicit preceding snapshots from June 2020, at five-minute spacing from September 2022, but historical access requires a paid plan. It is a concrete timing solution, not an authorized free input in this task.

[StatsBomb's first-party repository](https://github.com/hudl/open-data) provides historical lineup/event files, but they are completed-match exports. Position start/end and substitution records are match-relative, not publication receipts. Its [current competition catalog](https://raw.githubusercontent.com/statsbomb/open-data/master/data/competitions.json) has EPL 2015/16 and 2003/04, and men's Serie A 2015/16 and 1986/87, so it cannot supply our complete 2022–2026 EPL/Italy cohort. Treat this as a separate future player-data project; do not mistake final lineups, minutes or season aggregates for timestamped pre-match availability.

## Reproducible coverage and bindings

From the repository root, the following read-only check reproduces the core market coverage without using result columns:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv-f1/bin/python - <<'PY'
import csv, hashlib, json, math
from pathlib import Path
from collections import Counter
base = Path('data/football/performance_20260907')
totals = Counter(); bindings = []
for name in ('source_manifest.json', 'transfer_source_manifest.json'):
    for item in json.loads((base/name).read_text())['files']:
        path = base/item['name']
        if path.suffix != '.csv': continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == item['sha256']; bindings.append((path.name, digest))
        with path.open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream)
            avg = 'Avg' if 'AvgH' in reader.fieldnames else 'BbAv'
            for row in reader:
                totals['fixtures'] += 1
                for label, prefix in (('average', avg), ('bet365', 'B365')):
                    try: valid = all(math.isfinite(float(row[prefix+c])) and float(row[prefix+c]) > 1 for c in 'HDA')
                    except (ValueError, TypeError, KeyError): valid = False
                    totals[label] += valid
print(dict(totals))
print(hashlib.sha256(json.dumps(sorted(bindings), separators=(',', ':')).encode()).hexdigest())
PY
```

Expected: fixtures **10,260**, average **10,259**, Bet365 **10,257**; sorted CSV name/hash-list SHA256 `18384384718089dafb786b92754b4dd49dc937f478abee7aaaf4ed22917693f3`.

| Binding | SHA256 |
|---|---|
| `data/football/performance_20260907/source_manifest.json` | `8521af37c445777340cfa436fda3f9f6caed33bf632b3536cd6016a932f9298b` |
| `data/football/performance_20260907/transfer_source_manifest.json` | `ca5a485628cd231786e63558564eb7eeaba67bc45e456143e21506b48cf8ecc6` |
| Cached `notes.txt` | `6ecd41a98ad2751372817e7e6f1709bfeb433c53dd9aeda330fd926a5471452d` |
| Cached `data_source.html` | `aa43cae7f2949660a7f3916e757dafe22571290c7aab21a61aa752c5c8b584f1` |
| `artifacts/research/boundary_20260908/football/selection.json` | `e1db351c667da7ab438f8364e6ced76b06f3e34784efedafef7bcab18316e806` |
| `artifacts/research/boundary_20260908/football/evaluation.json` | `1c1d037ffbdf00a693728254e630792315ee7db8d8f79cbdb3545d34df66a795` |

Suggested commit: `research(football): specify market snapshot information pooling`.
