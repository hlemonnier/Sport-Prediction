# Market snapshot execution notes

The original README remains byte-identical, SHA256 `bb1ac1c9911631487d603394623110268bf286fd22abd49e45578cb4424433a2`. These implementation clarifications add no candidates, inputs or efficacy claims.

The shared and class-specific pools have **four and eight free coordinates**, respectively. Data identifiability is not guaranteed: for example, the internal and market vectors can coincide. The positive L2 penalty and orthonormal bias coordinates make the constrained objective strongly convex, so its penalized optimum is nevertheless unique. Market-only counterparts have three and five free coordinates. `BIAS_BASIS` has orthonormal columns perpendicular to `(1,1,1)`; therefore `b=B*u`, `sum(b)=0`, and `||b||²=||u||²`. Every free coefficient receives its declared penalty exactly once.

The power margin-removal root uses log-sum-exp and bracket expansion. It supports both overround and underround triples. Extreme valid odds may underflow an un-floored probability to zero; pooling log inputs use the frozen `1e-12` floor and renormalization. Raw references retain their original transformations and use the inherited common scoring implementation. There is no closing-odds substitution or learned quote-age feature.

## Causal data flow

1. `freeze` requires an independent review JSON with `approved_for_execution_lock=true` and the exact `source_files` map returned by `run.sources()`. **Put that review in a separate artifact directory or a nested `verification/` directory**, never as an immediate JSON in this source directory: the latter would create a self-hash cycle. It runs the synthetic suite, verifies all frozen input hashes and records Python/NumPy/SciPy/scikit-learn versions. An existing lock is immutable.
2. `prepare` verifies the lock before any historical fitting. It loads the three already frozen prequential feature caches for 2019/20–2023/24. From the 4,560 rows in 2020/21–2023/24 it reconstructs **3,800 missing selected-shot component vectors** with the unchanged historical correction code and reuses the **760 frozen EPL selection vectors verbatim**. The predecessor's result-availability condition is `<= cutoff`; this remains unchanged to preserve the incumbent. Only the **new market pooling fits** impose `< cutoff`. No goal, shot-count or previous candidate search is rerun. Every inherited source hash must match its original evidence.
3. Preparation joins only fixture identity/date and pre-closing Avg/Bet365 columns, writes a target-free table, records reconstruction parameters/training membership, then closes `data_lock.json`. Training rows lacking either complete price triple are excluded from **all four pooling/calibration fits** together and reported. Missing selection prices abort the planned complete-cohort experiment before scores. The loader retains a common internal fallback for explicit missing snapshots; this is not a permission to shrink the 760-row cohort.
4. `select` verifies that lock and trains both candidates and their matched market-only references on the same past rows at each inherited EPL refit cutoff. Timestamp parsing normalizes actual UTC offsets; legacy naive artifact timestamps mean UTC. Forecast and result-availability conditions are checked before accessing training prices, probabilities or labels. Current-batch and equal-availability rows cannot enter fitting. Previously resolved selection outcomes may enter later prequential refits.
5. All 760 current-row forecasts and fit parameters are closed and hashed before the final selection label-attachment/scoring pass. Every candidate and required reference is reported. Advancement requires the exact all-reference improvement and interval gates from `specification.json`. There is currently **no transfer command**; any later runner must enforce the hash-bound selection gate before reading later outcome values.

Neither the new refit clock nor the carried midnight internal forecast proves availability before the provider's **unknown** quote time. This remains a retrospective snapshot comparison with uncertified original receipt timing. It cannot support a same-horizon midnight upgrade, and neither `fit_cutoff_utc` nor a fixture kickoff time fills the missing quote timestamp.

## Commands after independent review

Run from the repository root with one numerical thread:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv-f1/bin/python -m pytest -q --import-mode=importlib -p no:cacheprovider research/experiments/boundary_20260908/football_information
```

The following stages are implemented but were not invoked while preparing this source. The review receipt must actually exist and approve the current source fingerprint; the path below is the intended location, not a claim that approval has already occurred.

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv-f1/bin/python -m research.experiments.boundary_20260908.football_information.run freeze --review artifacts/research/boundary_20260908/football_information/independent_review.json
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv-f1/bin/python -m research.experiments.boundary_20260908.football_information.run prepare
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv-f1/bin/python -m research.experiments.boundary_20260908.football_information.run select
```

The runner fixes numerical thread counts before imports and serializes fits. Each output is created exclusively; failures or partial runs are preserved, not overwritten. A clean retry uses an explicitly different `--out` directory and the same reviewed source/input hashes.

Suggested commit: `research(football): implement reviewed market snapshot selection pipeline`.
