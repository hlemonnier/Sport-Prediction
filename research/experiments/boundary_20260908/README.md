# Boundary experiments — 8 September 2026

The existing verified HGB point model is already the default in the global F1 prediction system. None of this directory's completed challengers passed its predeclared substantial-gain criteria. They remain research; the production model, feature contract and forecast issuance policy are unchanged.

Read the [publication and integration check](../../../docs/research/boundary_publication_20260908.md) for the latest status. The earlier [results snapshot](../../../docs/research/boundary_research_20260908.md) and [machine-readable journal](../../../docs/research/evidence/boundary_research_20260908_journal.json) are preserved as originally recorded; their weather-in-progress statement predates the completed weather experiment.

| Directory | Status |
| --- | --- |
| `pre_event/` and `pre_event/cycle2/` | Qualifying transfer completed; neither candidate passed |
| `online_residual/`, `online_refit/`, `anchor/` | Original-issuance live selection failed; no later candidate evaluation |
| `checkpoint/` | Later-issuance partial gain; below magnitude criteria and inconclusive contemporary transfer |
| `checkpoint/cycle2/` | Peer correction failed selection against checkpoint cycle1 |
| `composition/` | Exact fallback to checkpoint cycle1; no additional improvement |
| `football/` | Shot-strength correction yields a small improvement over the strong blend; promotion criteria failed |
| `football/cycle2/` | All three tree models lose on selection; incumbent retained |
| `weather/` | Discovery acquisition and quality checks complete; all four weather variants lose on selection |
| `control/` | Acquisition and parser verification only; paused before model design or fitting |
| `distributional/` | Frozen protocol and untested implementation draft; paused before fitting, with no performance claim |
| `live_review/` | Read-only input inventory |

The release contains experiment source, specifications, locks, result JSON and verification records. Large raw feeds, feature caches, pickles and row-level CSVs remain local, identified by the published manifests. Reproducing full experiments needs those inputs and the earlier frozen frontier artifacts. Published summary metrics and JSON evidence are reviewable without them. The publication manifest explicitly lists included files and omitted local artifact hashes.

Run the synthetic and mathematical regressions from the repository root:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$PWD:$PWD/research/projects/F1/rising_qualification_prediction/Python:$PWD/services/f1-platform:$PWD/services/f1-prediction-service" .venv-f1/bin/python -m pytest -q --tb=short --import-mode=importlib research/experiments/boundary_20260908
```

Result at publication: **83 passed**. Use `--import-mode=importlib` because independent experiment directories reuse test filenames. The distributional draft has no tests and is not covered by this count. Each completed lane's README gives its artifact-dependent replay verifier. Design-lock refusal to overwrite frozen results is intentional.

Suggested commit: `docs: publish boundary research and verify global integration`.
