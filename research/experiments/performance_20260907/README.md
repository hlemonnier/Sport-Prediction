**Performance research published from the 7 September 2026 cycle**

Start with the [findings and mathematical interpretation](../../../docs/research/performance_research_20260907.md), the [31-comparison scorecard](../../../docs/research/evidence/performance_research_20260907_scorecard.csv), and its [six canonical result hashes](../../../docs/research/evidence/performance_research_20260907_summary.json).

The publication includes all four research lanes and both frozen transfer extensions: source, fixed specifications, tests, selection locks, selected-model parameters, positive and negative summary results, and verification records. No model is promoted. The numerical findings and frozen experiment source files are unchanged from the verified research cycle.

| Lane | Reproduction guide | Canonical evidence |
|---|---|---|
| Pre-event F1 | [Guide](pre_event/README.md) | [Results](../../../artifacts/research/performance_20260907/pre_event/cycle1/results.json) |
| Race order | [Guide](race/README.md) | [Results](../../../artifacts/research/performance_20260907/race/v3/results.json) |
| Live next eligible lap | [Guide](live/README.md) | [Historical transfer](../../../artifacts/research/performance_20260907/live/corrected_cycle_1/results.json), [recent transfer](../../../artifacts/research/performance_20260907/live/recent_extension_final/results.json) |
| Football | [EPL guide](football/README.md), [transfer guide](football/transfer_README.md) | [EPL](../../../artifacts/research/performance_20260907/football/evidence.json), [Spain/Italy](../../../artifacts/research/performance_20260907/football/transfer_evidence.json) |

The [publication inventory](../../../docs/research/evidence/performance_research_20260907_publication.json) defines the shipped files and their exact hashes. Experiment-time manifests intentionally retain their original paths, timestamps, branch and local-run status. Some record files outside this publication, including raw inputs or a duplicate reproduction. The publication inventory is the packaging authority; the original manifests remain the provenance record. The main report received only portable links and a publication-status clarification, recorded with its before/after hashes.

Provider CSVs, downloaded web pages, row-level prediction/target tables, FastF1 caches, Python environments and duplicate replay outputs remain local. Matched-forecast hashes remain in the evidence; reproducing those rows requires the corresponding inputs and runners. Source manifests retain acquisition URLs and hashes; the acquisition scripts recover provider inputs. Complete F1 model refitting also needs the historical inputs named in each result manifest, recovered through the repository's data ingestion workflow. Later provider revisions can prevent an exact raw-input hash match and must not be silently treated as the original experiment.

The full source manifests for [EPL](../../../data/football/performance_20260907/source_manifest.json) and [Spain/Italy](../../../data/football/performance_20260907/transfer_source_manifest.json) accompany the derived evidence. The package does not assert that third-party source archives have an unrestricted redistribution license.

The failed first live execution and superseded race executions are excluded as canonical evidence. Their diagnoses are retained in the lane guides and main report; hashes of omitted research artifacts are recorded in the publication inventory. Their exclusion does not remove valid losing models or failed transfer results.

The shared scorecard can be regenerated from the committed result JSON files without downloading inputs:

```sh
python3 research/experiments/performance_20260907/summarize.py
```

The verified research suite passed 31 tests. Its [original log](../../../artifacts/research/performance_20260907/verification/focused_tests.log) and [verification snapshot](../../../artifacts/research/performance_20260907/verification/cycle_verification.json) are included. Model reruns require the documented Python scientific dependencies and the appropriate raw inputs. Output guards require fresh experiment directories.

Commit name for this publication: `docs(research): package verified performance findings for main`.
