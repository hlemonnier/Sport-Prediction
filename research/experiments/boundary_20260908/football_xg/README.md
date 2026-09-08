# Historical xG model — unfinished, unfitted draft

This directory preserves a proposed four-configuration experiment and initial model functions. It has no runner, tests, frozen execution lock, fitted model, predictions or performance result. It is not imported by the global prediction system. The separate [data acquisition](../football_xg_data/README.md) is complete and verified; data availability does not establish predictive improvement.

Work paused for the user's integration and publication request. The draft needs review before any execution. In particular, its `available_at` adds local calendar days before converting to UTC, whereas the subsequently finalized [data contract](../../../../artifacts/research/boundary_20260908/football_xg_data/availability_contract.json) converts next local midnight to UTC and then adds 168 or 336 elapsed hours. These differ around daylight-saving transitions. The draft also expects an actual boolean for join acceptance; raw CSV strings require explicit parsing. Neither behavior has generated forecasts or affected production.

Before fitting, reconcile that availability rule, implement the runner and strict earlier-data filtering, add mathematical and causal tests, obtain an independent pre-fit review, and freeze source/input/selection contracts. The proposed comparison remains against the stronger goal and shot models on the same forecast horizon and match population. No score, promotion or substantial gain is claimed.

Suggested commit: `research(football): preserve unfitted xG experiment proposal`.
