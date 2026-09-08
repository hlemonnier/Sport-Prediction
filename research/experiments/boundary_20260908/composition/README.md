The predeclared composition adds no gain. Both new components failed their
2023 advancement rules: the robust anchor increased original-issuance MAE,
and fresh-peer checkpoint corrections increased checkpoint MAE versus cycle1.
The fixed branch rule therefore retains current HGB at eligible observations
and the selected cycle1 checkpoint expert at ineligible observations.

The compositor was tested on shuffled keys, repeated targets, missing or
duplicated observations, mismatched targets and invalid point values. Its full
61,948-row output is exactly identical to cycle1. The 55,757 original eligible
forecasts remain unchanged. Historical checkpoint gain is still **6.879%**;
the composition fails its substantial-gain screen and its requirement to
improve on cycle1. No new anchor or peer transfer predictions were read.

`specification.json` was frozen before either component's new selection scores;
`component_lock.json` binds the failed screens before the old cycle1 transfer
table was opened for composition. `verify_fallback.py` preserves the resulting
identity check and per-year, target-balanced and subgroup reports under
`artifacts/research/boundary_20260908/composition/`.

Suggested commit: `research(f1-live): evaluate observed-regime expert composition`.
