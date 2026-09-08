# Football data opportunity after two completed model cycles

**Recommendation, not a measured gain:** prioritize timestamped shot-quality/xG events and squad availability or starting lineups. Aggregate shot volume cannot identify the quality of chances or which players will be available. Those inputs do not exist in the cached league CSVs. Another nearby architecture search on the same information is lower priority after the small affine gain and failed nonlinear challenge. Nothing here establishes that richer data will deliver a particular percentage gain.

The following existing files were inspected directly, with 380 rows each:

- `data/football/performance_20260907/E0_2025_2026.csv`
- `data/football/performance_20260907/transfer_SP1_2025_2026.csv`
- `data/football/performance_20260907/transfer_I1_2025_2026.csv`

In all three, `HTHG/HTAG/HTR` (halftime result), `HC/AC` (corners), `HF/AF` (fouls), `HY/AY` (yellow cards), `HR/AR` (red cards), and kickoff `Time` are complete. `Referee` is present and complete for EPL but absent in the Spanish and Italian files. These are concrete unconsumed or partially unconsumed aggregate signals, suitable for causal lagged features. They contain no shot locations, player lineups, injuries, player-event data or xG. They should not be sold as evidence of an imminent material gain.

A potentially stronger existing information source is the pre-closing market: `B365H/B365D/B365A`, `AvgH/AvgD/AvgA`, plus goals/handicap prices are complete in these three files. Their availability contract is essential. The cached primary schema at `data/football/performance_20260907/notes.txt:48` defines non-C fields as pre-closing and C fields as closing; it does not call every non-C value a literal opening quote. Line 198 describes Friday-afternoon collection for weekend games and Tuesday-afternoon collection for midweek games. Same-day Friday/Tuesday forecasts at midnight therefore cannot assume these quotes were already available. Individual observed quote/publication timestamps are missing. A future market-informed experiment needs an explicitly later cutoff or archived as-of snapshots; closing prices cannot be silently substituted into the current midnight contract.

The cached primary `data_source.html:194` also says Pinnacle delivery became unreliable from 2025-07-23 and its stale prices were removed from market averages. Pinnacle-only values are therefore an avoidable additional data-quality issue in that period. Line 244 describes at-least-twice-weekly file updates, Sunday/Wednesday nights. This does not prove an absence of faster updates, but it does mean next-midnight shot-statistics availability is an assumption, not a verified publication-time replay.

These source facts were checked against the already acquired primary pages. A fresh read of the provider's live notes/data pages returned HTTP 503 during this review. No paid data, credentials or new dataset was acquired. The current experiments retain their explicit availability caveat and remain retrospective.

Suggested commit: `research: document football data requirements after residual benchmarks`.
