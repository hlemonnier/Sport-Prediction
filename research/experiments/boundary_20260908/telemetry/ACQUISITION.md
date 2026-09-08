# Fixed discovery telemetry acquisition

Acquire CarData.z for exactly the original44 race sessions in2022–2023. Reuse the two completely verified Bahrain pilots without new requests or copies. The42 new jobs use the unchanged,128-test-validated pilot transport, configured in an isolated module instance with this experiment's output directory. Frozen source bytes remain unchanged.

Each job permits one mirror request and one official fallback, at most two jobs concurrently. Both wire and decoded bodies have64MiB caps; redirects are disabled. The150-second limit is a response-acceptance deadline checked before and after each read1, with a30-second socket timeout, rather than hard cancellation of all transport/DNS work. TwelveGiB free disk is required, exceeding worst-case body storage. Every attempt preserves exclusive body/receipt files; no completed or interrupted acquisition is silently overwritten. The contract, helper hash, orchestration source, tests and this note are locked before requests.

This stage reads session metadata and existing hashes only. It performs no model fitting, lap-label attachment, predictive scoring or later-season acquisition. Successful new responses remain downloaded_unparsed until the feature-input validation phase. An explicitly unavailable source retains the event's original forecast population with empty telemetry and incumbent fallback. A malformed downloaded stream stops feature construction before fitting; existing ledgers are preserved. No event is deleted for data quality or poor performance.

The separate model/feature protocol governs strict original issuance cutoffs and the controlled90-column experiment. The archive clock remains a proxy for source availability, not a historical client receipt. Raw files stay under data/f1/boundary_20260908/telemetry and closed manifests under the matching artifacts/research directory.

Suggested commit: research(f1-live): acquire fixed discovery telemetry with bounded transport.
