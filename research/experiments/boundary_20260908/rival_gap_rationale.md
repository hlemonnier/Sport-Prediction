# Race-order gaps as a separate information hypothesis

The current next-lap input builder includes race rank, a driver's recent lap
history and aggregate peer pace. Its `lag_*_gap` variables are differences
between that driver's own lap times, not intervals between cars. The current
80-feature HGB therefore has no explicit `IntervalToPositionAhead` or
`GapToLeader` input. The telemetry summaries and sequence experiment describe
the driver's own car and do not resolve that omission.

The hypothesis is specific: knowing the currently reported gap may distinguish
two otherwise similar histories with different opportunities to catch, follow
or separate from another driver. An absolute gap may add little because the
last observed lap already reflects traffic. Consequently, a successful
measurement pilot would establish availability only; it would not establish
incremental predictive information or justify a model promotion.

The [two-race pilot](rival_gap_pilot/README.md) measures original-issuance coverage
at fixed latency cutoffs without labels, fitting or prediction scores. Numeric
seconds, leader markers, lap deficits, missing observations and rank ambiguity
remain distinct. Race-order gaps do not identify the nearest physical car when
cars are lapped, nor do they establish the causal effect of following a car.

The state-space study by [Cappello and Hoegh](https://arxiv.org/html/2512.00640v1)
uses a single driver's comparatively uneventful race, assumes a linear fuel
trajectory and compares against ARIMA. It motivates robust observation models
and explicitly identifies traffic as a limitation. Those results do not
demonstrate an improvement over this repository's HGB on its fixed population.

There is also an identification issue in interpreting latent pace physically.
For the illustrative within-stint model

\[
y_t = a_s + \nu_s t + \gamma(F_s-c t)+\epsilon_t,
\]

only \(a_s+\gamma F_s\) and \(\nu_s-\gamma c\) enter its likelihood when the
intercept and slope are free. Replacing \(\gamma\) with \(\gamma+\delta\),
\(a_s\) with \(a_s-\delta F_s\), and \(\nu_s\) with
\(\nu_s+c\delta\) leaves every prediction unchanged. Independent fuel or
degradation information, or explicit identifying restrictions, is needed to
separate those components. Better predictions alone would not identify them.

The information hypothesis above was written before the gap pilot's first
historical execution. It does not change the frozen telemetry sequence test or
declare a new fitted candidate. Any subsequent prediction test must retain all
original issuances, preserve explicit fallback, compare against the current
strong reference and its missingness control, and state its selection and
later-season rules before fitting.

The operational assessment below was added after the target-free pilot. The
production code already subscribes to OpenF1's interval topic in
`services/f1-platform/f1_platform/openf1.py`. This is a configured transport
path, not a verified active entitlement or connectivity check. The current
reducer keeps latest gap strings without field-specific observation history;
its `or driver.interval` update also retains a prior value after an explicit
clear. The prediction-service normalization substitutes the interval when the
leader gap is absent, and `_gap_seconds` returns zero for unavailable or
unrecognized values. Those projections are unsuitable as inputs to the proposed
measurement model.

Any successful candidate would therefore need timestamped, presence-sensitive
gap and rank observations frozen at each original issuance, including the
distinction between provider time and API receipt time. The current HGB remains
the fallback. Appending the latest driver gap to earlier lap rows would violate
that information boundary. OpenF1's [official interval documentation](https://openf1.org/docs/#intervals)
describes roughly four-second updates during races and authenticated live
access; the archived TimingData pilot does not establish identical live-feed
latency or coverage. This operational note does not modify production code.

Suggested commit: `research(f1-live): motivate causal race-order gap information test`.
