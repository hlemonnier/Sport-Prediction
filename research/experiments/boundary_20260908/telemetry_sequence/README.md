# Causal telemetry sequence prototype

Status: **synthetic implementation only; no historical representation training,
lap-time fit, evaluation or runtime activation**. The existing global prediction
model remains the validated `frontier_live_hgb_l15_i150_20260907`.

The [preceding proposal](../telemetry_next/idea.md) was written before the
telemetry-summary selection result. That experiment subsequently failed its
frozen advancement gate; its [closed result](../../../../docs/research/boundary_telemetry_selection_20260908.md)
does not establish any gain from this separate sequence hypothesis.

`tokens.py` preserves up to 128 ordered driver packets as 34 bounded coordinates
per packet. It reuses the frozen channel semantics, exact archive prefix clock,
strict availability cutoff and full uncapped 30-second support gate. Its matched
permutation moves earlier measurement and quality bundles while retaining the
newest packet, chronological gaps and padding. The permutation key contains only
the event, driver and last admitted packet sequence.

`encoder.py` implements a 15,048-parameter causal convolutional encoder and
contrastive predictive coding objective. The ordered and permuted controls use
identical initialization and accept an externally supplied batch schedule. A
third representation uses the initial untrained weights. Inference accepts only
context tokens. The implementation fixes 800 Adam steps, batch size 32, a 0.001
learning rate and a final-checkpoint policy; it does not choose those settings
from historical scores.

The source is **not a complete experiment runner**. Before historical use, a new
execution specification must freeze the source graph, 2022-only corpus, sampling
schedule, resource limits, target-free feature and forecast closures, original
issuance parity, supervised controls and later evaluation/refit policy. The
sampler must enforce same-race/driver negatives at least 180 seconds away and
exclude all three positive packet identities. These conditions cannot be
established by the encoder's numerical input checks. No later-season telemetry
acquisition is part of this prototype.

Independent review and **86 synthetic tests** cover packet and stream parity,
future-data poisoning, equal-clock boundaries, support before sequence truncation,
permutation invariants, strictly causal outputs and gradients, manual InfoNCE
values/derivatives, parameter counts, seeded reproducibility and deadline failure.
A constructed pair has identical existing 170 predictors and different ordered
token inputs; that demonstrates additional representational information, not
predictive usefulness. These tests require PyTorch and run locally; the existing
research CI job does not install PyTorch or run this prototype.

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv-f1/bin/python -m pytest -q --import-mode=importlib -p no:cacheprovider research/experiments/boundary_20260908/telemetry_sequence/test_tokens.py research/experiments/boundary_20260908/telemetry_sequence/test_encoder.py
```

Suggested commit: `research(f1-live): preserve causal telemetry sequence prototype`.
