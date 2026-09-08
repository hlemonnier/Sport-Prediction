# Executing the fixed Elo-Odds follow-up

The original README and specification are the immutable experimental design. This implementation adds a matched result-Elo cursor, an execution runner, and a separate verifier. No outcomes or gains are asserted by this guide.

Use Python 3.12 with the repository research dependencies and one CPU thread:

```sh
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=.venv-f1/bin/python
MODULE=research.experiments.boundary_20260908.football_elo_followup

"$PY" -m pytest -q --import-mode=importlib -p no:cacheprovider research/experiments/boundary_20260908/football_elo_followup
"$PY" -m "$MODULE.run" freeze \
  --review artifacts/research/boundary_20260908/football_elo_followup_review.json \
  --feasibility artifacts/research/boundary_20260908/football_elo_followup_synthetic.json
"$PY" -m "$MODULE.run" prepare
"$PY" -m "$MODULE.run" selection
"$PY" -m "$MODULE.verify" selection --delay 1
```

The review must approve the exact current source map. The synthetic receipt must bind the new Elo cursor and inherited ordered-logit kernel. Freezing reruns the synthetic suite and checks the complete inherited input graph. Raw cached data and large historical ledgers are local, hash-bound prerequisites; a source checkout alone supports synthetic tests but cannot reproduce the historical run.

Advance only when both the result and independent verification report literal `passes_all_gates: true`. A failed gate ends this experiment. The runner enforces these dependencies:

```sh
# Only after independently verified primary selection passes:
"$PY" -m "$MODULE.run" sensitivity
"$PY" -m "$MODULE.verify" selection --delay 7

# Only after both independently verified selection delays pass:
"$PY" -m "$MODULE.run" transfer --delay 1
"$PY" -m "$MODULE.verify" transfer --delay 1

# Only after the first transfer also independently passes:
"$PY" -m "$MODULE.run" transfer --delay 7
"$PY" -m "$MODULE.verify" transfer --delay 7
```

The historical optimizer budget is one new readout in primary selection and two in sensitivity, with zero transfer or strength fits. Primary market readout bytes and candidate probability values are copied exactly. Later transfer uses the same saved readouts. The verifier independently reconstructs rating states, readout inputs and objective gradients, probabilities, uncertainty intervals and gates without fitting. It must run in a fresh process without operational modules imported.

Exclusive attempt files preserve interrupted or failed work. Do not restart a failed stage in another output directory to erase the failure. A genuine implementation failure requires an explicitly documented corrected version; a failed performance gate permits no retuning of this fixed candidate. Every stage closes its complete target-free forecast ledger before scoring labels are attached.

Suggested commit: `research(football): execute the fixed Elo-Odds follow-up`.
