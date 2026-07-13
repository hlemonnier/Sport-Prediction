from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[6]


def _run_import_probe(source: str) -> dict[str, object]:
    completed = subprocess.run(
        [sys.executable, "-c", source],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def test_narrow_f1_model_import_does_not_eagerly_load_training_backends() -> None:
    payload = _run_import_probe(
        """
import json
import sys
import packages.f1.models.probability
print(json.dumps({
    'xgboost': 'xgboost' in sys.modules,
    'lightgbm': 'lightgbm' in sys.modules,
    'training': 'packages.f1.models.training' in sys.modules,
}))
"""
    )

    assert payload == {"xgboost": False, "lightgbm": False, "training": False}


def test_package_root_keeps_existing_public_exports() -> None:
    payload = _run_import_probe(
        """
import json
import packages.f1 as f1
print(json.dumps({
    'circuit_card': f1.CircuitCard.__name__,
    'prediction_config': f1.PredictionConfig.__name__,
    'has_run_prediction': callable(f1.run_prediction),
}))
"""
    )

    assert payload == {
        "circuit_card": "CircuitCard",
        "prediction_config": "PredictionConfig",
        "has_run_prediction": True,
    }
