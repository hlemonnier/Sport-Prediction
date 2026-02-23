#!/usr/bin/env python3
"""Compatibility wrapper for profile workflows.

Use run_experiment.py as the canonical entrypoint.
"""

from __future__ import annotations

from run_experiment import main


if __name__ == "__main__":
    main(default_runner="profile")
