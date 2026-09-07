"""Rerun the fixed benchmark into a fresh isolated research artifact folder."""
import argparse
import re
import shutil
import sys
from research.experiments.performance_20260907.football import benchmark as b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_name):
        raise ValueError("Use letters, digits, underscores or hyphens for the replay name")
    target = b.OUT / "replays" / args.run_name
    target.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(b.OUT / "design_lock.json", target / "design_lock.json")
    b.OUT = target
    for phase in ("validation", "test", "report"):
        sys.argv = [str(b.LANE / "benchmark.py"), "--phase", phase]
        b.main()


if __name__ == "__main__":
    main()
