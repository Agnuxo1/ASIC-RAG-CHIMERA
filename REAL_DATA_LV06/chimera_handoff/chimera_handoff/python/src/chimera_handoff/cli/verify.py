from __future__ import annotations

import argparse
from pathlib import Path

from chimera_handoff.schema import validate_run_root


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="Run root to validate (must contain protocol/manifest and runs/*/seed_*/ artifacts).")
    args = ap.parse_args()
    validate_run_root(Path(args.path))
    print("ok:", str(Path(args.path)))

