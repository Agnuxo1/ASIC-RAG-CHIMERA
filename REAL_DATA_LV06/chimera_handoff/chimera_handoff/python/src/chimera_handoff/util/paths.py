from __future__ import annotations

import sys
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_ml_on_path() -> None:
    """
    Make `ml/eigenlearner` importable as `import eigenlearner` for the reservoir project.
    """
    root = repo_root()
    ml = root / "ml"
    if ml.exists() and str(ml) not in sys.path:
        sys.path.insert(0, str(ml))


def ensure_out_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
