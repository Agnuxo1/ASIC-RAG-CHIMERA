from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple


@dataclass(frozen=True)
class ManifestEntry:
    path: str
    sha256: str


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(int(chunk_size))
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def write_manifest_sha256(manifest_path: Path, *, files: Iterable[Path]) -> List[ManifestEntry]:
    rows: List[Tuple[str, str]] = []
    base = manifest_path.parent.resolve()
    for p in sorted({Path(x).resolve() for x in files}):
        if not p.is_file():
            continue
        rel = str(p.relative_to(base))
        rows.append((rel, sha256_file(p)))

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        for rel, digest in rows:
            f.write(f"{digest}  {rel}\n")

    return [ManifestEntry(path=rel, sha256=digest) for rel, digest in rows]


def list_files_for_manifest(root: Path) -> List[Path]:
    return [p for p in root.rglob("*") if p.is_file() and p.name != "MANIFEST.sha256"]

