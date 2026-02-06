from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional


@dataclass(frozen=True)
class ManifestItem:
    path: str
    id: str
    duration_sec: Optional[float] = None


FINAL_STATES = {"SUCCESS", "BAD_SAMPLE", "SKIPPED"}


def atomic_json_dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=str(path.parent)) as tmp:
        json.dump(payload, tmp, ensure_ascii=False, indent=2)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = tmp.name
    os.replace(tmp_path, path)


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_input_manifest(manifest_path: Path) -> List[ManifestItem]:
    items: List[ManifestItem] = []
    suffix = manifest_path.suffix.lower()
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if suffix == ".txt":
                p = line
                item_id = Path(p).stem
                items.append(ManifestItem(path=p, id=item_id))
            else:
                row = json.loads(line)
                p = row["path"]
                item_id = row.get("id") or Path(p).stem
                items.append(ManifestItem(path=p, id=item_id, duration_sec=row.get("duration_sec")))
    return items


def iter_glob(audio_dir: Path, pattern: str) -> Iterator[ManifestItem]:
    for p in sorted(audio_dir.glob(pattern)):
        rel = str(p.relative_to(audio_dir))
        yield ManifestItem(path=rel, id=p.stem)


def load_journal_latest(journal_path: Path) -> Dict[str, dict]:
    latest: Dict[str, dict] = {}
    if not journal_path.exists():
        return latest
    with open(journal_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            latest[row["input_path"]] = row
    return latest


def should_skip_by_resume(
    input_path: str,
    latest: Dict[str, dict],
    retry_failed: bool,
) -> bool:
    prev = latest.get(input_path)
    if not prev:
        return False
    status = prev.get("status")
    if status in FINAL_STATES:
        return True
    if status == "FAILED":
        return not retry_failed
    return False
