#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm

try:
    import mutagen
except Exception:
    print("mutagen is not installed. Install optional dependency: pip install mutagen")
    raise SystemExit(2)


def _duration_ffprobe(path: Path) -> Optional[float]:
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if p.returncode != 0:
            return None
        s = (p.stdout or "").strip()
        if not s:
            return None
        dur = float(s)
        if dur != dur or dur < 0:
            return None
        return dur
    except FileNotFoundError:
        return None
    except Exception:
        return None


def _duration_mutagen(path: Path) -> Optional[float]:
    try:
        mf = mutagen.File(str(path))
        if mf is None or getattr(mf, "info", None) is None:
            return None
        length = getattr(mf.info, "length", None)
        if length is None:
            return None
        dur = float(length)
        if dur != dur or dur < 0:
            return None
        return dur
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate JSONL manifest sorted by duration with mutagen")
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--glob", default="*.mp3")
    parser.add_argument(
        "--drop-zero",
        action="store_true",
        help="Drop files with duration_sec <= 0 after all duration detection attempts",
    )
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir)
    out_path = Path(args.out)
    files = sorted(audio_dir.glob(args.glob))

    rows: List[dict] = []
    dropped_zero = 0

    for p in tqdm(files, desc="Reading audio metadata"):
        st = p.stat()

        duration: Optional[float] = None
        duration_source = "none"

        duration = _duration_mutagen(p)
        if duration is not None:
            duration_source = "mutagen"
        else:
            duration = _duration_ffprobe(p)
            if duration is not None:
                duration_source = "ffprobe"
            else:
                duration = 0.0
                duration_source = "none"

        duration_f = float(duration)

        if args.drop_zero and duration_f <= 0.0:
            dropped_zero += 1
            continue

        rows.append(
            {
                "path": str(p.relative_to(audio_dir)),
                "duration_sec": duration_f,
                "duration_source": duration_source,
                "bytes": int(st.st_size),
                "mtime": float(st.st_mtime),
                "id": p.stem,
            }
        )

    rows.sort(key=lambda x: (x["duration_sec"], x["path"]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {"written": len(rows), "dropped_zero": dropped_zero, "out": str(out_path)},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
