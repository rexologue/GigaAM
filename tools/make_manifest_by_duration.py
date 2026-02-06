#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from tqdm import tqdm


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate JSONL manifest sorted by duration with mutagen")
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--glob", default="*.mp3")
    args = parser.parse_args()

    try:
        from mutagen import File as MutagenFile
    except Exception:
        print("mutagen is not installed. Install optional dependency: pip install mutagen")
        return 2

    audio_dir = Path(args.audio_dir)
    out_path = Path(args.out)
    files = sorted(audio_dir.glob(args.glob))

    rows: List[dict] = []
    for p in tqdm(files, desc="Reading audio metadata"):
        st = p.stat()
        mf = MutagenFile(str(p))
        duration = float(mf.info.length) if (mf is not None and getattr(mf, "info", None) is not None) else 0.0
        rows.append(
            {
                "path": str(p.relative_to(audio_dir)),
                "duration_sec": duration,
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

    print(json.dumps({"written": len(rows), "out": str(out_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
