#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm


Turn = Dict[str, object]
DialogRow = Dict[str, str]


def merge_adjacent_turns(turns: List[Turn]) -> List[Turn]:
    merged: List[Turn] = []

    for turn in turns:
        speaker = str(turn.get("speaker", "")).strip()
        text = str(turn.get("text", "")).strip()
        if not speaker or not text:
            continue

        if merged and merged[-1]["speaker"] == speaker:
            merged[-1]["text"] = f"{merged[-1]['text']} {text}".strip()
        else:
            merged.append({"speaker": speaker, "text": text})

    return merged


def turns_to_dialog_rows(turns: List[Turn]) -> List[DialogRow]:
    rows: List[DialogRow] = []
    current: DialogRow = {}

    for turn in turns:
        speaker = str(turn["speaker"])
        text = str(turn["text"])

        if speaker in current and len(current) >= 2:
            rows.append(current)
            current = {}

        if speaker in current:
            current[speaker] = f"{current[speaker]} {text}".strip()
        else:
            current[speaker] = text

    if current:
        rows.append(current)

    return rows


def process_file(path: Path) -> List[DialogRow]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    turns = payload.get("turns", [])
    if not isinstance(turns, list):
        return []

    merged_turns = merge_adjacent_turns(turns)
    return turns_to_dialog_rows(merged_turns)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read JSON transcripts and aggregate speaker replicas into dialog rows"
    )
    parser.add_argument(
        "result_dir",
        help="Path to the pipeline result directory (must contain transcripts/)",
    )
    parser.add_argument(
        "--transcripts-dir",
        default="transcripts",
        help="Transcripts directory name inside result_dir (default: transcripts)",
    )
    parser.add_argument(
        "--out-dir",
        default="dialogs",
        help="Output directory name inside result_dir (default: dialogs)",
    )
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    transcripts_dir = result_dir / args.transcripts_dir
    out_dir = result_dir / args.out_dir

    if not transcripts_dir.exists() or not transcripts_dir.is_dir():
        raise SystemExit(f"Transcripts directory not found: {transcripts_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(transcripts_dir.glob("*.json"))
    written = 0

    for path in tqdm(json_files, desc="Processing transcripts"):
        rows = process_file(path)
        out_path = out_dir / path.name
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
            f.write("\n")
        written += 1

    print(
        json.dumps(
            {
                "processed": len(json_files),
                "written": written,
                "transcripts_dir": str(transcripts_dir),
                "out_dir": str(out_dir),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
