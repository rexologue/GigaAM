#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from omegaconf import OmegaConf

from diarizer import Diarizer
from pipeline.config_schema import validate_config
from pipeline.manifest import (
    append_jsonl,
    atomic_json_dump,
    iter_glob,
    load_journal_latest,
    read_input_manifest,
    should_skip_by_resume,
)
from pipeline.modes import run_diar_cut_then_asr, run_full_asr_then_align
from word_timestamps import GigaAMWordTimestampTranscriber


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_input_paths(cfg: dict) -> List[dict]:
    audio_dir = Path(cfg["input"]["audio_dir"])
    manifest_path = cfg["input"].get("manifest_path")
    items = []
    if manifest_path:
        for it in read_input_manifest(Path(manifest_path)):
            p = Path(it.path)
            abs_path = p if p.is_absolute() else (audio_dir / p)
            items.append({"id": it.id, "path": str(abs_path.resolve()), "duration_sec": it.duration_sec})
    else:
        for it in iter_glob(audio_dir, cfg["input"].get("glob", "*.mp3")):
            items.append({"id": it.id, "path": str((audio_dir / it.path).resolve()), "duration_sec": it.duration_sec})
    return items


def main() -> int:
    parser = argparse.ArgumentParser(description="Dialog pipeline via YAML config")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    raw_cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    cfg = validate_config(raw_cfg).config

    out_dir = Path(cfg["output"]["out_dir"])
    transcripts_dir = out_dir / cfg["output"].get("transcripts_dir", "transcripts")
    meta_asr_dir = out_dir / cfg["output"].get("meta_asr_dir", "meta_asr")
    meta_diar_dir = out_dir / cfg["output"].get("meta_diar_dir", "meta_diar")
    journal_path = out_dir / "manifest.jsonl"

    diarizer = Diarizer(
        model_name=cfg["diarizer"].get("model_name", "nvidia/diar_streaming_sortformer_4spk-v2.1"),
        device=cfg["diarizer"].get("device"),
        chunk_len=int(cfg["diarizer"].get("chunk_len", 340)),
        chunk_right_context=int(cfg["diarizer"].get("chunk_right_context", 40)),
        fifo_len=int(cfg["diarizer"].get("fifo_len", 40)),
        spkcache_update_period=int(cfg["diarizer"].get("spkcache_update_period", 300)),
    )
    asr = GigaAMWordTimestampTranscriber(
        model_name=cfg["asr"].get("model_name", "v3_ctc"),
        device=cfg["asr"].get("device"),
        use_vad=bool(cfg["asr"].get("use_vad", False)),
        max_chunk_s=float(cfg["asr"].get("max_sec", 22.0)),
        overlap_s=float(cfg["asr"].get("overlap_sec", 1.0)),
        hf_token=cfg["asr"].get("hf_token"),
        batch_size=int(cfg["asr"].get("batch_size", 8)),
    )

    mode_fn = run_full_asr_then_align if cfg["pipeline"]["mode"] == "full_asr_then_align" else run_diar_cut_then_asr
    latest = load_journal_latest(journal_path)
    items = _resolve_input_paths(cfg)

    for item in items:
        input_path = item["path"]
        item_id = item["id"]

        row_base = {
            "input_path": input_path,
            "id": item_id,
            "started_at": _utc_now(),
            "config": cfg,
        }

        if bool(cfg["resume"].get("enabled", True)) and should_skip_by_resume(
            input_path,
            latest,
            retry_failed=bool(cfg["resume"].get("retry_failed", False)),
        ):
            append_jsonl(journal_path, {**row_base, "status": "SKIPPED", "reason": "resume", "finished_at": _utc_now()})
            continue

        tr = transcripts_dir / f"{item_id}.json"
        ma = meta_asr_dir / f"{item_id}.json"
        md = meta_diar_dir / f"{item_id}.json"
        if bool(cfg["resume"].get("skip_if_outputs_exist", True)) and tr.exists() and ma.exists() and md.exists():
            append_jsonl(journal_path, {**row_base, "status": "SKIPPED", "reason": "outputs_exist", "finished_at": _utc_now()})
            continue

        try:
            diar_out = diarizer.diarize([input_path], batch_size=1)[0]
            result = mode_fn({"audio_path": input_path, "diar_out": diar_out, "asr": asr, "cfg": cfg})
            if result["status"] == "BAD_SAMPLE":
                append_jsonl(
                    journal_path,
                    {
                        **row_base,
                        "status": "BAD_SAMPLE",
                        "reason": result["reason"],
                        "shares": result.get("shares"),
                        "speakers": result.get("speakers"),
                        "finished_at": _utc_now(),
                    },
                )
                continue

            transcript = result["transcript"]
            transcript["file_id"] = item_id
            atomic_json_dump(tr, transcript)
            atomic_json_dump(ma, result["meta_asr"])
            atomic_json_dump(md, result["meta_diar"])
            append_jsonl(journal_path, {**row_base, "status": "SUCCESS", "reason": None, "finished_at": _utc_now()})
        except Exception as exc:  # noqa: BLE001
            append_jsonl(journal_path, {**row_base, "status": "FAILED", "reason": repr(exc), "finished_at": _utc_now()})

    print(json.dumps({"status": "done", "journal": str(journal_path), "num_items": len(items)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
