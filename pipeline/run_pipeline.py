#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import json
import multiprocessing as mp
import os
import sys
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, List, Optional, Sequence

sys.path.append(str(Path(__file__).parent.parent))

from omegaconf import OmegaConf
from tqdm import tqdm

from pipeline.config_schema import validate_config
from pipeline.diarizer import Diarizer
from pipeline.manifest import (
    append_jsonl,
    atomic_json_dump,
    iter_glob,
    load_journal_latest,
    read_input_manifest,
    should_skip_by_resume,
)
from pipeline.modes import run_diar_cut_then_asr, run_full_asr_then_align
from pipeline.word_timestamps import GigaAMWordTimestampTranscriber


_WORKER_CFG: Optional[dict] = None
_WORKER_DIARIZER: Optional[Diarizer] = None
_WORKER_ASR: Optional[GigaAMWordTimestampTranscriber] = None


def _batched(items: Sequence, batch_size: int) -> List[list]:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    return [list(items[i : i + batch_size]) for i in range(0, len(items), batch_size)]


@contextlib.contextmanager
def _suppress_output(enabled: bool) -> Iterator[None]:
    if not enabled:
        yield
        return
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


def _progress_postfix(counters: Counter[str], done: int) -> str:
    denom = max(1, done)

    def pct(key: str) -> str:
        return f"{(100.0 * counters.get(key, 0) / denom):.1f}%"

    return (
        f"ok={counters.get('SUCCESS', 0)}({pct('SUCCESS')}) "
        f"fail={counters.get('FAILED', 0)}({pct('FAILED')}) "
        f"bad={counters.get('BAD_SAMPLE', 0)}({pct('BAD_SAMPLE')}) "
        f"skip={counters.get('SKIPPED', 0)}({pct('SKIPPED')})"
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_input_paths(cfg: dict) -> List[dict]:
    audio_dir = Path(cfg["input"]["audio_dir"])
    manifest_path = cfg["input"].get("manifest_path")
    items = []
    if manifest_path:
        for idx, it in enumerate(read_input_manifest(Path(manifest_path))):
            p = Path(it.path)
            abs_path = p if p.is_absolute() else (audio_dir / p)
            items.append({"source_index": idx, "id": it.id, "path": str(abs_path.resolve()), "duration_sec": it.duration_sec})
    else:
        for idx, it in enumerate(iter_glob(audio_dir, cfg["input"].get("glob", "*.mp3"))):
            items.append({"source_index": idx, "id": it.id, "path": str((audio_dir / it.path).resolve()), "duration_sec": it.duration_sec})
    return items


def _validate_unique_output_ids(items: Sequence[dict]) -> None:
    seen: set[str] = set()
    duplicates: List[str] = []
    for item in items:
        item_id = item["id"]
        if item_id in seen:
            duplicates.append(item_id)
        seen.add(item_id)
    if duplicates:
        sample = ", ".join(sorted(set(duplicates))[:8])
        raise ValueError(f"Duplicate input ids would overwrite output files: {sample}")


def _make_models(cfg: dict):
    suppress = bool(cfg["pipeline"].get("suppress_internal_progress", True))
    with _suppress_output(suppress):
        diarizer = Diarizer(
            model_name=cfg["diarizer"].get("model_name", "nvidia/diar_streaming_sortformer_4spk-v2.1"),
            device=cfg["diarizer"].get("device"),
            revision=cfg["diarizer"].get("revision"),
            hf_token=cfg["diarizer"].get("hf_token"),
            local_files_only=bool(cfg["diarizer"].get("local_files_only", False)),
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
            hf_revision=cfg["asr"].get("hf_revision") or cfg["asr"].get("revision"),
            local_files_only=bool(cfg["asr"].get("local_files_only", False)),
            trust_remote_code=bool(cfg["asr"].get("trust_remote_code", True)),
            verify_checksum=cfg["asr"].get("verify_checksum"),
            batch_size=int(cfg["asr"].get("chunk_batch_size", cfg["asr"].get("batch_size", 8))),
        )
    return diarizer, asr


def _mode_fn(cfg: dict):
    return run_full_asr_then_align if cfg["pipeline"]["mode"] == "full_asr_then_align" else run_diar_cut_then_asr


def _failed_result(item: dict, reason: str, *, started_at: Optional[str] = None) -> dict:
    return {
        "item": item,
        "status": "FAILED",
        "reason": reason,
        "started_at": started_at or _utc_now(),
        "finished_at": _utc_now(),
    }


def _run_one_item(
    item: dict,
    cfg: dict,
    diarizer: Diarizer,
    asr: GigaAMWordTimestampTranscriber,
    *,
    diar_out: Optional[tuple] = None,
    words: Optional[list] = None,
    failed_reason: Optional[str] = None,
) -> dict:
    started_at = _utc_now()
    if failed_reason is not None:
        return _failed_result(item, failed_reason, started_at=started_at)

    input_path = item["path"]
    try:
        if diar_out is None:
            with _suppress_output(bool(cfg["pipeline"].get("suppress_internal_progress", True))):
                diar_out = diarizer.diarize([input_path], batch_size=1)[0]

        job_ctx = {"audio_path": input_path, "diar_out": diar_out, "asr": asr, "cfg": cfg}
        if words is not None:
            job_ctx["words"] = words
        result = _mode_fn(cfg)(job_ctx)

        if result["status"] == "BAD_SAMPLE":
            return {
                "item": item,
                "status": "BAD_SAMPLE",
                "reason": result["reason"],
                "shares": result.get("shares"),
                "speakers": result.get("speakers"),
                "started_at": started_at,
                "finished_at": _utc_now(),
            }

        transcript = result["transcript"]
        transcript["file_id"] = item["id"]
        return {
            "item": item,
            "status": "SUCCESS",
            "reason": None,
            "transcript": transcript,
            "meta_asr": result["meta_asr"],
            "meta_diar": result["meta_diar"],
            "started_at": started_at,
            "finished_at": _utc_now(),
        }
    except Exception as exc:  # noqa: BLE001
        return _failed_result(item, repr(exc), started_at=started_at)


def _run_batch_results(
    batch_items: List[dict],
    cfg: dict,
    diarizer: Diarizer,
    asr: GigaAMWordTimestampTranscriber,
    *,
    stage_cb=None,
) -> List[dict]:
    paths = [item["path"] for item in batch_items]
    diar_batch_size = int(cfg["diarizer"].get("batch_size") or len(paths))
    try:
        if stage_cb is not None:
            stage_cb(f"diar {len(batch_items)} files")
        with _suppress_output(bool(cfg["pipeline"].get("suppress_internal_progress", True))):
            diar_outputs = diarizer.diarize(paths, batch_size=diar_batch_size)
        if len(diar_outputs) != len(batch_items):
            raise RuntimeError(f"Diarizer returned {len(diar_outputs)} outputs for {len(batch_items)} inputs")
    except Exception as exc:  # noqa: BLE001
        if len(batch_items) == 1:
            return [_failed_result(batch_items[0], repr(exc))]
        return [_run_one_item(item, cfg, diarizer, asr) for item in batch_items]

    if cfg["pipeline"]["mode"] != "full_asr_then_align":
        out = []
        for idx, item in enumerate(batch_items):
            if stage_cb is not None:
                stage_cb(f"asr {item['id']}")
            out.append(_run_one_item(item, cfg, diarizer, asr, diar_out=diar_outputs[idx]))
        return out

    out: List[dict] = []
    asr_file_batch_size = int(cfg["pipeline"].get("asr_file_batch_size", len(batch_items)))
    indexed = list(enumerate(batch_items))
    for sub_batch in _batched(indexed, asr_file_batch_size):
        sub_indices = [idx for idx, _item in sub_batch]
        sub_items = [item for _idx, item in sub_batch]
        sub_paths = [item["path"] for item in sub_items]
        try:
            if stage_cb is not None:
                stage_cb(f"asr {len(sub_items)} files")
            with _suppress_output(bool(cfg["pipeline"].get("suppress_internal_progress", True))):
                words_outputs = asr.transcribe_words_many(sub_paths)
            if len(words_outputs) != len(sub_items):
                raise RuntimeError(f"ASR returned {len(words_outputs)} outputs for {len(sub_items)} inputs")
        except Exception:  # noqa: BLE001
            for idx, item in zip(sub_indices, sub_items):
                if stage_cb is not None:
                    stage_cb(f"asr {item['id']}")
                out.append(_run_one_item(item, cfg, diarizer, asr, diar_out=diar_outputs[idx]))
            continue

        for local_idx, item in enumerate(sub_items):
            out.append(_run_one_item(item, cfg, diarizer, asr, diar_out=diar_outputs[sub_indices[local_idx]], words=words_outputs[local_idx]))
    return out


def _worker_init(cfg: dict) -> None:
    global _WORKER_CFG, _WORKER_DIARIZER, _WORKER_ASR
    _WORKER_CFG = cfg
    _WORKER_DIARIZER, _WORKER_ASR = _make_models(cfg)


def _worker_process_batch(batch_items: List[dict]) -> List[dict]:
    if _WORKER_CFG is None or _WORKER_DIARIZER is None or _WORKER_ASR is None:
        raise RuntimeError("Worker models were not initialized")
    return _run_batch_results(batch_items, _WORKER_CFG, _WORKER_DIARIZER, _WORKER_ASR)


def _journal_row(result: dict, cfg: dict) -> dict:
    item = result["item"]
    row = {
        "input_path": item["path"],
        "id": item["id"],
        "started_at": result.get("started_at") or _utc_now(),
        "config": cfg,
        "status": result["status"],
        "reason": result.get("reason"),
        "finished_at": result.get("finished_at") or _utc_now(),
    }
    if result["status"] == "BAD_SAMPLE":
        row["shares"] = result.get("shares")
        row["speakers"] = result.get("speakers")
    return row


def _write_result(result: dict, cfg: dict, transcripts_dir: Path, meta_asr_dir: Path, meta_diar_dir: Path, journal_path: Path) -> str:
    item = result["item"]
    status = result["status"]
    if status == "SUCCESS":
        item_id = item["id"]
        atomic_json_dump(transcripts_dir / f"{item_id}.json", result["transcript"])
        atomic_json_dump(meta_asr_dir / f"{item_id}.json", result["meta_asr"])
        atomic_json_dump(meta_diar_dir / f"{item_id}.json", result["meta_diar"])
    append_jsonl(journal_path, _journal_row(result, cfg))
    return status


def _skip_result(item: dict, reason: str) -> dict:
    now = _utc_now()
    return {
        "item": item,
        "status": "SKIPPED",
        "reason": reason,
        "started_at": now,
        "finished_at": now,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Dialog pipeline via YAML config")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    raw_cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    cfg = validate_config(raw_cfg, base_dir=config_path.parent).config

    out_dir = Path(cfg["output"]["out_dir"])
    transcripts_dir = out_dir / cfg["output"].get("transcripts_dir", "transcripts")
    meta_asr_dir = out_dir / cfg["output"].get("meta_asr_dir", "meta_asr")
    meta_diar_dir = out_dir / cfg["output"].get("meta_diar_dir", "meta_diar")
    journal_path = out_dir / "manifest.jsonl"
    for path in (out_dir, transcripts_dir, meta_asr_dir, meta_diar_dir):
        path.mkdir(parents=True, exist_ok=True)

    latest = load_journal_latest(journal_path)
    items = _resolve_input_paths(cfg)
    _validate_unique_output_ids(items)
    counters: Counter[str] = Counter()
    pending_items: List[dict] = []
    show_progress = bool(cfg["pipeline"].get("show_progress", True))
    pbar = tqdm(total=len(items), desc="Dialog pipeline", unit="file", dynamic_ncols=True, disable=not show_progress)

    def set_stage(stage: str) -> None:
        pbar.set_description_str(f"Dialog pipeline | {stage}")
        pbar.refresh()

    def record_result(result: dict) -> None:
        status = _write_result(result, cfg, transcripts_dir, meta_asr_dir, meta_diar_dir, journal_path)
        counters[status] += 1
        pbar.update(1)
        pbar.set_postfix_str(_progress_postfix(counters, int(pbar.n)))

    for item in items:
        input_path = item["path"]
        if bool(cfg["resume"].get("enabled", True)) and should_skip_by_resume(
            input_path,
            latest,
            retry_failed=bool(cfg["resume"].get("retry_failed", False)),
        ):
            record_result(_skip_result(item, "resume"))
            continue

        tr = transcripts_dir / f"{item['id']}.json"
        ma = meta_asr_dir / f"{item['id']}.json"
        md = meta_diar_dir / f"{item['id']}.json"
        if bool(cfg["resume"].get("skip_if_outputs_exist", True)) and tr.exists() and ma.exists() and md.exists():
            record_result(_skip_result(item, "outputs_exist"))
            continue
        pending_items.append(item)

    execution = cfg["pipeline"].get("execution", "sequential")
    if execution in {"sequential", "batched"}:
        diarizer, asr = _make_models(cfg)
        if execution == "sequential":
            for item in pending_items:
                set_stage(f"file {item['id']}")
                record_result(_run_one_item(item, cfg, diarizer, asr))
        else:
            file_batch_size = int(cfg["pipeline"].get("file_batch_size", 1))
            for batch_items in _batched(pending_items, file_batch_size):
                for result in _run_batch_results(batch_items, cfg, diarizer, asr, stage_cb=set_stage):
                    record_result(result)
    elif execution == "multi_instance":
        num_instances = int(cfg["pipeline"].get("num_instances", 1))
        batches_per_instance = int(cfg["pipeline"].get("batches_per_instance", 1))
        file_batch_size = int(cfg["pipeline"].get("file_batch_size", 1))
        max_inflight = max(1, num_instances * batches_per_instance)
        batches = deque(_batched(pending_items, file_batch_size))
        inflight = {}

        set_stage(f"multi {num_instances} instances")
        mp_context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=num_instances, mp_context=mp_context, initializer=_worker_init, initargs=(cfg,)) as pool:
            def submit_next() -> None:
                if not batches:
                    return
                batch = batches.popleft()
                fut = pool.submit(_worker_process_batch, batch)
                inflight[fut] = batch

            while batches and len(inflight) < max_inflight:
                submit_next()

            while inflight:
                done, _pending = wait(inflight.keys(), return_when=FIRST_COMPLETED)
                for fut in done:
                    batch_items = inflight.pop(fut)
                    try:
                        results = fut.result()
                    except Exception as exc:  # noqa: BLE001
                        results = [_failed_result(item, f"worker_batch_failed:{repr(exc)}") for item in batch_items]
                    for result in results:
                        record_result(result)
                    while batches and len(inflight) < max_inflight:
                        submit_next()
    else:
        raise ValueError("pipeline.execution must be sequential, batched, or multi_instance")

    pbar.close()

    has_runtime_failures = counters.get("FAILED", 0) > 0
    summary_status = "done_with_errors" if has_runtime_failures else "done"
    print(
        json.dumps(
            {
                "status": summary_status,
                "journal": str(journal_path),
                "num_items": len(items),
                "counts": dict(sorted(counters.items())),
                "outputs": {
                    "transcripts_dir": str(transcripts_dir),
                    "meta_asr_dir": str(meta_asr_dir),
                    "meta_diar_dir": str(meta_diar_dir),
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 1 if has_runtime_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
