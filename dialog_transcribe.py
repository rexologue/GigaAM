#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

from word_timestamps import WordSpan

if TYPE_CHECKING:
    from diarizer import Segment


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json_dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=str(path.parent)) as tmp:
        json.dump(payload, tmp, ensure_ascii=False, indent=2)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = tmp.name
    os.replace(tmp_path, path)


def append_manifest_line(manifest_path: Path, row: dict) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_manifest_latest(manifest_path: Path) -> Dict[str, dict]:
    latest: Dict[str, dict] = {}
    if not manifest_path.exists():
        return latest
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            latest[row["input_path"]] = row
    return latest


@dataclass(frozen=True)
class FileJob:
    input_path: Path
    stem: str
    size: int
    mtime: float


@dataclass
class SpeakerFilterResult:
    keep_speakers: List[str]
    dropped_speakers: List[str]
    status: str  # OK | BAD_SAMPLE
    reason: Optional[str]


def decide_speaker_filter(
    shares: Dict[str, float],
    third_spk_max_share: float,
    equal_share_eps: float,
    min_dominant_share: float,
) -> SpeakerFilterResult:
    speakers = sorted(shares)
    n = len(speakers)
    if n == 0:
        return SpeakerFilterResult([], [], "BAD_SAMPLE", "no_speakers")
    if n == 1:
        spk = speakers[0]
        if shares[spk] < min_dominant_share:
            return SpeakerFilterResult([], [], "BAD_SAMPLE", "single_speaker_low_share")
        return SpeakerFilterResult([spk], [], "OK", None)
    if n == 2:
        mx = max(shares.values())
        if mx < min_dominant_share:
            return SpeakerFilterResult([], [], "BAD_SAMPLE", "two_speakers_low_dominance")
        return SpeakerFilterResult(speakers, [], "OK", None)
    if n > 3:
        return SpeakerFilterResult([], [], "BAD_SAMPLE", f"too_many_speakers:{n}")

    ranked = sorted(shares.items(), key=lambda x: x[1], reverse=True)
    max_share = ranked[0][1]
    min_share = ranked[-1][1]
    if max_share - min_share < equal_share_eps:
        return SpeakerFilterResult([], [], "BAD_SAMPLE", "three_speakers_equal_shares")

    third_spk, third_share = ranked[-1]
    if third_share <= third_spk_max_share:
        keep = sorted([ranked[0][0], ranked[1][0]])
        return SpeakerFilterResult(keep, [third_spk], "OK", None)

    return SpeakerFilterResult([], [], "BAD_SAMPLE", "three_speakers_no_small_third")


def _seg_intersection(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def assign_words_to_speakers(
    words: Sequence[WordSpan],
    segments: Sequence[Segment],
    allowed_speakers: Sequence[str],
    max_snap_sec: float,
) -> Tuple[List[Optional[str]], int]:
    allowed = set(allowed_speakers)
    segs = [s for s in segments if s.spk in allowed]
    segs.sort(key=lambda s: (s.start, s.end))

    assigned: List[Optional[str]] = []
    reassigned = 0
    j = 0

    for w in words:
        w0, w1 = float(w.start_s), float(w.end_s)
        while j < len(segs) and segs[j].end < w0:
            j += 1

        k = j
        overlap_scores: Dict[str, float] = {}
        while k < len(segs) and segs[k].start <= w1:
            ov = _seg_intersection(w0, w1, segs[k].start, segs[k].end)
            if ov > 0:
                overlap_scores[segs[k].spk] = overlap_scores.get(segs[k].spk, 0.0) + ov
            k += 1

        if overlap_scores:
            spk = max(overlap_scores.items(), key=lambda x: (x[1], x[0]))[0]
            assigned.append(spk)
            continue

        best_spk = None
        best_gap = math.inf
        for cand in (j - 1, j):
            if 0 <= cand < len(segs):
                s = segs[cand]
                gap = min(abs(w0 - s.end), abs(w1 - s.start), abs((w0 + w1) / 2 - (s.start + s.end) / 2))
                if gap < best_gap:
                    best_gap = gap
                    best_spk = s.spk
        if best_spk is not None and best_gap <= max_snap_sec:
            assigned.append(best_spk)
            reassigned += 1
        else:
            assigned.append(None)

    return assigned, reassigned


def smooth_speaker_islands(
    words: Sequence[WordSpan],
    speakers: List[Optional[str]],
    island_max_words: int,
    island_max_sec: float,
) -> int:
    changes = 0
    i = 0
    while i < len(speakers):
        cur = speakers[i]
        j = i + 1
        while j < len(speakers) and speakers[j] == cur:
            j += 1

        block_len = j - i
        block_dur = max(0.0, float(words[j - 1].end_s - words[i].start_s)) if words else 0.0
        left = speakers[i - 1] if i > 0 else None
        right = speakers[j] if j < len(speakers) else None

        if (
            cur is not None
            and left is not None
            and right is not None
            and left == right
            and cur != left
            and block_len <= island_max_words
            and block_dur <= island_max_sec
        ):
            for k in range(i, j):
                speakers[k] = left
                changes += 1
        i = j
    return changes


def build_turns(
    words: Sequence[WordSpan],
    speakers: Sequence[Optional[str]],
    pause_new_turn_sec: float,
    store_words: bool,
) -> List[dict]:
    turns: List[dict] = []
    cur: Optional[dict] = None

    def flush() -> None:
        nonlocal cur
        if cur is None:
            return
        cur["text"] = " ".join(cur.pop("_text_parts")).strip()
        turns.append(cur)
        cur = None

    for idx, (w, spk) in enumerate(zip(words, speakers)):
        if spk is None:
            continue
        gap = 0.0
        if idx > 0:
            gap = max(0.0, float(w.start_s - words[idx - 1].end_s))

        starts_new = (
            cur is None
            or spk != cur["speaker"]
            or gap > pause_new_turn_sec
        )
        if starts_new:
            flush()
            cur = {
                "speaker": spk,
                "start": round(float(w.start_s), 3),
                "end": round(float(w.end_s), 3),
                "_text_parts": [w.word],
            }
            if store_words:
                cur["words"] = [{"w": w.word, "s": round(float(w.start_s), 3), "e": round(float(w.end_s), 3), "spk": spk}]
        else:
            cur["end"] = round(float(w.end_s), 3)
            cur["_text_parts"].append(w.word)
            if store_words:
                cur["words"].append({"w": w.word, "s": round(float(w.start_s), 3), "e": round(float(w.end_s), 3), "spk": spk})

    flush()
    return turns


def collect_mp3_jobs(in_dir: Path, limit: Optional[int]) -> List[FileJob]:
    files = sorted(in_dir.glob("*.mp3"))
    if limit is not None:
        files = files[:limit]
    jobs: List[FileJob] = []
    for p in files:
        st = p.stat()
        jobs.append(FileJob(input_path=p.resolve(), stem=p.stem, size=st.st_size, mtime=st.st_mtime))
    return jobs


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch dialog transcription pipeline")
    parser.add_argument("--in-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)

    parser.add_argument("--diar-model-name", default="nvidia/diar_streaming_sortformer_4spk-v2.1")
    parser.add_argument("--diar-chunk-len", type=int, default=340)
    parser.add_argument("--diar-right-context", type=int, default=40)
    parser.add_argument("--diar-fifo-len", type=int, default=40)
    parser.add_argument("--diar-spkcache-update-period", type=int, default=300)
    parser.add_argument("--diar-batch-size", type=int, default=8)

    parser.add_argument("--asr-model-name", default="v3_ctc")
    parser.add_argument("--asr-use-vad", action="store_true")
    parser.add_argument("--asr-max-chunk-s", type=float, default=22.0)
    parser.add_argument("--asr-overlap-s", type=float, default=1.0)
    parser.add_argument("--asr-batch-size", type=int, default=8)
    parser.add_argument("--hf-token", default=None)

    parser.add_argument("--third-spk-max-share", type=float, default=0.12)
    parser.add_argument("--equal-share-eps", type=float, default=0.08)
    parser.add_argument("--min-dominant-share", type=float, default=0.2)
    parser.add_argument("--max-snap-sec", type=float, default=0.8)
    parser.add_argument("--pause-new-turn-sec", type=float, default=0.6)
    parser.add_argument("--island-max-words", type=int, default=2)
    parser.add_argument("--island-max-sec", type=float, default=1.0)
    parser.add_argument("--store-words-in-transcript", action="store_true")
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    transcripts_dir = out_dir / "transcripts"
    meta_asr_dir = out_dir / "meta_asr"
    meta_diar_dir = out_dir / "meta_diar"
    manifest_path = out_dir / "manifest.jsonl"

    jobs = collect_mp3_jobs(in_dir, args.limit)
    latest = load_manifest_latest(manifest_path)

    todo: List[FileJob] = []
    skipped = 0
    for job in jobs:
        tr = transcripts_dir / f"{job.stem}.json"
        ma = meta_asr_dir / f"{job.stem}.json"
        md = meta_diar_dir / f"{job.stem}.json"
        prev = latest.get(str(job.input_path))
        outputs_exist = tr.exists() and ma.exists() and md.exists()

        if prev and prev.get("status") in {"SUCCESS", "BAD_SAMPLE"} and prev.get("file_size") == job.size and abs(prev.get("mtime", 0.0) - job.mtime) < 1e-6:
            skipped += 1
            continue
        if prev and prev.get("status") in {"FAILED", "IN_PROGRESS"} and (not args.retry_failed):
            skipped += 1
            continue
        if outputs_exist and not args.retry_failed:
            skipped += 1
            continue
        todo.append(job)

    if args.dry_run:
        print(json.dumps({
            "total_found": len(jobs),
            "to_process": len(todo),
            "skipped": skipped,
            "sample_to_process": [str(j.input_path) for j in todo[:5]],
        }, ensure_ascii=False, indent=2))
        return 0

    from diarizer import Diarizer
    from word_timestamps import GigaAMWordTimestampTranscriber

    diarizer = Diarizer(
        model_name=args.diar_model_name,
        device=args.device,
        chunk_len=args.diar_chunk_len,
        chunk_right_context=args.diar_right_context,
        fifo_len=args.diar_fifo_len,
        spkcache_update_period=args.diar_spkcache_update_period,
    )
    asr = GigaAMWordTimestampTranscriber(
        model_name=args.asr_model_name,
        device=args.device,
        use_vad=args.asr_use_vad,
        max_chunk_s=args.asr_max_chunk_s,
        overlap_s=args.asr_overlap_s,
        hf_token=args.hf_token,
        batch_size=args.asr_batch_size,
    )

    print(f"Found {len(jobs)} files, processing {len(todo)}, skipped {skipped}")

    for offset in range(0, len(todo), max(1, args.diar_batch_size)):
        batch = todo[offset : offset + max(1, args.diar_batch_size)]
        batch_paths = [str(j.input_path) for j in batch]

        started_batch = time.perf_counter()
        diar_out = diarizer.diarize(batch_paths, batch_size=min(args.diar_batch_size, len(batch)))

        def run_asr(job: FileJob) -> List[WordSpan]:
            return asr.transcribe_words(str(job.input_path))

        asr_results: Dict[str, List[WordSpan]] = {}
        workers = max(1, int(args.num_workers))
        if workers == 1:
            for j in batch:
                asr_results[str(j.input_path)] = run_asr(j)
        else:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                fut_to_path = {ex.submit(run_asr, j): str(j.input_path) for j in batch}
                for fut in as_completed(fut_to_path):
                    asr_results[fut_to_path[fut]] = fut.result()

        for j, (segments, speakers, shares) in zip(batch, diar_out):
            start_t = time.perf_counter()
            input_path = str(j.input_path)
            manifest_base = {
                "input_path": input_path,
                "file_size": j.size,
                "mtime": j.mtime,
                "started_at": _utc_now(),
                "params": {
                    "device": args.device,
                    "diar_model_name": args.diar_model_name,
                    "asr_model_name": args.asr_model_name,
                    "asr_use_vad": args.asr_use_vad,
                    "diar_chunk_len": args.diar_chunk_len,
                    "diar_right_context": args.diar_right_context,
                    "diar_fifo_len": args.diar_fifo_len,
                    "diar_spkcache_update_period": args.diar_spkcache_update_period,
                    "asr_max_chunk_s": args.asr_max_chunk_s,
                    "asr_overlap_s": args.asr_overlap_s,
                },
            }

            try:
                filt = decide_speaker_filter(
                    shares,
                    third_spk_max_share=args.third_spk_max_share,
                    equal_share_eps=args.equal_share_eps,
                    min_dominant_share=args.min_dominant_share,
                )
                if filt.status == "BAD_SAMPLE":
                    append_manifest_line(manifest_path, {
                        **manifest_base,
                        "status": "BAD_SAMPLE",
                        "reason": filt.reason,
                        "shares": shares,
                        "finished_at": _utc_now(),
                        "processing_time_sec": round(time.perf_counter() - start_t, 3),
                    })
                    continue

                words = asr_results[input_path]
                raw_speakers, reassigned_words = assign_words_to_speakers(words, segments, filt.keep_speakers, args.max_snap_sec)
                island_reassigned = smooth_speaker_islands(words, raw_speakers, args.island_max_words, args.island_max_sec)

                ordered_kept = sorted(filt.keep_speakers, key=lambda s: shares.get(s, 0.0), reverse=True)
                norm_map = {orig: f"Spk{i}" for i, orig in enumerate(ordered_kept)}
                norm_speakers = [norm_map.get(spk) if spk is not None else None for spk in raw_speakers]

                turns = build_turns(words, norm_speakers, args.pause_new_turn_sec, args.store_words_in_transcript)

                word_durs = [max(0.0, float(w.end_s - w.start_s)) for w in words]
                total_covered = sum(word_durs)
                unk_words = sum(1 for s in norm_speakers if s is None)
                speaker_counts = Counter([s for s in norm_speakers if s is not None])

                transcript_payload = {
                    "audio_path": input_path,
                    "file_id": j.stem,
                    "status": "SUCCESS",
                    "speakers": [norm_map[s] for s in ordered_kept],
                    "mapping": {
                        "orig_to_norm": norm_map,
                        "dropped_speakers": filt.dropped_speakers,
                    },
                    "turns": turns,
                    "stats": {
                        "num_turns": len(turns),
                        "num_words": len(words),
                        "unk_words": unk_words,
                        "reassigned_words": reassigned_words + island_reassigned,
                        "processing_time_sec": round(time.perf_counter() - start_t, 3),
                    },
                    "params": {
                        "pause_new_turn_sec": args.pause_new_turn_sec,
                        "max_snap_sec": args.max_snap_sec,
                        "island_max_words": args.island_max_words,
                        "island_max_sec": args.island_max_sec,
                    },
                }

                meta_asr_payload = {
                    "audio_path": input_path,
                    "model_name": args.asr_model_name,
                    "device": args.device,
                    "sample_rate": 16000,
                    "chunking": {
                        "use_vad": args.asr_use_vad,
                        "max_chunk_s": args.asr_max_chunk_s,
                        "overlap_s": args.asr_overlap_s,
                        "batch_size": args.asr_batch_size,
                    },
                    "words": [{"word": w.word, "start": round(float(w.start_s), 3), "end": round(float(w.end_s), 3)} for w in words],
                    "stats": {
                        "num_words": len(words),
                        "total_covered_sec": round(total_covered, 3),
                        "mean_word_dur": round(mean(word_durs), 4) if word_durs else 0.0,
                        "max_word_dur": round(max(word_durs), 4) if word_durs else 0.0,
                    },
                }

                meta_diar_payload = {
                    "audio_path": input_path,
                    "model_name": args.diar_model_name,
                    "device": args.device,
                    "streaming_params": {
                        "chunk_len": args.diar_chunk_len,
                        "right_context": args.diar_right_context,
                        "fifo_len": args.diar_fifo_len,
                        "spkcache_update_period": args.diar_spkcache_update_period,
                        "batch_size": args.diar_batch_size,
                    },
                    "segments": [{"spk": s.spk, "start": round(float(s.start), 3), "end": round(float(s.end), 3)} for s in segments],
                    "shares": shares,
                    "speaker_filter": {
                        "kept": filt.keep_speakers,
                        "dropped": filt.dropped_speakers,
                    },
                    "stats": {
                        "num_segments": len(segments),
                        "speakers_count": len(speakers),
                        "assigned_word_counts": dict(speaker_counts),
                    },
                }

                atomic_json_dump(transcripts_dir / f"{j.stem}.json", transcript_payload)
                atomic_json_dump(meta_asr_dir / f"{j.stem}.json", meta_asr_payload)
                atomic_json_dump(meta_diar_dir / f"{j.stem}.json", meta_diar_payload)

                append_manifest_line(manifest_path, {
                    **manifest_base,
                    "status": "SUCCESS",
                    "reason": None,
                    "finished_at": _utc_now(),
                    "processing_time_sec": round(time.perf_counter() - start_t, 3),
                })

            except Exception as exc:
                append_manifest_line(manifest_path, {
                    **manifest_base,
                    "status": "FAILED",
                    "reason": repr(exc),
                    "finished_at": _utc_now(),
                    "processing_time_sec": round(time.perf_counter() - start_t, 3),
                })

        print(
            f"Processed batch {offset // max(1, args.diar_batch_size) + 1} / {math.ceil(max(1, len(todo)) / max(1, args.diar_batch_size))} "
            f"in {time.perf_counter() - started_batch:.2f}s"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
