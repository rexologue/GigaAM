from __future__ import annotations

import time
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Sequence, Tuple

from diarizer import Segment
from dialog_transcribe import (
    assign_words_to_speakers,
    build_turns,
    decide_speaker_filter,
    smooth_speaker_islands,
)
from gigaam.preprocess import SAMPLE_RATE, load_audio
from word_timestamps import WordSpan


def _normalize_speaker_map(kept: Sequence[str], shares: Dict[str, float]) -> Dict[str, str]:
    ordered_kept = sorted(kept, key=lambda s: shares.get(s, 0.0), reverse=True)
    return {orig: f"Spk{i}" for i, orig in enumerate(ordered_kept)}


def _reassign_dropped_segments(segments: Sequence[Segment], kept_speakers: Sequence[str], dropped: Sequence[str]) -> List[Segment]:
    dropped_set = set(dropped)
    keep_set = set(kept_speakers)
    kept_segments = [s for s in segments if s.spk in keep_set]
    if not kept_segments:
        return [s for s in segments if s.spk in keep_set]

    out: List[Segment] = []
    for s in segments:
        if s.spk not in dropped_set:
            if s.spk in keep_set:
                out.append(s)
            continue

        mid = (s.start + s.end) / 2.0
        best = min(kept_segments, key=lambda k: abs(((k.start + k.end) / 2.0) - mid))
        out.append(Segment(spk=best.spk, start=s.start, end=s.end))
    out.sort(key=lambda x: (x.start, x.end, x.spk))
    return out


def _build_common_meta(
    *,
    input_path: str,
    words: List[WordSpan],
    turns: List[dict],
    speaker_map: Dict[str, str],
    dropped_speakers: Sequence[str],
    processing_time: float,
    reassigned_words: int,
) -> Tuple[dict, dict]:
    word_durs = [max(0.0, float(w.end_s - w.start_s)) for w in words]
    meta_stats = {
        "num_words": len(words),
        "total_covered_sec": round(sum(word_durs), 3),
        "mean_word_dur": round(mean(word_durs), 4) if word_durs else 0.0,
        "max_word_dur": round(max(word_durs), 4) if word_durs else 0.0,
    }
    transcript_payload = {
        "audio_path": input_path,
        "status": "SUCCESS",
        "speakers": list(speaker_map.values()),
        "mapping": {"orig_to_norm": speaker_map, "dropped_speakers": list(dropped_speakers)},
        "turns": turns,
        "stats": {
            "num_turns": len(turns),
            "num_words": len(words),
            "reassigned_words": reassigned_words,
            "processing_time_sec": round(processing_time, 3),
        },
    }
    return transcript_payload, meta_stats


def run_full_asr_then_align(job_ctx: dict) -> dict:
    t0 = time.perf_counter()
    segments, speakers, shares = job_ctx["diar_out"]
    cfg = job_ctx["cfg"]
    words = job_ctx["asr"].transcribe_words(job_ctx["audio_path"])

    filt = decide_speaker_filter(
        shares=shares,
        third_spk_max_share=float(cfg["speaker_rules"]["third_spk_max_share"]),
        equal_share_eps=float(cfg["speaker_rules"]["equal_share_eps"]),
        min_dominant_share=0.0,
    )
    if len(speakers) > int(cfg["speaker_rules"].get("max_speakers_allowed", 3)):
        filt.status = "BAD_SAMPLE"
        filt.reason = f"too_many_speakers:{len(speakers)}"

    if filt.status != "OK":
        return {
            "status": "BAD_SAMPLE",
            "reason": filt.reason,
            "shares": shares,
            "speakers": speakers,
            "processing_time_sec": time.perf_counter() - t0,
        }

    remap_segments = _reassign_dropped_segments(segments, filt.keep_speakers, filt.dropped_speakers)
    raw_speakers, reassigned_words = assign_words_to_speakers(
        words,
        remap_segments,
        filt.keep_speakers,
        float(cfg["speaker_rules"].get("max_snap_sec", 0.8)),
    )
    reassigned_words += smooth_speaker_islands(
        words,
        raw_speakers,
        int(cfg["speaker_rules"].get("island_max_words", 2)),
        float(cfg["speaker_rules"].get("island_max_sec", 1.0)),
    )

    speaker_map = _normalize_speaker_map(filt.keep_speakers, shares)
    norm_speakers = [speaker_map.get(s) if s is not None else None for s in raw_speakers]
    turns = build_turns(words, norm_speakers, float(cfg["speaker_rules"].get("pause_new_turn_sec", 0.6)), False)

    transcript_payload, common_stats = _build_common_meta(
        input_path=job_ctx["audio_path"],
        words=words,
        turns=turns,
        speaker_map=speaker_map,
        dropped_speakers=filt.dropped_speakers,
        processing_time=time.perf_counter() - t0,
        reassigned_words=reassigned_words,
    )

    meta_asr = {
        "audio_path": job_ctx["audio_path"],
        "mode": "full_asr_then_align",
        "words": [{"word": w.word, "start": round(float(w.start_s), 3), "end": round(float(w.end_s), 3)} for w in words],
        "stats": common_stats,
    }
    meta_diar = {
        "audio_path": job_ctx["audio_path"],
        "speakers": speakers,
        "shares": shares,
        "postprocessed_segments": [{"spk": s.spk, "start": round(float(s.start), 3), "end": round(float(s.end), 3)} for s in remap_segments],
        "dropped_speakers": list(filt.dropped_speakers),
        "drop_reason": "third_spk_max_share" if filt.dropped_speakers else None,
    }
    return {"status": "SUCCESS", "transcript": transcript_payload, "meta_asr": meta_asr, "meta_diar": meta_diar, "processing_time_sec": time.perf_counter() - t0}


def run_diar_cut_then_asr(job_ctx: dict) -> dict:
    t0 = time.perf_counter()
    segments, speakers, shares = job_ctx["diar_out"]
    cfg = job_ctx["cfg"]
    filt = decide_speaker_filter(
        shares=shares,
        third_spk_max_share=float(cfg["speaker_rules"]["third_spk_max_share"]),
        equal_share_eps=float(cfg["speaker_rules"]["equal_share_eps"]),
        min_dominant_share=0.0,
    )
    if len(speakers) > int(cfg["speaker_rules"].get("max_speakers_allowed", 3)):
        filt.status = "BAD_SAMPLE"
        filt.reason = f"too_many_speakers:{len(speakers)}"

    if filt.status != "OK":
        return {
            "status": "BAD_SAMPLE",
            "reason": filt.reason,
            "shares": shares,
            "speakers": speakers,
            "processing_time_sec": time.perf_counter() - t0,
        }

    remap_segments = _reassign_dropped_segments(segments, filt.keep_speakers, filt.dropped_speakers)
    waveform = load_audio(job_ctx["audio_path"], SAMPLE_RATE)
    bounds = [(float(s.start), float(s.end)) for s in remap_segments]
    words, word_segment_ids = job_ctx["asr"].transcribe_words_from_segments(
        waveform,
        bounds,
        segment_max_sec=float(cfg["asr"]["segment_max_sec"]),
    )

    segment_speakers = [s.spk for s in remap_segments]
    speaker_map = _normalize_speaker_map(filt.keep_speakers, shares)
    norm_word_speakers = [speaker_map.get(segment_speakers[idx]) for idx in word_segment_ids]
    turns = build_turns(words, norm_word_speakers, float(cfg["speaker_rules"].get("pause_new_turn_sec", 0.6)), False)

    transcript_payload, common_stats = _build_common_meta(
        input_path=job_ctx["audio_path"],
        words=words,
        turns=turns,
        speaker_map=speaker_map,
        dropped_speakers=filt.dropped_speakers,
        processing_time=time.perf_counter() - t0,
        reassigned_words=0,
    )

    seg_word_counts = Counter(word_segment_ids)
    segment_items = []
    for idx, seg in enumerate(remap_segments):
        dur = max(0.0, seg.end - seg.start)
        num_chunks = 1 if dur <= float(cfg["asr"]["segment_max_sec"]) else max(1, int(dur // float(cfg["asr"]["segment_max_sec"])) + 1)
        segment_items.append(
            {
                "segment_id": idx,
                "spk": seg.spk,
                "start": round(float(seg.start), 3),
                "end": round(float(seg.end), 3),
                "dur": round(dur, 3),
                "num_chunks_after_split": num_chunks,
                "processing_time": None,
                "num_words": int(seg_word_counts.get(idx, 0)),
            }
        )

    meta_asr = {
        "audio_path": job_ctx["audio_path"],
        "mode": "diar_cut_then_asr",
        "asr_batch_items": segment_items,
        "words": [
            {
                "word": w.word,
                "start": round(float(w.start_s), 3),
                "end": round(float(w.end_s), 3),
                "segment_id": int(seg_id),
                "speaker_id": speaker_map.get(segment_speakers[seg_id]),
            }
            for w, seg_id in zip(words, word_segment_ids)
        ],
        "stats": common_stats,
    }
    meta_diar = {
        "audio_path": job_ctx["audio_path"],
        "speakers": speakers,
        "shares": shares,
        "postprocessed_segments": [{"spk": s.spk, "start": round(float(s.start), 3), "end": round(float(s.end), 3)} for s in remap_segments],
        "dropped_speakers": list(filt.dropped_speakers),
        "drop_reason": "third_spk_max_share_reassigned" if filt.dropped_speakers else None,
    }
    return {"status": "SUCCESS", "transcript": transcript_payload, "meta_asr": meta_asr, "meta_diar": meta_diar, "processing_time_sec": time.perf_counter() - t0}
