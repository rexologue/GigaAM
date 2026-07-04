from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

if TYPE_CHECKING:
    from pipeline.diarizer import Segment


@dataclass
class WordSpan:
    word: str
    start_s: float
    end_s: float


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

        starts_new = cur is None or spk != cur["speaker"] or gap > pause_new_turn_sec
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
