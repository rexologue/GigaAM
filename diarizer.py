from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple, Optional

import torch
import torchaudio
from nemo.collections.asr.models import SortformerEncLabelModel


_SEG_RE = re.compile(r"^\s*(?P<start>\d+(?:\.\d+)?)\s+(?P<end>\d+(?:\.\d+)?)\s+(?P<spk>\S+)\s*$")


@dataclass(frozen=True, slots=True)
class Segment:
    spk: str
    start: float
    end: float

    @property
    def dur(self) -> float:
        return max(0.0, self.end - self.start)


def _merge_intervals(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Merge overlapping/touching intervals."""
    if not intervals:
        return []
    intervals.sort(key=lambda x: (x[0], x[1]))
    out = [intervals[0]]
    for s, e in intervals[1:]:
        ps, pe = out[-1]
        if s <= pe:  # overlap/touch
            out[-1] = (ps, max(pe, e))
        else:
            out.append((s, e))
    return out


def _sum_intervals(intervals: Iterable[Tuple[float, float]]) -> float:
    return sum(max(0.0, e - s) for s, e in intervals)


class Diarizer:
    """
    Optimized wrapper:
    - GPU if available
    - inference_mode + autocast (CUDA)
    - fast audio normalization via sox_effects (mono + 16k) and only then (if needed) write temp wav
    - returns:
        1) segments: List[Segment]
        2) speakers: List[str]
        3) share per speaker (fraction of total audio duration)
    """

    def __init__(
        self,
        model_name: str = "nvidia/diar_streaming_sortformer_4spk-v2.1",
        device: Optional[str] = None,
        chunk_len: int = 340,
        chunk_right_context: int = 40,
        fifo_len: int = 40,
        spkcache_update_period: int = 300,
        use_amp: bool = True,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.use_amp = bool(use_amp) and (self.device.type == "cuda")

        self.diar_model = SortformerEncLabelModel.from_pretrained(model_name)
        self.diar_model.eval()
        self.diar_model.to(self.device)

        # Streaming params
        sm = self.diar_model.sortformer_modules
        sm.chunk_len = chunk_len
        sm.chunk_right_context = chunk_right_context
        sm.fifo_len = fifo_len
        sm.spkcache_update_period = spkcache_update_period

        # Minor speed knobs
        torch.set_grad_enabled(False)
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

    @staticmethod
    def _ensure_wav16k_mono_to_tmp(src_path: str, tmpdir: str, idx: int) -> Tuple[str, float]:
        """
        Convert any input to WAV 16k mono using sox effects (fast, uses libsox).
        Returns (dst_wav_path, duration_sec).
        """
        # sox effects: downmix to 1ch + resample to 16k
        effects = [
            ["remix", "1"],     # mono (take channel 1; for stereo this is usually enough and faster than mean)
            ["rate", "16000"],
        ]

        wav, sr = torchaudio.sox_effects.apply_effects_file(src_path, effects=effects)
        # wav: (1, T)
        dst = os.path.join(tmpdir, f"{idx}.wav")
        torchaudio.save(dst, wav, 16000, encoding="PCM_S", bits_per_sample=16)

        # duration from samples (post-conversion)
        duration_sec = float(wav.shape[1]) / 16000.0
        return dst, duration_sec

    @staticmethod
    def _parse_segments(raw_lines: Iterable[str]) -> List[Segment]:
        segs: List[Segment] = []
        for line in raw_lines:
            m = _SEG_RE.match(line)
            if not m:
                continue
            s = float(m.group("start"))
            e = float(m.group("end"))
            spk = m.group("spk")
            if e > s:
                segs.append(Segment(spk=spk, start=s, end=e))
        segs.sort(key=lambda x: (x.start, x.end, x.spk))
        return segs

    @staticmethod
    def _compute_shares(
        segments: List[Segment],
        audio_dur_sec: float,
    ) -> Dict[str, float]:
        """
        Share per speaker relative to full audio duration.
        Note: if diarization outputs overlaps, shares can sum to > 1.0.
        (This is usually fine; if you want "non-overlap" accounting, see comment below.)
        """
        if audio_dur_sec <= 0:
            return {s.spk: 0.0 for s in segments}

        per_spk: Dict[str, float] = {}
        for seg in segments:
            per_spk[seg.spk] = per_spk.get(seg.spk, 0.0) + seg.dur

        return {spk: min(1.0, dur / audio_dur_sec) for spk, dur in per_spk.items()}

    def diarize(
        self,
        paths_batch: List[str],
        batch_size: Optional[int] = None,
    ) -> List[Tuple[List[Segment], List[str], Dict[str, float]]]:
        """
        For each input file returns:
          (segments, speakers, share_by_speaker)

        segments: List[Segment] with (spk, start, end)
        speakers: sorted unique speakers
        shares: fraction of total audio duration (0..1)
        """
        if not paths_batch:
            return []

        batch_size = batch_size or len(paths_batch)

        with tempfile.TemporaryDirectory() as tmpdir:
            wav_paths: List[str] = []
            durations: List[float] = []

            # Convert in a tight loop (sox is typically faster than python resample/mean)
            for i, src in enumerate(paths_batch):
                dst, dur = self._ensure_wav16k_mono_to_tmp(src, tmpdir, i)
                wav_paths.append(dst)
                durations.append(dur)

            # Run diarization
            with torch.inference_mode():
                autocast_ctx = (
                    torch.autocast(device_type="cuda", dtype=torch.float16)
                    if self.use_amp else
                    torch.autocast(device_type="cpu", enabled=False)
                )
                with autocast_ctx:
                    predicted = self.diar_model.diarize(audio=wav_paths, batch_size=batch_size)

            # predicted: list per file; each element is list[str] like "7.680 8.400 speaker_0"
            out: List[Tuple[List[Segment], List[str], Dict[str, float]]] = []
            for file_idx, raw_lines in enumerate(predicted):
                segments = self._parse_segments(raw_lines)
                speakers = sorted({s.spk for s in segments})
                shares = self._compute_shares(segments, durations[file_idx])
                out.append((segments, speakers, shares))

            return out


if __name__ == "__main__":
    d = Diarizer()
    res = d.diarize(["/root/audio/3844580337_0f49b1df58159bd58ea5f08fd4bbbe10_79036263700.mp3"])

    segments, speakers, shares = res[0]
    print("Speakers:", speakers)
    print("Shares:", {k: round(v * 100, 2) for k, v in shares.items()}, "%")
    print("First 10 segments:")
    for seg in segments[:10]:
        print(seg)
