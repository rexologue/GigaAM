from __future__ import annotations

import os
import re
import tempfile
import wave
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple, Optional

import torch

from gigaam.preprocess import SAMPLE_RATE, load_audio


def _get_sortformer_model_cls():
    try:
        from nemo.collections.asr.models import SortformerEncLabelModel
    except ImportError as exc:  # pragma: no cover - dependency-dependent path
        raise RuntimeError(
            "Diarization requires NVIDIA NeMo. Install NeMo or use an environment "
            "that already provides nemo.collections.asr.models.SortformerEncLabelModel."
        ) from exc

    return SortformerEncLabelModel

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
    - audio normalization via the same ffmpeg-backed loader used by GigaAM ASR
    - returns:
        1) segments: List[Segment]
        2) speakers: List[str]
        3) share per speaker (fraction of total audio duration)
    """

    def __init__(
        self,
        model_name: str = "nvidia/diar_streaming_sortformer_4spk-v2.1",
        device: Optional[str] = None,
        revision: Optional[str] = None,
        hf_token: Optional[str] = None,
        local_files_only: bool = False,
        chunk_len: int = 340,
        chunk_right_context: int = 40,
        fifo_len: int = 40,
        spkcache_update_period: int = 300,
        use_amp: bool = True,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.use_amp = bool(use_amp) and (self.device.type == "cuda")

        self.diar_model = self._load_model(
            model_name=model_name,
            revision=revision,
            hf_token=hf_token,
            local_files_only=local_files_only,
        )
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
    def _looks_like_local_path(model_name: str) -> bool:
        path = Path(model_name).expanduser()
        return model_name.startswith((".", "~")) or path.is_absolute() or path.exists()

    @staticmethod
    def _restore_nemo_model(path: Path):
        try:
            return _get_sortformer_model_cls().restore_from(restore_path=str(path))
        except TypeError:
            return _get_sortformer_model_cls().restore_from(str(path))

    @classmethod
    def _load_local_model(cls, model_name: str):
        path = Path(model_name).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Local diarizer model path does not exist: {path}")

        if path.is_file():
            if path.suffix == ".nemo":
                return cls._restore_nemo_model(path)
            raise ValueError(f"Local diarizer model file must be a .nemo archive, got: {path}")

        nemo_files = sorted(path.glob("*.nemo"))
        if len(nemo_files) == 1:
            return cls._restore_nemo_model(nemo_files[0])
        if len(nemo_files) > 1:
            names = ", ".join(p.name for p in nemo_files[:8])
            raise ValueError(f"Multiple .nemo files found under {path}: {names}. Pass the exact .nemo path instead.")

        return _get_sortformer_model_cls().from_pretrained(str(path))

    @classmethod
    def _load_model(
        cls,
        *,
        model_name: str,
        revision: Optional[str],
        hf_token: Optional[str],
        local_files_only: bool,
    ):
        if cls._looks_like_local_path(model_name):
            return cls._load_local_model(model_name)

        if revision is None and hf_token is None and not local_files_only:
            return _get_sortformer_model_cls().from_pretrained(model_name)

        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:  # pragma: no cover - dependency-dependent path
            raise RuntimeError(
                "revision/hf_token/local_files_only for diarizer models requires huggingface_hub. "
                "Install it or pass a local model path."
            ) from exc

        local_path = snapshot_download(
            repo_id=model_name,
            revision=revision,
            token=hf_token,
            local_files_only=local_files_only,
        )
        return cls._load_local_model(local_path)

    @staticmethod
    def _write_wav16(path: str, audio: torch.Tensor, sample_rate: int) -> None:
        audio = audio.detach().cpu().flatten().clamp(-1.0, 1.0)
        pcm = (audio * 32767.0).round().to(torch.int16).numpy().tobytes()

        with wave.open(path, "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(sample_rate)
            f.writeframes(pcm)

    @classmethod
    def _ensure_wav16k_mono_to_tmp(cls, src_path: str, tmpdir: str, idx: int) -> Tuple[str, float]:
        """
        Convert any input to a temporary WAV 16 kHz mono file.

        This intentionally uses the same ffmpeg-backed loader as the ASR path.
        It avoids torchaudio.sox_effects, because many production images have
        ffmpeg but do not ship libsox.so.

        Returns (dst_wav_path, duration_sec).
        """
        audio = load_audio(src_path, SAMPLE_RATE)
        dst = os.path.join(tmpdir, f"{idx}.wav")
        cls._write_wav16(dst, audio, SAMPLE_RATE)

        duration_sec = float(audio.numel()) / float(SAMPLE_RATE)
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

            # Convert in a tight loop before sending normalized wavs to NeMo.
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
