#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import torch

from gigaam import load_model
from gigaam.preprocess import SAMPLE_RATE, load_audio

# Optional: VAD chunking (requires pyannote + HF_TOKEN for pyannote/segmentation-3.0)
try:
    from gigaam.vad_utils import segment_audio_file  # type: ignore

    _HAS_VAD = True
except Exception:
    _HAS_VAD = False


@dataclass
class TokenSpan:
    token_id: int
    start_t: int  # frame index (inclusive)
    end_t: int    # frame index (exclusive)
    start_s: float
    end_s: float
    text: str


@dataclass
class WordSpan:
    word: str
    start_s: float
    end_s: float


class GigaAMWordTimestampTranscriber:
    """
    GigaAM CTC -> word timestamps for arbitrary-length audio.

    Constructor args mirror former argparse flags:
      - model_name        (--model)
      - device            (--device)
      - use_vad           (--vad)
      - max_chunk_s       (--max-chunk-s)
      - overlap_s         (--overlap-s)
      - hf_token          (optional; used for pyannote VAD if needed)
      - batch_size        (new; chunk batching)
    """

    def __init__(
        self,
        model_name: str = "v3_ctc",
        device: Optional[str] = None,
        use_vad: bool = False,
        max_chunk_s: float = 22.0,
        overlap_s: float = 1.0,
        hf_token: Optional[str] = None,
        batch_size: int = 8,
    ) -> None:
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_vad = use_vad
        self.max_chunk_s = float(max_chunk_s)
        self.overlap_s = float(overlap_s)
        self.hf_token = hf_token
        self.batch_size = int(batch_size)

        if self.use_vad and not _HAS_VAD:
            raise RuntimeError("VAD chunking requested but gigaam.vad_utils is unavailable.")

        if self.hf_token:
            # pyannote pipelines typically read HF_TOKEN
            os.environ.setdefault("HF_TOKEN", self.hf_token)

        self.model = load_model(self.model_name, device=self.device)
        if not self._is_ctc_head(self.model):
            raise RuntimeError("This class expects a GigaAM ASR model with a CTC head.")

        self.tokenizer, self.blank_id = self._get_vocab_and_blank(self.model)

    # -----------------------------
    # Public API
    # -----------------------------

    def split_audio_to_chunks(self, audio_path: str) -> Tuple[List[torch.Tensor], List[Tuple[float, float]]]:
        """
        Returns:
          segments: list[Tensor] each shape [Nsamples]
          bounds:   list[(start_s, end_s)] per segment in original timeline
        """
        if self.use_vad:
            # segment_audio_file reads audio itself
            segments, bounds = segment_audio_file(audio_path, SAMPLE_RATE, device=torch.device(self.device))
            return segments, bounds

        audio = load_audio(audio_path, SAMPLE_RATE)
        return self._chunk_audio_fixed(audio, SAMPLE_RATE, self.max_chunk_s, self.overlap_s)

    def transcribe_words(self, audio_path: str) -> List[WordSpan]:
        """
        Chunk audio, run batched inference, return a flat list of WordSpan (global timestamps).
        """
        segments, bounds = self.split_audio_to_chunks(audio_path)

        all_words: List[WordSpan] = []
        for i in range(0, len(segments), self.batch_size):
            seg_batch = segments[i : i + self.batch_size]
            bnd_batch = bounds[i : i + self.batch_size]
            batch_words = self._infer_batch_words(seg_batch, bnd_batch)
            all_words.extend(batch_words)

        if (not self.use_vad) and self.overlap_s > 0:
            all_words = self._stitch_overlapped_words(all_words)

        return all_words

    def transcribe_to_json(self, audio_path: str) -> dict:
        """
        Convenience: same as old script output (but words only; transcript is optional here).
        """
        words = self.transcribe_words(audio_path)
        return {
            "audio_path": audio_path,
            "model": self.model_name,
            "sample_rate": SAMPLE_RATE,
            "words": [{"word": w.word, "start": round(w.start_s, 3), "end": round(w.end_s, 3)} for w in words],
        }

    # -----------------------------
    # Batched inference
    # -----------------------------

    def _infer_batch_words(
        self,
        segments: List[torch.Tensor],
        bounds: List[Tuple[float, float]],
    ) -> List[WordSpan]:
        """
        Run one forward pass for a batch of variable-length chunks.
        Returns flat list of WordSpan with global timestamps.
        """
        if not segments:
            return []

        wav_batch, lengths = self._pad_batch(segments)  # wav_batch: [B, Tmax]
        wav_batch = wav_batch.to(self.model._device).to(self.model._dtype)
        lengths = lengths.to(self.model._device)

        # GigaAM models typically accept [B, T] or [B, 1, T]. Your original code used [1, T].
        # We try [B, T] first; if it fails, fallback to [B, 1, T].
        encoded = None
        enc_len = None
        try:
            encoded, enc_len = self.model.forward(wav_batch, lengths)
        except Exception:
            encoded, enc_len = self.model.forward(wav_batch.unsqueeze(1), lengths)

        log_probs = self.model.head(encoded)  # expected [B, T, C]
        if log_probs.dim() != 3:
            raise RuntimeError(f"Unexpected head output shape: {tuple(log_probs.shape)} (need [B,T,C])")

        B, T, _C = log_probs.shape
        enc_len_list = enc_len.detach().tolist() if enc_len is not None else [T] * B

        out_words: List[WordSpan] = []
        for b in range(B):
            b0, b1 = bounds[b]
            chunk_dur_s = float(b1 - b0)
            effective_T = int(enc_len_list[b]) if b < len(enc_len_list) else T
            effective_T = max(1, min(T, effective_T))

            token_text_fn: Callable[[int], str] = lambda tid: self._token_id_to_text(self.tokenizer, tid)
            token_spans = self._ctc_token_spans_from_logprobs(
                log_probs[b], self.blank_id, token_text_fn, chunk_start_s=b0, chunk_dur_s=chunk_dur_s, effective_T=effective_T
            )
            seg_words = self._words_from_token_spans(self.tokenizer, token_spans)
            out_words.extend(seg_words)

        return out_words

    @staticmethod
    def _pad_batch(segments: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        segments: list of [Nsamples]
        Returns:
          wav_batch: [B, Tmax] float32
          lengths:   [B] int64 (original lengths in samples)
        """
        lengths = torch.tensor([int(s.numel()) for s in segments], dtype=torch.long)
        max_len = int(lengths.max().item()) if lengths.numel() else 0

        wav_batch = torch.zeros((len(segments), max_len), dtype=segments[0].dtype)
        for i, s in enumerate(segments):
            n = int(s.numel())
            if n:
                wav_batch[i, :n] = s
        return wav_batch, lengths

    # -----------------------------
    # Helpers (ported from script)
    # -----------------------------

    @staticmethod
    def _is_ctc_head(model) -> bool:
        return hasattr(model, "head") and callable(model.head)

    @staticmethod
    def _get_vocab_and_blank(model):
        if not hasattr(model, "decoding") or not hasattr(model.decoding, "tokenizer"):
            raise RuntimeError("Model decoding/tokenizer not found in model.cfg")
        tok = model.decoding.tokenizer
        blank_id = getattr(model.decoding, "blank_id", None)
        if blank_id is None:
            blank_id = len(tok)
        return tok, int(blank_id)

    @staticmethod
    def _token_id_to_text(tokenizer, token_id: int) -> str:
        if getattr(tokenizer, "charwise", False):
            return tokenizer.vocab[token_id]
        sp = getattr(tokenizer, "model", None)
        if sp is None:
            return tokenizer.decode([token_id])
        return sp.id_to_piece(token_id)

    @staticmethod
    def _chunk_audio_fixed(
        audio: torch.Tensor,
        sr: int,
        max_chunk_s: float,
        overlap_s: float,
    ) -> Tuple[List[torch.Tensor], List[Tuple[float, float]]]:
        assert max_chunk_s > 0
        assert 0 <= overlap_s < max_chunk_s

        total_s = audio.numel() / sr
        step_s = max_chunk_s - overlap_s

        segments: List[torch.Tensor] = []
        bounds: List[Tuple[float, float]] = []

        t0 = 0.0
        while t0 < total_s:
            t1 = min(total_s, t0 + max_chunk_s)
            s0 = int(round(t0 * sr))
            s1 = int(round(t1 * sr))
            segments.append(audio[s0:s1])
            bounds.append((t0, t1))
            if t1 >= total_s:
                break
            t0 += step_s

        return segments, bounds

    @staticmethod
    def _ctc_token_spans_from_logprobs(
        log_probs_1: torch.Tensor,   # [T, C]
        blank_id: int,
        token_text_fn: Callable[[int], str],
        chunk_start_s: float,
        chunk_dur_s: float,
        effective_T: int,
    ) -> List[TokenSpan]:
        labels = log_probs_1.argmax(dim=-1).tolist()
        T_all = len(labels)
        T = max(1, min(T_all, int(effective_T)))

        spans: List[Tuple[int, int, int]] = []  # (token_id, start_t, end_t)
        prev: Optional[int] = None
        cur_id: Optional[int] = None
        cur_start = 0

        for t in range(T):
            lab = labels[t]
            if lab == blank_id:
                if cur_id is not None:
                    spans.append((cur_id, cur_start, t))
                    cur_id = None
                prev = lab
                continue

            if cur_id is None:
                cur_id = lab
                cur_start = t
            else:
                if prev is not None and lab != prev:
                    spans.append((cur_id, cur_start, t))
                    cur_id = lab
                    cur_start = t

            prev = lab

        if cur_id is not None:
            spans.append((cur_id, cur_start, T))

        out: List[TokenSpan] = []
        denom = max(1, T)
        for tid, st, en in spans:
            st_s = chunk_start_s + (st / denom) * chunk_dur_s
            en_s = chunk_start_s + (en / denom) * chunk_dur_s
            out.append(
                TokenSpan(
                    token_id=tid,
                    start_t=st,
                    end_t=en,
                    start_s=st_s,
                    end_s=en_s,
                    text=token_text_fn(tid),
                )
            )
        return out

    @staticmethod
    def _words_from_token_spans(tokenizer, token_spans: List[TokenSpan]) -> List[WordSpan]:
        if not token_spans:
            return []

        # Charwise grouping
        if getattr(tokenizer, "charwise", False):
            words: List[WordSpan] = []
            cur_chars: List[str] = []
            cur_start: Optional[float] = None
            cur_end: Optional[float] = None

            def flush():
                nonlocal cur_chars, cur_start, cur_end
                if cur_chars and cur_start is not None and cur_end is not None:
                    w = "".join(cur_chars)
                    if w.strip():
                        words.append(WordSpan(word=w, start_s=cur_start, end_s=cur_end))
                cur_chars = []
                cur_start = None
                cur_end = None

            for ts in token_spans:
                ch = ts.text
                is_sep = (ch.isspace() or ch == "|")
                if is_sep:
                    flush()
                    continue
                if cur_start is None:
                    cur_start = ts.start_s
                cur_end = ts.end_s
                cur_chars.append(ch)

            flush()
            return words

        # SentencePiece grouping: boundary on pieces starting with "▁"
        sp = getattr(tokenizer, "model", None)
        if sp is None:
            return [WordSpan(word=ts.text, start_s=ts.start_s, end_s=ts.end_s) for ts in token_spans]

        words: List[WordSpan] = []
        cur_pieces: List[str] = []
        cur_start: Optional[float] = None
        cur_end: Optional[float] = None

        def flush():
            nonlocal cur_pieces, cur_start, cur_end
            if cur_pieces and cur_start is not None and cur_end is not None:
                text = "".join(cur_pieces).replace("▁", " ").strip()
                if text:
                    words.append(WordSpan(word=text, start_s=cur_start, end_s=cur_end))
            cur_pieces = []
            cur_start = None
            cur_end = None

        for ts in token_spans:
            piece = ts.text
            starts_word = piece.startswith("▁")
            if starts_word and cur_pieces:
                flush()
            if cur_start is None:
                cur_start = ts.start_s
            cur_end = ts.end_s
            cur_pieces.append(piece)

        flush()
        return words

    @staticmethod
    def _stitch_overlapped_words(words: List[WordSpan], max_gap_s: float = 0.25) -> List[WordSpan]:
        if not words:
            return []
        out: List[WordSpan] = [words[0]]
        for w in words[1:]:
            prev = out[-1]
            if w.word == prev.word and abs(w.start_s - prev.start_s) < max_gap_s:
                prev.end_s = max(prev.end_s, w.end_s)
                continue
            if w.word == prev.word and abs(w.end_s - prev.end_s) < 1e-3:
                continue
            out.append(w)
        return out


# -----------------------------
# Example usage (no argparse)
# -----------------------------
if __name__ == "__main__":
    tr = GigaAMWordTimestampTranscriber(
        model_name="v3_e2e_ctc",
        device="cuda",          # auto
        use_vad=True,
        max_chunk_s=22.0,
        overlap_s=1.0,
        hf_token="...",
        batch_size=8,
    )

    audio_path = "/root/audio/3844580337_0f49b1df58159bd58ea5f08fd4bbbe10_79036263700.mp3"
    words: List[WordSpan] = tr.transcribe_words(audio_path)
    print(json.dumps(
        [{"word": w.word, "start": round(w.start_s, 3), "end": round(w.end_s, 3)} for w in words],
        ensure_ascii=False,
        indent=2,
    ))
