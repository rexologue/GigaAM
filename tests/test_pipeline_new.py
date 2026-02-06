import json
from pathlib import Path

from dialog_transcribe import decide_speaker_filter
from pipeline.manifest import load_journal_latest, read_input_manifest, should_skip_by_resume
from word_timestamps import GigaAMWordTimestampTranscriber, WordSpan


def test_manifest_read_order_jsonl(tmp_path: Path):
    m = tmp_path / "in.jsonl"
    rows = [
        {"path": "b.mp3", "id": "b"},
        {"path": "a.mp3", "id": "a"},
        {"path": "c.mp3", "id": "c"},
    ]
    with open(m, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    items = read_input_manifest(m)
    assert [x.path for x in items] == ["b.mp3", "a.mp3", "c.mp3"]


def test_resume_skip_statuses(tmp_path: Path):
    journal = tmp_path / "manifest.jsonl"
    with open(journal, "w", encoding="utf-8") as f:
        f.write(json.dumps({"input_path": "x.mp3", "status": "SUCCESS"}) + "\n")
        f.write(json.dumps({"input_path": "y.mp3", "status": "BAD_SAMPLE"}) + "\n")
        f.write(json.dumps({"input_path": "z.mp3", "status": "FAILED"}) + "\n")

    latest = load_journal_latest(journal)
    assert should_skip_by_resume("x.mp3", latest, retry_failed=False)
    assert should_skip_by_resume("y.mp3", latest, retry_failed=False)
    assert should_skip_by_resume("z.mp3", latest, retry_failed=False)
    assert not should_skip_by_resume("z.mp3", latest, retry_failed=True)


def test_bad_sample_equal_three_shares():
    res = decide_speaker_filter(
        shares={"speaker_0": 0.33, "speaker_1": 0.34, "speaker_2": 0.35},
        third_spk_max_share=0.12,
        equal_share_eps=0.08,
        min_dominant_share=0.0,
    )
    assert res.status == "BAD_SAMPLE"


def test_diar_cut_offsets_words():
    tr = object.__new__(GigaAMWordTimestampTranscriber)
    tr.max_chunk_s = 22.0
    tr.overlap_s = 1.0

    def fake_from_waveform(audio, offset_s=0.0, max_chunk_s=None, overlap_s=None):
        return [WordSpan("ok", float(offset_s), float(offset_s) + 0.5)]

    tr.transcribe_words_from_waveform = fake_from_waveform
    import torch

    waveform = torch.zeros(16000 * 10)
    words, seg_ids = GigaAMWordTimestampTranscriber.transcribe_words_from_segments(
        tr,
        waveform,
        [(1.0, 2.0), (4.0, 5.0)],
        segment_max_sec=3.0,
    )

    assert [w.start_s for w in words] == [1.0, 4.0]
    assert seg_ids == [0, 1]
