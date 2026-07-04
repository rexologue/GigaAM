import json
from pathlib import Path

from pipeline.dialog_postprocess import decide_speaker_filter
from pipeline.config_schema import validate_config
from pipeline.manifest import load_journal_latest, read_input_manifest, should_skip_by_resume
from pipeline.word_timestamps import GigaAMWordTimestampTranscriber, WordSpan


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


def test_transcribe_words_many_keeps_file_mapping_and_stitches_per_file():
    import torch

    tr = object.__new__(GigaAMWordTimestampTranscriber)
    tr.use_vad = False
    tr.overlap_s = 1.0
    tr.batch_size = 2

    chunks_by_path = {
        "a.wav": (
            [torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([3.0])],
            [(0.0, 1.0), (0.1, 1.1), (2.0, 3.0)],
        ),
        "b.wav": ([torch.tensor([4.0])], [(0.0, 1.0)]),
    }

    def fake_split(audio_path):
        return chunks_by_path[audio_path]

    def fake_infer_groups(segments, bounds):
        groups = []
        for segment, (start, _end) in zip(segments, bounds):
            chunk_id = int(segment[0].item())
            word = "dup" if chunk_id in {1, 2, 4} else "tail"
            groups.append([WordSpan(word, float(start), float(start) + 0.2)])
        return groups

    tr.split_audio_to_chunks = fake_split
    tr._infer_batch_word_groups = fake_infer_groups

    words_by_file = GigaAMWordTimestampTranscriber.transcribe_words_many(tr, ["a.wav", "b.wav"])

    assert [[w.word for w in words] for words in words_by_file] == [["dup", "tail"], ["dup"]]
    assert [words_by_file[0][0].start_s, words_by_file[0][0].end_s] == [0.0, 0.3]
    assert [words_by_file[1][0].start_s, words_by_file[1][0].end_s] == [0.0, 0.2]


def test_config_defaults_and_legacy_asr_batch_size(tmp_path: Path):
    cfg = validate_config(
        {
            "input": {"audio_dir": "."},
            "output": {"out_dir": "out"},
            "asr": {"batch_size": 3},
        },
        base_dir=tmp_path,
    ).config

    assert cfg["pipeline"]["execution"] == "sequential"
    assert cfg["pipeline"]["file_batch_size"] == 1
    assert cfg["pipeline"]["asr_file_batch_size"] == 4
    assert cfg["pipeline"]["show_progress"] is True
    assert cfg["pipeline"]["suppress_internal_progress"] is True
    assert cfg["asr"]["chunk_batch_size"] == 3
    assert "batch_size" not in cfg["asr"]
