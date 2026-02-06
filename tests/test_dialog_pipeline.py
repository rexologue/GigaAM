import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from dataclasses import dataclass

@dataclass
class Segment:
    spk: str
    start: float
    end: float

from dialog_transcribe import (
    WordSpan,
    assign_words_to_speakers,
    build_turns,
    decide_speaker_filter,
    smooth_speaker_islands,
)


def test_speaker_filter_drop_small_third():
    res = decide_speaker_filter(
        shares={"speaker_0": 0.48, "speaker_1": 0.43, "speaker_2": 0.09},
        third_spk_max_share=0.12,
        equal_share_eps=0.08,
        min_dominant_share=0.2,
    )
    assert res.status == "OK"
    assert res.dropped_speakers == ["speaker_2"]
    assert len(res.keep_speakers) == 2


def test_speaker_filter_bad_equal_three():
    res = decide_speaker_filter(
        shares={"speaker_0": 0.34, "speaker_1": 0.33, "speaker_2": 0.30},
        third_spk_max_share=0.12,
        equal_share_eps=0.08,
        min_dominant_share=0.2,
    )
    assert res.status == "BAD_SAMPLE"
    assert "equal_shares" in str(res.reason)


def test_word_mapping_and_snap():
    words = [
        WordSpan("a", 0.0, 0.2),
        WordSpan("b", 0.22, 0.4),
        WordSpan("c", 1.01, 1.2),
    ]
    segs = [
        Segment("speaker_0", 0.0, 0.5),
        Segment("speaker_1", 1.3, 2.0),
    ]
    spk, reassigned = assign_words_to_speakers(words, segs, ["speaker_0", "speaker_1"], max_snap_sec=0.35)
    assert spk[:2] == ["speaker_0", "speaker_0"]
    assert spk[2] == "speaker_1"
    assert reassigned == 1


def test_turn_build_and_island_smoothing():
    words = [
        WordSpan("привет", 0.0, 0.2),
        WordSpan("как", 0.21, 0.4),
        WordSpan("дела", 0.41, 0.6),
        WordSpan("да", 0.61, 0.7),
        WordSpan("норм", 0.71, 0.9),
    ]
    speakers = ["Spk0", "Spk1", "Spk0", "Spk0", "Spk0"]
    changed = smooth_speaker_islands(words, speakers, island_max_words=1, island_max_sec=0.3)
    assert changed == 1
    assert speakers == ["Spk0", "Spk0", "Spk0", "Spk0", "Spk0"]

    turns = build_turns(words, speakers, pause_new_turn_sec=0.5, store_words=False)
    assert len(turns) == 1
    assert turns[0]["speaker"] == "Spk0"
    assert "привет" in turns[0]["text"]
