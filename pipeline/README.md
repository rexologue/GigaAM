# Dialog transcription pipeline

This directory contains the whole dialog transcription pipeline. The legacy root-level `dialog_transcribe.py` entrypoint was removed; run the YAML-based pipeline from this package.

## Layout

- `run_pipeline.py` — main CLI entrypoint.
- `config.yaml` — local editable config.
- `config.example.yaml` — documented config template.
- `config_schema.py` — validation and path/model-reference resolution.
- `diarizer.py` — NeMo Sortformer diarization wrapper.
- `word_timestamps.py` — GigaAM CTC word timestamp transcriber.
- `modes.py` — pipeline modes.
- `dialog_postprocess.py` — speaker filtering, word-to-speaker alignment, turn building.
- `manifest.py` — input manifest and resumable output journal helpers.
- `tools/make_manifest_by_duration.py` — helper for building JSONL input manifests.
- `tools/collect_speaker_dialogs.py` — helper for converting transcript JSON files into compact speaker-dialog rows.

## Run

From the repository root:

```bash
python -m pipeline.run_pipeline --config pipeline/config.yaml
```

Direct script execution also works:

```bash
python pipeline/run_pipeline.py --config pipeline/config.yaml
```

All relative paths in the YAML config are resolved relative to the YAML file location, not relative to the shell working directory.

## Input

The pipeline accepts either a directory glob or a JSONL manifest.

Directory mode:

```yaml
input:
  audio_dir: ../data
  manifest_path: null
  glob: "*.mp3"
```

Manifest mode:

```yaml
input:
  audio_dir: ../data
  manifest_path: ./manifest.jsonl
  glob: null
```

Each manifest row may look like this:

```json
{"id":"sample_001","path":"subdir/file.mp3","duration_sec":12.34}
```

`path` may be absolute or relative to `input.audio_dir`.

## Output

For each accepted file the pipeline writes:

- `out_dir/transcripts/<id>.json` — normalized dialog turns, speaker mapping, basic stats.
- `out_dir/meta_asr/<id>.json` — ASR word timestamps and ASR-side metadata.
- `out_dir/meta_diar/<id>.json` — diarization segments, speaker shares, filtering metadata.
- `out_dir/manifest.jsonl` — resumable processing journal with `SUCCESS`, `FAILED`, `SKIPPED`, `BAD_SAMPLE` statuses.

## Pipeline modes

`full_asr_then_align`:

1. Run diarization over the whole file.
2. Run ASR over the whole file.
3. Assign ASR words to diarization segments.
4. Build speaker turns.

This is usually the better default when word timestamps are stable enough.

`diar_cut_then_asr`:

1. Run diarization over the whole file.
2. Cut audio logically by diarization segments.
3. Run ASR on each diarized segment.
4. Build turns directly from segment-linked words.

This can be useful when full-file ASR alignment is unstable, but it is usually slower.

## Pipeline execution

`pipeline.execution` controls how many audio files are processed together:

```yaml
pipeline:
  mode: full_asr_then_align
  execution: sequential
  file_batch_size: 4
  show_progress: true
  suppress_internal_progress: true
```

- `sequential` keeps the old behavior: one audio file is diarized and transcribed at a time.
- `batched` groups up to `file_batch_size` files for Sortformer diarization.
- In `full_asr_then_align`, `batched` also groups GigaAM CTC chunks across files. The result mapping is positional, so output `i` always belongs to input file `i`.
- In `diar_cut_then_asr`, `batched` currently batches the diarization stage; ASR remains per file because it depends on diarization segments.

ASR chunk batching is controlled separately:

```yaml
asr:
  chunk_batch_size: 8
  max_sec: 22.0
  overlap_sec: 1.0
```

`overlap_sec` keeps the same quality behavior in both execution modes. Chunks may overlap, and duplicate words are stitched independently inside each file after inference. The batched path does not stitch words across files.

When `show_progress` is enabled, the CLI shows one pipeline-level progress bar with throughput, ETA, and a status postfix. `suppress_internal_progress` redirects noisy model-level stdout/stderr during inference so library progress bars, including NeMo's per-sample progress, do not clutter the output. The final JSON summary is still printed to stdout.

## Model references

Both ASR and diarization model fields support local paths and Hugging Face identifiers.

ASR examples:

```yaml
asr:
  model_name: v3_ctc
```

```yaml
asr:
  model_name: ../models/gigaam/v3_ctc.ckpt
  verify_checksum: false
```

```yaml
asr:
  model_name: ai-sage/GigaAM-v3
  hf_revision: ctc
  local_files_only: false
  trust_remote_code: true
```

Diarizer examples:

```yaml
diarizer:
  model_name: nvidia/diar_streaming_sortformer_4spk-v2.1
```

```yaml
diarizer:
  model_name: ../models/diarizer/model.nemo
```

```yaml
diarizer:
  model_name: ../models/nvidia/diar_streaming_sortformer_4spk-v2.1
  local_files_only: true
```

A model value is treated as a local path only when it is absolute, starts with `.` or `~`, or already exists relative to the config directory. Values like `org/model` stay Hugging Face repo ids.

## Resume behavior

Resume is controlled by:

```yaml
resume:
  enabled: true
  retry_failed: false
  skip_if_outputs_exist: true
```

When enabled, already successful, skipped, bad, or failed files are not reprocessed unless `retry_failed` is set to `true` for failed rows. If all expected output files already exist, the file can also be skipped via `skip_if_outputs_exist`.

If a previous run produced only `FAILED` rows, either set `retry_failed: true` or remove `out_dir/manifest.jsonl` before rerunning. Otherwise resume will intentionally skip those failed rows.

The final CLI summary includes per-status counts and returns a non-zero exit code when at least one row failed with a runtime exception. `BAD_SAMPLE` is a data-quality decision, not a runtime failure.

## Bad sample policy

Speaker filtering is controlled by `speaker_rules`. The default behavior is conservative:

- too many speakers -> `BAD_SAMPLE`;
- three near-equal speakers -> `BAD_SAMPLE`;
- small third speaker -> dropped and reassigned to the nearest kept speaker where possible.

## Helper tools

Build a duration-sorted input manifest:

```bash
python -m pipeline.tools.make_manifest_by_duration --audio-dir ../data --out pipeline/manifest.jsonl --glob "*.mp3"
```

Collect transcript JSON files into compact speaker-dialog rows:

```bash
python -m pipeline.tools.collect_speaker_dialogs ../output --transcripts-dir transcripts --out-dir dialogs
```

## Dependencies

Core pipeline dependencies are listed in the repository-level `requirements.txt` and `requirements_extra.txt`. In practice, the pipeline requires the base GigaAM dependencies, NeMo, torchaudio, OmegaConf, and `ffmpeg` available in `PATH`. The pipeline no longer requires `libsox.so` for diarization audio normalization; it uses the same ffmpeg-backed loader as the ASR path. Hugging Face Hub access is optional and only needed when `revision`, `hf_token`, or `local_files_only` are used for remote diarizer snapshots.
