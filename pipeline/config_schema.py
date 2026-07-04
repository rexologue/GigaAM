from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class ConfigValidationResult:
    config: Dict[str, Any]


DEFAULT_CONFIG: Dict[str, Any] = {
    "input": {
        "manifest_path": None,
        "glob": "*.mp3",
    },
    "output": {
        "transcripts_dir": "transcripts",
        "meta_asr_dir": "meta_asr",
        "meta_diar_dir": "meta_diar",
    },
    "resume": {
        "enabled": True,
        "retry_failed": False,
        "skip_if_outputs_exist": True,
    },
    "pipeline": {
        "mode": "full_asr_then_align",
        "execution": "sequential",
        "num_instances": 1,
        "batches_per_instance": 1,
        "file_batch_size": 1,
        "asr_file_batch_size": 4,
        "show_progress": True,
        "suppress_internal_progress": True,
    },
    "diarizer": {
        "model_name": "nvidia/diar_streaming_sortformer_4spk-v2.1",
        "revision": None,
        "hf_token": None,
        "local_files_only": False,
        "device": None,
        "batch_size": None,
        "chunk_len": 340,
        "chunk_right_context": 40,
        "fifo_len": 40,
        "spkcache_update_period": 300,
    },
    "asr": {
        "model_name": "v3_ctc",
        "hf_revision": None,
        "local_files_only": False,
        "trust_remote_code": True,
        "verify_checksum": None,
        "device": None,
        "use_vad": False,
        "max_sec": 22.0,
        "overlap_sec": 1.0,
        "chunk_batch_size": 8,
        "hf_token": None,
        "segment_max_sec": 22.0,
    },
    "speaker_rules": {
        "third_spk_max_share": 0.10,
        "equal_share_eps": 0.05,
        "max_speakers_allowed": 3,
        "max_snap_sec": 0.8,
        "island_max_words": 2,
        "island_max_sec": 1.0,
        "pause_new_turn_sec": 0.6,
    },
}


def _merge_defaults(cfg: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(defaults)
    for key, value in cfg.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_defaults(value, merged[key])
        else:
            merged[key] = value
    return merged


def _require(cfg: Dict[str, Any], key: str, section: str) -> Any:
    if key not in cfg:
        raise ValueError(f"Missing required key '{section}.{key}'")
    return cfg[key]


def _resolve_path(value: Any, *, base_dir: Optional[Path], must_be_path: bool = True) -> Any:
    if value is None or not isinstance(value, str) or not value.strip():
        return value

    path = Path(value).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path

    if must_be_path:
        return str(path)
    return str(path.resolve()) if path.exists() else value


def _looks_like_local_path(value: str) -> bool:
    path = Path(value).expanduser()
    return value.startswith((".", "~")) or path.is_absolute() or path.exists()


def _resolve_model_ref(value: Any, *, base_dir: Optional[Path]) -> Any:
    """
    Resolve only explicit local model references.

    HF repo ids also contain '/', so we must not turn every slash-containing value into
    a filesystem path. A value is treated as local only when it is absolute, starts
    with '.', starts with '~', or already exists relative to cwd/base_dir.
    """
    if value is None or not isinstance(value, str) or not value.strip():
        return value

    expanded = Path(value).expanduser()
    if expanded.exists() or _looks_like_local_path(value):
        if expanded.is_absolute():
            return str(expanded)
        if base_dir is not None:
            candidate = (base_dir / expanded).expanduser()
            return str(candidate)
        return str(expanded)

    if base_dir is not None:
        candidate = (base_dir / value).expanduser()
        if candidate.exists():
            return str(candidate)

    return value


def validate_config(cfg: Dict[str, Any], *, base_dir: Optional[Path] = None) -> ConfigValidationResult:
    raw_asr_cfg = (cfg or {}).get("asr", {}) if isinstance((cfg or {}).get("asr", {}), dict) else {}
    cfg = _merge_defaults(cfg or {}, DEFAULT_CONFIG)

    input_cfg = cfg["input"]
    output_cfg = cfg["output"]
    pipeline_cfg = cfg["pipeline"]
    diarizer_cfg = cfg["diarizer"]
    asr_cfg = cfg["asr"]

    _require(input_cfg, "audio_dir", "input")
    if input_cfg.get("manifest_path") is None and input_cfg.get("glob") is None:
        raise ValueError("Either input.manifest_path or input.glob must be provided")

    _require(output_cfg, "out_dir", "output")
    mode = _require(pipeline_cfg, "mode", "pipeline")
    if mode not in {"full_asr_then_align", "diar_cut_then_asr"}:
        raise ValueError("pipeline.mode must be full_asr_then_align or diar_cut_then_asr")
    execution = _require(pipeline_cfg, "execution", "pipeline")
    if execution not in {"sequential", "batched", "multi_instance"}:
        raise ValueError("pipeline.execution must be sequential, batched, or multi_instance")
    if int(pipeline_cfg.get("num_instances", 1)) <= 0:
        raise ValueError("pipeline.num_instances must be > 0")
    if int(pipeline_cfg.get("batches_per_instance", 1)) <= 0:
        raise ValueError("pipeline.batches_per_instance must be > 0")
    if int(pipeline_cfg.get("file_batch_size", 1)) <= 0:
        raise ValueError("pipeline.file_batch_size must be > 0")
    if int(pipeline_cfg.get("asr_file_batch_size", 4)) <= 0:
        raise ValueError("pipeline.asr_file_batch_size must be > 0")

    if "batch_size" in raw_asr_cfg and "chunk_batch_size" not in raw_asr_cfg:
        asr_cfg["chunk_batch_size"] = asr_cfg["batch_size"]
    asr_cfg.pop("batch_size", None)
    if int(asr_cfg.get("chunk_batch_size", 8)) <= 0:
        raise ValueError("asr.chunk_batch_size must be > 0")
    diar_batch_size = diarizer_cfg.get("batch_size")
    if diar_batch_size is not None and int(diar_batch_size) <= 0:
        raise ValueError("diarizer.batch_size must be > 0 when set")

    if mode == "diar_cut_then_asr" and "segment_max_sec" not in cfg["asr"]:
        raise ValueError("asr.segment_max_sec is required for diar_cut_then_asr mode")

    input_cfg["audio_dir"] = _resolve_path(input_cfg["audio_dir"], base_dir=base_dir)
    input_cfg["manifest_path"] = _resolve_path(input_cfg.get("manifest_path"), base_dir=base_dir)
    output_cfg["out_dir"] = _resolve_path(output_cfg["out_dir"], base_dir=base_dir)

    cfg["diarizer"]["model_name"] = _resolve_model_ref(cfg["diarizer"].get("model_name"), base_dir=base_dir)
    cfg["asr"]["model_name"] = _resolve_model_ref(cfg["asr"].get("model_name"), base_dir=base_dir)
    return ConfigValidationResult(config=cfg)
