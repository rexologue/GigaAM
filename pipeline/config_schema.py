from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass
class ConfigValidationResult:
    config: Dict[str, Any]


def _require(cfg: Dict[str, Any], key: str, section: str) -> Any:
    if key not in cfg:
        raise ValueError(f"Missing required key '{section}.{key}'")
    return cfg[key]


def validate_config(cfg: Dict[str, Any]) -> ConfigValidationResult:
    for section in ("input", "output", "resume", "pipeline", "diarizer", "asr", "speaker_rules"):
        if section not in cfg:
            raise ValueError(f"Missing required section '{section}'")

    input_cfg = cfg["input"]
    output_cfg = cfg["output"]
    pipeline_cfg = cfg["pipeline"]

    _require(input_cfg, "audio_dir", "input")
    if input_cfg.get("manifest_path") is None and input_cfg.get("glob") is None:
        raise ValueError("Either input.manifest_path or input.glob must be provided")

    _require(output_cfg, "out_dir", "output")
    mode = _require(pipeline_cfg, "mode", "pipeline")
    if mode not in {"full_asr_then_align", "diar_cut_then_asr"}:
        raise ValueError("pipeline.mode must be full_asr_then_align or diar_cut_then_asr")

    if mode == "diar_cut_then_asr" and "segment_max_sec" not in cfg["asr"]:
        raise ValueError("asr.segment_max_sec is required for diar_cut_then_asr mode")

    cfg["input"]["audio_dir"] = str(Path(cfg["input"]["audio_dir"]).expanduser())
    cfg["output"]["out_dir"] = str(Path(cfg["output"]["out_dir"]).expanduser())
    return ConfigValidationResult(config=cfg)
