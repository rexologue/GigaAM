from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class ConfigValidationResult:
    config: Dict[str, Any]


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

    input_cfg["audio_dir"] = _resolve_path(input_cfg["audio_dir"], base_dir=base_dir)
    input_cfg["manifest_path"] = _resolve_path(input_cfg.get("manifest_path"), base_dir=base_dir)
    output_cfg["out_dir"] = _resolve_path(output_cfg["out_dir"], base_dir=base_dir)

    cfg["diarizer"]["model_name"] = _resolve_model_ref(cfg["diarizer"].get("model_name"), base_dir=base_dir)
    cfg["asr"]["model_name"] = _resolve_model_ref(cfg["asr"].get("model_name"), base_dir=base_dir)
    return ConfigValidationResult(config=cfg)
