from __future__ import annotations

import hashlib
import logging
import os
import urllib.request
import warnings
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
from tqdm import tqdm

__all__ = [
    "GigaAM",
    "GigaAMASR",
    "GigaAMEmo",
    "load_audio",
    "format_time",
    "load_model",
]


def __getattr__(name: str):
    if name in {"GigaAM", "GigaAMASR", "GigaAMEmo"}:
        from .model import GigaAM, GigaAMASR, GigaAMEmo

        return {"GigaAM": GigaAM, "GigaAMASR": GigaAMASR, "GigaAMEmo": GigaAMEmo}[name]
    if name == "load_audio":
        from .preprocess import load_audio

        return load_audio
    if name == "format_time":
        from .utils import format_time

        return format_time
    raise AttributeError(name)

# Default cache directory
_CACHE_DIR = os.path.expanduser("~/.cache/gigaam")
# Url with model checkpoints
_URL_DIR = "https://cdn.chatwm.opensmodel.sberdevices.ru/GigaAM"
_MODEL_HASHES = {
    "emo": "7ce76f9535cb254488985057c0d33006",
    "v1_ctc": "f027f199e590a391d015aeede2e66174",
    "v1_rnnt": "02c758999bcdc6afcb2087ef256d47ef",
    "v1_ssl": "dc7f7b231f7f91c4968dc21910e7b396",
    "v2_ctc": "e00f59cb5d39624fb30d1786044795bf",
    "v2_rnnt": "547460139acfebd842323f59ed54ab54",
    "v2_ssl": "cd4cf819c8191a07b9d7edcad111668e",
    "v3_ctc": "73413e7be9c6a5935827bfab5c0dd678",
    "v3_rnnt": "0fd2c9a1ff66abd8d32a3a07f7592815",
    "v3_e2e_ctc": "367074d6498f426d960b25f49531cf68",
    "v3_e2e_rnnt": "2730de7545ac43ad256485a462b0a27a",
    "v3_ssl": "70cbf5ed7303a0ed242ddb257e9dc6a6",
}
_SHORT_NAMES = ["ctc", "rnnt", "e2e_ctc", "e2e_rnnt", "ssl"]
_BUILTIN_MODEL_NAMES = _SHORT_NAMES + list(_MODEL_HASHES.keys())


def _download_file(file_url: str, file_path: str):
    """Helper to download a file if not already cached."""
    if os.path.exists(file_path):
        return file_path

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with urllib.request.urlopen(file_url) as source, open(file_path, "wb") as output:
        with tqdm(
            total=int(source.info().get("Content-Length", 0)),
            ncols=80,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    return file_path


def _normalize_builtin_model_name(model_name: str) -> str:
    if model_name in _SHORT_NAMES:
        return f"v3_{model_name}"
    return model_name


def _download_model(model_name: str, download_root: str) -> Tuple[str, str]:
    """Download the model weights if not already cached."""
    if model_name not in _BUILTIN_MODEL_NAMES:
        raise ValueError(
            f"Model '{model_name}' not found. Available built-in model names: {_BUILTIN_MODEL_NAMES}. "
            "Use a local .ckpt path, a local directory with a .ckpt file, or a Hugging Face repo id."
        )

    model_name = _normalize_builtin_model_name(model_name)
    model_url = f"{_URL_DIR}/{model_name}.ckpt"
    model_path = os.path.join(download_root, model_name + ".ckpt")
    return model_name, _download_file(model_url, model_path)


def _download_tokenizer(model_name: str, download_root: str) -> Optional[str]:
    """Download the tokenizer if required and return its path."""
    if model_name != "v1_rnnt" and "e2e" not in model_name:
        return None  # No tokenizer required for this model

    tokenizer_url = f"{_URL_DIR}/{model_name}_tokenizer.model"
    tokenizer_path = os.path.join(download_root, model_name + "_tokenizer.model")
    return _download_file(tokenizer_url, tokenizer_path)


def _looks_like_local_path(model_name: str) -> bool:
    path = Path(model_name).expanduser()
    return model_name.startswith((".", "~")) or path.is_absolute() or path.exists()


def _single_file(files: list[Path], *, kind: str, root: Path) -> Optional[Path]:
    if not files:
        return None
    if len(files) == 1:
        return files[0]
    names = ", ".join(str(p.relative_to(root)) for p in files[:8])
    raise ValueError(f"Multiple {kind} files found under {root}: {names}. Pass the exact file path instead.")


def _resolve_local_checkpoint(model_ref: str) -> Tuple[str, str, Optional[str]]:
    path = Path(model_ref).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Local model path does not exist: {path}")

    if path.is_file():
        if path.suffix != ".ckpt":
            raise ValueError(f"Local GigaAM checkpoint must be a .ckpt file, got: {path}")
        tokenizer_path = path.with_name(f"{path.stem}_tokenizer.model")
        if not tokenizer_path.exists():
            generic_tokenizer = path.with_name("tokenizer.model")
            tokenizer_path = generic_tokenizer if generic_tokenizer.exists() else None
        return path.stem, str(path), str(tokenizer_path) if tokenizer_path else None

    ckpt = path / f"{path.name}.ckpt"
    if not ckpt.exists():
        ckpt = _single_file(sorted(path.glob("*.ckpt")), kind="checkpoint", root=path)
    if ckpt is None:
        ckpt = _single_file(sorted(path.rglob("*.ckpt")), kind="checkpoint", root=path)
    if ckpt is None:
        raise FileNotFoundError(f"No .ckpt file found under local model directory: {path}")

    tokenizer_candidates = [
        ckpt.with_name(f"{ckpt.stem}_tokenizer.model"),
        ckpt.with_name("tokenizer.model"),
        path / f"{ckpt.stem}_tokenizer.model",
        path / "tokenizer.model",
    ]
    tokenizer_path = next((p for p in tokenizer_candidates if p.exists()), None)
    if tokenizer_path is None:
        tokenizer_path = _single_file(sorted(path.glob("*_tokenizer.model")), kind="tokenizer", root=path)

    return ckpt.stem, str(ckpt), str(tokenizer_path) if tokenizer_path else None


def hash_path(ckpt_path: str) -> str:
    """Calculate binary file hash for checksum."""
    md5 = hashlib.md5()
    with open(ckpt_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            md5.update(chunk)
    return md5.hexdigest()


def _normalize_device(device: Optional[Union[str, torch.device]]) -> torch.device:
    """Normalize device parameter to torch.device."""
    if device is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.device(device_str)
    if isinstance(device, str):
        return torch.device(device)
    return device


def _apply_runtime_options(
    model: Union[GigaAM, GigaAMEmo, GigaAMASR, torch.nn.Module],
    *,
    device_obj: torch.device,
    fp16_encoder: bool,
) -> Union[GigaAM, GigaAMEmo, GigaAMASR, torch.nn.Module]:
    model = model.eval()
    if fp16_encoder and device_obj.type != "cpu" and hasattr(model, "encoder"):
        model.encoder = model.encoder.half()
    return model.to(device_obj)


def _load_checkpoint_model(
    *,
    model_name: str,
    model_path: str,
    tokenizer_path: Optional[str],
    fp16_encoder: bool,
    use_flash: Optional[bool],
    device_obj: torch.device,
    verify_checksum: bool,
) -> Union[GigaAM, GigaAMEmo, GigaAMASR]:
    from .model import GigaAM, GigaAMASR, GigaAMEmo

    if verify_checksum:
        expected_hash = _MODEL_HASHES.get(model_name)
        if expected_hash is None:
            raise ValueError(f"No checksum is registered for model '{model_name}'. Set verify_checksum=false for local checkpoints.")
        assert hash_path(model_path) == expected_hash, f"Model checksum failed. Please run `rm {model_path}` and reload the model"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=(FutureWarning))
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    if use_flash is not None:
        checkpoint["cfg"].encoder.flash_attn = use_flash
    if checkpoint["cfg"].encoder.get("flash_attn", False) and device_obj.type == "cpu":
        logging.warning("flash_attn is not supported on CPU. Disabling it...")
        checkpoint["cfg"].encoder.flash_attn = False

    if tokenizer_path is not None:
        checkpoint["cfg"].decoding.model_path = tokenizer_path

    if "ssl" in model_name:
        model = GigaAM(checkpoint["cfg"])
    elif "emo" in model_name:
        model = GigaAMEmo(checkpoint["cfg"])
    else:
        model = GigaAMASR(checkpoint["cfg"])

    model.load_state_dict(checkpoint["state_dict"])
    checkpoint["cfg"].model_name = model_name
    return _apply_runtime_options(model, device_obj=device_obj, fp16_encoder=fp16_encoder)


def _load_hf_auto_model(
    *,
    model_name: str,
    hf_revision: Optional[str],
    hf_token: Optional[str],
    local_files_only: bool,
    trust_remote_code: bool,
    fp16_encoder: bool,
    device_obj: torch.device,
) -> torch.nn.Module:
    try:
        from transformers import AutoModel
    except ImportError as exc:  # pragma: no cover - dependency-dependent path
        raise RuntimeError(
            "Loading Hugging Face model ids requires transformers. Install it or pass a local .ckpt path."
        ) from exc

    model = AutoModel.from_pretrained(
        model_name,
        revision=hf_revision,
        token=hf_token,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )
    return _apply_runtime_options(model, device_obj=device_obj, fp16_encoder=fp16_encoder)


def load_model(
    model_name: str,
    fp16_encoder: bool = True,
    use_flash: Optional[bool] = False,
    device: Optional[Union[str, torch.device]] = None,
    download_root: Optional[str] = None,
    hf_revision: Optional[str] = None,
    hf_token: Optional[str] = None,
    local_files_only: bool = False,
    trust_remote_code: bool = True,
    verify_checksum: Optional[bool] = None,
) -> Union[GigaAM, GigaAMEmo, GigaAMASR, torch.nn.Module]:
    """
    Load a GigaAM model from one of three sources:
      - built-in CDN aliases: ctc, rnnt, ssl, v3_ctc, ...
      - local .ckpt file or local directory containing a .ckpt file
      - Hugging Face model id, for example ai-sage/GigaAM-v3 with hf_revision="ctc"

    Local checkpoints are not checksum-verified by default because their hashes are
    not known. Built-in CDN checkpoints keep checksum verification enabled by default.
    """
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name must be a non-empty string")

    device_obj = _normalize_device(device)
    if download_root is None:
        download_root = _CACHE_DIR

    if model_name in _BUILTIN_MODEL_NAMES:
        resolved_name, model_path = _download_model(model_name, download_root)
        tokenizer_path = _download_tokenizer(resolved_name, download_root)
        should_verify = True if verify_checksum is None else bool(verify_checksum)
        return _load_checkpoint_model(
            model_name=resolved_name,
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            fp16_encoder=fp16_encoder,
            use_flash=use_flash,
            device_obj=device_obj,
            verify_checksum=should_verify,
        )

    if _looks_like_local_path(model_name):
        resolved_name, model_path, tokenizer_path = _resolve_local_checkpoint(model_name)
        should_verify = False if verify_checksum is None else bool(verify_checksum)
        return _load_checkpoint_model(
            model_name=resolved_name,
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            fp16_encoder=fp16_encoder,
            use_flash=use_flash,
            device_obj=device_obj,
            verify_checksum=should_verify,
        )

    if "/" not in model_name:
        raise ValueError(
            f"Unknown model '{model_name}'. Use one of {_BUILTIN_MODEL_NAMES}, "
            "a local checkpoint path, or a Hugging Face repo id like 'org/model'."
        )

    return _load_hf_auto_model(
        model_name=model_name,
        hf_revision=hf_revision,
        hf_token=hf_token,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
        fp16_encoder=fp16_encoder,
        device_obj=device_obj,
    )
