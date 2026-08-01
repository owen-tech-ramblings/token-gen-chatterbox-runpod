"""RunPod Serverless audition worker for IndexTTS2."""

from __future__ import annotations

import base64
import binascii
from hashlib import sha256
import io
import os
from pathlib import Path
import random
import tempfile
import threading
import time
from typing import Any
import wave


MODEL_ID = "IndexTeam/IndexTTS-2"
MODEL_REVISION = os.getenv(
    "MODEL_REVISION", "740dcaff396282ffb241903d150ac011cd4b1ede"
)
SOURCE_REVISION = "13495845e3028f0bb6ca1462ad22aa0e76349e40"
MODEL_DIR = Path("/models/index-tts-2")
MAX_REFERENCE_BYTES = 12 * 1024 * 1024
DELIVERIES = (
    "neutral",
    "warm",
    "concerned",
    "surprised",
    "joyful",
    "reflective",
)
EMOTION_VECTORS = {
    # [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
    "neutral": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.60],
    "warm": [0.30, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.35],
    "concerned": [0.0, 0.0, 0.35, 0.15, 0.0, 0.0, 0.0, 0.10],
    "surprised": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.55, 0.10],
    "joyful": [0.65, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "reflective": [0.0, 0.0, 0.0, 0.0, 0.0, 0.40, 0.0, 0.20],
}

_runtime: Any | None = None
_runtime_lock = threading.Lock()
_generation_lock = threading.Lock()


class InputError(ValueError):
    pass


def _decode_reference(value: Any) -> tuple[bytes, str]:
    if not isinstance(value, dict):
        raise InputError("reference_audio must be an object")
    encoded = value.get("base64")
    if not isinstance(encoded, str) or not encoded:
        raise InputError("reference_audio.base64 is required")
    try:
        audio = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise InputError("reference_audio.base64 is invalid") from exc
    if not audio or len(audio) > MAX_REFERENCE_BYTES:
        raise InputError("reference audio must be between 1 byte and 12 MiB")
    content_type = str(value.get("content_type") or "audio/wav").lower()
    suffixes = {
        "audio/wav": ".wav",
        "audio/wave": ".wav",
        "audio/x-wav": ".wav",
        "audio/mpeg": ".mp3",
        "audio/mp3": ".mp3",
        "audio/flac": ".flac",
        "audio/ogg": ".ogg",
    }
    if content_type not in suffixes:
        raise InputError("unsupported reference audio content type")
    return audio, suffixes[content_type]


def _validate(job: dict[str, Any]) -> dict[str, Any]:
    data = job.get("input", job)
    if not isinstance(data, dict):
        raise InputError("input must be an object")
    action = str(data.get("action") or "generate").strip().lower()
    if action == "info":
        return {"action": "info"}
    text = data.get("text")
    if not isinstance(text, str) or not text.strip():
        raise InputError("text is required")
    text = text.strip()
    if len(text) > 2_000:
        raise InputError("text must not exceed 2000 characters")
    audio, suffix = _decode_reference(data.get("reference_audio"))
    delivery = str(data.get("delivery") or "neutral").strip().lower()
    if delivery not in DELIVERIES:
        raise InputError(f"delivery must be one of: {', '.join(DELIVERIES)}")
    try:
        strength = float(data.get("strength", 0.60))
    except (TypeError, ValueError) as exc:
        raise InputError("strength must be a number") from exc
    if not 0.0 <= strength <= 1.0:
        raise InputError("strength must be between 0 and 1")
    try:
        seed = int(data.get("seed", 42))
    except (TypeError, ValueError) as exc:
        raise InputError("seed must be an integer") from exc
    return {
        "action": "generate",
        "text": text,
        "reference_audio": audio,
        "reference_suffix": suffix,
        "delivery": delivery,
        "strength": strength,
        "seed": seed,
    }


def _get_runtime() -> Any:
    global _runtime
    if _runtime is not None:
        return _runtime
    with _runtime_lock:
        if _runtime is not None:
            return _runtime
        from huggingface_hub import snapshot_download
        from indextts.infer_v2 import IndexTTS2

        if not (MODEL_DIR / "config.yaml").is_file():
            print(f"Downloading {MODEL_ID} at {MODEL_REVISION}", flush=True)
            snapshot_download(
                repo_id=MODEL_ID,
                revision=MODEL_REVISION,
                local_dir=MODEL_DIR,
            )
        print("Loading isolated IndexTTS2 audition runtime", flush=True)
        _runtime = IndexTTS2(
            cfg_path=str(MODEL_DIR / "config.yaml"),
            model_dir=str(MODEL_DIR),
            use_fp16=True,
            use_cuda_kernel=False,
            use_deepspeed=False,
            use_accel=False,
            use_torch_compile=False,
        )
    return _runtime


def _wav_metadata(audio: bytes) -> tuple[int, float]:
    with wave.open(io.BytesIO(audio), "rb") as wav:
        rate = wav.getframerate()
        return rate, wav.getnframes() / rate


def handler(job: dict[str, Any]) -> dict[str, Any]:
    try:
        request = _validate(job)
        if request["action"] == "info":
            return {
                "backend": "indextts2",
                "model": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "source_revision": SOURCE_REVISION,
                "deliveries": list(DELIVERIES),
                "emotion_vector_order": [
                    "happy",
                    "angry",
                    "sad",
                    "afraid",
                    "disgusted",
                    "melancholic",
                    "surprised",
                    "calm",
                ],
                "isolated_audition_worker": True,
            }
        runtime = _get_runtime()
        started = time.monotonic()
        random.seed(request["seed"])
        import numpy as np
        import torch

        np.random.seed(request["seed"] % (2**32 - 1))
        torch.manual_seed(request["seed"])
        torch.cuda.manual_seed_all(request["seed"])
        with tempfile.TemporaryDirectory(prefix="indextts2-audition-") as temp_dir:
            root = Path(temp_dir)
            reference = root / ("reference" + request["reference_suffix"])
            output = root / "output.wav"
            reference.write_bytes(request["reference_audio"])
            vector = runtime.normalize_emo_vec(
                list(EMOTION_VECTORS[request["delivery"]]), apply_bias=True
            )
            with _generation_lock:
                result = runtime.infer(
                    spk_audio_prompt=str(reference),
                    text=request["text"],
                    output_path=str(output),
                    emo_vector=vector,
                    emo_alpha=request["strength"],
                    use_random=False,
                    verbose=False,
                    max_text_tokens_per_segment=120,
                )
            if result is None or not output.is_file():
                raise RuntimeError("IndexTTS2 returned no audio")
            audio = output.read_bytes()
        sample_rate, duration = _wav_metadata(audio)
        return {
            "audio_base64": base64.b64encode(audio).decode("ascii"),
            "mime_type": "audio/wav",
            "output_format": "wav",
            "sha256": sha256(audio).hexdigest(),
            "duration_seconds": round(duration, 3),
            "sample_rate": sample_rate,
            "model": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "delivery": request["delivery"],
            "strength": request["strength"],
            "seed": request["seed"],
            "generation_seconds": round(time.monotonic() - started, 3),
            "used_reference_voice": True,
            "use_random": False,
        }
    except InputError as exc:
        return {"error": str(exc), "error_type": "invalid_input"}
    except Exception as exc:
        return {"error": str(exc), "error_type": "generation_failed"}


if __name__ == "__main__":
    import runpod

    runpod.serverless.start({"handler": handler})
