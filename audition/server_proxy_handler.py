"""Isolated RunPod audition worker for Fish S2 Pro, Higgs TTS 3, or ZONOS2.

The selected upstream server is launched lazily inside the first job so the
RunPod worker can register before public model weights are downloaded. Reference
audio exists only in a request-scoped temporary directory and is never logged.
"""

from __future__ import annotations

import base64
import binascii
from hashlib import sha256
import io
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import time
from typing import Any
import wave


BACKEND = os.getenv("AUDITION_BACKEND", "fish").strip().lower()
if BACKEND not in {"fish", "higgs", "zonos2"}:
    raise RuntimeError("AUDITION_BACKEND must be fish, higgs, or zonos2")

MODEL_REVISIONS = {
    "fish": os.getenv(
        "MODEL_REVISION", "1de9996b6be38b745688de084d87a5633f714e4e"
    ),
    "higgs": os.getenv(
        "MODEL_REVISION", "7556c17e05201fccd9c8cc120bc216dcc7b5d561"
    ),
    "zonos2": os.getenv(
        "MODEL_REVISION", "65f1e80f94b599d474bb6af9094a803dc52f60bd"
    ),
}
MODEL_IDS = {
    "fish": "fishaudio/s2-pro",
    "higgs": "bosonai/higgs-tts-3-4b",
    "zonos2": "Zyphra/ZONOS2",
}
SOURCE_REVISIONS = {
    "fish": "e5e292632cb11e7a27b2b7487f58f612bc101e13",
    "higgs": "sglang-omni image sha256:46235435997d1fa93fc81fb1c2d5b7fd8470d77395a5c348c0176094ffddf95e",
    "zonos2": "194c0a3ab67b90383a67646289f28d4ecb1c1f64",
}
MAX_TEXT_CHARS = 2_000
MAX_REFERENCE_TEXT_CHARS = 4_000
MAX_REFERENCE_BYTES = 12 * 1024 * 1024
DELIVERIES = (
    "neutral",
    "warm",
    "concerned",
    "surprised",
    "joyful",
    "reflective",
)

_server_process: subprocess.Popen[bytes] | None = None
_server_lock = threading.Lock()
_generation_lock = threading.Lock()


class InputError(ValueError):
    """A safe error caused by invalid request input."""


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
    if not audio:
        raise InputError("reference audio is empty")
    if len(audio) > MAX_REFERENCE_BYTES:
        raise InputError("reference audio exceeds 12 MiB")
    content_type = str(value.get("content_type") or "audio/wav").lower()
    extensions = {
        "audio/wav": ".wav",
        "audio/wave": ".wav",
        "audio/x-wav": ".wav",
        "audio/mpeg": ".mp3",
        "audio/mp3": ".mp3",
        "audio/flac": ".flac",
        "audio/ogg": ".ogg",
    }
    suffix = extensions.get(content_type)
    if suffix is None:
        raise InputError("unsupported reference audio content type")
    return audio, suffix


def _validate_request(job: dict[str, Any]) -> dict[str, Any]:
    data = job.get("input", job)
    if not isinstance(data, dict):
        raise InputError("input must be an object")
    action = str(data.get("action") or "generate").strip().lower()
    if action == "info":
        return {"action": "info"}
    if action != "generate":
        raise InputError("action must be generate or info")
    text = data.get("text")
    if not isinstance(text, str) or not text.strip():
        raise InputError("text is required")
    text = text.strip()
    if len(text) > MAX_TEXT_CHARS:
        raise InputError(f"text must not exceed {MAX_TEXT_CHARS} characters")
    reference_audio, suffix = _decode_reference(data.get("reference_audio"))
    reference_text = data.get("reference_text")
    if not isinstance(reference_text, str) or not reference_text.strip():
        raise InputError("reference_text is required")
    reference_text = reference_text.strip()
    if len(reference_text) > MAX_REFERENCE_TEXT_CHARS:
        raise InputError(
            f"reference_text must not exceed {MAX_REFERENCE_TEXT_CHARS} characters"
        )
    delivery = str(data.get("delivery") or "neutral").strip().lower()
    if delivery not in DELIVERIES:
        raise InputError(f"delivery must be one of: {', '.join(DELIVERIES)}")
    try:
        strength = float(data.get("strength", 0.65))
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
        "reference_audio": reference_audio,
        "reference_suffix": suffix,
        "reference_text": reference_text,
        "delivery": delivery,
        "strength": strength,
        "seed": seed,
    }


def _download_fish_model() -> Path:
    from huggingface_hub import snapshot_download

    destination = Path("/models/fish-s2-pro")
    if not (destination / "codec.pth").is_file():
        print(
            f"Downloading {MODEL_IDS['fish']} at {MODEL_REVISIONS['fish']}",
            flush=True,
        )
        snapshot_download(
            repo_id=MODEL_IDS["fish"],
            revision=MODEL_REVISIONS["fish"],
            local_dir=destination,
        )
    return destination


def _server_configuration() -> tuple[list[str], str, tuple[str, ...]]:
    if BACKEND == "fish":
        model_path = _download_fish_model()
        return (
            [
                "/opt/fish-speech/.venv/bin/python",
                "tools/api_server.py",
                "--llama-checkpoint-path",
                str(model_path),
                "--decoder-checkpoint-path",
                str(model_path / "codec.pth"),
                "--decoder-config-name",
                "modded_dac_vq",
                "--listen",
                "127.0.0.1:8080",
                "--half",
            ],
            "http://127.0.0.1:8080",
            ("/v1/health",),
        )
    if BACKEND == "higgs":
        return (
            [
                "sgl-omni",
                "serve",
                "--model-path",
                MODEL_IDS["higgs"],
                "--port",
                "8000",
            ],
            "http://127.0.0.1:8000",
            ("/v1/models", "/health"),
        )
    return (
        [
            "/opt/zonos2/.venv/bin/python",
            "-m",
            "zonos2",
            "--model-path",
            MODEL_IDS["zonos2"],
            "--host",
            "127.0.0.1",
            "--port",
            "1919",
            "--tts-emotion-directions-dir",
            "/opt/zonos2/emotion_directions",
        ],
        "http://127.0.0.1:1919",
        ("/v1/models", "/health"),
    )


def _wait_for_server(base_url: str, health_paths: tuple[str, ...]) -> None:
    import httpx

    deadline = time.monotonic() + 1_200
    with httpx.Client(timeout=10.0) as client:
        while time.monotonic() < deadline:
            if _server_process is not None and _server_process.poll() is not None:
                raise RuntimeError(
                    f"{BACKEND} model server exited with code "
                    f"{_server_process.returncode}"
                )
            for path in health_paths:
                try:
                    response = client.get(base_url + path)
                    if response.status_code < 500:
                        return
                except httpx.HTTPError:
                    pass
            time.sleep(2)
    raise TimeoutError(f"{BACKEND} model server did not become ready")


def _ensure_server() -> str:
    global _server_process
    command, base_url, health_paths = _server_configuration()
    if _server_process is not None and _server_process.poll() is None:
        return base_url
    with _server_lock:
        if _server_process is not None and _server_process.poll() is None:
            return base_url
        print(f"Starting isolated {BACKEND} audition server", flush=True)
        _server_process = subprocess.Popen(
            command,
            cwd={
                "fish": "/opt/fish-speech",
                "higgs": "/workspace",
                "zonos2": "/opt/zonos2",
            }[BACKEND],
        )
        _wait_for_server(base_url, health_paths)
        print(f"{BACKEND} audition server ready", flush=True)
    return base_url


FISH_TAGS = {
    "neutral": "",
    "warm": "[delight] ",
    "concerned": "[sad] ",
    "surprised": "[surprised] ",
    "joyful": "[excited, joyful tone] ",
    "reflective": "[low voice] ",
}
HIGGS_TAGS = {
    "neutral": "",
    "warm": "<|emotion:affection|>",
    "concerned": "<|emotion:sadness|>",
    "surprised": "<|emotion:surprise|>",
    "joyful": "<|emotion:elation|>",
    "reflective": "<|emotion:contemplation|><|prosody:speed_slow|>",
}
ZONOS_EMOTIONS = {
    "neutral": {},
    "warm": {"happy": 0.45},
    "concerned": {"sad": 0.35},
    "surprised": {"surprised": 0.55},
    "joyful": {"happy": 0.65},
    "reflective": {"sad": 0.20},
}


def _fish_request(base_url: str, request: dict[str, Any], reference_b64: str) -> bytes:
    import httpx

    tag = FISH_TAGS[request["delivery"]] if request["strength"] >= 0.2 else ""
    payload = {
        "text": tag + request["text"],
        "format": "wav",
        "references": [
            {"audio": reference_b64, "text": request["reference_text"]}
        ],
        "seed": request["seed"],
        "normalize": True,
        "streaming": False,
        "top_p": 0.8,
        "temperature": 0.8,
        "repetition_penalty": 1.1,
    }
    response = httpx.post(base_url + "/v1/tts", json=payload, timeout=900.0)
    response.raise_for_status()
    return response.content


def _higgs_request(
    base_url: str,
    request: dict[str, Any],
    reference_path: Path,
) -> bytes:
    import httpx

    tag = HIGGS_TAGS[request["delivery"]] if request["strength"] >= 0.2 else ""
    if request["delivery"] != "neutral":
        expressive = (
            "<|prosody:expressive_high|>"
            if request["strength"] >= 0.75
            else "<|prosody:expressive_low|>"
        )
        tag += expressive
    payload = {
        "input": tag + request["text"],
        "references": [
            {
                "audio_path": str(reference_path),
                "text": request["reference_text"],
            }
        ],
        "temperature": 0.8,
        "top_k": 50,
        "max_new_tokens": 2_048,
        "seed": request["seed"],
    }
    response = httpx.post(
        base_url + "/v1/audio/speech", json=payload, timeout=900.0
    )
    response.raise_for_status()
    return response.content


def _pcm_to_wav(pcm: bytes, sample_rate: int = 44_100) -> bytes:
    import numpy as np

    samples = np.frombuffer(pcm, dtype=np.float32)
    samples = np.nan_to_num(samples, nan=0.0, posinf=1.0, neginf=-1.0)
    int16 = (np.clip(samples, -1.0, 1.0) * 32_767).astype("<i2")
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(int16.tobytes())
    return output.getvalue()


def _zonos_request(base_url: str, request: dict[str, Any], reference_b64: str) -> bytes:
    import httpx

    delivery = request["delivery"]
    sliders = ZONOS_EMOTIONS[delivery]
    payload = {
        "text": request["text"],
        "language": "en_gb",
        "text_normalization": True,
        "seed": request["seed"],
        "stream": False,
        "speaker_audio_base64": reference_b64,
        "speaker_audio_name": "reference.wav",
        "clean_speaker_background": True,
        "accurate_mode": True,
        "emotion_enabled": bool(sliders) and request["strength"] >= 0.2,
        "emotion_sliders": sliders or None,
        "emotion_strength": request["strength"],
        "emotion_cfg_scale": 1.0,
        "quality_values": {"trailing_silence_s": 0.4},
    }
    if delivery == "reflective":
        payload["emotion_arousal"] = -0.25
    elif delivery in {"surprised", "joyful"}:
        payload["emotion_arousal"] = 0.2
    response = httpx.post(
        base_url + "/tts/generate", json=payload, timeout=900.0
    )
    response.raise_for_status()
    return _pcm_to_wav(response.content)


def _wav_metadata(audio: bytes) -> tuple[int, float]:
    with wave.open(io.BytesIO(audio), "rb") as wav:
        sample_rate = wav.getframerate()
        duration = wav.getnframes() / sample_rate
    return sample_rate, duration


def _info() -> dict[str, Any]:
    return {
        "backend": BACKEND,
        "model": MODEL_IDS[BACKEND],
        "model_revision": MODEL_REVISIONS[BACKEND],
        "source_revision": SOURCE_REVISIONS[BACKEND],
        "deliveries": list(DELIVERIES),
        "reference_text_required": BACKEND in {"fish", "higgs"},
        "isolated_audition_worker": True,
    }


def handler(job: dict[str, Any]) -> dict[str, Any]:
    try:
        request = _validate_request(job)
        if request["action"] == "info":
            return _info()
        base_url = _ensure_server()
        started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix=f"{BACKEND}-audition-") as temp_dir:
            reference_path = Path(temp_dir) / (
                "reference" + request["reference_suffix"]
            )
            reference_path.write_bytes(request["reference_audio"])
            reference_b64 = base64.b64encode(
                request["reference_audio"]
            ).decode("ascii")
            with _generation_lock:
                if BACKEND == "fish":
                    audio = _fish_request(base_url, request, reference_b64)
                elif BACKEND == "higgs":
                    audio = _higgs_request(base_url, request, reference_path)
                else:
                    audio = _zonos_request(base_url, request, reference_b64)
        sample_rate, duration = _wav_metadata(audio)
        return {
            "audio_base64": base64.b64encode(audio).decode("ascii"),
            "mime_type": "audio/wav",
            "output_format": "wav",
            "sha256": sha256(audio).hexdigest(),
            "duration_seconds": round(duration, 3),
            "sample_rate": sample_rate,
            "model": MODEL_IDS[BACKEND],
            "model_revision": MODEL_REVISIONS[BACKEND],
            "delivery": request["delivery"],
            "strength": request["strength"],
            "seed": request["seed"],
            "generation_seconds": round(time.monotonic() - started, 3),
            "used_reference_voice": True,
        }
    except InputError as exc:
        return {"error": str(exc), "error_type": "invalid_input"}
    except Exception as exc:
        return {"error": str(exc), "error_type": "generation_failed"}


if __name__ == "__main__":
    import runpod

    runpod.serverless.start({"handler": handler})
