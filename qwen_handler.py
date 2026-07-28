"""RunPod Serverless handler for Qwen3-TTS voice cloning and voice design."""

from __future__ import annotations

import base64
import binascii
import hashlib
import os
from pathlib import Path
import re
import subprocess
import tempfile
import threading
from typing import Any


MODEL_VARIANT = os.getenv("QWEN_MODEL_VARIANT", "base").strip().lower()
if MODEL_VARIANT not in {"base", "design"}:
    raise RuntimeError("QWEN_MODEL_VARIANT must be 'base' or 'design'")

DEFAULT_MODEL_IDS = {
    "base": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "design": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
}
MODEL_ID = os.getenv("QWEN_MODEL_ID", DEFAULT_MODEL_IDS[MODEL_VARIANT]).strip()
MODEL_PATH = os.getenv("QWEN_MODEL_PATH", MODEL_ID).strip()
MODEL_LABEL = f"qwen3-tts-1.7b-{'base' if MODEL_VARIANT == 'base' else 'voice-design'}"
ATTENTION_IMPLEMENTATION = os.getenv(
    "QWEN_ATTN_IMPLEMENTATION", "sdpa"
).strip()
MAX_TEXT_CHARS = int(os.getenv("MAX_TEXT_CHARS", "2000"))
MAX_REFERENCE_TEXT_CHARS = int(os.getenv("MAX_REFERENCE_TEXT_CHARS", "4000"))
MAX_VOICE_DESCRIPTION_CHARS = int(
    os.getenv("MAX_VOICE_DESCRIPTION_CHARS", "1200")
)
MAX_REFERENCE_BYTES = int(
    os.getenv("MAX_REFERENCE_BYTES", str(10 * 1024 * 1024))
)
MAX_AUDIO_BYTES = int(os.getenv("MAX_AUDIO_BYTES", str(14 * 1024 * 1024)))
DEFAULT_QUALITY_PRESET = os.getenv(
    "DEFAULT_QUALITY_PRESET", "publication"
).strip().lower()

# The balanced preset follows Qwen's published generation defaults. Publication
# reduces randomness modestly and masters the result without suppressing natural
# prosody; expressive leaves more variation for character work.
QUALITY_PRESETS = {
    "publication": {
        "temperature": 0.80,
        "top_p": 0.95,
        "top_k": 50,
        "repetition_penalty": 1.05,
        "max_segment_chars": 420,
        "segment_pause_ms": 120,
        "master_output": True,
    },
    "balanced": {
        "temperature": 0.90,
        "top_p": 1.0,
        "top_k": 50,
        "repetition_penalty": 1.05,
        "max_segment_chars": 600,
        "segment_pause_ms": 110,
        "master_output": False,
    },
    "expressive": {
        "temperature": 0.95,
        "top_p": 1.0,
        "top_k": 50,
        "repetition_penalty": 1.05,
        "max_segment_chars": 420,
        "segment_pause_ms": 100,
        "master_output": False,
    },
}
if DEFAULT_QUALITY_PRESET not in QUALITY_PRESETS:
    DEFAULT_QUALITY_PRESET = "publication"

_runtime: QwenRuntime | None = None
_runtime_lock = threading.Lock()
_generation_lock = threading.Lock()

_FORMAT_EXTENSIONS = {
    "wav": ".wav",
    "wave": ".wav",
    "mp3": ".mp3",
    "flac": ".flac",
    "ogg": ".ogg",
}
_MIME_FORMATS = {
    "audio/wav": "wav",
    "audio/wave": "wav",
    "audio/x-wav": "wav",
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
    "audio/flac": "flac",
    "audio/x-flac": "flac",
    "audio/ogg": "ogg",
}
_OUTPUT_FORMATS = {
    "wav": ("audio/wav", ".wav"),
    "mp3": ("audio/mpeg", ".mp3"),
}


class InputError(ValueError):
    """Raised when a request cannot be safely processed."""


def split_text_segments(text: str, maximum: int) -> list[str]:
    """Split long English text at natural boundaries."""

    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    segments: list[str] = []
    remaining = normalized
    while len(remaining) > maximum:
        window = remaining[: maximum + 1]
        boundary = -1
        include_boundary = False
        for markers in ((". ", "? ", "! "), ("; ", ": "), (", ",), (" ",)):
            candidates = [
                (window.rfind(marker), marker)
                for marker in markers
                if window.rfind(marker) >= maximum // 2
            ]
            if candidates:
                boundary, marker = max(candidates)
                include_boundary = marker in (". ", "? ", "!")
                break
        if boundary < 0:
            boundary = maximum
        elif include_boundary:
            boundary += 1
        segment = remaining[:boundary].strip()
        if not segment:
            boundary = maximum
            segment = remaining[:boundary].strip()
        segments.append(segment)
        remaining = remaining[boundary:].strip()
    if remaining:
        segments.append(remaining)
    return segments


class QwenRuntime:
    """Own the GPU model and builds request-scoped voice prompts."""

    def __init__(self) -> None:
        import torch
        from qwen_tts import Qwen3TTSModel

        if not torch.cuda.is_available():
            raise RuntimeError("Qwen3-TTS requires a CUDA GPU, but CUDA is unavailable")

        self.torch = torch
        self.variant = MODEL_VARIANT
        self.device_name = torch.cuda.get_device_name(0)
        self.dtype_name = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
        dtype = torch.bfloat16 if self.dtype_name == "bfloat16" else torch.float16
        self.model = Qwen3TTSModel.from_pretrained(
            MODEL_PATH,
            device_map="cuda:0",
            dtype=dtype,
            attn_implementation=ATTENTION_IMPLEMENTATION,
        )
        self.sample_rate = 24_000

    def _generation_options(
        self,
        *,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
    ) -> dict[str, Any]:
        return {
            "do_sample": True,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "subtalker_dosample": True,
            "subtalker_temperature": temperature,
            "subtalker_top_p": top_p,
            "subtalker_top_k": top_k,
            "max_new_tokens": 8192,
        }

    def _join(
        self, waveforms: list[Any], sample_rate: int, pause_ms: int
    ) -> Any:
        import numpy as np

        if not waveforms:
            raise RuntimeError("Qwen3-TTS returned no audio")
        normalized = [
            np.asarray(waveform, dtype=np.float32).squeeze()
            for waveform in waveforms
        ]
        if len(normalized) == 1:
            return normalized[0]
        pause = np.zeros(max(0, int(sample_rate * pause_ms / 1000)), np.float32)
        joined: list[np.ndarray] = []
        for index, waveform in enumerate(normalized):
            if index and pause.size:
                joined.append(pause)
            joined.append(waveform)
        return np.concatenate(joined)

    def generate_clone(
        self,
        text: str,
        reference_path: str,
        reference_text: str | None,
        *,
        x_vector_only_mode: bool,
        seed: int,
        max_segment_chars: int,
        segment_pause_ms: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
    ) -> Any:
        if self.variant != "base":
            raise InputError("this worker image does not support voice cloning")
        prompt = self.model.create_voice_clone_prompt(
            ref_audio=reference_path,
            ref_text=reference_text,
            x_vector_only_mode=x_vector_only_mode,
        )
        options = self._generation_options(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )
        waveforms: list[Any] = []
        sample_rate = self.sample_rate
        for index, segment in enumerate(
            split_text_segments(text, max_segment_chars)
        ):
            segment_seed = (seed + index) % 2_147_483_648
            self.torch.manual_seed(segment_seed)
            self.torch.cuda.manual_seed_all(segment_seed)
            with self.torch.inference_mode():
                generated, sample_rate = self.model.generate_voice_clone(
                    text=segment,
                    language="English",
                    voice_clone_prompt=prompt,
                    non_streaming_mode=True,
                    **options,
                )
            waveforms.append(generated[0])
        self.sample_rate = int(sample_rate)
        return self._join(waveforms, self.sample_rate, segment_pause_ms)

    def generate_design(
        self,
        text: str,
        voice_description: str,
        *,
        seed: int,
        max_segment_chars: int,
        segment_pause_ms: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
    ) -> Any:
        if self.variant != "design":
            raise InputError("this worker image does not support voice design")
        options = self._generation_options(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )
        waveforms: list[Any] = []
        sample_rate = self.sample_rate
        for index, segment in enumerate(
            split_text_segments(text, max_segment_chars)
        ):
            segment_seed = (seed + index) % 2_147_483_648
            self.torch.manual_seed(segment_seed)
            self.torch.cuda.manual_seed_all(segment_seed)
            with self.torch.inference_mode():
                generated, sample_rate = self.model.generate_voice_design(
                    text=segment,
                    language="English",
                    instruct=voice_description,
                    non_streaming_mode=True,
                    **options,
                )
            waveforms.append(generated[0])
        self.sample_rate = int(sample_rate)
        return self._join(waveforms, self.sample_rate, segment_pause_ms)


def get_runtime() -> QwenRuntime:
    global _runtime
    if _runtime is None:
        with _runtime_lock:
            if _runtime is None:
                _runtime = QwenRuntime()
    return _runtime


def _number(
    data: dict[str, Any],
    key: str,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    value = data.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InputError(f"{key} must be a number")
    number = float(value)
    if not minimum <= number <= maximum:
        raise InputError(f"{key} must be between {minimum} and {maximum}")
    return number


def _integer(
    data: dict[str, Any],
    key: str,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    value = data.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise InputError(f"{key} must be an integer")
    if not minimum <= value <= maximum:
        raise InputError(f"{key} must be between {minimum} and {maximum}")
    return value


def _parse_data_url(value: str) -> tuple[str, str]:
    header, separator, encoded = value.partition(",")
    if not separator or not header.lower().endswith(";base64"):
        raise InputError("reference_audio data URL must contain base64 audio")
    mime_type = header[5:-7].lower()
    audio_format = _MIME_FORMATS.get(mime_type)
    if audio_format is None:
        raise InputError(f"unsupported reference audio content type: {mime_type}")
    return encoded, audio_format


def decode_reference(data: dict[str, Any]) -> tuple[bytes, str] | None:
    value = data.get("reference_audio")
    if value is None:
        return None

    format_hint: Any = data.get("reference_audio_format", "wav")
    if isinstance(value, dict):
        encoded = value.get("base64")
        content_type = value.get("content_type")
        format_hint = value.get("format", format_hint)
        if content_type is not None:
            if not isinstance(content_type, str):
                raise InputError("reference_audio.content_type must be a string")
            format_hint = _MIME_FORMATS.get(content_type.lower())
            if format_hint is None:
                raise InputError(
                    f"unsupported reference audio content type: {content_type}"
                )
    elif isinstance(value, str):
        if value.startswith("data:"):
            encoded, format_hint = _parse_data_url(value)
        else:
            encoded = value
    else:
        raise InputError("reference_audio must be a base64 string or object")

    if not isinstance(encoded, str) or not encoded:
        raise InputError("reference_audio base64 content is required")
    if not isinstance(format_hint, str):
        raise InputError("reference_audio_format must be a string")
    normalized_format = format_hint.lower().lstrip(".")
    extension = _FORMAT_EXTENSIONS.get(normalized_format)
    if extension is None:
        raise InputError("unsupported reference audio format; use WAV, MP3, FLAC, or OGG")

    encoded = "".join(encoded.split())
    if len(encoded) > ((MAX_REFERENCE_BYTES + 2) // 3) * 4 + 4:
        raise InputError(
            f"reference_audio exceeds the {MAX_REFERENCE_BYTES}-byte limit"
        )
    try:
        decoded = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise InputError("reference_audio is not valid base64") from exc
    if not decoded:
        raise InputError("reference_audio decoded to an empty file")
    if len(decoded) > MAX_REFERENCE_BYTES:
        raise InputError(
            f"reference_audio exceeds the {MAX_REFERENCE_BYTES}-byte limit"
        )
    return decoded, extension


def _save_wave(waveform: Any, sample_rate: int, path: Path) -> float:
    import numpy as np
    import soundfile

    samples = np.asarray(waveform, dtype=np.float32).squeeze()
    if samples.ndim != 1 or samples.size == 0:
        raise RuntimeError("model returned invalid audio")
    if not np.isfinite(samples).all():
        raise RuntimeError("model returned non-finite audio samples")
    samples = np.clip(samples, -1.0, 1.0)
    soundfile.write(path, samples, sample_rate, format="WAV", subtype="PCM_16")
    return float(samples.size / sample_rate)


def _encode_mp3(wave_path: Path, mp3_path: Path) -> None:
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-y",
                "-i",
                str(wave_path),
                "-map_metadata",
                "-1",
                "-codec:a",
                "libmp3lame",
                "-b:a",
                "160k",
                str(mp3_path),
            ],
            check=True,
            timeout=120,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("MP3 output is unavailable because ffmpeg is missing") from exc
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("failed to encode generated audio as MP3") from exc
    if not mp3_path.is_file() or mp3_path.stat().st_size <= 0:
        raise RuntimeError("MP3 encoder returned an empty output")


def _master_wave(wave_path: Path, sample_rate: int) -> None:
    mastered_path = wave_path.with_name("mastered.wav")
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-y",
                "-i",
                str(wave_path),
                "-af",
                "loudnorm=I=-19:TP=-1.5:LRA=7",
                "-ar",
                str(sample_rate),
                "-ac",
                "1",
                "-codec:a",
                "pcm_s16le",
                str(mastered_path),
            ],
            check=True,
            timeout=120,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("publication mastering is unavailable because ffmpeg is missing") from exc
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("failed to master generated publication audio") from exc
    if not mastered_path.is_file() or mastered_path.stat().st_size <= 0:
        raise RuntimeError("publication mastering returned an empty output")
    mastered_path.replace(wave_path)


def _info(runtime: QwenRuntime) -> dict[str, Any]:
    return {
        "model": MODEL_LABEL,
        "model_id": MODEL_ID,
        "model_variant": runtime.variant,
        "language": "English",
        "supported_languages": [
            "Chinese",
            "English",
            "Japanese",
            "Korean",
            "German",
            "French",
            "Russian",
            "Portuguese",
            "Spanish",
            "Italian",
        ],
        "sample_rate": runtime.sample_rate,
        "gpu": runtime.device_name,
        "dtype": runtime.dtype_name,
        "attention_implementation": ATTENTION_IMPLEMENTATION,
        "voice_cloning": runtime.variant == "base",
        "voice_design": runtime.variant == "design",
        "exact_transcript_cloning": runtime.variant == "base",
        "built_in_voice": False,
        "watermarked": False,
        "max_text_chars": MAX_TEXT_CHARS,
        "max_reference_text_chars": MAX_REFERENCE_TEXT_CHARS,
        "max_reference_bytes": MAX_REFERENCE_BYTES,
        "accepted_reference_formats": ["wav", "mp3", "flac", "ogg"],
        "output_formats": list(_OUTPUT_FORMATS),
        "default_quality_preset": DEFAULT_QUALITY_PRESET,
        "quality_presets": list(QUALITY_PRESETS),
    }


def handle_input(data: dict[str, Any], runtime: QwenRuntime) -> dict[str, Any]:
    action = data.get("action", "generate")
    if action == "info":
        return _info(runtime)
    if runtime.variant == "base" and action not in {"generate", "clone"}:
        raise InputError("action must be 'generate', 'clone', or 'info'")
    if runtime.variant == "design" and action not in {"generate", "design"}:
        raise InputError("action must be 'generate', 'design', or 'info'")

    text = data.get("text")
    if not isinstance(text, str) or not text.strip():
        raise InputError("text must be a non-empty string")
    text = text.strip()
    if len(text) > MAX_TEXT_CHARS:
        raise InputError(f"text must not exceed {MAX_TEXT_CHARS} characters")

    preset_name = data.get("quality_preset", DEFAULT_QUALITY_PRESET)
    if not isinstance(preset_name, str):
        raise InputError("quality_preset must be a string")
    preset_name = preset_name.strip().lower()
    if preset_name == "publish":
        preset_name = "publication"
    preset = QUALITY_PRESETS.get(preset_name)
    if preset is None:
        raise InputError(
            "quality_preset must be publication, balanced, or expressive"
        )

    seed = _integer(data, "seed", 0, 0, 2_147_483_647)
    temperature = _number(
        data, "temperature", preset["temperature"], 0.05, 2.0
    )
    top_p = _number(data, "top_p", preset["top_p"], 0.05, 1.0)
    top_k = _integer(data, "top_k", preset["top_k"], 1, 2000)
    repetition_penalty = _number(
        data, "repetition_penalty", preset["repetition_penalty"], 1.0, 2.5
    )
    output_format = data.get("output_format", "wav")
    if not isinstance(output_format, str):
        raise InputError("output_format must be a string")
    output_format = output_format.lower().lstrip(".")
    if output_format not in _OUTPUT_FORMATS:
        raise InputError("output_format must be 'wav' or 'mp3'")

    reference = None
    reference_text = None
    x_vector_only_mode = False
    voice_description = None
    if runtime.variant == "base":
        reference = decode_reference(data)
        if reference is None:
            raise InputError("reference_audio is required for voice cloning")
        reference_text_value = data.get("reference_text", data.get("ref_text"))
        if reference_text_value is not None:
            if not isinstance(reference_text_value, str):
                raise InputError("reference_text must be a string")
            reference_text = reference_text_value.strip()
            if not reference_text:
                reference_text = None
            elif len(reference_text) > MAX_REFERENCE_TEXT_CHARS:
                raise InputError(
                    f"reference_text must not exceed {MAX_REFERENCE_TEXT_CHARS} characters"
                )
        requested_x_vector = data.get(
            "x_vector_only_mode", reference_text is None
        )
        if not isinstance(requested_x_vector, bool):
            raise InputError("x_vector_only_mode must be true or false")
        x_vector_only_mode = requested_x_vector
        if not x_vector_only_mode and not reference_text:
            raise InputError(
                "reference_text is required unless x_vector_only_mode is true"
            )
    else:
        voice_description_value = data.get(
            "voice_description", data.get("instruct")
        )
        if not isinstance(voice_description_value, str) or not voice_description_value.strip():
            raise InputError("voice_description is required for voice design")
        voice_description = voice_description_value.strip()
        if len(voice_description) > MAX_VOICE_DESCRIPTION_CHARS:
            raise InputError(
                "voice_description must not exceed "
                f"{MAX_VOICE_DESCRIPTION_CHARS} characters"
            )

    with tempfile.TemporaryDirectory(prefix="qwen3-tts-job-") as temp_dir:
        temp_path = Path(temp_dir)
        with _generation_lock:
            if runtime.variant == "base":
                reference_bytes, extension = reference
                reference_path = temp_path / f"reference{extension}"
                reference_path.write_bytes(reference_bytes)
                waveform = runtime.generate_clone(
                    text,
                    str(reference_path),
                    reference_text,
                    x_vector_only_mode=x_vector_only_mode,
                    seed=seed,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    max_segment_chars=preset["max_segment_chars"],
                    segment_pause_ms=preset["segment_pause_ms"],
                )
            else:
                waveform = runtime.generate_design(
                    text,
                    voice_description,
                    seed=seed,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    max_segment_chars=preset["max_segment_chars"],
                    segment_pause_ms=preset["segment_pause_ms"],
                )

        wave_path = temp_path / "output.wav"
        duration = _save_wave(waveform, runtime.sample_rate, wave_path)
        if preset["master_output"]:
            _master_wave(wave_path, runtime.sample_rate)
        mime_type, extension = _OUTPUT_FORMATS[output_format]
        output_path = temp_path / f"output{extension}"
        if output_format == "mp3":
            _encode_mp3(wave_path, output_path)
        audio = output_path.read_bytes()

    if len(audio) > MAX_AUDIO_BYTES:
        raise RuntimeError(
            "generated audio is too large for an inline RunPod response; "
            "split the text into shorter chunks"
        )

    clone_mode = None
    if runtime.variant == "base":
        clone_mode = (
            "speaker_embedding" if x_vector_only_mode else "exact_transcript"
        )
    return {
        "audio_base64": base64.b64encode(audio).decode("ascii"),
        "mime_type": mime_type,
        "output_format": output_format,
        "sample_rate": runtime.sample_rate,
        "duration_seconds": round(duration, 3),
        "sha256": hashlib.sha256(audio).hexdigest(),
        "model": MODEL_LABEL,
        "model_id": MODEL_ID,
        "model_variant": runtime.variant,
        "seed": seed,
        "quality_preset": preset_name,
        "segment_count": len(
            split_text_segments(text, preset["max_segment_chars"])
        ),
        "mastering": "podcast_mono_-19_lufs" if preset["master_output"] else None,
        "used_reference_voice": runtime.variant == "base",
        "reference_text_used": bool(reference_text),
        "clone_mode": clone_mode,
        "watermarked": False,
    }


def handler(job: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(job, dict):
        raise InputError("job must be an object")
    data = job.get("input")
    if not isinstance(data, dict):
        raise InputError("job.input must be an object")
    return handle_input(data, get_runtime())


if __name__ == "__main__":
    get_runtime()
    import runpod

    runpod.serverless.start({"handler": handler})
