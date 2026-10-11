"""RunPod Serverless handler for Chatterbox Turbo text-to-speech."""

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


MODEL_ID = "ResembleAI/chatterbox-turbo"
MODEL_LABEL = "chatterbox-turbo"
MAX_TEXT_CHARS = int(os.getenv("MAX_TEXT_CHARS", "2000"))
MAX_REFERENCE_BYTES = int(os.getenv("MAX_REFERENCE_BYTES", str(10 * 1024 * 1024)))
MAX_AUDIO_BYTES = int(os.getenv("MAX_AUDIO_BYTES", str(14 * 1024 * 1024)))
DEFAULT_QUALITY_PRESET = os.getenv("DEFAULT_QUALITY_PRESET", "native").strip().lower()

# Native uses Chatterbox Turbo 0.1.7 sampling defaults. Other presets are
# explicit style choices, not an ordered quality scale. Mastering is separate.
QUALITY_PRESETS = {
    "native": {
        "temperature": 0.8,
        "top_p": 0.95,
        "top_k": 1000,
        "repetition_penalty": 1.2,
        "normalize_loudness": True,
        "max_segment_chars": 700,
        "segment_pause_ms": 120,
    },
    "publication": {
        "temperature": 0.65,
        "top_p": 0.90,
        "top_k": 500,
        "repetition_penalty": 1.20,
        "normalize_loudness": True,
        "max_segment_chars": 360,
        "segment_pause_ms": 140,
    },
    "balanced": {
        "temperature": 0.80,
        "top_p": 0.95,
        "top_k": 1000,
        "repetition_penalty": 1.20,
        "normalize_loudness": True,
        "max_segment_chars": 700,
        "segment_pause_ms": 120,
    },
    "expressive": {
        "temperature": 0.90,
        "top_p": 0.97,
        "top_k": 1000,
        "repetition_penalty": 1.15,
        "normalize_loudness": True,
        "max_segment_chars": 360,
        "segment_pause_ms": 120,
    },
}
if DEFAULT_QUALITY_PRESET not in QUALITY_PRESETS:
    DEFAULT_QUALITY_PRESET = "native"

_runtime: ChatterboxRuntime | None = None
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


from audio_transfer import download_reference, upload_audio, validate_transfer


class InputError(ValueError):
    """Raised when a request cannot be safely processed."""


def split_text_segments(text: str, maximum: int) -> list[str]:
    """Split long text at natural English boundaries for stable generation."""

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


class ChatterboxRuntime:
    """Owns the GPU model and prevents voice state crossing job boundaries."""

    def __init__(self) -> None:
        import torch
        from chatterbox.tts_turbo import ChatterboxTurboTTS

        if not torch.cuda.is_available():
            raise RuntimeError("Chatterbox requires a CUDA GPU, but CUDA is unavailable")

        self.torch = torch
        self.model = ChatterboxTurboTTS.from_pretrained(device="cuda")
        self.sample_rate = int(self.model.sr)
        self.device_name = torch.cuda.get_device_name(0)

        # generate(audio_prompt_path=...) replaces model.conds. Retaining the
        # original object lets us restore the built-in voice after every job,
        # preventing one user's reference voice from leaking into another job.
        self._builtin_conditionals = self.model.conds
        self.has_builtin_voice = self._builtin_conditionals is not None

    def generate(
        self,
        text: str,
        reference_path: str | None,
        *,
        seed: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        normalize_loudness: bool,
        max_segment_chars: int,
        segment_pause_ms: int,
    ) -> Any:
        if reference_path is None and not self.has_builtin_voice:
            raise InputError(
                "reference_audio is required because this model image has no built-in voice"
            )

        segments = split_text_segments(text, max_segment_chars)
        self.model.conds = self._builtin_conditionals
        try:
            waveforms = []
            for index, segment in enumerate(segments):
                segment_seed = (seed + index) % 2_147_483_648
                self.torch.manual_seed(segment_seed)
                self.torch.cuda.manual_seed_all(segment_seed)
                with self.torch.inference_mode():
                    waveforms.append(
                        self.model.generate(
                            segment,
                            # The first call prepares the reference voice. Later
                            # calls reuse it so every segment has the same voice.
                            audio_prompt_path=reference_path if index == 0 else None,
                            temperature=temperature,
                            top_p=top_p,
                            top_k=top_k,
                            repetition_penalty=repetition_penalty,
                            norm_loudness=normalize_loudness,
                        )
                    )
            if len(waveforms) == 1:
                return waveforms[0]
            pause_samples = max(0, int(self.sample_rate * segment_pause_ms / 1000))
            joined = []
            for index, waveform in enumerate(waveforms):
                if index and pause_samples:
                    joined.append(
                        self.torch.zeros(
                            (waveform.shape[0], pause_samples),
                            dtype=waveform.dtype,
                            device=waveform.device,
                        )
                    )
                joined.append(waveform)
            return self.torch.cat(joined, dim=-1)
        finally:
            self.model.conds = self._builtin_conditionals


def get_runtime() -> ChatterboxRuntime:
    global _runtime
    if _runtime is None:
        with _runtime_lock:
            if _runtime is None:
                _runtime = ChatterboxRuntime()
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
    value = float(value)
    if not minimum <= value <= maximum:
        raise InputError(f"{key} must be between {minimum} and {maximum}")
    return value


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


def _boolean(data: dict[str, Any], key: str, default: bool) -> bool:
    value = data.get(key, default)
    if not isinstance(value, bool):
        raise InputError(f"{key} must be true or false")
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
    if "reference_download" in data:
        return download_reference(data)
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
        supported = ", ".join(sorted(set(_FORMAT_EXTENSIONS.values())))
        raise InputError(f"unsupported reference audio format; use one of: {supported}")

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

    if hasattr(waveform, "detach"):
        waveform = waveform.detach().cpu().numpy()
    samples = np.asarray(waveform, dtype=np.float32).squeeze()
    if samples.ndim != 1 or samples.size == 0:
        raise RuntimeError("model returned an invalid audio tensor")
    if not np.isfinite(samples).all():
        raise RuntimeError("model returned non-finite audio samples")
    soundfile.write(path, samples, sample_rate, format="WAV", subtype="FLOAT")
    return float(samples.size / sample_rate)


def _encode_pcm16(wave_path: Path) -> None:
    """Quantise only when the caller explicitly requests compatibility PCM16."""
    import numpy as np
    import soundfile

    samples, sample_rate = soundfile.read(wave_path, dtype="float32")
    converted = wave_path.with_name("pcm16.wav")
    soundfile.write(converted, np.clip(samples, -1.0, 1.0), sample_rate,
                    format="WAV", subtype="PCM_16")
    converted.replace(wave_path)


def _encode_mp3(wave_path: Path, mp3_path: Path) -> None:
    """Encode a generated PCM WAV as a broadly compatible constant-bitrate MP3."""

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
                # 160 kbit/s is the maximum MPEG-2 Layer III rate at the
                # model's native 24 kHz sample rate.
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
    """Apply a consistent mono spoken-word master for publication output."""

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
                "pcm_f32le",
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


def _info(runtime: ChatterboxRuntime) -> dict[str, Any]:
    return {
        "audio_transfer": {"version": 1, "chunk_bytes": 8 * 1024 * 1024, "max_bytes": 768 * 1024 * 1024},
        "model": MODEL_LABEL,
        "model_id": MODEL_ID,
        "language": "English",
        "sample_rate": runtime.sample_rate,
        "gpu": runtime.device_name,
        "voice_cloning": True,
        "built_in_voice": runtime.has_builtin_voice,
        "watermarked": True,
        "max_text_chars": MAX_TEXT_CHARS,
        "max_reference_bytes": MAX_REFERENCE_BYTES,
        "accepted_reference_formats": ["wav", "mp3", "flac", "ogg"],
        "output_formats": list(_OUTPUT_FORMATS),
        "default_quality_preset": DEFAULT_QUALITY_PRESET,
        "quality_presets": list(QUALITY_PRESETS),
        "default_wav_encoding": "float32",
        "wav_encodings": ["float32", "pcm16"],
        "default_mastering": "native",
        "mastering_modes": ["native", "podcast_mono_-19_lufs"],
    }


def handle_input(data: dict[str, Any], runtime: ChatterboxRuntime) -> dict[str, Any]:
    action = data.get("action", "generate")
    if action == "info":
        return _info(runtime)
    if action != "generate":
        raise InputError("action must be 'generate' or 'info'")

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
        supported = ", ".join(QUALITY_PRESETS)
        raise InputError(f"quality_preset must be one of: {supported}")

    seed = _integer(data, "seed", 0, 0, 2_147_483_647)
    temperature = _number(
        data, "temperature", preset["temperature"], 0.05, 2.0
    )
    top_p = _number(data, "top_p", preset["top_p"], 0.05, 1.0)
    top_k = _integer(data, "top_k", preset["top_k"], 1, 2000)
    repetition_penalty = _number(
        data, "repetition_penalty", preset["repetition_penalty"], 1.0, 2.5
    )
    normalize_loudness = _boolean(
        data, "normalize_loudness", preset["normalize_loudness"]
    )
    reference = decode_reference(data)
    output_format = data.get("output_format", "wav")
    if not isinstance(output_format, str):
        raise InputError("output_format must be a string")
    output_format = output_format.lower().lstrip(".")
    if output_format not in _OUTPUT_FORMATS:
        raise InputError("output_format must be 'wav' or 'mp3'")

    wav_encoding = data.get("wav_encoding", "float32")
    if not isinstance(wav_encoding, str) or wav_encoding not in {"float32", "pcm16"}:
        raise InputError("wav_encoding must be 'float32' or 'pcm16'")
    if output_format != "wav" and wav_encoding != "float32":
        raise InputError("wav_encoding='pcm16' requires output_format='wav'")
    mastering = data.get("mastering", "native")
    if not isinstance(mastering, str) or mastering not in {"native", "podcast_mono_-19_lufs"}:
        raise InputError("mastering must be 'native' or 'podcast_mono_-19_lufs'")

    transfer = validate_transfer(data["transfer"]) if "transfer" in data else None

    with tempfile.TemporaryDirectory(prefix="chatterbox-job-") as temp_dir:
        temp_path = Path(temp_dir)
        reference_path: Path | None = None
        if reference is not None:
            reference_bytes, extension = reference
            reference_path = temp_path / f"reference{extension}"
            reference_path.write_bytes(reference_bytes)

        wave_path = temp_path / "output.wav"
        with _generation_lock:
            waveform = runtime.generate(
                text,
                str(reference_path) if reference_path else None,
                seed=seed,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                normalize_loudness=normalize_loudness,
                max_segment_chars=preset["max_segment_chars"],
                segment_pause_ms=preset["segment_pause_ms"],
            )
        duration = _save_wave(waveform, runtime.sample_rate, wave_path)
        if mastering != "native":
            _master_wave(wave_path, runtime.sample_rate)
        if output_format == "wav" and wav_encoding == "pcm16":
            _encode_pcm16(wave_path)
        mime_type, extension = _OUTPUT_FORMATS[output_format]
        output_path = temp_path / f"output{extension}"
        if output_format == "mp3":
            _encode_mp3(wave_path, output_path)
        result = {
            "mime_type": mime_type,
            "output_format": output_format,
            "sample_rate": runtime.sample_rate,
            "duration_seconds": round(duration, 3),
            "model": MODEL_LABEL,
            "seed": seed,
            "quality_preset": preset_name,
            "segment_count": len(split_text_segments(text, preset["max_segment_chars"])),
            "mastering": mastering if mastering != "native" else None,
            "wav_encoding": wav_encoding if output_format == "wav" else None,
            "audio_encoding": ("pcm_f32le" if wav_encoding == "float32" else "pcm_s16le") if output_format == "wav" else "mp3",
            "bit_depth": (32 if wav_encoding == "float32" else 16) if output_format == "wav" else None,
            "native_audio_preserved": output_format == "wav" and wav_encoding == "float32" and mastering == "native",
            "used_reference_voice": reference is not None,
            "watermarked": True,
        }

        if transfer is not None:
            return upload_audio(output_path, transfer, result)
        if output_path.stat().st_size > MAX_AUDIO_BYTES:
            raise RuntimeError("generated audio is too large for an inline RunPod response; use authenticated audio transfer")
        audio = output_path.read_bytes()
        return {**result, "audio_base64": base64.b64encode(audio).decode("ascii"),
                "sha256": hashlib.sha256(audio).hexdigest()}


def handler(job: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(job, dict):
        raise InputError("job must be an object")
    data = job.get("input")
    if not isinstance(data, dict):
        raise InputError("job.input must be an object")
    return handle_input(data, get_runtime())


if __name__ == "__main__":
    # Load model weights before registering the worker as ready.
    get_runtime()
    import runpod

    runpod.serverless.start({"handler": handler})
