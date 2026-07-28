from __future__ import annotations

import base64
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import handler


class FakeRuntime:
    sample_rate = 24_000
    device_name = "Fake GPU"
    has_builtin_voice = True

    def __init__(self) -> None:
        self.calls = []

    def generate(self, text, reference_path, **options):
        self.calls.append((text, reference_path, options))
        return object()


def fake_save_wave(_waveform, sample_rate: int, path: Path) -> float:
    assert sample_rate == 24_000
    path.write_bytes(b"RIFF-test-audio")
    return 1.25


def fake_encode_mp3(_wave_path: Path, mp3_path: Path) -> None:
    mp3_path.write_bytes(b"ID3-test-audio")


class HandlerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runtime = FakeRuntime()
        master_patcher = patch("handler._master_wave")
        self.master_wave = master_patcher.start()
        self.addCleanup(master_patcher.stop)

    def test_info(self) -> None:
        result = handler.handle_input({"action": "info"}, self.runtime)
        self.assertEqual(result["model"], "chatterbox-turbo")
        self.assertEqual(result["gpu"], "Fake GPU")
        self.assertTrue(result["voice_cloning"])
        self.assertEqual(result["default_quality_preset"], "publication")
        self.assertIn("publication", result["quality_presets"])

    @patch("handler._save_wave", fake_save_wave)
    def test_generate_with_builtin_voice(self) -> None:
        result = handler.handle_input(
            {"text": "Hello from Chatterbox.", "seed": 42}, self.runtime
        )
        self.assertEqual(base64.b64decode(result["audio_base64"]), b"RIFF-test-audio")
        self.assertEqual(result["duration_seconds"], 1.25)
        self.assertEqual(result["mime_type"], "audio/wav")
        self.assertEqual(result["output_format"], "wav")
        self.assertFalse(result["used_reference_voice"])
        self.assertEqual(self.runtime.calls[0][2]["seed"], 42)
        self.assertEqual(result["quality_preset"], "publication")
        self.assertEqual(self.runtime.calls[0][2]["temperature"], 0.65)
        self.assertEqual(result["mastering"], "podcast_mono_-19_lufs")
        self.master_wave.assert_called_once()

    @patch("handler._encode_mp3", fake_encode_mp3)
    @patch("handler._save_wave", fake_save_wave)
    def test_generate_mp3(self) -> None:
        result = handler.handle_input(
            {"text": "Return an MP3.", "output_format": "mp3"}, self.runtime
        )
        self.assertEqual(base64.b64decode(result["audio_base64"]), b"ID3-test-audio")
        self.assertEqual(result["mime_type"], "audio/mpeg")
        self.assertEqual(result["output_format"], "mp3")

    @patch("handler._save_wave", fake_save_wave)
    def test_generate_with_reference_voice(self) -> None:
        reference = base64.b64encode(b"fake-wave-data").decode()
        result = handler.handle_input(
            {
                "text": "Use this reference.",
                "reference_audio": {
                    "base64": reference,
                    "content_type": "audio/wav",
                },
            },
            self.runtime,
        )
        _, reference_path, _ = self.runtime.calls[0]
        self.assertTrue(reference_path.endswith("reference.wav"))
        self.assertTrue(result["used_reference_voice"])

    def test_decode_data_url(self) -> None:
        encoded = base64.b64encode(b"audio").decode()
        decoded, extension = handler.decode_reference(
            {"reference_audio": f"data:audio/mpeg;base64,{encoded}"}
        )
        self.assertEqual(decoded, b"audio")
        self.assertEqual(extension, ".mp3")

    def test_rejects_invalid_inputs(self) -> None:
        invalid_inputs = [
            {},
            {"text": ""},
            {"text": "x" * (handler.MAX_TEXT_CHARS + 1)},
            {"text": "hello", "temperature": "hot"},
            {"text": "hello", "top_p": 2},
            {"text": "hello", "reference_audio": "not-base64"},
            {"text": "hello", "output_format": "flac"},
            {"text": "hello", "quality_preset": "perfect-magic"},
            {"action": "delete"},
        ]
        for value in invalid_inputs:
            with self.subTest(value=value):
                with self.assertRaises(handler.InputError):
                    handler.handle_input(value, self.runtime)

    def test_reference_file_is_removed_after_job(self) -> None:
        reference = base64.b64encode(b"voice").decode()
        seen_path = None

        def capture_generate(text, reference_path, **options):
            nonlocal seen_path
            seen_path = reference_path
            return object()

        self.runtime.generate = capture_generate
        with patch("handler._save_wave", fake_save_wave):
            handler.handle_input(
                {"text": "Temporary.", "reference_audio": reference}, self.runtime
            )
        self.assertIsNotNone(seen_path)
        self.assertFalse(Path(seen_path).exists())

    def test_publication_preset_splits_at_sentence_boundaries(self) -> None:
        text = (
            "This is the first sentence. "
            "This is the second sentence with a little more detail. "
            "This is the final sentence."
        )
        segments = handler.split_text_segments(text, 90)
        self.assertGreater(len(segments), 1)
        self.assertEqual(" ".join(segments), text)
        self.assertTrue(segments[0].endswith("."))

    @patch("handler._save_wave", fake_save_wave)
    def test_balanced_preset_retains_upstream_sampling_defaults(self) -> None:
        result = handler.handle_input(
            {"text": "A balanced voice.", "quality_preset": "balanced"},
            self.runtime,
        )
        options = self.runtime.calls[0][2]
        self.assertEqual(result["quality_preset"], "balanced")
        self.assertEqual(options["temperature"], 0.8)
        self.assertEqual(options["top_p"], 0.95)
        self.assertEqual(options["top_k"], 1000)
        self.assertIsNone(result["mastering"])


if __name__ == "__main__":
    unittest.main()
