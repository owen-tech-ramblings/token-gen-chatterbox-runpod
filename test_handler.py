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


class HandlerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runtime = FakeRuntime()

    def test_info(self) -> None:
        result = handler.handle_input({"action": "info"}, self.runtime)
        self.assertEqual(result["model"], "chatterbox-turbo")
        self.assertEqual(result["gpu"], "Fake GPU")
        self.assertTrue(result["voice_cloning"])

    @patch("handler._save_wave", fake_save_wave)
    def test_generate_with_builtin_voice(self) -> None:
        result = handler.handle_input(
            {"text": "Hello from Chatterbox.", "seed": 42}, self.runtime
        )
        self.assertEqual(base64.b64decode(result["audio_base64"]), b"RIFF-test-audio")
        self.assertEqual(result["duration_seconds"], 1.25)
        self.assertFalse(result["used_reference_voice"])
        self.assertEqual(self.runtime.calls[0][2]["seed"], 42)

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


if __name__ == "__main__":
    unittest.main()

