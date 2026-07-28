from __future__ import annotations

import base64
from pathlib import Path
import unittest
from unittest.mock import patch

import qwen_handler


class FakeRuntime:
    sample_rate = 24_000
    device_name = "Fake GPU"
    dtype_name = "bfloat16"

    def __init__(self, variant="base") -> None:
        self.variant = variant
        self.calls = []

    def generate_clone(self, text, reference_path, reference_text, **options):
        self.calls.append(
            ("clone", text, reference_path, reference_text, options)
        )
        return object()

    def generate_design(self, text, voice_description, **options):
        self.calls.append(("design", text, voice_description, options))
        return object()


def fake_save_wave(_waveform, sample_rate: int, path: Path) -> float:
    assert sample_rate == 24_000
    path.write_bytes(b"RIFF-test-audio")
    return 1.0


def fake_encode_mp3(_wave_path: Path, mp3_path: Path) -> None:
    mp3_path.write_bytes(b"ID3-test-audio")


class QwenHandlerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runtime = FakeRuntime()
        master_patcher = patch("qwen_handler._master_wave")
        self.master_wave = master_patcher.start()
        self.addCleanup(master_patcher.stop)

    def test_info_describes_exact_transcript_cloning(self) -> None:
        result = qwen_handler.handle_input({"action": "info"}, self.runtime)
        self.assertEqual(result["model"], "qwen3-tts-1.7b-base")
        self.assertTrue(result["voice_cloning"])
        self.assertTrue(result["exact_transcript_cloning"])
        self.assertFalse(result["built_in_voice"])
        self.assertFalse(result["watermarked"])

    @patch("qwen_handler._save_wave", fake_save_wave)
    def test_clone_uses_reference_text(self) -> None:
        result = qwen_handler.handle_input(
            {
                "text": "A natural clone.",
                "reference_audio": base64.b64encode(b"voice").decode(),
                "reference_text": "These are the exact spoken words.",
                "seed": 17,
            },
            self.runtime,
        )
        call = self.runtime.calls[0]
        self.assertEqual(call[0], "clone")
        self.assertEqual(call[3], "These are the exact spoken words.")
        self.assertFalse(call[4]["x_vector_only_mode"])
        self.assertEqual(result["clone_mode"], "exact_transcript")
        self.assertTrue(result["reference_text_used"])
        self.master_wave.assert_called_once()

    @patch("qwen_handler._save_wave", fake_save_wave)
    def test_legacy_reference_uses_reduced_quality_embedding_mode(self) -> None:
        result = qwen_handler.handle_input(
            {
                "text": "A compatible clone.",
                "reference_audio": base64.b64encode(b"voice").decode(),
            },
            self.runtime,
        )
        self.assertTrue(self.runtime.calls[0][4]["x_vector_only_mode"])
        self.assertEqual(result["clone_mode"], "speaker_embedding")
        self.assertFalse(result["reference_text_used"])

    @patch("qwen_handler._encode_mp3", fake_encode_mp3)
    @patch("qwen_handler._save_wave", fake_save_wave)
    def test_mp3_output(self) -> None:
        result = qwen_handler.handle_input(
            {
                "text": "Return MP3.",
                "reference_audio": base64.b64encode(b"voice").decode(),
                "output_format": "mp3",
            },
            self.runtime,
        )
        self.assertEqual(
            base64.b64decode(result["audio_base64"]), b"ID3-test-audio"
        )
        self.assertEqual(result["mime_type"], "audio/mpeg")

    @patch("qwen_handler._save_wave", fake_save_wave)
    def test_voice_design_worker(self) -> None:
        runtime = FakeRuntime("design")
        result = qwen_handler.handle_input(
            {
                "action": "design",
                "text": "This becomes the reference.",
                "voice_description": "A natural adult woman speaking English.",
            },
            runtime,
        )
        self.assertEqual(runtime.calls[0][0], "design")
        self.assertEqual(result["model_variant"], "design")
        self.assertIsNone(result["clone_mode"])

    def test_rejects_invalid_inputs(self) -> None:
        invalid = [
            {},
            {"text": ""},
            {"text": "hello"},
            {
                "text": "hello",
                "reference_audio": base64.b64encode(b"voice").decode(),
                "reference_text": "",
                "x_vector_only_mode": False,
            },
            {
                "text": "hello",
                "reference_audio": "not-base64",
            },
            {
                "text": "hello",
                "reference_audio": base64.b64encode(b"voice").decode(),
                "output_format": "flac",
            },
        ]
        for value in invalid:
            with self.subTest(value=value):
                with self.assertRaises(qwen_handler.InputError):
                    qwen_handler.handle_input(value, self.runtime)

    @patch("qwen_handler._save_wave", fake_save_wave)
    def test_reference_file_is_removed(self) -> None:
        seen_path = None

        def capture(text, reference_path, reference_text, **options):
            nonlocal seen_path
            seen_path = reference_path
            return object()

        self.runtime.generate_clone = capture
        qwen_handler.handle_input(
            {
                "text": "Temporary reference.",
                "reference_audio": base64.b64encode(b"voice").decode(),
            },
            self.runtime,
        )
        self.assertIsNotNone(seen_path)
        self.assertFalse(Path(seen_path).exists())


if __name__ == "__main__":
    unittest.main()
