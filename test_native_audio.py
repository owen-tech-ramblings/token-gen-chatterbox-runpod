"""Synthetic, offline tests of the actual output bytes across both workers."""
import base64
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile

import handler
import qwen_handler
from test_handler import FakeRuntime as ChatterboxRuntime
from test_qwen_handler import FakeRuntime as QwenRuntime


def cases(samples):
    reference = base64.b64encode(b"synthetic-reference").decode()
    for module, runtime, request in (
        (handler, ChatterboxRuntime(), {"text": "Unchanged authored text."}),
        (handler, ChatterboxRuntime(), {"text": "Reference voice.", "reference_audio": reference}),
        (qwen_handler, QwenRuntime(), {"text": "Exact clone.", "reference_audio": reference,
                                      "reference_text": "Exact reference transcript."}),
        (qwen_handler, QwenRuntime(), {"text": "Embedding clone.", "reference_audio": reference}),
        (qwen_handler, QwenRuntime("design"), {"text": "Designed voice.", "voice_description": "A clear voice."}),
    ):
        calls = runtime.calls
        def generate(*args, _calls=calls, **kwargs):
            _calls.append((args, kwargs))
            return samples.copy()
        runtime.generate = runtime.generate_clone = runtime.generate_design = generate
        yield module, runtime, request


class NativeAudioTests(unittest.TestCase):
    def test_native_wav_preserves_samples_for_all_modes(self):
        # More precision than PCM16, signed zero, and headroom outside +/-1.
        samples = np.resize(np.array([0, -0., 2**-24, .123456789, 1.25, -1.125], np.float32), 60000)
        for module, runtime, request in cases(samples):
            for preset in (None, "native", "publication", "balanced", "expressive"):
                with self.subTest(worker=module.__name__, request=request["text"], preset=preset):
                    options = {} if preset is None else {"quality_preset": preset}
                    # Native output requires no subprocess or mastering dependency.
                    with patch.object(module.subprocess, "run", side_effect=AssertionError("Unexpected processing")):
                        result = module.handle_input({**request, **options}, runtime)
                    audio = base64.b64decode(result["audio_base64"], validate=True)
                    actual, rate = soundfile.read(io.BytesIO(audio), dtype="float32")
                    np.testing.assert_array_equal(actual.view(np.uint32), samples.view(np.uint32))
                    self.assertEqual(soundfile.info(io.BytesIO(audio)).subtype, "FLOAT")
                    self.assertEqual((rate, result["sample_rate"]), (24000, 24000))
                    self.assertEqual(result["duration_seconds"], 2.5)
                    self.assertEqual(result["sha256"], hashlib.sha256(audio).hexdigest())
                    self.assertEqual(result["audio_encoding"], "pcm_f32le")
                    self.assertEqual(result["wav_encoding"], "float32")
                    self.assertEqual(result["bit_depth"], 32)
                    self.assertIsNone(result["mastering"])
                    self.assertTrue(result["native_audio_preserved"])
                    self.assertEqual(result["quality_preset"], preset or "native")
                    self.assertEqual(runtime.calls[-1][0][0], request["text"])
                    if preset in (None, "native"):
                        expected = (.8, .95, 1000, 1.2) if module is handler else (.9, 1., 50, 1.05)
                        self.assertEqual(tuple(runtime.calls[-1][1][key] for key in
                                         ("temperature", "top_p", "top_k", "repetition_penalty")), expected)
            info = module.handle_input({"action": "info"}, runtime)
            self.assertEqual(info["default_quality_preset"], "native")
            self.assertEqual(info["default_wav_encoding"], "float32")
            self.assertEqual(info["default_mastering"], "native")

    def test_explicit_encodings_and_mastering_remain_available(self):
        samples = (.031234567 * np.sin(np.arange(96000, dtype=np.float32) * .2)).astype(np.float32)
        samples[0:2] = (1.25, -1.125)
        for module, runtime, request in cases(samples):
            with self.subTest(worker=module.__name__, request=request["text"]):
                pcm = module.handle_input({**request, "wav_encoding": "pcm16"}, runtime)
                data = base64.b64decode(pcm["audio_base64"])
                actual, rate = soundfile.read(io.BytesIO(data), dtype="float32")
                self.assertEqual(soundfile.info(io.BytesIO(data)).subtype, "PCM_16")
                np.testing.assert_allclose(actual, np.clip(samples, -1, 1 - 2**-15), atol=2**-15, rtol=0)
                self.assertEqual((pcm["audio_encoding"], pcm["bit_depth"], pcm["wav_encoding"]), ("pcm_s16le", 16, "pcm16"))
                self.assertFalse(pcm["native_audio_preserved"])
                self.assertIsNone(pcm["mastering"])

                mastered = module.handle_input({**request, "mastering": "podcast_mono_-19_lufs"}, runtime)
                data = base64.b64decode(mastered["audio_base64"])
                actual, rate = soundfile.read(io.BytesIO(data), dtype="float32")
                self.assertEqual((soundfile.info(io.BytesIO(data)).subtype, rate), ("FLOAT", 24000))
                self.assertFalse(np.array_equal(actual, samples))
                self.assertTrue(np.isfinite(actual).all())
                self.assertEqual(mastered["mastering"], "podcast_mono_-19_lufs")
                self.assertFalse(mastered["native_audio_preserved"])

                mp3 = module.handle_input({**request, "output_format": "mp3"}, runtime)
                data = base64.b64decode(mp3["audio_base64"])
                self.assertEqual(soundfile.info(io.BytesIO(data)).format, "MP3")
                self.assertEqual((mp3["audio_encoding"], mp3["mime_type"]), ("mp3", "audio/mpeg"))
                self.assertIsNone(mp3["wav_encoding"])
                self.assertIsNone(mp3["bit_depth"])
                self.assertFalse(mp3["native_audio_preserved"])

    def test_audio_validation_and_inline_limit_fail_without_fallback(self):
        for module, runtime, request in cases(np.ones(24000, np.float32)):
            for invalid in ({"wav_encoding": "pcm24"}, {"wav_encoding": True},
                            {"mastering": "automatic"}, {"mastering": True},
                            {"output_format": "mp3", "wav_encoding": "pcm16"}):
                with self.subTest(worker=module.__name__, invalid=invalid):
                    with self.assertRaises(module.InputError):
                        module.handle_input({**request, **invalid}, runtime)
                    self.assertEqual(runtime.calls, [])
            with patch.object(module, "MAX_AUDIO_BYTES", 100):
                with self.assertRaisesRegex(RuntimeError, "too large"):
                    module.handle_input(request, runtime)
            self.assertEqual(len(runtime.calls), 1)
        for module in (handler, qwen_handler):
            for invalid in (np.array([], np.float32), np.ones((2, 3), np.float32),
                            np.array([0, np.nan]), np.array([0, np.inf])):
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "invalid.wav"
                    with self.assertRaises(RuntimeError):
                        module._save_wave(invalid, 24000, path)
                    self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
