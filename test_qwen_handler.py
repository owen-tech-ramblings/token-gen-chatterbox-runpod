from __future__ import annotations

import base64
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import qwen_handler


class FakeRuntime:
    sample_rate = 24_000
    device_name = "Fake GPU"
    dtype_name = "bfloat16"
    compute_capability = "sm_120"
    torch_version = "2.11.0+cu128"
    cuda_runtime = "12.8"
    compiled_cuda_architectures = [
        "sm_80",
        "sm_86",
        "sm_90",
        "sm_100",
        "sm_120",
    ]
    supported_compute_capabilities = [
        "sm_80",
        "sm_86",
        "sm_89",
        "sm_90",
        "sm_100",
        "sm_120",
    ]
    cuda_preflight = "passed"

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


def fake_torch_module(capability: tuple[int, int], events: list[str]) -> ModuleType:
    module = ModuleType("torch")

    class FakeProbe:
        def add_(self, _value):
            events.append("cuda-op")
            return self

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def get_device_name(_index: int) -> str:
            return "NVIDIA RTX PRO 6000 Blackwell"

        @staticmethod
        def get_device_capability(_index: int) -> tuple[int, int]:
            return capability

        @staticmethod
        def get_arch_list() -> list[str]:
            return ["sm_80", "sm_86", "sm_90", "sm_100", "sm_120"]

        @staticmethod
        def synchronize(_index: int = 0) -> None:
            events.append("synchronize")

        @staticmethod
        def is_bf16_supported() -> bool:
            return True

    def empty(_size: int, *, device: str):
        assert device == "cuda:0"
        events.append("allocate")
        return FakeProbe()

    module.cuda = FakeCuda()
    module.empty = empty
    module.version = SimpleNamespace(cuda="12.8")
    module.__version__ = "2.11.0+cu128"
    module.bfloat16 = object()
    module.float16 = object()
    return module


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
        self.assertEqual(result["compute_capability"], "sm_120")
        self.assertEqual(result["torch_version"], "2.11.0+cu128")
        self.assertEqual(result["cuda_runtime"], "12.8")
        self.assertEqual(
            result["compiled_cuda_architectures"],
            ["sm_80", "sm_86", "sm_90", "sm_100", "sm_120"],
        )
        self.assertIn("sm_89", result["supported_compute_capabilities"])
        self.assertEqual(result["cuda_preflight"], "passed")

    def test_blackwell_cuda_preflight_precedes_model_loading(self) -> None:
        events: list[str] = []
        torch_module = fake_torch_module((12, 0), events)
        qwen_module = ModuleType("qwen_tts")

        class FakeModel:
            @classmethod
            def from_pretrained(cls, *_args, **_kwargs):
                events.append("model-load")
                return object()

        qwen_module.Qwen3TTSModel = FakeModel
        with patch.dict(
            sys.modules,
            {"torch": torch_module, "qwen_tts": qwen_module},
        ):
            runtime = qwen_handler.QwenRuntime()

        self.assertEqual(runtime.compute_capability, "sm_120")
        self.assertEqual(runtime.cuda_preflight, "passed")
        self.assertEqual(
            events,
            ["allocate", "cuda-op", "synchronize", "model-load"],
        )

        ada_events: list[str] = []
        ada_preflight = qwen_handler._cuda_preflight(
            fake_torch_module((8, 9), ada_events)
        )
        self.assertEqual(ada_preflight["compute_capability"], "sm_89")
        self.assertEqual(ada_preflight["cuda_preflight"], "passed")
        self.assertEqual(ada_events, ["allocate", "cuda-op", "synchronize"])

    def test_unsupported_gpu_fails_before_model_initialization(self) -> None:
        events: list[str] = []
        torch_module = fake_torch_module((7, 5), events)
        qwen_module = ModuleType("qwen_tts")

        class UnexpectedModel:
            @classmethod
            def from_pretrained(cls, *_args, **_kwargs):
                events.append("model-load")
                return object()

        qwen_module.Qwen3TTSModel = UnexpectedModel
        with patch.dict(
            sys.modules,
            {"torch": torch_module, "qwen_tts": qwen_module},
        ):
            with self.assertRaisesRegex(RuntimeError, "sm_75.*not supported"):
                qwen_handler.QwenRuntime()

        self.assertEqual(events, [])

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
