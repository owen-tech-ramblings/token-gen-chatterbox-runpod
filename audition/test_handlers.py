import base64
import importlib.util
import os
from pathlib import Path
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ProxyHandlerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load("audition_proxy", "server_proxy_handler.py")

    def request(self, **overrides):
        value = {
            "text": "Hello there.",
            "reference_audio": {
                "base64": base64.b64encode(b"RIFF-test").decode(),
                "content_type": "audio/wav",
            },
            "reference_text": "Reference words.",
            "delivery": "warm",
            "strength": 0.65,
            "seed": 42,
        }
        value.update(overrides)
        return {"input": value}

    def test_validates_common_contract(self):
        result = self.module._validate_request(self.request())
        self.assertEqual(result["delivery"], "warm")
        self.assertEqual(result["reference_audio"], b"RIFF-test")

    def test_rejects_unknown_delivery(self):
        with self.assertRaises(self.module.InputError):
            self.module._validate_request(self.request(delivery="theatrical"))

    def test_zonos_controls_keep_accurate_mode(self):
        self.assertIn("happy", self.module.ZONOS_EMOTIONS["warm"])
        self.assertEqual(self.module.ZONOS_EMOTIONS["neutral"], {})
        request = self.module._validate_request(self.request(delivery="neutral"))
        payload = self.module._zonos_payload(request, "reference-data")
        self.assertIsNone(payload["quality_buckets"])
        self.assertEqual(payload["quality_values"]["trailing_silence_s"], 0.4)

    def test_higgs_uses_base_image_virtualenv_and_allows_temp_references(self):
        with patch.dict(os.environ, {"AUDITION_BACKEND": "higgs"}):
            module = load("audition_proxy_higgs", "server_proxy_handler.py")
        command, _, _ = module._server_configuration()
        self.assertEqual(command[0], "/opt/omni/bin/sgl-omni")
        self.assertIn("--allowed-local-media-path", command)
        self.assertIn("/tmp", command)
        self.assertEqual(module.HIGGS_MAX_NEW_TOKENS, 320)


class IndexHandlerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load("audition_index", "index_handler.py")

    def test_vectors_use_published_order_and_safe_sum(self):
        for vector in self.module.EMOTION_VECTORS.values():
            self.assertEqual(len(vector), 8)
            self.assertLessEqual(sum(vector), 0.8)

    def test_info_does_not_load_model(self):
        result = self.module.handler({"input": {"action": "info"}})
        self.assertEqual(result["model"], "IndexTeam/IndexTTS-2")
        self.assertTrue(result["isolated_audition_worker"])


class ContainerDefinitionTests(unittest.TestCase):
    def test_higgs_installs_worker_deps_into_omni_virtualenv(self):
        dockerfile = (ROOT / "Dockerfile.higgs").read_text(encoding="utf-8")
        self.assertIn("uv pip install --python /opt/omni/bin/python", dockerfile)

    def test_zonos_image_includes_cuda_compiler_for_jit_kernels(self):
        dockerfile = (ROOT / "Dockerfile.zonos2").read_text(encoding="utf-8")
        self.assertIn("cudnn-devel-ubuntu24.04", dockerfile)


if __name__ == "__main__":
    unittest.main()
