"""Synthetic provider state verifies the owner-local release transaction."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).parent / "scripts"))
import release_native_audio as release


class LocalReleaseTests(unittest.TestCase):
    def test_cutover_preserves_provider_configuration_and_verifies_rollback(self):
        commit = "a" * 40
        old = release.IMAGE_REPOSITORY + "@sha256:" + "b" * 64
        new = release.IMAGE_REPOSITORY + "@sha256:" + "c" * 64
        before = {"endpoint": {"id": "synthetic-endpoint", "templateId": "synthetic-template",
                  "gpuTypeIds": ["synthetic-gpu"], "allowedCudaVersions": ["12.8"],
                  "cudaVersion": "12.8", "workersMin": 0, "workersMax": 1},
                  "template": {"id": "synthetic-template", "imageName": old,
                               "env": {"SYNTHETIC_VALUE": "private-fixture-value"}},
                  "health": {"jobs": {"inQueue": 0, "inProgress": 0}}}
        proof = {"image": new, "source_commit": commit}
        for fault in (None, "drift", "canary", "rollback"):
            state = copy.deepcopy(before)
            actions = []
            def bridge(action, details):
                actions.append(action)
                if action in ("patch_template_image", "restore_template_image"):
                    self.assertEqual(set(details["body"]), {"imageName"})
                    state["template"]["imageName"] = details["body"]["imageName"]
                    if action == "patch_template_image" and fault == "drift":
                        state["endpoint"]["workersMax"] = 2
                    if action == "restore_template_image":
                        if fault == "rollback":
                            raise OSError("simulated lost rollback response")
                        state["endpoint"]["workersMax"] = 1
                if action == "run_info_canary":
                    return {"status": "FAILED" if fault in ("canary", "rollback") else "COMPLETED", "id": "synthetic-job",
                            "output": {"model": "qwen3-tts-1.7b-base", "source_commit": commit,
                                       "default_quality_preset": "native", "default_wav_encoding": "float32",
                                       "default_mastering": "native", "cuda_preflight": "passed",
                                       "dtype": "bfloat16", "sample_rate": 24000}}
                return copy.deepcopy(state)
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                if fault:
                    with self.assertRaises(release.ReleaseError):
                        release.transaction(directory, "synthetic-endpoint", before, proof, {"conclusion": "success"}, bridge)
                else:
                    release.transaction(directory, "synthetic-endpoint", before, proof, {"conclusion": "success"}, bridge)
                receipt = release.private_json(directory / "receipt.json")
                self.assertEqual(receipt["status"], "complete" if fault is None else "rollback_required" if fault == "rollback" else "rolled_back")
                self.assertNotIn("private-fixture-value", (directory / "receipt.json").read_text())
                self.assertEqual((directory / "provider-before.json").stat().st_mode & 0o777, 0o600)
                self.assertEqual(release.private_json(directory / "provider-before.json"), before)
                if fault:
                    self.assertLess(actions.index("quiesce_canary"), actions.index("restore_template_image"))
        busy = copy.deepcopy(before)
        busy["health"]["jobs"]["inProgress"] = 1
        with tempfile.TemporaryDirectory() as temporary, self.assertRaises(release.ReleaseError):
            release.transaction(Path(temporary), "synthetic-endpoint", busy, proof, {},
                                lambda *_: self.fail("Busy endpoint must not reach MCP mutation"))
