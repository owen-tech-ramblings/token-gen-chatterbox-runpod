"""Synthetic provider state verifies the owner-local release transaction."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent / "scripts"))
import release_native_audio as release


class LocalReleaseTests(unittest.TestCase):
    def test_cutover_preserves_provider_configuration_and_verifies_rollback(self):
        commit = "a" * 40
        old = release.IMAGE_REPOSITORY + "@sha256:" + "b" * 64
        new = release.IMAGE_REPOSITORY + "@sha256:" + "c" * 64
        # Shape matches current v2 get_endpoint, with synthetic values only.
        before = {"endpoint": {"id": "synthetic-endpoint", "image": old,
                  "args": "synthetic-private-args", "createdAt": "2026-01-01T00:00:00Z",
                  "dataCenterIds": [], "disk": 50, "flashboot": "FLASHBOOT",
                  "gpu": {"pools": ["synthetic-pool"], "count": 1,
                          "allowedCudaVersions": ["12.8"], "minCudaVersion": "12.8"},
                  "name": "synthetic", "networkVolumes": [], "requestUrls": {},
                  "scaling": {"type": "QUEUE_DELAY", "queueDelay": 1},
                  "timeout": 600000, "type": "QUEUE",
                  "workers": {"min": 0, "max": 1, "idleTimeout": 5}},
                  "health": {"jobs": {"inQueue": 0, "inProgress": 0}}}
        proof = {"image": new, "source_commit": commit}
        for fault in (None, "drift", "canary", "rollback"):
            state = copy.deepcopy(before)
            actions = []
            def bridge(action, details):
                actions.append(action)
                if action in ("patch_endpoint_image", "restore_endpoint_image"):
                    self.assertEqual(set(details["body"]), {"image"})
                    self.assertEqual(details["id"], "synthetic-endpoint")
                    state["endpoint"]["image"] = details["body"]["image"]
                    if action == "patch_endpoint_image" and fault == "drift":
                        state["endpoint"]["workers"]["max"] = 2
                    if action == "restore_endpoint_image":
                        if fault == "rollback":
                            raise OSError("simulated lost rollback response")
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
                self.assertEqual(receipt["status"], "complete" if fault is None else "rollback_required" if fault in ("rollback", "drift") else "rolled_back")
                self.assertNotIn("synthetic-private-args", (directory / "receipt.json").read_text())
                self.assertEqual((directory / "provider-before.json").stat().st_mode & 0o777, 0o600)
                self.assertEqual(release.private_json(directory / "provider-before.json"), before)
                if fault:
                    self.assertLess(actions.index("quiesce_canary"), actions.index("restore_endpoint_image"))
        for key in before["endpoint"]:
            if key == "image":
                continue
            changed = copy.deepcopy(before)
            changed["endpoint"][key] = "synthetic-drift"
            with self.subTest(protected_key=key), self.assertRaises(release.ReleaseError):
                release.readback_matches(before, changed, "synthetic-endpoint", old)
        future_setting = copy.deepcopy(before)
        future_setting["endpoint"]["futureSetting"] = "preserve-this-too"
        changed = copy.deepcopy(future_setting)
        changed["endpoint"]["futureSetting"] = "changed"
        with self.assertRaises(release.ReleaseError):
            release.readback_matches(future_setting, changed, "synthetic-endpoint", old)
        busy = copy.deepcopy(before)
        busy["health"]["jobs"]["inProgress"] = 1
        with tempfile.TemporaryDirectory() as temporary, self.assertRaises(release.ReleaseError):
            release.transaction(Path(temporary), "synthetic-endpoint", busy, proof, {},
                                lambda *_: self.fail("Busy endpoint must not reach MCP mutation"))
        with tempfile.TemporaryDirectory() as temporary:
            nested = Path(temporary) / "state" / "receipts"
            with patch.object(release.os, "fsync", side_effect=OSError("simulated directory barrier failure")), self.assertRaises(OSError):
                release.durable_mkdir(nested)
            self.assertTrue(nested.is_dir())
            real_fsync = release.os.fsync
            with patch.object(release.os, "fsync", wraps=real_fsync) as barriers:
                release.durable_mkdir(nested)
            self.assertGreaterEqual(barriers.call_count, 3)
