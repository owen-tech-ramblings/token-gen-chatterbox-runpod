"""Offline transfer tests: synthetic samples, fake HTTPS, no provider jobs."""
import base64
import hashlib
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

import numpy as np
import soundfile

import qwen_handler
from test_native_audio import cases


def capability():
    return {"url": "https://token-gen-api.owenonthenet.com/api/tts/worker-transfers/owner/" + "a" * 32,
            "token": str(int(time.time()) + 21600) + "." + "b" * 64,
            "chunk_bytes": 8 * 1024 * 1024, "max_bytes": 768 * 1024 * 1024}


class AudioTransferTests(unittest.TestCase):
    def test_transfer_preserves_native_audio_beyond_inline_limit_for_all_modes(self):
        samples = np.resize(np.array([0., -0., 2**-24, .123456789, 1.25, -1.125], np.float32), 60000)
        for module, runtime, request in cases(samples):
            def upload(path, transfer, metadata):
                actual, rate = soundfile.read(path, dtype="float32")
                np.testing.assert_array_equal(actual.view(np.uint32), samples.view(np.uint32))
                self.assertEqual(rate, 24000)
                self.assertTrue(metadata["native_audio_preserved"])
                self.assertNotIn("audio_base64", metadata)
                return {**metadata, "stored_on_nas": True, "bytes": path.stat().st_size,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            with self.subTest(mode=request["text"]), patch.object(module, "MAX_AUDIO_BYTES", 100), patch.object(module, "upload_audio", side_effect=upload, create=True) as sink:
                result = module.handle_input({**request, "transfer": capability()}, runtime)
                self.assertTrue(result["stored_on_nas"])
                self.assertNotIn("audio_base64", result)
                sink.assert_called_once()
        # A real waveform above the old 14 MiB limit uses the same path.
        samples = np.resize(samples, 4_000_000)
        module, runtime, request = next(case for case in cases(samples) if case[0] is qwen_handler)
        with patch.object(module, "upload_audio", side_effect=upload, create=True):
            result = module.handle_input({**request, "transfer": capability()}, runtime)
        self.assertGreater(result["bytes"], 14 * 1024 * 1024)

    def test_authenticated_chunks_replay_and_reference_integrity_fail_closed(self):
        import audio_transfer as transfer
        cap = capability()
        contents = b"original-float-wave-bytes" * 400000
        received, requests = {}, []
        reference = b"original-lossless-reference"
        reference_info = {"url": cap["url"] + "/reference", "format": "wav",
                          "bytes": len(reference), "sha256": hashlib.sha256(reference).hexdigest()}
        class Response(io.BytesIO):
            status = 200
        class Opener:
            def open(inner, request, timeout):
                requests.append(request)
                self.assertEqual(request.get_header("Authorization"), "Bearer " + cap["token"])
                self.assertEqual(timeout, 60)
                if request.method == "GET":
                    return Response(reference)
                if request.method == "PUT":
                    index = int(request.full_url.rsplit("/", 1)[1])
                    if index in received:
                        self.assertEqual(received[index], request.data)
                    received[index] = request.data
                    if len(requests) == 1:
                        raise URLError("simulated lost ACK")
                    return Response(json.dumps({"ok": True, "index": index, "bytes": len(request.data)}).encode())
                manifest = json.loads(request.data)
                assembled = b"".join(received[i] for i in range(manifest["chunks"]))
                self.assertEqual(assembled, contents)
                self.assertEqual(manifest["sha256"], hashlib.sha256(assembled).hexdigest())
                return Response(json.dumps({"ok": True, "stored_on_nas": True,
                                           "sha256": manifest["sha256"], "bytes": len(assembled)}).encode())
        with tempfile.TemporaryDirectory() as directory, patch.object(transfer, "build_opener", return_value=Opener()), patch.object(transfer.time, "sleep"):
            path = Path(directory) / "native.wav"
            path.write_bytes(contents)
            result = transfer.upload_audio(path, cap, {"output_format": "wav", "native_audio_preserved": True})
            self.assertTrue(result["stored_on_nas"])
            self.assertEqual([r.full_url for r in requests[:2]], [cap["url"] + "/chunks/0"] * 2)
            self.assertEqual(transfer.download_reference({"transfer": cap, "reference_download": reference_info}), (reference, ".wav"))
            with self.assertRaises(transfer.TransferError):
                transfer.download_reference({"transfer": cap, "reference_download": {**reference_info, "sha256": "0" * 64}})
            for invalid in ({**cap, "url": cap["url"].replace("https:", "http:")},
                            {**cap, "url": cap["url"] + "?token=bad"},
                            {**cap, "url": cap["url"].replace("token-gen-api.owenonthenet.com", "example.org")},
                            {**cap, "token": "1000000000." + "b" * 64}):
                with self.subTest(invalid=invalid["url"]), self.assertRaises(transfer.TransferError):
                    transfer.upload_audio(path, invalid, {})
            with self.assertRaises(transfer.TransferError):
                transfer.download_reference({"transfer": cap, "reference_download": {**reference_info, "url": cap["url"] + "/other"}})
        # Redirects cannot forward a capability; permanent rejections are
        # neither followed nor retried, and provider bodies remain private.
        self.assertIsNone(transfer.NoRedirect().redirect_request(None, None, 302, "", {}, "https://example.org"))
        class Reject:
            def open(inner, *args, **kwargs):
                raise HTTPError("private", 302, "private", {}, io.BytesIO(b"private body"))
        with patch.object(transfer, "build_opener", return_value=Reject()), patch.object(transfer.time, "sleep") as sleep:
            with self.assertRaisesRegex(transfer.TransferError, "^audio_transfer_http_failure$"):
                transfer.download_reference({"transfer": cap, "reference_download": reference_info})
            sleep.assert_not_called()
