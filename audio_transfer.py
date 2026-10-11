"""Bounded, job-scoped HTTPS transport for native speech files."""
from __future__ import annotations

import hashlib
import hmac
import json
import re
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

CHUNK_BYTES = 8 * 1024 * 1024
MAX_AUDIO_BYTES = 768 * 1024 * 1024
MAX_REFERENCE_BYTES = 10 * 1024 * 1024
_BASE = re.compile(r"/api/tts/worker-transfers/(?:owner|user-[a-f0-9]{32})/[a-f0-9]{32}")
_TOKEN = re.compile(r"([0-9]{10})\.[a-f0-9]{64}")
_HASH = re.compile(r"[a-f0-9]{64}")


class TransferError(RuntimeError):
    """Safe transport failure; never includes credentials or response bodies."""


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def validate_transfer(value):
    if not isinstance(value, dict):
        raise TransferError("invalid_audio_transfer")
    url, token = value.get("url"), value.get("token")
    if not isinstance(url, str) or not isinstance(token, str):
        raise TransferError("invalid_audio_transfer")
    parsed = urlsplit(url)
    match = _TOKEN.fullmatch(token)
    if (parsed.scheme != "https" or parsed.netloc != "token-gen-api.owenonthenet.com"
            or parsed.query or parsed.fragment or not _BASE.fullmatch(parsed.path)
            or not match or int(match[1]) <= time.time()
            or type(value.get("chunk_bytes")) is not int or value["chunk_bytes"] != CHUNK_BYTES
            or type(value.get("max_bytes")) is not int or not 0 < value["max_bytes"] <= MAX_AUDIO_BYTES):
        raise TransferError("invalid_audio_transfer")
    return value


def _request(transfer, method, suffix, body=None, *, limit=4096, binary=False):
    opener = build_opener(NoRedirect())
    for attempt in range(3):
        validate_transfer(transfer)
        request = Request(transfer["url"] + suffix, method=method, data=body,
                          headers={"Authorization": "Bearer " + transfer["token"],
                                   "Content-Type": "application/octet-stream" if method == "PUT" else "application/json"})
        try:
            with opener.open(request, timeout=60) as response:
                if response.status != 200:
                    raise TransferError("audio_transfer_rejected")
                payload = response.read(limit + 1)
                if len(payload) > limit:
                    raise TransferError("audio_transfer_response_too_large")
                if binary:
                    return payload
                result = json.loads(payload)
                if not isinstance(result, dict) or result.get("ok") is not True:
                    raise TransferError("audio_transfer_invalid_ack")
                return result
        except HTTPError as exc:
            retryable = exc.code in (408, 429) or 500 <= exc.code <= 599
            exc.close()
            if not retryable or attempt == 2:
                raise TransferError("audio_transfer_http_failure") from None
        except (URLError, OSError):
            if attempt == 2:
                raise TransferError("audio_transfer_unavailable") from None
        except (ValueError, UnicodeError):
            raise TransferError("audio_transfer_invalid_ack") from None
        time.sleep(attempt + 1)
    raise TransferError("audio_transfer_unavailable")


def download_reference(data):
    reference = data.get("reference_download")
    if reference is None:
        return None
    transfer = validate_transfer(data.get("transfer"))
    if (not isinstance(reference, dict) or reference.get("url") != transfer["url"] + "/reference"
            or any(key in data for key in ("reference_audio", "reference_audio_base64"))
            or type(reference.get("bytes")) is not int or not 0 < reference["bytes"] <= MAX_REFERENCE_BYTES
            or reference.get("format") not in ("wav", "mp3", "flac", "ogg")
            or not isinstance(reference.get("sha256"), str) or not _HASH.fullmatch(reference["sha256"])):
        raise TransferError("invalid_reference_download")
    audio = _request(transfer, "GET", "/reference", limit=reference["bytes"], binary=True)
    if len(audio) != reference["bytes"] or not hmac.compare_digest(hashlib.sha256(audio).hexdigest(), reference["sha256"]):
        raise TransferError("reference_download_integrity_failed")
    return audio, "." + reference["format"]


def upload_audio(path, transfer, metadata):
    transfer = validate_transfer(transfer)
    size = path.stat().st_size
    if not 0 < size <= transfer["max_bytes"]:
        raise TransferError("audio_transfer_output_exceeds_budget")
    digest, count = hashlib.sha256(), 0
    with path.open("rb") as source:
        while chunk := source.read(CHUNK_BYTES):
            digest.update(chunk)
            ack = _request(transfer, "PUT", f"/chunks/{count}", chunk)
            if ack.get("index") != count or ack.get("bytes") != len(chunk):
                raise TransferError("audio_transfer_invalid_ack")
            count += 1
    manifest = {**metadata, "chunks": count, "bytes": size, "sha256": digest.hexdigest()}
    body = json.dumps(manifest, allow_nan=False, separators=(",", ":")).encode()
    if len(body) > 4096:
        raise TransferError("audio_transfer_manifest_too_large")
    ack = _request(transfer, "POST", "/complete", body)
    if ack.get("stored_on_nas") is not True or ack.get("sha256") != manifest["sha256"] or ack.get("bytes") != size:
        raise TransferError("audio_transfer_invalid_ack")
    return {**metadata, "stored_on_nas": True, "bytes": size, "sha256": manifest["sha256"]}
