"""Read-only immutable image proof; credentials remain only in process memory."""
import base64
from datetime import datetime, timezone
import hashlib
import json
import re
import subprocess
from urllib.error import HTTPError
from urllib.parse import urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

REPOSITORY = "owen-tech-ramblings/token-gen-chatterbox-runpod"
IMAGE_REPOSITORY = "ghcr.io/" + REPOSITORY
ACCEPT = ", ".join(("application/vnd.oci.image.index.v1+json", "application/vnd.oci.image.manifest.v1+json",
                    "application/vnd.docker.distribution.manifest.list.v2+json", "application/vnd.docker.distribution.manifest.v2+json"))


class ProofError(RuntimeError):
    pass


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def _request(url, *, token=None, method="GET", limit=2 * 1024 * 1024, basic=False):
    headers = {"Accept": ACCEPT, "User-Agent": "Token-Gen-Image-Proof/1.0"}
    if token:
        headers["Authorization"] = ("Basic " if basic else "Bearer ") + token
    opener = build_opener(NoRedirect())
    for _ in range(4):
        try:
            with opener.open(Request(url, method=method, headers=headers), timeout=60) as response:
                body = response.read(limit + 1) if method != "HEAD" else b""
                if len(body) > limit:
                    raise ProofError("Registry document exceeds bound")
                return response.status, dict(response.headers), body
        except HTTPError as exc:
            if exc.code not in (301, 302, 303, 307, 308):
                code = exc.code
                exc.close()
                raise ProofError(f"Registry request failed with HTTP {code}") from None
            target = exc.headers.get("Location", "")
            exc.close()
            parsed = urlsplit(target)
            if parsed.scheme != "https" or parsed.username or parsed.password or not parsed.hostname:
                raise ProofError("Unsafe registry redirect")
            # Blob redirects may carry a signed storage URL. Never forward
            # registry credentials to it or write that URL into a receipt.
            headers.pop("Authorization", None)
            url = target
    raise ProofError("Registry redirect limit exceeded")


def _digest(body):
    return "sha256:" + hashlib.sha256(body).hexdigest()


def verify_image(commit):
    if not re.fullmatch(r"[a-f0-9]{40}", commit):
        raise ProofError("Invalid source commit")
    credential = subprocess.run(["gh", "auth", "token"], check=True, capture_output=True, text=True).stdout.strip()
    if not credential:
        raise ProofError("Existing GitHub authentication is unavailable")
    basic = base64.b64encode((REPOSITORY.split("/", 1)[0] + ":" + credential).encode()).decode()
    _, _, body = _request("https://ghcr.io/token?" + urlencode({"service": "ghcr.io", "scope": "repository:" + REPOSITORY + ":pull"}), token=basic, basic=True)
    token = json.loads(body)["token"]
    base = "https://ghcr.io/v2/" + REPOSITORY
    tag = "qwen3-base-" + commit
    _, headers, body = _request(base + "/manifests/" + tag, token=token)
    top_digest = _digest(body)
    reported = next((v for k, v in headers.items() if k.lower() == "docker-content-digest"), None)
    if reported != top_digest:
        raise ProofError("Tagged manifest checksum differs")
    _, _, immutable = _request(base + "/manifests/" + top_digest, token=token)
    if immutable != body:
        raise ProofError("Immutable manifest differs from tag")
    manifest = json.loads(body)
    media_type = manifest.get("mediaType")
    platform_digest = top_digest
    if "manifests" in manifest:
        matches = [item for item in manifest["manifests"] if item.get("platform", {}).get("os") == "linux" and item.get("platform", {}).get("architecture") == "amd64"]
        if len(matches) != 1:
            raise ProofError("Expected exactly one Linux amd64 image")
        descriptor = matches[0]
        platform_digest = descriptor["digest"]
        _, _, body = _request(base + "/manifests/" + platform_digest, token=token)
        if _digest(body) != platform_digest or len(body) != descriptor["size"]:
            raise ProofError("Platform manifest checksum differs")
        manifest = json.loads(body)
    config_descriptor = manifest["config"]
    _, _, body = _request(base + "/blobs/" + config_descriptor["digest"], token=token)
    if _digest(body) != config_descriptor["digest"] or len(body) != config_descriptor["size"]:
        raise ProofError("Image configuration checksum differs")
    config = json.loads(body)
    if config.get("os") != "linux" or config.get("architecture") != "amd64":
        raise ProofError("Unexpected image platform")
    revision = config.get("config", {}).get("Labels", {}).get("org.opencontainers.image.revision")
    if revision != commit:
        raise ProofError("Image source revision differs")
    descriptors = [config_descriptor, *manifest["layers"]]
    if not 1 <= len(descriptors) <= 100:
        raise ProofError("Unexpected image blob count")
    proofs = []
    for descriptor in descriptors:
        digest, size = descriptor["digest"], descriptor["size"]
        if not re.fullmatch(r"sha256:[a-f0-9]{64}", digest) or type(size) is not int or size < 0:
            raise ProofError("Invalid image blob descriptor")
        status, headers, _ = _request(base + "/blobs/" + digest, token=token, method="HEAD")
        length = next((v for k, v in headers.items() if k.lower() == "content-length"), None)
        if status != 200 or length is None or int(length) != size:
            raise ProofError("Image blob pull availability or size differs")
        proofs.append({"digest": digest, "bytes": size, "status": status, "content_length": int(length)})
    return {"schema": 1, "verified_at": datetime.now(timezone.utc).isoformat(),
            "repository": REPOSITORY, "source_commit": commit, "immutable_tag": tag,
            "image": IMAGE_REPOSITORY + "@" + top_digest, "top_level_digest": top_digest,
            "top_level_media_type": media_type, "platform_manifest_digest": platform_digest,
            "platform": "linux/amd64", "revision_label": revision,
            "layer_count": len(manifest["layers"]), "blob_count": len(proofs),
            "compressed_layer_bytes": sum(item["size"] for item in manifest["layers"]),
            "authenticated_blob_heads": proofs,
            "scope": "Manifest/config hashes and all authenticated blob HEAD lengths verified; layer payloads not downloaded or rehashed; no GPU inference."}
