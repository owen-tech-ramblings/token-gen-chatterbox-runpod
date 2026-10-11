#!/usr/bin/env python3
"""Owner-local release transaction; the parent agent performs authenticated MCP calls.

This process holds its release lock while emitting bounded JSON actions. Reply
on stdin with {"path": "/private/mode600/provider-or-canary.json"}. It never
accepts an API credential or performs a provider mutation itself.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import uuid

from registry_proof import IMAGE_REPOSITORY, REPOSITORY, verify_image


class ReleaseError(RuntimeError):
    pass


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def checksum(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def durable_mkdir(path):
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    current, ancestors = path.resolve(), []
    while True:
        ancestors.append(current)
        if current == current.parent or current.is_mount():
            break
        current = current.parent
    for directory in reversed(ancestors):
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def private_json(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077 or info.st_size > 1024 * 1024:
            raise ReleaseError("Evidence must be an owner-only bounded regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as source:
            value = json.load(source)
        if not isinstance(value, dict):
            raise ReleaseError("Evidence must contain an object")
        return value
    finally:
        os.close(descriptor)


def durable_json(path, value):
    temporary = path.with_name(path.name + ".new")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as output:
            output.write(canonical(value) + b"\n")
            output.flush()
            os.fsync(descriptor)
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        os.close(descriptor)
        temporary.unlink(missing_ok=True)


def validate_snapshot(snapshot, endpoint_id, *, idle=False):
    endpoint, template = snapshot.get("endpoint"), snapshot.get("template")
    if not isinstance(endpoint, dict) or not isinstance(template, dict):
        raise ReleaseError("Provider snapshot must contain full endpoint and template objects")
    if endpoint.get("id") != endpoint_id or endpoint.get("templateId") != template.get("id") or not template.get("id"):
        raise ReleaseError("Provider snapshot identity differs")
    image = template.get("imageName")
    if not isinstance(image, str) or not image.startswith(IMAGE_REPOSITORY + "@sha256:") or not re.fullmatch(r"sha256:[a-f0-9]{64}", image.rsplit("@", 1)[1]):
        raise ReleaseError("Provider image must be an immutable image in the reviewed repository")
    if not isinstance(template.get("env"), dict):
        raise ReleaseError("Snapshot is missing the complete template environment")
    for key in ("gpuTypeIds", "allowedCudaVersions", "cudaVersion", "workersMin", "workersMax"):
        if key not in endpoint:
            raise ReleaseError("Snapshot is missing GPU, CUDA or scaling configuration")
    if idle:
        jobs = snapshot.get("health", {}).get("jobs", {})
        if any(type(jobs.get(key)) is not int or jobs[key] != 0 for key in ("inQueue", "inProgress")):
            raise ReleaseError("Endpoint must have no queued or active jobs before image cutover")
    return image


def protected_settings(snapshot):
    # Provider timestamps may change after PATCH. Every other returned setting
    # must survive exactly, including environment values and scaling policy.
    ignored = {"updatedAt", "lastUpdatedAt"}
    endpoint = {k: v for k, v in snapshot["endpoint"].items() if k not in ignored}
    template = {k: v for k, v in snapshot["template"].items() if k not in ignored | {"imageName"}}
    return {"endpoint": endpoint, "template": template}


def readback_matches(before, after, endpoint_id, image):
    actual = validate_snapshot(after, endpoint_id)
    if actual != image or protected_settings(after) != protected_settings(before):
        raise ReleaseError("Image readback or protected provider settings differ")


def validate_canary(value, commit):
    if value.get("status") != "COMPLETED" or not isinstance(value.get("id"), str):
        raise ReleaseError("Info canary did not complete")
    output = value.get("output", {})
    if (output.get("model") != "qwen3-tts-1.7b-base" or output.get("source_commit", commit) != commit
            or output.get("default_quality_preset") != "native" or output.get("default_wav_encoding") != "float32"
            or output.get("default_mastering") != "native" or output.get("cuda_preflight") != "passed"
            or output.get("dtype") != "bfloat16" or output.get("sample_rate") != 24000):
        raise ReleaseError("Info canary does not attest the native image contract")


def transaction(directory, endpoint_id, before, proof, ci, bridge, *, active_path=None):
    image = proof["image"]
    previous_image = validate_snapshot(before, endpoint_id, idle=True)
    before_hash = checksum(before)
    snapshot_path, receipt_path = directory / "provider-before.json", directory / "receipt.json"
    durable_json(snapshot_path, before)
    receipt = {"schema": 1, "source_commit": proof["source_commit"], "controller_commit": proof.get("controller_commit"), "image": image,
               "previous_image": previous_image, "endpoint_id": endpoint_id,
               "template_id": before["template"]["id"], "before_sha256": before_hash,
               "protected_settings_sha256": checksum(protected_settings(before)),
               "registry_proof_sha256": checksum(proof), "ci": ci,
               "started_at": datetime.now(timezone.utc).isoformat(), "status": "prepared"}
    durable_json(receipt_path, receipt)
    if active_path is not None:
        durable_json(active_path, {"receipt": str(receipt_path)})
    mutation_started = False
    try:
        # Refresh under the held local lock immediately before emitting PATCH.
        fresh = bridge("read_provider", {"endpoint_id": endpoint_id})
        validate_snapshot(fresh, endpoint_id, idle=True)
        if protected_settings(fresh) != protected_settings(before) or fresh["template"]["imageName"] != previous_image:
            raise ReleaseError("Provider changed after preparation")
        receipt["status"] = "cutover_pending"
        durable_json(receipt_path, receipt)
        mutation_started = True
        after = bridge("patch_template_image", {"template_id": before["template"]["id"],
                                               "body": {"imageName": image}, "then": "read full endpoint and template"})
        readback_matches(before, after, endpoint_id, image)
        canary = bridge("run_info_canary", {"endpoint_id": endpoint_id,
                                            "input": {"action": "info"}, "allow_replacement": False})
        if isinstance(canary.get("id"), str):
            receipt["canary_job_id"] = canary["id"]
            durable_json(receipt_path, receipt)
        validate_canary(canary, proof["source_commit"])
        final = bridge("read_provider", {"endpoint_id": endpoint_id})
        readback_matches(before, final, endpoint_id, image)
        receipt.update(status="complete", after_sha256=checksum(final), canary_job_id=canary["id"],
                       canary_source_commit_attested=canary.get("output", {}).get("source_commit") == proof["source_commit"],
                       completed_at=datetime.now(timezone.utc).isoformat())
        durable_json(receipt_path, receipt)
        return receipt
    except BaseException:
        if mutation_started:
            receipt["status"] = "rollback_required"
            durable_json(receipt_path, receipt)
            try:
                quiescent = bridge("quiesce_canary", {"endpoint_id": endpoint_id,
                                   "canary_job_id": receipt.get("canary_job_id"),
                                   "instruction": "Confirm submitted info job terminal or cancelled and zero queued/active jobs; never submit a replacement"})
                validate_snapshot(quiescent, endpoint_id, idle=True)
                restored = bridge("restore_template_image", {"template_id": before["template"]["id"],
                                  "body": {"imageName": previous_image},
                                  "then": "read full endpoint and template"})
                readback_matches(before, restored, endpoint_id, previous_image)
                validate_snapshot(restored, endpoint_id, idle=True)
                receipt.update(status="rolled_back", rollback_sha256=checksum(restored))
                durable_json(receipt_path, receipt)
            except BaseException:
                # A lost MCP response must never manufacture rollback success.
                raise ReleaseError("Release failed; rollback still requires verified operator action") from None
        else:
            receipt["status"] = "aborted_before_mutation"
            durable_json(receipt_path, receipt)
        raise ReleaseError("Release failed; consult the private transaction receipt") from None


def mcp_bridge(action, details):
    print(json.dumps({"action": action, **details}), flush=True)
    line = sys.stdin.readline(4097)
    if not line or len(line) > 4096:
        raise ReleaseError("MCP evidence was not supplied")
    message = json.loads(line)
    if not isinstance(message, dict) or set(message) != {"path"}:
        raise ReleaseError("Reply with one owner-only evidence file path")
    return private_json(message["path"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--image-commit", required=True)
    parser.add_argument("--ci-run", required=True, type=int)
    parser.add_argument("--endpoint-id", required=True)
    parser.add_argument("--provider-before", required=True, type=Path)
    parser.add_argument("--state-root", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    def git(*arguments):
        return subprocess.run(["git", "-C", str(root), *arguments], capture_output=True, text=True, check=True).stdout.strip()
    if (not re.fullmatch(r"[a-f0-9]{40}", args.source_commit) or git("rev-parse", "HEAD") != args.source_commit
            or git("remote", "get-url", "origin") != "https://github.com/" + REPOSITORY + ".git"
            or git("status", "--porcelain", "--untracked-files=normal")):
        raise ReleaseError("Release requires the clean exact reviewed source commit")
    if git("ls-remote", "origin", "refs/heads/master").split()[0] != args.source_commit:
        raise ReleaseError("Release commit must be the canonical master head")
    state_root = args.state_root.resolve()
    if state_root.is_relative_to(root):
        raise ReleaseError("Private release state must remain outside the repository")
    durable_mkdir(state_root)
    if state_root.stat().st_uid != os.getuid() or state_root.stat().st_mode & 0o077:
        raise ReleaseError("Release state root must be owner-only")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", args.endpoint_id):
        raise ReleaseError("Invalid endpoint identity")
    lock_root = Path.home() / ".local/state/token-gen-runpod-releases"
    durable_mkdir(lock_root)
    if lock_root.stat().st_uid != os.getuid() or lock_root.stat().st_mode & 0o077:
        raise ReleaseError("Release lock directory must be owner-only")
    descriptor = os.open(lock_root / (args.endpoint_id + ".lock"), os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        active_path = lock_root / (args.endpoint_id + ".active.json")
        if active_path.exists():
            previous = private_json(active_path)
            record = private_json(previous["receipt"])
            if record.get("status") not in {"complete", "rolled_back", "aborted_before_mutation"}:
                raise ReleaseError("The endpoint has an unresolved earlier release transaction")
        for prior in state_root.glob("*/receipt.json"):
            record = private_json(prior)
            if record.get("endpoint_id") == args.endpoint_id and record.get("status") not in {"complete", "rolled_back", "aborted_before_mutation"}:
                raise ReleaseError("An earlier transaction still requires verified rollback")
        if not re.fullmatch(r"[a-f0-9]{40}", args.image_commit):
            raise ReleaseError("Invalid image commit")
        subprocess.run(["git", "-C", str(root), "merge-base", "--is-ancestor", args.image_commit, args.source_commit], check=True, capture_output=True)
        ci = json.loads(subprocess.run(["gh", "run", "view", str(args.ci_run), "--repo", REPOSITORY,
                         "--json", "headSha,status,conclusion,workflowName,url"], capture_output=True, text=True, check=True).stdout)
        if ci.get("headSha") != args.image_commit or ci.get("status") != "completed" or ci.get("conclusion") != "success" or ci.get("workflowName") != "Universal GPU image":
            raise ReleaseError("Exact native image master CI must have succeeded")
        proof = verify_image(args.image_commit)
        proof["controller_commit"] = args.source_commit
        directory = state_root / (args.image_commit + "-" + uuid.uuid4().hex)
        directory.mkdir(mode=0o700, exist_ok=False)
        durable_mkdir(directory)
        durable_json(directory / "registry-proof.json", proof)
        result = transaction(directory, args.endpoint_id, private_json(args.provider_before), proof, ci, mcp_bridge, active_path=active_path)
        print(json.dumps({"status": result["status"], "receipt": str(directory / "receipt.json")}), flush=True)
    finally:
        os.close(descriptor)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # Provider snapshots, registry URLs, credentials and canary bodies must
        # never escape through an exception repr or a traceback.
        print("Release transaction stopped; inspect the private receipt and required rollback action.", file=sys.stderr)
        raise SystemExit(1)
