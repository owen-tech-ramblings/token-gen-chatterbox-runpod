#!/usr/bin/env python3
"""Submit a Chatterbox job, wait for completion, and save its WAV output."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
from pathlib import Path
import sys
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen


API_BASE = "https://api.runpod.ai/v2"
TERMINAL_STATES = {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}


def request_json(url: str, key: str, payload: dict | None = None) -> dict:
    body = json.dumps(payload).encode() if payload is not None else None
    request = Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="POST" if payload is not None else "GET",
    )
    try:
        with urlopen(request, timeout=60) as response:
            return json.load(response)
    except HTTPError as exc:
        details = exc.read().decode(errors="replace")
        raise RuntimeError(f"RunPod returned HTTP {exc.code}: {details}") from exc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("text", help="English text to synthesize")
    parser.add_argument("--reference", type=Path, help="5-10 second voice clip")
    parser.add_argument("--output", type=Path, default=Path("chatterbox-output.wav"))
    parser.add_argument(
        "--format",
        choices=("wav", "mp3"),
        help="Output format; defaults to the --output extension or WAV",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()

    key = os.getenv("RUNPOD_API_KEY")
    endpoint_id = os.getenv("CHATTERBOX_ENDPOINT_ID")
    if not key or not endpoint_id:
        parser.error("RUNPOD_API_KEY and CHATTERBOX_ENDPOINT_ID must be set")

    output_format = args.format
    if output_format is None:
        output_format = args.output.suffix.lower().lstrip(".") or "wav"
    if output_format not in {"wav", "mp3"}:
        parser.error("--output must end in .wav or .mp3 unless --format is supplied")

    job_input: dict = {
        "text": args.text,
        "seed": args.seed,
        "output_format": output_format,
    }
    if args.reference:
        content_type = mimetypes.guess_type(args.reference.name)[0] or "audio/wav"
        job_input["reference_audio"] = {
            "base64": base64.b64encode(args.reference.read_bytes()).decode(),
            "content_type": content_type,
        }

    submitted = request_json(
        f"{API_BASE}/{endpoint_id}/run", key, {"input": job_input}
    )
    job_id = submitted.get("id")
    if not job_id:
        raise RuntimeError(f"RunPod did not return a job id: {submitted}")
    print(f"Submitted {job_id}; waiting for the scale-to-zero worker...")

    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        status = request_json(f"{API_BASE}/{endpoint_id}/status/{job_id}", key)
        state = status.get("status")
        if state in TERMINAL_STATES:
            break
        time.sleep(3)
    else:
        raise TimeoutError(f"job {job_id} did not finish within {args.timeout}s")

    if state != "COMPLETED":
        raise RuntimeError(f"job {job_id} ended as {state}: {status.get('error')}")

    output = status.get("output") or {}
    if output.get("output_format") not in (None, output_format):
        raise RuntimeError("Chatterbox returned an unexpected audio format")
    audio = base64.b64decode(output["audio_base64"], validate=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(audio)
    print(
        f"Saved {args.output} ({output.get('duration_seconds')}s, "
        f"{output.get('model')})"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, TimeoutError, OSError, KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
