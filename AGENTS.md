# Token Gen Chatterbox RunPod Agent Instructions

This repository is the canonical source for the Token Gen Chatterbox RunPod
Serverless worker and its immutable container images. It must not contain API
keys, voice recordings, generated audio, provider responses, or runtime state.

## Repository identity and workflow

- Canonical root: `/home/jesse/.openclaw/sources/token-gen-chatterbox-runpod`
- Canonical remote:
  `https://github.com/owen-tech-ramblings/token-gen-chatterbox-runpod.git`
- Release branch: `master`
- Integration branch: `dev`
- Expected GitHub identity: `owen-tech-ramblings`
- Approved secret locators: Google Secret Manager project `lil-zen-oc` and
  `D:\openclaw` (`/mnt/d/openclaw` in WSL) only

Owner-authorized local authoring is supported in an isolated clean GitHub
checkout under `/home/zenfree/codex` or another ecosystem device. OpenClaw PC or
SSH availability is not a prerequisite. The live API/runtime remains a release
target, never an authoring checkout. Existing owner authorization takes
precedence over host-specific authoring paths; never fabricate a managed
context receipt to make local work appear to run on another host.

When operating through the OpenClaw control plane, before planning or editing
use the installed
`openclaw-enhancement-coding-guide` skill and read its complete generated
context package. Create only a receipt-bound managed `codex/*` worktree with
`openclaw_managed_worktree.py`. Start from aligned local and origin `dev` and
`master`, push the managed branch, use a pull request, and merge the reviewed
change to `master`. Do not author directly in this canonical checkout.

GitHub CI uses `risk-based-targeted-v2`: one validation workflow runs only on
a push to `master`, uses the repository-scoped self-hosted `lil-zen-ci` runner,
and selects one to five tests for the changed behaviour and protected
neighbours (ten is the absolute limit). Do not run complete-suite discovery or
duplicate checks on feature, pull-request, or `dev` events. Publish production
images only with commit-addressed tags and retain the resolved manifest digest
as release evidence.

## Required checks and build

Select the smallest relevant subset for the change:

- Qwen handler: `python3 -m unittest -v test_qwen_handler.py`
- Chatterbox handler: `python3 -m unittest -v test_handler.py`
- Syntax: `python3 -m py_compile handler.py qwen_handler.py scripts/chatterbox_client.py`
- Universal Qwen image:
  `docker build --file Dockerfile.qwen --build-arg MODEL_VARIANT=base --tag token-gen-chatterbox-runpod:qwen3-base-check .`

The production Qwen build must use the contract-pinned CUDA 12.8+ base and
matching PyTorch CUDA wheels. Its build preflight and startup preflight must
cover `sm_80`, `sm_86`, `sm_89`, `sm_90`, `sm_100`, and `sm_120`; startup must
perform a real CUDA allocation and synchronization before loading model
weights. Never install or select a different framework at worker runtime.

## Live deployment and rollback

The owner-local path uses the clean reviewed exact GitHub `master` commit and
`scripts/release_native_audio.py`. It verifies successful exact-commit CI and
authenticated immutable registry blobs, holds a local release lock, durably
records the complete private provider snapshot, checks there are no queued or
active jobs, and emits narrow image-only actions for the authorized parent
agent to perform through RunPod MCP. It accepts private readback evidence and
checks all other returned endpoint settings are unchanged. It uses current v2
`update_endpoint` with only `body.image`; templates are not a live update path. It never handles
RunPod credentials. A failed check requests canary quiescence and rollback to
the recorded image; a missing acknowledgement stays `rollback_required`.
Complete the recorded rollback before starting another transaction. Follow
`docs/local-image-release.md`; keep all snapshots/receipts outside this public
repository. Align canonical source and `dev` only after verified success.

The registered control-plane path remains the `token-gen-chatterbox-runpod`
release scope in the Lil Zen control plane. Run its release gate, then invoke
`platform-ops/scripts/openclaw_runpod_image_release_cutover.py` from the exact
released control-plane source with the reviewed image commit and its valid
context receipt. That transaction verifies exact-commit CI, resolves the GHCR
digest, snapshots the RunPod template and endpoint CUDA policy, deploys only
endpoint `usexk8jki4y8v3`, runs the `info` canary, records the immutable release,
and aligns `dev` only after success.

Rollback is owned by the same transaction. On any failure it restores the
captured prior image, endpoint CUDA policy, repository refs, canonical
checkout, and release pointer, then re-runs the provider post-check. A later
manual console edit is drift and is not a rollback or deployment path.

## Forbidden actions

- Do not edit the RunPod endpoint or template outside the reviewed owner-local
  transaction or registered control-plane transaction. Authenticated MCP
  performs only the exact action emitted by the active local transaction.
- Do not deploy mutable tags such as `latest`, `main`, `master`, or `dev`.
- Do not use GitHub-hosted runners, broad test discovery, or duplicated CI
  triggers for governed production changes.
- Do not release from a dirty, unreviewed, unlocked, unexpected-origin,
  runtime, or similarly named checkout.
- Do not commit credentials or read them from repository files, environment
  examples, command arguments, logs, or release records.
- Do not change global GitHub authentication to make a command pass.
- Do not give Token Gen callers direct RunPod access or introduce another TTS
  provider fallback; callers continue through the released Token Gen voice
  capability.
