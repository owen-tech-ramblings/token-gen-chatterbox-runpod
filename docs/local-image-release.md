# Owner-local immutable image release

This is a real local transaction for an authorized parent agent with RunPod MCP.
It has no dependency on an OpenClaw PC or SSH receipt. It does not replace the
registered control-plane path when that path is being used.

Author and review changes in an isolated branch. After the reviewed PR reaches
`master`, wait for the one master CI run to succeed. Use a clean exact checkout
of that commit and the existing GitHub identity; no global auth switching.
The transaction verifies its controller source is the clean remote master head,
that the image commit is an ancestor, successful image CI SHA/workflow,
immutable image manifest/config hashes, Linux amd64 platform and source label,
and every authenticated blob HEAD byte count. This proves artifact availability,
not GPU execution. It does not download/re-hash entire layers.

Before invocation, the parent uses read-only MCP calls to write a private
mode-0600 JSON snapshot containing full `endpoint`, full `template`, and
`health` objects. Include the endpoint's `id`, `templateId`, `gpuTypeIds`,
`allowedCudaVersions`, `cudaVersion`, `workersMin`, `workersMax`, template `id`,
`imageName`, complete `env`, and job health counters. Never commit that file or
copy its environment values into chat/logs. The prior image must be immutable
so rollback has an exact target. Store private state in a mode-0700 directory
outside the repository.

```sh
python3 scripts/release_native_audio.py \
  --source-commit REVIEWED_CONTROLLER_MASTER_SHA \
  --image-commit REVIEWED_IMAGE_MASTER_SHA --ci-run SUCCESSFUL_IMAGE_RUN_ID \
  --endpoint-id AUTHORIZED_ENDPOINT \
  --provider-before /private/provider-before.json \
  --state-root /private/native-audio-releases
```

Keep this process alive and its stdin open: its filesystem lock spans provider
readback, mutation, canary and rollback. For each JSON action on stdout, the
parent performs exactly that authenticated MCP operation, saves the fresh
result in another owner-only file, and replies with one line:
`{"path":"/private/evidence.json"}`. The controller accepts no API credential.
Only `template.imageName` may change. GPU/CUDA allow-lists, environment,
timeouts, volume, scaling and all other returned settings must remain equal;
only provider update timestamps are excluded from comparison.

The controller refreshes idle status immediately before mutation, verifies the
new digest after mutation, requests one `action: info` canary, and verifies the
native output defaults, BF16 and CUDA preflight. If the worker reports a source
commit, it must match; older native workers lack that field, so source identity
comes from the verified OCI revision label and immutable provider readbacks.
The receipt distinguishes that from a canary source-commit attestation. The parent must retain a canary
job ID immediately after acknowledgement, poll that same job to terminal, and
never replace an ambiguous submission. Provide the complete final provider
job envelope as evidence. A final provider readback completes the receipt.
No speech canary is implied by `info`; a separately authorized native WAV
generation and byte-level API check establish the actual end-to-end result.

On failure after mutation, the transaction first asks the parent to confirm
the info job is terminal/cancelled and endpoint queue/in-progress counts are
zero, then restore the exact prior image and supply fresh provider readback.
Failed/lost rollback evidence stays `rollback_required`. Do not claim success
or start another release while such a receipt is unresolved. If the controller
itself is interrupted, use its durable private snapshot and `cutover_pending`
or `rollback_required` receipt to perform the same verified rollback under the
same release lock before another release. Never infer rollback success from a
timeout or lost acknowledgement.

Receipts contain image/source hashes, CI evidence, protected-setting hashes,
job ID and outcome. Full snapshots remain private mode-0600 files. Align the
canonical checkout and `dev` only after a `complete` receipt and actual service
readiness. Preserve the receipt and previous immutable image for rollback.
