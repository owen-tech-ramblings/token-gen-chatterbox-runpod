# Native speech transfer

The paired API supplies an expiring `transfer` capability before submission.
Both Chatterbox modes and Qwen Base/VoiceDesign upload the finished original
file directly in 8 MiB chunks, then send its SHA256, byte count, format and safe
encoding/processing metadata to `/complete`. Only an acknowledged, verified NAS
file produces `stored_on_nas: true`. The provider result contains a small
receipt rather than base64 audio. Network retries repeat the same chunk and
manifest; they never generate speech again or change audio quality.

The capability has `url`, `token`, `chunk_bytes: 8388608` and a per-request
`max_bytes` no larger than 805306368 (768 MiB). URLs must use the canonical
HTTPS API origin and `/api/tts/worker-transfers/<owner-scope>/<32-hex-job-id>`.
Redirects are refused. The token is confined to that origin/job. The API
verifies scope, expiry and job state before accepting bounded request bodies.

An optional `reference_download` contains the same capability URL plus
`/reference`, exact `bytes`, `sha256` and `format`. Its bearer comes from
`transfer`. Original reference files up to the existing 10 MiB bound are fetched
without base64 expansion, recompression or resampling. A reference must use
one transport; mixing download and inline reference fields is rejected.

The 768 MiB ceiling is an operational storage/memory bound, not a reason to
shorten normal authored input. For the supported 2,000-character limit,
Qwen's 600-character native splitting may make six segments; the 420-character
explicit style may make nine because boundaries can occur halfway through a
segment. The existing 8192-token per-segment bound at 12 Hz gives upper
projections of about 68.3 and 102.4 minutes. At mono 24 kHz float32, nine
segments need about 562.5 MiB; a 0.8× speed derivative needs about 703.1 MiB,
plus small pauses/headers. These are bounds, not measured normal speech times.
Chatterbox's existing sampling and segmentation remain unchanged. The API can
choose a tighter request budget within this hard ceiling. Exceeding it fails
explicitly without truncation or format fallback.

Direct clients without the paired capability keep the legacy inline response
and its 14 MiB guard. The paired API uses durable transfer by default. The API
retains the uploaded original and stores any explicitly requested speed
derivative separately. Its existing catalogue owns final artifact registration
and access control; this worker does not create a second catalogue.

Offline tests use synthetic samples, a >14 MiB WAV, fake authenticated HTTPS,
lost acknowledgements, checksum failures and refused redirects. They establish
the protocol and sample preservation, not a paid generation or live transfer.
