# Chatterbox Turbo on RunPod Serverless

A scale-to-zero RunPod worker for English text-to-speech and zero-shot voice
cloning with Resemble AI's Chatterbox Turbo model.

The model weights are baked into the container to reduce cold-start time.
Reference audio and generated WAV/MP3 files are written only to a per-job temporary
directory and removed before the handler returns. The worker never logs audio or
prompt contents. Chatterbox adds its built-in PerTh watermark to generated audio.

## Endpoint configuration

Use the public container:

```text
ghcr.io/owen-tech-ramblings/token-gen-chatterbox-runpod:sha-c71bc70
```

Recommended RunPod Serverless settings:

- GPU: A4000 / A4500 / RTX 4000 (16 GB), with L4 / A5000 / RTX 3090 fallback
- GPUs per worker: 1
- Active workers: 0
- Max workers: 1
- Scaling: queue delay, 1 second
- Idle timeout: 5 seconds
- Execution timeout: 600 seconds
- FlashBoot: enabled
- Container disk: at least 20 GB

No network volume or Hugging Face token is required.

## Current deployment

The production scale-to-zero endpoint is:

```text
Endpoint ID: usexk8jki4y8v3
Run:         https://api.runpod.ai/v2/usexk8jki4y8v3/run
Run sync:    https://api.runpod.ai/v2/usexk8jki4y8v3/runsync
```

It uses one GPU from the `AMPERE_16` pool, with `AMPERE_24` as an
availability fallback. Minimum workers is 0 and maximum workers is 1.

The native-MP3 image is published as `sha-c71bc70`. Until the endpoint template
is rolled to that tag, the Token-Gen API remains backward compatible by
detecting the prior worker's PCM WAV response and encoding requested MP3 output
at the gateway.

The first deployment canary took 238 seconds because RunPod had to pull the
image for the first time. A second request after scale-down completed in
33 seconds through FlashBoot, confirming that the endpoint returned to zero
rather than remaining warm.

## Request contract

The default action is `generate`:

```json
{
  "input": {
    "text": "Hello from Chatterbox [chuckle].",
    "quality_preset": "publication",
    "seed": 42,
    "temperature": 0.65,
    "top_p": 0.90,
    "top_k": 500,
    "repetition_penalty": 1.2,
    "normalize_loudness": true,
    "output_format": "mp3"
  }
}
```

`publication` is the default. It uses conservative sampling, splits long inputs
at English sentence boundaries before generation, and masters the result to
-19 LUFS with a -1.5 dB true-peak ceiling for consistent mono spoken-word audio.
`balanced` keeps the upstream Turbo sampling defaults, while `expressive` is
intended for deliberate character performance. Explicit sampling values
override the selected preset.

For voice cloning, add a clean 5-10 second reference clip:

```json
{
  "input": {
    "text": "This uses the reference voice.",
    "reference_audio": {
      "base64": "<base64 audio>",
      "content_type": "audio/wav"
    }
  }
}
```

WAV, MP3, FLAC, and OGG references are accepted. References are limited to
10 MiB and text to 2,000 characters. Chatterbox itself requires reference audio
longer than five seconds.

The worker does not persist or log reference audio. RunPod still handles the
request and result as platform job data; asynchronous results are retained by
RunPod for 30 minutes.

`output_format` may be `wav` (the default) or `mp3`. MP3 output is encoded at
160 kbit/s, the maximum MPEG-2 Layer III rate at the model's native 24 kHz
sample rate. The response contains inline base64 audio:

```json
{
  "audio_base64": "<base64 audio>",
  "mime_type": "audio/mpeg",
  "output_format": "mp3",
  "sample_rate": 24000,
  "duration_seconds": 2.5,
  "sha256": "...",
  "model": "chatterbox-turbo",
  "seed": 42,
  "quality_preset": "publication",
  "segment_count": 1,
  "mastering": "podcast_mono_-19_lufs",
  "used_reference_voice": false,
  "watermarked": true
}
```

Use `{"input":{"action":"info"}}` to inspect the live model and GPU.

## Client

Set the endpoint ID and an API key that has AI API access to this specific
endpoint. Do not commit the key:

```bash
export CHATTERBOX_ENDPOINT_ID="usexk8jki4y8v3"
export RUNPOD_API_KEY="your-runpod-api-key"
python3 scripts/chatterbox_client.py \
  "This is a live Chatterbox test." \
  --output canary-output/test.mp3
```

Add `--reference voice.wav` to exercise voice cloning. The client uses the
asynchronous RunPod API so it can wait through a scale-to-zero cold start.

## Local validation

The unit tests do not load the GPU model:

```bash
python3 -m unittest -v
python3 -m py_compile handler.py scripts/chatterbox_client.py
```
