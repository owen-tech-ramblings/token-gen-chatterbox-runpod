# Chatterbox Turbo on RunPod Serverless

A scale-to-zero RunPod worker for English text-to-speech and zero-shot voice
cloning with Resemble AI's Chatterbox Turbo model.

The model weights are baked into the container to reduce cold-start time.
Reference audio and generated WAV files are written only to a per-job temporary
directory and removed before the handler returns. The worker never logs audio or
prompt contents. Chatterbox adds its built-in PerTh watermark to generated audio.

## Endpoint configuration

Use the public container:

```text
ghcr.io/owen-tech-ramblings/token-gen-chatterbox-runpod:latest
```

Recommended RunPod Serverless settings:

- GPU: A4000 / A4500 / RTX 4000 (16 GB), with L4 / A5000 / RTX 3090 fallback
- GPUs per worker: 1
- Active workers: 0
- Max workers: 1
- Scaling: queue delay, 1 second
- Idle timeout: 30 seconds
- Execution timeout: 600 seconds
- FlashBoot: enabled
- Container disk: at least 20 GB

No network volume or Hugging Face token is required.

## Request contract

The default action is `generate`:

```json
{
  "input": {
    "text": "Hello from Chatterbox [chuckle].",
    "seed": 42,
    "temperature": 0.8,
    "top_p": 0.95,
    "top_k": 1000,
    "repetition_penalty": 1.2,
    "normalize_loudness": true
  }
}
```

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

The response contains an inline base64 PCM WAV:

```json
{
  "audio_base64": "<base64 WAV>",
  "mime_type": "audio/wav",
  "sample_rate": 24000,
  "duration_seconds": 2.5,
  "sha256": "...",
  "model": "chatterbox-turbo",
  "seed": 42,
  "used_reference_voice": false,
  "watermarked": true
}
```

Use `{"input":{"action":"info"}}` to inspect the live model and GPU.

## Client

Set the endpoint ID and API key without committing either value:

```bash
export CHATTERBOX_ENDPOINT_ID="your-endpoint-id"
export RUNPOD_API_KEY="your-runpod-api-key"
python3 scripts/chatterbox_client.py \
  "This is a live Chatterbox test." \
  --output canary-output/test.wav
```

Add `--reference voice.wav` to exercise voice cloning. The client uses the
asynchronous RunPod API so it can wait through a scale-to-zero cold start.

## Local validation

The unit tests do not load the GPU model:

```bash
python3 -m unittest -v
python3 -m py_compile handler.py scripts/chatterbox_client.py
```

