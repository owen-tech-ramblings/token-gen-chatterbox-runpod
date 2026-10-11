# Token-Gen TTS on RunPod Serverless

A scale-to-zero RunPod worker for English text-to-speech and zero-shot voice
cloning with Resemble AI's Chatterbox Turbo model.

The repository also contains a reversible Qwen3-TTS 1.7B trial. The Qwen Base
worker uses a reference recording plus its exact transcript for higher-fidelity
voice cloning; the separate VoiceDesign image creates original synthetic voice
references. Both image variants are built from `Dockerfile.qwen` and are
deployed to the same TTS endpoint at different stages, so the trial does not
create another billable RunPod endpoint.

The model weights are baked into the container to reduce cold-start time.
Reference audio and generated WAV/MP3 files are written only to a per-job temporary
directory and removed before the handler returns. The worker never logs audio or
prompt contents. Chatterbox adds its built-in PerTh watermark to generated audio.

## Release and images

Publish only the reviewed commit through the master-only self-hosted workflow,
then use the registered rollback-protected image release transaction described
in `AGENTS.md`. Deployment uses the resolved image digest. This repository does
not declare a mutable tag or a historical canary to be the current deployment.
No network volume or Hugging Face token is required by the baked model runtime.

## Qwen3-TTS trial images

The `Universal GPU image` workflow publishes the Base variant with a full
commit-addressed tag. VoiceDesign uses the same Dockerfile with its own explicit
model ID and revision; it requires a separate reviewed build:

```text
ghcr.io/owen-tech-ramblings/token-gen-chatterbox-runpod:qwen3-design-<commit>
ghcr.io/owen-tech-ramblings/token-gen-chatterbox-runpod:qwen3-base-<commit>
```

The 1.7B model and tokenizer are pinned and baked into each image. The worker
uses BF16 on supported GPUs, PyTorch SDPA for broad Ampere compatibility, one
request at a time, and the same native WAV and opt-in MP3/mastering contract as
the Chatterbox worker.

High-quality Qwen cloning supplies both fields:

```json
{
  "input": {
    "text": "This uses the exact-transcript clone.",
    "reference_audio": {
      "base64": "<base64 audio>",
      "content_type": "audio/wav"
    },
    "reference_text": "The exact words spoken in the reference recording.",
    "x_vector_only_mode": false,
    "quality_preset": "native",
    "output_format": "wav"
  }
}
```

Legacy profiles without a transcript remain usable with
`x_vector_only_mode: true`, but the response reports
`clone_mode: speaker_embedding` because that path has lower cloning fidelity.

The VoiceDesign image accepts `action: design` and a `voice_description`.
Designed clips are references for the Base image; VoiceDesign is not the
long-term production image.

## Request contract

The default action is `generate`:

```json
{
  "input": {
    "text": "Hello from Chatterbox [chuckle].",
    "quality_preset": "native",
    "seed": 42,
    "temperature": 0.80,
    "top_p": 0.95,
    "top_k": 1000,
    "repetition_penalty": 1.2,
    "normalize_loudness": true,
    "output_format": "wav",
    "wav_encoding": "float32",
    "mastering": "native"
  }
}
```

`native` is the default sampling preset. Chatterbox uses the defaults verified
in the pinned `chatterbox-tts==0.1.7` package: temperature 0.8, top-p 0.95,
top-k 1000 and repetition penalty 1.2. Qwen uses temperature 0.9, top-p 1.0,
top-k 50 and repetition penalty 1.05, matching its
[pinned model generation configuration](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base/blob/fd4b254389122332181a7c3db7f27e918eec64e3/generation_config.json)
and `qwen-tts==0.1.1`. Qwen retains its 8192-token generation limit and matching
subtalker sampling. Both workers retain bounded sentence segmentation and
preserve the supplied words and reference information.

`publication`, `balanced` and `expressive` remain explicit sampling choices.
Stochastic sampling values are not an ordered quality scale. Explicit numeric
values override the chosen preset. Chatterbox's `normalize_loudness` controls
reference conditioning, following upstream; it does not master generated output.

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

`output_format` may be `wav` (default) or explicitly selected `mp3`. Native WAV
uses `wav_encoding: "float32"` and `mastering: "native"`: finite returned float32
samples are preserved exactly, including headroom outside +/-1, at the model's
native sample rate. There is no output clipping, normalisation or resampling.
This preserves the generated waveform; it does not manufacture extra model
precision or reverse synthesis/watermarking performed by the model.

Use `wav_encoding: "pcm16"` for an explicit smaller compatibility WAV. It clips
to the PCM range and quantises to 16 bits. Use
`mastering: "podcast_mono_-19_lufs"` to explicitly apply the existing mono
-19 LUFS, -1.5 dB true-peak loudnorm filter; its intermediate WAV stays float32.
To request the former publication processing, supply `quality_preset:
"publication"`, that mastering mode, and `wav_encoding: "pcm16"` for WAV.
No sampling preset implicitly enables mastering.

MP3 remains an explicit lossy choice at 160 kbit/s, the maximum MPEG-2 Layer III
rate at 24 kHz. `wav_encoding: "pcm16"` is rejected with MP3. Native float WAV
uses twice the sample payload of PCM16; the unchanged 14 MiB inline audio limit
fails explicitly when exceeded instead of silently converting or compressing.
Split overly long text into shorter requests.

The response retains its existing audio/MIME/hash fields and reports actual
encoding and processing. `native_audio_preserved` is true only for untouched
float32 WAV; it is false for podcast mastering, PCM16 or MP3. `mastering` is
null when no output processing was applied. WAV fields are null for MP3.

```json
{
  "audio_base64": "<base64 audio>",
  "mime_type": "audio/wav",
  "output_format": "wav",
  "wav_encoding": "float32",
  "audio_encoding": "pcm_f32le",
  "bit_depth": 32,
  "native_audio_preserved": true,
  "sample_rate": 24000,
  "duration_seconds": 2.5,
  "sha256": "...",
  "model": "chatterbox-turbo",
  "seed": 42,
  "quality_preset": "native",
  "segment_count": 1,
  "mastering": null,
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
python3 -m unittest -v test_handler.py test_qwen_handler.py
python3 -m py_compile handler.py qwen_handler.py scripts/chatterbox_client.py
```

## Isolated emotion-model auditions

The `audition/` directory contains temporary, scale-to-zero workers for a
voice-retention and emotional-range comparison. They are deliberately separate
from the production Qwen endpoint and are never selected by the Token-Gen API.

Four images are published from one workflow: Fish Audio S2 Pro, Higgs TTS 3,
IndexTTS2, and ZONOS2. One temporary RunPod endpoint can be updated between
images after it has scaled to zero, avoiding concurrent audition workers. Every
worker accepts the same `text`, `reference_audio`, `reference_text`, `delivery`,
`strength`, and `seed` contract. The fixed six-segment script is in
`audition/matrix.json`.

Hume Octave 2 is not included in the RunPod images because Hume does not publish
self-hostable Octave 2 weights. It must be tested through Hume's hosted API with
an account credential.
