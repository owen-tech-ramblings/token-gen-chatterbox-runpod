FROM nvidia/cuda:12.9.0-cudnn-runtime-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG FISH_SOURCE_REVISION=e5e292632cb11e7a27b2b7487f58f612bc101e13

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/models/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1 \
    AUDITION_BACKEND=fish

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        build-essential ca-certificates cmake curl ffmpeg git libasound2-dev \
        libportaudio2 libportaudiocpp0 libsox-dev portaudio19-dev python3-dev \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.8.15 /uv /uvx /usr/local/bin/

RUN git clone https://github.com/fishaudio/fish-speech.git /opt/fish-speech \
    && git -C /opt/fish-speech checkout "${FISH_SOURCE_REVISION}"

WORKDIR /opt/fish-speech
RUN uv sync --extra cu129 --frozen \
    && uv pip install --python .venv/bin/python runpod==1.11.0 httpx==0.28.1

COPY audition/server_proxy_handler.py /app/server_proxy_handler.py

CMD ["/opt/fish-speech/.venv/bin/python", "-u", "/app/server_proxy_handler.py"]
