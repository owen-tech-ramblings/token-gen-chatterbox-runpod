FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG SOURCE_REPOSITORY=https://github.com/owen-tech-ramblings/token-gen-chatterbox-runpod

LABEL org.opencontainers.image.source="${SOURCE_REPOSITORY}"
LABEL org.opencontainers.image.description="Chatterbox Turbo text-to-speech worker for RunPod Serverless"
LABEL org.opencontainers.image.licenses="MIT"

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HUB_DISABLE_XET=1 \
    TRANSFORMERS_OFFLINE=1

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates \
        ffmpeg \
        git \
        libsndfile1 \
        python3 \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install \
        --index-url https://download.pytorch.org/whl/cu124 \
        torch==2.6.0 \
        torchaudio==2.6.0

WORKDIR /app
COPY requirements.txt .
RUN python3 -m pip install --requirement requirements.txt

# Bake the public model into the image so a scale-to-zero cold start does not
# have to download model weights.
RUN TRANSFORMERS_OFFLINE=0 python3 -c \
    "from huggingface_hub import snapshot_download; snapshot_download(repo_id='ResembleAI/chatterbox-turbo', allow_patterns=['*.safetensors', '*.json', '*.txt', '*.pt', '*.model'])"

COPY handler.py .

CMD ["python3", "-u", "handler.py"]

