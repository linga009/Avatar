FROM nvidia/cuda:12.6.3-base-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Physics engine (JAX CUDA)
RUN pip3 install --no-cache-dir --break-system-packages "jax[cuda12]>=0.4.35"
RUN pip3 install --no-cache-dir --break-system-packages \
    equinox>=0.11.0 optax>=0.2.2 diffrax>=0.5.0 chex>=0.1.86
RUN pip3 install --no-cache-dir --break-system-packages \
    numpy>=1.24 einops>=0.7.0 pyyaml pyarrow sentencepiece

# Perception
RUN pip3 install --no-cache-dir --break-system-packages \
    sentence-transformers ddgs

# Prefrontal cortex: LoRA fine-tuning during dreaming
RUN pip3 install --no-cache-dir --break-system-packages \
    torch --index-url https://download.pytorch.org/whl/cpu
RUN pip3 install --no-cache-dir --break-system-packages \
    transformers peft

# --- Senses deps (v3.7: FNO runs on JAX/GPU, only need Pillow for image loading) ---
RUN pip3 install --no-cache-dir --break-system-packages Pillow

# --- Speech (v3.8: TTS, v3.10: speech recognition + neural TTS) ---
RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends espeak-ng && \
    rm -rf /var/lib/apt/lists/*
RUN pip3 install --no-cache-dir --break-system-packages faster-whisper piper-tts

COPY halo3/ /app/halo3/
COPY train_halo3.py /app/train_halo3.py
COPY train_tinystories.py /app/train_tinystories.py

RUN mkdir -p /app/data/checkpoints /app/data/episodes /app/data/pfc_adapter /app/data/dream_training /app/data/xla_cache /app/data/model_cache /app/data/senses

CMD ["python3", "-m", "halo3.main"]
