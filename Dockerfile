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

# Pre-download Qwen3 0.6B into the image (avoids dream-time download)
RUN python3 -c "from transformers import AutoTokenizer, AutoModelForCausalLM; \
    AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B', trust_remote_code=True); \
    AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B', trust_remote_code=True)" \
    || echo "Pre-download skipped (will download at first dream)"

COPY halo3/ /app/halo3/
COPY train_halo3.py /app/train_halo3.py
COPY train_tinystories.py /app/train_tinystories.py

RUN mkdir -p /app/data/checkpoints /app/data/episodes /app/data/pfc_adapter /app/data/dream_training

CMD ["python3", "-m", "halo3.main"]
