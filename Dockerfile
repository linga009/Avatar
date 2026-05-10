FROM nvidia/cuda:12.6.3-base-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip3 install --no-cache-dir --break-system-packages "jax[cuda12]>=0.4.35"
RUN pip3 install --no-cache-dir --break-system-packages \
    equinox>=0.11.0 optax>=0.2.2 diffrax>=0.5.0 chex>=0.1.86
RUN pip3 install --no-cache-dir --break-system-packages \
    numpy>=1.24 einops>=0.7.0 pyyaml
RUN pip3 install --no-cache-dir --break-system-packages \
    sentence-transformers duckduckgo-search
# Prefrontal cortex: LoRA fine-tuning during dreaming
RUN pip3 install --no-cache-dir --break-system-packages \
    torch --index-url https://download.pytorch.org/whl/cpu
RUN pip3 install --no-cache-dir --break-system-packages \
    transformers peft

COPY halo3/ /app/halo3/
COPY train_halo3.py /app/train_halo3.py

RUN mkdir -p /app/data/checkpoints /app/data/episodes

CMD ["python3", "-m", "halo3.main"]
