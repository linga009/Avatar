FROM nvidia/cuda:12.6.3-base-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends python3 python3-pip libsndfile1 && \
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
    sentence-transformers ddgs soundfile Pillow

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

# Pre-download Wav2Vec2-base and CLIP ViT-B/32 (avoids first-tick download)
RUN python3 -c "\
from transformers import Wav2Vec2Processor, Wav2Vec2Model; \
Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base', cache_dir='/app/data/model_cache'); \
Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base', cache_dir='/app/data/model_cache')" \
    || echo "Wav2Vec2 pre-download skipped"

RUN python3 -c "\
from transformers import CLIPProcessor, CLIPVisionModel; \
CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32', cache_dir='/app/data/model_cache'); \
CLIPVisionModel.from_pretrained('openai/clip-vit-base-patch32', cache_dir='/app/data/model_cache')" \
    || echo "CLIP pre-download skipped"

COPY halo3/ /app/halo3/
COPY train_halo3.py /app/train_halo3.py
COPY train_tinystories.py /app/train_tinystories.py

RUN mkdir -p /app/data/checkpoints /app/data/episodes /app/data/pfc_adapter /app/data/dream_training /app/data/xla_cache /app/data/model_cache /app/data/senses

CMD ["python3", "-m", "halo3.main"]
