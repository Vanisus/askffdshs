# syntax=docker/dockerfile:1.7
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf-cache \
    SENTENCE_TRANSFORMERS_HOME=/opt/hf-cache \
    HF_HUB_DOWNLOAD_TIMEOUT=600 \
    TOKENIZERS_PARALLELISM=false

RUN sed -i \
        -e 's|http://archive.ubuntu.com/ubuntu|https://mirror.yandex.ru/ubuntu|g' \
        -e 's|http://security.ubuntu.com/ubuntu|https://mirror.yandex.ru/ubuntu|g' \
        /etc/apt/sources.list 2>/dev/null || true \
    && sed -i \
        -e 's|http://archive.ubuntu.com/ubuntu|https://mirror.yandex.ru/ubuntu|g' \
        -e 's|http://security.ubuntu.com/ubuntu|https://mirror.yandex.ru/ubuntu|g' \
        /etc/apt/sources.list.d/*.list /etc/apt/sources.list.d/*.sources 2>/dev/null || true \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Модели в образ НЕ вшиваются: их скачивает сервис model-init
# в именованный volume hf-cache (один раз, дальше берутся из кеша).
COPY app ./app

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1"]
