FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf-cache \
    TRANSFORMERS_CACHE=/opt/hf-cache \
    SENTENCE_TRANSFORMERS_HOME=/opt/hf-cache \
    TOKENIZERS_PARALLELISM=false

# Меняем только apt-зеркала Ubuntu на Yandex.
# PyPI оставляем официальный — pip тянет пакеты с pypi.org.
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
RUN pip install -r requirements.txt

COPY app ./app

RUN python -c "\
import os; \
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '600'; \
from transformers import MarianMTModel, MarianTokenizer; \
from sentence_transformers import SentenceTransformer, CrossEncoder; \
MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-ru-en'); \
MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-ru-en'); \
SentenceTransformer('intfloat/multilingual-e5-small'); \
CrossEncoder('BAAI/bge-reranker-v2-m3'); \
"

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1"]