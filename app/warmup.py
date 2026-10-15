import os
import sys
import time

os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")

TRANSLATION_MODEL = os.getenv("SIEM_TRANSLATION_MODEL", "Helsinki-NLP/opus-mt-ru-en")
EMBEDDING_MODEL = os.getenv("SIEM_EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
RERANKER_MODEL = os.getenv("SIEM_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")


def _step(name, fn):
    started = time.time()
    print(f"[model-init] {name} ...", flush=True)
    fn()
    print(f"[model-init] {name} ok ({time.time() - started:.1f}s)", flush=True)


def main() -> int:
    from transformers import MarianMTModel, MarianTokenizer
    from sentence_transformers import CrossEncoder, SentenceTransformer

    def marian():
        MarianTokenizer.from_pretrained(TRANSLATION_MODEL)
        MarianMTModel.from_pretrained(TRANSLATION_MODEL)

    _step(f"translator {TRANSLATION_MODEL}", marian)
    _step(f"embeddings {EMBEDDING_MODEL}", lambda: SentenceTransformer(EMBEDDING_MODEL, device="cpu"))
    _step(f"reranker {RERANKER_MODEL}", lambda: CrossEncoder(RERANKER_MODEL, device="cpu"))
    print("[model-init] all models cached", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
