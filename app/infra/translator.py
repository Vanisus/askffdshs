import logging
import re
from collections import OrderedDict

import torch

logger = logging.getLogger(__name__)


_TRANSLATION_FIXES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bIdico\b", re.IGNORECASE), "Ideco"),
    (re.compile(r"\bIdica\b", re.IGNORECASE), "Ideco"),
    (re.compile(r"\bIdeko\b", re.IGNORECASE), "Ideco"),
    (re.compile(r"\bKS\s*MG\b", re.IGNORECASE), "KSMG"),
    (re.compile(r"\bPower\s+Shell\b", re.IGNORECASE), "PowerShell"),
    (re.compile(r"\bKasper\s+sky\b", re.IGNORECASE), "Kaspersky"),
    (re.compile(r"\bSuspiciousus\b", re.IGNORECASE), "Suspicious"),
    (re.compile(r"\bSuspicous\b", re.IGNORECASE), "Suspicious"),
    (re.compile(r"\bSuspicoius\b", re.IGNORECASE), "Suspicious"),
    (re.compile(r"\bwith\s+a?\s*Mark\b", re.IGNORECASE), "with a label"),
    (re.compile(r"\bwith\s+Mark\b", re.IGNORECASE), "with a label"),
)


class Translator:
    def __init__(
        self,
        model_name: str = "Helsinki-NLP/opus-mt-ru-en",
        device: str | None = None,
        max_input_tokens: int = 512,
        cache_size: int = 8192,
        *,
        num_beams: int = 2,
        length_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        use_fp16: bool = False,
        probe_text: str = "подозрительный DNS-туннель",
        probe_required: bool = True,
    ) -> None:
        from transformers import MarianMTModel, MarianTokenizer

        self.model_name = model_name
        self.max_input_tokens = max_input_tokens
        self.cache_size = cache_size
        self.num_beams = num_beams
        self.length_penalty = length_penalty
        self.no_repeat_ngram_size = no_repeat_ngram_size

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        logger.info("Loading translator %s on %s", model_name, device)
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name)
        self.model.to(device).eval()

        if use_fp16 and device.startswith("cuda"):
            self.model = self.model.half()
            logger.info("Translator using FP16 on %s", device)

        self._cache: OrderedDict[str, str] = OrderedDict()

        if probe_required:
            self._probe(probe_text)

    @torch.inference_mode()
    def translate(self, text: str) -> str:
        return self.translate_batch([text])[0]

    @torch.inference_mode()
    def translate_batch(self, texts: list[str]) -> list[str]:
        if not texts:
            return []

        results: list[str | None] = [None] * len(texts)
        pending: list[str] = []
        pending_idx: list[int] = []

        for i, t in enumerate(texts):
            if not t or not t.strip():
                results[i] = ""
                continue
            cached = self._cache_get(t)
            if cached is not None:
                results[i] = cached
            else:
                pending.append(t)
                pending_idx.append(i)

        if pending:
            encoded = self.tokenizer(
                pending,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_input_tokens,
            ).to(self.device)

            generate_kwargs: dict[str, object] = {
                "max_length": self.max_input_tokens,
                "num_beams": self.num_beams,
                "length_penalty": self.length_penalty,
                "do_sample": False,
            }
            if self.no_repeat_ngram_size > 0:
                generate_kwargs["no_repeat_ngram_size"] = self.no_repeat_ngram_size

            generated = self.model.generate(**encoded, **generate_kwargs)
            decoded = self.tokenizer.batch_decode(
                generated, skip_special_tokens=True
            )

            for idx, src, dst in zip(pending_idx, pending, decoded):
                fixed = self._apply_fixes(dst)
                results[idx] = fixed
                self._cache_put(src, fixed)

        return [r or "" for r in results]

    @staticmethod
    def _apply_fixes(text: str) -> str:
        if not text:
            return text
        for pattern, replacement in _TRANSLATION_FIXES:
            text = pattern.sub(replacement, text)
        return text

    def _probe(self, probe_text: str) -> None:
        try:
            probe_out = self.translate(probe_text)
        except Exception as exc:
            raise RuntimeError(
                f"Translator probe raised on {probe_text!r}: {exc}"
            ) from exc

        normalized_in = re.sub(r"\W+", " ", probe_text.lower()).strip()
        normalized_out = re.sub(r"\W+", " ", probe_out.lower()).strip()

        if not normalized_out:
            raise RuntimeError(
                f"Translator probe returned empty output for {probe_text!r}"
            )

        if normalized_in == normalized_out:
            raise RuntimeError(
                f"Translator probe returned unchanged input for "
                f"{probe_text!r}: {probe_out!r}"
            )

        logger.info("Translator probe OK: %r → %r", probe_text, probe_out)

    def _cache_get(self, key: str) -> str | None:
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def _cache_put(self, key: str, value: str) -> None:
        self._cache[key] = value
        self._cache.move_to_end(key)
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
