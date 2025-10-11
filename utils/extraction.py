from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Callable, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


class HttpLLMExtractor:
    """Simple HTTP client for JSON-extraction prompts."""

    def __init__(
        self,
        endpoint: str,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: int = 15,
        temperature: float = 0.0,
        system_prompt: Optional[str] = None,
        extra_payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.endpoint = endpoint
        self.model = model
        self.api_key = api_key
        self.headers = headers or {}
        self.timeout = timeout
        self.temperature = temperature
        self.system_prompt = system_prompt or (
            "You extract structured JSON that conforms exactly to the caller's instructions."
        )
        self.extra_payload = extra_payload or {}

        if api_key and "Authorization" not in self.headers:
            self.headers["Authorization"] = f"Bearer {api_key}"

        if "Content-Type" not in self.headers:
            self.headers["Content-Type"] = "application/json"

    def __call__(self, prompt: str) -> str:
        payload: Dict[str, Any] = {
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
        }
        if self.model:
            payload["model"] = self.model
        payload.update(self.extra_payload)

        try:
            response = requests.post(
                self.endpoint,
                headers=self.headers,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("HTTP extractor failed: %s", exc)
            raise

        # OpenAI compatible schema
        if isinstance(data, dict):
            choices = data.get("choices")
            if isinstance(choices, list) and choices:
                message = choices[0].get("message", {})
                content = message.get("content")
                if isinstance(content, str):
                    return content
            if "content" in data and isinstance(data["content"], str):
                return data["content"]

        raise RuntimeError("Unsupported response schema from extractor")


class RegexFallbackExtractor:
    """Deterministic extractor when LLM access is unavailable."""

    _STOPWORDS = {
        "the",
        "and",
        "for",
        "with",
        "that",
        "have",
        "this",
        "about",
        "from",
        "your",
        "into",
        "their",
        "will",
        "would",
        "there",
        "which",
        "should",
        "could",
        "really",
        "going",
        "because",
        "while",
        "these",
    }

    def __call__(self, prompt: str) -> str:
        text = self._extract_text(prompt)
        if not text:
            return "[]"

        if "personal" in prompt.lower():
            units = self._extract_personal(text)
        else:
            units = self._extract_public(text)
        return json.dumps(units[:4])

    @staticmethod
    def _extract_text(prompt: str) -> str:
        match = re.search(r"Input:\s*(.*)", prompt, re.DOTALL)
        return match.group(1).strip() if match else ""

    def _extract_public(self, text: str) -> List[Dict[str, Any]]:
        units: List[Dict[str, Any]] = []
        for sentence in self._sentences(text):
            if len(sentence) < 12:
                continue
            stype = "concept" if " is " in sentence.lower() else "pattern"
            units.append(
                {
                    "type": stype,
                    "title": self._titleize(sentence),
                    "content": sentence[:200],
                    "symbols": self._symbols(sentence),
                }
            )
        return units

    def _extract_personal(self, text: str) -> List[Dict[str, Any]]:
        units: List[Dict[str, Any]] = []
        for sentence in self._sentences(text):
            lowered = sentence.lower()
            if "i need" in lowered or "i have to" in lowered or "i must" in lowered:
                stype = "todo"
            elif "i like" in lowered or "i love" in lowered or "my favorite" in lowered:
                stype = "preference"
            elif "i decided" in lowered or "i chose" in lowered:
                stype = "decision"
            elif lowered.startswith("i wonder") or "?" in sentence:
                stype = "open_question"
            else:
                stype = "fact"
            units.append(
                {
                    "type": stype,
                    "title": self._titleize(sentence),
                    "content": sentence[:200],
                    "symbols": self._symbols(sentence),
                }
            )
        return units

    @staticmethod
    def _sentences(text: str) -> List[str]:
        raw = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in raw if s.strip()]

    def _symbols(self, sentence: str) -> List[str]:
        tokens = re.findall(r"[A-Za-z]{3,}", sentence.lower())
        unique: List[str] = []
        for token in tokens:
            if token in self._STOPWORDS:
                continue
            if token not in unique:
                unique.append(token)
            if len(unique) == 6:
                break
        if len(unique) < 2:
            unique.append("memory")
        return unique[:6]

    @staticmethod
    def _titleize(sentence: str) -> str:
        words = sentence.split()
        return " ".join(words[:6]).strip().rstrip(".,;:")


class ThreadAskResolver:
    """Creates a resilient `_thread_ask` callable."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg or {}
        self.fallback = None
        if self.cfg.get("fallback", "regex").lower() == "regex":
            self.fallback = RegexFallbackExtractor()

        self.primary: Optional[Callable[[str], str]] = None
        llm_cfg = self.cfg.get("llm") or {}
        endpoint = llm_cfg.get("endpoint")
        if endpoint:
            api_key = llm_cfg.get("api_key")
            api_key_env = llm_cfg.get("api_key_env")
            if not api_key and api_key_env:
                api_key = os.getenv(api_key_env)
            headers = llm_cfg.get("headers") or {}
            extra_payload = llm_cfg.get("extra_payload") or {}
            try:
                self.primary = HttpLLMExtractor(
                    endpoint=endpoint,
                    model=llm_cfg.get("model"),
                    api_key=api_key,
                    headers=headers,
                    timeout=int(llm_cfg.get("timeout", 15)),
                    temperature=float(llm_cfg.get("temperature", 0.0)),
                    system_prompt=llm_cfg.get("system_prompt"),
                    extra_payload=extra_payload,
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Failed to initialize HTTP extractor: %s", exc)
                self.primary = None

    def __call__(self, host_ask: Optional[Callable[[str], str]]) -> Callable[[str], str]:
        def ask(prompt: str) -> str:
            # 1. Host-provided callable
            if host_ask:
                try:
                    return host_ask(prompt)
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning("Host _thread_ask failed: %s", exc)

            # 2. Configured HTTP extractor
            if self.primary:
                try:
                    return self.primary(prompt)
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning("Primary extractor failed: %s", exc)

            # 3. Deterministic fallback
            if self.fallback:
                try:
                    return self.fallback(prompt)
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning("Fallback extractor failed: %s", exc)

            return "[]"

        return ask


def build_thread_ask(cfg: Dict[str, Any]) -> ThreadAskResolver:
    return ThreadAskResolver(cfg.get("extractor", {}) if cfg else {})
