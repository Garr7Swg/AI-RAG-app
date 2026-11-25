"""LLM adapters.

Two providers, one interface:

``ollama``        - fully free and local. ``qwen2.5:3b-instruct`` (Q4) is ~2 GB,
                    fits a 6 GB RTX 2060, and is strong enough for grounded
                    extraction. This is the default.
``openai_compat`` - any OpenAI-compatible endpoint (Groq / OpenRouter / Freebuff)
                    for when there is no GPU or no local Ollama, e.g. Streamlit
                    Community Cloud. Groq is the default hosted provider.

Ollama also exposes an OpenAI-compatible ``/v1`` endpoint, which is what the
Ragas judge uses - so evaluation needs no extra SDK or paid key.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterator, Sequence

from .config import Settings
from .logging_utils import get_logger, redact

log = get_logger("llm")


class LLMError(RuntimeError):
    """User-presentable LLM failure (never contains secrets)."""


def _endpoint_error(model: str, exc: Exception) -> LLMError:
    """Turn provider exceptions into an actionable message.

    Hosted providers retire model IDs (Groq removed ``llama-3.1-8b-instant``
    from its free tier in August 2026), so a bare provider name is not enough
    to debug a failure.
    """
    detail = str(exc).lower()
    if any(token in detail for token in ("decommission", "model_not_found", "does not exist", "no longer supported")):
        return LLMError(
            f"Model '{model}' is no longer served by this endpoint. "
            "Set RAG_OPENAI_MODEL to a current one (Groq free tier: "
            "openai/gpt-oss-20b or openai/gpt-oss-120b)."
        )
    return LLMError(f"LLM request failed ({type(exc).__name__}). Check the API key and base URL.")


@dataclass
class Message:
    role: str
    content: str

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


class BaseLLM:
    name = "base"
    model = ""

    def health(self) -> tuple[bool, str]:
        raise NotImplementedError

    def complete(self, messages: Sequence[Message], temperature: float | None = None) -> str:
        raise NotImplementedError

    def stream(self, messages: Sequence[Message], temperature: float | None = None) -> Iterator[str]:
        yield self.complete(messages, temperature=temperature)


# ---------------------------------------------------------------------- ollama
class OllamaLLM(BaseLLM):
    name = "ollama"

    def __init__(self, base_url: str, model: str, timeout_s: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self._resolved: str | None = None

    def _installed(self) -> list[str]:
        import httpx

        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(f"{self.base_url}/api/tags")
                if response.status_code != 200:
                    return []
                return [str(m.get("name", "")) for m in response.json().get("models", [])]
        except Exception:
            return []

    def resolve_model(self) -> str:
        """Map the configured name onto an installed tag.

        ``qwen2.5:3b-instruct`` is commonly pulled as
        ``qwen2.5:3b-instruct-q4_K_M``. Rather than fail with a 404, use the
        quantisation variant of the same model when it is the only option.
        """
        if self._resolved:
            return self._resolved
        names = self._installed()
        if not names:
            return self.model
        if self.model in names:
            self._resolved = self.model
        else:
            prefixed = [n for n in names if n.startswith(self.model + "-")]
            family = [n for n in names if n.split(":")[0] == self.model.split(":")[0]]
            chosen = (prefixed or family or [self.model])[0]
            if chosen != self.model:
                log.info("using installed Ollama tag '%s' for '%s'", chosen, self.model)
            self._resolved = chosen
        return self._resolved

    def _payload(self, messages: Sequence[Message], stream: bool, temperature: float | None) -> dict[str, Any]:
        return {
            "model": self.resolve_model(),
            "messages": [m.as_dict() for m in messages],
            "stream": stream,
            "options": {"temperature": temperature, "num_predict": 512},
        }

    def health(self) -> tuple[bool, str]:
        names = self._installed()
        if not names:
            return False, "Ollama is not reachable on this machine."
        resolved = self.resolve_model()
        if resolved not in names:
            return False, f"Model '{self.model}' is not pulled. Run: ollama pull {self.model}"
        label = resolved if resolved == self.model else f"installed as {resolved}"
        return True, f"Ollama ready ({label})."

    def complete(self, messages: Sequence[Message], temperature: float | None = None) -> str:
        import httpx

        try:
            with httpx.Client(timeout=self.timeout_s) as client:
                response = client.post(
                    f"{self.base_url}/api/chat",
                    json=self._payload(messages, False, temperature),
                )
        except Exception as exc:
            raise LLMError(
                "Could not reach the local model. Is Ollama running? "
                "Start it with `ollama serve`, or switch RAG_LLM_PROVIDER=openai_compat."
            ) from exc
        if response.status_code >= 400:
            detail = redact(response.text, 300)
            if response.status_code == 404:
                raise LLMError(
                    f"Model '{self.model}' is not available. Run: ollama pull {self.model}"
                )
            raise LLMError(f"Ollama returned HTTP {response.status_code}: {detail}")
        try:
            return str(response.json().get("message", {}).get("content", "")).strip()
        except json.JSONDecodeError as exc:  # pragma: no cover
            raise LLMError("Ollama returned a malformed response.") from exc

    def stream(self, messages: Sequence[Message], temperature: float | None = None) -> Iterator[str]:
        import httpx

        try:
            with httpx.Client(timeout=self.timeout_s) as client:
                with client.stream(
                    "POST", f"{self.base_url}/api/chat", json=self._payload(messages, True, temperature)
                ) as response:
                    if response.status_code >= 400:
                        response.read()
                        raise LLMError(f"Ollama returned HTTP {response.status_code}.")
                    for line in response.iter_lines():
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        piece = (data.get("message") or {}).get("content") or ""
                        if piece:
                            yield piece
                        if data.get("done"):
                            break
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError("The local model connection dropped mid-answer.") from exc


# ------------------------------------------------------------ openai-compatible
class OpenAICompatLLM(BaseLLM):
    name = "openai_compat"

    def __init__(self, base_url: str, model: str, api_key: str, timeout_s: float = 120.0) -> None:
        self.base_url = base_url
        self.model = model
        self._api_key = api_key
        self.timeout_s = timeout_s
        self._client = None

    def _lc(self):
        if self._client is not None:
            return self._client
        if not self._api_key:
            raise LLMError("RAG_OPENAI_API_KEY is not set for the openai_compat provider.")
        try:
            from langchain_openai import ChatOpenAI
        except Exception as exc:
            raise LLMError("langchain-openai is not installed.") from exc
        self._client = ChatOpenAI(
            model=self.model,
            base_url=self.base_url,
            api_key=self._api_key,
            temperature=0.0,
            timeout=self.timeout_s,
            max_retries=2,
        )
        return self._client

    def health(self) -> tuple[bool, str]:
        if not self._api_key:
            return False, "No API key configured for the OpenAI-compatible endpoint."
        try:
            self._lc().invoke([("human", "Reply with the single word: ready")])
            return True, f"{self.model} reachable."
        except Exception as exc:
            return False, str(_endpoint_error(self.model, exc))

    def _lc_messages(self, messages: Sequence[Message]) -> list[tuple[str, str]]:
        mapping = {"system": "system", "assistant": "ai", "user": "human"}
        return [(mapping.get(m.role, "human"), m.content) for m in messages]

    def complete(self, messages: Sequence[Message], temperature: float | None = None) -> str:
        try:
            result = self._lc().invoke(self._lc_messages(messages))
        except LLMError:
            raise
        except Exception as exc:
            raise _endpoint_error(self.model, exc) from exc
        content = getattr(result, "content", "")
        if isinstance(content, list):  # some providers return content parts
            content = "".join(str(part.get("text", "")) if isinstance(part, dict) else str(part) for part in content)
        return str(content).strip()

    def stream(self, messages: Sequence[Message], temperature: float | None = None) -> Iterator[str]:
        try:
            for chunk in self._lc().stream(self._lc_messages(messages)):
                text = getattr(chunk, "content", "")
                if isinstance(text, list):
                    text = "".join(str(p.get("text", "")) if isinstance(p, dict) else str(p) for p in text)
                if text:
                    yield str(text)
        except LLMError:
            raise
        except Exception as exc:
            raise _endpoint_error(self.model, exc) from exc


def get_llm(settings: Settings) -> BaseLLM:
    if settings.llm_provider == "openai_compat":
        return OpenAICompatLLM(
            base_url=settings.openai_base_url,
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            timeout_s=settings.llm_timeout_s,
        )
    return OllamaLLM(settings.ollama_url, settings.ollama_model, timeout_s=settings.llm_timeout_s)


def get_chat_model(settings: Settings, temperature: float = 0.0):
    """A LangChain chat model for the Ragas judge.

    For Ollama we use its OpenAI-compatible ``/v1`` endpoint, so the local model
    can judge the pipeline for free.
    """
    try:
        from langchain_openai import ChatOpenAI
    except Exception as exc:  # pragma: no cover
        raise LLMError("langchain-openai is not installed.") from exc

    if settings.llm_provider == "ollama":
        # Resolve quantisation suffixes (qwen2.5:3b-instruct -> ...-q4_K_M) so the
        # judge uses a tag that actually exists.
        model = OllamaLLM(settings.ollama_url, settings.ollama_model).resolve_model()
        return ChatOpenAI(
            model=model,
            base_url=f"{settings.ollama_url}/v1",
            api_key="ollama",  # local only, not a secret
            temperature=temperature,
            timeout=settings.llm_timeout_s,
            max_retries=2,
        )
    if not settings.openai_api_key:
        raise LLMError("RAG_OPENAI_API_KEY is required for evaluation with the openai_compat provider.")
    return ChatOpenAI(
        model=settings.openai_model,
        base_url=settings.openai_base_url,
        api_key=settings.openai_api_key,
        temperature=temperature,
        timeout=settings.llm_timeout_s,
        max_retries=2,
    )
