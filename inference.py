import os
import time
import requests
from abc import ABC, abstractmethod
from typing import Optional


class ChatRunner(ABC):
    """Abstract base class for running chat models."""

    @abstractmethod
    def generate(
        self,
        messages: list[dict],
        max_tokens: int = 1000,
        temperature: float = 0.7,
        seed: int = 1234,
        response_format: Optional[dict] = None,
    ) -> dict:
        """Generate a response given a list of messages.

        Must return a dict with keys:
            text, prompt_tokens, completion_tokens, total_tokens,
            latency_ms, model, backend, finish_reason
        """
        pass


class LlamaCppRunner(ChatRunner):
    """ChatRunner backed by a llama-cpp-python Llama instance."""

    def __init__(self, model_path: str, n_ctx: int = 2048, n_threads: int = 4, verbose: bool = False):
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise RuntimeError(
                f"llama-cpp-python failed to import: {exc}\n"
                "Install it: pip install llama-cpp-python --extra-index-url "
                "https://abetlen.github.io/llama-cpp-python/whl/cpu"
            ) from exc
        self._llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            verbose=verbose,
        )

    def generate(
        self,
        messages: list[dict],
        max_tokens: int = 1000,
        temperature: float = 0.7,
        seed: int = 1234,
        response_format: Optional[dict] = None,
    ) -> dict:
        t0 = time.perf_counter()

        args = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "seed": seed,
        }
        if response_format:
            args["response_format"] = response_format

        response = self._llm.create_chat_completion(**args)

        t1 = time.perf_counter()

        usage = response.get("usage", {})
        choices = response.get("choices", [{}])
        choice = choices[0] if choices else {}
        message = choice.get("message", {})

        return {
            "text": message.get("content", ""),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "latency_ms": int((t1 - t0) * 1000),
            "model": response.get("model", "unknown-llama-cpp"),
            "backend": "llama_cpp",
            "finish_reason": choice.get("finish_reason", "unknown"),
        }


class OpenAICompatibleRunner(ChatRunner):
    """ChatRunner backed by any OpenAI-compatible REST API (vLLM, LM Studio, etc.)."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model_id: str,
        timeout: int = 120,
        retries: int = 3,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model_id
        self.timeout = timeout
        self.retries = retries

    def generate(
        self,
        messages: list[dict],
        max_tokens: int = 1000,
        temperature: float = 0.7,
        seed: int = 1234,
        response_format: Optional[dict] = None,
    ) -> dict:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "seed": seed,
        }
        if response_format:
            payload["response_format"] = response_format

        endpoint = f"{self.base_url}/chat/completions"

        last_err = None
        for attempt in range(self.retries):
            t0 = time.perf_counter()
            try:
                resp = requests.post(endpoint, headers=headers, json=payload, timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()
                t1 = time.perf_counter()

                usage = data.get("usage", {})
                choices = data.get("choices", [{}])
                choice = choices[0] if choices else {}
                message = choice.get("message", {})

                return {
                    "text": message.get("content", ""),
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                    "latency_ms": int((t1 - t0) * 1000),
                    "model": data.get("model", self.model),
                    "backend": "openai_compatible",
                    "finish_reason": choice.get("finish_reason", "unknown"),
                }
            except Exception as e:
                last_err = str(e)
                time.sleep(2 ** attempt)

        return {
            "text": "",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "latency_ms": 0,
            "model": self.model,
            "backend": "openai_compatible",
            "finish_reason": "error",
            "error": last_err,
        }
