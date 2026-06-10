import os
import time
import logging
from typing import Optional

from openai import OpenAI

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
        proxy: Optional[str] = None,
        max_retries: int = 3,
    ):
        self.model = model
        self.max_retries = max_retries
        kwargs = {}
        if proxy:
            kwargs["http_client"] = _build_proxied_client(proxy)
        self.client = OpenAI(
            api_key=api_key or os.environ.get("DEEPSEEK_API_KEY", ""),
            base_url=base_url,
            **kwargs,
        )

    def generate(
        self,
        prompt: str,
        system: str = "You are a helpful programming assistant. Return only code, no explanation.",
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> dict:
        last_error = None
        for attempt in range(self.max_retries):
            try:
                start = time.time()
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                elapsed = time.time() - start
                usage = response.usage
                choice = response.choices[0].message
                content = choice.content or ""
                logger.debug(
                    "LLM generation: %.1fs, prompt_tokens=%d, completion_tokens=%d",
                    elapsed,
                    usage.prompt_tokens if usage else 0,
                    usage.completion_tokens if usage else 0,
                )
                return {
                    "code": _extract_code(content),
                    "raw_response": content,
                    "prompt_tokens": usage.prompt_tokens if usage else 0,
                    "completion_tokens": usage.completion_tokens if usage else 0,
                    "wall_time_s": elapsed,
                }
            except Exception as e:
                last_error = e
                logger.warning("LLM API attempt %d/%d failed: %s", attempt + 1, self.max_retries, e)
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"LLM generation failed after {self.max_retries} attempts: {last_error}")


def _extract_code(raw: str) -> str:
    if "```python" in raw:
        parts = raw.split("```python", 1)
        if len(parts) > 1:
            inner = parts[1].split("```", 1)
            return inner[0].strip()
    if "```" in raw:
        parts = raw.split("```", 1)
        if len(parts) > 1:
            inner = parts[1].split("```", 1)
            return inner[0].strip()
    return raw.strip()


def _build_proxied_client(proxy_url: str):
    import httpx

    return httpx.Client(proxy=proxy_url)
