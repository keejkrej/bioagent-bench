"""Minimal OpenAI-compatible client for Vercel AI Gateway (stdlib only)."""

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional


DEFAULT_BASE_URL = "https://ai-gateway.vercel.sh/v1"
DEFAULT_MODEL = "anthropic/claude-haiku-5.5"


class GatewayError(RuntimeError):
    pass


class GatewayClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout_seconds: int = 300,
        max_attempts: int = 5,
    ) -> None:
        self.api_key = api_key or os.environ.get("AI_GATEWAY_API_KEY") or os.environ.get("VERCEL_OIDC_TOKEN")
        if not self.api_key:
            raise GatewayError("Set AI_GATEWAY_API_KEY (or VERCEL_OIDC_TOKEN) before running an agent")
        self.base_url = (base_url or os.environ.get("AI_GATEWAY_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max_attempts

    def chat(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        max_tokens: int = 8192,
        deadline: Optional[float] = None,
    ) -> Dict[str, Any]:
        payload = {
            "model": model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "max_tokens": max_tokens,
        }
        body = json.dumps(payload).encode("utf-8")
        for attempt in range(1, self.max_attempts + 1):
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                raise GatewayError("Agent deadline reached before a gateway request completed")
            request_timeout = self.timeout_seconds if remaining is None else min(self.timeout_seconds, remaining)
            request = urllib.request.Request(
                self.base_url + "/chat/completions",
                data=body,
                headers={
                    "Authorization": "Bearer " + self.api_key,
                    "Content-Type": "application/json",
                    "User-Agent": "bioagent-bench/0.1",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=request_timeout) as response:
                    result = json.load(response)
                if not result.get("choices"):
                    raise GatewayError("Gateway response contained no choices")
                return result
            except urllib.error.HTTPError as exc:
                detail = exc.read(4096).decode("utf-8", errors="replace")
                retryable = exc.code == 429 or exc.code >= 500
                if not retryable or attempt == self.max_attempts:
                    raise GatewayError("Gateway HTTP %s: %s" % (exc.code, detail)) from exc
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                if attempt == self.max_attempts:
                    raise GatewayError("Gateway request failed: %s" % exc) from exc
            delay = min(2 ** (attempt - 1), 16)
            if remaining is not None:
                delay = min(delay, max(0, deadline - time.monotonic()))
            if delay <= 0:
                raise GatewayError("Agent deadline reached during gateway retry backoff")
            time.sleep(delay)
        raise GatewayError("Gateway request exhausted retries")
