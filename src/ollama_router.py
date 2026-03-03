"""
Ollama Router
=============
Routes AI-assistant @mentions directly to a local Ollama instance.

Supported trigger mentions (case-insensitive):
  @copilot  @lucidia  @blackboxprogramming  @ollama

All requests go to the local Ollama HTTP API – no external AI provider is used.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL: str = "http://localhost:11434"
DEFAULT_MODEL: str = "llama3"

# Mentions that trigger Ollama routing instead of any external AI provider
ROUTED_MENTIONS: frozenset[str] = frozenset(
    {"@copilot", "@lucidia", "@blackboxprogramming", "@ollama"}
)

_MENTION_RE = re.compile(
    r"(?i)\B(@(?:copilot|lucidia|blackboxprogramming|ollama))\b"
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class OllamaRequest:
    prompt: str
    model: str = DEFAULT_MODEL
    stream: bool = False
    options: dict = field(default_factory=dict)


@dataclass
class OllamaResponse:
    model: str
    response: str
    done: bool
    mention: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def detect_mention(text: str) -> Optional[str]:
    """Return the first routed @mention found in *text* (lower-cased), or None."""
    match = _MENTION_RE.search(text)
    return match.group(1).lower() if match else None


def strip_mention(text: str) -> str:
    """Remove all routed @mentions from *text* and return the clean prompt."""
    return _MENTION_RE.sub("", text).strip()


# ---------------------------------------------------------------------------
# Ollama API client
# ---------------------------------------------------------------------------

def call_ollama(
    prompt: str,
    model: str = DEFAULT_MODEL,
    base_url: str = OLLAMA_BASE_URL,
    options: Optional[dict] = None,
    timeout: int = 120,
) -> OllamaResponse:
    """Send *prompt* to the local Ollama API and return an OllamaResponse.

    Raises:
        ConnectionError: when Ollama is not reachable.
        ValueError:      when the API returns an unexpected response.
    """
    payload: dict = {"model": model, "prompt": prompt, "stream": False}
    if options:
        payload["options"] = options

    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{base_url}/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode())
    except urllib.error.URLError as exc:
        raise ConnectionError(
            f"Cannot reach Ollama at {base_url}. "
            "Ensure Ollama is running locally: https://ollama.com"
        ) from exc

    if not isinstance(body, dict):
        raise ValueError(f"Unexpected Ollama response: {body!r}")

    return OllamaResponse(
        model=body.get("model", model),
        response=body.get("response", ""),
        done=body.get("done", True),
    )


# ---------------------------------------------------------------------------
# Main routing entry point
# ---------------------------------------------------------------------------

def route(
    text: str,
    model: str = DEFAULT_MODEL,
    base_url: str = OLLAMA_BASE_URL,
    options: Optional[dict] = None,
) -> Optional[OllamaResponse]:
    """Route *text* to Ollama if it contains a supported @mention.

    Returns an OllamaResponse when a routed mention is found, or None when the
    message contains no recognized mention (so the caller can handle it
    differently or ignore it).

    Supported mentions: @copilot, @lucidia, @blackboxprogramming, @ollama
    """
    mention = detect_mention(text)
    if mention is None:
        return None

    prompt = strip_mention(text)
    response = call_ollama(prompt, model=model, base_url=base_url, options=options)
    response.mention = mention
    return response
