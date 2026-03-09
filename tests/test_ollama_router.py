"""Tests for the Ollama router."""
import json
from unittest.mock import MagicMock, patch

import pytest

from ollama_router import (
    DEFAULT_MODEL,
    ROUTED_MENTIONS,
    OllamaResponse,
    call_ollama,
    detect_mention,
    route,
    strip_mention,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_urlopen(response_text: str = "Hello!", model: str = DEFAULT_MODEL):
    """Return a context-manager mock that simulates a successful Ollama response."""
    body = json.dumps({"model": model, "response": response_text, "done": True}).encode()
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=MagicMock(read=MagicMock(return_value=body)))
    cm.__exit__ = MagicMock(return_value=False)
    return cm


# ---------------------------------------------------------------------------
# ROUTED_MENTIONS
# ---------------------------------------------------------------------------

class TestRoutedMentions:
    def test_contains_copilot(self):
        assert "@copilot" in ROUTED_MENTIONS

    def test_contains_lucidia(self):
        assert "@lucidia" in ROUTED_MENTIONS

    def test_contains_blackboxprogramming(self):
        assert "@blackboxprogramming" in ROUTED_MENTIONS

    def test_contains_ollama(self):
        assert "@ollama" in ROUTED_MENTIONS

    def test_no_external_providers(self):
        # External AI providers must NOT be in the routing table
        for provider in ("@chatgpt", "@claude", "@openai", "@anthropic"):
            assert provider not in ROUTED_MENTIONS


# ---------------------------------------------------------------------------
# detect_mention
# ---------------------------------------------------------------------------

class TestDetectMention:
    def test_detects_at_copilot(self):
        assert detect_mention("@copilot explain this") == "@copilot"

    def test_detects_at_lucidia(self):
        assert detect_mention("@lucidia what is k8s?") == "@lucidia"

    def test_detects_at_blackboxprogramming(self):
        assert detect_mention("@blackboxprogramming help me") == "@blackboxprogramming"

    def test_detects_at_ollama(self):
        assert detect_mention("@ollama write a poem") == "@ollama"

    def test_case_insensitive(self):
        assert detect_mention("@Copilot do something") == "@copilot"
        assert detect_mention("@OLLAMA do something") == "@ollama"

    def test_returns_none_for_no_mention(self):
        assert detect_mention("just a plain message") is None

    def test_returns_none_for_unknown_mention(self):
        assert detect_mention("@chatgpt explain this") is None

    def test_returns_first_mention(self):
        result = detect_mention("@copilot and @ollama help")
        assert result == "@copilot"


# ---------------------------------------------------------------------------
# strip_mention
# ---------------------------------------------------------------------------

class TestStripMention:
    def test_strips_copilot(self):
        assert strip_mention("@copilot explain this") == "explain this"

    def test_strips_lucidia(self):
        assert strip_mention("@lucidia what is k8s?") == "what is k8s?"

    def test_strips_blackboxprogramming(self):
        assert strip_mention("@blackboxprogramming help me") == "help me"

    def test_strips_ollama(self):
        assert strip_mention("@ollama write a poem") == "write a poem"

    def test_strips_multiple_mentions(self):
        result = strip_mention("@copilot and @ollama help me")
        assert "@copilot" not in result
        assert "@ollama" not in result
        assert "help me" in result

    def test_passthrough_when_no_mention(self):
        assert strip_mention("no mention here") == "no mention here"

    def test_strips_case_insensitive(self):
        assert strip_mention("@Copilot do this") == "do this"


# ---------------------------------------------------------------------------
# call_ollama
# ---------------------------------------------------------------------------

class TestCallOllama:
    def test_returns_ollama_response(self):
        with patch("urllib.request.urlopen", return_value=_mock_urlopen("Hi there!")):
            resp = call_ollama("Hello", base_url="http://localhost:11434")
        assert isinstance(resp, OllamaResponse)
        assert resp.response == "Hi there!"
        assert resp.done is True

    def test_uses_specified_model(self):
        with patch("urllib.request.urlopen", return_value=_mock_urlopen(model="mistral")):
            resp = call_ollama("Hello", model="mistral", base_url="http://localhost:11434")
        assert resp.model == "mistral"

    def test_raises_connection_error_on_url_error(self):
        import urllib.error
        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("connection refused"),
        ):
            with pytest.raises(ConnectionError, match="Cannot reach Ollama"):
                call_ollama("Hello", base_url="http://localhost:11434")

    def test_passes_options_in_payload(self):
        captured = {}

        def fake_urlopen(req, timeout=None):
            captured["data"] = json.loads(req.data.decode())
            return _mock_urlopen()

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            call_ollama("Hi", options={"temperature": 0.7}, base_url="http://localhost:11434")

        assert captured["data"]["options"] == {"temperature": 0.7}

    def test_request_sent_to_generate_endpoint(self):
        captured = {}

        def fake_urlopen(req, timeout=None):
            captured["url"] = req.full_url
            return _mock_urlopen()

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            call_ollama("Hi", base_url="http://localhost:11434")

        assert captured["url"] == "http://localhost:11434/api/generate"


# ---------------------------------------------------------------------------
# route
# ---------------------------------------------------------------------------

class TestRoute:
    def test_routes_copilot_to_ollama(self):
        with patch("urllib.request.urlopen", return_value=_mock_urlopen("pong")):
            resp = route("@copilot ping", base_url="http://localhost:11434")
        assert resp is not None
        assert resp.response == "pong"
        assert resp.mention == "@copilot"

    def test_routes_lucidia_to_ollama(self):
        with patch("urllib.request.urlopen", return_value=_mock_urlopen("ok")):
            resp = route("@lucidia help", base_url="http://localhost:11434")
        assert resp is not None
        assert resp.mention == "@lucidia"

    def test_routes_blackboxprogramming_to_ollama(self):
        with patch("urllib.request.urlopen", return_value=_mock_urlopen("sure")):
            resp = route("@blackboxprogramming build this", base_url="http://localhost:11434")
        assert resp is not None
        assert resp.mention == "@blackboxprogramming"

    def test_routes_ollama_to_ollama(self):
        with patch("urllib.request.urlopen", return_value=_mock_urlopen("done")):
            resp = route("@ollama generate code", base_url="http://localhost:11434")
        assert resp is not None
        assert resp.mention == "@ollama"

    def test_returns_none_when_no_mention(self):
        resp = route("just a plain message", base_url="http://localhost:11434")
        assert resp is None

    def test_strips_mention_before_sending_prompt(self):
        captured = {}

        def fake_urlopen(req, timeout=None):
            captured["payload"] = json.loads(req.data.decode())
            return _mock_urlopen()

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            route("@copilot explain kubernetes", base_url="http://localhost:11434")

        assert "@copilot" not in captured["payload"]["prompt"]
        assert "explain kubernetes" in captured["payload"]["prompt"]

    def test_no_external_provider_called(self):
        """Ensure route() never calls an external AI provider URL."""
        captured_urls = []

        def fake_urlopen(req, timeout=None):
            captured_urls.append(req.full_url)
            return _mock_urlopen()

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            route("@ollama hello", base_url="http://localhost:11434")

        for url in captured_urls:
            assert "openai" not in url
            assert "anthropic" not in url
            assert "copilot" not in url
            assert "claude" not in url
