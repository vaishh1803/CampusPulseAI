"""Thin Ollama client for IBM Granite. Every call fails soft so agents can fall back to rules."""
import json
import logging
import os

import requests

log = logging.getLogger("campuspulse")
URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
CHAT_MODEL = os.getenv("GRANITE_MODEL", "granite3.3:8b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "granite-embedding:278m")
MOCK = os.getenv("MOCK_LLM", "0") == "1"


def chat_json(prompt, retries=2):
    """Ask Granite for a JSON object. Returns a dict, or None if unavailable/malformed."""
    if MOCK:
        return None
    for _ in range(retries + 1):
        try:
            r = requests.post(f"{URL}/api/chat", timeout=60, json={
                "model": CHAT_MODEL, "stream": False, "format": "json",
                "messages": [{"role": "user", "content": prompt}]})
            out = json.loads(r.json()["message"]["content"])
            if isinstance(out, dict):
                return out
        except Exception as e:  # network down, bad JSON, model missing
            log.warning("Granite call failed: %s", e)
    return None


def embed(text):
    """Return an embedding vector, or None so callers use the keyword fallback."""
    if MOCK:
        return None
    try:
        r = requests.post(f"{URL}/api/embeddings", timeout=30, json={"model": EMBED_MODEL, "prompt": text})
        return r.json()["embedding"]
    except Exception as e:
        log.warning("Embedding failed: %s", e)
        return None
