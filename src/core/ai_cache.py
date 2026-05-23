"""
ai_cache.py — AI response caching + API connection testing.

The cache hashes (provider, model, config_content, context_block) -> response.
Avoids paying for the same query twice.
"""
import os
import json
import hashlib
import time
import urllib.request
import urllib.error
import urllib.parse


CACHE_FILE = os.path.join(os.path.expanduser("~"), ".usr_studio_ai_cache.json")
CACHE_MAX_AGE_DAYS = 30


def _cache_key(provider: str, model: str, prompt: str) -> str:
    """Build a SHA-256 key from the request params."""
    h = hashlib.sha256()
    h.update(provider.encode("utf-8"))
    h.update(b"|")
    h.update(model.encode("utf-8"))
    h.update(b"|")
    h.update(prompt.encode("utf-8"))
    return h.hexdigest()


def _load_cache() -> dict:
    if not os.path.exists(CACHE_FILE):
        return {}
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Prune expired entries
        now = time.time()
        max_age = CACHE_MAX_AGE_DAYS * 86400
        pruned = {k: v for k, v in data.items()
                  if now - v.get("ts", 0) < max_age}
        return pruned
    except Exception:
        return {}


def _save_cache(data: dict):
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[AI Cache] Erreur sauvegarde: {e}")


def get_cached_response(provider: str, model: str, prompt: str) -> str:
    """Return cached AI response or None if not cached."""
    cache = _load_cache()
    key = _cache_key(provider, model, prompt)
    entry = cache.get(key)
    if entry:
        return entry.get("response")
    return None


def store_response(provider: str, model: str, prompt: str, response: str):
    """Store an AI response in the cache."""
    cache = _load_cache()
    key = _cache_key(provider, model, prompt)
    cache[key] = {
        "ts": time.time(),
        "provider": provider,
        "model": model,
        "response": response,
    }
    _save_cache(cache)


def clear_cache() -> int:
    """Clear all cached responses. Returns count of entries removed."""
    cache = _load_cache()
    count = len(cache)
    if os.path.exists(CACHE_FILE):
        try:
            os.remove(CACHE_FILE)
        except Exception:
            pass
    return count


def cache_stats() -> dict:
    """Return stats about the cache."""
    cache = _load_cache()
    if not cache:
        return {"count": 0, "size_kb": 0, "providers": {}}
    providers = {}
    for entry in cache.values():
        p = entry.get("provider", "unknown")
        providers[p] = providers.get(p, 0) + 1
    size = os.path.getsize(CACHE_FILE) if os.path.exists(CACHE_FILE) else 0
    return {
        "count": len(cache),
        "size_kb": size / 1024,
        "providers": providers,
    }


# ─── API Connection Tester ────────────────────────────────────────────

def test_api_connection(provider: str, api_key: str, model: str, timeout: int = 15) -> tuple:
    """
    Test if an API key works by sending a minimal "Hello" request.

    Args:
        provider: Provider name (matches choices in tab_config)
        api_key: The API key to test
        model: Model id to use for the test
        timeout: Request timeout in seconds

    Returns:
        (success: bool, message: str)
    """
    if not api_key or len(api_key) < 5:
        return (False, "Cle API vide ou trop courte.")

    test_prompt = "Reply with exactly: OK"

    try:
        if "OpenRouter" in provider:
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/Universal-SR-Studio",
            }
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": test_prompt}],
                "max_tokens": 10,
            }
        elif "GitHub" in provider:
            url = "https://models.github.ai/inference/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "X-GitHub-Api-Version": "2022-11-28",
                "Accept": "application/vnd.github+json",
            }
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": test_prompt}],
                "max_tokens": 10,
            }
        elif "Anthropic" in provider:
            url = "https://api.anthropic.com/v1/messages"
            headers = {
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            }
            payload = {
                "model": model,
                "max_tokens": 10,
                "messages": [{"role": "user", "content": test_prompt}],
            }
        elif "OpenAI" in provider:
            url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": test_prompt}],
                "max_completion_tokens": 10,
            }
        elif "Google" in provider or "Gemini" in provider:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={urllib.parse.quote(api_key, safe='')}"
            headers = {"Content-Type": "application/json"}
            payload = {
                "contents": [{"parts": [{"text": test_prompt}]}],
                "generationConfig": {"maxOutputTokens": 10},
            }
        elif "xAI" in provider or "Grok" in provider:
            url = "https://api.x.ai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": test_prompt}],
                "max_tokens": 10,
            }
        elif "DeepSeek" in provider:
            url = "https://api.deepseek.com/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": test_prompt}],
                "max_tokens": 10,
            }
        else:
            return (False, f"Provider inconnu: {provider}")

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")

        start = time.time()
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            elapsed = (time.time() - start) * 1000
            body = resp.read().decode("utf-8")
            try:
                json.loads(body)  # Validate JSON
            except Exception:
                return (False, f"Reponse non-JSON ({resp.status})")
            return (True, f"OK — {elapsed:.0f} ms, modele: {model}")

    except urllib.error.HTTPError as e:
        try:
            err_body = e.read().decode("utf-8")[:300]
        except Exception:
            err_body = ""
        if e.code == 401:
            return (False, f"Cle API invalide (401 Unauthorized)")
        elif e.code == 403:
            return (False, f"Acces refuse (403). Verifiez vos permissions.")
        elif e.code == 404:
            return (False, f"Modele '{model}' non trouve (404). Verifiez le nom.")
        elif e.code == 429:
            return (False, f"Rate limit atteint (429). Reessayez plus tard.")
        else:
            return (False, f"HTTP {e.code}: {err_body[:150]}")
    except urllib.error.URLError as e:
        return (False, f"Erreur reseau: {e.reason}")
    except Exception as e:
        return (False, f"Erreur: {e}")
