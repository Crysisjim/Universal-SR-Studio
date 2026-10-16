"""
ai_providers.py — single source of truth for the AI providers + model lists used
across the app (Config › Vérification IA, Training › Analyse IA, Publier Modèle).

AI_MODELS maps the provider display name → (settings_key, [models]). The settings
key is where the per-provider API key is stored (api_key_*).
"""

# Updated 2026-08-29 — GitHub Models was fully retired by GitHub on 2026-07-30
# (playground, model catalog, inference API and BYOK all shut down); the provider
# entry is removed entirely rather than kept as a permanently-broken dead option.
# Other lists refreshed to each provider's current lineup as of the same date —
# check the provider's own docs periodically, this rotates over time (especially
# OpenRouter's free tier and DeepSeek's legacy aliases, which get retired outright).
AI_MODELS = {
    # Refreshed 2026-10-29. OpenRouter + NVIDIA NIM: read from their public catalogs
    # (https://openrouter.ai/api/v1/models filtered on pricing.prompt == "0", and
    # https://integrate.api.nvidia.com/v1/models). Google / OpenAI / xAI / DeepSeek: from
    # each provider's own models documentation page. Anthropic: current Claude 5 family.
    # OpenRouter's free tier rotates constantly (5 of the 7 previous free models had left
    # it): "openrouter/free" is OpenRouter's own router to whatever free model is up, so
    # it stays valid when the others rotate out — it is the default.
    "OpenRouter (Gratuit)": ("api_key_openrouter", [
        "openrouter/free", "thinkingmachines/inkling:free", "thinkingmachines/inkling-small:free",
        "nvidia/nemotron-3-ultra-550b-a55b:free", "nvidia/nemotron-3.5-super-vl:free",
        "nvidia/nemotron-3.5-lightning:free", "google/gemma-4-31b-it:free",
        "google/gemma-4-26b-a4b-it:free"]),
    # gemini-2.5-* are now restricted to projects that already used them: left out.
    "Google (Gemini)": ("api_key_google", [
        "gemini-3.8-flash", "gemini-4-argon", "gemini-3.6-flash", "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite"]),
    "Anthropic (Claude)": ("api_key_anthropic", [
        "claude-sonnet-5-5", "claude-opus-5-5", "claude-haiku-5-5", "claude-fable-5-1"]),
    "OpenAI (ChatGPT)": ("api_key_openai", [
        "gpt-6.1-sol", "gpt-6-astra", "gpt-6.1-terra", "gpt-6-luna"]),
    "xAI (Grok)": ("api_key_xai", [
        "grok-4.7", "grok-4.6", "grok-4.5", "grok-4.3",
        "grok-4.20-0309-reasoning", "grok-4.20-0309-non-reasoning", "grok-build-0.1"]),
    # deepseek-v4-flash is a legacy alias routed to V4.1 Flash: the current name is used.
    "DeepSeek": ("api_key_deepseek", ["deepseek-v4-pro", "deepseek-flash"]),
    # deepseek-ai/deepseek-v4-pro-0813 left the NIM catalog.
    "NVIDIA NIM": ("api_key_nvidia", [
        "nvidia/llama-3.1-nemotron-70b-instruct", "nvidia/nemotron-3-ultra-550b-a55b",
        "nvidia/nemotron-3-super-120b-a12b", "nvidia/llama-3.1-nemotron-ultra-253b-v1",
        "deepseek-ai/deepseek-v4.1-flash", "moonshotai/kimi-k2.6",
        "mistralai/mistral-large-4-1t-a52b", "z-ai/glm-5.3-flash",
        "meta/llama-3.2-90b-vision-instruct"]),
}
