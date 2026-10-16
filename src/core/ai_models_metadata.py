"""
ai_models_metadata.py — Descriptions and metadata for AI models used in config check.

Kept in sync with src/core/ai_providers.py — every model_id below should exist in
AI_MODELS there. Rewritten 2026-08-29 (old ids were all stale: GitHub Models fully
retired, OpenRouter's free tier had rotated out every previous entry, DeepSeek's
legacy chat/reasoner aliases retired, NVIDIA NIM's llama-3.3/3.1-*-instruct EOL'd —
see ai_providers.py's own comments for how each replacement was verified live).
"""

# Detailed descriptions for the model picker tooltip.
# Format: model_id -> "Short label : detailed description"
AI_MODEL_DESCRIPTIONS = {
    'openrouter/free': "Free Models Router (OpenRouter)\n  Routeur GRATUIT d'OpenRouter : envoie la requête au modèle gratuit disponible.\n  Reste valable quand la liste des modèles gratuits change (recommandé).",
    'thinkingmachines/inkling:free': 'Inkling (Free via OpenRouter)\n  Modèle Thinking Machines GRATUIT, contexte 1M tokens.\n  Bon généraliste, raisonnement.',
    'thinkingmachines/inkling-small:free': "Inkling Small (Free via OpenRouter)\n  Version réduite et plus rapide d'Inkling, GRATUITE.",
    'nvidia/nemotron-3-ultra-550b-a55b:free': 'Nemotron 3 Ultra 550B (Free via OpenRouter)\n  Très grand modèle NVIDIA GRATUIT.\n  Le plus capable de la gamme Nemotron 3 gratuite, plus lent.',
    'nvidia/nemotron-3.5-super-vl:free': 'Nemotron 3.5 Super VL (Free via OpenRouter)\n  Modèle NVIDIA GRATUIT texte + image, contexte 1M tokens.',
    'nvidia/nemotron-3.5-lightning:free': 'Nemotron 3.5 Lightning (Free via OpenRouter)\n  Modèle NVIDIA GRATUIT rapide, contexte 1M tokens.',
    'google/gemma-4-31b-it:free': 'Gemma 4 31B (Free via OpenRouter)\n  Modèle Google open-weights GRATUIT.\n  Bon reasoning et coding.',
    'google/gemma-4-26b-a4b-it:free': 'Gemma 4 26B MoE (Free via OpenRouter)\n  Modèle Google MoE GRATUIT (architecture sparse).\n  Rapide malgré sa taille.',
    'gemini-3.8-flash': 'Gemini 3.8 Flash (Google)\n  Flash le plus intelligent de Google, recommandé pour les nouveaux projets.',
    'gemini-4-argon': "Gemini 4 Argon (Google)\n  Flagship Google orienté raisonnement, contexte 1M tokens.\n  Recommandé pour l'analyse de configs complexes.",
    'gemini-3.6-flash': 'Gemini 3.6 Flash (Google)\n  Flash de la génération précédente.',
    'gemini-3.5-flash-lite': 'Gemini 3.5 Flash-Lite (Google)\n  Le plus rapide et économique de la série 3.5.',
    'gemini-3.1-flash-lite': 'Gemini 3.1 Flash-Lite (Google)\n  Modèle léger et économique.',
    'claude-sonnet-5-5': 'Claude Sonnet 5.5 (Anthropic)\n  Équilibre vitesse / intelligence.\n  Bon défaut pour la plupart des tâches.',
    'claude-opus-5-5': "Claude Opus 5.5 (Anthropic)\n  Modèle le plus avancé d'Anthropic.\n  Excellent en analyse fine, raisonnement complexe.\n  Recommandé pour l'évaluation de configs avancées.",
    'claude-haiku-5-5': 'Claude Haiku 5.5 (Anthropic)\n  Modèle rapide et économique.\n  Bon pour une vérification simple.',
    'claude-fable-5-1': 'Claude Fable 5.1 (Anthropic)\n  Modèle Fable de la famille Claude 5.',
    'gpt-6.1-sol': "GPT-6.1 Sol (OpenAI)\n  Milieu de gamme GPT-6 : bon équilibre capacité / coût.\n  Bon défaut pour l'analyse de config.",
    'gpt-6-astra': 'GPT-6 Astra (OpenAI)\n  Flagship GPT-6, le plus capable (et le plus cher).',
    'gpt-6.1-terra': 'GPT-6.1 Terra (OpenAI)\n  Équilibre intelligence / coût de la famille GPT-6.',
    'gpt-6-luna': 'GPT-6 Luna (OpenAI)\n  Le plus économique de la famille GPT-6, gros volume.',
    'grok-4.7': 'Grok 4.7 (xAI)\n  Flagship xAI (septembre 2026), contexte 500K tokens.\n  Code, tâches agentiques.',
    'grok-4.6': 'Grok 4.6 (xAI)\n  Flagship précédent, contexte 500K tokens.',
    'grok-4.5': 'Grok 4.5 (xAI)\n  Orienté code et workflows agentiques.',
    'grok-4.3': 'Grok 4.3 (xAI)\n  Contexte 1M tokens, rapide, bon rapport qualité/prix.',
    'grok-4.20-0309-reasoning': 'Grok 4.20 Reasoning (xAI)\n  Contexte 1M tokens, variante avec raisonnement.',
    'grok-4.20-0309-non-reasoning': 'Grok 4.20 Non-Reasoning (xAI)\n  Contexte 1M tokens, variante rapide sans raisonnement.',
    'grok-build-0.1': 'Grok Build 0.1 (xAI)\n  Modèle spécialisé code, contexte 256K tokens.',
    'deepseek-v4-pro': 'DeepSeek V4 Pro (DeepSeek)\n  Flagship DeepSeek pour le raisonnement complexe et le code agentique.',
    'deepseek-flash': 'DeepSeek V4.1 Flash (DeepSeek)\n  Rapide et économique, contexte 1M tokens, entrée image.',
    'nvidia/llama-3.1-nemotron-70b-instruct': 'Llama 3.1 Nemotron 70B (NVIDIA NIM)\n  Llama 3.1 affiné par NVIDIA (Nemotron).',
    'nvidia/nemotron-3-ultra-550b-a55b': 'Nemotron 3 Ultra 550B (NVIDIA NIM)\n  Très grand modèle NVIDIA, le plus capable de la gamme Nemotron 3.',
    'nvidia/nemotron-3-super-120b-a12b': 'Nemotron 3 Super 120B (NVIDIA NIM)\n  Modèle NVIDIA 120B, excellent en raisonnement et analyse.',
    'nvidia/llama-3.1-nemotron-ultra-253b-v1': 'Llama 3.1 Nemotron Ultra 253B (NVIDIA NIM)\n  Très grand modèle affiné par NVIDIA.\n  Excellent en raisonnement, plus lent.',
    'deepseek-ai/deepseek-v4.1-flash': "DeepSeek V4.1 Flash (NVIDIA NIM)\n  DeepSeek V4.1 Flash hébergé sur l'infra NVIDIA.",
    'moonshotai/kimi-k2.6': 'Kimi K2.6 (NVIDIA NIM)\n  Modèle Moonshot AI, bon en code et raisonnement.',
    'mistralai/mistral-large-4-1t-a52b': 'Mistral Large 4 (NVIDIA NIM)\n  Flagship Mistral (MoE 1T), bon en code et instructions.',
    'z-ai/glm-5.3-flash': 'GLM-5.3 Flash (NVIDIA NIM)\n  Modèle Zhipu AI rapide.',
    'meta/llama-3.2-90b-vision-instruct': 'Llama 3.2 90B Vision (NVIDIA NIM)\n  Modèle Meta multimodal (texte + image).',
}


# English body of each description (the first line, the model's name, is shared).
AI_MODEL_DESCRIPTIONS_EN = {
    'openrouter/free': "OpenRouter's FREE router: sends the request to whichever free model is available.\n  Stays valid when the free model list rotates (recommended).",
    'thinkingmachines/inkling:free': 'FREE Thinking Machines model, 1M-token context.\n  Good generalist, reasoning.',
    'thinkingmachines/inkling-small:free': 'Smaller, faster Inkling, FREE.',
    'nvidia/nemotron-3-ultra-550b-a55b:free': 'Very large FREE NVIDIA model.\n  The most capable of the free Nemotron 3 range, slower.',
    'nvidia/nemotron-3.5-super-vl:free': 'FREE NVIDIA text + image model, 1M-token context.',
    'nvidia/nemotron-3.5-lightning:free': 'Fast FREE NVIDIA model, 1M-token context.',
    'google/gemma-4-31b-it:free': 'FREE Google open-weights model.\n  Good reasoning and coding.',
    'google/gemma-4-26b-a4b-it:free': 'FREE Google MoE model (sparse architecture).\n  Fast for its size.',
    'gemini-3.8-flash': "Google's most intelligent Flash, recommended for new projects.",
    'gemini-4-argon': "Google's reasoning flagship, 1M-token context.\n  Recommended for complex config analysis.",
    'gemini-3.6-flash': 'Previous-generation Flash.',
    'gemini-3.5-flash-lite': 'The fastest and cheapest of the 3.5 series.',
    'gemini-3.1-flash-lite': 'Light, cheap model.',
    'claude-sonnet-5-5': 'Speed / intelligence balance.\n  A good default for most tasks.',
    'claude-opus-5-5': "Anthropic's most advanced model.\n  Excellent at fine analysis and complex reasoning.\n  Recommended for advanced config reviews.",
    'claude-haiku-5-5': 'Fast and cheap model.\n  Good for a simple check.',
    'claude-fable-5-1': 'Fable model of the Claude 5 family.',
    'gpt-6.1-sol': 'GPT-6 mid tier: good capability / cost balance.\n  A good default for config analysis.',
    'gpt-6-astra': 'GPT-6 flagship, the most capable (and most expensive).',
    'gpt-6.1-terra': 'Intelligence / cost balance of the GPT-6 family.',
    'gpt-6-luna': 'The cheapest of the GPT-6 family, high volume.',
    'grok-4.7': 'xAI flagship (September 2026), 500K-token context.\n  Code, agentic tasks.',
    'grok-4.6': 'Previous flagship, 500K-token context.',
    'grok-4.5': 'Oriented to code and agentic workflows.',
    'grok-4.3': '1M-token context, fast, good quality/price ratio.',
    'grok-4.20-0309-reasoning': '1M-token context, reasoning variant.',
    'grok-4.20-0309-non-reasoning': '1M-token context, fast non-reasoning variant.',
    'grok-build-0.1': 'Coding model, 256K-token context.',
    'deepseek-v4-pro': 'DeepSeek flagship for complex reasoning and agentic code.',
    'deepseek-flash': 'Fast and cheap, 1M-token context, image input.',
    'nvidia/llama-3.1-nemotron-70b-instruct': 'Llama 3.1 fine-tuned by NVIDIA (Nemotron).',
    'nvidia/nemotron-3-ultra-550b-a55b': 'Very large NVIDIA model, the most capable of the Nemotron 3 range.',
    'nvidia/nemotron-3-super-120b-a12b': 'NVIDIA 120B model, excellent at reasoning and analysis.',
    'nvidia/llama-3.1-nemotron-ultra-253b-v1': 'Very large model fine-tuned by NVIDIA.\n  Excellent at reasoning, slower.',
    'deepseek-ai/deepseek-v4.1-flash': "DeepSeek V4.1 Flash hosted on NVIDIA's infrastructure.",
    'moonshotai/kimi-k2.6': 'Moonshot AI model, good at code and reasoning.',
    'mistralai/mistral-large-4-1t-a52b': 'Mistral flagship (1T MoE), good at code and instructions.',
    'z-ai/glm-5.3-flash': 'Fast Zhipu AI model.',
    'meta/llama-3.2-90b-vision-instruct': 'Meta multimodal model (text + image).',
}


def get_model_description(model_id: str) -> str:
    """Description for a model in the UI language. Generic message if unknown."""
    try:
        from qt_ui.i18n import get_lang
        en = get_lang() == "en"
    except Exception:
        en = False
    if model_id in AI_MODEL_DESCRIPTIONS:
        txt = AI_MODEL_DESCRIPTIONS[model_id]
        if en and model_id in AI_MODEL_DESCRIPTIONS_EN:
            return txt.split("\n", 1)[0] + "\n  " + AI_MODEL_DESCRIPTIONS_EN[model_id]
        return txt
    if en:
        return f"Model: {model_id}\n  (No detailed description available)"
    return f"Modèle : {model_id}\n  (Pas de description détaillée disponible)"


def get_provider_default_model(provider: str) -> str:
    """Get the recommended default model for each provider."""
    defaults = {
        "OpenRouter (Gratuit)": "openrouter/free",
        "Google (Gemini)": "gemini-3.8-flash",
        "Anthropic (Claude)": "claude-sonnet-5-5",
        "OpenAI (ChatGPT)": "gpt-6.1-sol",
        "xAI (Grok)": "grok-4.7",
        "DeepSeek": "deepseek-v4-pro",
        "NVIDIA NIM": "nvidia/llama-3.1-nemotron-70b-instruct",
    }
    return defaults.get(provider, "")
