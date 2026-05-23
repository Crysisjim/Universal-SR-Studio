"""
ai_models_metadata.py — Descriptions and metadata for AI models used in config check.
"""

# Detailed descriptions for the model picker tooltip.
# Format: model_id -> "Short label : detailed description"
AI_MODEL_DESCRIPTIONS = {
    # ============ Anthropic Claude ============
    "claude-opus-4-7": "Claude Opus 4.7 (Anthropic)\n"
        "  Modele le plus avance d'Anthropic.\n"
        "  Excellent en analyse fine, raisonnement complexe.\n"
        "  Recommande pour evaluation de configs avancees.\n"
        "  Contexte : 200k tokens. Cout : eleve.",
    "claude-sonnet-4-6": "Claude Sonnet 4.6 (Anthropic)\n"
        "  Equilibre vitesse / intelligence.\n"
        "  Bon defaut pour la plupart des taches.\n"
        "  Contexte : 200k tokens. Cout : moyen.",
    "claude-haiku-4-5-20251001": "Claude Haiku 4.5 (Anthropic)\n"
        "  Modele rapide et economique.\n"
        "  Bon pour verification simple, prototypage.\n"
        "  Contexte : 200k tokens. Cout : tres bas.",

    # ============ OpenAI ============
    "gpt-5": "GPT-5 (OpenAI)\n"
        "  Flagship OpenAI generation 5.\n"
        "  Raisonnement avance, multimodal.\n"
        "  Recommande pour analyse de config complexe.",
    "gpt-5-mini": "GPT-5 Mini (OpenAI)\n"
        "  Version compacte et economique de GPT-5.\n"
        "  Bon equilibre qualite/cout/vitesse.",
    "gpt-4.1": "GPT-4.1 (OpenAI)\n"
        "  Modele OpenAI (avril 2025).\n"
        "  Excellent en code, instruction following.\n"
        "  Contexte : 1M tokens.",
    "gpt-4.1-mini": "GPT-4.1 Mini (OpenAI)\n"
        "  Version rapide et economique de GPT-4.1.\n"
        "  Bon equilibre qualite/cout.",
    "gpt-4.1-nano": "GPT-4.1 Nano (OpenAI)\n"
        "  La plus petite/rapide de la famille 4.1.\n"
        "  Pour classification simple, latence critique.",
    "gpt-4o": "GPT-4o (OpenAI)\n"
        "  Modele multimodal phare.\n"
        "  Tres bon en code et analyse.\n"
        "  Contexte : 128k tokens.",
    "gpt-4o-mini": "GPT-4o Mini (OpenAI)\n"
        "  Version rapide et economique de GPT-4o.\n"
        "  Bon pour taches simples.",
    "o3": "o3 (OpenAI)\n"
        "  Modele de reasoning avance.\n"
        "  Pour les problemes complexes necessitant reflexion.",
    "o3-mini": "o3 Mini (OpenAI)\n"
        "  Reasoning rapide et economique.\n"
        "  Bon rapport qualite/prix pour l'analyse.",

    # ============ Google Gemini ============
    "gemini-3.1-pro-preview": "Gemini 3.1 Pro Preview (Google)\n"
        "  Flagship Google generation 3.1 (preview).\n"
        "  Reasoning avance, contexte long.",
    "gemini-3.1-flash-lite": "Gemini 3.1 Flash Lite (Google)\n"
        "  Version rapide et economique de Gemini 3.1.\n"
        "  Bon equilibre vitesse/qualite.",
    "gemini-2.5-pro": "Gemini 2.5 Pro (Google)\n"
        "  Flagship Google avec reasoning adaptatif.\n"
        "  Tres bon en code et analyse technique.\n"
        "  Contexte : 1M tokens.",
    "gemini-2.5-flash": "Gemini 2.5 Flash (Google)\n"
        "  Rapide, equilibre intelligence/latence.\n"
        "  Tier gratuit genereux.",
    "gemini-2.5-flash-lite": "Gemini 2.5 Flash Lite (Google)\n"
        "  Le plus rapide et economique de la 2.5.\n"
        "  Pour high-volume, cost-sensitive.",
    "gemini-2.0-flash": "Gemini 2.0 Flash (Google)\n"
        "  Rapide, multimodal, generation precedente.\n"
        "  Tier gratuit dispo.",

    # ============ xAI Grok ============
    "grok-4.3": "Grok 4.3 (xAI)\n"
        "  Flagship xAI generation 4.3.\n"
        "  Excellent en raisonnement et code.\n"
        "  Contexte large.",
    "grok-4.3-fast": "Grok 4.3 Fast (xAI)\n"
        "  Version rapide de Grok 4.3.\n"
        "  Bon pour taches repetitives et pipelines.",
    "grok-4.3-mini": "Grok 4.3 Mini (xAI)\n"
        "  Modele compact Grok 4.3 avec reasoning.\n"
        "  Tres bon rapport qualite/prix.",
    "grok-3": "Grok 3 (xAI)\n"
        "  Flagship xAI (fevrier 2025).\n"
        "  Excellent en raisonnement et code.\n"
        "  Contexte : 131k tokens.",
    "grok-3-fast": "Grok 3 Fast (xAI)\n"
        "  Version rapide de Grok 3.\n"
        "  Bon pour taches repetitives et pipelines.",
    "grok-3-mini": "Grok 3 Mini (xAI)\n"
        "  Modele compact avec reasoning.\n"
        "  Tres bon rapport qualite/prix.",
    "grok-3-mini-fast": "Grok 3 Mini Fast (xAI)\n"
        "  Version la plus rapide et economique.\n"
        "  Pour iteration rapide.",

    # ============ DeepSeek ============
    "deepseek-chat": "DeepSeek Chat (DeepSeek)\n"
        "  Modele general DeepSeek V3.\n"
        "  Excellent en code, tres bon rapport qualite/prix.",
    "deepseek-reasoner": "DeepSeek Reasoner R1 (DeepSeek)\n"
        "  Modele de reasoning open-weights.\n"
        "  Tres bon pour analyse technique approfondie.",

    # ============ OpenRouter Free ============
    "meta-llama/llama-3.3-70b-instruct:free": "Llama 3.3 70B (Free via OpenRouter)\n"
        "  Modele Meta open-weights GRATUIT.\n"
        "  Bon pour taches generales.",
    "nvidia/nemotron-3-super-120b-a12b:free": "Nemotron 3 Super 120B (Free via OpenRouter)\n"
        "  Modele NVIDIA 120B GRATUIT.\n"
        "  Excellent en raisonnement et analyse.",
    "z-ai/glm-4.5-air:free": "GLM-4.5 Air (Free via OpenRouter)\n"
        "  Modele Zhipu AI GRATUIT.\n"
        "  Rapide et capable.",
    "google/gemma-4-31b-it:free": "Gemma 4 31B (Free via OpenRouter)\n"
        "  Modele Google open-weights GRATUIT.\n"
        "  Bon reasoning et coding.",
    "google/gemma-4-26b-a4b-it:free": "Gemma 4 26B MoE (Free via OpenRouter)\n"
        "  Modele Google MoE GRATUIT (architecture sparse).\n"
        "  Rapide malgre sa taille.",
    "qwen/qwen3-coder:free": "Qwen3 Coder (Free via OpenRouter)\n"
        "  Modele code Alibaba GRATUIT.\n"
        "  Tres fort en code et analyse technique.",
    "openai/gpt-oss-120b:free": "GPT-OSS 120B (Free via OpenRouter)\n"
        "  Grand modele open-source OpenAI GRATUIT.\n"
        "  Excellent rapport qualite/cout.",

    # ============ GitHub Models Free ============
    "gpt-4o": "GPT-4o (Free via GitHub Models)\n"
        "  GPT-4o disponible gratuitement avec compte GitHub.\n"
        "  Rate limit quotidien.",
    "gpt-4o-mini": "GPT-4o Mini (Free via GitHub Models)\n"
        "  Version reduite GRATUITE, plus de quota.\n"
        "  Recommande comme defaut GitHub.",
    "Phi-4": "Phi-4 (Free via GitHub Models)\n"
        "  Modele Microsoft compact mais puissant.\n"
        "  Bon pour reasoning sur petites configs.",
    "DeepSeek-R1": "DeepSeek R1 (Free via GitHub Models)\n"
        "  Reasoning model open-source.\n"
        "  Disponible gratuitement avec compte GitHub.",
    "Llama-3.3-70B-Instruct": "Llama 3.3 70B (Free via GitHub Models)\n"
        "  Modele Meta GRATUIT.",
}


def get_model_description(model_id: str) -> str:
    """Get the description for a model. Returns generic message if unknown."""
    if model_id in AI_MODEL_DESCRIPTIONS:
        return AI_MODEL_DESCRIPTIONS[model_id]
    return f"Modele : {model_id}\n  (Pas de description detaillee disponible)"


def get_provider_default_model(provider: str) -> str:
    """Get the recommended default model for each provider."""
    defaults = {
        "OpenRouter (Gratuit)": "meta-llama/llama-3.3-70b-instruct:free",
        "GitHub Models (Gratuit)": "gpt-4o-mini",
        "Google (Gemini)": "gemini-2.5-pro",
        "Anthropic (Claude)": "claude-sonnet-4-6",
        "OpenAI (ChatGPT)": "gpt-5",
        "xAI (Grok)": "grok-4.3",
        "DeepSeek": "deepseek-reasoner",
    }
    return defaults.get(provider, "")
