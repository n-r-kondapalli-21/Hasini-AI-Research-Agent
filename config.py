import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
GOOGLE_AI_STUDIO_API_KEY = os.getenv("GOOGLE_AI_STUDIO_API_KEY")
ZAI_API_KEY = os.getenv("ZAI_API_KEY")

# Select LLM provider from .env
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openrouter")
MODEL_NAME = os.getenv("MODEL_NAME")

# LLM Providers
PROVIDERS = {
    "openrouter": {
        "api_key": OPENROUTER_API_KEY,
        "base_url": "https://openrouter.ai/api/v1",
    },

    "google": {
        "api_key": GOOGLE_AI_STUDIO_API_KEY,
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
    },

    "zai": {
        "api_key": ZAI_API_KEY,
        "base_url": "https://api.z.ai/api/paas/v4",
    },
}

# Get selected provider
provider = PROVIDERS.get(LLM_PROVIDER)

if provider is None:
    raise ValueError(
        f"Unsupported provider: {LLM_PROVIDER}. "
        f"Supported providers: {', '.join(PROVIDERS.keys())}"
    )

# Validate selected provider API key
if not provider["api_key"]:
    raise ValueError(
        f"{LLM_PROVIDER.upper()} API key is missing from .env"
    )


Tavily_api_key=os.getenv("TAVILY_API_KEY")

# Set Tavily API key in environment for web_search.py compatibility
if Tavily_api_key:
    os.environ["Tavily_api_key"] = Tavily_api_key

# ============================================================
# Telegram Bot Configuration
# ============================================================

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# ============================================================
# Conversation Memory
# ============================================================

# Set to False to disable conversation history entirely.
MEMORY_ENABLED = os.getenv("MEMORY_ENABLED", "true").lower() == "true"

# Maximum number of messages (user + assistant) to keep in memory.
MEMORY_HISTORY_LIMIT = int(os.getenv("MEMORY_HISTORY_LIMIT", "10"))

if MEMORY_HISTORY_LIMIT < 2:
    raise ValueError("MEMORY_HISTORY_LIMIT must be at least 2. Check in .env file")

# ============================================================
# Tool Enable/Disable Flags
# ============================================================

ENABLE_WEB_TOOL = os.getenv("ENABLE_WEB_TOOL", "true").lower() == "true"
ENABLE_WEATHER_TOOL = os.getenv("ENABLE_WEATHER_TOOL", "true").lower() == "true"
ENABLE_CALCULATOR_TOOL = os.getenv("ENABLE_CALCULATOR_TOOL", "true").lower() == "true"

# ============================================================
# MCP Server Enable/Disable Flags
# ============================================================

ENABLE_MCP_FILESYSTEM = os.getenv("ENABLE_MCP_FILESYSTEM", "true").lower() == "true"
ENABLE_MCP_GITHUB = os.getenv("ENABLE_MCP_GITHUB", "true").lower() == "true"
ENABLE_MCP_OPENALGO = os.getenv("ENABLE_MCP_OPENALGO", "true").lower() == "true"