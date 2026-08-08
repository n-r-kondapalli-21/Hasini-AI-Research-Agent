import os
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ZAI_API_KEY = os.getenv("ZAI_API_KEY")

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openrouter")
MODEL_NAME = os.getenv("MODEL_NAME")

PROVIDERS = {
    "openrouter": {
        "api_key": OPENROUTER_API_KEY,
        "base_url": "https://openrouter.ai/api/v1",
    },
    "zai": {
        "api_key": ZAI_API_KEY,
        "base_url": "https://api.z.ai/api/paas/v4",
    },
}

provider = PROVIDERS.get(LLM_PROVIDER)

if provider is None:
    raise ValueError(f"Unsupported provider: {LLM_PROVIDER}")