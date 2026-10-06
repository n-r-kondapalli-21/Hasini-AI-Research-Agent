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
try:
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
except (ValueError, TypeError):
    TEMPERATURE = 0.7


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

# ============================================================
# RAG Knowledge System Configuration
# ============================================================

RAG_ENABLED = os.getenv("RAG_ENABLED", "true").lower() == "true"
CHROMADB_PATH = os.getenv("CHROMADB_PATH", "./knowledge_base/chroma_db")
BM25_INDEX_PATH = os.getenv("BM25_INDEX_PATH") or os.getenv("RAG_BM25_INDEX_PATH") or "./models/bm25_index"
RAG_BM25_INDEX_PATH = BM25_INDEX_PATH
RAG_EMBEDDING_MODEL_NAME = os.getenv("RAG_EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
RAG_EMBEDDING_MODEL_PATH = os.getenv("RAG_EMBEDDING_MODEL_PATH", "models/embeddings/all-MiniLM-L6-v2")
RAG_EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", RAG_EMBEDDING_MODEL_NAME)

RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))

# ============================================================
# Hybrid Search Configuration
# ============================================================

# Enable/disable BM25 keyword search
RAG_BM25_ENABLED = os.getenv("RAG_BM25_ENABLED", "true").lower() == "true"
RAG_BM25_INDEX_PATH = os.getenv("RAG_BM25_INDEX_PATH", "models/bm25_index")

# Number of top results from vector search before fusion
RAG_VECTOR_TOP_K = int(os.getenv("RAG_VECTOR_TOP_K", "10"))

# Number of top results from BM25 search before fusion
RAG_BM25_TOP_K = int(os.getenv("RAG_BM25_TOP_K", "10"))

# RRF constant k (default 60 as per original paper)
RAG_RRF_K = int(os.getenv("RAG_RRF_K", "60"))

# Enable/disable cross-encoder reranking
RAG_RERANKER_ENABLED = os.getenv("RAG_RERANKER_ENABLED", "true").lower() == "true"

# Number of top candidates to pass to reranker
RAG_RERANKER_TOP_K = int(os.getenv("RAG_RERANKER_TOP_K", "20"))

# Reranker relevance threshold (0.0 to disable filtering)
RAG_RERANKER_THRESHOLD = float(os.getenv("RAG_RERANKER_THRESHOLD", "0.0"))

# Final number of chunks to return after all processing
RAG_FINAL_TOP_K = int(os.getenv("RAG_FINAL_TOP_K", "5"))
