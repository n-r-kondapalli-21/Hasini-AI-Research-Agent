# Hasini RAG Knowledge System

The **RAG (Retrieval-Augmented Generation) Knowledge System** is an independent module for the Hasini AI Research Agent. It provides persistent vector storage and semantic retrieval over manually curated reference documents, technical manuals, papers, and web pages.

> **Important Note:** This module is **NOT a conversational memory system**. It does not automatically capture or extract user chat history, queries, or personal interactions. It is strictly a **manually managed knowledge base**.

---

## 🏗 Architecture & Flow

### Indexing Flow

```
Document / URL (PDF, TXT, DOCX, MD, Web Page)
        │
        ▼
RAGDocumentLoader (Text & Metadata Extraction)
        │
        ▼
RAGTextSplitter (Chunking & Metadata Preserving)
        │
        ▼
Local SentenceTransformer Model (models/embeddings/all-MiniLM-L6-v2/)
        │
        ▼
RAGVectorStore (ChromaDB Persistent Storage)
```

### Query & Agent Context Flow

```
User Query (CLI, Voice, Telegram)
        │
        ▼
RAGRetriever (Vector Similarity Search)
        │
        ▼
Relevance Threshold Filter (similarity >= RAG_SIMILARITY_THRESHOLD)
        │
        ├──► No relevant chunks met threshold ──► Agent executes normally
        │
        ▼ Relevant chunks found
Formatted RAG Context (Source Filename, Chunk ID, Similarity, Date, Text)
        │
        ▼
Hasini Agent (LangChain / Gemini / OpenRouter / ZAI)
        │
        ▼
Tools & MCP Servers (Web Search, GitHub, Filesystem, OpenAlgo)
        │
        ▼
Final Response
```

---

## 📁 Directory Structure

| File | Description |
| :--- | :--- |
| [`__init__.py`](file:///d:/Hasini_Ai_Research_Agent/rag/__init__.py) | Package initialization exposing `RAGVectorStore`, `RAGIndexer`, `RAGRetriever`, `RAGDocumentLoader`, `RAGTextSplitter`. |
| [`vector_store.py`](file:///d:/Hasini_Ai_Research_Agent/rag/vector_store.py) | ChromaDB persistent storage client, collection management, cosine similarity queries, document listing, and chunk deletion. |
| [`document_loader.py`](file:///d:/Hasini_Ai_Research_Agent/rag/document_loader.py) | Document parser supporting PDF (`pypdf`), DOCX (`python-docx`), TXT, Markdown, and Web URLs (`requests` + `BeautifulSoup`). |
| [`text_splitter.py`](file:///d:/Hasini_Ai_Research_Agent/rag/text_splitter.py) | Recursive text chunking with metadata attachment (`source`, `filename`, `file_type`, `chunk_id`, `timestamp`). |
| [`embedding.py`](file:///d:/Hasini_Ai_Research_Agent/rag/embedding.py) | Local `sentence-transformers` loader downloading to `models/embeddings/all-MiniLM-L6-v2/` on 1st setup and loading locally on subsequent runs. |
| [`indexer.py`](file:///d:/Hasini_Ai_Research_Agent/rag/indexer.py) | Document lifecycle controller (`index_document`, `reindex_document`, `remove_document`, `list_documents`, `clear_all`). |
| [`retriever.py`](file:///d:/Hasini_Ai_Research_Agent/rag/retriever.py) | Semantic search query processor, similarity score threshold filter, logging reporter, and prompt context formatter. |
| [`cli.py`](file:///d:/Hasini_Ai_Research_Agent/rag/cli.py) | Command-line management tool interface. |

---

## ⚙️ Configuration (`.env`)

Configure the RAG Knowledge System in your root `.env` file:

```env
# Enable or disable RAG knowledge retrieval (true/false)
RAG_ENABLED=true

# Path to persistent ChromaDB storage directory
CHROMADB_PATH=./knowledge_base/chroma_db

# Path to persistent BM25 index directory
BM25_INDEX_PATH=./models/bm25_index

# Local embedding model storage directory (relative to project root)
RAG_EMBEDDING_MODEL_PATH=models/embeddings/all-MiniLM-L6-v2

# HuggingFace model identifier for initial setup download
RAG_EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2

# Maximum number of top relevant chunks to retrieve per query
RAG_TOP_K=5

# Minimum similarity threshold (0.0 to 1.0)
# Chunks with similarity score below this value will be ignored
RAG_SIMILARITY_THRESHOLD=0.3
```

---

## 🚀 Management Commands

You can manage the RAG Knowledge Base using the top-level script `rag_manage.py` or terminal slash commands inside `main.py`.

### A. Using `rag_manage.py` CLI

```bash
# Add / Index a document file or Web URL
python rag_manage.py add path/to/document.pdf
python rag_manage.py add https://example.com/research_paper

# List all currently indexed knowledge documents
python rag_manage.py list

# Test vector search for a query
python rag_manage.py search "quantum computing"

# View system statistics & vector store status
python rag_manage.py stats

# Re-index an existing document
python rag_manage.py reindex path/to/document.pdf

# Remove a document from the knowledge base
python rag_manage.py remove path/to/document.pdf

# Clear the entire knowledge base
python rag_manage.py clear
```

### B. Using Terminal Slash Commands (`main.py`)

When running text mode (`python main.py`), use `/rag` commands directly:
- `/rag list` — Display indexed documents.
- `/rag add <path_or_url>` — Index a new document or URL.
- `/rag remove <path_or_url>` — Remove an indexed document.
- `/rag search <query>` — Perform a test retrieval search.
- `/rag stats` — Display RAG configuration and stats.
- `/rag clear` — Clear all knowledge.

---

## 📄 Supported File Formats

- **PDF Documents** (`.pdf`) — Page-by-page text extraction via `pypdf`.
- **Microsoft Word** (`.docx`, `.doc`) — Paragraph and table extraction via `python-docx`.
- **Plain Text** (`.txt`) — UTF-8 / fallback encoded plain text.
- **Markdown** (`.md`, `.markdown`) — Structured text parsing.
- **Web Pages** (`http://`, `https://`) — Main article content extraction via `BeautifulSoup`.

---

## 🧪 Testing

Run the RAG unit test suite from the project root:

```bash
python test_rag.py
```

Tests cover:
- Document parsing and text chunking
- Local `sentence-transformers` model loading
- ChromaDB vector store indexing & querying
- Relevance threshold score filtering
- Document deletion and collection reset
- Empty results and disabled RAG handling
