"""
Top-level entry point for managing Hasini RAG Knowledge Base.

Usage:
    python rag_manage.py               (Launches native Tkinter Desktop GUI)
    python rag_manage.py gui           (Launches native Tkinter Desktop GUI)
    python rag_manage.py add <path_or_url>
    python rag_manage.py remove <path_or_url>
    python rag_manage.py list
    python rag_manage.py search "<query>"
    python rag_manage.py stats
    python rag_manage.py clear
"""

from rag.cli import main

if __name__ == "__main__":
    main()

