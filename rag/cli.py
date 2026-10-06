"""
CLI tool for manual management of Hasini RAG Knowledge Base.

Usage:
  python -m rag.cli add <path_or_url>
  python -m rag.cli remove <path_or_url>
  python -m rag.cli reindex <path_or_url>
  python -m rag.cli list
  python -m rag.cli search "<query>"
  python -m rag.cli clear
  python -m rag.cli stats
"""

import sys
import argparse
import logging
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from .indexer import RAGIndexer
from .retriever import RAGRetriever
from .vector_store import RAGVectorStore
from config import RAG_ENABLED, CHROMADB_PATH, RAG_EMBEDDING_MODEL, RAG_TOP_K

console = Console()


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_add(args, indexer: RAGIndexer):
    console.print(f"[bold cyan]Indexing document/URL:[/bold cyan] {args.source}")
    res = indexer.index_document(args.source)
    if res["success"]:
        console.print(f"[bold green][OK]:[/bold green] {res['message']} (Filename: [cyan]{res['filename']}[/cyan])")
    else:
        console.print(f"[bold red][ERROR]:[/bold red] {res['message']}")


def cmd_remove(args, indexer: RAGIndexer):
    console.print(f"[bold yellow]Removing document/URL:[/bold yellow] {args.source}")
    res = indexer.remove_document(args.source)
    if res["success"]:
        console.print(f"[bold green][OK]:[/bold green] {res['message']}")
    else:
        console.print(f"[bold red][ERROR]:[/bold red] {res['message']}")


def cmd_reindex(args, indexer: RAGIndexer):
    console.print(f"[bold cyan]Re-indexing document/URL:[/bold cyan] {args.source}")
    res = indexer.reindex_document(args.source)
    if res["success"]:
        console.print(f"[bold green][OK]:[/bold green] {res['message']}")
    else:
        console.print(f"[bold red][ERROR]:[/bold red] {res['message']}")


def cmd_list(args, indexer: RAGIndexer):
    docs = indexer.list_documents()
    if not docs:
        console.print("[bold yellow]No documents indexed in the RAG Knowledge Base.[/bold yellow]")
        return

    table = Table(title="Hasini RAG Knowledge Base Documents", border_style="cyan")
    table.add_column("Filename", style="bold green")
    table.add_column("Type", style="cyan")
    table.add_column("Chunks", style="yellow", justify="right")
    table.add_column("Indexed At", style="dim white")
    table.add_column("Source Path / URL", style="dim cyan")

    for d in docs:
        table.add_row(
            d.get("filename", "N/A"),
            d.get("file_type", "N/A"),
            str(d.get("chunk_count", 0)),
            d.get("timestamp", "N/A"),
            d.get("source", "N/A"),
        )

    console.print(table)


def cmd_search(args, retriever: RAGRetriever):
    console.print(f"[bold cyan]Searching RAG Knowledge Base for query:[/bold cyan] [bold white]'{args.query}'[/bold white]")
    chunks = retriever.retrieve(
        query=args.query,
        top_k=args.top_k,
    )

    if not chunks:
        console.print(f"[bold yellow]No chunks found for query '{args.query}'.[/bold yellow]")
        return

    console.print(f"[bold green]Found {len(chunks)} relevant chunk(s):[/bold green]\n")
    for idx, c in enumerate(chunks, 1):
        meta = c["metadata"]
        panel_title = f"Chunk {idx} | Source: {meta.get('filename')} | Similarity: {c['similarity']:.4f} | Dist: {c['distance']:.4f}"
        console.print(Panel(c["text"], title=panel_title, border_style="green"))


def cmd_clear(args, indexer: RAGIndexer):
    console.print("[bold red]Clearing all documents from RAG Knowledge Base...[/bold red]")
    removed = indexer.clear_all()
    console.print(f"[bold green][OK]:[/bold green] Cleared {removed} chunk(s) from knowledge base.")


def cmd_stats(args, vector_store: RAGVectorStore):
    stats = vector_store.get_stats()
    console.print(Panel(
        f"[bold white]RAG Status:[/bold white] [{'green' if RAG_ENABLED else 'red'}]{'ENABLED' if RAG_ENABLED else 'DISABLED'}[/]\n"
        f"[bold white]Storage Path:[/bold white] {stats['db_path']}\n"
        f"[bold white]Embedding Model:[/bold white] {RAG_EMBEDDING_MODEL}\n"
        f"[bold white]Total Unique Documents:[/bold white] {stats['total_documents']}\n"
        f"[bold white]Total Text Chunks:[/bold white] {stats['total_chunks']}\n"
        f"[bold white]Top-K Default:[/bold white] {RAG_TOP_K}\n",
        title="RAG Knowledge System Statistics",
        border_style="bright_blue",
    ))


def main():
    configure_logging()
    parser = argparse.ArgumentParser(description="Hasini RAG Knowledge Base Manager")
    subparsers = parser.add_subparsers(dest="command", help="Sub-command help")

    # GUI
    subparsers.add_parser("gui", help="Launch native Tkinter Desktop GUI")
    subparsers.add_parser("ui", help="Launch native Tkinter Desktop GUI")

    # Add
    p_add = subparsers.add_parser("add", help="Index a document file or URL")
    p_add.add_argument("source", help="Path to PDF, TXT, DOCX, Markdown file, or URL")

    # Remove
    p_rem = subparsers.add_parser("remove", help="Remove an indexed document or URL")
    p_rem.add_argument("source", help="Source path or URL to remove")

    # Reindex
    p_rei = subparsers.add_parser("reindex", help="Re-index an existing document or URL")
    p_rei.add_argument("source", help="Source path or URL to reindex")

    # List
    subparsers.add_parser("list", help="List all indexed documents")

    # Search
    p_sch = subparsers.add_parser("search", help="Test vector search for a query")
    p_sch.add_argument("query", help="Query text to search")
    p_sch.add_argument("--top-k", type=int, default=None, help="Top K results to retrieve")

    # Clear
    subparsers.add_parser("clear", help="Clear all indexed knowledge")

    # Stats
    subparsers.add_parser("stats", help="View knowledge system statistics")

    args = parser.parse_args()
    if not args.command or args.command in ("gui", "ui"):
        from .gui import launch_gui
        launch_gui()
        return

    vector_store = RAGVectorStore()
    indexer = RAGIndexer(vector_store=vector_store)
    retriever = RAGRetriever(vector_store=vector_store)

    if args.command == "add":
        cmd_add(args, indexer)
    elif args.command == "remove":
        cmd_remove(args, indexer)
    elif args.command == "reindex":
        cmd_reindex(args, indexer)
    elif args.command == "list":
        cmd_list(args, indexer)
    elif args.command == "search":
        cmd_search(args, retriever)
    elif args.command == "clear":
        cmd_clear(args, indexer)
    elif args.command == "stats":
        cmd_stats(args, vector_store)


if __name__ == "__main__":
    main()

