
"""
Terminal tool-command helpers.

Provides commands for inspecting the tools registered by the research
agent. Tool categories are resolved dynamically from ToolRegistry.
"""

from __future__ import annotations

import logging
from typing import Iterable

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

logger = logging.getLogger("hasini.tool_commands")
_default_console = Console()


def show_tools(
    tools: Iterable,
    title: str = "Available Tools",
    console: Console | None = None,
) -> None:
    """Display a list of registered tools in a Rich table."""
    if console is None:
        console = _default_console

    try:
        tool_list = list(tools)

        if not tool_list:
            console.print(f"\n[yellow]No tools found for: {title}[/yellow]")
            return

        table = Table(
            title=title,
            title_style="bold bright_green",
            border_style="green",
            header_style="bold white on dark_green",
            expand=True,
        )
        table.add_column("Tool Name", style="bold cyan", no_wrap=True)
        table.add_column("Description", style="dim white")

        for tool in tool_list:
            tool_name = getattr(tool, "name", "<unnamed tool>")
            doc = getattr(tool, "description", None) or getattr(tool, "__doc__", "") or "No description provided."
            # Show first line or brief description
            first_line = doc.strip().split("\n")[0]
            table.add_row(tool_name, first_line)

        console.print()
        console.print(table)

    except Exception:
        logger.exception("Failed to display tools.")
        console.print("\n[bold red]❌ Failed to display tools.[/bold red]")


def show_categories(registry, console: Console | None = None) -> None:
    """Display all currently registered tool categories in a Rich table."""
    if console is None:
        console = _default_console

    try:
        categories = registry.categories()

        if not categories:
            console.print("\n[yellow]No tool categories found.[/yellow]")
            return

        table = Table(
            title="Available Tool Categories",
            title_style="bold bright_cyan",
            border_style="cyan",
            header_style="bold white on blue",
            expand=True,
        )
        table.add_column("Category", style="bold yellow", no_wrap=True)
        table.add_column("Count", style="bold green", justify="right")
        table.add_column("Inspection Command", style="bold magenta")

        for category in categories:
            cat_tools = registry.get_tools(category)
            table.add_row(
                category,
                str(len(cat_tools)),
                f"[bold cyan]/tools {category}[/bold cyan]",
            )

        console.print()
        console.print(table)

    except Exception:
        logger.exception("Failed to display tool categories.")
        console.print("\n[bold red]❌ Failed to display tool categories.[/bold red]")


def handle_tool_command(
    command: str,
    registry,
    console: Console | None = None,
) -> bool:
    """
    Handle a supported terminal tool command with Rich formatting.

    Commands:
        /tools            List available tool categories.
        /tools <category> List tools belonging to a category.
        /tool <tool_name> Inspect a specific tool.

    Returns:
        True  -> command was recognized and handled.
        False -> command is not a tool command.
    """
    if console is None:
        console = _default_console

    if not isinstance(command, str):
        logger.warning(
            "Ignoring invalid tool command type: %s",
            type(command).__name__,
        )
        return False

    command = command.strip()

    if not command:
        return False

    try:
        # --------------------------------------------------
        # /tools
        # --------------------------------------------------
        if command.lower() == "/tools":
            show_categories(registry, console=console)
            return True

        # --------------------------------------------------
        # /tools <category>
        # --------------------------------------------------
        if command.lower().startswith("/tools "):
            category = command[7:].strip()

            if not category:
                show_categories(registry, console=console)
                return True

            tools = registry.get_tools(category)

            if tools:
                show_tools(
                    tools,
                    f"{category.title()} Category Tools",
                    console=console,
                )
            else:
                console.print(
                    f"\n[yellow]No tools found for category '[bold]{category}[/bold]'.[/yellow]"
                )
                show_categories(registry, console=console)

            return True

        # --------------------------------------------------
        # /tool <tool_name>
        # --------------------------------------------------
        if command.lower().startswith("/tool "):
            tool_name = command[6:].strip()

            if not tool_name:
                console.print("\n[bold red]❌ Tool name cannot be empty.[/bold red]")
                return True

            tool = registry.get_tool(tool_name)

            if tool:
                name = getattr(tool, "name", tool_name)
                description = getattr(tool, "description", None) or getattr(tool, "__doc__", "No description provided.")
                
                content_text = Text()
                content_text.append("Name: ", style="bold cyan")
                content_text.append(f"{name}\n\n")
                content_text.append("Description:\n", style="bold yellow")
                content_text.append(f"{description}\n")

                panel = Panel(
                    content_text,
                    title=f"[bold bright_magenta]🛠️ Tool Inspection: {name}[/bold bright_magenta]",
                    border_style="magenta",
                    expand=False,
                )
                console.print()
                console.print(panel)
            else:
                console.print(
                    f"\n[bold red]❌ Tool '[yellow]{tool_name}[/yellow]' not found.[/bold red]"
                )

            return True

        # --------------------------------------------------
        # /rag commands
        # --------------------------------------------------
        if command.lower().startswith("/rag"):
            _handle_rag_command(command, console=console)
            return True

    except Exception:
        logger.exception(
            "Failed to handle tool command: %s",
            command,
        )
        console.print("\n[bold red]❌ Failed to process the command.[/bold red]")
        return True

    return False


def _handle_rag_command(command: str, console: Console | None = None) -> None:
    """Handle /rag commands in terminal interface."""
    if console is None:
        console = _default_console

    parts = command.strip().split(maxsplit=2)
    subcmd = parts[1].lower() if len(parts) > 1 else "stats"
    target = parts[2] if len(parts) > 2 else ""

    try:
        from rag.indexer import RAGIndexer
        from rag.retriever import RAGRetriever
        from rag.vector_store import RAGVectorStore

        indexer = RAGIndexer()
        retriever = RAGRetriever(vector_store=indexer.vector_store)

        if subcmd == "list":
            docs = indexer.list_documents()
            if not docs:
                console.print("\n[yellow]No documents indexed in RAG Knowledge Base.[/yellow]")
                return
        if subcmd == "list":
            docs = indexer.list_documents()
            if not docs:
                console.print("\n[yellow]No documents indexed in RAG Knowledge Base.[/yellow]")
                return
            table = Table(title="Hasini RAG Knowledge Base Documents", border_style="cyan")
            table.add_column("Filename", style="bold green")
            table.add_column("Type", style="cyan")
            table.add_column("Chunks", style="yellow", justify="right")
            table.add_column("Indexed At", style="dim white")
            table.add_column("Source", style="dim cyan")
            for d in docs:
                table.add_row(
                    d.get("filename", "N/A"),
                    d.get("file_type", "N/A"),
                    str(d.get("chunk_count", 0)),
                    d.get("timestamp", "N/A"),
                    d.get("source", "N/A"),
                )
            console.print()
            console.print(table)

        elif subcmd == "add":
            if not target:
                console.print("\n[bold red][ERROR]: Please provide a file path or URL to add: /rag add <path_or_url>[/bold red]")
                return
            console.print(f"\n[bold cyan]Indexing document/URL:[/bold cyan] {target}")
            res = indexer.index_document(target)
            if res["success"]:
                console.print(f"[bold green][OK]:[/bold green] {res['message']}")
            else:
                console.print(f"[bold red][ERROR]:[/bold red] {res['message']}")

        elif subcmd == "remove":
            if not target:
                console.print("\n[bold red][ERROR]: Please provide a file path or URL to remove: /rag remove <path_or_url>[/bold red]")
                return
            res = indexer.remove_document(target)
            if res["success"]:
                console.print(f"\n[bold green][OK]:[/bold green] {res['message']}")
            else:
                console.print(f"\n[bold red][ERROR]:[/bold red] {res['message']}")

        elif subcmd == "search":
            if not target:
                console.print("\n[bold red][ERROR]: Please provide a search query: /rag search <query>[/bold red]")
                return
            chunks = retriever.retrieve(query=target)
            if not chunks:
                console.print("\n[yellow]No relevant chunks found.[/yellow]")
                return
            console.print(f"\n[bold green]Found {len(chunks)} relevant chunk(s):[/bold green]\n")
            for idx, c in enumerate(chunks, 1):
                meta = c["metadata"]
                title = f"Chunk {idx} | Source: {meta.get('filename')} | Similarity: {c['similarity']:.4f}"
                console.print(Panel(c["text"], title=title, border_style="green"))

        elif subcmd == "stats":
            stats = indexer.vector_store.get_stats()
            from config import RAG_ENABLED, RAG_EMBEDDING_MODEL, RAG_TOP_K
            console.print()
            console.print(Panel(
                f"[bold white]Status:[/bold white] [{'green' if RAG_ENABLED else 'red'}]{'ENABLED' if RAG_ENABLED else 'DISABLED'}[/]\n"
                f"[bold white]Storage Path:[/bold white] {stats['db_path']}\n"
                f"[bold white]Embedding Model:[/bold white] {RAG_EMBEDDING_MODEL}\n"
                f"[bold white]Total Unique Documents:[/bold white] {stats['total_documents']}\n"
                f"[bold white]Total Text Chunks:[/bold white] {stats['total_chunks']}\n"
                f"[bold white]Top-K Default:[/bold white] {RAG_TOP_K}",
                title="RAG Knowledge System Statistics",
                border_style="bright_blue",
            ))

        elif subcmd == "clear":
            count = indexer.clear_all()
            console.print(f"\n[bold green][OK]:[/bold green] Cleared {count} chunk(s) from knowledge base.")

        else:
            console.print(f"\n[yellow]Unknown /rag command: '{subcmd}'. Available: list, add, remove, search, stats, clear.[/yellow]")

    except Exception as exc:
        logger.exception("Failed to execute /rag command: %s", command)
        console.print(f"\n[bold red][ERROR]: Error executing RAG command: {exc}[/bold red]")

