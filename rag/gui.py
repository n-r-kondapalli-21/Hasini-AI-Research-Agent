"""
Tkinter Desktop GUI for Hasini RAG Knowledge System.

Provides a clean, simple native desktop interface to:
- Browse, add, re-index, delete, or clear RAG documents (PDF, TXT, DOCX, MD, URLs, Folders).
- Test hybrid vector & keyword search queries with live similarity threshold & top-k sliders.
- View system statistics and ChromaDB storage details.
"""

import os
import sys
import threading
import logging
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from typing import List, Dict, Any, Optional

from .indexer import RAGIndexer
from .retriever import RAGRetriever
from .vector_store import RAGVectorStore
from config import (
    RAG_ENABLED,
    CHROMADB_PATH,
    RAG_EMBEDDING_MODEL,
    RAG_TOP_K,
    RAG_BM25_ENABLED,
    RAG_RERANKER_ENABLED,
)

logger = logging.getLogger("hasini.rag.gui")


# ----------------------------------------------------------------------------
# Visual theme (UI only)
# ----------------------------------------------------------------------------
if sys.platform.startswith("win"):
    FONT = "Segoe UI"
    MONO = "Consolas"
elif sys.platform == "darwin":
    FONT = "Helvetica Neue"
    MONO = "Menlo"
else:
    FONT = "DejaVu Sans"
    MONO = "DejaVu Sans Mono"

C = {
    "bg": "#f1f5f9",          # window background
    "card": "#ffffff",        # card / panel surface
    "border": "#e2e8f0",
    "header": "#0f172a",
    "header_sub": "#94a3b8",
    "text": "#0f172a",
    "muted": "#64748b",
    "primary": "#2563eb",
    "primary_hover": "#1d4ed8",
    "primary_press": "#1e40af",
    "danger": "#dc2626",
    "danger_hover": "#b91c1c",
    "neutral": "#e2e8f0",
    "neutral_hover": "#cbd5e1",
    "row_alt": "#f8fafc",
    "select": "#dbeafe",
    "ok": "#16a34a",
    "warn": "#d97706",
    "err": "#dc2626",
    "log_bg": "#0f172a",
    "log_fg": "#e2e8f0",
}


class RAGManagerGUI:
    """Native Python Tkinter Desktop Application for RAG Management."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Hasini AI - RAG Knowledge Base Manager")
        self.root.geometry("1020x760")
        self.root.minsize(840, 620)
        self.root.configure(background=C["bg"])

        # Apply modern ttk theme if available
        self.style = ttk.Style()
        available_themes = self.style.theme_names()
        if "clam" in available_themes:
            self.style.theme_use("clam")
        elif "vista" in available_themes:
            self.style.theme_use("vista")

        # Custom styles
        self._configure_styles()

        # Initialize backend engines lazily or with loading indicator
        self.vector_store: Optional[RAGVectorStore] = None
        self.indexer: Optional[RAGIndexer] = None
        self.retriever: Optional[RAGRetriever] = None

        # Build UI layout
        self._build_header()
        self._build_statusbar()  # packed at bottom first so it stays visible
        self._build_notebook()

        # Initialize RAG background load
        self.root.after(100, self._async_init_backend)

    def _configure_styles(self):
        """Define custom fonts and colors for ttk components."""
        s = self.style

        # Base
        s.configure(".", background=C["bg"], foreground=C["text"], font=(FONT, 9))
        s.configure("TFrame", background=C["bg"])
        s.configure("TLabel", background=C["bg"], foreground=C["text"])

        # Header
        s.configure("Header.TFrame", background=C["header"])
        s.configure("Header.TLabel", background=C["header"], foreground="#f8fafc", font=(FONT, 16, "bold"))
        s.configure("HeaderSub.TLabel", background=C["header"], foreground=C["header_sub"], font=(FONT, 9))

        # Cards
        s.configure("Card.TFrame", background=C["card"])
        s.configure("Card.TLabel", background=C["card"], foreground=C["text"])
        s.configure("CardMuted.TLabel", background=C["card"], foreground=C["muted"])
        s.configure(
            "TLabelframe",
            background=C["card"],
            bordercolor=C["border"],
            lightcolor=C["border"],
            darkcolor=C["border"],
            relief="solid",
            borderwidth=1,
        )
        s.configure("TLabelframe.Label", background=C["card"], foreground=C["muted"], font=(FONT, 9, "bold"))

        # Buttons
        s.configure(
            "TButton",
            font=(FONT, 9),
            padding=(10, 6),
            background=C["neutral"],
            foreground=C["text"],
            bordercolor=C["border"],
            focuscolor=C["neutral"],
            relief="flat",
            borderwidth=1,
        )
        s.map(
            "TButton",
            background=[("pressed", C["neutral_hover"]), ("active", C["neutral_hover"])],
        )

        s.configure(
            "Action.TButton",
            font=(FONT, 9, "bold"),
            padding=(12, 6),
            background=C["primary"],
            foreground="#ffffff",
            bordercolor=C["primary"],
            focuscolor=C["primary"],
        )
        s.map(
            "Action.TButton",
            background=[("pressed", C["primary_press"]), ("active", C["primary_hover"])],
            foreground=[("disabled", "#cbd5e1")],
        )

        s.configure("Accent.TButton", font=(FONT, 9, "bold"), padding=(12, 6))

        s.configure(
            "Danger.TButton",
            font=(FONT, 9, "bold"),
            padding=(12, 6),
            background="#fee2e2",
            foreground=C["danger"],
            bordercolor="#fecaca",
            focuscolor="#fee2e2",
        )
        s.map(
            "Danger.TButton",
            background=[("pressed", C["danger_hover"]), ("active", C["danger"])],
            foreground=[("pressed", "#ffffff"), ("active", "#ffffff")],
        )

        # Notebook tabs
        s.configure("TNotebook", background=C["bg"], borderwidth=0, tabmargins=(0, 0, 0, 0))
        s.configure(
            "TNotebook.Tab",
            font=(FONT, 10, "bold"),
            padding=(18, 8),
            background=C["neutral"],
            foreground=C["muted"],
            borderwidth=0,
        )
        s.map(
            "TNotebook.Tab",
            background=[("selected", C["card"]), ("active", C["neutral_hover"])],
            foreground=[("selected", C["primary"])],
        )

        # Treeview
        s.configure(
            "Treeview",
            font=(FONT, 9),
            rowheight=30,
            background=C["card"],
            fieldbackground=C["card"],
            foreground=C["text"],
            bordercolor=C["border"],
            borderwidth=0,
        )
        s.configure(
            "Treeview.Heading",
            font=(FONT, 9, "bold"),
            background=C["neutral"],
            foreground=C["text"],
            padding=(8, 7),
            relief="flat",
        )
        s.map(
            "Treeview",
            background=[("selected", C["select"])],
            foreground=[("selected", C["text"])],
        )
        s.map("Treeview.Heading", background=[("active", C["neutral_hover"])])

        # Inputs
        s.configure("TEntry", padding=6, fieldbackground="#ffffff", bordercolor=C["border"])
        s.map("TEntry", bordercolor=[("focus", C["primary"])])
        s.configure("TSpinbox", padding=4, fieldbackground="#ffffff", bordercolor=C["border"])
        s.configure("Horizontal.TScale", background=C["card"], troughcolor=C["neutral"])

        # Scrollbars
        s.configure("Vertical.TScrollbar", background=C["neutral"], troughcolor=C["bg"], bordercolor=C["bg"])
        s.configure("Horizontal.TScrollbar", background=C["neutral"], troughcolor=C["bg"], bordercolor=C["bg"])

        # Status bar
        s.configure("Status.TLabel", background=C["card"], foreground=C["muted"], font=(FONT, 9))

        # Stats value labels
        s.configure("StatKey.TLabel", background=C["card"], foreground=C["muted"], font=(FONT, 9, "bold"))
        s.configure("StatVal.TLabel", background=C["card"], foreground=C["text"], font=(FONT, 10))

    def _build_header(self):
        """Build top banner with title and system status indicator."""
        header_frame = ttk.Frame(self.root, style="Header.TFrame", padding=(20, 14))
        header_frame.pack(fill=tk.X, side=tk.TOP)

        title_label = ttk.Label(
            header_frame,
            text="🧠 Hasini RAG Knowledge Base Manager",
            style="Header.TLabel",
        )
        title_label.pack(side=tk.LEFT, anchor=tk.W)

        subtitle_label = ttk.Label(
            header_frame,
            text="Manually Managed Vector & BM25 Hybrid Knowledge Retrieval",
            style="HeaderSub.TLabel",
        )
        subtitle_label.pack(side=tk.LEFT, anchor=tk.S, padx=(14, 0), pady=(0, 3))

        status_text = "🟢 RAG Active" if RAG_ENABLED else "🔴 RAG Disabled"
        self.status_badge = ttk.Label(
            header_frame,
            text=status_text,
            style="HeaderSub.TLabel",
            foreground="#4ade80" if RAG_ENABLED else "#f87171",
            font=(FONT, 10, "bold"),
        )
        self.status_badge.pack(side=tk.RIGHT, anchor=tk.E)

        # Thin accent line under the header
        tk.Frame(self.root, height=3, background=C["primary"]).pack(fill=tk.X, side=tk.TOP)

    def _build_notebook(self):
        """Build main notebook tabs: Documents, Search, Stats."""
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=14, pady=(12, 8))

        # Tab 1: Documents
        self.tab_docs = ttk.Frame(self.notebook, padding=14, style="Card.TFrame")
        self.notebook.add(self.tab_docs, text=" 📂 Documents ")
        self._build_docs_tab()

        # Tab 2: Search & Retrieval
        self.tab_search = ttk.Frame(self.notebook, padding=14, style="Card.TFrame")
        self.notebook.add(self.tab_search, text=" 🔍 Hybrid Search ")
        self._build_search_tab()

        # Tab 3: System Statistics
        self.tab_stats = ttk.Frame(self.notebook, padding=14, style="Card.TFrame")
        self.notebook.add(self.tab_stats, text=" 📊 Statistics & Info ")
        self._build_stats_tab()

    def _build_docs_tab(self):
        """Build document management interface with buttons, treeview table, and log console."""
        # Top Action Toolbar
        toolbar = ttk.Frame(self.tab_docs, style="Card.TFrame")
        toolbar.pack(fill=tk.X, side=tk.TOP, pady=(0, 12))

        btn_add_file = ttk.Button(
            toolbar,
            text="📄 Add File",
            style="Action.TButton",
            command=self._on_add_file,
        )
        btn_add_file.pack(side=tk.LEFT, padx=(0, 6))

        btn_add_url = ttk.Button(
            toolbar,
            text="🌐 Add Web URL",
            style="Action.TButton",
            command=self._on_add_url,
        )
        btn_add_url.pack(side=tk.LEFT, padx=6)

        btn_add_dir = ttk.Button(
            toolbar,
            text="📁 Add Folder",
            style="Action.TButton",
            command=self._on_add_folder,
        )
        btn_add_dir.pack(side=tk.LEFT, padx=6)

        btn_refresh = ttk.Button(
            toolbar,
            text="🔄 Refresh",
            command=self._refresh_documents,
        )
        btn_refresh.pack(side=tk.LEFT, padx=6)

        # Right-side action buttons
        btn_clear = ttk.Button(
            toolbar,
            text="⚠️ Clear All",
            style="Danger.TButton",
            command=self._on_clear_all,
        )
        btn_clear.pack(side=tk.RIGHT, padx=(6, 0))

        btn_delete = ttk.Button(
            toolbar,
            text="🗑️ Delete Selected",
            command=self._on_delete_selected,
        )
        btn_delete.pack(side=tk.RIGHT, padx=6)

        btn_reindex = ttk.Button(
            toolbar,
            text="♻️ Re-index Selected",
            command=self._on_reindex_selected,
        )
        btn_reindex.pack(side=tk.RIGHT, padx=6)

        # Filter entry box
        filter_frame = ttk.Frame(self.tab_docs, style="Card.TFrame")
        filter_frame.pack(fill=tk.X, side=tk.TOP, pady=(0, 8))

        ttk.Label(filter_frame, text="🔎 Filter:", style="Card.TLabel", font=(FONT, 9, "bold")).pack(
            side=tk.LEFT, padx=(0, 8)
        )
        self.doc_filter_var = tk.StringVar()
        self.doc_filter_var.trace_add("write", lambda *args: self._apply_doc_filter())
        entry_filter = ttk.Entry(filter_frame, textvariable=self.doc_filter_var)
        entry_filter.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Operation Log Output Box (packed before the tree so it keeps its space)
        log_frame = ttk.LabelFrame(self.tab_docs, text="Operation Activity Log", padding=6)
        log_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=(10, 0))

        log_inner = ttk.Frame(log_frame, style="Card.TFrame")
        log_inner.pack(fill=tk.BOTH, expand=True)

        self.log_text = tk.Text(
            log_inner,
            height=6,
            state=tk.DISABLED,
            font=(MONO, 9),
            wrap=tk.WORD,
            background=C["log_bg"],
            foreground=C["log_fg"],
            insertbackground=C["log_fg"],
            relief=tk.FLAT,
            padx=10,
            pady=8,
            borderwidth=0,
            highlightthickness=0,
        )
        log_vsb = ttk.Scrollbar(log_inner, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_vsb.set)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_vsb.pack(side=tk.RIGHT, fill=tk.Y)

        self.log_text.tag_configure("time", foreground="#64748b")
        self.log_text.tag_configure("ok", foreground="#4ade80")
        self.log_text.tag_configure("error", foreground="#f87171")
        self.log_text.tag_configure("info", foreground=C["log_fg"])

        # Documents Treeview Table
        tree_frame = ttk.Frame(self.tab_docs, style="Card.TFrame")
        tree_frame.pack(fill=tk.BOTH, expand=True)

        columns = ("filename", "file_type", "chunks", "timestamp", "source")
        self.doc_tree = ttk.Treeview(
            tree_frame,
            columns=columns,
            show="headings",
            selectmode="browse",
        )

        self.doc_tree.heading("filename", text="Filename", anchor=tk.W)
        self.doc_tree.heading("file_type", text="Type", anchor=tk.CENTER)
        self.doc_tree.heading("chunks", text="Chunks", anchor=tk.E)
        self.doc_tree.heading("timestamp", text="Indexed Date", anchor=tk.W)
        self.doc_tree.heading("source", text="Source Path / URL", anchor=tk.W)

        self.doc_tree.column("filename", width=220, minwidth=120)
        self.doc_tree.column("file_type", width=80, minwidth=60, anchor=tk.CENTER)
        self.doc_tree.column("chunks", width=80, minwidth=50, anchor=tk.E)
        self.doc_tree.column("timestamp", width=160, minwidth=120)
        self.doc_tree.column("source", width=360, minwidth=200)

        # Zebra striping for readability
        self.doc_tree.tag_configure("odd", background=C["card"])
        self.doc_tree.tag_configure("even", background=C["row_alt"])

        # Scrollbars for treeview
        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=self.doc_tree.yview)
        hsb = ttk.Scrollbar(tree_frame, orient="horizontal", command=self.doc_tree.xview)
        self.doc_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        self.doc_tree.grid(column=0, row=0, sticky="nsew")
        vsb.grid(column=1, row=0, sticky="ns")
        hsb.grid(column=0, row=1, sticky="ew")

        tree_frame.grid_columnconfigure(0, weight=1)
        tree_frame.grid_rowconfigure(0, weight=1)

    def _build_search_tab(self):
        """Build hybrid query test panel with threshold sliders and results view."""
        # Top Query Bar
        query_frame = ttk.Frame(self.tab_search, style="Card.TFrame")
        query_frame.pack(fill=tk.X, side=tk.TOP, pady=(0, 12))

        ttk.Label(query_frame, text="Query:", style="Card.TLabel", font=(FONT, 10, "bold")).pack(
            side=tk.LEFT, padx=(0, 8)
        )
        self.search_entry = ttk.Entry(query_frame, font=(FONT, 11))
        self.search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        self.search_entry.bind("<Return>", lambda event: self._on_search())

        btn_search = ttk.Button(
            query_frame,
            text="🔍 Search RAG",
            style="Action.TButton",
            command=self._on_search,
        )
        btn_search.pack(side=tk.LEFT)

        # Options & Sliders
        params_frame = ttk.LabelFrame(self.tab_search, text="Retrieval Parameters", padding=12)
        params_frame.pack(fill=tk.X, side=tk.TOP, pady=(0, 12))
        params_frame.grid_columnconfigure(5, weight=1)

        # Top-K
        ttk.Label(params_frame, text="Top-K Chunks:", style="Card.TLabel").grid(
            row=0, column=0, sticky=tk.W, padx=5, pady=2
        )
        self.top_k_var = tk.IntVar(value=RAG_TOP_K)
        top_k_spin = ttk.Spinbox(params_frame, from_=1, to=30, textvariable=self.top_k_var, width=5)
        top_k_spin.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)

        # Mode summary
        mode_text = f"Mode: Vector + BM25 {'+ CrossEncoder Reranker' if RAG_RERANKER_ENABLED else ''}"
        ttk.Label(params_frame, text=mode_text, style="CardMuted.TLabel", font=(FONT, 8, "italic")).grid(
            row=0, column=2, sticky=tk.E, padx=(20, 5), pady=2
        )

        # Search Results Notebook / Panes
        self.search_results_nb = ttk.Notebook(self.tab_search)
        self.search_results_nb.pack(fill=tk.BOTH, expand=True)

        # Tab: Formatted Context
        context_frame = ttk.Frame(self.search_results_nb, style="Card.TFrame")
        self.search_results_nb.add(context_frame, text=" 📝 Formatted LLM Context ")

        self.context_text = tk.Text(
            context_frame,
            font=(MONO, 10),
            wrap=tk.WORD,
            background="#f8fafc",
            foreground=C["text"],
            relief=tk.FLAT,
            padx=14,
            pady=12,
            borderwidth=0,
            highlightthickness=1,
            highlightbackground=C["border"],
            highlightcolor=C["primary"],
            spacing1=2,
            spacing3=2,
        )
        context_vsb = ttk.Scrollbar(context_frame, orient="vertical", command=self.context_text.yview)
        self.context_text.configure(yscrollcommand=context_vsb.set)
        self.context_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        context_vsb.pack(side=tk.RIGHT, fill=tk.Y)

        # Tab: Detailed Chunk Cards
        chunks_frame = ttk.Frame(self.search_results_nb, style="Card.TFrame")
        self.search_results_nb.add(chunks_frame, text=" 🧩 Individual Chunks ")

        self.chunks_text = tk.Text(
            chunks_frame,
            font=(FONT, 10),
            wrap=tk.WORD,
            background="#ffffff",
            foreground=C["text"],
            relief=tk.FLAT,
            padx=14,
            pady=12,
            borderwidth=0,
            highlightthickness=1,
            highlightbackground=C["border"],
            highlightcolor=C["primary"],
            spacing1=2,
            spacing3=2,
        )
        chunks_vsb = ttk.Scrollbar(chunks_frame, orient="vertical", command=self.chunks_text.yview)
        self.chunks_text.configure(yscrollcommand=chunks_vsb.set)
        self.chunks_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        chunks_vsb.pack(side=tk.RIGHT, fill=tk.Y)

        # Visual tags for chunk cards
        self.chunks_text.tag_configure(
            "chunk_header",
            font=(FONT, 9, "bold"),
            foreground="#1e3a8a",
            background="#dbeafe",
            spacing1=10,
            spacing3=6,
            lmargin1=0,
            lmargin2=0,
        )
        self.chunks_text.tag_configure("chunk_body", lmargin1=10, lmargin2=10, spacing3=4)
        self.chunks_text.tag_configure("empty_hint", foreground=C["muted"], font=(FONT, 10, "italic"))

    def _build_stats_tab(self):
        """Build statistics display tab."""
        stats_frame = ttk.LabelFrame(self.tab_stats, text="Knowledge Store Information", padding=18)
        stats_frame.pack(fill=tk.BOTH, expand=True)
        stats_frame.grid_columnconfigure(1, weight=1)

        self.stats_labels: Dict[str, ttk.Label] = {}
        items = [
            ("RAG Status", "rag_status"),
            ("Database Path", "db_path"),
            ("Embedding Model", "embed_model"),
            ("Total Unique Documents", "total_docs"),
            ("Total Vector Chunks", "total_chunks"),
            ("BM25 Search", "bm25_status"),
            ("Cross-Encoder Reranker", "reranker_status"),
            ("Default Top-K", "default_top_k"),
        ]

        for idx, (label_title, key) in enumerate(items):
            row = idx * 2  # leave odd rows for separators
            ttk.Label(stats_frame, text=f"{label_title}", style="StatKey.TLabel").grid(
                row=row, column=0, sticky=tk.W, padx=10, pady=8
            )
            val_lbl = ttk.Label(stats_frame, text="Loading...", style="StatVal.TLabel")
            val_lbl.grid(row=row, column=1, sticky=tk.W, padx=10, pady=8)
            self.stats_labels[key] = val_lbl

            if idx < len(items) - 1:
                ttk.Separator(stats_frame, orient="horizontal").grid(
                    row=row + 1, column=0, columnspan=2, sticky="ew", padx=6
                )

        btn_refresh_stats = ttk.Button(
            stats_frame,
            text="🔄 Refresh Statistics",
            style="Action.TButton",
            command=self._refresh_stats,
        )
        btn_refresh_stats.grid(row=len(items) * 2, column=0, columnspan=2, sticky=tk.W, padx=10, pady=(20, 4))

    def _build_statusbar(self):
        """Build status bar at bottom of application window."""
        tk.Frame(self.root, height=1, background=C["border"]).pack(fill=tk.X, side=tk.BOTTOM)
        self.statusbar = ttk.Label(
            self.root,
            text="Initializing RAG Backend...",
            style="Status.TLabel",
            anchor=tk.W,
            padding=(14, 6),
        )
        self.statusbar.pack(fill=tk.X, side=tk.BOTTOM)

    # ================= Background Tasks & Event Handlers =================

    def _async_init_backend(self):
        """Initialize ChromaDB and RAG Indexer/Retriever in background thread."""
        def task():
            try:
                self.vector_store = RAGVectorStore()
                self.indexer = RAGIndexer(vector_store=self.vector_store)
                self.retriever = RAGRetriever(vector_store=self.vector_store)
                self.root.after(0, self._on_backend_ready)
            except Exception as exc:
                logger.error("Failed to initialize backend: %s", exc)
                self.root.after(0, lambda: self._set_status(f"Error loading RAG backend: {exc}"))

        threading.Thread(target=task, daemon=True).start()

    def _on_backend_ready(self):
        """Called on main UI thread once RAG engines are initialized."""
        self._set_status("RAG Knowledge Base Ready.")
        self._log("RAG System loaded successfully.")
        self._refresh_documents()
        self._refresh_stats()

    def _set_status(self, message: str):
        """Update status bar text."""
        self.statusbar.config(text=message)

    def _log(self, message: str):
        """Append line to operational activity log box."""
        lowered = message.lower()
        if "error" in lowered or "failed" in lowered:
            tag = "error"
        elif any(k in lowered for k in ("success", "indexed", "cleared", "removed", "ready", "complete")):
            tag = "ok"
        else:
            tag = "info"

        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, time.strftime("[%H:%M:%S] "), "time")
        self.log_text.insert(tk.END, f"{message}\n", tag)
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def _refresh_documents(self):
        """Fetch and populate indexed documents into treeview."""
        if not self.indexer:
            return

        def task():
            try:
                docs = self.indexer.list_documents()
                self.root.after(0, lambda: self._populate_doc_tree(docs))
            except Exception as exc:
                self.root.after(0, lambda: self._log(f"Error listing documents: {exc}"))

        threading.Thread(target=task, daemon=True).start()

    def _populate_doc_tree(self, docs: List[Dict[str, Any]]):
        """Populate treeview table on UI thread."""
        self.raw_documents = docs
        self._apply_doc_filter()
        total_docs = len(docs)
        total_chunks = sum(d.get("chunk_count", 0) for d in docs)
        self._set_status(f"Knowledge Base: {total_docs} document(s), {total_chunks} total chunk(s).")

    def _apply_doc_filter(self):
        """Filter items in document treeview based on search filter input."""
        if not hasattr(self, "raw_documents"):
            return

        query = self.doc_filter_var.get().lower().strip()
        self.doc_tree.delete(*self.doc_tree.get_children())

        row_index = 0
        for d in self.raw_documents:
            filename = d.get("filename", "N/A")
            file_type = d.get("file_type", "N/A")
            chunks = str(d.get("chunk_count", 0))
            timestamp = d.get("timestamp", "N/A")
            source = d.get("source", "N/A")

            if query and not (query in filename.lower() or query in source.lower() or query in file_type.lower()):
                continue

            self.doc_tree.insert(
                "",
                tk.END,
                values=(filename, file_type, chunks, timestamp, source),
                tags=("even" if row_index % 2 else "odd",),
            )
            row_index += 1

    def _on_add_file(self):
        """Prompt file chooser and index selected file."""
        if not self.indexer:
            messagebox.showwarning("RAG Loading", "RAG Engine is still initializing. Please wait.")
            return

        file_types = [
            ("Supported Documents", "*.pdf *.txt *.docx *.md *.markdown *.csv *.json *.html *.htm"),
            ("PDF Documents", "*.pdf"),
            ("Text Files", "*.txt *.md *.markdown"),
            ("Word Documents", "*.docx"),
            ("Data Files", "*.csv *.json *.html"),
            ("All Files", "*.*"),
        ]
        file_path = filedialog.askopenfilename(title="Select Document to Index", filetypes=file_types)
        if not file_path:
            return

        self._run_indexing_task(file_path)

    def _on_add_url(self):
        """Prompt dialog for Web URL input and index it."""
        if not self.indexer:
            messagebox.showwarning("RAG Loading", "RAG Engine is still initializing. Please wait.")
            return

        url = simpledialog.askstring("Add Web URL", "Enter website URL to index into RAG knowledge base:")
        if not url or not url.strip():
            return

        url = url.strip()
        if not (url.startswith("http://") or url.startswith("https://")):
            url = "https://" + url

        self._run_indexing_task(url)

    def _on_add_folder(self):
        """Prompt directory picker and batch index folder."""
        if not self.indexer:
            messagebox.showwarning("RAG Loading", "RAG Engine is still initializing. Please wait.")
            return

        folder_path = filedialog.askdirectory(title="Select Directory to Batch Index")
        if not folder_path:
            return

        self._set_status(f"Batch indexing folder: {folder_path}...")
        self._log(f"Starting batch index for directory: {folder_path}")

        def task():
            try:
                results = self.indexer.index_directory(folder_path, recursive=True)
                success_count = sum(1 for r in results if r.get("success"))
                msg = f"Indexed {success_count}/{len(results)} file(s) from folder."
                self.root.after(0, lambda: self._on_operation_done(msg))
            except Exception as exc:
                self.root.after(0, lambda: self._log(f"Folder indexing error: {exc}"))

        threading.Thread(target=task, daemon=True).start()

    def _run_indexing_task(self, source: str):
        """Run single document indexing in background thread."""
        self._set_status(f"Indexing source: {source}...")
        self._log(f"Indexing: {source}")

        def task():
            try:
                res = self.indexer.index_document(source)
                msg = res.get("message", "Indexing complete.")
                self.root.after(0, lambda: self._on_operation_done(msg))
            except Exception as exc:
                self.root.after(0, lambda: self._log(f"Error indexing '{source}': {exc}"))

        threading.Thread(target=task, daemon=True).start()

    def _on_delete_selected(self):
        """Delete selected document from treeview."""
        selected = self.doc_tree.selection()
        if not selected:
            messagebox.showinfo("Select Document", "Please select a document from the table to delete.")
            return

        item = self.doc_tree.item(selected[0])
        source = item["values"][4]
        filename = item["values"][0]

        if not messagebox.askyesno("Confirm Deletion", f"Remove '{filename}' from RAG Knowledge Base?"):
            return

        self._set_status(f"Deleting '{filename}'...")

        def task():
            try:
                res = self.indexer.remove_document(source)
                msg = res.get("message", "Document removed.")
                self.root.after(0, lambda: self._on_operation_done(msg))
            except Exception as exc:
                self.root.after(0, lambda: self._log(f"Error deleting '{source}': {exc}"))

        threading.Thread(target=task, daemon=True).start()

    def _on_reindex_selected(self):
        """Re-index selected document from treeview."""
        selected = self.doc_tree.selection()
        if not selected:
            messagebox.showinfo("Select Document", "Please select a document from the table to re-index.")
            return

        item = self.doc_tree.item(selected[0])
        source = item["values"][4]
        filename = item["values"][0]

        self._run_indexing_task(source)

    def _on_clear_all(self):
        """Clear all knowledge from store."""
        if not messagebox.askyesno(
            "⚠️ Confirm Clear Knowledge Base",
            "Are you sure you want to delete ALL indexed documents and vectors?\nThis operation cannot be undone.",
            icon="warning",
        ):
            return

        self._set_status("Clearing knowledge base...")

        def task():
            try:
                count = self.indexer.clear_all()
                msg = f"Cleared knowledge base ({count} chunk(s) removed)."
                self.root.after(0, lambda: self._on_operation_done(msg))
            except Exception as exc:
                self.root.after(0, lambda: self._log(f"Error clearing knowledge base: {exc}"))

        threading.Thread(target=task, daemon=True).start()

    def _on_operation_done(self, message: str):
        """Callback when an operation finishes on backend."""
        self._log(message)
        self._set_status(message)
        self._refresh_documents()
        self._refresh_stats()

    def _on_search(self):
        """Execute RAG search and display formatted results."""
        if not self.retriever:
            messagebox.showwarning("RAG Loading", "RAG Engine is still initializing. Please wait.")
            return

        query = self.search_entry.get().strip()
        if not query:
            messagebox.showinfo("Empty Query", "Please enter a search query string.")
            return

        top_k = self.top_k_var.get()

        self._set_status(f"Searching knowledge base for: '{query}'...")

        def task():
            try:
                chunks = self.retriever.retrieve(
                    query=query,
                    top_k=top_k,
                )
                formatted_context = self.retriever.format_context(chunks) if chunks else "No relevant context found."
                self.root.after(0, lambda: self._display_search_results(query, chunks, formatted_context))
            except Exception as exc:
                self.root.after(0, lambda: self._log(f"Search error: {exc}"))

        threading.Thread(target=task, daemon=True).start()

    def _display_search_results(self, query: str, chunks: List[Dict[str, Any]], context_str: str):
        """Display search results on UI thread."""
        self._set_status(f"Search complete for '{query}'. {len(chunks)} result(s) returned.")

        # Update Context tab
        self.context_text.config(state=tk.NORMAL)
        self.context_text.delete("1.0", tk.END)
        self.context_text.insert(tk.END, context_str)
        self.context_text.config(state=tk.NORMAL)

        # Update Chunks tab
        self.chunks_text.config(state=tk.NORMAL)
        self.chunks_text.delete("1.0", tk.END)

        if not chunks:
            self.chunks_text.insert(
                tk.END,
                f"No relevant chunks matched query '{query}'.\n\n"
                "Tip: Check if documents are indexed or try a different search term.",
                "empty_hint",
            )
        else:
            for idx, c in enumerate(chunks, 1):
                meta = c.get("metadata", {})
                sim = c.get("similarity", 0.0)
                dist = c.get("distance", 0.0)
                filename = meta.get("filename", "Unknown")
                chunk_id = meta.get("chunk_id", f"chunk_{idx}")
                text = c.get("text", "").strip()

                header = f"--- Chunk {idx} | Source: {filename} | ID: {chunk_id} | Similarity: {sim:.4f} (Dist: {dist:.4f}) ---\n"
                self.chunks_text.insert(tk.END, header, "chunk_header")
                self.chunks_text.insert(tk.END, f"{text}\n\n", "chunk_body")

        self.chunks_text.config(state=tk.NORMAL)

    def _refresh_stats(self):
        """Update system statistics tab."""
        if not self.vector_store:
            return

        def task():
            try:
                stats = self.vector_store.get_stats()
                self.root.after(0, lambda: self._update_stats_ui(stats))
            except Exception as exc:
                logger.error("Failed to fetch stats: %s", exc)

        threading.Thread(target=task, daemon=True).start()

    def _update_stats_ui(self, stats: Dict[str, Any]):
        """Render statistics on UI thread."""
        self.stats_labels["rag_status"].config(
            text="ENABLED" if RAG_ENABLED else "DISABLED",
            foreground="#16a34a" if RAG_ENABLED else "#dc2626",
        )
        self.stats_labels["db_path"].config(text=stats.get("db_path", CHROMADB_PATH))
        self.stats_labels["embed_model"].config(text=RAG_EMBEDDING_MODEL)
        self.stats_labels["total_docs"].config(text=str(stats.get("total_documents", 0)))
        self.stats_labels["total_chunks"].config(text=str(stats.get("total_chunks", 0)))
        self.stats_labels["bm25_status"].config(text="Active" if RAG_BM25_ENABLED else "Disabled")
        self.stats_labels["reranker_status"].config(text="Active" if RAG_RERANKER_ENABLED else "Disabled")
        self.stats_labels["default_top_k"].config(text=str(RAG_TOP_K))


def launch_gui():
    """Launch native Tkinter Desktop Application."""
    root = tk.Tk()
    app = RAGManagerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    launch_gui()