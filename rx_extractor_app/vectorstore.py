"""
vectorstore.py
--------------
FAISS-backed storage/retrieval of the system prompt with graceful fallback.

Design:
  - System prompt is embedded/indexed once.
  - Stored index is reused on subsequent queries.
  - If FAISS or GPT4AllEmbeddings is not yet configured, provides a reliable
    in-memory prompt store so the app never crashes.
"""
import os
from typing import Any, Optional

import config
from prompt import SYSTEM_PROMPT

_embeddings = None


def get_embeddings() -> Any:
    global _embeddings
    if _embeddings is None:
        try:
            from langchain_community.embeddings import GPT4AllEmbeddings
            _embeddings = GPT4AllEmbeddings(model_name=config.EMBEDDINGS_MODEL)
        except Exception:
            # Fallback simple embedding object if GPT4AllEmbeddings cannot be loaded
            class SimplePromptEmbedding:
                def embed_query(self, text: str):
                    return [0.0] * 384
                def embed_documents(self, texts):
                    return [[0.0] * 384 for _ in texts]
            _embeddings = SimplePromptEmbedding()
    return _embeddings


def _index_exists() -> bool:
    return os.path.exists(os.path.join(config.FAISS_DIR, "index.faiss"))


def load_or_create_index() -> Any:
    """Embed + store the system prompt exactly once; load it on every call after that."""
    try:
        from langchain_community.vectorstores import FAISS
        from langchain_core.documents import Document

        embeddings = get_embeddings()

        if _index_exists():
            return FAISS.load_local(
                config.FAISS_DIR, embeddings, allow_dangerous_deserialization=True
            )

        doc = Document(page_content=SYSTEM_PROMPT, metadata={"type": "system_prompt"})
        store = FAISS.from_documents([doc], embeddings)
        store.save_local(config.FAISS_DIR)
        return store
    except Exception:
        # Graceful fallback: return system prompt wrapper
        class InMemPromptStore:
            def __init__(self, content: str):
                self.content = content
            def similarity_search(self, *args, **kwargs):
                from langchain_core.documents import Document
                return [Document(page_content=self.content)]
        return InMemPromptStore(SYSTEM_PROMPT)


def retrieve_system_prompt(store: Any) -> str:
    """Pull the stored system prompt back out of FAISS (top-1 retrieval)."""
    if store is None:
        return SYSTEM_PROMPT

    if hasattr(store, "content"):
        return store.content

    # 1. Fast direct docstore lookup if index mapping is present
    if hasattr(store, "index_to_docstore_id") and store.index_to_docstore_id:
        doc_id = store.index_to_docstore_id.get(0)
        if doc_id is not None and hasattr(store, "docstore"):
            doc = store.docstore.search(doc_id)
            if hasattr(doc, "page_content") and doc.page_content:
                return doc.page_content

    # 2. Fast lookup via docstore values dict if present
    if hasattr(store, "docstore") and hasattr(store.docstore, "_dict") and store.docstore._dict:
        doc = next(iter(store.docstore._dict.values()))
        if hasattr(doc, "page_content") and doc.page_content:
            return doc.page_content

    # 3. Fallback similarity search
    try:
        results = store.similarity_search("expert doctor assistant extraction format", k=1)
        if results and results[0].page_content:
            return results[0].page_content
    except Exception:
        pass

    return SYSTEM_PROMPT
