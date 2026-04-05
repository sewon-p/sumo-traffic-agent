"""
Traffic Regulation RAG (Retrieval-Augmented Generation)

Chunks Korean traffic law text by article, embeds into ChromaDB,
and retrieves relevant regulations to enrich FT prompts.

Data sources:
  - data/road_traffic_act.txt (도로교통법, ~160 articles)
  - data/road_traffic_act_enforcement.txt (시행규칙, ~80 articles)
"""

import json
import os
import re
from typing import Optional

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
_CHROMA_DIR = os.path.join(_DATA_DIR, "chroma_db")

_LAW_FILES = [
    ("도로교통법", os.path.join(_DATA_DIR, "road_traffic_act.txt")),
    ("도로교통법 시행규칙", os.path.join(_DATA_DIR, "road_traffic_act_enforcement.txt")),
]

_collection = None


def _chunk_law_text(text: str, source: str) -> list[dict]:
    """Split law text into article-level chunks."""
    chunks = []
    # Match 제X조, 제X조의2 patterns
    pattern = re.compile(r'^(제\d+조(?:의\d+)?)\(([^)]+)\)', re.MULTILINE)

    matches = list(pattern.finditer(text))
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        article_id = match.group(1)
        article_title = match.group(2)
        article_text = text[start:end].strip()

        # Skip very short articles (just a title with no content)
        if len(article_text) < 20:
            continue

        # Truncate very long articles (definitions, etc.) to 1500 chars
        if len(article_text) > 1500:
            article_text = article_text[:1500] + "..."

        chunks.append({
            "id": f"{source}_{article_id}",
            "source": source,
            "article_id": article_id,
            "title": article_title,
            "text": article_text,
        })

    return chunks


def _get_openai_embedding_fn():
    """Create OpenAI embedding function for ChromaDB."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return None
    try:
        from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
        return OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name="text-embedding-3-small",
        )
    except Exception:
        return None


def _get_collection():
    """Get or create the ChromaDB collection."""
    global _collection
    if _collection is not None:
        return _collection

    try:
        import chromadb
    except ImportError:
        return None

    embedding_fn = _get_openai_embedding_fn()
    if not embedding_fn:
        return None

    client = chromadb.Client(chromadb.Settings(
        persist_directory=_CHROMA_DIR,
        anonymized_telemetry=False,
    ))

    _collection = client.get_or_create_collection(
        name="traffic_regulations",
        embedding_function=embedding_fn,
    )

    # Build index if empty
    if _collection.count() == 0:
        _build_index()

    return _collection


def _build_index():
    """Load law files, chunk, and add to ChromaDB."""
    if _collection is None:
        return

    all_chunks = []
    for source, path in _LAW_FILES:
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            text = f.read()
        chunks = _chunk_law_text(text, source)
        all_chunks.extend(chunks)

    if not all_chunks:
        return

    # Add in batches (ChromaDB limit)
    batch_size = 100
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i + batch_size]
        _collection.add(
            ids=[c["id"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[{
                "source": c["source"],
                "article_id": c["article_id"],
                "title": c["title"],
            } for c in batch],
        )

    print(f"RAG index built: {len(all_chunks)} articles from {len(_LAW_FILES)} law files")


def _keyword_search(query: str, top_k: int = 3) -> list[dict]:
    """Fallback: simple keyword matching when ChromaDB unavailable."""
    results = []
    for source, path in _LAW_FILES:
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            text = f.read()
        chunks = _chunk_law_text(text, source)

        query_words = set(query.lower().split())
        for chunk in chunks:
            chunk_text = chunk["text"].lower()
            score = sum(1 for w in query_words if w in chunk_text)
            # Boost for title match
            title = chunk.get("title", "").lower()
            score += sum(3 for w in query_words if w in title)
            if score >= 2:
                results.append((score, chunk))

    results.sort(key=lambda x: -x[0])
    return [r for _, r in results[:top_k]]


def search(query: str, top_k: int = 3) -> list[dict]:
    """
    Search for relevant traffic regulations.

    Uses ChromaDB vector search if available, falls back to keyword matching.

    Returns:
        List of dicts with keys: source, article_id, title, text
    """
    collection = _get_collection()

    if collection is not None and collection.count() > 0:
        try:
            results = collection.query(
                query_texts=[query],
                n_results=top_k,
            )
            items = []
            for i, doc in enumerate(results["documents"][0]):
                meta = results["metadatas"][0][i]
                items.append({
                    "source": meta.get("source", ""),
                    "article_id": meta.get("article_id", ""),
                    "title": meta.get("title", ""),
                    "text": doc,
                })
            return items
        except Exception:
            pass

    return _keyword_search(query, top_k)


def format_context(regulations: list[dict]) -> str:
    """Format retrieved regulations as prompt context."""
    if not regulations:
        return ""
    lines = []
    for reg in regulations:
        source = reg.get("source", "")
        article = reg.get("article_id", "")
        # Use first 300 chars of text to keep prompt concise
        text = reg.get("text", "")
        if len(text) > 300:
            text = text[:300] + "..."
        lines.append(f"[{source} {article}] {text}")
    return "\n".join(lines)


def enrich_prompt(user_input: str, top_k: int = 3, api_key: str = None) -> str:
    """
    Enrich user input with relevant traffic regulations.

    Returns the original input with appended regulation context,
    or the original input unchanged if no relevant regulations found.
    """
    regs = search(user_input, top_k=top_k)
    if not regs:
        return user_input

    context = format_context(regs)
    return f"{user_input}\n\n[참고 규정]\n{context}"
