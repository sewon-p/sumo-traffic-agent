"""
Traffic Regulation RAG (Retrieval-Augmented Generation)

Retrieves relevant Korean traffic regulations and injects them
as context into the FT prompt. Covers scenarios the training data
does not: school zones, construction zones, weather rules, etc.
"""

import json
import os
from typing import Optional

_REGULATIONS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "traffic_regulations.jsonl",
)

_regulations = []
_embeddings_cache = {}


def _load_regulations():
    """Load regulations from JSONL file."""
    global _regulations
    if _regulations:
        return _regulations
    if not os.path.exists(_REGULATIONS_PATH):
        return []
    with open(_REGULATIONS_PATH, encoding="utf-8") as f:
        _regulations = [json.loads(line) for line in f if line.strip()]
    return _regulations


def _get_embedding(text: str, api_key: str = None) -> list:
    """Get OpenAI embedding for a text string."""
    if text in _embeddings_cache:
        return _embeddings_cache[text]
    try:
        from openai import OpenAI
        key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not key:
            return []
        client = OpenAI(api_key=key)
        resp = client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
        )
        emb = resp.data[0].embedding
        _embeddings_cache[text] = emb
        return emb
    except Exception:
        return []


def _cosine_similarity(a: list, b: list) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _keyword_search(query: str, top_k: int = 3) -> list:
    """Fallback: simple keyword matching when embeddings unavailable."""
    regs = _load_regulations()
    if not regs:
        return []

    query_lower = query.lower()
    scored = []
    for reg in regs:
        text = reg.get("text", "") + " " + reg.get("category", "")
        score = sum(1 for word in query_lower.split() if word in text.lower())
        # Boost for category match
        cat = reg.get("category", "")
        if cat and cat in query:
            score += 3
        if score >= 2:
            scored.append((score, reg))

    scored.sort(key=lambda x: -x[0])
    return [r for _, r in scored[:top_k]]


def search(query: str, top_k: int = 3, api_key: str = None) -> list:
    """
    Search for relevant traffic regulations.

    Uses OpenAI embeddings if available, falls back to keyword matching.

    Returns:
        List of regulation dicts with keys: id, category, law, text, params
    """
    regs = _load_regulations()
    if not regs:
        return []

    # Try embedding-based search
    query_emb = _get_embedding(query, api_key)
    if not query_emb:
        return _keyword_search(query, top_k)

    scored = []
    for reg in regs:
        search_text = f"{reg.get('category', '')} {reg.get('text', '')}"
        reg_emb = _get_embedding(search_text, api_key)
        if reg_emb:
            sim = _cosine_similarity(query_emb, reg_emb)
            scored.append((sim, reg))

    scored.sort(key=lambda x: -x[0])
    return [r for _, r in scored[:top_k]]


def format_context(regulations: list) -> str:
    """Format retrieved regulations as prompt context."""
    if not regulations:
        return ""
    lines = []
    for reg in regulations:
        law = reg.get("law", "")
        text = reg.get("text", "")
        lines.append(f"[{law}] {text}")
    return "\n".join(lines)


def enrich_prompt(user_input: str, top_k: int = 3, api_key: str = None) -> str:
    """
    Enrich user input with relevant traffic regulations.

    Returns the original input with appended regulation context,
    or the original input unchanged if no relevant regulations found.
    """
    regs = search(user_input, top_k=top_k, api_key=api_key)
    if not regs:
        return user_input

    context = format_context(regs)
    return f"{user_input}\n\n[참고 규정]\n{context}"
