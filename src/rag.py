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

# ---------------------------------------------------------------------------
# Query expansion dictionary: short Korean traffic terms → related synonyms
# ---------------------------------------------------------------------------
_QUERY_EXPANSION: dict[str, list[str]] = {
    "학교": ["어린이", "보호구역", "스쿨존"],
    "등하교": ["어린이", "보호구역", "통학"],
    "스쿨존": ["어린이", "보호구역", "학교"],
    "공사": ["공사구간", "도로공사"],
    "터널": ["터널", "앞지르기", "등화", "다리"],
    "비": ["우천", "노면", "감속"],
    "비오는": ["우천", "노면", "감속", "비"],
    "우천": ["비", "노면", "감속"],
    "눈": ["적설", "노면", "감속"],
    "안개": ["가시거리", "감속", "폭우"],
    "자율주행": ["자율주행", "시범구역", "시험운행", "자율주행자동차", "자율주행시스템"],
    "고속도로": ["고속도로", "속도", "최고속도", "최저속도"],
    "속도": ["속도", "최고속도", "감속", "제한"],
    "음주": ["음주", "혈중알코올", "음주운전"],
    "횡단보도": ["횡단보도", "보행자", "일시정지"],
    "신호": ["신호기", "신호등", "교차로"],
    "주차": ["주차", "정차", "주차금지"],
    "어린이": ["어린이", "보호구역", "통학버스"],
    "노인": ["노인", "보호구역", "장애인"],
}

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def _expand_query(query: str) -> str:
    """Expand a short Korean traffic query with synonyms/related terms.

    Looks up each word in ``_QUERY_EXPANSION`` and appends any additional
    terms that are not already present in the query.  Returns the enriched
    query string (original + expansion terms).
    """
    words = query.split()
    extra: list[str] = []
    for word in words:
        # Try exact match first, then check if any key is a substring of the word
        expansions = _QUERY_EXPANSION.get(word, [])
        if not expansions:
            for key, values in _QUERY_EXPANSION.items():
                if key in word:
                    expansions = values
                    break
        for term in expansions:
            if term not in query and term not in extra:
                extra.append(term)
    if extra:
        return query + " " + " ".join(extra)
    return query


_CHROMA_DIR = os.path.join(_DATA_DIR, "chroma_db")

_LAW_FILES = [
    ("Road Traffic Act", os.path.join(_DATA_DIR, "road_traffic_act_en.txt")),
    ("Enforcement Rules", os.path.join(_DATA_DIR, "road_traffic_act_enforcement_en.txt")),
    ("도로교통법", os.path.join(_DATA_DIR, "road_traffic_act.txt")),
    ("도로교통법 시행규칙", os.path.join(_DATA_DIR, "road_traffic_act_enforcement.txt")),
]

_collection = None


def _chunk_law_text(text: str, source: str) -> list[dict]:
    """Split law text into article-level chunks."""
    chunks = []
    # Match Korean 제X조 or English Article X patterns
    pattern = re.compile(
        r'^(제\d+조(?:의\d+)?)\(([^)]+)\)|^(Article\s+\d+(?:-\d+)?)\s*\(([^)]+)\)',
        re.MULTILINE,
    )

    matches = list(pattern.finditer(text))
    seen_ids = {}
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        # Korean groups: 1,2 / English groups: 3,4
        article_id = match.group(1) or match.group(3)
        article_title = match.group(2) or match.group(4)
        article_text = text[start:end].strip()

        # Skip very short articles (just a title with no content)
        if len(article_text) < 20:
            continue

        # Truncate very long articles (definitions, etc.) to 1500 chars
        if len(article_text) > 1500:
            article_text = article_text[:1500] + "..."

        base_id = f"{source}_{article_id}"
        seen_ids[base_id] = seen_ids.get(base_id, 0) + 1
        chunk_id = base_id if seen_ids[base_id] == 1 else f"{base_id}_{seen_ids[base_id]}"

        chunks.append({
            "id": chunk_id,
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
    """Keyword matching with 5x title weighting.

    Scores each article: +1 per query-word found in body, +5 per
    query-word found in the article title.  Articles scoring below
    ``min_score`` are discarded.
    """
    min_score = 2
    results: list[tuple[int, dict]] = []
    for source, path in _LAW_FILES:
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            text = f.read()
        chunks = _chunk_law_text(text, source)

        query_words = set(query.lower().split())
        for chunk in chunks:
            chunk_text = chunk["text"].lower()
            title = chunk.get("title", "").lower()

            # Body match: +1 per word
            score = sum(1 for w in query_words if w in chunk_text)
            # Title match: +5 per word (primary match target)
            score += sum(5 for w in query_words if w in title)

            if score >= min_score:
                results.append((score, chunk))

    results.sort(key=lambda x: -x[0])
    return [r for _, r in results[:top_k]]


def _rank_score(rank: int, n: int) -> float:
    """Convert a 0-based rank into a score in [0, 1].

    The top result (rank 0) receives 1.0; the last result receives 1/n.
    """
    if n <= 0:
        return 0.0
    return (n - rank) / n


def search(query: str, top_k: int = 3) -> list[dict]:
    """
    Search for relevant traffic regulations.

    Uses hybrid search when ChromaDB is available:
      1. Expand the query with domain synonyms.
      2. Run embedding search (top-10) and keyword search (top-10).
      3. Combine: ``0.6 * embedding_rank_score + 0.4 * keyword_rank_score``.
      4. Return top-``top_k`` by combined score.

    Falls back to keyword-only search when ChromaDB is unavailable.

    Returns:
        List of dicts with keys: source, article_id, title, text
    """
    expanded_query = _expand_query(query)

    collection = _get_collection()
    _HYBRID_K = 10  # candidates from each source

    if collection is not None and collection.count() > 0:
        try:
            # --- Embedding search (top-10) ---------------------------------
            emb_results = collection.query(
                query_texts=[expanded_query],
                n_results=_HYBRID_K,
            )
            emb_items: list[dict] = []
            for i, doc in enumerate(emb_results["documents"][0]):
                meta = emb_results["metadatas"][0][i]
                emb_items.append({
                    "source": meta.get("source", ""),
                    "article_id": meta.get("article_id", ""),
                    "title": meta.get("title", ""),
                    "text": doc,
                })

            # --- Keyword search (top-10) -----------------------------------
            kw_items = _keyword_search(expanded_query, top_k=_HYBRID_K)

            # --- Merge via rank-based scoring ------------------------------
            # Key = (source, article_id) to deduplicate
            scores: dict[tuple[str, str], float] = {}
            item_map: dict[tuple[str, str], dict] = {}

            for rank, item in enumerate(emb_items):
                key = (item["source"], item["article_id"])
                scores[key] = scores.get(key, 0.0) + 0.6 * _rank_score(rank, len(emb_items))
                item_map.setdefault(key, item)

            for rank, item in enumerate(kw_items):
                key = (item["source"], item["article_id"])
                scores[key] = scores.get(key, 0.0) + 0.4 * _rank_score(rank, len(kw_items))
                item_map.setdefault(key, item)

            ranked = sorted(scores.items(), key=lambda kv: -kv[1])
            return [item_map[k] for k, _ in ranked[:top_k]]

        except Exception:
            pass

    # Fallback: keyword-only
    return _keyword_search(expanded_query, top_k)


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
