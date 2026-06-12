"""Embedding + Vector Store + Retrieval (Milestone 4).

Loads the reviews produced by the ingestion pipeline (`rmp_reviews_raw.json`),
token-splits them, embeds each chunk locally with all-MiniLM-L6-v2 (via
sentence-transformers, no API key / no rate limits), stores the chunks in a
persistent ChromaDB collection with source metadata, and exposes a `retrieve()`
function for cosine-similarity search.

Run directly to (re)build the index and print a sample retrieval:

    python embed_index.py

Or import the pieces:

    from embed_index import build_index, retrieve
    build_index()
    hits = retrieve("which professor is most lenient with late work?", top_k=40)
"""

from __future__ import annotations

import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

# --- Configuration (from planning.md + diagram.JPG) ----------------------------
DEFAULT_REVIEWS = "rmp_reviews_raw.json"   # output of the ingestion pipeline
CHROMA_DIR = "chroma_db"                    # persistent vector store (gitignored)
COLLECTION = "professor_reviews"
MODEL_NAME = "all-MiniLM-L6-v2"             # local embedding model, no API key
CHUNK_SIZE = 250                             # tokens, per planning.md
OVERLAP_RATIO = 0.18                         # 18% overlap, per planning.md
TOP_K = 40                                   # chunks retrieved per query, per planning.md

# Lazy singleton so the model is loaded from disk only once per process.
_MODEL: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Load (and cache) the sentence-transformers embedding model.

    The first call downloads ~80 MB from the HuggingFace hub; subsequent calls
    and runs reuse the local cache. Runs entirely locally — no API key required.
    """
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(MODEL_NAME)
    return _MODEL


def load_reviews(path: str | Path = DEFAULT_REVIEWS) -> list[dict]:
    """Read the JSON array of review records from the ingestion pipeline."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run the ingestion pipeline (parse_rmp_html.py / "
            f"fetch_rmp_reviews.py) first to produce it."
        )
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def chunk_text(
    text: str,
    tokenizer,
    chunk_size: int = CHUNK_SIZE,
    overlap_ratio: float = OVERLAP_RATIO,
) -> list[str]:
    """Token-aware splitter using the embedding model's own tokenizer.

    Short reviews (<= chunk_size tokens) are returned whole as a single chunk.
    Longer ones are split into overlapping windows of `chunk_size` tokens with a
    stride of `chunk_size - round(chunk_size * overlap_ratio)`.
    """
    if not text or not text.strip():
        return []

    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) <= chunk_size:
        return [text.strip()]

    overlap = round(chunk_size * overlap_ratio)   # 45 tokens for 250 / 18%
    step = chunk_size - overlap                    # 205 tokens
    chunks: list[str] = []
    for start in range(0, len(token_ids), step):
        window = token_ids[start:start + chunk_size]
        piece = tokenizer.decode(window, skip_special_tokens=True).strip()
        if piece:
            chunks.append(piece)
        if start + chunk_size >= len(token_ids):
            break
    return chunks


def build_metadata(review: dict, source: str, chunk_index: int, n_chunks: int) -> dict:
    """Build a ChromaDB-safe metadata dict for one chunk.

    ChromaDB metadata values must be str/int/float/bool and non-None, so None
    values are dropped and a few key text fields fall back to "unknown".
    """
    metadata = {
        "professor_name": review.get("professor_name") or "unknown",
        "course": review.get("course") or "unknown",
        "rating_overall": review.get("rating_overall"),
        "difficulty": review.get("difficulty"),
        "date": review.get("date") or "unknown",
        "school_name": review.get("school_name") or "unknown",
        "source": source,
        "chunk_index": chunk_index,
        "n_chunks": n_chunks,
    }
    # source_url is currently null in the data, but carry it through if present.
    if review.get("source_url"):
        metadata["source_url"] = review["source_url"]
    # Drop any None values — ChromaDB rejects them.
    return {k: v for k, v in metadata.items() if v is not None}


def build_index(
    reviews_path: str | Path = DEFAULT_REVIEWS,
    persist_dir: str = CHROMA_DIR,
    collection_name: str = COLLECTION,
    rebuild: bool = True,
):
    """Chunk, embed, and store all reviews in a persistent ChromaDB collection.

    Returns the populated collection.
    """
    model = get_model()
    tokenizer = model.tokenizer
    reviews = load_reviews(reviews_path)

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []

    for i, review in enumerate(reviews):
        pieces = chunk_text(review.get("review_text", ""), tokenizer)
        # A review with no professor_name comes from the Reddit thread.
        source = "RateMyProfessors" if review.get("professor_name") else "Reddit (r/uhd)"
        for j, piece in enumerate(pieces):
            ids.append(f"review{i}_chunk{j}")
            documents.append(piece)
            metadatas.append(build_metadata(review, source, j, len(pieces)))

    if not documents:
        raise ValueError("No chunks produced — check that reviews contain review_text.")

    print(f"{len(reviews)} reviews -> {len(documents)} chunks. Embedding with {MODEL_NAME} ...")
    embeddings = model.encode(
        documents,
        batch_size=32,
        normalize_embeddings=True,   # unit vectors -> clean cosine scores
        show_progress_bar=True,
    )

    client = chromadb.PersistentClient(path=persist_dir)
    if rebuild:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass  # collection didn't exist yet
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},   # cosine similarity, per diagram
    )
    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
    )
    print(f"Stored {collection.count()} chunks in ChromaDB collection "
          f"'{collection_name}' at ./{persist_dir}")
    return collection


def get_collection(persist_dir: str = CHROMA_DIR, collection_name: str = COLLECTION):
    """Open an existing persistent collection (raises if it hasn't been built)."""
    client = chromadb.PersistentClient(path=persist_dir)
    return client.get_collection(collection_name)


def retrieve(
    query: str,
    top_k: int = TOP_K,
    persist_dir: str = CHROMA_DIR,
    collection_name: str = COLLECTION,
) -> list[dict]:
    """Embed `query` and return the top-k most similar chunks.

    Each hit is a dict: {"id", "text", "metadata", "distance", "score"} where
    score = 1 - cosine_distance (higher is more similar).
    """
    model = get_model()
    collection = get_collection(persist_dir, collection_name)

    query_embedding = model.encode([query], normalize_embeddings=True)
    n_results = min(top_k, collection.count())
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=n_results,
    )

    hits: list[dict] = []
    for id_, doc, meta, dist in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append({
            "id": id_,
            "text": doc,
            "metadata": meta,
            "distance": dist,
            "score": 1 - dist,
        })
    return hits


if __name__ == "__main__":
    build_index()

    sample_query = "Which professor is most lenient with late assignment due dates?"
    print(f"\nSample retrieval for: {sample_query!r}\n" + "-" * 60)
    for rank, hit in enumerate(retrieve(sample_query, top_k=5), start=1):
        meta = hit["metadata"]
        print(f"{rank}. [{hit['score']:.3f}] {meta.get('professor_name')} "
              f"({meta.get('course')}, rating {meta.get('rating_overall')}, "
              f"difficulty {meta.get('difficulty')})")
        print(f"   {hit['text'][:160]}{'...' if len(hit['text']) > 160 else ''}")
