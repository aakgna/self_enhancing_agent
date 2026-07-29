"""pgvector-backed manifest registry.

Embeddings are local (fastembed, BAAI/bge-small-en-v1.5, 384-dim) so retrieval
needs no API key. The embedder is isolated behind embed_document/embed_query;
swapping in an API embedder later is a change to those two functions plus the
column dimension.

Search is scoped to the requesting user's own manifests plus _shared ones —
per-user agents never leak across users.
"""

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from db.conn import get_conn
from orchestrator.manifest import Manifest

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384
SHARED_USER = "_shared"

# One row per embedded chunk: the description plus each example request.
# Search scores a manifest by its best-matching chunk — user requests resemble
# example requests far more than they resemble capability descriptions.
_REGISTRY_DDL = f"""
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS manifest_embeddings (
  manifest_id TEXT NOT NULL,
  chunk_no    INT  NOT NULL,
  chunk_text  TEXT NOT NULL,
  user_id     TEXT NOT NULL,
  embedding   VECTOR({EMBED_DIM}) NOT NULL,
  manifest    JSONB NOT NULL,
  PRIMARY KEY (manifest_id, chunk_no)
);
CREATE INDEX IF NOT EXISTS manifest_embeddings_user ON manifest_embeddings (user_id);
"""


@dataclass
class SearchHit:
    manifest: Manifest
    similarity: float  # cosine similarity in [0, 1]


@lru_cache(maxsize=1)
def _embedder():
    from fastembed import TextEmbedding  # deferred: first call downloads the model

    return TextEmbedding(model_name=EMBED_MODEL)


def embed_text(text: str) -> list[float]:
    return [float(x) for x in next(iter(_embedder().embed([text])))]


def init_registry() -> None:
    get_conn().execute(_REGISTRY_DDL)


def upsert(manifest: Manifest) -> None:
    conn = get_conn()
    conn.execute("DELETE FROM manifest_embeddings WHERE manifest_id = %s", (manifest.key,))
    chunks = [manifest.description, *manifest.example_requests]
    manifest_json = manifest.model_dump_json()
    for chunk_no, chunk in enumerate(chunks):
        conn.execute(
            """
            INSERT INTO manifest_embeddings
              (manifest_id, chunk_no, chunk_text, user_id, embedding, manifest)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (manifest.key, chunk_no, chunk, manifest.user_id, str(embed_text(chunk)), manifest_json),
        )


def search(
    query: str,
    user_id: str,
    top_k: int = 5,
    type_filter: str | None = None,
) -> list[SearchHit]:
    vector = embed_text(query)
    type_clause = "AND manifest->>'type' = %(type)s" if type_filter else ""
    rows = get_conn().execute(
        f"""
        SELECT manifest, max(1 - (embedding <=> %(vec)s::vector)) AS similarity
        FROM manifest_embeddings
        WHERE user_id IN (%(user)s, %(shared)s) {type_clause}
        GROUP BY manifest_id, manifest
        ORDER BY similarity DESC
        LIMIT %(k)s
        """,
        {
            "vec": str(vector),
            "user": user_id,
            "shared": SHARED_USER,
            "type": type_filter,
            "k": top_k,
        },
    ).fetchall()
    return [
        SearchHit(manifest=Manifest.model_validate(row[0]), similarity=float(row[1]))
        for row in rows
    ]


def set_flagged(manifest_key: str, flagged: bool) -> None:
    """Phase 5 uses this: visibility only, never removal (create-only lifecycle)."""
    get_conn().execute(
        """
        UPDATE manifest_embeddings
        SET manifest = jsonb_set(manifest, '{flagged_for_review}', %s::jsonb)
        WHERE manifest_id = %s
        """,
        (json.dumps(flagged), manifest_key),
    )


def backfill(repo_root: Path | None = None) -> list[str]:
    """Load every manifest.json under agents/ and tools/ into the registry."""
    root = repo_root or Path(__file__).resolve().parent.parent
    loaded = []
    for manifest_path in sorted(
        [*root.glob("agents/*/*/manifest.json"), *root.glob("tools/**/manifest.json")]
    ):
        manifest = Manifest.model_validate_json(manifest_path.read_text())
        upsert(manifest)
        loaded.append(manifest.key)
    return loaded
