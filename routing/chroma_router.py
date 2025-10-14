from __future__ import annotations
from typing import List, Tuple, Optional, Dict
import threading, numpy as np, logging, warnings
import chromadb
from chromadb.config import Settings

# Lightweight motif model used internally by store/router
class Motif:
    def __init__(self, _id: str, content: str, symbols: list[str], metadata: dict | None = None):
        self.id = _id
        self.content = content
        self.symbols = symbols
        self.metadata = metadata or {}

def _l2_norm(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
    return (mat / norms).astype(np.float32)

def _as_sim(dist: float | None) -> float:
    if dist is None:
        return 0.0
    d = float(dist)
    if d < 0:
        d = 0.0
    if d > 1:
        d = min(d, 2.0)
    return max(0.0, 1.0 - d)

log = logging.getLogger(__name__)


class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", allow_hash_fallback: bool = False):
        self.st = None
        self.model_name = model_name
        self.allow_hash_fallback = allow_hash_fallback
        self._hash_fallback = False
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:
            message = (
                "sentence-transformers is required to load embedding model "
                f"'{model_name}'. Install the dependency or enable allow_hash_fallback."
            )
            if not allow_hash_fallback:
                raise RuntimeError(message) from exc
            warnings.warn(message + " Falling back to hashing encoder.", RuntimeWarning)
            log.warning(message)
            self._hash_fallback = True
            return

        try:
            self.st = SentenceTransformer(model_name)
        except Exception as exc:
            message = f"Failed to load embedding model '{model_name}': {exc}"
            if not allow_hash_fallback:
                raise RuntimeError(message) from exc
            warnings.warn(message + " Falling back to hashing encoder.", RuntimeWarning)
            log.warning(message)
            self._hash_fallback = True

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        if self.st is not None:
            vecs = self.st.encode(texts, normalize_embeddings=True)
            return np.asarray(vecs, dtype=np.float32)
        # fallback: trivial tf-idf-like hashing (compact 384-d)
        if not self.allow_hash_fallback and not self._hash_fallback:
            raise RuntimeError(
                "Embedding model is not initialised and allow_hash_fallback is disabled."
            )
        import hashlib
        D = 384
        out = np.zeros((len(texts), D), dtype=np.float32)
        for i, t in enumerate(texts):
            toks = set(t.lower().split())
            for tok in toks:
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16) % D
                out[i, h] += 1.0
        return _l2_norm(out)

class ChromaRouter:
    def __init__(
        self,
        persist_dir="data/chroma",
        collection_name="motifs__public",
        metric="cosine",
        embedding_model: str = "all-MiniLM-L6-v2",
        allow_hash_fallback: bool = False,
    ):
        self.emb = Embedder(model_name=embedding_model, allow_hash_fallback=allow_hash_fallback)
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.metric = metric

        self._lock = threading.RLock()
        try:
            self._client = chromadb.PersistentClient(path=persist_dir, settings=Settings(allow_reset=False))
        except Exception as exc:  # pragma: no cover - defensive
            raise RuntimeError(
                f"Failed to initialise Chroma client at '{persist_dir}': {exc}"
            ) from exc
        self._coll = self._get_or_create_collection()
        self._cache: Dict[str, Motif] = {}

    def _get_or_create_collection(self):
        try:
            return self._client.get_collection(self.collection_name)
        except Exception:
            return self._client.create_collection(self.collection_name, metadata={"hnsw:space": self.metric})

    def rebuild_cache(self):
        with self._lock:
            res = self._coll.get(include=["ids","documents","metadatas"], where={})
            self._cache.clear()
            for mid, doc, meta in zip(res.get("ids", []), res.get("documents", []), res.get("metadatas", [])):
                self._cache[mid] = Motif(mid, doc, (meta.get("symbols", "").split(",") if meta and meta.get("symbols") else []), meta or {})

    def add_many(self, motifs: List[Motif]) -> None:
        if not motifs:
            return
        with self._lock:
            embs = self.emb.encode_texts([m.content for m in motifs])
            self._coll.upsert(
                ids=[m.id for m in motifs],
                documents=[m.content for m in motifs],
                embeddings=[row.tolist() for row in embs],
                metadatas=[{"symbols": ",".join(m.symbols), **(m.metadata or {})} for m in motifs],
            )
            for m in motifs:
                self._cache[m.id] = m

    def add_one(self, m: Motif) -> None:
        self.add_many([m])

    def delete(self, motif_id: str) -> None:
        with self._lock:
            try:
                self._coll.delete(ids=[motif_id])
            except Exception:
                pass
            self._cache.pop(motif_id, None)

    def search_text(self, text: str, top_k: int = 5) -> List[Tuple[Motif, float]]:
        q = self.emb.encode_texts([text])[0].tolist()
        with self._lock:
            res = self._coll.query(query_embeddings=[q], n_results=max(1, top_k))
        out: List[Tuple[Motif, float]] = []
        ids = res.get("ids", [[]])
        dists = res.get("distances", [[]])
        if not ids or not ids[0]:
            return out
        for i in range(len(ids[0])):
            mid = ids[0][i]
            sim = _as_sim(dists[0][i] if dists and dists[0] else None)
            m = self._cache.get(mid)
            if m:
                out.append((m, sim))
        return out

    def suggest_for_motif(self, motif_id: str, top_k: int = 8) -> List[Tuple[Motif, float]]:
        m = self._cache.get(motif_id)
        if not m:
            return []
        hits = self.search_text(m.content, top_k=top_k + 1)
        return [(cand, s) for cand, s in hits if cand.id != motif_id][:top_k]
