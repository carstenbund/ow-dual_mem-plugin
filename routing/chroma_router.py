from __future__ import annotations
from typing import List, Tuple, Optional, Dict
import threading, numpy as np
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

class Embedder:
    def __init__(self):
        self.st = None
        try:
            from sentence_transformers import SentenceTransformer
            self.st = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            self.st = None

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        if self.st is not None:
            vecs = self.st.encode(texts, normalize_embeddings=True)
            return np.asarray(vecs, dtype=np.float32)
        # fallback: trivial tf-idf-like hashing (compact 384-d)
        import hashlib, math
        D = 384
        out = np.zeros((len(texts), D), dtype=np.float32)
        for i, t in enumerate(texts):
            toks = set(t.lower().split())
            for tok in toks:
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16) % D
                out[i, h] += 1.0
        return _l2_norm(out)

class ChromaRouter:
    def __init__(self, persist_dir="data/chroma", collection_name="motifs__public", metric="cosine"):
        self.emb = Embedder()
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.metric = metric

        self._lock = threading.RLock()
        self._client = chromadb.PersistentClient(path=persist_dir, settings=Settings(allow_reset=False))
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
