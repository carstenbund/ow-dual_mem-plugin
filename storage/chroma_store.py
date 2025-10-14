from __future__ import annotations
from typing import List, Optional, Dict, Tuple
import time, uuid, chromadb
from chromadb.config import Settings
from routing.chroma_router import ChromaRouter, Motif
from policy.runtime_policy import Policy

# ----------------- small helpers -----------------

def _jaccard(a: list[str], b: list[str]) -> float:
    A, B = set(a), set(b)
    return len(A & B) / max(1, len(A | B))

def _clamp_small(x: float, eps: float = 1e-9) -> float:
    return 0.0 if abs(x) < eps else float(x)

def _novelty_scores(router: ChromaRouter, m: Motif, k: int = 5) -> Dict[str, float]:
    """
    Quick, production-ish novelty:
      semantic  = 1 - top1_sim                          (higher = more novel)
      symbolic  = 1 - max_jaccard(symbols)              (higher = more novel)
      structural= 1 - mean(sim over top-k hits)         (higher = more novel)
    Combined   = 0.5*semantic + 0.3*symbolic + 0.2*structural
    """
    hits = router.search_text(m.content, top_k=max(1, k))
    if not hits:
        return {"semantic": 1.0, "symbolic": 1.0, "structural": 1.0, "novelty_index": 1.0}

    sims = [s for _, s in hits]
    top1 = sims[0]
    mean_k = sum(sims) / len(sims)

    # max symbol overlap among the neighbors we saw
    max_j = 0.0
    for neigh, _ in hits:
        max_j = max(max_j, _jaccard(m.symbols, neigh.symbols))

    semantic = _clamp_small(1.0 - float(top1))
    symbolic = _clamp_small(1.0 - float(max_j))
    structural = _clamp_small(1.0 - float(mean_k))
    novelty_index = _clamp_small(0.5 * semantic + 0.3 * symbolic + 0.2 * structural)

    return {
        "semantic": semantic,
        "symbolic": symbolic,
        "structural": structural,
        "novelty_index": novelty_index,
        "top1_sim": float(top1),
    }

# ----------------- store -----------------

class ChromaStore:
    def __init__(
        self,
        persist_dir="data/chroma",
        public_name="motifs__public",
        personal_prefix="motifs__personal__",
        embedding_model: str = "all-MiniLM-L6-v2",
        allow_hash_fallback: bool = False,
    ):
        self.persist_dir = persist_dir
        self.public_name = public_name
        self.personal_prefix = personal_prefix
        self.embedding_model = embedding_model
        self.allow_hash_fallback = allow_hash_fallback
        try:
            self.client = chromadb.PersistentClient(path=persist_dir, settings=Settings(allow_reset=False))
        except Exception as exc:  # pragma: no cover - defensive
            raise RuntimeError(
                f"Failed to initialise Chroma persistence at '{persist_dir}': {exc}"
            ) from exc

    # ---- collection names ----
    def _motif_coll_name(self, layer: str, user_id: Optional[str]) -> str:
        return self.public_name if layer == "public" else f"{self.personal_prefix}{user_id or 'anon'}"

    def _edge_coll_name(self, layer: str, user_id: Optional[str]) -> str:
        base = "edges__public" if layer == "public" else f"edges__personal__{user_id or 'anon'}"
        return base

    def _get_or_create(self, name: str):
        try:
            return self.client.get_collection(name)
        except Exception:
            return self.client.create_collection(name, metadata={"hnsw:space": "cosine"})

    # ---- routers ----
    def router(self, layer: str, user_id: Optional[str]) -> ChromaRouter:
        coll = self._motif_coll_name(layer, user_id)
        r = ChromaRouter(
            persist_dir=self.persist_dir,
            collection_name=coll,
            embedding_model=self.embedding_model,
            allow_hash_fallback=self.allow_hash_fallback,
        )
        r.rebuild_cache()
        return r

    # ---- CRUD: motifs ----
    def list_motifs(self, layer: str, user_id: Optional[str]):
        r = self.router(layer, user_id)
        return [{"id": m.id, "content": m.content, "symbols": m.symbols, "metadata": m.metadata} for m in r._cache.values()]

    def upsert_motifs(self, layer: str, user_id: Optional[str], motifs: List[Motif]):
        r = self.router(layer, user_id)
        r.add_many(motifs)

    def delete_motif(self, layer: str, user_id: Optional[str], motif_id: str):
        self.router(layer, user_id).delete(motif_id)

    # ---- CRUD: edges ----
    def link(self, layer: str, user_id: Optional[str], src_id: str, dst_id: str, rel: str = "linked") -> None:
        edges = self._get_or_create(self._edge_coll_name(layer, user_id))
        eid = f"{src_id}__{rel}__{dst_id}"
        edges.upsert(ids=[eid], documents=[""], metadatas=[{"src": src_id, "dst": dst_id, "rel": rel, "ts": time.time()}])

    def links_for(self, layer: str, user_id: Optional[str], motif_id: str) -> List[Dict]:
        edges = self._get_or_create(self._edge_coll_name(layer, user_id))
        res = edges.get(include=["ids","metadatas"], where={"src": motif_id})
        out = []
        for eid, meta in zip(res.get("ids", []), res.get("metadatas", [])):
            out.append({"id": eid, **(meta or {})})
        return out

    def wipe_namespace(self, layer: str, user_id: Optional[str]):
        for name in (self._motif_coll_name(layer, user_id), self._edge_coll_name(layer, user_id)):
            try:
                self.client.delete_collection(name)
            except Exception:
                pass

    # ---- Policy-gated persist with novelty + thresholded links ----
    def _link_with_policy(self, layer: str, user_id: Optional[str], r: ChromaRouter, m: Motif, policy: Policy) -> int:
        proposals = r.suggest_for_motif(m.id, top_k=8)
        linked = 0
        for cand, score in proposals:
            if cand.id == m.id:
                continue
            jacc = _jaccard(m.symbols, cand.symbols)
            if policy.should_link(score, jacc, linked):
                self.link(layer, user_id, m.id, cand.id, rel="linked")
                linked += 1
        return linked

    def persist_units(self, layer: str, user_id: Optional[str], units: List[Dict], policy: Policy) -> List[str]:
        if not units:
            return []
        r = self.router(layer, user_id)
        accepted: List[Motif] = []
        ids: List[str] = []

        for u in units:
            mid = str(uuid.uuid4())
            content = f"[{u.get('type','unit')}] {u.get('title','')}\n{u.get('content','')}".strip()
            symbols = list(dict.fromkeys((u.get("symbols") or []) + [u.get("type","unit"), layer]))
            meta = {"layer": layer, "owner": (user_id if layer == "personal" else None), "created_at": time.time(), "updated_at": time.time()}
            m = Motif(mid, content, symbols, meta)

            # --- dedup (nearest neighbor) ---
            hits = r.search_text(m.content, top_k=1)
            top1 = float(hits[0][1]) if hits else 0.0
            if policy.is_duplicate(top1):
                # merge symbols into nearest motif, update and skip new
                if hits:
                    nearest = hits[0][0]
                    nearest.symbols = sorted(set(nearest.symbols) | set(m.symbols))
                    r.add_one(nearest)  # update tags on existing node
                continue

            # --- novelty (real scoring) ---
            n = _novelty_scores(r, m, k=5)
            if not policy.should_persist(n["novelty_index"]):
                continue

            accepted.append(m); ids.append(mid)

        # batch add accepted
        if accepted:
            r.add_many(accepted)
            # thresholded linking (now that motifs exist)
            for m in accepted:
                self._link_with_policy(layer, user_id, r, m, policy)

        return ids
