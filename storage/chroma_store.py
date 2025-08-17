from __future__ import annotations
from typing import List, Optional, Dict
import time, uuid, chromadb
from chromadb.config import Settings
from routing.chroma_router import ChromaRouter, Motif
from policy.runtime_policy import Policy

def _jaccard(a: list[str], b: list[str]) -> float:
    A, B = set(a), set(b)
    return len(A & B) / max(1, len(A | B))

def _novelty_stub(m: Motif) -> Dict[str, float]:
    # Minimal novelty proxy; replace with your real novelty_index if desired.
    structural = 0.8  # pretend you measured structure difference
    return {"semantic": 0.0, "symbolic": 0.0, "structural": structural, "novelty_index": structural * 0.3}

class ChromaStore:
    def __init__(self, persist_dir="data/chroma", public_name="motifs__public", personal_prefix="motifs__personal__"):
        self.persist_dir = persist_dir
        self.public_name = public_name
        self.personal_prefix = personal_prefix
        self.client = chromadb.PersistentClient(path=persist_dir, settings=Settings(allow_reset=False))

    def _motif_coll_name(self, layer: str, user_id: Optional[str]) -> str:
        return self.public_name if layer == "public" else f"{self.personal_prefix}{user_id or 'anon'}"

    def router(self, layer: str, user_id: Optional[str]) -> ChromaRouter:
        coll = self._motif_coll_name(layer, user_id)
        r = ChromaRouter(persist_dir=self.persist_dir, collection_name=coll)
        r.rebuild_cache()
        return r

    # ----- CRUD -----

    def list_motifs(self, layer: str, user_id: Optional[str]):
        r = self.router(layer, user_id)
        return [{"id": m.id, "content": m.content, "symbols": m.symbols, "metadata": m.metadata} for m in r._cache.values()]

    def upsert_motifs(self, layer: str, user_id: Optional[str], motifs: List[Motif]):
        r = self.router(layer, user_id)
        r.add_many(motifs)

    def delete_motif(self, layer: str, user_id: Optional[str], motif_id: str):
        self.router(layer, user_id).delete(motif_id)

    def link(self, layer: str, user_id: Optional[str], src_id: str, dst_id: str, rel: str = "linked"):
        # Optional: keep edges in a separate collection; for MVP we skip persistent edges.
        pass

    def links_for(self, layer: str, user_id: Optional[str], motif_id: str):
        return []

    def wipe_namespace(self, layer: str, user_id: Optional[str]):
        name = self._motif_coll_name(layer, user_id)
        try:
            self.client.delete_collection(name)
        except Exception:
            pass

    # ----- Policy-gated persist for extracted units -----

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

            # dedup
            hits = r.search_text(m.content, top_k=1)
            top1 = float(hits[0][1]) if hits else 0.0
            if policy.is_duplicate(top1):
                # merge symbols into nearest motif
                if hits:
                    nearest = hits[0][0]
                    nearest.symbols = sorted(set(nearest.symbols) | set(m.symbols))
                    r.add_one(nearest)  # update symbols
                continue

            # novelty
            n = _novelty_stub(m)
            if not policy.should_persist(n["novelty_index"]):
                continue

            accepted.append(m)
            ids.append(mid)

        if accepted:
            r.add_many(accepted)

        return ids
