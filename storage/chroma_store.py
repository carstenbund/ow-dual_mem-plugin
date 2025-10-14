from __future__ import annotations
from typing import List, Optional, Dict, Tuple
import logging
import threading
from collections import defaultdict, deque
import time, uuid, chromadb
from chromadb.config import Settings
from routing.chroma_router import ChromaRouter, Motif
from policy.runtime_policy import Policy

log = logging.getLogger(__name__)

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
        self._router_lock = threading.RLock()
        self._routers: Dict[Tuple[str, str], ChromaRouter] = {}
        self._metrics_lock = threading.Lock()
        self._metrics: Dict[str, Dict[str, float]] = defaultdict(lambda: {"count": 0, "errors": 0, "latency_ms": 0.0})
        self._link_windows: Dict[Tuple[str, str], deque[float]] = defaultdict(deque)
        self._link_window_seconds = 60.0
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
    def _observe(self, name: str, start: float, error: bool = False) -> float:
        elapsed = (time.perf_counter() - start) * 1000.0
        with self._metrics_lock:
            stats = self._metrics[name]
            stats["count"] += 1
            if error:
                stats["errors"] += 1
            stats["latency_ms"] += elapsed
        return elapsed

    def metrics_snapshot(self) -> Dict[str, Dict[str, float]]:
        with self._metrics_lock:
            return {k: dict(v) for k, v in self._metrics.items()}

    def _router_key(self, layer: str, user_id: Optional[str]) -> Tuple[str, str]:
        return layer, (user_id or "__shared__")

    def router(self, layer: str, user_id: Optional[str]) -> ChromaRouter:
        key = self._router_key(layer, user_id)
        with self._router_lock:
            router = self._routers.get(key)
            if router is None:
                coll = self._motif_coll_name(layer, user_id)
                router = ChromaRouter(
                    persist_dir=self.persist_dir,
                    collection_name=coll,
                    embedding_model=self.embedding_model,
                    allow_hash_fallback=self.allow_hash_fallback,
                )
                router.rebuild_cache()
                self._routers[key] = router
        return router

    def search_text(self, layer: str, user_id: Optional[str], text: str, top_k: int = 5):
        start = time.perf_counter()
        error = False
        try:
            router = self.router(layer, user_id)
            return router.search_text(text, top_k=top_k)
        except Exception as exc:  # pragma: no cover - unexpected
            error = True
            log.exception("Search failed for layer=%s user_id=%s: %s", layer, user_id, exc)
            raise
        finally:
            elapsed = self._observe("search_text", start, error)
            log.debug("memory.search_text layer=%s user=%s top_k=%s took %.2fms (error=%s)", layer, user_id, top_k, elapsed, error)

    # ---- CRUD: motifs ----
    def list_motifs(self, layer: str, user_id: Optional[str]):
        start = time.perf_counter()
        error = False
        try:
            r = self.router(layer, user_id)
            return [{"id": m.id, "content": m.content, "symbols": m.symbols, "metadata": m.metadata} for m in r._cache.values()]
        except Exception as exc:  # pragma: no cover - unexpected
            error = True
            log.exception("Failed to list motifs for %s/%s: %s", layer, user_id, exc)
            raise
        finally:
            elapsed = self._observe("list_motifs", start, error)
            log.debug("memory.list_motifs layer=%s user=%s took %.2fms (error=%s)", layer, user_id, elapsed, error)

    def upsert_motifs(self, layer: str, user_id: Optional[str], motifs: List[Motif]):
        start = time.perf_counter()
        error = False
        try:
            r = self.router(layer, user_id)
            r.add_many(motifs)
        except Exception as exc:  # pragma: no cover - unexpected
            error = True
            log.exception("Failed to upsert motifs for %s/%s: %s", layer, user_id, exc)
            raise
        finally:
            elapsed = self._observe("upsert_motifs", start, error)
            log.debug("memory.upsert_motifs layer=%s user=%s count=%s took %.2fms (error=%s)", layer, user_id, len(motifs), elapsed, error)

    def delete_motif(self, layer: str, user_id: Optional[str], motif_id: str):
        start = time.perf_counter()
        error = False
        try:
            self.router(layer, user_id).delete(motif_id)
        except Exception as exc:  # pragma: no cover - unexpected
            error = True
            log.exception("Failed to delete motif %s for %s/%s: %s", motif_id, layer, user_id, exc)
            raise
        finally:
            elapsed = self._observe("delete_motif", start, error)
            log.debug("memory.delete_motif layer=%s user=%s took %.2fms (error=%s)", layer, user_id, elapsed, error)

    # ---- CRUD: edges ----
    def _consume_link_budget(self, layer: str, user_id: Optional[str], policy: Policy) -> bool:
        limit = getattr(policy, "max_links_per_minute", 0)
        if limit <= 0:
            return True
        key = self._router_key(layer, user_id)
        window = self._link_windows[key]
        now = time.time()
        cutoff = now - self._link_window_seconds
        while window and window[0] < cutoff:
            window.popleft()
        if len(window) >= limit:
            return False
        window.append(now)
        return True

    def link(self, layer: str, user_id: Optional[str], src_id: str, dst_id: str, rel: str = "linked") -> None:
        start = time.perf_counter()
        error = False
        try:
            edges = self._get_or_create(self._edge_coll_name(layer, user_id))
            eid = f"{src_id}__{rel}__{dst_id}"
            edges.upsert(ids=[eid], documents=[""], metadatas=[{"src": src_id, "dst": dst_id, "rel": rel, "ts": time.time()}])
        except Exception as exc:  # pragma: no cover - unexpected
            error = True
            log.exception("Failed to link %s -> %s for %s/%s: %s", src_id, dst_id, layer, user_id, exc)
            raise
        finally:
            elapsed = self._observe("link", start, error)
            log.debug("memory.link layer=%s user=%s took %.2fms (error=%s)", layer, user_id, elapsed, error)

    def links_for(self, layer: str, user_id: Optional[str], motif_id: str) -> List[Dict]:
        start = time.perf_counter()
        error = False
        try:
            edges = self._get_or_create(self._edge_coll_name(layer, user_id))
            res = edges.get(include=["ids","metadatas"], where={"src": motif_id})
            out = []
            for eid, meta in zip(res.get("ids", []), res.get("metadatas", [])):
                out.append({"id": eid, **(meta or {})})
            return out
        except Exception as exc:  # pragma: no cover - unexpected
            error = True
            log.exception("Failed to list links for motif %s in %s/%s: %s", motif_id, layer, user_id, exc)
            raise
        finally:
            elapsed = self._observe("links_for", start, error)
            log.debug("memory.links_for layer=%s user=%s took %.2fms (error=%s)", layer, user_id, elapsed, error)

    def wipe_namespace(self, layer: str, user_id: Optional[str]):
        start = time.perf_counter()
        error = False
        try:
            for name in (self._motif_coll_name(layer, user_id), self._edge_coll_name(layer, user_id)):
                try:
                    self.client.delete_collection(name)
                except Exception:
                    pass
            key = self._router_key(layer, user_id)
            with self._router_lock:
                self._routers.pop(key, None)
            self._link_windows.pop(key, None)
        except Exception as exc:  # pragma: no cover - unexpected
            error = True
            log.exception("Failed to wipe namespace %s/%s: %s", layer, user_id, exc)
            raise
        finally:
            elapsed = self._observe("wipe_namespace", start, error)
            log.debug("memory.wipe_namespace layer=%s user=%s took %.2fms (error=%s)", layer, user_id, elapsed, error)

    # ---- Policy-gated persist with novelty + thresholded links ----
    def _link_with_policy(self, layer: str, user_id: Optional[str], r: ChromaRouter, m: Motif, policy: Policy) -> int:
        proposals = r.suggest_for_motif(m.id, top_k=8)
        linked = 0
        for cand, score in proposals:
            if cand.id == m.id:
                continue
            jacc = _jaccard(m.symbols, cand.symbols)
            if not policy.should_link(score, jacc, linked):
                continue
            if not self._consume_link_budget(layer, user_id, policy):
                log.debug(
                    "memory.link budget exhausted for layer=%s user=%s; skipping remaining proposals", layer, user_id
                )
                break
            self.link(layer, user_id, m.id, cand.id, rel="linked")
            linked += 1
        return linked

    def persist_units(self, layer: str, user_id: Optional[str], units: List[Dict], policy: Policy) -> List[str]:
        if not units:
            return []
        start = time.perf_counter()
        error = False
        accepted_count = 0
        try:
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

            accepted_count = len(accepted)
            return ids
        except Exception as exc:
            error = True
            log.exception("Failed to persist motifs for %s/%s: %s", layer, user_id, exc)
            raise
        finally:
            elapsed = self._observe("persist_units", start, error)
            log.debug("memory.persist_units layer=%s user=%s accepted=%s took %.2fms (error=%s)", layer, user_id, accepted_count, elapsed, error)
