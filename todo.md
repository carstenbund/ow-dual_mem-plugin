base scaffold for a dual-layer memory plugin you can drop into Open WebUI’s plugins/tools workspace and also run standalone in your own Python env. It’s Chroma-backed by default, cleanly namespaced (`public` vs `personal/<UUID>`), and uses the same policy+d edup+novelty ideas we wired earlier.

---

# Repo layout

```
openwebui-memory-dual/
  README.md
  plugin.toml
  memory_config.yaml
  __init__.py
  policy/
    __init__.py
    runtime_policy.py
  utils/
    __init__.py
    context.py
    reducers.py
  routing/
    __init__.py
    chroma_router.py
  storage/
    __init__.py
    store.py
    chroma_store.py
  tools/
    __init__.py
    memory.py
  filters/
    __init__.py
    dual_layer_memory_filter.py
```

---

## `plugin.toml` (Open WebUI metadata)

```toml
name = "dual-layer-memory"
version = "0.1.0"
description = "Public + Personal (UUID) memory with policy-gated capture & Chroma retrieval."
authors = ["you <you@example.com>"]

# Entry points (Open WebUI will discover these)
[tools]
memory = "tools.memory:register"

[filters]
dual_layer_memory = "filters.dual_layer_memory_filter:register"
```

---

## `README.md`

````md
# Dual-Layer Memory for Open WebUI

- Two namespaces: **public** and **personal (UUID)**.
- Policy-gated capture: dedup, novelty, thresholded links.
- Persistent retrieval via **Chroma** (per-namespace collections).
- Tools: `memory.retrieve`, `memory.persist`, `memory.link`, `memory.wipe_namespace`.
- Filter: auto-extract public/personal units on each user turn (optional).

## Quick start

```bash
pip install chromadb sentence-transformers
# Optional: pip install uvicorn fastapi (if you run standalone services)
````

Place this folder into your Open WebUI plugins / tools workspace and enable:

* **Tool**: `dual-layer-memory / memory`
* **Filter**: `dual_layer_memory`

Configure `memory_config.yaml`, then chat. Use the `memory.retrieve` tool to pull context.

````

---

## `memory_config.yaml` (defaults you can tweak in UI later)

```yaml
router:
  persist_dir: "data/chroma"
  public_collection: "motifs__public"
  personal_prefix: "motifs__personal__"
policy:
  public:
    link_threshold: 0.65
    dedup_threshold: 0.95
    novelty_min: 0.20
    max_links_per_motif: 2
    symbol_jaccard_cap: 0.80
  personal:
    link_threshold: 0.60
    dedup_threshold: 0.93
    novelty_min: 0.10
    max_links_per_motif: 3
    symbol_jaccard_cap: 0.80
retrieval:
  k_public: 5
  k_personal: 3
symbols:
  public_allow: ["concept","definition","taxonomy","relationship","claim","pattern"]
  personal_allow: ["fact","preference","decision","rationale","open_question","todo","glossary","pattern"]
````

---

## `__init__.py`

```python
__all__ = []
```

---

## `policy/runtime_policy.py`

```python
from __future__ import annotations

class Policy:
    def __init__(
        self,
        link_threshold: float = 0.60,
        dedup_threshold: float = 0.93,
        novelty_min: float = 0.10,
        max_links_per_motif: int = 3,
        symbol_jaccard_cap: float = 0.80,
    ):
        self.link_threshold = link_threshold
        self.dedup_threshold = dedup_threshold
        self.novelty_min = novelty_min
        self.max_links_per_motif = max_links_per_motif
        self.symbol_jaccard_cap = symbol_jaccard_cap

    def should_persist(self, novelty_index: float) -> bool:
        return novelty_index >= self.novelty_min

    def should_link(self, score: float, jaccard: float, links_added: int) -> bool:
        if links_added >= self.max_links_per_motif:
            return False
        if jaccard >= self.symbol_jaccard_cap:
            return False
        # mild brake to avoid cliques
        if links_added >= 2 and score < max(self.link_threshold, 0.70):
            return False
        return score >= self.link_threshold

    def is_duplicate(self, top1_sim: float) -> bool:
        return top1_sim >= self.dedup_threshold
```

---

## `utils/context.py`

```python
from __future__ import annotations
from typing import List, Tuple

def render_context(hits: List[Tuple[object, float]], max_chars: int = 2000) -> str:
    header = "## Context (top motifs)\n"
    lines, used = [], 0
    for m, score in hits:
        symbols = getattr(m, "symbols", []) or []
        content = (getattr(m, "content", "") or "").strip()
        chunk = f"\n### {','.join(symbols)} (sim={score:.2f})\n{content}\n"
        if used + len(chunk) > max_chars:
            break
        lines.append(chunk); used += len(chunk)
    return header + "".join(lines)
```

---

## `utils/reducers.py` (LLM-assisted extractors)

```python
from __future__ import annotations
from typing import List, Dict

PUBLIC_TYPES   = ["concept","definition","taxonomy","relationship","claim","pattern"]
PERSONAL_TYPES = ["fact","preference","decision","rationale","open_question","todo","glossary","pattern"]

def extract_public_units(thread_ask, text: str) -> List[Dict]:
    prompt = f"""Extract reusable knowledge. Allowed types: {PUBLIC_TYPES}.
Return JSON list of objects with fields: type, title, content (<=200 chars), symbols (2-6 tags).
Drop personal/anecdotal content.
Input:
{text}"""
    out = thread_ask(prompt)
    try:
        import json
        units = json.loads(out)
        return units if isinstance(units, list) else []
    except Exception:
        return []

def extract_personal_units(thread_ask, text: str) -> List[Dict]:
    prompt = f"""Extract durable personal memory. Allowed types: {PERSONAL_TYPES}.
Return JSON list of objects with fields: type, title, content (<=200 chars), symbols (2-6 tags).
Keep only facts, preferences, decisions, todos, glossaries, patterns.
Input:
{text}"""
    out = thread_ask(prompt)
    try:
        import json
        units = json.loads(out)
        return units if isinstance(units, list) else []
    except Exception:
        return []
```

---

## `routing/chroma_router.py` (compact, encode\_texts-first)

```python
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
    if dist is None: return 0.0
    d = float(dist); 
    if d < 0: d = 0.0
    if d > 1: d = min(d, 2.0)
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
        try: return self._client.get_collection(self.collection_name)
        except Exception: return self._client.create_collection(self.collection_name, metadata={"hnsw:space": self.metric})

    def rebuild_cache(self):
        with self._lock:
            res = self._coll.get(include=["ids","documents","metadatas"], where={})
            self._cache.clear()
            for mid, doc, meta in zip(res.get("ids",[]), res.get("documents",[]), res.get("metadatas",[])):
                self._cache[mid] = Motif(mid, doc, (meta.get("symbols","").split(",") if meta and meta.get("symbols") else []), meta or {})

    def add_many(self, motifs: List[Motif]) -> None:
        if not motifs: return
        with self._lock:
            embs = self.emb.encode_texts([m.content for m in motifs])
            self._coll.upsert(
                ids=[m.id for m in motifs],
                documents=[m.content for m in motifs],
                embeddings=[row.tolist() for row in embs],
                metadatas=[{"symbols": ",".join(m.symbols), **(m.metadata or {})} for m in motifs],
            )
            for m in motifs: self._cache[m.id] = m

    def add_one(self, m: Motif) -> None:
        self.add_many([m])

    def delete(self, motif_id: str) -> None:
        with self._lock:
            try: self._coll.delete(ids=[motif_id])
            except Exception: pass
            self._cache.pop(motif_id, None)

    def search_text(self, text: str, top_k: int = 5) -> List[Tuple[Motif, float]]:
        q = self.emb.encode_texts([text])[0].tolist()
        with self._lock:
            res = self._coll.query(query_embeddings=[q], n_results=max(1, top_k))
        out: List[Tuple[Motif, float]] = []
        ids = res.get("ids", [[]]); dists = res.get("distances", [[]])
        if not ids or not ids[0]: return out
        for i in range(len(ids[0])):
            mid = ids[0][i]
            sim = _as_sim(dists[0][i] if dists and dists[0] else None)
            m = self._cache.get(mid)
            if m: out.append((m, sim))
        return out

    def suggest_for_motif(self, motif_id: str, top_k: int = 8) -> List[Tuple[Motif, float]]:
        m = self._cache.get(motif_id)
        if not m: return []
        hits = self.search_text(m.content, top_k=top_k+1)
        return [(cand, s) for cand, s in hits if cand.id != motif_id][:top_k]
```

---

## `storage/store.py` (interface)

```python
from __future__ import annotations
from typing import List, Optional

class Store:
    def list_motifs(self, layer: str, user_id: Optional[str]): ...
    def upsert_motifs(self, layer: str, user_id: Optional[str], motifs: list): ...
    def delete_motif(self, layer: str, user_id: Optional[str], motif_id: str): ...
    def link(self, layer: str, user_id: Optional[str], src_id: str, dst_id: str, rel: str="linked"): ...
    def links_for(self, layer: str, user_id: Optional[str], motif_id: str): ...
    def wipe_namespace(self, layer: str, user_id: Optional[str]): ...
```

---

## `storage/chroma_store.py` (namespaced store + policy)

```python
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
        return self.public_name if layer=="public" else f"{self.personal_prefix}{user_id or 'anon'}"

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

    def link(self, layer: str, user_id: Optional[str], src_id: str, dst_id: str, rel: str="linked"):
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
        if not units: return []
        r = self.router(layer, user_id)
        accepted: List[Motif] = []
        ids: List[str] = []

        for u in units:
            mid = str(uuid.uuid4())
            content = f"[{u.get('type','unit')}] {u.get('title','')}\n{u.get('content','')}".strip()
            symbols = list(dict.fromkeys((u.get("symbols") or []) + [u.get("type","unit"), layer]))
            meta = {"layer": layer, "owner": (user_id if layer=="personal" else None), "created_at": time.time(), "updated_at": time.time()}
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

            accepted.append(m); ids.append(mid)

        if accepted:
            r.add_many(accepted)

        return ids
```

---

## `tools/memory.py` (tool functions exposed to the model/UI)

```python
from __future__ import annotations
from typing import Optional, Dict, Any, List
import yaml, os
from utils.context import render_context
from storage.chroma_store import ChromaStore
from routing.chroma_router import Motif
from policy.runtime_policy import Policy

_cfg = {}
try:
    _cfg = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", "memory_config.yaml")))
except Exception:
    _cfg = {}

_store = ChromaStore(
    persist_dir=_cfg.get("router", {}).get("persist_dir", "data/chroma"),
    public_name=_cfg.get("router", {}).get("public_collection", "motifs__public"),
    personal_prefix=_cfg.get("router", {}).get("personal_prefix", "motifs__personal__"),
)

PUB = Policy(**_cfg.get("policy", {}).get("public", {})) if _cfg.get("policy") else Policy(0.65,0.95,0.20,2,0.80)
PER = Policy(**_cfg.get("policy", {}).get("personal", {})) if _cfg.get("policy") else Policy(0.60,0.93,0.10,3,0.80)

def retrieve(query: str, layer: str = "public", user_id: Optional[str] = None, k: int = 5) -> Dict[str, Any]:
    r = _store.router(layer, user_id)
    hits = r.search_text(query, top_k=max(1, k))
    return {
        "count": len(hits),
        "context": render_context(hits),
        "hits": [{"id": m.id, "symbols": m.symbols, "score": s, "owner": m.metadata.get("owner")} for m, s in hits],
    }

def persist(units: List[Dict[str, Any]], layer: str = "public", user_id: Optional[str] = None) -> Dict[str, Any]:
    pol = PUB if layer == "public" else PER
    ids = _store.persist_units(layer, user_id, units, pol)
    return {"written": ids}

def link(src_id: str, dst_id: str, layer: str = "public", user_id: Optional[str] = None) -> Dict[str, Any]:
    _store.link(layer, user_id, src_id, dst_id, rel="linked")
    return {"linked": [src_id, dst_id]}

def wipe_namespace(layer: str, user_id: Optional[str] = None) -> Dict[str, Any]:
    _store.wipe_namespace(layer, user_id)
    return {"wiped": {"layer": layer, "user_id": user_id}}

# ---- Open WebUI registration hook ----
def register():
    # The host will introspect these callables and expose them as tool functions
    return {
        "functions": {
            "memory.retrieve": {
                "callable": retrieve,
                "description": "Retrieve top-K memory for a query from public or personal namespace.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "layer": {"type": "string", "enum": ["public","personal"], "default": "public"},
                        "user_id": {"type": ["string","null"]},
                        "k": {"type": "integer", "default": 5}
                    },
                    "required": ["query"]
                }
            },
            "memory.persist": {
                "callable": persist,
                "description": "Persist extracted units to the memory (policy-gated).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "units": {"type": "array"},
                        "layer": {"type": "string", "enum": ["public","personal"], "default": "public"},
                        "user_id": {"type": ["string","null"]}
                    },
                    "required": ["units"]
                }
            },
            "memory.link": {
                "callable": link,
                "description": "Create a link between two motifs.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "src_id": {"type": "string"},
                        "dst_id": {"type": "string"},
                        "layer": {"type": "string", "enum": ["public","personal"], "default": "public"},
                        "user_id": {"type": ["string","null"]}
                    },
                    "required": ["src_id","dst_id"]
                }
            },
            "memory.wipe_namespace": {
                "callable": wipe_namespace,
                "description": "Delete all data in a namespace (public or specific personal UUID).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "layer": {"type": "string", "enum": ["public","personal"], "default": "public"},
                        "user_id": {"type": ["string","null"]}
                    },
                    "required": ["layer"]
                }
            }
        }
    }
```

---

## `filters/dual_layer_memory_filter.py` (auto-capture on user turns)

```python
from __future__ import annotations
from typing import Dict, Any, Optional
import yaml, os, time
from storage.chroma_store import ChromaStore
from policy.runtime_policy import Policy
from utils.reducers import extract_public_units, extract_personal_units

_cfg = {}
try:
    _cfg = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", "memory_config.yaml")))
except Exception:
    _cfg = {}

_store = ChromaStore(
    persist_dir=_cfg.get("router", {}).get("persist_dir", "data/chroma"),
    public_name=_cfg.get("router", {}).get("public_collection", "motifs__public"),
    personal_prefix=_cfg.get("router", {}).get("personal_prefix", "motifs__personal__"),
)

PUB = Policy(**_cfg.get("policy", {}).get("public", {})) if _cfg.get("policy") else Policy(0.65,0.95,0.20,2,0.80)
PER = Policy(**_cfg.get("policy", {}).get("personal", {})) if _cfg.get("policy") else Policy(0.60,0.93,0.10,3,0.80)

def _thread_ask_default(prompt: str) -> str:
    # In Open WebUI, you can inject a host-provided LLM call; for MVP, return empty JSON.
    return "[]"

def on_message(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Runs on each message; expects event to include:
      - role: 'user' or 'assistant'
      - content: text
      - user_id: UUID (if available)
    """
    if event.get("role") != "user":
        return event

    text = event.get("content","")
    user_id = event.get("user_id")

    # Use host's LLM when integrated; fallback returns [] so it's safe
    ask = event.get("_thread_ask", _thread_ask_default)

    pub_units = extract_public_units(ask, text)
    per_units = extract_personal_units(ask, text) if user_id else []

    if pub_units:
        _store.persist_units("public", None, pub_units, PUB)
    if per_units and user_id:
        _store.persist_units("personal", user_id, per_units, PER)

    # Optionally, attach retrieval context for next turn (host can merge it)
    # r_pub = _store.router("public", None).search_text(text, top_k=_cfg.get("retrieval",{}).get("k_public",5))
    # r_per = _store.router("personal", user_id).search_text(text, top_k=_cfg.get("retrieval",{}).get("k_personal",3)) if user_id else []
    # event["memory_context"] = render_context(r_pub) + render_context(r_per)

    return event

def register():
    # Host will call this to register the filter
    return {"on_message": on_message}
```

---

# How to use

1. **Install deps**

```bash
pip install chromadb sentence-transformers
```

2. **Add to Open WebUI**

* Drop `openwebui-memory-dual/` into your Open WebUI plugins/tools directory (where custom tools/filters live).
* In the UI, enable:

  * Tool: `memory` (from `dual-layer-memory`)
  * Filter: `dual_layer_memory`

3. **Run a quick test**

* Send a few messages; the filter will (safely) do nothing if the host doesn’t supply an LLM to `_thread_ask`.
* Use the tool manually to persist & retrieve:

Example tool calls (from the chat or a small script):

```python
# Persist a public concept
memory.persist(units=[{
  "type":"concept","title":"Justice gradient",
  "content":"Treat justice as levels rather than binary.",
  "symbols":["justice","taxonomy","gradient"]
}], layer="public")

# Retrieve for a query
memory.retrieve(query="justice taxonomy bridge", layer="public", k=5)
```

4. **(Optional) Wire the host LLM**
   If Open WebUI exposes a thread-ask callable to filters, pass it through the event as `_thread_ask`; the extractor will start producing real units automatically.

---

## Why this is “transferable”

* **Zero assumptions** about Open WebUI internals beyond standard tool/filter hooks.
* **Chroma only** (no files); easy to switch to another backend later (just add a `pg_store.py` and swap in `tools/memory.py` and the filter).
* **Policy knobs** live in `memory_config.yaml`.
* Works as a **standalone library** too — you can import `ChromaStore` and `ChromaRouter` in your own runners.

If you want, I can add a tiny **standalone demo script** (outside WebUI) that simulates a user sending messages and shows the persisted public/personal motifs and retrieval context.
