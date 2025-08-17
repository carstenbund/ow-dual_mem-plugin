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

PUB = Policy(**_cfg.get("policy", {}).get("public", {})) if _cfg.get("policy") else Policy(0.65, 0.95, 0.20, 2, 0.80)
PER = Policy(**_cfg.get("policy", {}).get("personal", {})) if _cfg.get("policy") else Policy(0.60, 0.93, 0.10, 3, 0.80)

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
                        "layer": {"type": "string", "enum": ["public", "personal"], "default": "public"},
                        "user_id": {"type": ["string", "null"]},
                        "k": {"type": "integer", "default": 5},
                    },
                    "required": ["query"],
                },
            },
            "memory.persist": {
                "callable": persist,
                "description": "Persist extracted units to the memory (policy-gated).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "units": {"type": "array"},
                        "layer": {"type": "string", "enum": ["public", "personal"], "default": "public"},
                        "user_id": {"type": ["string", "null"]},
                    },
                    "required": ["units"],
                },
            },
            "memory.link": {
                "callable": link,
                "description": "Create a link between two motifs.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "src_id": {"type": "string"},
                        "dst_id": {"type": "string"},
                        "layer": {"type": "string", "enum": ["public", "personal"], "default": "public"},
                        "user_id": {"type": ["string", "null"]},
                    },
                    "required": ["src_id", "dst_id"],
                },
            },
            "memory.wipe_namespace": {
                "callable": wipe_namespace,
                "description": "Delete all data in a namespace (public or specific personal UUID).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "layer": {"type": "string", "enum": ["public", "personal"], "default": "public"},
                        "user_id": {"type": ["string", "null"]},
                    },
                    "required": ["layer"],
                },
            },
        }
    }
