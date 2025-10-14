from __future__ import annotations
from typing import Optional, Dict, Any, List
from utils.context import render_context
from storage.chroma_store import ChromaStore
from policy.runtime_policy import Policy
from utils.config_loader import get_config

_cfg = get_config()

_store = ChromaStore(
    persist_dir=_cfg["router"]["persist_dir"],
    public_name=_cfg["router"]["public_collection"],
    personal_prefix=_cfg["router"]["personal_prefix"],
    embedding_model=_cfg["embedding"]["model_name"],
    allow_hash_fallback=_cfg["embedding"].get("allow_hash_fallback", False),
)

PUB = Policy(**_cfg["policy"]["public"])
PER = Policy(**_cfg["policy"]["personal"])

def retrieve(query: str, layer: str = "public", user_id: Optional[str] = None, k: int = 5) -> Dict[str, Any]:
    hits = _store.search_text(layer, user_id, query, top_k=max(1, k))
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


def links_for(motif_id: str, layer: str="public", user_id: Optional[str]=None) -> Dict[str, Any]:
    return {"edges": _store.links_for(layer, user_id, motif_id)}

def wipe_namespace(layer: str, user_id: Optional[str] = None) -> Dict[str, Any]:
    _store.wipe_namespace(layer, user_id)
    return {"wiped": {"layer": layer, "user_id": user_id}}


def admin_links_for(motif_id: str, layer: str = "public", user_id: Optional[str] = None) -> Dict[str, Any]:
    """Return link metadata plus destination motif snippets for admins."""

    edges = _store.links_for(layer, user_id, motif_id)
    router = _store.router(layer, user_id)
    cache = router._cache  # cached motifs for this namespace
    summary = []
    for edge in edges:
        dst_id = edge.get("dst")
        dst = cache.get(dst_id)
        summary.append(
            {
                "edge_id": edge.get("id"),
                "dst_id": dst_id,
                "relation": edge.get("rel", "linked"),
                "score_hint": edge.get("score"),
                "created_at": edge.get("ts"),
                "dst_symbols": list(getattr(dst, "symbols", []) or []),
                "dst_excerpt": (getattr(dst, "content", "") or "").strip()[:200],
            }
        )

    return {
        "motif_id": motif_id,
        "layer": layer,
        "user_id": user_id,
        "count": len(summary),
        "links": summary,
    }


def admin_wipe_namespace(layer: str = "public", user_id: Optional[str] = None, confirm: bool = False) -> Dict[str, Any]:
    """Safety wrapper around wipe_namespace for administrative tooling."""

    if not confirm:
        return {
            "confirmation_required": True,
            "message": "Set confirm=true to wipe the namespace.",
            "layer": layer,
            "user_id": user_id,
        }

    return wipe_namespace(layer=layer, user_id=user_id)

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
            "memory.links_for": {
                "callable": links_for,
                "description": "List outgoing links for a motif.",
                "parameters": {
                    "type":"object",
                    "properties":{
                        "motif_id":{"type":"string"},
                        "layer":{"type":"string","enum":["public","personal"],"default":"public"},
                        "user_id":{"type":["string","null"]}
                    },
                    "required":["motif_id"]
                }
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
            "memory.admin_links_for": {
                "callable": admin_links_for,
                "description": "Admin helper that summarizes outgoing links and destination snippets.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "motif_id": {"type": "string"},
                        "layer": {"type": "string", "enum": ["public", "personal"], "default": "public"},
                        "user_id": {"type": ["string", "null"]},
                    },
                    "required": ["motif_id"],
                },
            },
            "memory.admin_wipe_namespace": {
                "callable": admin_wipe_namespace,
                "description": "Require explicit confirmation before wiping a namespace.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "layer": {"type": "string", "enum": ["public", "personal"], "default": "public"},
                        "user_id": {"type": ["string", "null"]},
                        "confirm": {"type": "boolean", "default": False},
                    },
                    "required": ["layer"],
                },
            },
        }
    }
