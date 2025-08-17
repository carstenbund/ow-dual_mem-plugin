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

PUB = Policy(**_cfg.get("policy", {}).get("public", {})) if _cfg.get("policy") else Policy(0.65, 0.95, 0.20, 2, 0.80)
PER = Policy(**_cfg.get("policy", {}).get("personal", {})) if _cfg.get("policy") else Policy(0.60, 0.93, 0.10, 3, 0.80)

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

    text = event.get("content", "")
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
