from __future__ import annotations
from typing import Dict, Any
import yaml, os
from storage.chroma_store import ChromaStore
from policy.runtime_policy import Policy
from utils.reducers import extract_public_units, extract_personal_units
from utils.context import render_context
from utils.extraction import build_thread_ask

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

_ask_resolver = build_thread_ask(_cfg)

def on_message(event: Dict[str, Any]) -> Dict[str, Any]:
    if event.get("role") != "user":
        return event

    text = event.get("content","")
    user_id = event.get("user_id")
    host_ask = event.get("_thread_ask")
    ask = _ask_resolver(host_ask)

    # 1) extract + persist
    pub_units = extract_public_units(ask, text)
    per_units = extract_personal_units(ask, text) if user_id else []

    if pub_units:
        _store.persist_units("public", None, pub_units, PUB)
    if per_units and user_id:
        _store.persist_units("personal", user_id, per_units, PER)

    # 2) Optional: attach retrieval context for the next turn
    cfg_ret = _cfg.get("retrieval", {}) or {}
    if cfg_ret.get("attach_context", True):
        k_pub = int(cfg_ret.get("k_public", 5))
        k_per = int(cfg_ret.get("k_personal", 3))
        ctx = ""

        r_pub = _store.router("public", None)
        hits_pub = r_pub.search_text(text, top_k=max(0, k_pub))
        if hits_pub: ctx += render_context(hits_pub)

        if user_id and k_per > 0:
            r_per = _store.router("personal", user_id)
            hits_per = r_per.search_text(text, top_k=k_per)
            if hits_per: ctx += render_context(hits_per)

        # Host can read this and prepend to system/user prompt for the model
        if ctx:
            event["memory_context"] = ctx

    return event

def register():
    return {"on_message": on_message}
