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
