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
        lines.append(chunk)
        used += len(chunk)
    return header + "".join(lines)
