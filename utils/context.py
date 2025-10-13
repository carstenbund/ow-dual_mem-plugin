from __future__ import annotations
from typing import List, Tuple, Dict, Any

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


def structured_hits(layer: str, hits: List[Tuple[object, float]]) -> List[Dict[str, Any]]:
    """Return a serialisable representation of retrieval hits.

    The Open WebUI host can surface this structure alongside the chat turn
    without having to parse the Markdown context block.
    """

    out: List[Dict[str, Any]] = []
    for motif, score in hits:
        out.append(
            {
                "id": getattr(motif, "id", None),
                "layer": layer,
                "score": float(score),
                "symbols": list(getattr(motif, "symbols", []) or []),
                "content": getattr(motif, "content", ""),
                "owner": getattr(motif, "metadata", {}).get("owner") if getattr(motif, "metadata", None) else None,
            }
        )
    return out


def attachment_payload(layer: str, hits: List[Tuple[object, float]]) -> Dict[str, Any]:
    """Small helper to build a markdown attachment payload for the UI."""

    block = render_context(hits).replace("## Context (top motifs)", f"## {layer.title()} memory", 1)
    return {
        "type": "markdown",
        "title": f"{layer.title()} memory",  # Public / Personal
        "content": block,
        "meta": {
            "layer": layer,
            "hits": len(hits),
        },
    }
