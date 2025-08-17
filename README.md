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
```

Place this folder into your Open WebUI plugins / tools workspace and enable:

* **Tool**: `dual-layer-memory / memory`
* **Filter**: `dual_layer_memory`

Configure `memory_config.yaml`, then chat. Use the `memory.retrieve` tool to pull context.
