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

## Implementation task list

1. **Complete functional integration with Open WebUI**
   - Wire the filter's `_thread_ask` calls to an available LLM endpoint so `extract_public_units` and `extract_personal_units` capture memories during real conversations.
   - Provide a configurable fallback extractor (e.g., deterministic regex or local model) for environments without hosted LLM access to keep the plugin operational.
2. **Surface retrieved memory inside the chat experience**
   - Render `memory_context` blocks or add UI affordances in Open WebUI so users can inspect which memories were retrieved for each turn.
   - Expose admin-facing commands or panels that call `memory.links_for` and `memory.wipe_namespace` to manage personal and shared memories.
3. **Validate configuration and dependencies on startup**
   - Add schema checks for `memory_config.yaml`, ensuring embedding model names, collection paths, and thresholds are present and valid.
   - Emit clear errors (or health endpoints) when Chroma persistence or the embedding model cannot be initialized, instead of silently degrading to hashing.
4. **Package for predictable deployment**
   - Publish dependency metadata (`pyproject.toml` or `requirements.txt`) with pinned versions for Chroma, sentence-transformers, and other runtime libraries.
   - Document or script pre-download of embedding models and required volume mounts for Chroma persistence to avoid runtime surprises.
5. **Improve observability and performance safeguards**
   - Reuse shared router/store instances to avoid per-request cache rebuilds, and instrument latency/error metrics for memory operations.
   - Add rate limiting or guardrails around automatic link creation to prevent runaway cross-namespace associations.
6. **Establish automated testing coverage**
   - Create unit tests for policy decisions (deduplication, novelty gating, link thresholds) and storage operations across namespaces.
   - Add integration tests that simulate a chat flow through the filter to confirm capture, retrieval, and context rendering behave as expected.
