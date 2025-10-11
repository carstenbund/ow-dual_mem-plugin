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

### Memory extractor configuration

The filter attempts to use the host-provided `_thread_ask` hook first. When the host does not supply one, the plugin now reads
`memory_config.yaml > extractor`:

```yaml
extractor:
  fallback: "regex"            # deterministic extractor used when HTTP calls fail or are disabled
  llm:
    endpoint: "http://127.0.0.1:3000/api/chat/completions"  # OpenAI compatible endpoint (leave blank to disable)
    model: "gpt-4o-mini"       # Optional model hint for OpenAI compatible servers
    api_key_env: "OPENWEBUI_API_KEY"  # Environment variable that stores the bearer token (optional)
    timeout: 15
    temperature: 0.0
```

If the HTTP call fails or is not configured, the regex fallback will still derive structured memories from user turns so capture
and retrieval remain operational.

## Implementation task list

1. **Complete functional integration with Open WebUI** ✅
   - `_thread_ask` now routes through a configurable HTTP client and automatically falls back to a deterministic regex-based extractor when a hosted LLM is unavailable.
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
