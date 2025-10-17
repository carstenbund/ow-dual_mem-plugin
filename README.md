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

## Project status

### Usability and functionality assessment

- **Filter + tool wiring works end-to-end.** The filter persists both public and personal units through a shared `ChromaStore`, applies policy gating, and attaches retrieval context, structured metadata, and markdown attachments for the host UI to consume. Tools expose retrieval, persistence, linking, and admin helpers over the same storage backend.【F:filters/dual_layer_memory_filter.py†L1-L83】【F:tools/memory.py†L1-L173】
- **Configuration is validated at import time.** `utils.config_loader` enforces schema requirements, normalises paths, and ensures the persistence directory exists before instantiating storage or policy objects, preventing silent misconfiguration.【F:utils/config_loader.py†L1-L104】
- **Extraction is resilient to missing LLMs.** The thread ask resolver chains the host-provided callable, an optional HTTP LLM client, and a deterministic regex fallback so capture still works offline.【F:utils/extraction.py†L1-L244】
- **Chroma integration is production-ready.** The storage layer caches router instances, applies novelty and dedup policies, records simple metrics, and rate-limits automatic linking to avoid runaway graph growth.【F:storage/chroma_store.py†L1-L239】【F:storage/chroma_store.py†L239-L366】

Collectively, the plugin is usable today inside Open WebUI for capturing and retrieving memories as long as Chroma and embedding dependencies are available. The remaining work is mostly around packaging, observability exposure, and host-level UX.

### Completed milestones

1. ✅ **Functional Open WebUI integration.** Filter + tool registration work with configurable extraction fallbacks and Chroma persistence.【F:filters/dual_layer_memory_filter.py†L1-L83】【F:tools/memory.py†L1-L173】
2. ✅ **In-chat memory surfacing hooks.** Retrieval context, metadata, and attachments are emitted on each user turn so the host can display them in the UI.【F:filters/dual_layer_memory_filter.py†L35-L76】
3. ✅ **Configuration validation and dependency checks.** Import-time schema validation and embedder initialisation guard against missing models or directories, with optional hashing fallback.【F:utils/config_loader.py†L1-L104】【F:routing/chroma_router.py†L1-L104】

### Outstanding work

1. **Ship packaging metadata and dependency pins.** Provide a `pyproject.toml`/`requirements.txt` plus documentation for downloading embedding models so deployments are reproducible.【F:tools/memory.py†L1-L35】【F:routing/chroma_router.py†L23-L83】
2. **Document or build UI affordances for memory context.**
   - **Topic.** Give operators and contributors guidance on displaying the structured payload emitted by the filter so the surfaced memories are actionable within chat. The filter currently passes a `memory_context` string, per-hit metadata in `metadata.memory_hits`, and any markdown attachments alongside the assistant prompt for each user turn.【F:filters/dual_layer_memory_filter.py†L35-L76】
   - **Recommendation.** Extend the Open WebUI message renderer to: (a) show a collapsible "Memories" panel that lists each hit title, score, and namespace from `metadata.memory_hits`; (b) inline the `memory_context` block as summarised bullet points at the top of that panel; and (c) render markdown attachments using the existing markdown component with download links for embedded files. Document these affordances in the Open WebUI README so self-hosters know how the memory context appears and how to toggle it.【F:filters/dual_layer_memory_filter.py†L35-L76】
3. **Expose metrics and health diagnostics.** The store records latency/error counters but nothing exports them; add tooling or endpoints so operators can monitor persistence and link budgets.【F:storage/chroma_store.py†L53-L113】
4. **Add automated tests.** Cover policy gating, novelty scoring, link rate limiting, and extractor fallbacks with unit and integration tests to prevent regressions.【F:storage/chroma_store.py†L239-L366】【F:utils/extraction.py†L1-L244】
