from __future__ import annotations

import os
from typing import Any, Dict

import yaml


class ConfigError(RuntimeError):
    """Raised when the memory configuration cannot be loaded or validated."""


_CONFIG_CACHE: Dict[str, Any] | None = None


def _config_path() -> str:
    return os.path.join(os.path.dirname(__file__), "..", "memory_config.yaml")


def load_config(path: str | None = None) -> Dict[str, Any]:
    """Load the configuration YAML and validate the schema."""

    cfg_path = path or _config_path()
    if not os.path.exists(cfg_path):
        raise ConfigError(f"memory_config.yaml not found at {cfg_path}")

    try:
        with open(cfg_path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
    except Exception as exc:  # pragma: no cover - defensive
        raise ConfigError(f"Failed to load memory_config.yaml: {exc}") from exc

    return validate_config(cfg)


def get_config(path: str | None = None) -> Dict[str, Any]:
    global _CONFIG_CACHE
    if _CONFIG_CACHE is None:
        _CONFIG_CACHE = load_config(path)
    return _CONFIG_CACHE


def validate_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    errors: list[str] = []

    router = cfg.get("router") or {}
    persist_dir = router.get("persist_dir")
    public_collection = router.get("public_collection")
    personal_prefix = router.get("personal_prefix")

    if not isinstance(persist_dir, str) or not persist_dir.strip():
        errors.append("router.persist_dir must be a non-empty string")
    if not isinstance(public_collection, str) or not public_collection.strip():
        errors.append("router.public_collection must be a non-empty string")
    if not isinstance(personal_prefix, str) or not personal_prefix.strip():
        errors.append("router.personal_prefix must be a non-empty string")

    embedding = cfg.get("embedding") or {}
    model_name = embedding.get("model_name")
    allow_hash_fallback = embedding.get("allow_hash_fallback", False)
    if not isinstance(model_name, str) or not model_name.strip():
        errors.append("embedding.model_name must be provided")
    if not isinstance(allow_hash_fallback, bool):
        errors.append("embedding.allow_hash_fallback must be a boolean if provided")

    policy = cfg.get("policy") or {}
    for layer in ("public", "personal"):
        section = policy.get(layer)
        if not isinstance(section, dict):
            errors.append(f"policy.{layer} must be a mapping with threshold values")
            continue
        for key in ("link_threshold", "dedup_threshold", "novelty_min", "symbol_jaccard_cap"):
            value = section.get(key)
            if not isinstance(value, (int, float)):
                errors.append(f"policy.{layer}.{key} must be a number between 0 and 1")
                continue
            if not (0.0 <= float(value) <= 1.0):
                errors.append(f"policy.{layer}.{key} must be between 0 and 1")
        max_links = section.get("max_links_per_motif")
        if not isinstance(max_links, int) or max_links < 0:
            errors.append(f"policy.{layer}.max_links_per_motif must be a non-negative integer")
        rate_limit = section.get("max_links_per_minute", 30)
        if not isinstance(rate_limit, int) or rate_limit < 0:
            errors.append(f"policy.{layer}.max_links_per_minute must be a non-negative integer")

    retrieval = cfg.get("retrieval") or {}
    for key in ("k_public", "k_personal"):
        value = retrieval.get(key)
        if value is None:
            errors.append(f"retrieval.{key} must be set")
        elif not isinstance(value, int) or value < 0:
            errors.append(f"retrieval.{key} must be a non-negative integer")
    attach_context = retrieval.get("attach_context", True)
    if not isinstance(attach_context, bool):
        errors.append("retrieval.attach_context must be a boolean")

    extractor = cfg.get("extractor") or {}
    fallback = extractor.get("fallback")
    if fallback is not None and not isinstance(fallback, str):
        errors.append("extractor.fallback must be a string if provided")

    if errors:
        raise ConfigError("Invalid memory_config.yaml:\n - " + "\n - ".join(errors))

    # Normalise strings by stripping whitespace
    router["persist_dir"] = persist_dir.strip()
    router["public_collection"] = public_collection.strip()
    router["personal_prefix"] = personal_prefix.strip()
    embedding["model_name"] = model_name.strip()
    embedding["allow_hash_fallback"] = bool(allow_hash_fallback)

    try:
        os.makedirs(router["persist_dir"], exist_ok=True)
    except OSError as exc:  # pragma: no cover - defensive
        raise ConfigError(f"Unable to create persist directory '{router['persist_dir']}': {exc}") from exc

    return cfg

