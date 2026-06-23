"""
semantic.py  —  Orb v2 Semantic Layer

Loads semantic_layer.yaml and exposes helpers used by:
  - router.py      → normalise() before routing
  - ai_engine.py   → thresholds() in retrieve_data() and system prompt
  - vector_store.py → synonym expansion in document text
  - debug panel    → show current config

All functions are safe to call even if the YAML is missing or malformed
(falls back to sensible defaults so the app never crashes).
"""
import re
import yaml
import streamlit as st
from pathlib import Path
from functools import lru_cache

_YAML_PATH = Path(__file__).parent / "semantic_layer.yaml"

# ── LOADER ────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=60)   # reload at most once per minute so edits take effect
def _load(_hash: str = "") -> dict:
    """Load and parse the YAML. Returns empty dict on any error."""
    try:
        return yaml.safe_load(_YAML_PATH.read_text()) or {}
    except Exception:
        return {}

def _cfg() -> dict:
    """Always returns the current config (re-reads if file changed)."""
    try:
        mtime = str(_YAML_PATH.stat().st_mtime)
    except Exception:
        mtime = ""
    return _load(mtime)

# ── DEFAULTS (used when YAML is absent or key missing) ────────────────────────
_DEFAULT_THRESHOLDS = {
    "consecutive_miss_min":           3,
    "consecutive_miss_severe":        5,
    "low_payout_pct":                 50,
    "near_miss_pct":                  5,
    "high_attainment_pct":            90,
    "high_pmgm_bands":                ["Exceptional", "Exceeds Expectations"],
    "low_pmgm_bands":                 ["Below Expectations", "Unsatisfactory"],
    "new_joiner_months":              6,
    "top_n_default":                  10,
    "ranking_min_payout":             0,
    "anomaly_high_pmgm_low_payout_pct":  50,
    "anomaly_low_pmgm_high_payout_pct":  95,
    "proration_threshold":            1.0,
    "global_scope_keyword":           "ALL",
}

# ── PUBLIC API ────────────────────────────────────────────────────────────────

def thresholds() -> dict:
    """Return all threshold values, merged with defaults."""
    cfg = _cfg()
    t = dict(_DEFAULT_THRESHOLDS)
    t.update(cfg.get("thresholds", {}))
    return t

def get_threshold(key: str, default=None):
    """Get a single threshold value by key."""
    return thresholds().get(key, default)

def synonyms() -> dict:
    """Return the full synonym dict {canonical: [phrases]}."""
    return _cfg().get("synonyms", {})

def field_aliases() -> dict:
    """Return field alias map {user term: column name}."""
    return _cfg().get("field_aliases", {})

def intent_hints() -> dict:
    """Return intent hint map {intent: [phrases]}."""
    return _cfg().get("intent_hints", {})

def resolve_field(user_term: str) -> str:
    """Map a user-facing term to its actual column name, or return as-is."""
    aliases = field_aliases()
    return aliases.get(user_term.lower().strip(), user_term)

def normalise(text: str) -> str:
    """
    Replace synonym phrases in text with their canonical terms.
    Applied before routing and vector retrieval so patterns match reliably.

    Example:
        normalise("who has remuneration below 50%")
        → "who has payout below 50%"
    """
    if not text:
        return text
    result = text
    for canonical, phrases in synonyms().items():
        for phrase in (phrases or []):
            # Case-insensitive whole-phrase replacement
            pattern = re.compile(re.escape(phrase), re.IGNORECASE)
            result  = pattern.sub(canonical, result)
    return result

def expand_for_vector(text: str) -> str:
    """
    Append synonym expansions to a document string for richer TF-IDF matching.
    Used in vector_store._fr_to_text() and ._fh_to_text().

    Example:
        expand_for_vector("Bea Reyes E0016 Scheme B zero payout qualifier failed")
        → "Bea Reyes ... + all synonyms for zero, payout, qualifier, failed"
    """
    t      = text.lower()
    extras = []
    for canonical, phrases in synonyms().items():
        # If canonical term appears, append its synonym phrases as additional vocab
        if canonical.replace("_", " ") in t or canonical in t:
            extras.extend(phrases or [])
        # Also check reverse — if a phrase appears, add the canonical
        for phrase in (phrases or []):
            if phrase.lower() in t:
                extras.append(canonical)
                break
    return text + " " + " ".join(extras) if extras else text

def hint_intent(text: str) -> str | None:
    """
    Check intent_hints for a forced intent match.
    Returns the intent name if found, else None.
    Used as a pre-check before regex patterns.
    """
    t_lower = text.lower()
    for intent, phrases in intent_hints().items():
        for phrase in (phrases or []):
            if phrase.lower() in t_lower:
                return intent
    return None

def threshold_summary_for_prompt() -> str:
    """
    Returns a compact threshold summary to embed in the AI system prompt,
    ensuring the AI uses consistent business definitions in its responses.
    """
    t = thresholds()
    lines = [
        "Business rule definitions (use these consistently in your responses):",
        f"- Consecutive miss threshold: {t['consecutive_miss_min']}+ cycles = underperformer flag",
        f"- Severe consecutive miss: {t['consecutive_miss_severe']}+ cycles = critical flag",
        f"- Low payout: < {t['low_payout_pct']}% of scheme maximum",
        f"- Near miss: within {t['near_miss_pct']}% of scheme maximum",
        f"- High attainment: >= {t['high_attainment_pct']}% of scheme maximum",
        f"- High PMGM bands: {', '.join(t['high_pmgm_bands'])}",
        f"- Low PMGM bands: {', '.join(t['low_pmgm_bands'])}",
        f"- New joiner: joined within last {t['new_joiner_months']} months",
        f"- Default top N: {t['top_n_default']} when not specified",
        f"- Proration: ProrFactor < {t['proration_threshold']}",
        f"- Anomaly (high performer low pay): PMGM in top bands AND payout < {t['anomaly_high_pmgm_low_payout_pct']}% of max",
        f"- Anomaly (low performer high pay): PMGM in low bands AND payout >= {t['anomaly_low_pmgm_high_payout_pct']}% of max",
    ]
    return "\n".join(lines)

def raw_config() -> dict:
    """Return the full parsed YAML for display in the config UI."""
    return _cfg()
