"""
debug_logger.py  —  Orb v2 interaction logger.

Uses a JSON file in /tmp so log entries persist when the user navigates
between Streamlit pages (session_state is NOT shared across pages on
Streamlit Cloud). Falls back to session_state only if /tmp is unavailable.
"""
import json
import os
import streamlit as st
from datetime import datetime
from pathlib import Path

MAX_ENTRIES  = 50
_LOG_FILE    = Path("/tmp/orb_v2_debug_log.json")
_SS_KEY      = "debug_log"   # session_state fallback key


# ── FILE HELPERS ──────────────────────────────────────────────────────────────

def _read_file() -> list:
    try:
        if _LOG_FILE.exists():
            return json.loads(_LOG_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return []


def _write_file(log: list):
    try:
        _LOG_FILE.write_text(
            json.dumps(log, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
    except Exception:
        pass   # /tmp not writable — fall through to session_state


# ── SESSION STATE FALLBACK ────────────────────────────────────────────────────

def _ss_read() -> list:
    return st.session_state.get(_SS_KEY, [])


def _ss_write(log: list):
    st.session_state[_SS_KEY] = log


# ── PUBLIC API ────────────────────────────────────────────────────────────────

def log_interaction(
    question:       str,
    routing:        dict,
    retrieval_mode: str,
    intent:         str,
    data_context:   str,
    system_prompt:  str,
    ai_response:    str,
    vector_docs:    list = None,
    latency_ms:     dict = None,
    max_tokens:     int  = None,
    rewritten_q:    str  = None,
):
    """Append one interaction. Newest entry first. Max MAX_ENTRIES kept."""
    entry = {
        "ts":             datetime.now().strftime("%H:%M:%S"),
        "question":       question,
        "rewritten_q":    rewritten_q,
        "routing":        routing,
        "retrieval_mode": retrieval_mode,
        "intent":         intent,
        "data_context":   data_context,
        "system_prompt":  system_prompt,
        "ai_response":    ai_response,
        "vector_docs":    vector_docs or [],
        "latency_ms":     latency_ms or {},
        "max_tokens":     max_tokens,
    }

    # Try file store first
    log = _read_file()
    if log is not None:
        log.insert(0, entry)
        log = log[:MAX_ENTRIES]
        _write_file(log)

    # Always mirror to session_state so Debug Panel can read it in same page nav
    ss_log = _ss_read()
    ss_log.insert(0, entry)
    _ss_write(ss_log[:MAX_ENTRIES])


def get_log() -> list:
    """Return log entries — prefers file store so cross-page nav works."""
    file_log = _read_file()
    if file_log:
        # Keep session_state in sync
        _ss_write(file_log)
        return file_log
    return _ss_read()


def clear_log():
    try:
        _LOG_FILE.unlink(missing_ok=True)
    except Exception:
        pass
    st.session_state[_SS_KEY] = []
