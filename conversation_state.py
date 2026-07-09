"""
conversation_state.py  —  Orb v2 Conversation Memory

Maintains a lightweight "conversation context" object in session_state
across turns so the router and AI engine always have a richer picture
of the current conversation than just raw message history.

What it tracks
--------------
  topic_intent      : the dominant intent of the current conversation thread
  active_filters    : filters set during this thread (country, scheme, cycle, status)
  topic_summary     : a one-line plain-English description of what's being discussed
  turns_in_topic    : how many turns have been spent on the current topic
  response_summaries: rolling list of short AI response summaries (last 6)
  last_df_columns   : column names of the last dataframe returned
  clarification_pending : whether we're waiting for the user to answer a question
  clarification_context : what question was asked + what options were given

Usage
-----
  from conversation_state import get_ctx, update_ctx, reset_ctx, ctx_for_router, ctx_for_ai
"""

import streamlit as st

_SS_KEY = "conv_ctx"

_DEFAULTS = {
    "topic_intent":            None,
    "active_filters":          {
        "country":  None,
        "scheme":   None,
        "cycle":    None,
        "status":   None,
        "top_n":    None,
    },
    "topic_summary":           "",
    "turns_in_topic":          0,
    "response_summaries":      [],   # list of str, max 6
    "last_df_columns":         [],
    "clarification_pending":   False,
    "clarification_context":   None,
    "pinned_country":          None,  # set by sidebar scope pin
}


def _init():
    if _SS_KEY not in st.session_state:
        st.session_state[_SS_KEY] = dict(_DEFAULTS)


def get_ctx() -> dict:
    _init()
    return st.session_state[_SS_KEY]


def update_ctx(**kwargs):
    """Partial update — only supplied keys are changed."""
    _init()
    ctx = st.session_state[_SS_KEY]

    # Merge active_filters instead of replacing
    if "active_filters" in kwargs:
        for k, v in kwargs.pop("active_filters").items():
            if v is not None:
                ctx["active_filters"][k] = v

    ctx.update(kwargs)


def add_response_summary(summary: str):
    """Append a short summary of the latest AI response. Keeps last 6."""
    _init()
    ctx = st.session_state[_SS_KEY]
    ctx["response_summaries"].append(summary[:200])
    ctx["response_summaries"] = ctx["response_summaries"][-6:]


def set_clarification(question: str, options: list, intent_candidates: list,
                       clarification_type: str = "intent"):
    """
    Mark that a clarification question has been sent to the user.

    clarification_type:
        "intent"   — we're unsure which intent the user wants
        "filter"   — we need a specific filter value (country, scheme, etc.)
        "scope"    — the question applies to multiple scopes
        "detail"   — we need more detail before fetching data
    """
    update_ctx(
        clarification_pending=True,
        clarification_context={
            "question":           question,
            "options":            options,
            "intent_candidates":  intent_candidates,
            "clarification_type": clarification_type,
        },
    )


def clear_clarification():
    update_ctx(
        clarification_pending=False,
        clarification_context=None,
    )


def reset_ctx():
    """Call when a new chat starts."""
    _init()
    st.session_state[_SS_KEY] = dict(_DEFAULTS)


def bump_topic(new_intent: str, topic_summary: str, router_filters: dict):
    """
    Called after each successful turn. Updates topic tracking.
    Clears status filter when moving to a workforce-wide intent.
    """
    _init()
    ctx = st.session_state[_SS_KEY]

    _TOPIC_CHANGE_INTENTS = {
        "headcount", "attrition", "pmgm", "cycle_summary",
        "country_compare", "employee_list",
    }
    _DETAIL_INTENTS = {
        "cross_join", "ranking", "anomaly", "cross_check",
        "qualifier", "proration", "new_joiner",
    }
    # Intents that should never carry a status filter forward
    _CLEAR_STATUS_INTENTS = {
        "anomaly", "attainment", "cycle_summary", "ranking",
        "country_compare", "project_compare", "tenure_compare",
        "kpi_trend", "pmgm", "underperformance", "headcount",
        "employee_list", "missing_kpi", "scheme_config",
    }

    prev = ctx["topic_intent"]
    if (prev is not None
            and new_intent != prev
            and new_intent not in _DETAIL_INTENTS
            and prev not in _DETAIL_INTENTS
            and new_intent in _TOPIC_CHANGE_INTENTS):
        ctx["turns_in_topic"] = 0

    ctx["topic_intent"]   = new_intent
    ctx["topic_summary"]  = topic_summary
    ctx["turns_in_topic"] = ctx.get("turns_in_topic", 0) + 1

    # Clear status filter for workforce-wide intents — never bleed "Non-Active" into anomaly
    if new_intent in _CLEAR_STATUS_INTENTS:
        ctx["active_filters"]["status"] = None

    # Merge in filters from router (non-null values only)
    if router_filters:
        for k, v in router_filters.items():
            if v is not None and k in ctx["active_filters"]:
                ctx["active_filters"][k] = v


def ctx_for_router() -> str:
    """
    Returns a compact string summary of current conversation context
    to inject into the router prompt.
    """
    _init()
    ctx = st.session_state[_SS_KEY]

    lines = []

    if ctx["topic_intent"]:
        lines.append(f"Current topic: {ctx['topic_intent']} — {ctx['topic_summary']}")
        lines.append(f"Turns on this topic: {ctx['turns_in_topic']}")

    active = {k: v for k, v in ctx["active_filters"].items() if v is not None}
    if active:
        lines.append(f"Active filters in this conversation: {active}")

    if ctx["last_df_columns"]:
        lines.append(f"Last result columns: {', '.join(ctx['last_df_columns'])}")

    if ctx.get("pinned_country"):
        lines.append(f"SCOPE PIN active: user has pinned all queries to country={ctx['pinned_country']}")

    if ctx["response_summaries"]:
        lines.append("Recent AI responses (brief):")
        for s in ctx["response_summaries"][-3:]:
            lines.append(f"  • {s}")

    if ctx["clarification_pending"] and ctx["clarification_context"]:
        cc = ctx["clarification_context"]
        lines.append(f"[CLARIFICATION PENDING] We asked: \"{cc['question']}\"")
        lines.append(f"Options offered: {cc['options']}")

    return "\n".join(lines) if lines else "(no prior context)"


def ctx_for_ai() -> str:
    """
    Returns a richer context block to embed in the AI system prompt.
    """
    _init()
    ctx = st.session_state[_SS_KEY]

    lines = []

    if ctx["topic_intent"]:
        lines.append(
            f"Conversation topic: {ctx['topic_intent']} ({ctx['topic_summary']}). "
            f"Turn {ctx['turns_in_topic']} of this thread."
        )

    active = {k: v for k, v in ctx["active_filters"].items() if v is not None}
    if active:
        filt_str = ", ".join(f"{k}={v}" for k, v in active.items())
        lines.append(
            f"Filters established in this conversation: {filt_str}. "
            "Apply these unless the user explicitly changes them."
        )

    if ctx["response_summaries"]:
        lines.append("What you have already told the user (most recent first):")
        for s in reversed(ctx["response_summaries"][-4:]):
            lines.append(f"  • {s}")

    return "\n".join(lines) if lines else ""
