"""
clarifier.py  —  Orb v2 Clarification Layer

Decides whether a question is clear enough to answer directly, or whether
Orb should ask the user a focused follow-up question first — like how
you'd ask a colleague to be more specific before pulling a report.

The clarifier can ask multiple turns of questions, each with clickable
option buttons rendered in the chat. The user's selection is then treated
as a refined question that goes back through the normal routing pipeline.

Architecture
------------
  needs_clarification(question, routing, ctx)
      → (bool, ClarificationRequest | None)

  build_clarification_message(cr)
      → str   (the message Orb sends to the user)

  resolve_clarification(user_reply, cr)
      → str   (a new, fully-specified question to re-route)

  CLARIFICATION_BUTTONS in session_state
      → list of {label, value} shown as clickable chips in app.py
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Optional


# ── DATA CLASSES ──────────────────────────────────────────────────────────────

@dataclass
class ClarificationOption:
    label: str          # Short text shown on the button
    value: str          # The refined question text if this option is picked


@dataclass
class ClarificationRequest:
    question:    str                          # What Orb asks the user
    options:     list[ClarificationOption]    # Clickable options
    follow_up:   Optional[str] = None         # A second-level question (if needed)
    intent_hint: Optional[str] = None         # Preferred intent once resolved
    clarification_type: str = "intent"        # intent | filter | scope | detail


# ── AMBIGUITY RULES ───────────────────────────────────────────────────────────
# Each rule maps a situation to a clarification request.
# Rules are evaluated in order; first match wins.

def _is_short(q: str) -> bool:
    return len(q.split()) <= 5

def _mentions_country(q: str) -> bool:
    return bool(re.search(
        r"\b(sg|my|ph|th|id|singapore|malaysia|philippines|thailand|indonesia|"
        r"all countries|every country|across all countries|"
        r"company.?wide|global(ly)?)\b",
        q, re.IGNORECASE
    ))

def _mentions_name(q: str) -> bool:
    return bool(re.search(
        r"\b(who|name|identify|tell me who|which employee|which person)\b",
        q, re.IGNORECASE
    ))

def _mentions_time(q: str) -> bool:
    return bool(re.search(
        r"\b(last|previous|prior|this|latest|q[1-4]|quarter|cycle|period)\b",
        q, re.IGNORECASE
    ))

def _is_comparison(q: str) -> bool:
    return bool(re.search(r"\b(compare|vs|versus|against|difference)\b", q, re.IGNORECASE))

def _mentions_anomaly_direction(q: str) -> bool:
    """True if the question already specifies which anomaly direction it wants,
    so we don't ask a question the user effectively already answered."""
    return bool(re.search(
        r"\b(high.{0,40}low|low.{0,40}high|both direction|either direction|"
        r"underpaid|overpaid|under.?paid|over.?paid)\b",
        q, re.IGNORECASE
    ))

# Intents where "which country/region" is a meaningful ambiguity for a
# multi-country user. Shared between needs_clarification (Rule 4) and the
# live query-clarity estimator below, so the two stay consistent.
_COUNTRY_SENSITIVE = {"ranking", "anomaly", "underperformance", "qualifier", "cross_check"}


# ── CORE DECISION FUNCTION ────────────────────────────────────────────────────

def needs_clarification(
    question:  str,
    routing:   dict,
    ctx:       dict,   # from conversation_state.get_ctx()
    user:      dict,   # auth user dict
) -> tuple[bool, Optional[ClarificationRequest]]:
    """
    Returns (True, ClarificationRequest) if we should ask for more info,
    or (False, None) if the question is clear enough to answer directly.

    Deliberately conservative — only fires when genuinely ambiguous.
    Does NOT fire on follow-up turns where clarification is already pending.
    """
    intent   = routing.get("intent", "free_form")
    q_lower  = question.lower().strip()
    is_short = _is_short(question)

    # Never clarify if we're already mid-clarification
    if ctx.get("clarification_pending"):
        return False, None

    # Never clarify simple follow-ups — the router/context handles those
    if routing.get("is_followup"):
        return False, None

    # Never clarify if the question already has enough specificity — either
    # because THIS turn names a country/scheme, or because one was already
    # established earlier in the conversation (pinned scope, or inherited
    # from a prior turn's filters). Without checking the latter, scope set
    # once at the start of a chat was invisible here on any later turn that
    # wasn't itself flagged as a follow-up, so Rule 4 kept re-asking.
    if routing.get("filters", {}).get("country") or routing.get("filters", {}).get("scheme"):
        return False, None
    if ctx.get("pinned_country") or (ctx.get("active_filters") or {}).get("country"):
        return False, None

    # ── RULE 1: Vague "show me everything" on free_form ─────────────────────
    if intent == "free_form" and is_short:
        countries = user.get("countries", [])
        has_multi_country = len(countries) > 1 or "ALL" in countries
        return True, ClarificationRequest(
            question="What would you like to know? I can cover several areas:",
            options=[
                ClarificationOption("Incentive Attainment", "What % of employees hit max payout this cycle?"),
                ClarificationOption("Underperformers", "Who has missed targets for 3+ consecutive cycles?"),
                ClarificationOption("Headcount & Status", "Give me a headcount breakdown by country and status"),
                ClarificationOption("PMGM Ratings", "Show me the performance rating distribution"),
                ClarificationOption("Cycle Summary", "Give me an overview of the latest incentive cycle"),
                ClarificationOption("Anomalies", "Flag any mismatches between PMGM rating and payout"),
            ],
            clarification_type="intent",
        )

    # ── RULE 2: Ranking without knowing top N or direction ───────────────────
    if intent == "ranking" and is_short and not re.search(r"\d+", question):
        return True, ClarificationRequest(
            question="I can rank employees by payout. What would you like to see?",
            options=[
                ClarificationOption("Top 10 earners", "Show me the top 10 highest earning employees"),
                ClarificationOption("Bottom 10 earners", "Show me the bottom 10 lowest earning employees"),
                ClarificationOption("Above average only", "Show employees earning above the average payout"),
                ClarificationOption("Full ranked list", "Show all employees ranked by payout, highest to lowest"),
            ],
            clarification_type="detail",
        )

    # ── RULE 3: Underperformance without a threshold specified ───────────────
    if intent == "underperformance" and is_short and not re.search(r"\d+", question):
        return True, ClarificationRequest(
            question="How many consecutive cycles of missed targets should I flag?",
            options=[
                ClarificationOption("2+ cycles (broader)", "Show employees who missed targets in 2 or more consecutive cycles"),
                ClarificationOption("3+ cycles (default)", "Show employees who missed targets in 3 or more consecutive cycles"),
                ClarificationOption("5+ cycles (severe)", "Show employees with 5 or more consecutive cycles below target"),
                ClarificationOption("All with any miss", "Show all employees who have missed a target in any cycle"),
            ],
            clarification_type="detail",
        )

    # ── RULE 4: Country ambiguity for multi-country users ───────────────────
    countries = user.get("countries", [])
    has_multi = len(countries) > 1 or "ALL" in countries
    if (intent in _COUNTRY_SENSITIVE
            and has_multi
            and not _mentions_country(q_lower)):
        if "ALL" in countries:
            scope_options = [
                ClarificationOption("Global (all countries)", f"{question} — across all countries"),
                ClarificationOption("Singapore (SG)", f"{question} — for Singapore only"),
                ClarificationOption("Malaysia (MY)", f"{question} — for Malaysia only"),
                ClarificationOption("Philippines (PH)", f"{question} — for Philippines only"),
                ClarificationOption("Thailand (TH)", f"{question} — for Thailand only"),
            ]
        else:
            scope_options = [
                ClarificationOption("All my countries", f"{question} — across {', '.join(countries)}"),
            ] + [
                ClarificationOption(c, f"{question} — for {c} only")
                for c in countries
            ]
        return True, ClarificationRequest(
            question=f"Which scope should I use for this?",
            options=scope_options,
            clarification_type="scope",
        )

    # ── RULE 5: Attainment — current cycle or trend? ─────────────────────────
    if intent == "attainment" and is_short and not _mentions_time(q_lower):
        return True, ClarificationRequest(
            question="Are you looking at the latest cycle or a trend over time?",
            options=[
                ClarificationOption("Latest cycle only", "What % hit max payout in the latest cycle?"),
                ClarificationOption("Trend across cycles", "Show attainment rate across all cycles as a trend"),
                ClarificationOption("By scheme (latest)", "Break down attainment by incentive scheme for the latest cycle"),
                ClarificationOption("By country (latest)", "Compare attainment rates across countries for the latest cycle"),
            ],
            clarification_type="detail",
        )

    # ── RULE 6: Anomaly — which direction? ───────────────────────────────────
    if intent == "anomaly" and not _mentions_anomaly_direction(q_lower):
        return True, ClarificationRequest(
            question="Which type of anomaly are you looking for?",
            options=[
                ClarificationOption("High rating, low payout", "Who has a high PMGM rating but a low incentive payout?"),
                ClarificationOption("Low rating, high payout", "Who has a low PMGM rating but a high incentive payout?"),
                ClarificationOption("Both directions", "Show all PMGM vs payout mismatches in both directions"),
            ],
            clarification_type="detail",
        )

    # No clarification needed
    return False, None


# ── MESSAGE BUILDER ───────────────────────────────────────────────────────────

# ── LIVE QUERY CLARITY (non-blocking) ─────────────────────────────────────────

def estimate_query_clarity(question: str, routing: dict, needs_clar: bool, user: dict) -> tuple:
    """
    A 0-100 "how well-specified was this question" score shown right after
    the user sends it — purely informational, never blocks sending.

    Deliberately reuses the exact same signals as needs_clarification() above
    rather than a separate heuristic, so the score and the actual
    clarification behaviour can never disagree with each other. This is
    about the QUESTION's phrasing/specificity, not the reply's reliability —
    see _estimate_confidence() in ai_engine.py for that, which factors in
    retrieval mechanics as well.

    Returns (score, factors) — factors are short, user-facing explanations
    ("no country specified"), not internal routing details.
    """
    intent  = routing.get("intent", "free_form")
    q_lower = question.lower().strip()
    score = 95
    factors = []

    if needs_clar:
        # The system is about to ask a genuine clarifying question — that's
        # the clearest possible signal the phrasing was ambiguous.
        score -= 35
        factors.append("the question needed a follow-up to answer accurately")

    if intent == "free_form":
        score -= 20
        factors.append("no specific data area was recognised — try naming a metric (e.g. attainment, headcount, attrition)")

    if "regex fallback" in (routing.get("reasoning") or "").lower():
        score -= 10
        factors.append("the AI router was unavailable, so this was matched by keyword only")

    countries = user.get("countries", [])
    has_multi = len(countries) > 1 or "ALL" in countries
    if intent in _COUNTRY_SENSITIVE and has_multi and not _mentions_country(q_lower):
        score -= 15
        factors.append("no country/region specified — add one, or say 'across all countries'")

    if intent == "anomaly" and not _mentions_anomaly_direction(q_lower):
        score -= 10
        factors.append("didn't specify which direction of anomaly (e.g. high rating, low payout)")

    if intent in ("ranking", "cycle_summary") and not re.search(r"\d+", question) \
            and not re.search(r"\b(top|bottom|highest|lowest|all|everyone)\b", q_lower):
        score -= 8
        factors.append("no specific number or ranking direction given")

    if len(question.split()) <= 2:
        score -= 10
        factors.append("very short — a bit more detail usually helps")

    score = max(30, min(97, score))
    return score, factors


def build_clarification_message(cr: ClarificationRequest) -> str:
    """
    Returns the text Orb sends to the user when asking for clarification.
    The options are stored separately in session_state as buttons.
    """
    lines = [cr.question]
    for i, opt in enumerate(cr.options, 1):
        lines.append(f"  {i}. {opt.label}")
    lines.append("\nClick an option above, or just type your answer.")
    return "\n".join(lines)


def store_clarification_buttons(cr: ClarificationRequest):
    """Store option buttons in session_state so app.py can render them."""
    import streamlit as st
    st.session_state["_clarification_buttons"] = [
        {"label": opt.label, "value": opt.value}
        for opt in cr.options
    ]


def resolve_clarification(user_reply: str, cr: ClarificationRequest) -> str:
    """
    Given the user's reply (typed or from a button click), return
    the refined question to re-route. If the reply matches an option
    label, return that option's value; otherwise use the raw reply.
    """
    reply_lower = user_reply.lower().strip()
    for opt in cr.options:
        if (opt.label.lower() in reply_lower
                or reply_lower in opt.label.lower()
                or user_reply.strip() == opt.value.strip()):
            return opt.value
    # Typed a number?
    m = re.match(r"^(\d+)$", reply_lower)
    if m:
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(cr.options):
            return cr.options[idx].value
    # Raw typed reply — use as-is
    return user_reply
