"""
router.py  —  AI-powered intent router for Orb v2

Uses DeepSeek V4 Flash (fast, cheap) as the routing model.
Falls back to regex detection if the call fails.

Improvements over v13c:
  - Few-shot disambiguation examples for the 6 hardest intent pairs
  - Injects conversation_state context so history > 4 turns is handled
  - Passes last 6 messages (up from 4) to the router
  - Smarter is_followup detection using pronoun + context continuity
  - Regex fallback also uses conversation context
"""
import json, os, re
import requests
import streamlit as st

DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
ROUTER_MODEL     = "deepseek-v4-flash"

KNOWN_INTENTS = [
    "attainment",       # % hitting max payout, payout achievement
    "underperformance", # missed targets, consecutive misses
    "qualifier",        # qualifier failures, blocked payouts
    "proration",        # attendance proration, absenteeism
    "anomaly",          # PMGM vs payout mismatch
    "cross_check",      # non-active with payouts, leavers
    "new_joiner",       # first cycle, recently joined
    "cross_join",       # follow-up: identify / name employees from prior result
    "ranking",          # top N, bottom N, above/below average payout ranking
    "employee_list",    # full employee directory with names
    "headcount",        # how many employees, count by status/country
    "attrition",        # leavers, resignations, turnover
    "pmgm",             # performance rating distribution
    "cycle_summary",    # overall cycle overview / brief me
    "country_compare",  # compare countries on any metric
    "kpi_trend",        # time-series: KPI / payout / proration trend over cycles
    "project_compare",  # compare projects on any metric (like country_compare but by project)
    "tenure_compare",   # compare payout/KPI cohorts by tenure band
    "missing_kpi",      # employees on scheme with no KPI record this cycle
    "adjustment",       # KPI adjustments, qualifying status adjustments, approval workflow
    "scheme_config",    # scheme structure: tiers, KPI weightage, acknowledgement, expiry
    "login",            # last login, login activity
    "announcement",     # announcements active during a cycle
    "free_form",        # anything else / complex multi-part
]

ROUTER_SYSTEM = """You are a data routing assistant for a workforce analytics platform called Orb.
Your ONLY job is to classify the user question and decide what data retrieval is needed.

Available data:
- Flash Reward: EmployeeID, Scheme, SchemeMaxPayout, Cycle, Metric, Target, Achieved,
  MetricPayout, QualifierFailed, ProrationType, ProrFactor, TotalCyclePayout
- Flash Home: EmployeeID, EmployeeName, EmployeeStatus, JoinDate, LastDate,
  Country, Project, PMGMRating

Respond with ONLY a valid JSON object. No explanation, no markdown, no code fences.

Schema:
{
  "intent": "<one of the known intents>",
  "needs_fresh_join": <true|false>,
  "needs_flash_reward": <true|false>,
  "needs_flash_home": <true|false>,
  "is_followup": <true|false>,
  "filters": {
    "country": "<2-letter code or null>",
    "scheme": "<scheme name or null>",
    "cycle": "<cycle string or null>",
    "threshold": "<number or null>",
    "employee_id": "<ID or null>",
    "status": "<Active|Non-Active or null>"
  },
  "reasoning": "<one short sentence>"
}

Set needs_fresh_join = true when:
- Question asks to identify, name, or find specific employees
- Question combines payout with names or HR fields
- Prior result columns lack either payout or employee names
- Question is a follow-up needing data not visible in the previous result

Set is_followup = true when:
- Question references prior context using words like: they, those, these, them, same,
  among, of those, of them, their, those employees, the above, that group, those people
- The conversation context shows a topic has been established and this is a drill-down
- The question is too short to stand alone without prior context (e.g. "what about SG?",
  "and the bottom 10?", "break that down by country")

=== DISAMBIGUATION EXAMPLES (hard cases) ===

Q: "who earns the most?" — intent: ranking (not employee_list — user wants sorted by payout)
Q: "list all employees" — intent: employee_list (not ranking — user wants a directory)
Q: "show me attainment" — intent: attainment (not ranking — user wants % hitting max, not a leaderboard)
Q: "who hit max payout?" — intent: attainment (about hitting the cap, not who earns most)
Q: "top 10 by payout" — intent: ranking (explicit N + sort = ranking)
Q: "who are the underperformers?" — intent: underperformance (not ranking — about consecutive misses)

Q: "who are they?" (after underperformance result) — intent: cross_join, is_followup: true
Q: "name those employees" (after any result) — intent: cross_join, is_followup: true
Q: "show me in SG" (after headcount result) — same intent as prior, is_followup: true, filter country: SG
Q: "break that down by country" — same intent as prior, is_followup: true
Q: "what about the ones who left?" — intent: cross_check, is_followup: true

Q: "compare countries" — intent: country_compare (not headcount — user wants a comparison table)
Q: "how many in each country?" — intent: headcount (count, not comparison metric)
Q: "attrition vs headcount" — intent: country_compare (comparing two metrics)
Q: "who resigned?" — intent: attrition (not cross_check — no payout concern mentioned)
Q: "non-active with payouts" — intent: cross_check (specific: leavers who still received pay)

Q: "anomaly" / "anything unusual?" — intent: anomaly (needs PMGM vs payout check)
Q: "mismatch in ratings and pay" — intent: anomaly
Q: "who got paid despite bad ratings?" — intent: anomaly, filter needs_fresh_join: true
Q: "are there any errors in the data?" — intent: anomaly (interpret as data quality check)

Q: "qualifier" / "who's blocked?" — intent: qualifier
Q: "why did someone not get paid?" — intent: qualifier (most likely reason is qualifier failure)
Q: "attendance issues?" — intent: proration
Q: "partial payout" — intent: proration (attendance deduction)

Q: "compare projects" — intent: project_compare (not country_compare — grouping by project not country)
Q: "which project has the highest payout?" — intent: project_compare
Q: "compare kpi between project a and project b" — intent: project_compare
Q: "payout by project" — intent: project_compare

Q: "show me the trend" / "over the last 6 months" — intent: kpi_trend (time-series, not attainment)
Q: "payout history for employee" — intent: kpi_trend (historical lookup per person)
Q: "kpi trend over last year" — intent: kpi_trend
Q: "proration trend across cycles" — intent: kpi_trend

Q: "compare employees with 1 year vs 3 years tenure" — intent: tenure_compare
Q: "incentive by tenure band" — intent: tenure_compare
Q: "do longer tenured employees earn more?" — intent: tenure_compare

Q: "who hasn't had kpi uploaded?" — intent: missing_kpi (not employee_list — looking for gaps)
Q: "which employees are on a scheme but have no kpi this cycle?" — intent: missing_kpi
Q: "kpi not submitted" — intent: missing_kpi

Q: "any kpi adjustments?" / "pending adjustments" — intent: adjustment
Q: "was employee's kpi adjusted?" — intent: adjustment
Q: "approval status of adjustment" — intent: adjustment
Q: "qualifying status changes" — intent: adjustment

Q: "what tiers does the scheme have?" — intent: scheme_config (not attainment)
Q: "kpi weightage for scheme" — intent: scheme_config
Q: "has employee acknowledged their scheme?" — intent: scheme_config
Q: "when does the scheme expire?" — intent: scheme_config

Q: "when did employee last log in?" — intent: login
Q: "last login for employee" — intent: login

Q: "any announcements this cycle?" — intent: announcement
Q: "were there notices active during cycle?" — intent: announcement

=== END DISAMBIGUATION EXAMPLES ==="""


def _get_secret(key: str) -> str:
    try:
        return st.secrets[key]
    except Exception:
        return os.environ.get(key, "")


def _regex_fallback(question: str, ctx: dict = None) -> dict:
    """Pure-regex fallback — same output schema as the AI router.
    Uses conversation context to improve follow-up detection."""
    from ai_engine import detect_intent
    intent = detect_intent(question)
    q      = question.lower()

    # Enhanced follow-up detection — also checks conversation context
    followup_patterns = r"\b(they|those|these|them|same|among|of those|of them|their|" \
                        r"those employees|the above|that group|those people|" \
                        r"break that|break it|drill down|zoom in|and the|what about)\b"
    is_followup = bool(re.search(followup_patterns, q))

    # Short question in an active conversation is likely a follow-up
    if not is_followup and ctx and ctx.get("topic_intent") and len(question.split()) <= 4:
        is_followup = True

    needs_fresh = is_followup or intent in ("cross_join", "free_form")

    # Inherit active filters from conversation context
    filters = {
        "country": None, "scheme": None, "cycle": None,
        "threshold": None, "employee_id": None, "status": None,
    }
    if ctx and ctx.get("active_filters"):
        for k, v in ctx["active_filters"].items():
            if v is not None and k in filters:
                filters[k] = v

    return {
        "intent":             intent,
        "needs_fresh_join":   needs_fresh,
        "needs_flash_reward": intent not in ("employee_list", "headcount", "attrition", "pmgm"),
        "needs_flash_home":   intent not in ("attainment", "qualifier", "proration", "cycle_summary"),
        "is_followup":        is_followup,
        "filters":            filters,
        "reasoning":          "Regex fallback — DeepSeek router unavailable.",
    }


def _safe_text(s: str, maxlen: int = 200) -> str:
    """Strip characters that can break f-string prompt injection."""
    if not s:
        return ""
    # Remove control chars, curly braces (break f-strings), backticks
    import unicodedata
    cleaned = "".join(
        c for c in s
        if unicodedata.category(c)[0] != "C"   # no control characters
        and c not in '`{}'
    )
    return cleaned[:maxlen]
    """
    Call DeepSeek to classify intent and data needs.
    Injects conversation_state context for multi-turn coherence.
    Falls back to regex on any failure.
    """
    from semantic import normalise, hint_intent
    from conversation_state import get_ctx, ctx_for_router

    ctx        = get_ctx()
    normalised = normalise(question)

    # Check intent_hints first — these override the AI router for known phrases
    forced_intent = hint_intent(normalised)

    api_key = _get_secret("DEEPSEEK_API_KEY")
    if not api_key:
        result = _regex_fallback(normalised, ctx)
        if forced_intent:
            result["intent"]    = forced_intent
            result["reasoning"] = f"Intent hint matched: '{question}' → {forced_intent}"
        return result

    # ── Build conversation snippet (last 6 messages) ─────────────────────────
    history_snippet = ""
    if history:
        recent = history[-6:]
        history_snippet = "\n".join(
            f"{m['role'].upper()}: {_safe_text(m['content'], 200)}" for m in recent
        )

    last_cols_str = ""
    if last_df_columns:
        last_cols_str = f"\nPrevious result columns: {', '.join(last_df_columns)}"

    # ── Inject conversation context ───────────────────────────────────────────
    conv_context = ctx_for_router()

    safe_q         = _safe_text(question, 400)
    safe_normalised = _safe_text(normalised, 400)

    user_prompt = (
        "=== CONVERSATION CONTEXT ===\n"
        + conv_context
        + "\n\n=== RECENT MESSAGES ===\n"
        + (history_snippet or "(no prior messages)")
        + last_cols_str
        + "\n\n=== CURRENT QUESTION ===\n"
        + f"Normalised: {safe_normalised}\n"
        + f"Original:   {safe_q}\n\n"
        + f"Known intents: {', '.join(KNOWN_INTENTS)}\n\n"
        + "Respond with ONLY the JSON object."
    )

    try:
        r = requests.post(
            DEEPSEEK_API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type":  "application/json",
            },
            json={
                "model":      ROUTER_MODEL,
                "max_tokens": 350,
                "messages": [
                    {"role": "system", "content": ROUTER_SYSTEM},
                    {"role": "user",   "content": user_prompt},
                ],
            },
            timeout=10,
        )
        r.raise_for_status()
        raw = r.json()["choices"][0]["message"]["content"].strip()
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$",       "", raw)
        raw = raw.strip()

        data = json.loads(raw)

        # forced_intent from semantic hints overrides AI router
        if forced_intent:
            data["intent"]    = forced_intent
            data["reasoning"] = f"Semantic hint override: {data.get('reasoning', '')}"

        if data.get("intent") not in KNOWN_INTENTS:
            data["intent"] = "free_form"

        # Inherit active filters from conversation context for follow-ups
        if data.get("is_followup") and ctx.get("active_filters"):
            for k, v in ctx["active_filters"].items():
                if v is not None:
                    data["filters"].setdefault(k, None)
                    if data["filters"].get(k) is None:
                        data["filters"][k] = v

        data.setdefault("needs_fresh_join",   False)
        data.setdefault("needs_flash_reward", True)
        data.setdefault("needs_flash_home",   False)
        data.setdefault("is_followup",        False)
        data.setdefault("filters", {})
        for k in ["country", "scheme", "cycle", "threshold", "employee_id", "status"]:
            data["filters"].setdefault(k, None)
        data.setdefault("reasoning", "")

        return data

    except Exception as e:
        result = _regex_fallback(normalised, ctx)
        if forced_intent:
            result["intent"]    = forced_intent
            result["reasoning"] = f"Semantic hint: {forced_intent} (router error: {str(e)[:60]})"
        else:
            result["reasoning"] = f"Regex fallback (router error: {str(e)[:80]})"
        return result
