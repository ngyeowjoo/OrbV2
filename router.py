"""
router.py  —  AI-powered intent router for Orb v2

Uses DeepSeek V4 Flash (fast, cheap) as the routing model.
Falls back to regex detection if the call fails.
"""
import json
import os
import re
import unicodedata

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
    "project_compare",  # compare projects on any metric
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
  MetricPayout, TierAchieved, QualifierFailed, ProrationType, ProrFactor, TotalCyclePayout
- Flash Home: EmployeeID, EmployeeName, EmployeeStatus, JoinDate, LastDate,
  Country, Project, PMGMRating, JobTitle, EmployeeGrade, EmployeeDepartment, SupervisorID

Respond with ONLY a valid JSON object. No explanation, no markdown, no code fences.

Schema:
{"intent":"<one of the known intents>","needs_fresh_join":true,"needs_flash_reward":true,"needs_flash_home":false,"is_followup":false,"filters":{"country":null,"scheme":null,"cycle":null,"threshold":null,"employee_id":null,"status":null},"reasoning":"one short sentence"}

Set needs_fresh_join = true when:
- Question asks to identify, name, or find specific employees
- Question asks for join date, last date, job title, grade, department, supervisor
- Question is a follow-up needing HR data not visible in the previous result
- Prior result may lack full employee profile

Set is_followup = true when:
- Question uses: they, those, these, them, same, among, of those, their, that employee, that group
- Question is too short to stand alone (e.g. "when did they join?", "what is their grade?")
- Conversation context shows an established topic and this drills deeper

FILTER EXTRACTION — do not confuse "grouped by X" with "filtered to X":
- "break that down by country" / "a breakdown by country" / "across countries" / "by country"
  means GROUP the result by country — the user wants to see every country dimensioned out,
  not narrowed to one. Do NOT set filters.country in this case (leave it null), and prefer
  intent: country_compare over keeping the prior intent, since only country_compare actually
  produces a per-country breakdown. The same logic applies to "by project" -> project_compare
  and "by tenure" -> tenure_compare.
- Only set filters.country when the user names a SPECIFIC country/region they want narrowed
  to (a country name, code, or "just Singapore", "only for Malaysia", etc.) — never guess or
  default to a country that merely appeared somewhere in the prior answer's text.

=== DISAMBIGUATION EXAMPLES ===

Q: "when did that employee join?" — intent: cross_join, is_followup: true, needs_fresh_join: true
Q: "what is their job title?" — intent: cross_join, is_followup: true, needs_fresh_join: true
Q: "what grade is that employee?" — intent: cross_join, is_followup: true, needs_fresh_join: true
Q: "who is their supervisor?" — intent: cross_join, is_followup: true, needs_fresh_join: true
Q: "when did they leave?" — intent: cross_join, is_followup: true, needs_fresh_join: true

Q: "who earns the most?" — intent: ranking
Q: "list all employees" — intent: employee_list
Q: "show me attainment" — intent: attainment
Q: "who hit max payout?" — intent: attainment
Q: "top 10 by payout" — intent: ranking
Q: "who are the underperformers?" — intent: underperformance

Q: "who are they?" (after underperformance result) — intent: cross_join, is_followup: true
Q: "name those employees" (after any result) — intent: cross_join, is_followup: true
Q: "show me in SG" (after headcount result) — same intent as prior, is_followup: true, filter country: SG
Q: "break that down by country" (after any result) — intent: country_compare, is_followup: true, filter country: null
Q: "need a breakdown by country or by specific employee names" (after a pmgm/headcount/etc result) — intent: cross_join, is_followup: true, filter country: null — the user wants individual records to work from, not a single country filtered out
Q: "what about the ones who left?" — intent: cross_check, is_followup: true

Q: "compare countries" — intent: country_compare
Q: "how many in each country?" — intent: headcount
Q: "who resigned?" — intent: attrition
Q: "non-active with payouts" — intent: cross_check

Q: "anomaly" / "anything unusual?" — intent: anomaly
Q: "mismatch in ratings and pay" — intent: anomaly

Q: "qualifier" / "who is blocked?" — intent: qualifier
Q: "attendance issues?" — intent: proration
Q: "partial payout" — intent: proration

Q: "compare projects" — intent: project_compare
Q: "payout by project" — intent: project_compare

Q: "show me the trend" — intent: kpi_trend
Q: "payout history for employee" — intent: kpi_trend

Q: "compare employees with 1 year vs 3 years tenure" — intent: tenure_compare
Q: "incentive by tenure band" — intent: tenure_compare
Q: "can I see performance by employee tenure?" — intent: tenure_compare
Q: "do longer-tenured employees earn more?" — intent: tenure_compare

Q: "who has not had kpi uploaded?" — intent: missing_kpi
Q: "kpi not submitted" — intent: missing_kpi

Q: "any kpi adjustments?" — intent: adjustment
Q: "pending adjustments" — intent: adjustment

Q: "what tiers does the scheme have?" — intent: scheme_config
Q: "kpi weightage for scheme" — intent: scheme_config
Q: "has employee acknowledged their scheme?" — intent: scheme_config

Q: "when did employee last log in?" — intent: login
Q: "any announcements this cycle?" — intent: announcement

=== END DISAMBIGUATION EXAMPLES ==="""


def _get_secret(key: str) -> str:
    try:
        return st.secrets[key]
    except Exception:
        return os.environ.get(key, "")


def _safe(s: str, maxlen: int = 300) -> str:
    """Remove characters that corrupt JSON prompts: control chars, curly braces, backticks."""
    if not s:
        return ""
    return "".join(
        c for c in str(s)
        if unicodedata.category(c)[0] != "C" and c not in "`{}"
    )[:maxlen]


def _recompute_derived_flags(result: dict, intent: str, is_followup: bool) -> None:
    """
    Recompute needs_fresh_join / needs_flash_reward / needs_flash_home for a
    given intent. Call this any time an intent is forced/overridden after the
    original flags were computed for a *different* intent, so they don't go
    stale (e.g. a semantic hint switching "attainment" -> "login" but leaving
    needs_flash_home=False, which is wrong for login).
    """
    result["needs_fresh_join"]   = is_followup or intent in ("cross_join", "free_form")
    result["needs_flash_reward"] = intent not in ("employee_list", "headcount", "attrition", "pmgm")
    result["needs_flash_home"]   = intent not in ("attainment", "qualifier", "proration", "cycle_summary")


def _regex_fallback(question: str, ctx: dict = None) -> dict:
    """Pure-regex fallback — same output schema as the AI router."""
    from ai_engine import detect_intent
    intent = detect_intent(question)
    q      = question.lower()

    followup_patterns = (
        r"\b(they|those|these|them|same|among|of those|of them|their|"
        r"that employee|those employees|the above|that group|those people|"
        r"break that|break it|drill down|zoom in|and the|what about|"
        r"when did|what is their|what was their|who is their)\b"
    )
    is_followup = bool(re.search(followup_patterns, q))

    # Short question in an active conversation is likely a follow-up
    if not is_followup and ctx and ctx.get("topic_intent") and len(question.split()) <= 6:
        is_followup = True

    needs_fresh = is_followup or intent in ("cross_join", "free_form")

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


def route(question: str, history: list, last_df_columns: list = None) -> dict:
    """
    Call DeepSeek to classify intent and data needs.
    Falls back to regex on any failure.
    """
    from semantic import normalise, hint_intent
    from conversation_state import get_ctx, ctx_for_router

    ctx        = get_ctx()
    normalised = normalise(question)

    # Semantic hint overrides — fire before API call
    forced_intent = hint_intent(normalised)

    api_key = _get_secret("DEEPSEEK_API_KEY")
    if not api_key:
        result = _regex_fallback(normalised, ctx)
        if forced_intent and forced_intent != result["intent"]:
            result["intent"]    = forced_intent
            result["reasoning"] = f"Intent hint: {forced_intent}"
            _recompute_derived_flags(result, forced_intent, result["is_followup"])
        return result

    # ── Build prompt as plain string concatenation — no f-strings with user content ──
    history_lines = []
    if history:
        for m in history[-6:]:
            role    = m.get("role", "user").upper()
            content = _safe(m.get("content", ""), 200)
            history_lines.append(role + ": " + content)

    cols_line = ""
    if last_df_columns:
        cols_line = "\nPrevious result columns: " + ", ".join(_safe(c, 40) for c in last_df_columns)

    conv_ctx = _safe(ctx_for_router(), 800)
    safe_q   = _safe(question, 400)
    safe_n   = _safe(normalised, 400)

    # A semantic-layer keyword match is a useful signal but not proof — a naive
    # substring/phrase match can misfire on unrelated questions. Feed it to the
    # model as context and let it weigh it against the full conversation,
    # rather than silently overriding a confident classification below.
    hint_line = ""
    if forced_intent:
        hint_line = (
            "\n\nBUSINESS-RULE HINT: a configured keyword rule suggests this question "
            "matches intent '" + forced_intent + "'. Treat it as a strong signal, but if "
            "the conversation context or question clearly points to a different intent, "
            "use your own judgement instead."
        )

    user_prompt = (
        "CONVERSATION CONTEXT:\n" + (conv_ctx or "(none)") +
        "\n\nRECENT MESSAGES:\n" + ("\n".join(history_lines) or "(none)") +
        cols_line +
        "\n\nCURRENT QUESTION: " + safe_q +
        "\nNORMALISED: " + safe_n +
        "\n\nKNOWN INTENTS: " + ", ".join(KNOWN_INTENTS) +
        hint_line +
        "\n\nRespond with ONLY the JSON object."
    )

    try:
        r = requests.post(
            DEEPSEEK_API_URL,
            headers={
                "Authorization": "Bearer " + api_key,
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
            timeout=12,
        )
        r.raise_for_status()

        raw = r.json()["choices"][0]["message"]["content"].strip()

        # Strip markdown fences if model wraps in ```json ... ```
        raw = re.sub(r"^```[a-zA-Z]*\s*", "", raw)
        raw = re.sub(r"\s*```$",          "", raw)
        raw = raw.strip()

        # Sometimes model returns trailing comma before } — fix it
        raw = re.sub(r",\s*([}\]])", r"\1", raw)

        data = json.loads(raw)

        # Guard unknown intents
        if data.get("intent") not in KNOWN_INTENTS:
            data["intent"] = "free_form"

        # The hint was already shown to the model above. Only step in here if
        # the model still couldn't commit (fell back to free_form) — in that
        # case a matched business-rule keyword is more useful than free_form.
        if forced_intent and data.get("intent") == "free_form":
            data["intent"]    = forced_intent
            data["reasoning"] = "Semantic hint (model was unsure): " + forced_intent
            _recompute_derived_flags(data, forced_intent, data.get("is_followup", False))

        # Inherit active filters from context on follow-ups
        if data.get("is_followup") and ctx.get("active_filters"):
            for k, v in ctx["active_filters"].items():
                if v is not None:
                    data.setdefault("filters", {})
                    if data["filters"].get(k) is None:
                        data["filters"][k] = v

        # Ensure all expected keys exist
        data.setdefault("needs_fresh_join",   False)
        data.setdefault("needs_flash_reward", True)
        data.setdefault("needs_flash_home",   False)
        data.setdefault("is_followup",        False)
        data.setdefault("filters",            {})
        for k in ["country", "scheme", "cycle", "threshold", "employee_id", "status"]:
            data["filters"].setdefault(k, None)
        data.setdefault("reasoning", "")

        return data

    except Exception as e:
        result = _regex_fallback(normalised, ctx)
        if forced_intent and forced_intent != result["intent"]:
            result["intent"]    = forced_intent
            result["reasoning"] = "Hint: " + forced_intent + " (fallback: " + str(e)[:60] + ")"
            _recompute_derived_flags(result, forced_intent, result["is_followup"])
        else:
            result["reasoning"] = "Regex fallback (router error: " + str(e)[:80] + ")"
        return result
