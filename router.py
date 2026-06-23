"""
router.py  —  AI-powered intent router for Orb v2

Uses DeepSeek V4 Flash (fast, cheap) as the routing model.
Falls back to regex detection if the call fails.
"""
import json, os, re
import requests
import streamlit as st

DEEPSEEK_API_URL   = "https://api.deepseek.com/chat/completions"
ROUTER_MODEL       = "deepseek-v4-flash"   # cheapest/fastest for routing

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
    "free_form",        # anything else / complex multi-part
]

ROUTER_SYSTEM = """You are a data routing assistant for a workforce analytics platform.
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

Set is_followup = true when question references prior context using words like:
they, those, these, them, same, among, of those, of them, their, those employees, the above"""


def _get_secret(key: str) -> str:
    try:
        return st.secrets[key]
    except Exception:
        return os.environ.get(key, "")


def _regex_fallback(question: str) -> dict:
    """Pure-regex fallback — same output schema as the AI router."""
    from ai_engine import detect_intent
    intent      = detect_intent(question)
    q           = question.lower()
    is_followup = bool(re.search(
        r"\b(they|those|these|them|same|among|of those|of them|their|those employees)\b", q
    ))
    needs_fresh = is_followup or intent in ("cross_join", "free_form")
    return {
        "intent":             intent,
        "needs_fresh_join":   needs_fresh,
        "needs_flash_reward": intent not in ("employee_list", "headcount", "attrition", "pmgm"),
        "needs_flash_home":   intent not in ("attainment", "qualifier", "proration", "cycle_summary"),
        "is_followup":        is_followup,
        "filters": {
            "country": None, "scheme": None, "cycle": None,
            "threshold": None, "employee_id": None, "status": None,
        },
        "reasoning": "Regex fallback — DeepSeek router unavailable.",
    }


def route(question: str, history: list, last_df_columns: list = None) -> dict:
    """
    Call DeepSeek to classify intent and data needs.
    Normalises question with semantic layer first.
    Falls back to regex instantly on any failure.
    """
    from semantic import normalise, hint_intent

    # Apply semantic normalisation — replaces synonyms before routing
    normalised = normalise(question)

    # Check intent_hints first — these override the AI router for known phrases
    forced_intent = hint_intent(normalised)

    api_key = _get_secret("DEEPSEEK_API_KEY")
    if not api_key:
        result = _regex_fallback(normalised)
        if forced_intent:
            result["intent"] = forced_intent
            result["reasoning"] = f"Intent hint matched: '{question}' → {forced_intent}"
        return result
    if not api_key:
        return _regex_fallback(question)

    # Compact conversation context
    history_snippet = ""
    if history:
        recent = history[-4:]
        history_snippet = "\n".join(
            f"{m['role'].upper()}: {m['content'][:120]}" for m in recent
        )

    last_cols_str = ""
    if last_df_columns:
        last_cols_str = f"\nPrevious result columns: {', '.join(last_df_columns)}"

    user_prompt = f"""Conversation so far:
{history_snippet or "(no prior messages)"}
{last_cols_str}

Current question (normalised): {normalised}
Original question: {question}

Known intents: {", ".join(KNOWN_INTENTS)}

Respond with ONLY the JSON object."""

    try:
        r = requests.post(
            DEEPSEEK_API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type":  "application/json",
            },
            json={
                "model":      ROUTER_MODEL,
                "max_tokens": 300,
                "messages": [
                    {"role": "system",  "content": ROUTER_SYSTEM},
                    {"role": "user",    "content": user_prompt},
                ],
            },
            timeout=8,
        )
        r.raise_for_status()
        raw = r.json()["choices"][0]["message"]["content"].strip()
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "",     raw)
        raw = raw.strip()

        data = json.loads(raw)

        # forced_intent from semantic hints overrides AI router
        if forced_intent:
            data["intent"] = forced_intent
            data["reasoning"] = f"Semantic hint override: {data.get('reasoning','')}"

        if data.get("intent") not in KNOWN_INTENTS:
            data["intent"] = "free_form"

        data.setdefault("needs_fresh_join",   False)
        data.setdefault("needs_flash_reward", True)
        data.setdefault("needs_flash_home",   False)
        data.setdefault("is_followup",        False)
        data.setdefault("filters", {
            "country": None, "scheme": None, "cycle": None,
            "threshold": None, "employee_id": None, "status": None,
        })
        for k in ["country","scheme","cycle","threshold","employee_id","status"]:
            data["filters"].setdefault(k, None)
        data.setdefault("reasoning", "")
        return data

    except Exception as e:
        result = _regex_fallback(normalised)
        if forced_intent:
            result["intent"] = forced_intent
            result["reasoning"] = f"Semantic hint: {forced_intent} (router error: {str(e)[:60]})"
        else:
            result["reasoning"] = f"Regex fallback (router error: {str(e)[:80]})"
        return result
