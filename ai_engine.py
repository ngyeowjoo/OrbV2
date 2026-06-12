"""
ai_engine.py  —  Intent detection, data retrieval, multi-model reasoning
Supports: Claude (Anthropic) and DeepSeek (OpenAI-compatible)
"""
import os, re
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import requests

# ── MODEL REGISTRY ────────────────────────────────────────────────────────────
MODELS = {
    "Claude Sonnet (Default)": {
        "provider":    "anthropic",
        "model_id":    "claude-sonnet-4-20250514",
        "description": "Best reasoning & narrative",
        "tag":         "Anthropic",
        "tag_color":   "#D97706",
    },
    "DeepSeek V4 Pro": {
        "provider":    "deepseek",
        "model_id":    "deepseek-v4-pro",
        "description": "Strong reasoning, lower cost",
        "tag":         "DeepSeek",
        "tag_color":   "#2563EB",
    },
    "DeepSeek V4 Flash": {
        "provider":    "deepseek",
        "model_id":    "deepseek-v4-flash",
        "description": "Fast & cheapest",
        "tag":         "DeepSeek",
        "tag_color":   "#2563EB",
    },
}
MODEL_NAMES   = list(MODELS.keys())
DEFAULT_MODEL = MODEL_NAMES[0]

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
DEEPSEEK_API_URL  = "https://api.deepseek.com/chat/completions"

C_AMBER = "#F9A602"
C_WARN  = "#E05C00"
C_BAD   = "#C0392B"
C_GOOD  = "#27AE60"
PALETTE = [C_AMBER, "#FFD166", C_WARN, C_BAD, "#9B59B6", C_GOOD, "#3498DB"]

PLOT_THEME = dict(
    paper_bgcolor="#111111", plot_bgcolor="#111111",
    font=dict(color="#CCCCCC", family="DM Mono, monospace", size=11),
    margin=dict(l=32, r=16, t=40, b=32),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
)

def _fmt_axes(fig):
    fig.update_xaxes(gridcolor="#222", zeroline=False)
    fig.update_yaxes(gridcolor="#222", zeroline=False)

# ── INTENT CLASSIFIER ─────────────────────────────────────────────────────────
INTENT_PATTERNS = {
    "attainment":        r"(hit|reach|attain|max payout|maximum|full pay)",
    "underperformance":  r"(miss|under.?perform|below target|not hit|consistent)",
    "qualifier":         r"(qualifier|blocked|fail.*qual|qual.*fail)",
    "proration":         r"(prorat|attendance|absent|proration)",
    "anomaly":           r"(anomaly|mismatch|high.*rating.*low|low.*rating.*high|pmgm.*payout|payout.*pmgm)",
    "cross_check":       r"(non.?active|inactive|left.*payout|payout.*left|leaver|exit)",
    "new_joiner":        r"(new joiner|first cycle|recently joined|new hire)",
    "headcount":         r"(headcount|how many|count|active.*employee|employee.*active|workforce size)",
    "attrition":         r"(attrition|left|resign|turnover|leavers)",
    "pmgm":              r"(pmgm|performance rating|rating distribution|appraisal)",
    "cycle_summary":     r"(summary|overview|this cycle|cycle summary|brief me)",
    "country_compare":   r"(compare.*country|country.*compare|vs.*country|country.*vs|\bvs\b.*[A-Z]{2}|compare.*(sg|my|ph|th|id))",
    "free_form":         r".*",
}

def detect_intent(question: str) -> str:
    q = question.lower()
    for intent, pattern in INTENT_PATTERNS.items():
        if re.search(pattern, q):
            return intent
    return "free_form"

# ── DATA RETRIEVAL PER INTENT ─────────────────────────────────────────────────
def retrieve_data(intent: str, countries: list, question: str):
    """Returns (dataframe, description_for_ai)"""
    from data import (attainment_summary, underperformer_summary,
                      qualifier_summary, proration_summary, anomaly_summary,
                      get_flash_home, get_flash_reward, get_joined)

    scope = "Global" if "ALL" in countries else ", ".join(countries)

    if intent == "attainment":
        df, cycle = attainment_summary(countries)
        desc = f"Incentive attainment data for cycle {cycle}, scope: {scope}.\n{df.to_string(index=False)}"
        return df, desc

    elif intent == "underperformance":
        df = underperformer_summary(countries)
        fh  = get_flash_home(countries)[["EmployeeID", "Country", "Project"]]
        df  = df.merge(fh, on="EmployeeID", how="left")
        desc = f"Employees with >=3 consecutive cycles below target, scope: {scope}.\n{df.to_string(index=False)}"
        return df, desc

    elif intent == "qualifier":
        df  = qualifier_summary(countries)
        fh  = get_flash_home(countries)[["EmployeeID", "Country"]]
        df  = df.merge(fh, on="EmployeeID", how="left")
        desc = f"Qualifier failure data, scope: {scope}.\n{df.to_string(index=False)}"
        return df, desc

    elif intent == "proration":
        df  = proration_summary(countries)
        desc = f"Attendance proration data, scope: {scope}.\nAffected: {df['EmployeeID'].nunique()} employees, total payout impact: {df['PayoutLost'].sum():,.2f}\n{df[['EmployeeID','Cycle','ProrFactor','PayoutLost']].head(40).to_string(index=False)}"
        return df, desc

    elif intent == "anomaly":
        high_low, low_high, cycle = anomaly_summary(countries)
        df_combined = pd.concat([
            high_low.assign(AnomalyType="High PMGM / Low Payout"),
            low_high.assign(AnomalyType="Low PMGM / High Payout"),
        ])
        desc = f"Performance vs payout anomaly for cycle {cycle}, scope: {scope}.\n{df_combined.to_string(index=False)}"
        return df_combined, desc

    elif intent == "cross_check":
        joined = get_joined(countries)
        fr_latest = joined[joined["Cycle"] == joined["Cycle"].max()]
        non_active_paid = fr_latest[
            (fr_latest["EmployeeStatus"] == "Non-Active") &
            (fr_latest["TotalCyclePayout"] > 0)
        ][["EmployeeID","Country","LastDate","TotalCyclePayout","Cycle"]].drop_duplicates("EmployeeID")
        desc = f"Non-active employees with payouts in latest cycle, scope: {scope}.\n{non_active_paid.to_string(index=False)}"
        return non_active_paid, desc

    elif intent == "new_joiner":
        fh  = get_flash_home(countries)
        fr  = get_flash_reward(countries)
        cutoff = pd.Timestamp.today() - pd.DateOffset(months=6)
        new = fh[(fh["JoinDate"] >= cutoff) & (fh["EmployeeStatus"] == "Active")]
        latest = fr["Cycle"].max()
        fr_latest = fr[fr["Cycle"] == latest].drop_duplicates("EmployeeID")
        df = new.merge(fr_latest[["EmployeeID","Scheme","TotalCyclePayout","ProrFactor"]], on="EmployeeID", how="left")
        desc = f"New joiners (last 6 months) on incentive, scope: {scope}.\n{df[['EmployeeID','JoinDate','Country','Scheme','TotalCyclePayout','ProrFactor']].to_string(index=False)}"
        return df, desc

    elif intent == "headcount":
        fh   = get_flash_home(countries)
        summ = fh.groupby(["Country","EmployeeStatus"]).size().reset_index(name="Count")
        desc = f"Headcount by country and status, scope: {scope}.\n{summ.to_string(index=False)}"
        return summ, desc

    elif intent == "attrition":
        fh = get_flash_home(countries)
        leavers = fh[fh["EmployeeStatus"] == "Non-Active"].copy()
        leavers["YearLeft"] = leavers["LastDate"].dt.year
        summ = leavers.groupby(["Country","YearLeft"]).size().reset_index(name="Count")
        desc = f"Attrition data, scope: {scope}.\n{summ.to_string(index=False)}"
        return summ, desc

    elif intent == "pmgm":
        fh   = get_flash_home(countries)
        dist = fh.groupby(["PMGMRating","Country"]).size().reset_index(name="Count")
        desc = f"PMGM rating distribution, scope: {scope}.\n{dist.to_string(index=False)}"
        return dist, desc

    elif intent == "cycle_summary":
        fr = get_flash_reward(countries)
        latest = fr["Cycle"].max()
        cyc = fr[fr["Cycle"] == latest].drop_duplicates(["EmployeeID"])
        total_pay   = cyc["TotalCyclePayout"].sum()
        avg_pct     = (cyc["TotalCyclePayout"] / cyc["SchemeMaxPayout"]).mean()
        hit_max     = (cyc["TotalCyclePayout"] >= cyc["SchemeMaxPayout"] * 0.999).sum()
        qual_fail   = fr[(fr["Cycle"] == latest) & (fr["QualifierFailed"] != "")]["EmployeeID"].nunique()
        prorated    = fr[(fr["Cycle"] == latest) & (fr["ProrFactor"] < 1.0)]["EmployeeID"].nunique()
        summary_str = (
            f"Cycle: {latest}, Scope: {scope}\n"
            f"Total eligible employees: {len(cyc)}\n"
            f"Total payout: {total_pay:,.2f}\n"
            f"Average payout as % of max: {avg_pct:.1%}\n"
            f"Hit max payout: {hit_max} ({hit_max/len(cyc):.1%})\n"
            f"Qualifier failures: {qual_fail}\n"
            f"Prorated for attendance: {prorated}\n"
        )
        return cyc[["EmployeeID","Scheme","TotalCyclePayout","SchemeMaxPayout","ProrFactor"]].head(50), summary_str

    elif intent == "country_compare":
        fr   = get_flash_reward(countries)
        fh   = get_flash_home(countries)
        latest = fr["Cycle"].max()
        cyc  = fr[fr["Cycle"] == latest].drop_duplicates("EmployeeID")
        cyc  = cyc.merge(fh[["EmployeeID","Country"]], on="EmployeeID", how="left")
        comp = cyc.groupby("Country").agg(
            Employees=("EmployeeID","count"),
            AvgPayout=("TotalCyclePayout","mean"),
            TotalPayout=("TotalCyclePayout","sum"),
            HitMax=("TotalCyclePayout", lambda x: (x >= cyc.loc[x.index,"SchemeMaxPayout"] * 0.999).sum()),
        ).reset_index()
        comp["HitMaxPct"] = comp["HitMax"] / comp["Employees"]
        desc = f"Country comparison for cycle {latest}, scope: {scope}.\n{comp.to_string(index=False)}"
        return comp, desc

    else:  # free_form
        fh = get_flash_home(countries)
        fr = get_flash_reward(countries)
        latest = fr["Cycle"].max()
        cyc = fr[fr["Cycle"] == latest].drop_duplicates("EmployeeID")
        desc = (
            f"Scope: {scope}. Latest cycle: {latest}.\n"
            f"Flash Home - {len(fh)} employees, countries: {fh['Country'].unique().tolist()}\n"
            f"Flash Reward (latest cycle) - {len(cyc)} employees\n"
            f"Schemes: {cyc['Scheme'].unique().tolist()}\n"
            f"Payout range: {cyc['TotalCyclePayout'].min():.0f} - {cyc['TotalCyclePayout'].max():.0f}\n"
            f"Qualifier failures: {(fr[(fr['Cycle']==latest) & (fr['QualifierFailed']!='')]['EmployeeID'].nunique())}\n"
            f"Sample Flash Home:\n{fh.head(10).to_string(index=False)}\n"
            f"Sample Flash Reward:\n{cyc.head(10).to_string(index=False)}\n"
        )
        return None, desc

# ── CHART BUILDER ─────────────────────────────────────────────────────────────
def build_chart(intent: str, df):
    if df is None or df.empty:
        return None
    try:
        if intent == "attainment":
            by_scheme = df.groupby("Scheme")["HitMax"].agg(["sum","count"]).reset_index()
            by_scheme["HitMaxPct"] = by_scheme["sum"] / by_scheme["count"] * 100
            fig = px.bar(by_scheme, x="Scheme", y="HitMaxPct",
                         title="% Hitting Max Payout by Scheme",
                         color="Scheme", color_discrete_sequence=PALETTE)
            fig.update_layout(**PLOT_THEME, yaxis_title="% Hit Max"); _fmt_axes(fig)
            return fig

        elif intent == "underperformance":
            if df.empty: return None
            fig = px.bar(df.sort_values("MaxConsecutiveMisses", ascending=False).head(20),
                         x="EmployeeID", y="MaxConsecutiveMisses",
                         color="Country" if "Country" in df.columns else None,
                         title="Consecutive Cycles Below Target (Top 20)",
                         color_discrete_sequence=PALETTE)
            fig.update_layout(**PLOT_THEME); _fmt_axes(fig)
            return fig

        elif intent == "qualifier":
            top = df.groupby("QualifierFailed")["TimesFailed"].sum().reset_index()
            fig = px.bar(top, x="QualifierFailed", y="TimesFailed",
                         title="Qualifier Failure Frequency",
                         color_discrete_sequence=[C_WARN])
            fig.update_layout(**PLOT_THEME); _fmt_axes(fig)
            return fig

        elif intent == "proration":
            monthly = df.groupby("Cycle")["EmployeeID"].nunique().reset_index(name="ProatedCount")
            fig = px.line(monthly, x="Cycle", y="ProatedCount",
                          title="Prorated Employees per Cycle",
                          markers=True, color_discrete_sequence=[C_AMBER])
            fig.update_layout(**PLOT_THEME); _fmt_axes(fig)
            return fig

        elif intent == "anomaly":
            if df.empty: return None
            fig = px.scatter(df, x="PayoutPct", y="PMGMRating",
                             color="AnomalyType",
                             title="Performance vs Payout Anomaly",
                             color_discrete_sequence=[C_BAD, C_WARN],
                             hover_data=["EmployeeID"])
            fig.update_layout(**PLOT_THEME); _fmt_axes(fig)
            return fig

        elif intent == "headcount":
            fig = px.bar(df, x="Country", y="Count", color="EmployeeStatus",
                         title="Headcount by Country & Status",
                         color_discrete_sequence=[C_AMBER, C_WARN])
            fig.update_layout(**PLOT_THEME, barmode="group"); _fmt_axes(fig)
            return fig

        elif intent == "attrition":
            fig = px.bar(df, x="YearLeft", y="Count", color="Country",
                         title="Attrition by Year & Country",
                         color_discrete_sequence=PALETTE)
            fig.update_layout(**PLOT_THEME); _fmt_axes(fig)
            return fig

        elif intent == "pmgm":
            order = ["Exceptional","Exceeds Expectations","Meets Expectations",
                     "Below Expectations","Unsatisfactory"]
            fig = px.bar(df, x="PMGMRating", y="Count", color="Country",
                         title="PMGM Rating Distribution",
                         category_orders={"PMGMRating": order},
                         color_discrete_sequence=PALETTE)
            fig.update_layout(**PLOT_THEME); _fmt_axes(fig)
            return fig

        elif intent == "country_compare":
            fig = px.bar(df, x="Country", y="HitMaxPct",
                         title="% Hitting Max Payout by Country",
                         color="Country", color_discrete_sequence=PALETTE)
            fig.update_layout(**PLOT_THEME, yaxis_tickformat=".0%"); _fmt_axes(fig)
            return fig

        elif intent == "cross_check":
            if df.empty: return None
            fig = px.bar(df, x="EmployeeID", y="TotalCyclePayout",
                         title="Non-Active Employees with Payouts",
                         color_discrete_sequence=[C_BAD])
            fig.update_layout(**PLOT_THEME); _fmt_axes(fig)
            return fig

    except Exception:
        return None
    return None

# ── API CALLERS ───────────────────────────────────────────────────────────────
def _get_secret(key: str) -> str:
    """Read from st.secrets (Streamlit Cloud) with fallback to os.environ (local dev)."""
    try:
        return st.secrets[key]
    except Exception:
        return os.environ.get(key, "")


def _call_anthropic(messages: list, system: str, model_id: str) -> str:
    api_key = _get_secret("ANTHROPIC_API_KEY")
    r = requests.post(
        ANTHROPIC_API_URL,
        headers={
            "x-api-key":         api_key,
            "anthropic-version": "2023-06-01",
            "content-type":      "application/json",
        },
        json={
            "model":      model_id,
            "max_tokens": 1024,
            "system":     system,
            "messages":   messages,
        },
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["content"][0]["text"]


def _call_deepseek(messages: list, system: str, model_id: str) -> str:
    api_key = _get_secret("DEEPSEEK_API_KEY")
    # DeepSeek uses OpenAI-compatible format: system prompt goes as first message
    openai_messages = [{"role": "system", "content": system}] + messages
    r = requests.post(
        DEEPSEEK_API_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json",
        },
        json={
            "model":      model_id,
            "max_tokens": 1024,
            "messages":   openai_messages,
        },
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def call_model(messages: list, system: str, model_name: str) -> str:
    """Dispatches to the correct provider based on model_name."""
    cfg      = MODELS.get(model_name, MODELS[DEFAULT_MODEL])
    provider = cfg["provider"]
    model_id = cfg["model_id"]

    if provider == "anthropic":
        return _call_anthropic(messages, system, model_id)
    elif provider == "deepseek":
        return _call_deepseek(messages, system, model_id)
    else:
        raise ValueError(f"Unknown provider: {provider}")

# ── MAIN ENTRY POINT ──────────────────────────────────────────────────────────
def answer(question: str, history: list, user: dict, model_name: str = DEFAULT_MODEL):
    """
    Returns (text_answer, plotly_fig_or_None, dataframe_or_None)
    history: list of {role, content} dicts (prior turns)
    model_name: key from MODELS dict
    """
    countries = user["countries"]
    intent    = detect_intent(question)
    df, data_context = retrieve_data(intent, countries, question)
    chart = build_chart(intent, df)

    scope = "Global" if "ALL" in countries else ", ".join(countries)
    cfg   = MODELS.get(model_name, MODELS[DEFAULT_MODEL])

    system = f"""You are the Orb v2 AI assistant — an executive-grade workforce intelligence analyst.
You are answering a {user['role']} named {user['display_name']}.
Their data scope: {scope}.
Data sources available: Flash Reward (Incentive System) and Flash Home (HR System).
Today's date: {pd.Timestamp.today().strftime('%d %b %Y')}.
You are running on: {cfg['tag']} — {model_name}.

Rules:
- Be concise, executive-grade. Lead with the insight, not methodology.
- If data shows something concerning, flag it clearly.
- Reference specific numbers from the data provided.
- If a chart has been generated, mention it naturally ("as shown in the chart").
- Do NOT mention SQL, dataframes, or technical implementation details.
- Keep answers to 3-5 sentences for simple queries; use brief bullet points for lists.
- Always state the data scope and cycle period you are referencing.
- If the data is insufficient to answer, say so clearly and suggest what data would help.

Data context:
{data_context}
"""

    messages = history[-10:] + [{"role": "user", "content": question}]
    text = call_model(messages, system, model_name)
    return text, chart, df
