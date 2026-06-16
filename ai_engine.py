"""
ai_engine.py  —  Intent detection, data retrieval, streaming multi-model reasoning
Supports: Claude (Anthropic) and DeepSeek (OpenAI-compatible)
"""
import os, re
import pandas as pd
import plotly.express as px
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

C_AMBER = "#D97706"
C_WARN  = "#E05C00"
C_BAD   = "#C0392B"
C_GOOD  = "#27AE60"
PALETTE = [C_AMBER, "#F59E0B", C_WARN, C_BAD, "#9B59B6", C_GOOD, "#3498DB"]

PLOT_THEME = dict(
    paper_bgcolor="#FFFFFF", plot_bgcolor="#FAFAFA",
    font=dict(color="#374151", family="Inter, sans-serif", size=11),
    margin=dict(l=32, r=16, t=40, b=32),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
)

def _fmt_axes(fig):
    fig.update_xaxes(gridcolor="#E5E7EB", zeroline=False, linecolor="#E5E7EB")
    fig.update_yaxes(gridcolor="#E5E7EB", zeroline=False, linecolor="#E5E7EB")

# ── INTENT CLASSIFIER ─────────────────────────────────────────────────────────
INTENT_PATTERNS = {
    "attainment":       r"(hit|reach|attain|max payout|maximum|full pay)",
    "underperformance": r"(miss|under.?perform|below target|not hit|consistent)",
    "qualifier":        r"(qualifier|blocked|fail.*qual|qual.*fail)",
    "proration":        r"(prorat|attendance|absent|proration)",
    "anomaly":          r"(anomaly|mismatch|high.*rating.*low|low.*rating.*high|pmgm.*payout|payout.*pmgm)",
    "cross_check":      r"(non.?active|inactive|left.*payout|payout.*left|leaver|exit)",
    "new_joiner":       r"(new joiner|first cycle|recently joined|new hire)",
    "employee_list":    r"(show.*name|list.*name|employee.*name|name.*employee|who are|all employee|employee list|staff list|roster|directory)",
    "headcount":        r"(headcount|how many|count|active.*employee|employee.*active|workforce size)",
    "attrition":        r"(attrition|left|resign|turnover|leavers)",
    "pmgm":             r"(pmgm|performance rating|rating distribution|appraisal)",
    "cycle_summary":    r"(summary|overview|this cycle|cycle summary|brief me)",
    "country_compare":  r"(compare.*country|country.*compare|vs.*country|country.*vs|\bvs\b.*[A-Z]{2}|compare.*(sg|my|ph|th|id))",
    "free_form":        r".*",
}

def detect_intent(question: str) -> str:
    q = question.lower()
    for intent, pattern in INTENT_PATTERNS.items():
        if re.search(pattern, q):
            return intent
    return "free_form"

# ── EMPLOYEE NAME ENRICHMENT ──────────────────────────────────────────────────
def _add_names(df, countries):
    """Merge EmployeeName from Flash Home, placed right after EmployeeID.
    Safe if EmployeeName is missing from the source (e.g. old mock_data.xlsx)."""
    if df is None or df.empty or "EmployeeID" not in df.columns:
        return df
    if "EmployeeName" in df.columns:
        return df
    try:
        from data import get_flash_home
        fh = get_flash_home(countries)
        if "EmployeeName" not in fh.columns:
            return df   # source doesn't have names yet — skip silently
        fh = fh[["EmployeeID", "EmployeeName"]]
        df = df.merge(fh, on="EmployeeID", how="left")
        cols = df.columns.tolist()
        cols.remove("EmployeeName")
        idx = cols.index("EmployeeID") + 1
        cols.insert(idx, "EmployeeName")
        return df[cols]
    except Exception:
        return df   # never crash the app over a name lookup

# ── DATA RETRIEVAL ────────────────────────────────────────────────────────────
def retrieve_data(intent: str, countries: list, question: str):
    from data import (attainment_summary, underperformer_summary,
                      qualifier_summary, proration_summary, anomaly_summary,
                      get_flash_home, get_flash_reward, get_joined)
    scope = "Global" if "ALL" in countries else ", ".join(countries)

    if intent == "attainment":
        df, cycle = attainment_summary(countries)
        df = _add_names(df, countries)
        return df, f"Incentive attainment data for cycle {cycle}, scope: {scope}.\n{df.to_string(index=False)}"

    elif intent == "underperformance":
        df = underperformer_summary(countries)
        fh = get_flash_home(countries)[["EmployeeID","Country","Project"]]
        df = df.merge(fh, on="EmployeeID", how="left")
        df = _add_names(df, countries)
        return df, f"Employees with >=3 consecutive cycles below target, scope: {scope}.\n{df.to_string(index=False)}"

    elif intent == "qualifier":
        df = qualifier_summary(countries)
        fh = get_flash_home(countries)[["EmployeeID","Country"]]
        df = df.merge(fh, on="EmployeeID", how="left")
        df = _add_names(df, countries)
        return df, f"Qualifier failure data, scope: {scope}.\n{df.to_string(index=False)}"

    elif intent == "proration":
        df = proration_summary(countries)
        df = _add_names(df, countries)
        show_cols = [c for c in ["EmployeeID","EmployeeName","Cycle","ProrFactor","PayoutLost"] if c in df.columns]
        return df, (f"Attendance proration data, scope: {scope}.\n"
                    f"Affected: {df['EmployeeID'].nunique()} employees, "
                    f"total payout impact: {df['PayoutLost'].sum():,.2f}\n"
                    f"{df[show_cols].head(40).to_string(index=False)}")

    elif intent == "anomaly":
        high_low, low_high, cycle = anomaly_summary(countries)
        df = pd.concat([
            high_low.assign(AnomalyType="High PMGM / Low Payout"),
            low_high.assign(AnomalyType="Low PMGM / High Payout"),
        ])
        df = _add_names(df, countries)
        return df, f"Performance vs payout anomaly for cycle {cycle}, scope: {scope}.\n{df.to_string(index=False)}"

    elif intent == "cross_check":
        joined = get_joined(countries)
        latest = joined["Cycle"].max()
        fr_l   = joined[joined["Cycle"] == latest]
        base_cols = ["EmployeeID","EmployeeName","Country","LastDate","TotalCyclePayout","Cycle"]
        avail_cols = [c for c in base_cols if c in fr_l.columns]
        df     = fr_l[(fr_l["EmployeeStatus"]=="Non-Active") & (fr_l["TotalCyclePayout"]>0)]                    [avail_cols]                    .drop_duplicates("EmployeeID")
        return df, f"Non-active employees with payouts in latest cycle, scope: {scope}.\n{df.to_string(index=False)}"

    elif intent == "new_joiner":
        fh = get_flash_home(countries)
        fr = get_flash_reward(countries)
        cutoff = pd.Timestamp.today() - pd.DateOffset(months=6)
        new    = fh[(fh["JoinDate"]>=cutoff) & (fh["EmployeeStatus"]=="Active")]
        latest = fr["Cycle"].max()
        fr_l   = fr[fr["Cycle"]==latest].drop_duplicates("EmployeeID")
        df     = new.merge(fr_l[["EmployeeID","Scheme","TotalCyclePayout","ProrFactor"]], on="EmployeeID", how="left")
        show_cols = [c for c in ["EmployeeID","EmployeeName","JoinDate","Country","Scheme","TotalCyclePayout","ProrFactor"] if c in df.columns]
        return df, f"New joiners (last 6 months) on incentive, scope: {scope}.\n{df[show_cols].to_string(index=False)}"

    elif intent == "employee_list":
        fh = get_flash_home(countries)
        cols = [c for c in ["EmployeeID","EmployeeName","Country","Project",
                             "EmployeeStatus","PMGMRating"] if c in fh.columns]
        df = fh[cols].copy()
        has_names = "EmployeeName" in df.columns
        return df, (f"Employee directory, scope: {scope}. Total: {len(df)} employees.\n"
                    f"{'EmployeeName is included in the data.' if has_names else 'Note: EmployeeName not available.'}\n"
                    f"{df.to_string(index=False)}")

    elif intent == "headcount":
        fh   = get_flash_home(countries)
        summ = fh.groupby(["Country","EmployeeStatus"]).size().reset_index(name="Count")
        return summ, f"Headcount by country and status, scope: {scope}.\n{summ.to_string(index=False)}"

    elif intent == "attrition":
        fh      = get_flash_home(countries)
        leavers = fh[fh["EmployeeStatus"]=="Non-Active"].copy()
        leavers["YearLeft"] = leavers["LastDate"].dt.year
        summ    = leavers.groupby(["Country","YearLeft"]).size().reset_index(name="Count")
        return summ, f"Attrition data, scope: {scope}.\n{summ.to_string(index=False)}"

    elif intent == "pmgm":
        fh   = get_flash_home(countries)
        dist = fh.groupby(["PMGMRating","Country"]).size().reset_index(name="Count")
        return dist, f"PMGM rating distribution, scope: {scope}.\n{dist.to_string(index=False)}"

    elif intent == "cycle_summary":
        fr      = get_flash_reward(countries)
        latest  = fr["Cycle"].max()
        cyc     = fr[fr["Cycle"]==latest].drop_duplicates(["EmployeeID"])
        total_p = cyc["TotalCyclePayout"].sum()
        avg_pct = (cyc["TotalCyclePayout"]/cyc["SchemeMaxPayout"]).mean()
        hit_max = (cyc["TotalCyclePayout"]>=cyc["SchemeMaxPayout"]*0.999).sum()
        q_fail  = fr[(fr["Cycle"]==latest)&(fr["QualifierFailed"]!="")]["EmployeeID"].nunique()
        prorat  = fr[(fr["Cycle"]==latest)&(fr["ProrFactor"]<1.0)]["EmployeeID"].nunique()
        desc    = (f"Cycle: {latest}, Scope: {scope}\n"
                   f"Total eligible employees: {len(cyc)}\nTotal payout: {total_p:,.2f}\n"
                   f"Average payout as % of max: {avg_pct:.1%}\n"
                   f"Hit max payout: {hit_max} ({hit_max/len(cyc):.1%})\n"
                   f"Qualifier failures: {q_fail}\nProrated for attendance: {prorat}\n")
        result_df = _add_names(cyc[["EmployeeID","Scheme","TotalCyclePayout","SchemeMaxPayout","ProrFactor"]].head(50), countries)
        return result_df, desc

    elif intent == "country_compare":
        fr     = get_flash_reward(countries)
        fh     = get_flash_home(countries)
        latest = fr["Cycle"].max()
        cyc    = fr[fr["Cycle"]==latest].drop_duplicates("EmployeeID")
        cyc    = cyc.merge(fh[["EmployeeID","Country"]], on="EmployeeID", how="left")
        comp   = cyc.groupby("Country").agg(
            Employees=("EmployeeID","count"),
            AvgPayout=("TotalCyclePayout","mean"),
            TotalPayout=("TotalCyclePayout","sum"),
            HitMax=("TotalCyclePayout", lambda x: (x>=cyc.loc[x.index,"SchemeMaxPayout"]*0.999).sum()),
        ).reset_index()
        comp["HitMaxPct"] = comp["HitMax"]/comp["Employees"]
        return comp, f"Country comparison for cycle {latest}, scope: {scope}.\n{comp.to_string(index=False)}"

    else:  # free_form
        fh     = get_flash_home(countries)
        fr     = get_flash_reward(countries)
        latest = fr["Cycle"].max()
        cyc    = fr[fr["Cycle"]==latest].drop_duplicates("EmployeeID")
        desc   = (f"Scope: {scope}. Latest cycle: {latest}.\n"
                  f"Flash Home - {len(fh)} employees, countries: {fh['Country'].unique().tolist()}\n"
                  f"Flash Reward (latest cycle) - {len(cyc)} employees\n"
                  f"Schemes: {cyc['Scheme'].unique().tolist()}\n"
                  f"Payout range: {cyc['TotalCyclePayout'].min():.0f}-{cyc['TotalCyclePayout'].max():.0f}\n"
                  f"Qualifier failures: {(fr[(fr['Cycle']==latest)&(fr['QualifierFailed']!='')]['EmployeeID'].nunique())}\n"
                  f"Sample Flash Home:\n{fh.head(10).to_string(index=False)}\n"
                  f"Sample Flash Reward:\n{cyc.head(10).to_string(index=False)}\n")
        return None, desc

# ── CHART BUILDER ─────────────────────────────────────────────────────────────
def build_chart(intent: str, df):
    if df is None or df.empty:
        return None
    try:
        if intent == "attainment":
            by_scheme = df.groupby("Scheme")["HitMax"].agg(["sum","count"]).reset_index()
            by_scheme["HitMaxPct"] = by_scheme["sum"]/by_scheme["count"]*100
            fig = px.bar(by_scheme, x="Scheme", y="HitMaxPct",
                         title="% Hitting Max Payout by Scheme",
                         color="Scheme", color_discrete_sequence=PALETTE)
            fig.update_layout(**PLOT_THEME, yaxis_title="% Hit Max"); _fmt_axes(fig); return fig

        elif intent == "underperformance":
            x_col = "EmployeeName" if "EmployeeName" in df.columns else "EmployeeID"
            fig = px.bar(df.sort_values("MaxConsecutiveMisses", ascending=False).head(20),
                         x=x_col, y="MaxConsecutiveMisses",
                         color="Country" if "Country" in df.columns else None,
                         title="Consecutive Cycles Below Target (Top 20)",
                         color_discrete_sequence=PALETTE)
            fig.update_layout(**PLOT_THEME, xaxis_title="Employee"); _fmt_axes(fig); return fig

        elif intent == "qualifier":
            top = df.groupby("QualifierFailed")["TimesFailed"].sum().reset_index()
            fig = px.bar(top, x="QualifierFailed", y="TimesFailed",
                         title="Qualifier Failure Frequency",
                         color_discrete_sequence=[C_WARN])
            fig.update_layout(**PLOT_THEME); _fmt_axes(fig); return fig

        elif intent == "proration":
            monthly = df.groupby("Cycle")["EmployeeID"].nunique().reset_index(name="ProatedCount")
            fig = px.line(monthly, x="Cycle", y="ProatedCount",
                          title="Prorated Employees per Cycle",
                          markers=True, color_discrete_sequence=[C_AMBER])
            fig.update_layout(**PLOT_THEME); _fmt_axes(fig); return fig

        elif intent == "anomaly":
            fig = px.scatter(df, x="PayoutPct", y="PMGMRating",
                             color="AnomalyType",
                             title="Performance vs Payout Anomaly",
                             color_discrete_sequence=[C_BAD, C_WARN],
                             hover_data=["EmployeeID"])
            fig.update_layout(**PLOT_THEME); _fmt_axes(fig); return fig

        elif intent == "headcount":
            fig = px.bar(df, x="Country", y="Count", color="EmployeeStatus",
                         title="Headcount by Country & Status",
                         color_discrete_sequence=[C_AMBER, C_WARN])
            fig.update_layout(**PLOT_THEME, barmode="group"); _fmt_axes(fig); return fig

        elif intent == "attrition":
            fig = px.bar(df, x="YearLeft", y="Count", color="Country",
                         title="Attrition by Year & Country",
                         color_discrete_sequence=PALETTE)
            fig.update_layout(**PLOT_THEME); _fmt_axes(fig); return fig

        elif intent == "pmgm":
            order = ["Exceptional","Exceeds Expectations","Meets Expectations",
                     "Below Expectations","Unsatisfactory"]
            fig = px.bar(df, x="PMGMRating", y="Count", color="Country",
                         title="PMGM Rating Distribution",
                         category_orders={"PMGMRating": order},
                         color_discrete_sequence=PALETTE)
            fig.update_layout(**PLOT_THEME); _fmt_axes(fig); return fig

        elif intent == "country_compare":
            fig = px.bar(df, x="Country", y="HitMaxPct",
                         title="% Hitting Max Payout by Country",
                         color="Country", color_discrete_sequence=PALETTE)
            fig.update_layout(**PLOT_THEME, yaxis_tickformat=".0%"); _fmt_axes(fig); return fig

        elif intent == "cross_check":
            if df.empty: return None
            x_col = "EmployeeName" if "EmployeeName" in df.columns else "EmployeeID"
            fig = px.bar(df, x=x_col, y="TotalCyclePayout",
                         title="Non-Active Employees with Payouts",
                         color_discrete_sequence=[C_BAD])
            fig.update_layout(**PLOT_THEME, xaxis_title="Employee"); _fmt_axes(fig); return fig

    except Exception:
        return None
    return None

# ── SECRET READER ─────────────────────────────────────────────────────────────
def _get_secret(key: str) -> str:
    try:
        return st.secrets[key]
    except Exception:
        return os.environ.get(key, "")

# ── STREAMING API CALLERS ─────────────────────────────────────────────────────
def stream_anthropic(messages: list, system: str, model_id: str):
    """Yields text chunks from Anthropic streaming API."""
    api_key = _get_secret("ANTHROPIC_API_KEY")
    with requests.post(
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
            "stream":     True,
        },
        stream=True,
        timeout=60,
    ) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8") if isinstance(line, bytes) else line
            if line.startswith("data: "):
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    import json
                    evt = json.loads(data)
                    if evt.get("type") == "content_block_delta":
                        delta = evt.get("delta", {})
                        if delta.get("type") == "text_delta":
                            yield delta.get("text", "")
                except Exception:
                    continue


def stream_deepseek(messages: list, system: str, model_id: str):
    """Yields text chunks from DeepSeek streaming API (OpenAI-compatible)."""
    import json
    api_key = _get_secret("DEEPSEEK_API_KEY")
    openai_messages = [{"role": "system", "content": system}] + messages
    with requests.post(
        DEEPSEEK_API_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json",
        },
        json={
            "model":      model_id,
            "max_tokens": 1024,
            "messages":   openai_messages,
            "stream":     True,
        },
        stream=True,
        timeout=60,
    ) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8") if isinstance(line, bytes) else line
            if line.startswith("data: "):
                data = line[6:]
                if data.strip() == "[DONE]":
                    break
                try:
                    evt   = json.loads(data)
                    delta = evt["choices"][0]["delta"]
                    chunk = delta.get("content", "")
                    if chunk:
                        yield chunk
                except Exception:
                    continue


def stream_model(messages: list, system: str, model_name: str):
    """Dispatches to the correct streaming provider."""
    cfg      = MODELS.get(model_name, MODELS[DEFAULT_MODEL])
    provider = cfg["provider"]
    model_id = cfg["model_id"]
    if provider == "anthropic":
        yield from stream_anthropic(messages, system, model_id)
    elif provider == "deepseek":
        yield from stream_deepseek(messages, system, model_id)


# ── FALLBACK: non-streaming call ──────────────────────────────────────────────
def _call_anthropic(messages, system, model_id):
    api_key = _get_secret("ANTHROPIC_API_KEY")
    r = requests.post(ANTHROPIC_API_URL,
        headers={"x-api-key": api_key, "anthropic-version": "2023-06-01",
                 "content-type": "application/json"},
        json={"model": model_id, "max_tokens": 1024, "system": system,
              "messages": messages},
        timeout=60)
    r.raise_for_status()
    return r.json()["content"][0]["text"]

def _call_deepseek(messages, system, model_id):
    api_key = _get_secret("DEEPSEEK_API_KEY")
    r = requests.post(DEEPSEEK_API_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": model_id, "max_tokens": 1024,
              "messages": [{"role": "system", "content": system}] + messages},
        timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

def call_model(messages, system, model_name):
    cfg = MODELS.get(model_name, MODELS[DEFAULT_MODEL])
    if cfg["provider"] == "anthropic":
        return _call_anthropic(messages, system, cfg["model_id"])
    return _call_deepseek(messages, system, cfg["model_id"])


# ── MAIN ENTRY POINT ──────────────────────────────────────────────────────────
def answer(question: str, history: list, user: dict,
           model_name: str = DEFAULT_MODEL, last_df=None):
    """
    Returns (stream_generator, plotly_fig_or_None, dataframe_or_None, system_prompt)
    Caller should consume the generator with st.write_stream() and capture the text.
    """
    countries = user["countries"]
    intent    = detect_intent(question)
    df, data_context = retrieve_data(intent, countries, question)
    chart = build_chart(intent, df)

    # Inject previous df for follow-up context
    if (intent == "free_form" or df is None) and last_df is not None:
        try:
            data_context += f"\n\nPrevious query result (use for follow-up reasoning):\n{last_df.to_string(index=False)}"
            if df is None:
                df = last_df
        except Exception:
            pass

    scope = "Global" if "ALL" in countries else ", ".join(countries)
    cfg   = MODELS.get(model_name, MODELS[DEFAULT_MODEL])

    # Chart generated — ensure text always accompanies it
    chart_note = ""
    if chart is not None:
        chart_note = "\n- A chart has been generated and will be shown alongside your response. Always provide a written summary — never return only a chart."

    system = f"""You are the Orb v2 AI assistant — an executive-grade workforce intelligence analyst.
You are answering a {user["role"]} named {user["display_name"]}.
Their data scope: {scope}.
Data sources: Flash Reward (Incentive System) and Flash Home (HR System).
Today: {pd.Timestamp.today().strftime("%d %b %Y")}. Model: {model_name}.

Rules:
- Be concise and executive-grade. Lead with the insight.
- Flag concerning data clearly.
- Reference specific numbers from the data.
- When the data includes EmployeeName, refer to employees by name (not EmployeeID) in your narrative. You may mention EmployeeID alongside the name for cross-referencing if helpful (e.g. "Aisyah Torres (E0001)").
- Do NOT use markdown bold (**) or italic (*) formatting.
- Do NOT mention SQL, dataframes, or technical details.
- Always write a response — never return an empty string.
- Keep answers to 3-5 sentences for simple queries; use plain bullet points (- item) for lists.
- State the scope and cycle period you are referencing.
- For follow-up questions, use the previous query result in the data context.{chart_note}

Data context:
{data_context}
"""

    messages = history[-10:] + [{"role": "user", "content": question}]
    stream   = stream_model(messages, system, model_name)
    return stream, chart, df
