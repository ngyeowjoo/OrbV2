"""
ai_engine.py  —  Intent detection, data retrieval, streaming multi-model reasoning
Supports: Claude (Anthropic) and DeepSeek (OpenAI-compatible)
Uses AI router (router.py) for intelligent intent classification.
Falls back to regex if router unavailable.
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

def _get_plot_theme():
    """Returns plotly layout theme dict based on current app theme."""
    is_dark = st.session_state.get("theme_mode", "light") == "dark"
    if is_dark:
        return dict(
            paper_bgcolor="#1A1D27", plot_bgcolor="#1A1D27",
            font=dict(color="#9CA3AF", family="Inter, sans-serif", size=11),
            margin=dict(l=32, r=16, t=40, b=32),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
        )
    return dict(
        paper_bgcolor="#FFFFFF", plot_bgcolor="#FAFAFA",
        font=dict(color="#374151", family="Inter, sans-serif", size=11),
        margin=dict(l=32, r=16, t=40, b=32),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )

# Keep PLOT_THEME as a static fallback (used during import before session state exists)
PLOT_THEME = dict(
    paper_bgcolor="#FFFFFF", plot_bgcolor="#FAFAFA",
    font=dict(color="#374151", family="Inter, sans-serif", size=11),
    margin=dict(l=32, r=16, t=40, b=32),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
)

def _fmt_axes(fig):
    is_dark = st.session_state.get("theme_mode", "light") == "dark"
    grid = "#2D3143" if is_dark else "#E5E7EB"
    fig.update_xaxes(gridcolor=grid, zeroline=False, linecolor=grid)
    fig.update_yaxes(gridcolor=grid, zeroline=False, linecolor=grid)

# ── INTENT CLASSIFIER ─────────────────────────────────────────────────────────
# Order matters: re.search stops at the first match, so narrower/more specific
# patterns must come before broader ones. free_form (`.*`) must stay LAST since
# it matches everything.
#
# NOTE: this classifier is a fallback used only when DeepSeek is unavailable
# (no API key, or the API call raised). Every intent the router can return
# needs a pattern here, or it silently collapses to free_form in that path.
INTENT_PATTERNS = {
    "cross_join":       r"(who are (they|those|these|them)|show.*their name|identify them|name them|which employee|tell me who|who (is|are) (it|that|this)|give me their name|who exactly|who specifically|which individuals|list those people|name those employees)",
    "missing_kpi":      r"(missing kpi|no kpi (this cycle|uploaded|recorded)|kpi.{0,15}not.{0,15}(uploaded|submitted|recorded|on file)|not.{0,15}kpi.{0,15}(uploaded|submitted|recorded|on file)|kpi gap|no performance data|scheme but no kpi|haven'?t submitted kpi|not in (the )?system)",
    "adjustment":       r"(kpi adjustment|adjusted kpi|adjustment status|kpi override|qualifying status adjustment|pending adjustment|approved adjustment)",
    "scheme_config":    r"(scheme.{0,15}tier|tier.{0,15}scheme|kpi weightage|scheme structure|scheme configuration|acknowledg.{0,20}scheme|scheme.{0,20}acknowledg|scheme expiry|when does.*scheme.*end|how is the scheme (set ?up|structured|configured))",
    "login":            r"(last\s*log(ged)?\s*in|login activity|logged in (on|since|recently)|log\s*in\s*(history|activity))",
    "announcement":     r"(announcement|active during (this|the) cycle|any (news|updates) (this|for the) cycle)",
    "tenure_compare":   r"(tenure band|by tenure|tenure cohort|years of service|long.?tenured|tenure vs|compare.*tenure|tenure.*compare)",
    "project_compare":  r"(compare.*project|project.*compare|project.{0,15}comparison|project vs project|per project breakdown|by project|across projects|which project (perform|has))",
    "kpi_trend":        r"(\btrend\b|over time|across cycles|payout history|kpi history|historical (kpi|payout|trend))",
    # cross_check = *non-active employees who still show a payout* (an anomaly
    # check). Plain "leaver"/"resign" questions belong to attrition, so this
    # requires the payout angle rather than matching "leaver" alone — otherwise
    # it swallows attrition questions before the attrition pattern is reached.
    "cross_check":      r"((non.?active|inactive|left|leaver|exit).{0,25}(payout|paid|compensat)|(payout|paid|compensat).{0,25}(non.?active|inactive|left|leaver|exit))",
    "country_compare":  r"(compare.*countr(y|ies)|countr(y|ies).*compare|vs.*countr(y|ies)|countr(y|ies).*vs|\bvs\b.*[A-Z]{2}|compare.*(sg|my|ph|th|id))",
    "ranking":          r"(top \d+|bottom \d+|highest pay|lowest pay|best perform|worst perform|above.?average|below.?average|hit.*above|hit.*below|rank(ing)?|most paid|least paid|who earn|top perform|highest earning|lowest earning)",
    "anomaly":          r"(anomaly|anomalies|mismatch|unusual|discrepanc|doesn'?t match|high.*rating.*low|low.*rating.*high|pmgm.*payout|payout.*pmgm)",
    "qualifier":        r"(qualifier|blocked|fail.*qual|qual.*fail)",
    "proration":        r"(prorat|attendance|absent|partial payout)",
    "new_joiner":       r"(new joiner|first cycle|recently joined|new hire)",
    "underperformance": r"(miss|under.?perform|below target|not hit|consistent.*miss|consistent.*below)",
    "attainment":       r"(\battainment\b|hit max|reach max|attain max|max payout|hit.*maximum|% hit|pct.*hit|what %.*payout|payout.*%|% of.*payout|hit full|full payout)",
    "employee_list":    r"(show.*name|list.*name|employee.*name|name.*employee|all employee|employee list|staff list|roster|directory)",
    "headcount":        r"(headcount|how many employee|count.*employee|employee.*count|workforce size|number of employee|how many.*(in|per|by) (each )?countr|breakdown by (country|status)|count by country)",
    "attrition":        r"(attrition|resign|turnover|leavers?)",
    "pmgm":             r"(pmgm|performance rating|rating distribution|appraisal)",
    "cycle_summary":    r"(summary|overview|this cycle|cycle summary|brief me)",
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
    """Merge name, job title, grade, department and supervisor from Flash Home.
    Places EmployeeName right after EmployeeID. Safe if columns missing."""
    if df is None or df.empty or "EmployeeID" not in df.columns:
        return df
    try:
        from data import get_flash_home
        fh = get_flash_home(countries)
        enrich = [c for c in ["EmployeeID","EmployeeName","JobTitle","EmployeeGrade",
                               "EmployeeDepartment","SupervisorID"]
                  if c in fh.columns and (c == "EmployeeID" or c not in df.columns)]
        if len(enrich) <= 1:
            return df   # nothing new to add
        df = df.merge(fh[enrich], on="EmployeeID", how="left")
        # Reorder: put EmployeeName right after EmployeeID
        if "EmployeeName" in df.columns:
            cols = df.columns.tolist()
            cols.remove("EmployeeName")
            cols.insert(cols.index("EmployeeID") + 1, "EmployeeName")
            df = df[cols]
        return df
    except Exception:
        return df

# ── DATA RETRIEVAL ────────────────────────────────────────────────────────────
def retrieve_data(intent: str, countries: list, question: str):
    from data import (
        attainment_summary, underperformer_summary,
        qualifier_summary, proration_summary, anomaly_summary,
        get_flash_home, get_flash_reward, get_joined,
        get_payout_cycle, get_project_cycle, get_incentive_scheme,
        get_scheme_tier, get_incentive_matrix, get_scheme_ack,
        get_scheme_audit, get_kpi_adjustment, get_qualifying_employee,
        get_qualifying_adj, get_login_audit, get_announcement, get_attendance,
    )
    scope = "Global" if "ALL" in countries else ", ".join(countries)

    if intent == "attainment":
        df, cycle = attainment_summary(countries)
        df = _add_names(df, countries)
        return df, f"Incentive attainment data for cycle {cycle}, scope: {scope}.\n{df.to_string(index=False)}"

    elif intent == "underperformance":
        from semantic import get_threshold, normalise
        min_cycles = get_threshold("consecutive_miss_min", 3)
        df = underperformer_summary(countries, min_cycles=min_cycles)
        fh = get_flash_home(countries)[["EmployeeID","Country","Project"]]
        df = df.merge(fh, on="EmployeeID", how="left")
        df = _add_names(df, countries)
        return df, f"Employees with >={min_cycles} consecutive cycles below target, scope: {scope}.\n{df.to_string(index=False)}"

    elif intent == "qualifier":
        df  = qualifier_summary(countries)
        fh  = get_flash_home(countries)
        # Also pull qualifying_employee detail for latest cycle
        qe  = get_qualifying_employee(countries)
        fr  = get_flash_reward(countries)
        latest = fr["Cycle"].max()
        qe_latest = qe[qe["Cycle"] == latest]
        fail_latest = qe_latest[qe_latest["QualifierStatus"] == 0]
        fail_summary = fail_latest.groupby("Qualifier").agg(
            FailCount=("EmployeeID","nunique")).reset_index()
        return df, (
            f"Qualifier failure data, scope: {scope}.\n"
            f"Latest cycle ({latest}) failures by qualifier:\n{fail_summary.to_string(index=False)}\n\n"
            f"All-time qualifier failure summary:\n{df.to_string(index=False)}"
        )

    elif intent == "proration":
        att  = get_attendance(countries)
        fh   = get_flash_home(countries)
        pror = att[att["ProrationFactor"] < 1.0].copy()
        pror["DaysAbsent"] = pror["MaxWorkingDays"] - pror["DaysWorked"]
        name_cols = [c for c in ["EmployeeID","EmployeeName","Country","Project"] if c in fh.columns]
        pror = pror.merge(fh[name_cols], on="EmployeeID", how="left", suffixes=("","_fh"))
        show = [c for c in ["EmployeeID","EmployeeName","Cycle","Country","Project",
                             "DaysWorked","MaxWorkingDays","DaysAbsent","ProrationFactor"] if c in pror.columns]
        total_rows = len(pror)
        shown = pror[show].head(50)
        cap_note = (
            f"NOTE: {total_rows} rows matched but only the first {len(shown)} are shown below. "
            f"Do not describe this as the complete list — say how many rows total, and offer to "
            f"narrow the filter (country/cycle/project) if the user wants the full set.\n"
            if total_rows > len(shown) else ""
        )
        return pror, (
            f"Attendance proration data, scope: {scope}.\n"
            f"Affected: {pror['EmployeeID'].nunique()} employees across {pror['Cycle'].nunique()} cycles\n"
            f"{cap_note}"
            f"{shown.to_string(index=False)}"
        )

    elif intent == "anomaly":
        from semantic import get_threshold
        # Anomaly is always run across the full active workforce —
        # never inherits status="Non-Active" from a prior cross_check turn.
        # Pass the base country list, not scoped_countries which may have a status filter.
        high_low, low_high, cycle = anomaly_summary(countries)
        total = len(high_low) + len(low_high)
        df = pd.concat([
            high_low.assign(AnomalyType="High PMGM / Low Payout"),
            low_high.assign(AnomalyType="Low PMGM / High Payout"),
        ], ignore_index=True) if total > 0 else pd.DataFrame()
        df = _add_names(df, countries) if not df.empty else df
        low_pct  = get_threshold("anomaly_high_pmgm_low_payout_pct", 50)
        high_pct = get_threshold("anomaly_low_pmgm_high_payout_pct", 95)
        return df, (
            f"PMGM vs payout anomaly for cycle {cycle}, scope: {scope} (Active employees only).\n"
            f"Criteria: High PMGM rating (Exceptional / Exceeds) + payout < {low_pct}% of max, "
            f"OR Low PMGM rating (Below / Unsatisfactory) + payout >= {high_pct}% of max.\n"
            f"High PMGM / Low Payout: {len(high_low)} employees\n"
            f"Low PMGM / High Payout: {len(low_high)} employees\n"
            f"{df.to_string(index=False) if not df.empty else 'No anomalies found under current thresholds.'}"
        )

    elif intent == "cross_check":
        fr  = get_flash_reward(countries)
        fh  = get_flash_home(countries)
        latest = fr["Cycle"].max()
        cyc    = fr[fr["Cycle"] == latest].drop_duplicates("EmployeeID")
        # All HR fields merged — so every follow-up (join date, grade, supervisor) is answerable
        hr_cols = [c for c in ["EmployeeID","EmployeeName","EmployeeStatus","Country","Project",
                                "JobTitle","EmployeeGrade","EmployeeDepartment",
                                "JoinDate","LastDate","SupervisorID","PMGMRating"] if c in fh.columns]
        merged  = cyc.merge(fh[hr_cols], on="EmployeeID", how="left")
        # Non-active with payouts
        df = merged[
            (merged.get("EmployeeStatus", pd.Series(dtype=str)) == "Non-Active") &
            (merged["TotalCyclePayout"] > 0)
        ].drop_duplicates("EmployeeID")
        show = [c for c in ["EmployeeID","EmployeeName","Country","Project","JobTitle",
                             "EmployeeGrade","JoinDate","LastDate","Scheme",
                             "TotalCyclePayout","SchemeMaxPayout","PMGMRating"] if c in df.columns]
        return df, (
            f"Non-active employees with incentive payouts in cycle {latest}, scope: {scope}.\n"
            f"Count: {len(df)} employees.\n"
            f"These employees have a recorded payout despite being Non-Active (LastDate is set).\n"
            f"{df[show].to_string(index=False)}"
        )

    elif intent == "new_joiner":
        import re as _re
        fh = get_flash_home(countries)
        fr = get_flash_reward(countries)
        # Extract month count from question (e.g. "last 3 months")
        m = _re.search(r'(\d+)\s*month', question, _re.IGNORECASE)
        months = int(m.group(1)) if m else 6
        cutoff = pd.Timestamp.today() - pd.DateOffset(months=months)
        new    = fh[(fh["JoinDate"] >= cutoff) & (fh["EmployeeStatus"] == "Active")]
        latest = fr["Cycle"].max()
        fr_l   = fr[fr["Cycle"] == latest].drop_duplicates("EmployeeID")
        df     = new.merge(fr_l[["EmployeeID","Scheme","TotalCyclePayout","ProrFactor"]], on="EmployeeID", how="left")
        show   = [c for c in ["EmployeeID","EmployeeName","JoinDate","Country","Project",
                               "JobTitle","EmployeeGrade","Scheme","TotalCyclePayout","ProrFactor"] if c in df.columns]
        return df, (
            f"New joiners (last {months} months) on incentive, scope: {scope}.\n"
            f"Count: {len(df)} employees joined since {cutoff.strftime('%d %b %Y')}\n"
            f"{df[show].to_string(index=False)}"
        )

    elif intent == "ranking":
        # Payout ranking — top/bottom N, above/below average, by country
        from semantic import get_threshold
        fr  = get_flash_reward(countries)
        fh  = get_flash_home(countries)
        latest = fr["Cycle"].max()
        cyc = fr[fr["Cycle"] == latest].drop_duplicates("EmployeeID")
        name_cols = [c for c in ["EmployeeID","EmployeeName","Country","Project","PMGMRating"]
                     if c in fh.columns]
        df = cyc.merge(fh[name_cols], on="EmployeeID", how="left")
        df["PayoutPct"]    = (df["TotalCyclePayout"] / df["SchemeMaxPayout"].replace(0,1) * 100).round(1)
        avg_payout         = df["TotalCyclePayout"].mean()
        df["AboveAverage"] = df["TotalCyclePayout"] > avg_payout
        low_pct  = get_threshold("low_payout_pct", 50)
        top_n    = get_threshold("top_n_default", 10)
        df["LowPayout"] = df["PayoutPct"] < low_pct
        show = [c for c in ["EmployeeID","EmployeeName","Scheme","TotalCyclePayout",
                             "SchemeMaxPayout","PayoutPct","QualifierFailed",
                             "Country","Project","PMGMRating","AboveAverage","LowPayout"] if c in df.columns]
        df = df[show].sort_values("TotalCyclePayout", ascending=False)
        return df, (
            f"Payout ranking for cycle {latest}, scope: {scope}.\n"
            f"Average payout: {avg_payout:,.0f} | Low payout threshold: <{low_pct}% of max | Default top N: {top_n}\n"
            f"Employees above average: {df['AboveAverage'].sum()} of {len(df)}\n"
            f"Full ranked list (highest to lowest payout):\n"
            f"{df.to_string(index=False)}"
        )

    elif intent == "employee_list":
        fh = get_flash_home(countries)
        cols = [c for c in ["EmployeeID","EmployeeName","Country","Project","EmployeeDepartment",
                             "EmployeeStatus","JobTitle","EmployeeGrade","PMGMRating",
                             "JoinDate","SupervisorID"] if c in fh.columns]
        df = fh[cols].copy()
        return df, (
            f"Employee directory, scope: {scope}. Total: {len(df)} employees.\n"
            f"Includes: job title, grade, department, supervisor, join date.\n"
            f"{df.to_string(index=False)}"
        )

    elif intent == "cross_join":
        fr  = get_flash_reward(countries)
        fh  = get_flash_home(countries)
        latest = fr["Cycle"].max()
        cyc = fr[fr["Cycle"] == latest].drop_duplicates("EmployeeID")
        name_cols = [c for c in ["EmployeeID","EmployeeName","Country","Project",
                                  "EmployeeStatus","PMGMRating","JobTitle","EmployeeGrade",
                                  "SupervisorID","JoinDate"] if c in fh.columns]
        df = cyc.merge(fh[name_cols], on="EmployeeID", how="left")
        show = [c for c in ["EmployeeID","EmployeeName","Scheme","TotalCyclePayout",
                             "SchemeMaxPayout","QualifierFailed","ProrFactor",
                             "Country","Project","PMGMRating","JobTitle","EmployeeGrade"] if c in df.columns]
        return df[show], (
            f"Full employee-level joined data for cycle {latest}, scope: {scope}.\n"
            f"Includes: name, grade, title, payout, qualifier, proration.\n"
            f"{df[show].to_string(index=False)}"
        )

    elif intent == "headcount":
        fh   = get_flash_home(countries)
        summ = fh.groupby(["Country","Project","EmployeeStatus"]).size().reset_index(name="Count")
        by_grade = fh.groupby(["EmployeeGrade","EmployeeStatus"]).size().reset_index(name="Count") if "EmployeeGrade" in fh.columns else pd.DataFrame()
        total = len(fh)
        active = (fh["EmployeeStatus"] == "Active").sum()
        desc = (
            f"Headcount by country, project and status, scope: {scope}. "
            f"Total: {total}, Active: {active}, Non-Active: {total-active}\n"
            f"{summ.to_string(index=False)}"
        )
        if not by_grade.empty:
            desc += f"\n\nBy grade:\n{by_grade.to_string(index=False)}"
        return summ, desc

    elif intent == "attrition":
        fh      = get_flash_home(countries)
        leavers = fh[fh["EmployeeStatus"] == "Non-Active"].copy()
        leavers["YearLeft"]    = leavers["LastDate"].dt.year
        leavers["TenureYears"] = ((leavers["LastDate"] - leavers["JoinDate"]).dt.days / 365.25).round(1)
        summ = leavers.groupby(["Country","YearLeft"]).size().reset_index(name="Count")
        show = [c for c in ["EmployeeID","EmployeeName","Country","Project","JobTitle",
                             "JoinDate","LastDate","TenureYears","PMGMRating"] if c in leavers.columns]
        fr   = get_flash_reward(countries)
        latest = fr["Cycle"].max()
        leavers_with_pay = leavers.merge(
            fr[fr["Cycle"]==latest].drop_duplicates("EmployeeID")[["EmployeeID","TotalCyclePayout"]],
            on="EmployeeID", how="left"
        )
        with_pay = leavers_with_pay[leavers_with_pay["TotalCyclePayout"].notna() &
                                    (leavers_with_pay["TotalCyclePayout"] > 0)]
        return leavers, (
            f"Attrition data, scope: {scope}. Total leavers: {len(leavers)}\n"
            f"By country and year:\n{summ.to_string(index=False)}\n\n"
            f"Leavers with payout in latest cycle ({latest}): {len(with_pay)}\n"
            f"Leaver details:\n{leavers[show].to_string(index=False)}"
        )

    elif intent == "pmgm":
        fh   = get_flash_home(countries)
        dist = fh.groupby(["PMGMRating","Country"]).size().reset_index(name="Count")
        return dist, f"PMGM rating distribution, scope: {scope}.\n{dist.to_string(index=False)}"

    elif intent == "cycle_summary":
        fr     = get_flash_reward(countries)
        fh     = get_flash_home(countries)
        qe     = get_qualifying_employee(countries)
        pc     = get_payout_cycle()
        latest = fr["Cycle"].max()
        cyc    = fr[fr["Cycle"] == latest].drop_duplicates("EmployeeID")
        total_p   = cyc["TotalCyclePayout"].sum()
        avg_pct   = (cyc["TotalCyclePayout"] / cyc["SchemeMaxPayout"]).mean()
        hit_max   = (cyc["TotalCyclePayout"] >= cyc["SchemeMaxPayout"] * 0.999).sum()
        att       = get_attendance(countries)
        prorat    = att[(att["Cycle"] == latest) & (att["ProrationFactor"] < 1.0)]["EmployeeID"].nunique()
        q_fail    = qe[(qe["Cycle"] == latest) & (qe["QualifierStatus"] == 0)]["EmployeeID"].nunique()
        # Cycle dates
        cycle_row = pc[pc["CycleName"] == latest]
        cutoff_info = ""
        if not cycle_row.empty:
            cr = cycle_row.iloc[0]
            cutoff_info = (f"\nCycle dates: {cr['CycleStartDate'].strftime('%d %b %Y')} — "
                           f"{cr['CycleEndDate'].strftime('%d %b %Y')}"
                           f"\nKPI upload cutoff: {cr['CycleUploadCutoffDate'].strftime('%d %b %Y')}")
        # By scheme
        by_scheme = cyc.groupby("Scheme").agg(
            Employees=("EmployeeID","count"),
            TotalPayout=("TotalCyclePayout","sum"),
            AvgPayout=("TotalCyclePayout","mean"),
        ).reset_index().round(2)
        result_df = _add_names(
            cyc[["EmployeeID","Scheme","TotalCyclePayout","SchemeMaxPayout","ProrFactor"]].head(50),
            countries
        )
        desc = (
            f"Cycle: {latest}, Scope: {scope}{cutoff_info}\n"
            f"Total eligible employees: {len(cyc)}\n"
            f"Total payout: {total_p:,.2f}\n"
            f"Average payout as % of max: {avg_pct:.1%}\n"
            f"Hit max payout: {hit_max} ({hit_max/len(cyc):.1%})\n"
            f"Qualifier failures: {q_fail}\n"
            f"Prorated for attendance: {prorat}\n\n"
            f"By scheme:\n{by_scheme.to_string(index=False)}"
        )
        return result_df, desc

    elif intent == "kpi_trend":
        fr  = get_flash_reward(countries)
        fh  = get_flash_home(countries)
        emp_id = None
        for eid in fh["EmployeeID"].unique():
            if eid.lower() in question.lower():
                emp_id = eid
                break
        if emp_id is None and "EmployeeName" in fh.columns:
            for _, row in fh.iterrows():
                if row["EmployeeName"] and str(row["EmployeeName"]).lower() in question.lower():
                    emp_id = row["EmployeeID"]
                    break
        if emp_id:
            emp_fr = fr[fr["EmployeeID"] == emp_id].copy()
            emp_name_vals = fh[fh["EmployeeID"] == emp_id]["EmployeeName"].values
            emp_name = emp_name_vals[0] if len(emp_name_vals) else emp_id
            cyc_agg = emp_fr.groupby("Cycle").agg(
                TotalPayout=("TotalCyclePayout", "first"),
                ProrFactor=("ProrFactor", "first"),
                Scheme=("Scheme", "first"),
            ).reset_index().sort_values("Cycle")
            metric_trend = emp_fr.groupby(["Cycle","Metric"]).agg(
                Target=("Target","first"),
                Achieved=("Achieved","first"),
                MetricPayout=("MetricPayout","sum"),
            ).reset_index().sort_values(["Cycle","Metric"])
            desc = (
                f"KPI trend for {emp_name} ({emp_id}), scope: {scope}\n"
                f"Cycle-level payout and proration:\n{cyc_agg.to_string(index=False)}\n\n"
                f"Per-metric performance across cycles:\n{metric_trend.to_string(index=False)}"
            )
            return cyc_agg, desc
        else:
            cyc_agg = fr.groupby("Cycle").agg(
                AvgPayout=("TotalCyclePayout","mean"),
                TotalPayout=("TotalCyclePayout","sum"),
                Employees=("EmployeeID","nunique"),
                AvgAchieved=("Achieved","mean"),
            ).reset_index().sort_values("Cycle")
            pror_trend = fr[fr["ProrFactor"]<1.0].groupby("Cycle").agg(
                ProratedCount=("EmployeeID","nunique"),
                AvgProrFactor=("ProrFactor","mean"),
            ).reset_index()
            cyc_agg = cyc_agg.merge(pror_trend, on="Cycle", how="left").fillna(0)
            return cyc_agg, f"KPI/payout trend across all cycles, scope: {scope}\n{cyc_agg.to_string(index=False)}"

    elif intent == "project_compare":
        fr  = get_flash_reward(countries)
        fh  = get_flash_home(countries)
        latest = fr["Cycle"].max()
        cyc    = fr[fr["Cycle"]==latest].drop_duplicates("EmployeeID")
        merged = cyc.merge(fh[["EmployeeID","Project","Country","PMGMRating"]], on="EmployeeID", how="left")
        comp = merged.groupby("Project").agg(
            Employees=("EmployeeID","count"),
            AvgPayout=("TotalCyclePayout","mean"),
            TotalPayout=("TotalCyclePayout","sum"),
            MaxPayout=("TotalCyclePayout","max"),
            AvgAchieved=("Achieved","mean"),
            QualFailures=("QualifierFailed", lambda x: x.notna().sum()),
            Prorated=("ProrFactor", lambda x: (x<1.0).sum()),
        ).reset_index().round(2)
        return comp, f"Project comparison for cycle {latest}, scope: {scope}\n{comp.to_string(index=False)}"

    elif intent == "tenure_compare":
        fr  = get_flash_reward(countries)
        fh  = get_flash_home(countries)
        latest = fr["Cycle"].max()
        cyc    = fr[fr["Cycle"]==latest].drop_duplicates("EmployeeID")
        fh_t = fh[["EmployeeID","JoinDate","EmployeeName","Project","Country"]].copy()
        fh_t["TenureYears"] = ((pd.Timestamp.today() - fh_t["JoinDate"]).dt.days / 365.25).round(1)
        def _band(y):
            if pd.isna(y): return "Unknown"
            if y < 1:  return "< 1 year"
            if y < 2:  return "1–2 years"
            if y < 3:  return "2–3 years"
            if y < 5:  return "3–5 years"
            if y < 10: return "5–10 years"
            return "10+ years"
        fh_t["TenureBand"] = fh_t["TenureYears"].apply(_band)
        merged = cyc.merge(fh_t, on="EmployeeID", how="left")
        band_order = ["< 1 year","1–2 years","2–3 years","3–5 years","5–10 years","10+ years","Unknown"]
        comp = merged.groupby("TenureBand").agg(
            Employees=("EmployeeID","count"),
            AvgPayout=("TotalCyclePayout","mean"),
            TotalPayout=("TotalCyclePayout","sum"),
            AvgAchieved=("Achieved","mean"),
        ).reset_index().round(2)
        comp["TenureBand"] = pd.Categorical(comp["TenureBand"], categories=band_order, ordered=True)
        comp = comp.sort_values("TenureBand")
        return comp, f"Incentive by tenure band, cycle {latest}, scope: {scope}\n{comp.to_string(index=False)}"

    elif intent == "missing_kpi":
        fr  = get_flash_reward(countries)
        fh  = get_flash_home(countries)
        latest = fr["Cycle"].max()
        all_emp    = set(fr["EmployeeID"].unique())
        in_latest  = set(fr[fr["Cycle"]==latest]["EmployeeID"].unique())
        missing    = list(all_emp - in_latest)
        fh_active  = fh[(fh["EmployeeStatus"]=="Active") & (fh["EmployeeID"].isin(missing))]
        name_cols  = [c for c in ["EmployeeID","EmployeeName","Project","Country","PMGMRating"] if c in fh_active.columns]
        last_scheme = (fr[fr["EmployeeID"].isin(missing)]
                       .sort_values("Cycle")
                       .groupby("EmployeeID")
                       .last()[["Scheme","Cycle"]]
                       .reset_index()
                       .rename(columns={"Scheme":"LastKnownScheme","Cycle":"LastCycle"}))
        df = fh_active[name_cols].merge(last_scheme, on="EmployeeID", how="left")
        return df, (
            f"Active employees on incentive with NO KPI record in latest cycle ({latest}), scope: {scope}\n"
            f"Count: {len(df)} employees missing\n{df.to_string(index=False)}"
        )

    elif intent == "adjustment":
        adj = get_kpi_adjustment(countries)
        qa  = get_qualifying_adj(countries)
        fh  = get_flash_home(countries)
        # KPI adjustments
        kpi_show = [c for c in ["AdjustmentID","EmployeeID","EmployeeName","Cycle","Metric",
                                 "Scheme","RecordedKPI","AdjustedKPI","RecordedPayout",
                                 "AdjustedPayout","RecordedTier","AdjStatus",
                                 "SubmittedBy","SubmittedDate","ApprovedBy","ApprovedDate"] if c in adj.columns]
        # Qualifier adjustments
        qa_show = [c for c in ["QualAdjID","EmployeeID","Cycle","Qualifier","RecordedStatus",
                                "AdjustedStatus","AdjStatus","ActionBy","ActionDate"] if c in qa.columns]
        pending_kpi  = adj[adj["AdjStatus"] == "Pending"] if not adj.empty else pd.DataFrame()
        pending_qual = qa[qa["AdjStatus"] == "Pending"]   if not qa.empty  else pd.DataFrame()
        desc = (
            f"Adjustment data, scope: {scope}.\n\n"
            f"KPI Adjustments — Total: {len(adj)}, "
            f"Pending: {len(pending_kpi)}, "
            f"Approved: {len(adj[adj['AdjStatus']=='Approved']) if not adj.empty else 0}, "
            f"Rejected: {len(adj[adj['AdjStatus']=='Rejected']) if not adj.empty else 0}\n"
            f"{adj[kpi_show].to_string(index=False) if not adj.empty else 'No KPI adjustments.'}\n\n"
            f"Qualifier Status Adjustments — Total: {len(qa)}, Pending: {len(pending_qual)}\n"
            f"{qa[qa_show].to_string(index=False) if not qa.empty else 'No qualifier adjustments.'}"
        )
        return adj, desc

    elif intent == "scheme_config":
        scheme   = get_incentive_scheme()
        tiers    = get_scheme_tier()
        matrix   = get_incentive_matrix()
        ack      = get_scheme_ack(countries)
        fr       = get_flash_reward(countries)
        latest   = fr["Cycle"].max()
        cyc      = fr[fr["Cycle"] == latest]
        # Active employees per scheme
        emp_by_scheme = cyc.groupby("Scheme")["EmployeeID"].nunique().reset_index(name="ActiveEmployees")
        # Ack summary
        ack_summ = ack.groupby(["SchemeName","AckStatus"]).size().reset_index(name="Count") if not ack.empty else pd.DataFrame()
        desc = (
            f"Incentive scheme configuration for cycle {latest}, scope: {scope}.\n\n"
            f"Schemes:\n{scheme[['SchemeName','MaxPayout','StartDate','EndDate','Status']].to_string(index=False)}\n\n"
            f"Active employees per scheme:\n{emp_by_scheme.to_string(index=False)}\n\n"
            f"KPI Matrix (metrics + weightage):\n{matrix[['SchemeName','Metric','Weightage']].to_string(index=False)}\n\n"
            f"Tier structure:\n{tiers[['SchemeName','Tier','TargetMin','TargetMax','PayoutPct','PayoutAmount']].to_string(index=False)}\n\n"
            f"Acknowledgement status:\n{ack_summ.to_string(index=False) if not ack_summ.empty else 'No ack data.'}"
        )
        return scheme, desc

    elif intent == "login":
        log = get_login_audit(countries)
        fh  = get_flash_home(countries)
        name_cols = [c for c in ["EmployeeID","EmployeeName","Country","Project",
                                  "EmployeeStatus","JobTitle"] if c in fh.columns]
        df = log.merge(fh[name_cols], on="EmployeeID", how="left")
        show = [c for c in ["EmployeeID","EmployeeName","Country","Project",
                             "EmployeeStatus","LastLoginDate"] if c in df.columns]
        df = df[show].sort_values("LastLoginDate", ascending=False)
        thirty_days_ago = pd.Timestamp.today() - pd.DateOffset(days=30)
        inactive_logins = df[df["LastLoginDate"] < thirty_days_ago] if "LastLoginDate" in df.columns else pd.DataFrame()
        return df, (
            f"Login activity, scope: {scope}. Total employees: {len(df)}\n"
            f"Not logged in for 30+ days: {len(inactive_logins)}\n"
            f"{df.to_string(index=False)}"
        )

    elif intent == "announcement":
        ann    = get_announcement()
        fr     = get_flash_reward(countries)
        latest = fr["Cycle"].max()
        active = ann[ann["IsActive"] == True] if "IsActive" in ann.columns else ann
        show   = [c for c in ["AnnouncementID","Message","CycleName","StartDate","EndDate"] if c in active.columns]
        return active, (
            f"Announcements, current cycle: {latest}, scope: {scope}.\n"
            f"Total active announcements: {len(active)}\n"
            f"{active[show].to_string(index=False)}"
        )

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
            fig.update_layout(**_get_plot_theme(), yaxis_title="% Hit Max"); _fmt_axes(fig); return fig

        elif intent == "underperformance":
            x_col = "EmployeeName" if "EmployeeName" in df.columns else "EmployeeID"
            fig = px.bar(df.sort_values("MaxConsecutiveMisses", ascending=False).head(20),
                         x=x_col, y="MaxConsecutiveMisses",
                         color="Country" if "Country" in df.columns else None,
                         title="Consecutive Cycles Below Target (Top 20)",
                         color_discrete_sequence=PALETTE)
            fig.update_layout(**_get_plot_theme(), xaxis_title="Employee"); _fmt_axes(fig); return fig

        elif intent == "qualifier":
            top = df.groupby("QualifierFailed")["TimesFailed"].sum().reset_index()
            fig = px.bar(top, x="QualifierFailed", y="TimesFailed",
                         title="Qualifier Failure Frequency",
                         color_discrete_sequence=[C_WARN])
            fig.update_layout(**_get_plot_theme()); _fmt_axes(fig); return fig

        elif intent == "proration":
            monthly = df.groupby("Cycle")["EmployeeID"].nunique().reset_index(name="ProatedCount")
            fig = px.line(monthly, x="Cycle", y="ProatedCount",
                          title="Prorated Employees per Cycle",
                          markers=True, color_discrete_sequence=[C_AMBER])
            fig.update_layout(**_get_plot_theme()); _fmt_axes(fig); return fig

        elif intent == "anomaly":
            fig = px.scatter(df, x="PayoutPct", y="PMGMRating",
                             color="AnomalyType",
                             title="Performance vs Payout Anomaly",
                             color_discrete_sequence=[C_BAD, C_WARN],
                             hover_data=["EmployeeID"])
            fig.update_layout(**_get_plot_theme()); _fmt_axes(fig); return fig

        elif intent == "headcount":
            fig = px.bar(df, x="Country", y="Count", color="EmployeeStatus",
                         title="Headcount by Country & Status",
                         color_discrete_sequence=[C_AMBER, C_WARN])
            fig.update_layout(**_get_plot_theme(), barmode="group"); _fmt_axes(fig); return fig

        elif intent == "attrition":
            fig = px.bar(df, x="YearLeft", y="Count", color="Country",
                         title="Attrition by Year & Country",
                         color_discrete_sequence=PALETTE)
            fig.update_layout(**_get_plot_theme()); _fmt_axes(fig); return fig

        elif intent == "pmgm":
            order = ["Exceptional","Exceeds Expectations","Meets Expectations",
                     "Below Expectations","Unsatisfactory"]
            fig = px.bar(df, x="PMGMRating", y="Count", color="Country",
                         title="PMGM Rating Distribution",
                         category_orders={"PMGMRating": order},
                         color_discrete_sequence=PALETTE)
            fig.update_layout(**_get_plot_theme()); _fmt_axes(fig); return fig

        elif intent == "country_compare":
            fig = px.bar(df, x="Country", y="HitMaxPct",
                         title="% Hitting Max Payout by Country",
                         color="Country", color_discrete_sequence=PALETTE)
            fig.update_layout(**_get_plot_theme(), yaxis_tickformat=".0%"); _fmt_axes(fig); return fig

        elif intent == "cross_check":
            if df.empty: return None
            x_col = "EmployeeName" if "EmployeeName" in df.columns else "EmployeeID"
            fig = px.bar(df, x=x_col, y="TotalCyclePayout",
                         title="Non-Active Employees with Payouts",
                         color_discrete_sequence=[C_BAD])
            fig.update_layout(**_get_plot_theme(), xaxis_title="Employee"); _fmt_axes(fig); return fig

        elif intent == "kpi_trend":
            if "Cycle" not in df.columns: return None
            y_col = "TotalPayout" if "TotalPayout" in df.columns else "AvgPayout"
            title = "Payout Trend Across Cycles"
            fig = px.line(df, x="Cycle", y=y_col, markers=True,
                          title=title, color_discrete_sequence=[C_AMBER])
            fig.update_layout(**_get_plot_theme(), yaxis_title="Payout"); _fmt_axes(fig); return fig

        elif intent == "project_compare":
            if "Project" not in df.columns: return None
            fig = px.bar(df.sort_values("AvgPayout", ascending=False),
                         x="Project", y="AvgPayout",
                         color="Project", title="Average Payout by Project",
                         color_discrete_sequence=PALETTE)
            fig.update_layout(**_get_plot_theme(), yaxis_title="Avg Payout"); _fmt_axes(fig); return fig

        elif intent == "tenure_compare":
            if "TenureBand" not in df.columns: return None
            fig = px.bar(df, x="TenureBand", y="AvgPayout",
                         title="Average Payout by Tenure Band",
                         color_discrete_sequence=[C_AMBER])
            fig.update_layout(**_get_plot_theme(), yaxis_title="Avg Payout"); _fmt_axes(fig); return fig

        elif intent == "adjustment":
            if df.empty: return None
            x_col = "EmployeeName" if "EmployeeName" in df.columns else "EmployeeID"
            if "PayoutDelta" not in df.columns: return None
            fig = px.bar(df.sort_values("PayoutDelta").head(30),
                         x=x_col, y="PayoutDelta",
                         title="Payout Delta (Adjustment Impact)",
                         color_discrete_sequence=[C_WARN])
            fig.update_layout(**_get_plot_theme(), xaxis_title="Employee"); _fmt_axes(fig); return fig

        elif intent == "scheme_config":
            if "Scheme" not in df.columns or "ActiveEmployees" not in df.columns: return None
            fig = px.bar(df, x="Scheme", y="ActiveEmployees",
                         color="Scheme", title="Active Employees per Scheme",
                         color_discrete_sequence=PALETTE)
            fig.update_layout(**_get_plot_theme(), yaxis_title="Employees"); _fmt_axes(fig); return fig

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
# ── PER-INTENT TOKEN BUDGETS ──────────────────────────────────────────────────
# Intents that produce long structured output get more tokens.
# All others default to 1200 (more than the old 1024 for breathing room).
INTENT_MAX_TOKENS = {
    "cycle_summary":    2000,
    "ranking":          1800,
    "country_compare":  1600,
    "underperformance": 1600,
    "employee_list":    1600,
    "cross_join":       1600,
    "anomaly":          1500,
    "attainment":       1400,
    "qualifier":        1400,
    "proration":        1400,
    "cross_check":      1400,
    "new_joiner":       1400,
    "free_form":        1400,
    "kpi_trend":        1800,
    "project_compare":  1600,
    "tenure_compare":   1400,
    "missing_kpi":      1400,
    "adjustment":       1600,
    "scheme_config":    1600,
    "login":            1000,
    "announcement":     1000,
    "attrition":        1200,
    "pmgm":             1200,
}
DEFAULT_MAX_TOKENS = 1200


def stream_anthropic(messages: list, system: str, model_id: str,
                     max_tokens: int = DEFAULT_MAX_TOKENS):
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
            "max_tokens": max_tokens,
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


def stream_deepseek(messages: list, system: str, model_id: str,
                    max_tokens: int = DEFAULT_MAX_TOKENS):
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
            "max_tokens": max_tokens,
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


def stream_model(messages: list, system: str, model_name: str,
                 max_tokens: int = DEFAULT_MAX_TOKENS):
    """Dispatches to the correct streaming provider."""
    cfg      = MODELS.get(model_name, MODELS[DEFAULT_MODEL])
    provider = cfg["provider"]
    model_id = cfg["model_id"]
    if provider == "anthropic":
        yield from stream_anthropic(messages, system, model_id, max_tokens)
    elif provider == "deepseek":
        yield from stream_deepseek(messages, system, model_id, max_tokens)


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

def _get_followup_context(countries: list, last_df) -> tuple:
    """
    Idea 1: last_df-first follow-up context.

    Priority order:
    1. If last_df has EmployeeID — enrich it with ALL HR + reward fields and use it.
       This keeps the AI focused on exactly the employees from the prior result.
    2. Fallback: fetch latest cycle company-wide (capped at 150 rows).

    This prevents "why are you talking about 150 random employees when I asked about
    the 7 anomaly employees from the previous answer?"
    """
    from data import get_flash_reward, get_flash_home
    fr  = get_flash_reward(countries)
    fh  = get_flash_home(countries)
    latest = fr["Cycle"].max()

    hr_cols = [c for c in [
        "EmployeeID","EmployeeName","Country","Project","EmployeeStatus",
        "PMGMRating","JobTitle","EmployeeGrade","EmployeeDepartment",
        "JoinDate","LastDate","SupervisorID",
    ] if c in fh.columns]

    reward_cols = [c for c in [
        "EmployeeID","Scheme","SchemeMaxPayout","TotalCyclePayout",
        "TierAchieved","QualifierFailed","ProrFactor","PayoutEarned",
    ] if c in fr.columns]

    # ── PATH 1: Enrich last_df with any missing HR and reward fields ───────────
    if last_df is not None and "EmployeeID" in last_df.columns and not last_df.empty:
        context_df = last_df.copy()

        # Merge in HR fields not already in last_df
        missing_hr = [c for c in hr_cols if c not in context_df.columns]
        if missing_hr:
            context_df = context_df.merge(
                fh[["EmployeeID"] + missing_hr], on="EmployeeID", how="left"
            )

        # Merge in reward fields from latest cycle not already present
        missing_rew = [c for c in reward_cols if c not in context_df.columns]
        if missing_rew:
            cyc_latest = (fr[fr["Cycle"] == latest]
                          .drop_duplicates("EmployeeID")
                          [["EmployeeID"] + missing_rew])
            context_df = context_df.merge(cyc_latest, on="EmployeeID", how="left")

        n = len(context_df)
        desc = (
            f"FOLLOW-UP CONTEXT — {n} employee(s) from the previous result.\n"
            f"These are the SPECIFIC employees the user is asking about. "
            f"Answer only from this set unless the user explicitly asks to broaden scope.\n"
            f"Cycle: {latest}. Columns: {', '.join(context_df.columns.tolist())}\n"
            f"{context_df.to_string(index=False)}"
        )
        return context_df, desc

    # ── PATH 2: No last_df — fetch company-wide latest cycle ──────────────────
    cyc = fr[fr["Cycle"] == latest].drop_duplicates("EmployeeID")
    joined = cyc.merge(fh[hr_cols], on="EmployeeID", how="left")

    show = [c for c in [
        "EmployeeID","EmployeeName","Scheme","TotalCyclePayout","SchemeMaxPayout",
        "TierAchieved","QualifierFailed","ProrFactor",
        "Country","Project","PMGMRating","EmployeeStatus",
        "JobTitle","EmployeeGrade","EmployeeDepartment","JoinDate","LastDate","SupervisorID",
    ] if c in joined.columns]
    joined = joined[show]

    total_rows = len(joined)
    ROW_CAP    = 150
    joined_display = (
        joined.sort_values("TotalCyclePayout", ascending=False).head(ROW_CAP)
        if "TotalCyclePayout" in joined.columns else joined.head(ROW_CAP)
    )

    cap_note = ""
    if total_rows > ROW_CAP:
        cap_note = (
            f"\n[{total_rows} employees total — showing top {ROW_CAP} by payout. "
            f"If the user asks about a specific employee not shown, state this limitation.]"
        )

    desc = (
        f"FOLLOW-UP CONTEXT — company-wide latest cycle ({latest}). "
        f"{min(total_rows, ROW_CAP)} of {total_rows} employees shown.{cap_note}\n"
        f"Columns: {', '.join(joined_display.columns.tolist())}\n"
        f"{joined_display.to_string(index=False)}"
    )
    return joined, desc


# Intents whose answer needs a data table that _get_followup_context's Path 1
# enrichment (HR fields + latest-cycle reward fields) does NOT cover. If a
# follow-up's classified intent falls in this set, reusing the enriched
# last_df will look like "no data" to the model — correctly, per the
# anti-hallucination rules, since the table genuinely isn't there. These get
# routed through _get_scoped_fresh_context instead. See retrieve_data() below
# for what each of these actually pulls from (get_attendance, get_login_audit,
# get_announcement, get_kpi_adjustment, get_incentive_scheme/tier/*, etc.)
_SPECIALIZED_DOMAIN_INTENTS = {
    "proration", "kpi_trend", "missing_kpi", "adjustment",
    "scheme_config", "login", "announcement",
}


def _get_scoped_fresh_context(intent: str, countries: list, last_df, question: str) -> tuple:
    """
    Fresh, intent-appropriate retrieval for a follow-up that switched to a
    specialized data domain. Where possible, narrows the fresh result to the
    same employees the user was just discussing (from last_df), so it still
    reads as a follow-up rather than a company-wide query. If none of those
    employees have rows in the new table, falls back to the full scoped
    result and says so explicitly, in both the data context (for the model)
    and the returned scope_note (for the UI to optionally surface).

    Returns (df, data_context, scope_note). scope_note is "" when nothing
    unusual happened (no prior employees to narrow to, or narrowing worked
    with no need to explain it beyond the data context itself).
    """
    df, data_context = retrieve_data(intent, countries, question)

    prior_ids = None
    if last_df is not None and "EmployeeID" in last_df.columns and not last_df.empty:
        prior_ids = set(last_df["EmployeeID"].dropna().tolist())

    if not prior_ids or df is None or df.empty or "EmployeeID" not in df.columns:
        return df, data_context, ""

    narrowed = df[df["EmployeeID"].isin(prior_ids)]
    label = intent.replace("_", " ")

    if narrowed.empty:
        scope_note = (
            f"Switched to {label} data — none of the employees from your previous "
            f"question have {label} records, so this is the full scope instead."
        )
        return df, data_context + f"\n\nNOTE TO AI: {scope_note}", scope_note

    scope_note = (
        f"Switched to {label} data — narrowed to the {len(narrowed)} employee(s) "
        f"from your previous question."
    )
    narrowed_desc = (
        f"FOLLOW-UP CONTEXT (topic switched to {intent}) — "
        f"{len(narrowed)} employee(s) from the previous result.\n"
        f"Columns: {', '.join(narrowed.columns.tolist())}\n"
        f"{narrowed.to_string(index=False)}"
    )
    return narrowed, narrowed_desc, scope_note


def rewrite_followup_question(question: str, history: list,
                              last_df, model_name: str) -> str:
    """
    Idea 2: Query rewriting for follow-ups.

    Resolves pronouns (they/them/that employee/those) into specific names/IDs
    using the conversation history and last_df as grounding.

    Only fires when:
    - Question is short (≤10 words) — longer questions are usually already specific
    - Question contains a pronoun or reference word
    - There is prior conversation history to resolve from

    Validation: the rewritten question must contain a name or ID that actually
    appears in last_df (or history) — otherwise return original to avoid hallucination.
    """
    import re as _re

    PRONOUN_PATTERN = _re.compile(
        r"\b(they|them|their|those|these|that employee|that person|"
        r"that one|this employee|the same|who are they|who is that)\b",
        _re.IGNORECASE
    )

    words = question.split()
    if len(words) > 10:
        return question
    if not PRONOUN_PATTERN.search(question):
        return question
    if not history:
        return question

    # Build grounding: names and IDs from last_df
    known_names = set()
    known_ids   = set()
    if last_df is not None and not last_df.empty:
        if "EmployeeName" in last_df.columns:
            known_names = set(last_df["EmployeeName"].dropna().tolist())
        if "EmployeeID" in last_df.columns:
            known_ids = set(last_df["EmployeeID"].dropna().tolist())

    # Also pull names/IDs mentioned in last assistant message
    last_assistant = next(
        (m["content"] for m in reversed(history) if m["role"] == "assistant"), ""
    )

    rewrite_system = (
        "You are a query rewriter for a workforce analytics assistant. "
        "The user asked a follow-up question with a pronoun. "
        "Using the conversation history, rewrite it as a standalone question "
        "that names the specific employee(s) or group being referenced. "
        "Output ONLY the rewritten question — no explanation, no prefix."
    )

    history_excerpt = "\n".join(
        f"{m['role'].upper()}: {m['content'][:300]}"
        for m in history[-4:]
    )
    rewrite_prompt = (
        f"Conversation so far:\n{history_excerpt}\n\n"
        f"Follow-up question to rewrite: {question}\n\n"
        f"Rewritten standalone question:"
    )

    try:
        rewritten = call_model(
            [{"role": "user", "content": rewrite_prompt}],
            rewrite_system,
            model_name,
        ).strip().strip('"').strip("'")

        if not rewritten or len(rewritten) < 5:
            return question

        # Validation: rewritten question must mention a name/ID we know about,
        # or reference something from the last assistant message.
        # This prevents the rewriter from hallucinating a wrong employee.
        rewritten_lower = rewritten.lower()
        has_known_ref = (
            any(n.lower() in rewritten_lower for n in known_names if n)
            or any(i.lower() in rewritten_lower for i in known_ids if i)
            or any(w.lower() in rewritten_lower for w in last_assistant.split()
                   if len(w) > 4 and w[0].isupper())  # proper nouns from assistant
        )

        return rewritten if has_known_ref else question

    except Exception:
        return question


# ── MAIN ENTRY POINT ──────────────────────────────────────────────────────────
def _estimate_confidence(routing: dict, retrieval_mode: str, intent: str,
                          df, data_context: str, scope_note: str) -> tuple:
    """
    Heuristic reliability indicator (0-100) for a reply, built ONLY from
    concrete pipeline signals — never asked of the LLM itself, since models
    are notoriously badly calibrated about their own confidence and asking
    would just add another thing that could be hallucinated. Every deduction
    below corresponds to a real, checkable fact about how the answer was
    produced (which router path fired, which retrieval path fired, whether
    data was truncated, etc.), returned alongside the score so the UI can
    show *why*, not just a bare number.

    This is a heuristic, not a statistically calibrated probability — treat
    it as "how many known risk factors applied," not "P(this answer is
    correct)".
    """
    score = 95
    factors = []
    reasoning = (routing.get("reasoning") or "").lower()

    if "regex fallback" in reasoning:
        score -= 25
        factors.append("DeepSeek router was unavailable — used keyword-based fallback routing")
    if "model was unsure" in reasoning:
        score -= 15
        factors.append("AI router wasn't confident — fell back to a configured keyword hint")

    if retrieval_mode == "vector":
        score -= 20
        factors.append("used fuzzy semantic search rather than a direct data query")

    if intent == "free_form":
        score -= 15
        factors.append("open-ended question without a specific data match")

    row_count = len(df) if df is not None else 0
    if row_count == 0:
        score -= 10
        factors.append("no matching rows were found for this question")

    ctx_text = data_context or ""
    if "only the first" in ctx_text.lower() or "truncated" in ctx_text.lower():
        score -= 10
        factors.append("result set was capped — not every matching row was shown to the model")

    if scope_note:
        score -= 8
        factors.append("follow-up switched topic and had to re-scope the data")

    score = max(30, min(97, score))
    return score, factors


def answer(question: str, history: list, user: dict,
           model_name: str = DEFAULT_MODEL, last_df=None):
    """
    Returns (stream_generator, plotly_fig_or_None, dataframe_or_None, debug_info)

    Improvements:
      - Role-aware system prompt (CEO vs HR Admin vs Country Head)
      - Intent-based max_tokens — no more silent truncation
      - free_form triggers cycle_summary + anomaly double-fetch
      - followup context capped at 150 rows
      - Empty-data instruction so AI suggests alternatives
      - Conversation context injected for 5+ turn coherence
    """
    from router import route
    from semantic import normalise, threshold_summary_for_prompt
    from conversation_state import (
        get_ctx, update_ctx, bump_topic, add_response_summary,
        clear_clarification, ctx_for_ai,
    )

    countries = user["countries"]
    ctx       = get_ctx()

    normalised_q = normalise(question)

    # ── Idea 2 & 3: Rewrite follow-up questions before routing ───────────────
    # Short pronoun-heavy questions ("when did they join?") get resolved into
    # specific questions ("when did Jian Patel join?") before the router sees them.
    # The original question is kept for display; only routing/retrieval uses rewritten.
    rewritten_q = rewrite_followup_question(question, history, last_df, model_name)
    route_q     = rewritten_q  # what the router and retrieve_data receive
    display_q   = question     # what appears in chat and debug panel

    # ── Step 1: AI Router ────────────────────────────────────────────────────
    last_df_cols = list(last_df.columns) if last_df is not None else []
    routing = route(normalise(route_q), history, last_df_cols)

    intent        = routing.get("intent", "free_form")
    needs_fresh   = routing.get("needs_fresh_join", False)
    is_followup   = routing.get("is_followup", False)
    filters       = routing.get("filters", {})
    router_reason = routing.get("reasoning", "")

    if ctx.get("clarification_pending"):
        clear_clarification()

    # ── Step 2: Country scope ─────────────────────────────────────────────────
    scoped_countries = countries
    if filters.get("country") and "ALL" not in countries:
        rc = filters["country"].upper()
        if rc in countries:
            scoped_countries = [rc]
    elif filters.get("country") and "ALL" in countries:
        scoped_countries = [filters["country"].upper()]

    # Inherit scoped country from pinned scope or conversation context
    pinned = ctx.get("pinned_country")
    if pinned and not filters.get("country"):
        if "ALL" in countries or pinned in countries:
            scoped_countries = [pinned]
    elif not filters.get("country") and ctx["active_filters"].get("country"):
        inherited = ctx["active_filters"]["country"]
        if "ALL" not in countries and inherited in countries:
            scoped_countries = [inherited]
        elif "ALL" in countries:
            scoped_countries = [inherited]

    # ── Step 3: Retrieve data ─────────────────────────────────────────────────
    from vector_store import VECTOR_STORE_ENABLED, vector_retrieve, status as vs_status

    _ALWAYS_LIVE = {
        "attainment", "underperformance", "qualifier", "proration",
        "anomaly", "cross_check", "new_joiner", "ranking",
        "cycle_summary", "country_compare", "headcount", "attrition",
        "pmgm", "employee_list", "cross_join",
        "kpi_trend", "project_compare", "tenure_compare",
        "missing_kpi", "adjustment", "scheme_config", "login", "announcement",
    }
    use_vector = (
        VECTOR_STORE_ENABLED
        and vs_status()["fr_indexed"] > 0
        and not (needs_fresh or is_followup)
        and intent not in _ALWAYS_LIVE
        and intent == "free_form"
    )

    scope_note = ""
    if needs_fresh or is_followup:
        # A follow-up needs fresh, intent-appropriate retrieval (rather than the
        # flat HR+reward enrichment below) whenever:
        #   (a) the classified intent genuinely changed from last turn's topic, or
        #   (b) the intent is one of the specialized-table intents, which can
        #       never be answered from HR+reward fields even on a repeat visit
        #       (e.g. staying on "login" two turns in a row).
        # cross_join is excluded from (a) — it's definitionally "tell me more
        # about the same employees", so the flat enrichment is exactly right
        # for it even though its "intent" differs from whatever came before.
        prev_intent = ctx.get("topic_intent")
        topic_switched = (
            prev_intent is not None and intent != prev_intent and intent != "cross_join"
        )
        needs_domain_switch = topic_switched or intent in _SPECIALIZED_DOMAIN_INTENTS

        if needs_domain_switch:
            try:
                df, data_context, scope_note = _get_scoped_fresh_context(
                    intent, scoped_countries, last_df, route_q
                )
                chart = build_chart(intent, df)
            except Exception:
                df, data_context = _get_followup_context(scoped_countries, last_df)
                chart = None
        else:
            try:
                df, data_context = _get_followup_context(scoped_countries, last_df)
                chart = None
            except Exception:
                df, data_context = retrieve_data(intent, scoped_countries, route_q)
                chart = build_chart(intent, df)

    elif intent == "free_form" and not use_vector:
        try:
            df_sum,  ctx_sum  = retrieve_data("cycle_summary", scoped_countries, route_q)
            df_anom, ctx_anom = retrieve_data("anomaly",       scoped_countries, route_q)
            df = df_sum
            data_context = (
                "[Free-form — combined cycle summary + anomaly context]\n"
                f"{ctx_sum}\n\n--- Anomaly Data ---\n{ctx_anom}"
            )
            chart = build_chart("cycle_summary", df_sum)
        except Exception:
            df, data_context = retrieve_data("free_form", scoped_countries, route_q)
            chart = None

    elif use_vector:
        country_filter = None if "ALL" in scoped_countries else scoped_countries
        df, data_context = vector_retrieve(
            route_q, top_k=30, source="both", country_filter=country_filter
        )
        chart = build_chart(intent, df) if df is not None and not df.empty else None
        data_context = f"[Semantic retrieval — top relevant records]\n{data_context}"

    else:
        df, data_context = retrieve_data(intent, scoped_countries, route_q)
        chart = build_chart(intent, df)

    # Apply scheme / status filter — only for intents where explicit filtering is appropriate.
    # Never filter by status on workforce-wide analysis intents (anomaly, attainment, etc.)
    _STATUS_FILTER_BLOCKED = {
        "anomaly", "attainment", "cycle_summary", "ranking",
        "country_compare", "project_compare", "tenure_compare",
        "kpi_trend", "pmgm", "underperformance",
    }
    if df is not None and not df.empty:
        if filters.get("scheme") and "Scheme" in df.columns:
            df = df[df["Scheme"].str.lower() == filters["scheme"].lower()]
            data_context += f"\n[Filtered to scheme: {filters['scheme']}]"
        if (filters.get("status")
                and "EmployeeStatus" in df.columns
                and intent not in _STATUS_FILTER_BLOCKED):
            df = df[df["EmployeeStatus"] == filters["status"]]
            data_context += f"\n[Filtered to status: {filters['status']}]"

    scope = "Global" if "ALL" in countries else ", ".join(countries)
    if scoped_countries != countries and "ALL" not in scoped_countries:
        scope = ", ".join(scoped_countries) + " (scoped)"

    # ── Role-aware response calibration ──────────────────────────────────────
    role = user.get("role", "")
    _EXEC_ROLES    = {"CEO", "COO", "COO — APAC"}
    _COUNTRY_ROLES = {"Country Head — SG", "Country Head — MY", "Country Head"}
    _HR_ROLES      = {"HR Admin"}

    if any(r in role for r in ["CEO", "COO"]):
        role_instruction = (
            "You are briefing a C-suite executive. Lead with the single most important "
            "finding in the first sentence. Give 2-3 specific numbers. Flag the top 1-2 "
            "concerns by name. Do not list every employee — synthesise."
        )
        list_depth = "Top 5 items max for lists unless the user asked for more."
    elif "Country Head" in role:
        role_instruction = (
            "You are briefing a Country Head who knows their team personally. "
            "Use employee names freely. Give full breakdowns for their country. "
            "Flag every individual concern — they want to act on specifics."
        )
        list_depth = "Show all flagged employees — do not cap the list."
    elif "HR" in role:
        role_instruction = (
            "You are briefing an HR Administrator who needs operational detail. "
            "Include employee IDs alongside names. Show full lists, not summaries. "
            "Be precise about dates, cycle references, and scheme names."
        )
        list_depth = "Show complete data — no truncation."
    else:
        role_instruction = "Be concise and lead with the insight."
        list_depth = "3-5 bullet points for lists unless asked for more."

    # ── Empty data instruction ────────────────────────────────────────────────
    empty_note = ""
    if df is None or (hasattr(df, "empty") and df.empty):
        empty_note = (
            "\n- The data returned is EMPTY. State this clearly. "
            "Then suggest 1-2 alternative questions the user could try "
            "(e.g. a related intent or a broader scope). "
            "Do not fabricate data or say 'no concerns found' without confirming the data is present."
        )

    chart_note = ""
    if chart is not None:
        chart_note = "\n- A chart has been generated. Always provide a written summary alongside it."

    threshold_rules = threshold_summary_for_prompt()
    conv_context_block = ctx_for_ai()
    conv_section = f"\nConversation context:\n{conv_context_block}\n" if conv_context_block else ""

    # ── Idea 4: Grounding block — tell AI exactly what data it has ───────────
    if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
        grounding_block = (
            f"\nDATA GROUNDING:\n"
            f"  Rows returned: {len(df)}\n"
            f"  Columns available: {', '.join(df.columns.tolist())}\n"
            f"  Answer ONLY using values from these columns. "
            f"  If a field is not in this list, say it is not in the current result.\n"
        )
        if rewritten_q != display_q:
            grounding_block += f"  [Question was internally resolved to: {rewritten_q}]\n"
    else:
        grounding_block = "\nDATA GROUNDING: No data rows returned for this query.\n"

    # ── Idea 5: Intent-specific answer format ─────────────────────────────────
    _FORMAT_HINTS = {
        "attainment":       "State the % hitting max, total payout, then list the top flagged names.",
        "underperformance": "List each employee with name, consecutive miss count, country, and project.",
        "anomaly":          "Group into two sections: (1) High rating / low payout, (2) Low rating / high payout. Name each employee.",
        "qualifier":        "State total failures, break down by qualifier type, then name the employees.",
        "proration":        "State how many were prorated, average factor, then list affected employees with days worked.",
        "ranking":          "Numbered list with name, payout amount, and % of max. Most/least first.",
        "cross_check":      "List each non-active employee with their name, last date, country, and payout amount.",
        "cycle_summary":    "Lead with cycle period, total payout, % hitting max, qualifier failures, then by-scheme breakdown.",
        "kpi_trend":        "Show cycle-by-cycle progression with numbers. Note direction of trend (improving/declining).",
        "project_compare":  "Ranked table: project, avg payout, employee count, qualifier failures. Highlight best and worst.",
        "country_compare":  "Ranked table: country, avg payout, employee count. Highlight gaps.",
        "tenure_compare":   "Table by tenure band with avg payout. State whether tenure correlates with payout.",
        "adjustment":       "State pending count, approved count, rejected count. List pending items with employee name and delta.",
        "scheme_config":    "List each scheme's tiers, max payout, KPI metrics and weightage, acknowledgement status.",
        "missing_kpi":      "List each missing employee with name, project, country, and last known scheme.",
        "headcount":        "Total count first, then breakdown by country/project/status table.",
        "attrition":        "Total leavers, by country and year, flag any with active payouts.",
        "pmgm":             "Distribution table (Exceptional → Unsatisfactory) with counts and % for each rating.",
        "employee_list":    "Table with name, grade, title, department, country, project, join date.",
        "cross_join":       "Answer the specific HR question directly using the employee data in context.",
        "login":            "List employees, their last login date, and flag anyone not logged in for 30+ days.",
        "announcement":     "List each announcement with message, active cycle, and date range.",
    }
    format_hint = _FORMAT_HINTS.get(intent, "Answer concisely with specific numbers from the data.")

    system = f"""You are the Orb v2 AI assistant — an executive-grade workforce intelligence analyst.
You are answering a {role} named {user["display_name"]}.
Their data scope: {scope}.
Data sources:
- Flash Reward: EmployeeID, Scheme, SchemeMaxPayout, Cycle, Metric, Target, Achieved,
  TierAchieved, MetricPayout, QualifierFailed, ProrationType, ProrFactor, PayoutEarned, TotalCyclePayout
- Flash Home: EmployeeID, EmployeeName, EmployeeStatus, JoinDate, LastDate, Country, Project,
  PMGMRating, JobTitle, EmployeeGrade, EmployeeDepartment, SupervisorID
- Supporting tables: payout_cycle, scheme_tier, incentive_matrix, scheme_acknowledgement,
  kpi_adjustment, qualifying_employee, login_audit, announcement, attendance
Today: {pd.Timestamp.today().strftime("%d %b %Y")}. Model: {model_name}.
Query intent: {intent}. Follow-up: {is_followup}. Rewritten: {rewritten_q != display_q}.
{grounding_block}
{conv_section}
{threshold_rules}

Response style for this user:
{role_instruction}
{list_depth}

ANTI-HALLUCINATION RULES — these are non-negotiable:
- Only state facts that appear in the Data Context below. Never infer, estimate, or fill gaps.
- If a specific field (e.g. JoinDate, SupervisorID) is in the COLUMNS AVAILABLE list above
  but not visible for a specific employee, say "not recorded" — do not fabricate a value.
- If a field is NOT in the columns list, say "that field is not in the current result"
  and suggest the user ask a more specific question to retrieve it.
- Never say an employee "likely" or "probably" has a value — only state what is in the data.
- Never round or approximate a number that is given exactly in the data.
- If the data has 0 rows, say exactly that and do not attempt to answer from memory.
- Do not carry over specific numbers, names, or dates from prior turns unless they appear
  in the current data context.
- If the Data Context contains a NOTE about rows being truncated/capped/a sample, treat any
  summary counts in that context as authoritative but NEVER imply the visible rows are the
  complete list — state the total count and that the user can narrow the filter to see more.

ANSWER FORMAT for this query type ({intent}):
{format_hint}

GENERAL RULES:
- Lead with the insight. Never open with "Based on the data..." or "According to..."
- Refer to employees by name. Add ID in brackets only when role is HR.
- Do NOT use markdown bold (**) or italic (*) formatting.
- Do NOT mention SQL, dataframes, router, or any internal technical details.
- Always state the cycle period and scope you are referencing.{chart_note}{empty_note}

Data context:
{data_context}
"""

    # ── Update conversation state ─────────────────────────────────────────────
    topic_summary = f"{intent} — {question[:80]}"
    bump_topic(new_intent=intent, topic_summary=topic_summary, router_filters=filters)
    update_ctx(last_df_columns=list(df.columns) if df is not None else [])

    # ── Intent-based token budget ─────────────────────────────────────────────
    max_tokens = INTENT_MAX_TOKENS.get(intent, DEFAULT_MAX_TOKENS)

    # ── Debug info ────────────────────────────────────────────────────────────
    retrieval_mode = (
        "fresh_join" if (needs_fresh or is_followup) else
        "vector"     if use_vector else
        "live_query"
    )
    # ── Human-readable "what the AI understood" label, for UI transparency ────
    _filter_bits = []
    for fkey, flabel in (("country", None), ("scheme", "Scheme"), ("cycle", "Cycle"), ("threshold", "Threshold")):
        fval = filters.get(fkey)
        if fval:
            _filter_bits.append(str(fval) if flabel is None else f"{flabel}: {fval}")
    understood_label = intent.replace("_", " ")
    if _filter_bits:
        understood_label += " · " + " · ".join(_filter_bits)

    confidence_score, confidence_factors = _estimate_confidence(
        routing, retrieval_mode, intent, df, data_context, scope_note
    )

    debug_info = {
        "routing":            routing,
        "intent":             intent,
        "retrieval_mode":     retrieval_mode,
        "data_context":       data_context,
        "system_prompt":      system,
        "conv_context":       conv_context_block,
        "max_tokens":         max_tokens,
        "rewritten_q":        rewritten_q if rewritten_q != display_q else None,
        "understood_label":   understood_label,
        "scope_note":         scope_note,
        "confidence_score":   confidence_score,
        "confidence_factors": confidence_factors,
    }

    # Use original question in history so conversation reads naturally
    messages = history[-14:] + [{"role": "user", "content": display_q}]
    stream   = stream_model(messages, system, model_name, max_tokens)
    return stream, chart, df, debug_info
