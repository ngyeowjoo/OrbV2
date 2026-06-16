"""
data.py  —  Data access layer for Orb v2 POC
Loads mock_data.xlsx and exposes scoped query helpers.
Swap load_data() for a MySQL connector in production.
"""
import pandas as pd
import streamlit as st
from pathlib import Path
import hashlib

DATA_FILE = Path(__file__).parent / "mock_data.xlsx"

def _file_hash() -> str:
    """MD5 of the Excel file — used to bust Streamlit's data cache when file changes."""
    try:
        return hashlib.md5(DATA_FILE.read_bytes()).hexdigest()
    except Exception:
        return "unknown"

@st.cache_data(ttl=0)   # ttl=0 means: re-read on every app restart, no stale cache
def load_data(_cache_key: str = ""):
    """_cache_key is the file hash — changing it forces a cache bust."""
    fh = pd.read_excel(DATA_FILE, sheet_name="flash_home",   parse_dates=["JoinDate", "LastDate"])
    fr = pd.read_excel(DATA_FILE, sheet_name="flash_reward")
    return fh, fr

def get_data():
    """Always passes the current file hash so the cache reloads when mock_data.xlsx changes."""
    return load_data(_cache_key=_file_hash())

# ── COUNTRY-SCOPED ACCESSORS ──────────────────────────────────────────────────
def get_flash_home(countries: list[str]) -> pd.DataFrame:
    fh, _ = get_data()
    if "ALL" in countries:
        return fh.copy()
    return fh[fh["Country"].isin(countries)].copy()

def get_flash_reward(countries: list[str]) -> pd.DataFrame:
    fh, fr = get_data()
    if "ALL" not in countries:
        allowed_ids = fh[fh["Country"].isin(countries)]["EmployeeID"]
        fr = fr[fr["EmployeeID"].isin(allowed_ids)]
    return fr.copy()

def get_joined(countries: list[str]) -> pd.DataFrame:
    """Flash Reward joined with Flash Home — country scoped."""
    fh = get_flash_home(countries)
    fr = get_flash_reward(countries)
    name_col = ["EmployeeName"] if "EmployeeName" in fh.columns else []
    return fr.merge(fh[["EmployeeID"] + name_col + ["Country", "Project", "EmployeeStatus",
                         "JoinDate", "LastDate", "PMGMRating"]],
                    on="EmployeeID", how="left")

# ── SUMMARY HELPERS (used by intent handlers) ─────────────────────────────────
def attainment_summary(countries):
    fr = get_flash_reward(countries)
    fh = get_flash_home(countries)
    latest = fr["Cycle"].max()
    cyc    = fr[fr["Cycle"] == latest].drop_duplicates(["EmployeeID", "Cycle"])
    cyc    = cyc.groupby("EmployeeID").agg(
        TotalCyclePayout=("TotalCyclePayout", "first"),
        SchemeMaxPayout=("SchemeMaxPayout",   "first"),
        Scheme=("Scheme",                     "first"),
    ).reset_index()
    cyc["HitMax"] = cyc["TotalCyclePayout"] >= cyc["SchemeMaxPayout"] * 0.999
    # Enrich with name, country, project from Flash Home
    name_cols = [c for c in ["EmployeeID","EmployeeName","Country","Project"] if c in fh.columns]
    cyc = cyc.merge(fh[name_cols], on="EmployeeID", how="left")
    return cyc, latest

def underperformer_summary(countries, min_cycles=3):
    fr = get_flash_reward(countries)
    fh = get_flash_home(countries)
    fr["BelowTarget"] = fr["Achieved"] < fr["Target"]
    emp_cycle = fr.groupby(["EmployeeID", "Cycle"])["BelowTarget"].any().reset_index()
    emp_cycle = emp_cycle.sort_values(["EmployeeID", "Cycle"])

    results = []
    for emp, grp in emp_cycle.groupby("EmployeeID"):
        consecutive = 0
        max_consec  = 0
        for val in grp["BelowTarget"]:
            if val:
                consecutive += 1
                max_consec = max(max_consec, consecutive)
            else:
                consecutive = 0
        if max_consec >= min_cycles:
            results.append({"EmployeeID": emp, "MaxConsecutiveMisses": max_consec})
    df = pd.DataFrame(results)
    if df.empty:
        return df
    name_cols = [c for c in ["EmployeeID","EmployeeName","Country","Project"] if c in fh.columns]
    df = df.merge(fh[name_cols], on="EmployeeID", how="left")
    return df

def qualifier_summary(countries):
    fr = get_flash_reward(countries)
    fh = get_flash_home(countries)
    failed = fr[fr["QualifierFailed"] != ""].copy()
    df = failed.groupby(["EmployeeID", "QualifierFailed"]).agg(
        TimesFailed=("Cycle", "count"),
        TotalPayoutBlocked=("MetricPayout", "sum")
    ).reset_index()
    name_cols = [c for c in ["EmployeeID","EmployeeName","Country"] if c in fh.columns]
    df = df.merge(fh[name_cols], on="EmployeeID", how="left")
    return df

def proration_summary(countries):
    fr = get_flash_reward(countries)
    prorated = fr[fr["ProrFactor"] < 1.0].drop_duplicates(["EmployeeID", "Cycle"])
    prorated["PayoutLost"] = (prorated["SchemeMaxPayout"] - prorated["TotalCyclePayout"]).clip(lower=0)
    return prorated

def anomaly_summary(countries):
    fh = get_flash_home(countries)
    fr = get_flash_reward(countries)
    latest = fr["Cycle"].max()
    cyc = fr[fr["Cycle"] == latest].groupby("EmployeeID").agg(
        TotalCyclePayout=("TotalCyclePayout", "first"),
        SchemeMaxPayout=("SchemeMaxPayout",   "first"),
    ).reset_index()
    name_cols = [c for c in ["EmployeeID","EmployeeName","Country","PMGMRating"] if c in fh.columns]
    merged = cyc.merge(fh[name_cols], on="EmployeeID", how="inner")
    merged["PayoutPct"] = merged["TotalCyclePayout"] / merged["SchemeMaxPayout"]
    top_ratings   = ["Exceptional", "Exceeds Expectations"]
    bottom_ratings= ["Below Expectations", "Unsatisfactory"]
    high_low = merged[(merged["PMGMRating"].isin(top_ratings))    & (merged["PayoutPct"] < 0.5)]
    low_high  = merged[(merged["PMGMRating"].isin(bottom_ratings)) & (merged["PayoutPct"] >= 0.95)]
    return high_low, low_high, latest
