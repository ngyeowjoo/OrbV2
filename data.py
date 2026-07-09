"""
data.py  —  Data access layer for Orb v2
Loads mock_data.xlsx and exposes country-scoped query helpers.
Swap load_data() for a real DB connector in production.

Sheets:
  flash_home              tblEmployeeSGP + enriched fields
  flash_reward            Incentive payout per metric per cycle
  payout_cycle            PayoutCycle config (dates, cutoffs)
  project_cycle           ProjectCycle dates per project
  incentive_scheme        Scheme definitions
  scheme_tier             IncentiveSchemeTier (tier thresholds)
  incentive_matrix        KPI matrix (metrics + weightage per scheme)
  scheme_acknowledgement  Employee acknowledgement status
  scheme_audit            Scheme change history
  kpi_adjustment          KPI adjustment + approval workflow
  qualifying_employee     Qualifier pass/fail per cycle
  qualifying_adj          Qualifier status adjustments
  login_audit             Last login per employee
  announcement            Active announcements per cycle
  attendance              Attendance + proration data
"""
import pandas as pd
import streamlit as st
from pathlib import Path
import hashlib

DATA_FILE = Path(__file__).parent / "mock_data.xlsx"

_DATE_COLS = {
    "flash_home":             ["JoinDate", "LastDate", "LastLoginDate"],
    "payout_cycle":           ["CycleStartDate","CycleEndDate",
                               "CycleUploadCutoffDate","CycleAdjustmentCutoffDate",
                               "CycleCutoffDate"],
    "project_cycle":          ["ProjectCycleStartDate","ProjectCycleEndDate"],
    "incentive_scheme":       ["StartDate","EndDate","CreatedDate","UpdatedDate"],
    "scheme_acknowledgement": ["AcknowledgedAt"],
    "scheme_audit":           ["ActionDate"],
    "kpi_adjustment":         ["SubmittedDate","ApprovedDate"],
    "qualifying_adj":         ["ActionDate"],
    "login_audit":            ["LastLoginDate"],
    "announcement":           ["StartDate","EndDate","CreatedDate"],
}

def _file_hash() -> str:
    try:
        return hashlib.md5(DATA_FILE.read_bytes()).hexdigest()
    except Exception:
        return "unknown"

@st.cache_data(ttl=0)
def load_data(_cache_key: str = ""):
    """Loads all sheets. _cache_key (file hash) forces reload when file changes."""
    sheets = [
        "flash_home", "flash_reward", "payout_cycle", "project_cycle",
        "incentive_scheme", "scheme_tier", "incentive_matrix",
        "scheme_acknowledgement", "scheme_audit", "kpi_adjustment",
        "qualifying_employee", "qualifying_adj", "login_audit",
        "announcement", "attendance",
    ]
    data = {}
    for s in sheets:
        try:
            parse = _DATE_COLS.get(s, [])
            data[s] = pd.read_excel(DATA_FILE, sheet_name=s,
                                    parse_dates=parse if parse else False)
        except Exception as e:
            data[s] = pd.DataFrame()
    return data

def get_data() -> dict:
    return load_data(_cache_key=_file_hash())

# ── COUNTRY-SCOPED ACCESSORS ──────────────────────────────────────────────────

def _normalise_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Convert pandas StringDtype columns to object — required for st.dataframe compatibility."""
    for col in df.columns:
        try:
            if str(df[col].dtype) in ('string', 'String') or \
               hasattr(df[col], 'dtype') and 'StringDtype' in str(type(df[col].dtype)):
                df[col] = df[col].astype(object)
        except Exception:
            pass
    return df

def get_flash_home(countries: list) -> pd.DataFrame:
    d = get_data()
    fh = d["flash_home"].copy()
    if "ALL" not in countries:
        fh = fh[fh["Country"].isin(countries)].copy()
    return _normalise_dtypes(fh)

def get_flash_reward(countries: list) -> pd.DataFrame:
    d  = get_data()
    fh = d["flash_home"]
    fr = d["flash_reward"].copy()
    if "ALL" not in countries:
        allowed = fh[fh["Country"].isin(countries)]["EmployeeID"]
        fr = fr[fr["EmployeeID"].isin(allowed)]
    return _normalise_dtypes(fr.copy())

def get_joined(countries: list) -> pd.DataFrame:
    fh = get_flash_home(countries)
    fr = get_flash_reward(countries)
    keep = [c for c in ["EmployeeID","EmployeeName","Country","Project",
                         "EmployeeStatus","JoinDate","LastDate","PMGMRating",
                         "JobTitle","EmployeeGrade","EmployeeDepartment","SupervisorID"]
            if c in fh.columns]
    return fr.merge(fh[keep], on="EmployeeID", how="left")

def get_payout_cycle() -> pd.DataFrame:
    return get_data()["payout_cycle"].copy()

def get_project_cycle(countries: list = None) -> pd.DataFrame:
    pc  = get_data()["project_cycle"].copy()
    all_projects = _country_projects(countries)
    if all_projects:
        pc = pc[pc["ProjectName"].isin(all_projects)]
    return pc

def get_incentive_scheme() -> pd.DataFrame:
    return get_data()["incentive_scheme"].copy()

def get_scheme_tier() -> pd.DataFrame:
    return get_data()["scheme_tier"].copy()

def get_incentive_matrix() -> pd.DataFrame:
    return get_data()["incentive_matrix"].copy()

def get_scheme_ack(countries: list) -> pd.DataFrame:
    d   = get_data()
    fh  = get_flash_home(countries)
    ack = d["scheme_acknowledgement"].copy()
    return ack[ack["EmployeeID"].isin(fh["EmployeeID"])].copy()

def get_scheme_audit() -> pd.DataFrame:
    return get_data()["scheme_audit"].copy()

def get_kpi_adjustment(countries: list) -> pd.DataFrame:
    d   = get_data()
    fh  = get_flash_home(countries)
    adj = d["kpi_adjustment"].copy()
    return adj[adj["EmployeeID"].isin(fh["EmployeeID"])].copy()

def get_qualifying_employee(countries: list) -> pd.DataFrame:
    d  = get_data()
    fh = get_flash_home(countries)
    qe = d["qualifying_employee"].copy()
    return qe[qe["EmployeeID"].isin(fh["EmployeeID"])].copy()

def get_qualifying_adj(countries: list) -> pd.DataFrame:
    d  = get_data()
    fh = get_flash_home(countries)
    qa = d["qualifying_adj"].copy()
    return qa[qa["EmployeeID"].isin(fh["EmployeeID"])].copy()

def get_login_audit(countries: list) -> pd.DataFrame:
    d   = get_data()
    fh  = get_flash_home(countries)
    log = d["login_audit"].copy()
    return log[log["EmployeeID"].isin(fh["EmployeeID"])].copy()

def get_announcement() -> pd.DataFrame:
    return get_data()["announcement"].copy()

def get_attendance(countries: list) -> pd.DataFrame:
    d   = get_data()
    fh  = get_flash_home(countries)
    att = d["attendance"].copy()
    return att[att["EmployeeID"].isin(fh["EmployeeID"])].copy()

# ── HELPER ────────────────────────────────────────────────────────────────────

def _country_projects(countries: list = None) -> list:
    if not countries or "ALL" in countries:
        return []
    fh = get_flash_home(countries)
    return fh["Project"].dropna().unique().tolist()


# ── SUMMARY HELPERS (used by ai_engine intent handlers) ──────────────────────

def attainment_summary(countries):
    fr = get_flash_reward(countries)
    fh = get_flash_home(countries)
    latest = fr["Cycle"].max()
    cyc = (fr[fr["Cycle"] == latest]
           .groupby("EmployeeID")
           .agg(TotalCyclePayout=("TotalCyclePayout","first"),
                SchemeMaxPayout=("SchemeMaxPayout","first"),
                Scheme=("Scheme","first"))
           .reset_index())
    cyc["HitMax"] = cyc["TotalCyclePayout"] >= cyc["SchemeMaxPayout"] * 0.999
    name_cols = [c for c in ["EmployeeID","EmployeeName","Country","Project",
                              "JobTitle","EmployeeGrade"] if c in fh.columns]
    return cyc.merge(fh[name_cols], on="EmployeeID", how="left"), latest

def underperformer_summary(countries, min_cycles=3):
    fr = get_flash_reward(countries)
    fh = get_flash_home(countries)
    fr = fr.copy()
    fr["BelowTarget"] = fr["Achieved"] < fr["Target"]
    emp_cycle = fr.groupby(["EmployeeID","Cycle"])["BelowTarget"].any().reset_index()
    emp_cycle = emp_cycle.sort_values(["EmployeeID","Cycle"])
    results = []
    for emp, grp in emp_cycle.groupby("EmployeeID"):
        consec = max_c = 0
        for v in grp["BelowTarget"]:
            consec = consec + 1 if v else 0
            max_c  = max(max_c, consec)
        if max_c >= min_cycles:
            results.append({"EmployeeID": emp, "MaxConsecutiveMisses": max_c})
    df = pd.DataFrame(results)
    if df.empty:
        return df
    name_cols = [c for c in ["EmployeeID","EmployeeName","Country","Project",
                              "JobTitle","EmployeeGrade"] if c in fh.columns]
    return df.merge(fh[name_cols], on="EmployeeID", how="left")

def qualifier_summary(countries):
    qe = get_qualifying_employee(countries)
    fh = get_flash_home(countries)
    failed = qe[qe["QualifierStatus"] == 0].copy()
    df = (failed.groupby(["EmployeeID","Qualifier"])
          .agg(TimesFailed=("Cycle","count"),
               LastFailCycle=("Cycle","max"))
          .reset_index())
    name_cols = [c for c in ["EmployeeID","EmployeeName","Country","Project"] if c in fh.columns]
    return df.merge(fh[name_cols], on="EmployeeID", how="left")

def proration_summary(countries):
    att  = get_attendance(countries)
    fh   = get_flash_home(countries)
    pror = att[att["ProrationFactor"] < 1.0].copy()
    pror["DaysAbsent"] = pror["MaxWorkingDays"] - pror["DaysWorked"]
    name_cols = [c for c in ["EmployeeID","EmployeeName","Country","Project"] if c in fh.columns]
    return pror.merge(fh[name_cols], on="EmployeeID", how="left")

def anomaly_summary(countries):
    fh = get_flash_home(countries)
    fr = get_flash_reward(countries)
    latest = fr["Cycle"].max()
    cyc = (fr[fr["Cycle"]==latest]
           .groupby("EmployeeID")
           .agg(TotalCyclePayout=("TotalCyclePayout","first"),
                SchemeMaxPayout=("SchemeMaxPayout","first"))
           .reset_index())
    name_cols = [c for c in ["EmployeeID","EmployeeName","Country","PMGMRating",
                              "Project","JobTitle","EmployeeStatus"] if c in fh.columns]
    # Use left join so all payout records are kept even if fh row is missing
    merged = cyc.merge(fh[name_cols], on="EmployeeID", how="left")
    # Anomaly is a workforce-wide analysis — restrict to Active employees
    if "EmployeeStatus" in merged.columns:
        merged = merged[merged["EmployeeStatus"] == "Active"]
    merged["PayoutPct"] = merged["TotalCyclePayout"] / merged["SchemeMaxPayout"].replace(0, 1)
    top_r    = ["Exceptional","Exceeds Expectations"]
    bot_r    = ["Below Expectations","Unsatisfactory"]
    high_low = merged[(merged["PMGMRating"].isin(top_r))  & (merged["PayoutPct"] < 0.5)]
    low_high = merged[(merged["PMGMRating"].isin(bot_r))  & (merged["PayoutPct"] >= 0.95)]
    return high_low, low_high, latest
