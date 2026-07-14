"""
generate_mock_data.py  —  Orb v2 expanded mock data generator

Generates mock_data.xlsx covering every table needed for all 57 example questions:
  - flash_home         : SAP HR data (employees with JobTitle, Grade, Supervisor, Dept)
  - flash_reward       : Incentive payout per metric per cycle
  - payout_cycle       : PayoutCycle config (cutoff dates, cycle dates)
  - project_cycle      : ProjectCycle dates per project
  - incentive_scheme   : Scheme definitions (tiers, max payout, expiry)
  - scheme_tier        : IncentiveSchemeTier (tier definitions)
  - incentive_matrix   : KPI matrix (metrics + weightage per scheme)
  - scheme_ack         : IncentiveSchemeAcknowledgement (employee ack status)
  - scheme_audit       : IncentiveSchemeAuditLog (changes + who made them)
  - kpi_adjustment     : KPIAdjustment (recorded vs adjusted KPI, approval status)
  - qualifying_employee: QualifyingEmployee + adjustments (qualifier status per cycle)
  - login_audit        : LoginAuditLog (last login per employee)
  - announcement       : Announcement (messages active during cycles)
  - attendance         : Attendance (days worked, proration data)

Run: python generate_mock_data.py
Output: mock_data.xlsx (drop-in replacement)
"""
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

random.seed(42)
np.random.seed(42)

OUT = Path(__file__).parent / "mock_data.xlsx"

# ── CONFIG ────────────────────────────────────────────────────────────────────
COUNTRIES    = ["SG", "MY", "PH", "TH", "ID"]
COUNTRY_DIST = [30, 25, 20, 15, 10]   # % of employees per country
N_EMPLOYEES  = 150

PROJECTS = {
    "SG": ["Project Alpha", "Project Sigma", "Project Nexus"],
    "MY": ["Project Beta",  "Project Omega"],
    "PH": ["Project Gamma", "Project Delta"],
    "TH": ["Project Theta"],
    "ID": ["Project Epsilon"],
}

SCHEMES = {
    "Scheme A": {"max": 5000, "metrics": ["Revenue Target", "Customer Satisfaction", "Quality Score"]},
    "Scheme B": {"max": 4000, "metrics": ["Call Handling", "First Call Resolution", "CSAT"]},
    "Scheme C": {"max": 6000, "metrics": ["Sales Volume", "Revenue Target", "Customer Retention"]},
    "Scheme D": {"max": 3500, "metrics": ["Quality Score", "Compliance Score", "Attendance"]},
}

SCHEME_COUNTRY = {
    "SG": ["Scheme A", "Scheme C"],
    "MY": ["Scheme A", "Scheme B"],
    "PH": ["Scheme B", "Scheme D"],
    "TH": ["Scheme C"],
    "ID": ["Scheme D"],
}

CYCLES = ["2024-Q1", "2024-Q2", "2024-Q3", "2024-Q4", "2025-Q1", "2025-Q2"]
CYCLE_DATES = {
    "2024-Q1": ("2024-01-01", "2024-03-31", "2024-03-15", "2024-03-25"),
    "2024-Q2": ("2024-04-01", "2024-06-30", "2024-06-15", "2024-06-25"),
    "2024-Q3": ("2024-07-01", "2024-09-30", "2024-09-15", "2024-09-25"),
    "2024-Q4": ("2024-10-01", "2024-12-31", "2024-12-15", "2024-12-25"),
    "2025-Q1": ("2025-01-01", "2025-03-31", "2025-03-15", "2025-03-25"),
    "2025-Q2": ("2025-04-01", "2025-06-30", "2025-06-15", "2025-06-25"),
}

JOB_GRADES = ["G1", "G2", "G3", "G4", "G5"]
JOB_TITLES = {
    "G1": ["Customer Service Agent", "Sales Agent", "Support Specialist"],
    "G2": ["Senior Agent", "Team Lead Specialist", "Quality Analyst"],
    "G3": ["Team Lead", "Senior Analyst", "Operations Coordinator"],
    "G4": ["Assistant Manager", "Operations Manager"],
    "G5": ["Senior Manager", "Country Head"],
}
DEPARTMENTS = ["Operations", "Sales", "Customer Experience", "Quality Assurance", "Support"]
PMGM_RATINGS = [
    "Exceptional", "Exceeds Expectations", "Meets Expectations",
    "Below Expectations", "Unsatisfactory"
]
PMGM_WEIGHTS = [0.08, 0.25, 0.50, 0.12, 0.05]

QUALIFIERS = ["Compliance", "Attendance", "Ethics", "Performance Gate"]

MALE_FIRST  = ["Ahmad","Raj","Wei","Somchai","Budi","Kevin","James","Marcus","Daniel","Arif","Jian","Prayut","Deni"]
FEMALE_FIRST= ["Aisyah","Siti","Mei","Nok","Dewi","Sarah","Linda","Priya","Fiona","Nurul","Xin","Malee","Ratih"]
LAST_NAMES  = ["Torres","Lee","Lim","Tan","Hassan","Patel","Kumar","Wong","Chen","Santos",
               "Garcia","Reyes","Nguyen","Ibrahim","Suryadi","Chandra","Kowalski","Mendoza"]

def _name():
    first = random.choice(MALE_FIRST + FEMALE_FIRST)
    return f"{first} {random.choice(LAST_NAMES)}"

def _date(start: str, end: str) -> datetime:
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end,   "%Y-%m-%d")
    return s + timedelta(days=random.randint(0, (e - s).days))


# ═══════════════════════════════════════════════════════════════════════════════
# 1. FLASH HOME (tblEmployeeSGP enriched)
# ═══════════════════════════════════════════════════════════════════════════════
def build_flash_home():
    rows = []
    emp_id = 1

    # Distribute employees across countries
    country_counts = {c: int(N_EMPLOYEES * p / 100) for c, p in zip(COUNTRIES, COUNTRY_DIST)}
    # Make up the rounding difference
    country_counts["SG"] += N_EMPLOYEES - sum(country_counts.values())

    # Build supervisor pool first (G4/G5 — one per project)
    supervisors = {}  # project -> EmployeeID

    for country, count in country_counts.items():
        projects = PROJECTS[country]
        for i in range(count):
            eid      = f"E{emp_id:04d}"
            # First employee per project is a supervisor (G4/G5)
            is_super = (i < len(projects))
            grade    = random.choice(["G4","G5"]) if is_super else random.choice(["G1","G2","G3"])
            project  = projects[i % len(projects)]
            title    = random.choice(JOB_TITLES[grade])

            # Track first supervisor per project
            if is_super and project not in supervisors:
                supervisors[project] = eid

            # Join date: spread across 2018-2024
            join_date = _date("2018-01-01", "2024-10-01")

            # Status: ~12% non-active (left in last 2 years)
            if random.random() < 0.12:
                last_date = _date("2023-01-01", "2025-06-01")
                status    = "Non-Active"
            else:
                last_date = None
                status    = "Active"

            pmgm = np.random.choice(PMGM_RATINGS, p=PMGM_WEIGHTS)

            rows.append({
                "EmployeeID":            eid,
                "EmployeeName":          _name(),
                "EmployeeStatus":        status,
                "JoinDate":              join_date,
                "LastDate":              last_date,
                "Country":               country,
                "Project":               project,
                "PMGMRating":            pmgm,
                "JobTitle":              title,
                "EmployeeGrade":         grade,
                "EmployeeDepartment":    random.choice(DEPARTMENTS),
                "SupervisorID":          None,   # filled in below
                "EmployeeType":          "Full Time",
                "EmployeeEmail":         f"{eid.lower()}@company.com",
                "LastLoginDate":         _date("2025-03-01", "2025-06-30"),
            })
            emp_id += 1

    fh = pd.DataFrame(rows)

    # Assign supervisors (project head supervises everyone in that project)
    def _sup(row):
        sup = supervisors.get(row["Project"])
        return sup if sup != row["EmployeeID"] else None

    fh["SupervisorID"] = fh.apply(_sup, axis=1)

    today = datetime.today()
    recent_dates = [
        today - timedelta(days=15),
        today - timedelta(days=32),
        today - timedelta(days=55),
        today - timedelta(days=78),
    ]
    for i, d in enumerate(recent_dates):
        fh.loc[fh.index[-(i+1)], "JoinDate"]        = d.replace(hour=0, minute=0, second=0, microsecond=0)
        fh.loc[fh.index[-(i+1)], "EmployeeStatus"]  = "Active"
        fh.loc[fh.index[-(i+1)], "LastDate"]        = None

    return fh, supervisors


# ═══════════════════════════════════════════════════════════════════════════════
# 2. FLASH REWARD (enriched with TierAchieved)
# ═══════════════════════════════════════════════════════════════════════════════
def build_flash_reward(fh: pd.DataFrame):
    rows = []
    active = fh[fh["EmployeeStatus"] == "Active"]

    # Employees to skip from some cycles (to test missing_kpi intent)
    skip_employees = set(fh[fh["EmployeeStatus"] == "Active"]["EmployeeID"].sample(5).tolist())

    for _, emp in fh.iterrows():
        eid     = emp["EmployeeID"]
        country = emp["Country"]
        scheme  = random.choice(SCHEME_COUNTRY[country])
        metrics = SCHEMES[scheme]["metrics"]
        max_pay = SCHEMES[scheme]["max"]

        # Non-active: only include cycles before their last date
        last_date = emp["LastDate"]

        for cycle in CYCLES:
            cycle_end_str = CYCLE_DATES[cycle][1]
            cycle_end     = datetime.strptime(cycle_end_str, "%Y-%m-%d")

            # Skip if employee left before this cycle
            if last_date and pd.notna(last_date):
                ld = last_date if isinstance(last_date, datetime) else pd.Timestamp(last_date).to_pydatetime()
                if ld < cycle_end - timedelta(days=60):
                    continue

            # Skip last cycle for some active employees (missing_kpi scenario)
            if cycle == "2025-Q2" and eid in skip_employees:
                continue

            # Qualifier failure: ~8% chance
            qual_failed = None
            if random.random() < 0.08:
                qual_failed = random.choice(QUALIFIERS)

            # Proration: ~15% chance of attendance proration
            pror_type   = None
            pror_factor = 1.0
            if random.random() < 0.15:
                pror_type   = "Attendance"
                pror_factor = round(random.uniform(0.6, 0.95), 3)

            # Underperformer pattern: ~10% of employees consistently miss
            is_underperformer = hash(eid) % 10 == 0
            perf_bias = 0.5 if is_underperformer else random.uniform(0.7, 1.15)

            total_metric_payout = 0
            metric_rows = []
            for metric in metrics:
                target   = round(random.uniform(80, 120), 1)
                achieved = round(target * perf_bias * random.uniform(0.85, 1.15), 1)
                # Tier achieved
                pct = achieved / target if target > 0 else 0
                if pct >= 1.0:       tier = "Platinum"
                elif pct >= 0.9:     tier = "Gold"
                elif pct >= 0.75:    tier = "Silver"
                else:                tier = "Bronze"

                payout_per_metric = round((max_pay / len(metrics)) * min(pct, 1.0), 2)
                if qual_failed:
                    payout_per_metric = 0
                total_metric_payout += payout_per_metric

                metric_rows.append({
                    "EmployeeID":       eid,
                    "Scheme":           scheme,
                    "SchemeMaxPayout":  max_pay,
                    "Cycle":            cycle,
                    "Metric":           metric,
                    "Target":           target,
                    "Achieved":         achieved,
                    "TierAchieved":     tier,
                    "MetricPayout":     payout_per_metric,
                    "QualifierFailed":  qual_failed,
                    "ProrationType":    pror_type,
                    "ProrFactor":       pror_factor,
                })

            # Apply proration to total
            total_payout = round(total_metric_payout * pror_factor, 2)

            for mr in metric_rows:
                mr["PayoutEarned"]     = round(mr["MetricPayout"] * pror_factor, 2)
                mr["TotalCyclePayout"] = total_payout
            rows.extend(metric_rows)

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. PAYOUT CYCLE
# ═══════════════════════════════════════════════════════════════════════════════
def build_payout_cycle():
    rows = []
    for i, (cycle, dates) in enumerate(CYCLE_DATES.items(), start=1):
        start, end, cutoff, adj_cutoff = dates
        rows.append({
            "PayoutCycleID":              i,
            "CycleName":                  cycle,
            "SiteID":                     1,
            "CycleStartDate":             datetime.strptime(start,      "%Y-%m-%d"),
            "CycleEndDate":               datetime.strptime(end,        "%Y-%m-%d"),
            "CycleUploadCutoffDate":      datetime.strptime(cutoff,     "%Y-%m-%d"),
            "CycleAdjustmentCutoffDate":  datetime.strptime(adj_cutoff, "%Y-%m-%d"),
            "CycleCutoffDate":            datetime.strptime(end,        "%Y-%m-%d"),
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. PROJECT CYCLE
# ═══════════════════════════════════════════════════════════════════════════════
def build_project_cycle(payout_cycle_df: pd.DataFrame):
    all_projects = [p for ps in PROJECTS.values() for p in ps]
    rows = []
    for _, pc in payout_cycle_df.iterrows():
        for proj_id, proj in enumerate(all_projects, start=1):
            rows.append({
                "ProjectCycleID":       f"PC{pc['PayoutCycleID']:02d}{proj_id:02d}",
                "ProjectName":          proj,
                "PayoutCycleID":        pc["PayoutCycleID"],
                "CycleName":            pc["CycleName"],
                "ProjectCycleStartDate": pc["CycleStartDate"],
                "ProjectCycleEndDate":   pc["CycleEndDate"],
            })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. INCENTIVE SCHEME
# ═══════════════════════════════════════════════════════════════════════════════
def build_incentive_scheme():
    rows = []
    for i, (name, cfg) in enumerate(SCHEMES.items(), start=1):
        rows.append({
            "SchemeID":      i,
            "SchemeCode":    name.replace(" ", "_").upper(),
            "SchemeName":    name,
            "MaxPayout":     cfg["max"],
            "PayoutType":    "Monthly",
            "StartDate":     datetime(2024, 1, 1),
            "EndDate":       datetime(2025, 12, 31),
            "Status":        "Active",
            "Remarks":       f"Standard incentive scheme — {name}",
            "CreatedBy":     "admin",
            "CreatedDate":   datetime(2023, 12, 1),
            "UpdatedBy":     "admin",
            "UpdatedDate":   datetime(2024, 1, 15),
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. SCHEME TIERS (IncentiveSchemeTier)
# ═══════════════════════════════════════════════════════════════════════════════
def build_scheme_tier(scheme_df: pd.DataFrame):
    tiers = [
        ("Bronze",   0.0,  0.75,  ">=", 0.25),
        ("Silver",   0.75, 0.90,  ">=", 0.60),
        ("Gold",     0.90, 1.0,   ">=", 0.85),
        ("Platinum", 1.0,  999.0, ">=", 1.00),
    ]
    rows = []
    for _, scheme in scheme_df.iterrows():
        for tier_name, target_min, target_max, op, payout_pct in tiers:
            rows.append({
                "TierID":        f"{scheme['SchemeID']}_{tier_name}",
                "SchemeID":      scheme["SchemeID"],
                "SchemeName":    scheme["SchemeName"],
                "Tier":          tier_name,
                "TargetMin":     target_min,
                "TargetMax":     target_max if target_max < 999 else None,
                "Operator":      op,
                "PayoutPct":     payout_pct,
                "PayoutAmount":  round(scheme["MaxPayout"] * payout_pct, 2),
            })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. INCENTIVE MATRIX (KPI weightage per scheme)
# ═══════════════════════════════════════════════════════════════════════════════
def build_incentive_matrix(scheme_df: pd.DataFrame):
    rows = []
    for _, scheme in scheme_df.iterrows():
        metrics = SCHEMES[scheme["SchemeName"]]["metrics"]
        n       = len(metrics)
        # Distribute weightage: first metric gets more weight
        weights = [0.4] + [0.6 / (n - 1)] * (n - 1) if n > 1 else [1.0]
        for metric, weight in zip(metrics, weights):
            rows.append({
                "MatrixID":      f"{scheme['SchemeID']}_{metric[:8]}",
                "SchemeID":      scheme["SchemeID"],
                "SchemeName":    scheme["SchemeName"],
                "Metric":        metric,
                "Weightage":     round(weight * 100, 1),   # as %
                "Description":   f"{metric} KPI for {scheme['SchemeName']}",
            })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# 8. SCHEME ACKNOWLEDGEMENT
# ═══════════════════════════════════════════════════════════════════════════════
def build_scheme_ack(fh: pd.DataFrame, scheme_df: pd.DataFrame):
    rows = []
    country_scheme = {}
    for country, schemes in SCHEME_COUNTRY.items():
        for s in schemes:
            country_scheme.setdefault(s, []).append(country)

    for _, emp in fh[fh["EmployeeStatus"] == "Active"].iterrows():
        country = emp["Country"]
        for scheme_name in SCHEME_COUNTRY[country]:
            scheme_row = scheme_df[scheme_df["SchemeName"] == scheme_name]
            if scheme_row.empty:
                continue
            sid    = int(scheme_row.iloc[0]["SchemeID"])
            # 85% acknowledged, 15% pending
            status = "Acknowledged" if random.random() < 0.85 else "Pending"
            ack_date = _date("2024-01-10", "2024-02-15") if status == "Acknowledged" else None
            rows.append({
                "AckID":          f"ACK_{emp['EmployeeID']}_{sid}",
                "EmployeeID":     emp["EmployeeID"],
                "EmployeeName":   emp["EmployeeName"],
                "SchemeID":       sid,
                "SchemeName":     scheme_name,
                "AckStatus":      status,
                "AcknowledgedAt": ack_date,
            })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# 9. SCHEME AUDIT LOG
# ═══════════════════════════════════════════════════════════════════════════════
def build_scheme_audit(scheme_df: pd.DataFrame):
    actions = ["Created", "Approved", "Updated MaxPayout", "Updated Tier", "Activated", "Deactivated"]
    actors  = ["john.admin", "sarah.hrd", "mike.ops", "admin", "system"]
    rows = []
    for _, scheme in scheme_df.iterrows():
        n_changes = random.randint(2, 5)
        base_date = datetime(2023, 11, 1)
        for j in range(n_changes):
            action_date = base_date + timedelta(days=j * random.randint(3, 20))
            rows.append({
                "AuditID":       f"SA_{scheme['SchemeID']}_{j}",
                "SchemeID":      scheme["SchemeID"],
                "SchemeName":    scheme["SchemeName"],
                "ActionTaken":   random.choice(actions),
                "ActionRemark":  f"Change #{j+1} to {scheme['SchemeName']}",
                "ActionBy":      random.choice(actors),
                "ActionDate":    action_date,
            })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# 10. KPI ADJUSTMENTS
# ═══════════════════════════════════════════════════════════════════════════════
def build_kpi_adjustment(fr: pd.DataFrame, fh: pd.DataFrame):
    rows = []
    statuses = ["Pending", "Approved", "Rejected"]
    status_w = [0.3, 0.55, 0.15]

    latest_cycle = fr["Cycle"].max()
    latest_fr    = fr[fr["Cycle"] == latest_cycle].drop_duplicates(["EmployeeID", "Metric"])

    # ~15% of latest cycle records have an adjustment
    sample = latest_fr.sample(frac=0.15, random_state=42)

    for i, (_, row) in enumerate(sample.iterrows()):
        status        = np.random.choice(statuses, p=status_w)
        recorded_kpi  = row["Achieved"]
        adjusted_kpi  = round(recorded_kpi * random.uniform(0.9, 1.15), 2)
        adj_pct       = adjusted_kpi / row["Target"] if row["Target"] > 0 else 0
        adj_payout    = round((row["SchemeMaxPayout"] / 3) * min(adj_pct, 1.0), 2)
        emp_name      = fh[fh["EmployeeID"] == row["EmployeeID"]]["EmployeeName"].values
        emp_name      = emp_name[0] if len(emp_name) else row["EmployeeID"]

        rows.append({
            "AdjustmentID":    f"ADJ_{i:04d}",
            "EmployeeID":      row["EmployeeID"],
            "EmployeeName":    emp_name,
            "Cycle":           row["Cycle"],
            "Metric":          row["Metric"],
            "Scheme":          row["Scheme"],
            "RecordedKPI":     recorded_kpi,
            "AdjustedKPI":     adjusted_kpi,
            "RecordedPayout":  row["MetricPayout"],
            "AdjustedPayout":  adj_payout,
            "RecordedTier":    row["TierAchieved"],
            "AdjustmentRemark": f"System adjustment for {row['Metric']}",
            "ApprovalRemark":   "Reviewed and approved by team lead" if status == "Approved" else (
                                "Data verified — rejected" if status == "Rejected" else ""),
            "AdjStatus":       status,
            "SubmittedBy":     "team.lead",
            "SubmittedDate":   _date("2025-05-01", "2025-06-10"),
            "ApprovedBy":      "ops.manager" if status in ("Approved","Rejected") else None,
            "ApprovedDate":    _date("2025-05-10", "2025-06-20") if status in ("Approved","Rejected") else None,
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# 11. QUALIFYING EMPLOYEE (qualifier status per employee per cycle)
# ═══════════════════════════════════════════════════════════════════════════════
def build_qualifying_employee(fh: pd.DataFrame, fr: pd.DataFrame):
    rows = []
    adj_rows = []
    latest = fr["Cycle"].max()

    for _, emp in fh.iterrows():
        eid = emp["EmployeeID"]
        for cycle in CYCLES:
            cycle_end = datetime.strptime(CYCLE_DATES[cycle][1], "%Y-%m-%d")
            if emp["LastDate"] and pd.notna(emp["LastDate"]):
                ld = emp["LastDate"] if isinstance(emp["LastDate"], datetime) else pd.Timestamp(emp["LastDate"]).to_pydatetime()
                if ld < cycle_end - timedelta(days=60):
                    continue

            for qual in QUALIFIERS:
                # ~8% fail per qualifier per cycle
                failed = random.random() < 0.08
                status = 0 if failed else 1   # 0 = failed, 1 = passed

                qid = f"QE_{eid}_{cycle}_{qual[:3]}"
                rows.append({
                    "QualifyingID":   qid,
                    "EmployeeID":     eid,
                    "EmployeeName":   emp["EmployeeName"],
                    "Cycle":          cycle,
                    "Qualifier":      qual,
                    "QualifierStatus": status,  # 1=pass, 0=fail
                    "StatusLabel":    "Pass" if status else "Fail",
                    "Remarks":        "" if status else f"Failed {qual} check",
                })

                # ~20% of failures have a status adjustment
                if not failed:
                    continue
                if random.random() < 0.20:
                    adj_status = random.choice(["Approved", "Rejected"])
                    adj_rows.append({
                        "QualAdjID":      f"QA_{qid}",
                        "QualifyingID":   qid,
                        "EmployeeID":     eid,
                        "Cycle":          cycle,
                        "Qualifier":      qual,
                        "RecordedStatus": "Fail",
                        "AdjustedStatus": "Pass" if adj_status == "Approved" else "Fail",
                        "AdjStatus":      adj_status,
                        "AdjRemark":      "Special circumstance — override approved" if adj_status == "Approved"
                                          else "Override rejected after review",
                        "ActionBy":       "ops.manager",
                        "ActionDate":     _date("2025-05-01", "2025-06-15"),
                    })

    return pd.DataFrame(rows), pd.DataFrame(adj_rows)


# ═══════════════════════════════════════════════════════════════════════════════
# 12. LOGIN AUDIT LOG
# ═══════════════════════════════════════════════════════════════════════════════
def build_login_audit(fh: pd.DataFrame):
    rows = []
    for _, emp in fh.iterrows():
        # Last login within last 3 months; non-active employees less recent
        if emp["EmployeeStatus"] == "Active":
            last_login = _date("2025-04-01", "2025-06-30")
        else:
            last_login = _date("2024-01-01", "2025-01-01")
        rows.append({
            "LogID":            f"LOG_{emp['EmployeeID']}",
            "EmployeeID":       emp["EmployeeID"],
            "EmployeeName":     emp["EmployeeName"],
            "IsLoginSuccessful": True,
            "LastLoginDate":    last_login,
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# 13. ANNOUNCEMENT
# ═══════════════════════════════════════════════════════════════════════════════
def build_announcement():
    messages = [
        "Q2 2025 KPI upload window is now open. Please upload by June 15.",
        "Reminder: Q1 2025 incentive payout has been processed.",
        "New Scheme D effective from Q3 2024 — please acknowledge.",
        "System maintenance scheduled — portal unavailable June 28, 2-4AM.",
        "Q2 2025 payout cycle adjustment cutoff: June 25, 2025.",
        "Annual incentive scheme review completed. Updates effective Q3 2025.",
    ]
    rows = []
    for i, msg in enumerate(messages, start=1):
        # Active during a cycle window
        cycle_key = CYCLES[i % len(CYCLES)]
        s, e      = CYCLE_DATES[cycle_key][0], CYCLE_DATES[cycle_key][1]
        rows.append({
            "AnnouncementID": i,
            "Message":        msg,
            "StartDate":      datetime.strptime(s, "%Y-%m-%d"),
            "EndDate":        datetime.strptime(e, "%Y-%m-%d"),
            "CycleName":      cycle_key,
            "IsActive":       True,
            "CreatedBy":      "admin",
            "CreatedDate":    datetime.strptime(s, "%Y-%m-%d") - timedelta(days=5),
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# 14. ATTENDANCE
# ═══════════════════════════════════════════════════════════════════════════════
def build_attendance(fh: pd.DataFrame):
    rows = []
    for _, emp in fh.iterrows():
        for i, cycle in enumerate(CYCLES, start=1):
            max_days  = 65  # ~3-month quarter
            # Prorated employees: days_worked < threshold
            if hash(emp["EmployeeID"] + cycle) % 7 == 0:
                days_worked = random.randint(40, 58)
                threshold   = 60
            else:
                days_worked = random.randint(61, 65)
                threshold   = 60
            pror_factor = min(1.0, round(days_worked / threshold, 3))
            rows.append({
                "AttendanceID":          f"ATT_{emp['EmployeeID']}_{i}",
                "EmployeeID":            emp["EmployeeID"],
                "EmployeeName":          emp["EmployeeName"],
                "Cycle":                 cycle,
                "PayoutCycleID":         i,
                "MaxWorkingDays":         max_days,
                "DaysWorked":            days_worked,
                "ProrationUnitThreshold": threshold,
                "ProrationFactor":        pror_factor,
                "Country":               emp["Country"],
                "Project":               emp["Project"],
            })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN — BUILD AND WRITE
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Building mock data...")

    fh, supervisors = build_flash_home()
    print(f"  flash_home:           {len(fh):>5} rows")

    fr = build_flash_reward(fh)
    print(f"  flash_reward:         {len(fr):>5} rows")

    pc = build_payout_cycle()
    print(f"  payout_cycle:         {len(pc):>5} rows")

    proj_cycle = build_project_cycle(pc)
    print(f"  project_cycle:        {len(proj_cycle):>5} rows")

    scheme = build_incentive_scheme()
    print(f"  incentive_scheme:     {len(scheme):>5} rows")

    scheme_tier = build_scheme_tier(scheme)
    print(f"  scheme_tier:          {len(scheme_tier):>5} rows")

    matrix = build_incentive_matrix(scheme)
    print(f"  incentive_matrix:     {len(matrix):>5} rows")

    ack = build_scheme_ack(fh, scheme)
    print(f"  scheme_acknowledgement:{len(ack):>4} rows")

    audit = build_scheme_audit(scheme)
    print(f"  scheme_audit:         {len(audit):>5} rows")

    adj = build_kpi_adjustment(fr, fh)
    print(f"  kpi_adjustment:       {len(adj):>5} rows")

    qe, qa = build_qualifying_employee(fh, fr)
    print(f"  qualifying_employee:  {len(qe):>5} rows")
    print(f"  qualifying_adj:       {len(qa):>5} rows")

    login = build_login_audit(fh)
    print(f"  login_audit:          {len(login):>5} rows")

    ann = build_announcement()
    print(f"  announcement:         {len(ann):>5} rows")

    att = build_attendance(fh)
    print(f"  attendance:           {len(att):>5} rows")

    print(f"\nWriting to {OUT}...")
    with pd.ExcelWriter(OUT, engine="openpyxl") as writer:
        fh.to_excel(writer,         sheet_name="flash_home",            index=False)
        fr.to_excel(writer,         sheet_name="flash_reward",           index=False)
        pc.to_excel(writer,         sheet_name="payout_cycle",           index=False)
        proj_cycle.to_excel(writer, sheet_name="project_cycle",          index=False)
        scheme.to_excel(writer,     sheet_name="incentive_scheme",       index=False)
        scheme_tier.to_excel(writer,sheet_name="scheme_tier",            index=False)
        matrix.to_excel(writer,     sheet_name="incentive_matrix",       index=False)
        ack.to_excel(writer,        sheet_name="scheme_acknowledgement",  index=False)
        audit.to_excel(writer,      sheet_name="scheme_audit",           index=False)
        adj.to_excel(writer,        sheet_name="kpi_adjustment",         index=False)
        qe.to_excel(writer,         sheet_name="qualifying_employee",    index=False)
        qa.to_excel(writer,         sheet_name="qualifying_adj",         index=False)
        login.to_excel(writer,      sheet_name="login_audit",            index=False)
        ann.to_excel(writer,        sheet_name="announcement",           index=False)
        att.to_excel(writer,        sheet_name="attendance",             index=False)

    print(f"Done. {OUT}")
