"""
generate_mock_data.py
Generates mock Excel data for Orb v2 POC.
Two sheets: flash_home, flash_reward
Run once: python generate_mock_data.py
"""
import pandas as pd
import numpy as np
from datetime import date, timedelta
import random, os

random.seed(42)
np.random.seed(42)

COUNTRIES   = ["SG", "MY", "PH", "TH", "ID"]
PROJECTS    = ["Project Alpha", "Project Beta", "Project Gamma", "Project Delta", "Project Epsilon"]
SCHEMES     = ["Scheme A", "Scheme B", "Scheme C"]
SCHEME_MAX  = {"Scheme A": 5000, "Scheme B": 8000, "Scheme C": 3500}
METRICS     = ["Revenue Target", "Customer Satisfaction", "SLA Compliance"]
QUALIFIERS  = ["Attendance Qualifier", "Compliance Qualifier", "Ethics Qualifier"]
RATINGS     = ["Exceptional", "Exceeds Expectations", "Meets Expectations", "Below Expectations", "Unsatisfactory"]
CYCLES      = ["2024-Q1", "2024-Q2", "2024-Q3", "2024-Q4", "2025-Q1", "2025-Q2"]

# Name pools — mix of names common across SG/MY/PH/TH/ID workforces
FIRST_NAMES = [
    "Wei Ling", "Jun Hao", "Mei Xin", "Kai Ming", "Hui Min", "Zhi Yang", "Si Hui", "Jia Hao",
    "Aisyah", "Aiman", "Nur Aina", "Hafiz", "Farah", "Amirul", "Nadia", "Syafiq",
    "Maria", "Jose", "Carlo", "Angelica", "Mark", "Princess", "Joshua", "Bea",
    "Somchai", "Siriporn", "Anong", "Niran", "Chanya", "Krit", "Suda", "Pakorn",
    "Budi", "Siti", "Agus", "Dewi", "Eko", "Rina", "Hendra", "Putri",
]
LAST_NAMES = [
    "Tan", "Lim", "Lee", "Wong", "Ng", "Chen", "Goh", "Teo",
    "Abdullah", "Rahman", "Ismail", "Hassan", "Yusof", "Bakar",
    "Santos", "Reyes", "Cruz", "Garcia", "Torres", "Ramos",
    "Srisai", "Charoen", "Saetang", "Wong", "Phongsri",
    "Wijaya", "Santoso", "Hartono", "Saputra", "Kusuma",
]

def _gen_employee_name(seed: int) -> str:
    rnd = random.Random(seed)  # deterministic per-employee name
    return f"{rnd.choice(FIRST_NAMES)} {rnd.choice(LAST_NAMES)}"

N_EMP = 120

# ── FLASH HOME ──────────────────────────────────────────────────────────────
def gen_flash_home():
    rows = []
    for i in range(1, N_EMP + 1):
        emp_id     = f"E{i:04d}"
        country    = random.choice(COUNTRIES)
        project    = random.choice(PROJECTS)
        join_date  = date(2018, 1, 1) + timedelta(days=random.randint(0, 2400))
        is_active  = random.random() > 0.12
        last_date  = None if is_active else join_date + timedelta(days=random.randint(180, 1800))
        rating     = random.choices(RATINGS, weights=[10, 25, 40, 18, 7])[0]
        rows.append({
            "EmployeeID":       emp_id,
            "EmployeeName":     _gen_employee_name(i),
            "EmployeeStatus":   "Active" if is_active else "Non-Active",
            "JoinDate":         join_date,
            "LastDate":         last_date,
            "Country":          country,
            "Project":          project,
            "PMGMRating":       rating,
        })
    return pd.DataFrame(rows)

# ── FLASH REWARD ─────────────────────────────────────────────────────────────
def gen_flash_reward(fh: pd.DataFrame):
    rows = []
    active_emp = fh[fh["EmployeeStatus"] == "Active"]["EmployeeID"].tolist()
    # ~10% of non-active still appear (data issue — useful for cross-check query)
    non_active_sample = fh[fh["EmployeeStatus"] == "Non-Active"]["EmployeeID"].sample(
        min(5, len(fh[fh["EmployeeStatus"] == "Non-Active"])), random_state=42).tolist()
    emp_pool = active_emp + non_active_sample

    for emp_id in emp_pool:
        scheme = random.choice(SCHEMES)
        max_pay = SCHEME_MAX[scheme]

        # Some employees are consistent underperformers (useful for Q2 card)
        is_underperformer = random.random() < 0.15
        # Some are near-miss employees
        is_near_miss = (not is_underperformer) and random.random() < 0.12

        for cycle in CYCLES:
            # Attendance / proration
            attendance_ok = random.random() > 0.18
            proration     = round(random.uniform(0.70, 0.95), 2) if not attendance_ok else 1.0

            # Qualifier
            qual_failed = random.choice(QUALIFIERS) if random.random() < 0.14 else None

            # KPI achievement per metric
            metric_rows = []
            for metric in METRICS:
                target = round(random.uniform(70, 100), 1)
                if is_underperformer:
                    achieved = round(target * random.uniform(0.50, 0.82), 1)
                elif is_near_miss:
                    achieved = round(target * random.uniform(0.91, 0.99), 1)
                else:
                    achieved = round(target * random.uniform(0.75, 1.20), 1)

                metric_pct     = min(achieved / target, 1.0)
                metric_payout  = round((max_pay / len(METRICS)) * metric_pct, 2)

                metric_rows.append({
                    "EmployeeID":       emp_id,
                    "Scheme":           scheme,
                    "SchemeMaxPayout":  max_pay,
                    "Cycle":            cycle,
                    "Metric":           metric,
                    "Target":           target,
                    "Achieved":         achieved,
                    "MetricPayout":     metric_payout if not qual_failed else 0.0,
                    "QualifierFailed":  qual_failed or "",
                    "ProrationType":    "Attendance" if not attendance_ok else "",
                    "ProrFactor":       proration,
                    "PayoutEarned":     round(sum(
                        (max_pay / len(METRICS)) * min(a / t, 1.0)
                        for a, t in [(achieved, target)]
                    ) * proration, 2) if not qual_failed else 0.0,
                })

            # Compute total payout for the cycle (sum of metric payouts × proration)
            total_metric_pay = sum(
                (max_pay / len(METRICS)) * min(r["Achieved"] / r["Target"], 1.0)
                for r in metric_rows
            )
            total_payout = round(total_metric_pay * proration, 2) if not qual_failed else 0.0

            for r in metric_rows:
                r["TotalCyclePayout"] = total_payout

            rows.extend(metric_rows)

    return pd.DataFrame(rows)

# ── WRITE EXCEL ───────────────────────────────────────────────────────────────
def main():
    out = os.path.join(os.path.dirname(__file__), "mock_data.xlsx")
    fh = gen_flash_home()
    fr = gen_flash_reward(fh)

    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        fh.to_excel(writer, sheet_name="flash_home",   index=False)
        fr.to_excel(writer, sheet_name="flash_reward", index=False)

    print(f"✅  mock_data.xlsx written — {len(fh)} employees, {len(fr)} incentive rows")

if __name__ == "__main__":
    main()
