"""
auth.py  —  Login simulation for Orb v2 POC
Replace with Flashcredentials OAuth in production.
"""

# username → { password, display_name, role, countries, data_types }
USERS = {
    "ceo": {
        "password":     "demo",
        "display_name": "Sarah Chen",
        "role":         "CEO",
        "countries":    ["ALL"],
        "data_types":   ["HR", "Incentive", "Finance", "Payroll"],
        "avatar":       "SC",
    },
    "coo.apac": {
        "password":     "demo",
        "display_name": "James Lim",
        "role":         "COO — APAC",
        "countries":    ["SG", "MY", "PH", "TH", "ID"],
        "data_types":   ["HR", "Incentive"],
        "avatar":       "JL",
    },
    "head.sg": {
        "password":     "demo",
        "display_name": "Priya Nair",
        "role":         "Country Head — SG",
        "countries":    ["SG"],
        "data_types":   ["HR", "Incentive"],
        "avatar":       "PN",
    },
    "hr.admin": {
        "password":     "demo",
        "display_name": "Mark Tan",
        "role":         "HR Admin",
        "countries":    ["SG", "MY"],
        "data_types":   ["HR"],
        "avatar":       "MT",
    },
}

def authenticate(username: str, password: str):
    """Returns user dict or None."""
    u = USERS.get(username.lower().strip())
    if u and u["password"] == password:
        return {k: v for k, v in u.items() if k != "password"}
    return None

def scope_label(user: dict) -> str:
    c = user["countries"]
    return "Global" if "ALL" in c else " · ".join(c)
