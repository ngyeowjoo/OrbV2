Orb v2 — Workforce Intelligence Platform
> Conversational AI interface for executive workforce and incentive analytics.
> Chat-first. Side-panel visualisations. Country-scoped access control.
---
Demo Accounts
Username	Password	Role	Scope
`ceo`	`demo`	CEO	Global
`coo.apac`	`demo`	COO — APAC	SG MY PH TH ID
`head.sg`	`demo`	Country Head — SG	SG only
`hr.admin`	`demo`	HR Admin	SG + MY
---
Quick Start (Local)
```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/orb-v2.git
cd orb-v2

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate mock data
python generate_mock_data.py

# 5. Set your Anthropic API key
export ANTHROPIC_API_KEY=sk-ant-...   # Windows: set ANTHROPIC_API_KEY=sk-ant-...

# 6. Run
streamlit run app.py
```
---
Deploy to Streamlit Cloud
Push this repo to GitHub (all files including `mock_data.xlsx` after running step 4 locally, or let the cloud run generate it).
Go to share.streamlit.io → New app
Select your repo, branch `main`, main file `app.py`
Under Advanced settings → Secrets, add:
```toml
ANTHROPIC_API_KEY = "sk-ant-your-key-here"
DEEPSEEK_API_KEY  = "sk-your-deepseek-key-here"
```
Click Deploy. Done.
> ⚠️ Make sure `mock_data.xlsx` is committed to the repo, or add a startup step to generate it.  
> The `generate_mock_data.py` script can be run as a one-off before committing.
---
Project Structure
```
orb-v2/
├── app.py                   # Main Streamlit app — UI, login, chat layout
├── ai_engine.py             # Intent detection, data retrieval, Claude API
├── data.py                  # Data access layer — loads Excel, exposes scoped queries
├── auth.py                  # Simulated login — replace with Flashcredentials in prod
├── generate_mock_data.py    # One-off mock data generator
├── mock_data.xlsx           # Generated mock data (commit after running generator)
├── requirements.txt
├── .streamlit/
│   └── config.toml          # Theme and server config
└── README.md
```
---
Architecture
```
User question
     │
     ▼
Intent Detection          ← regex + keyword patterns → one of 13 intent types
     │
     ▼
Data Retrieval            ← country-scoped query on Excel (MySQL in prod)
     │
     ▼
Claude Reasoning          ← data slice + conversation history → executive answer
     │
     ├── Text response    → chat window (left)
     └── Chart + Table    → side panel (right, persistent, open/close)
```
---
Supported Query Patterns
Group	Themes
Incentive Performance	Attainment, underperformance, payout distribution, near-miss, top performers, scheme comparison, cycle trend
Qualifiers & Attendance	Qualifier failures, proration impact, absenteeism
Cross-Source	Performance vs payout anomaly, active/non-active cross-check, new joiners, leavers reconciliation
Workforce Health	Headcount, attrition, tenure, PMGM distribution
Executive Summary	Cycle summary, country comparison, anomaly flags
All free-form questions are also supported — the AI interprets intent dynamically.
---
Production Upgrade Path
POC	Production
Excel (`mock_data.xlsx`)	MySQL via SQLAlchemy connector
Simulated login	Flashcredentials OAuth/SSO
Config-file role map	Central IAM / database roles table
Single-instance Streamlit	Containerised deployment (Docker + cloud run)
---
Environment Variables
Variable	Description
`ANTHROPIC_API_KEY`	Required. Your Anthropic API key.
`DEEPSEEK_API_KEY`	Required for DeepSeek models. Get from platform.deepseek.com
---
Orb v2 POC — built for internal use. Not for production without auth hardening.
