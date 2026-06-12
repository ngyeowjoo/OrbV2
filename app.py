"""
app.py  —  Orb v2  |  Workforce Intelligence Platform
Light palette. Model selector on welcome/login screen. Clean side panel.
"""
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import re
from auth import authenticate, scope_label
from ai_engine import answer, MODELS, MODEL_NAMES, DEFAULT_MODEL

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Orb v2",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={"About": "Orb v2 — Workforce Intelligence Platform"},
)

# ── Light palette tokens ──────────────────────────────────────────────────────
AMBER   = "#D97706"   # primary accent (darker amber on white)
AMBERL  = "#FEF3C7"   # amber light tint
BG      = "#FAFAFA"   # page background
CARD    = "#FFFFFF"   # card / panel background
BORDER  = "#E5E7EB"   # subtle border
TEXT    = "#111827"   # primary text
SUBTEXT = "#6B7280"   # secondary text
USERBG  = "#F3F4F6"   # user message bubble

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS  —  light palette
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Mono:wght@300;400;500&family=Inter:wght@300;400;500;600&display=swap');

*, *::before, *::after {{ box-sizing: border-box; margin: 0; }}
html, body, .stApp {{ background: {BG} !important; color: {TEXT}; }}
section[data-testid="stSidebar"] {{ display: none !important; }}
#MainMenu, footer {{ display: none !important; }}
.block-container {{ padding: 0 !important; max-width: 100% !important; }}

::-webkit-scrollbar {{ width: 4px; }}
::-webkit-scrollbar-track {{ background: #f1f1f1; }}
::-webkit-scrollbar-thumb {{ background: #d1d5db; border-radius: 2px; }}

/* ── Inputs ── */
.stTextInput > div > div > input {{
    background: {CARD} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 10px !important;
    color: {TEXT} !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.92rem !important;
    padding: 12px 16px !important;
    transition: border-color 0.2s !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04) !important;
}}
.stTextInput > div > div > input:focus {{
    border-color: {AMBER} !important;
    box-shadow: 0 0 0 3px rgba(217,119,6,0.12) !important;
}}
.stTextInput > div > div > input::placeholder {{ color: #9CA3AF !important; }}
.stTextInput label {{ display: none !important; }}

/* ── Buttons ── */
.stButton > button {{
    background: {CARD} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 8px !important;
    color: {SUBTEXT} !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    padding: 7px 16px !important;
    transition: all 0.15s !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05) !important;
}}
.stButton > button:hover {{
    border-color: {AMBER} !important;
    color: {AMBER} !important;
    background: {AMBERL} !important;
    box-shadow: 0 2px 6px rgba(217,119,6,0.12) !important;
}}

/* ── Send / primary button ── */
.send-btn > button {{
    background: {AMBER} !important;
    border: none !important;
    border-radius: 8px !important;
    color: #fff !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    padding: 9px 22px !important;
    box-shadow: 0 2px 8px rgba(217,119,6,0.3) !important;
}}
.send-btn > button:hover {{
    background: #B45309 !important;
    box-shadow: 0 4px 12px rgba(217,119,6,0.35) !important;
}}

/* ── Selectbox ── */
div[data-testid="stSelectbox"] > div > div {{
    background: {CARD} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 8px !important;
    color: {TEXT} !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.84rem !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04) !important;
}}
div[data-testid="stSelectbox"] label {{
    font-family: 'Inter', sans-serif !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    color: {SUBTEXT} !important;
}}

/* ── Metrics ── */
[data-testid="stMetric"] {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 12px 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}}
[data-testid="stMetricLabel"] {{ color: {SUBTEXT} !important; font-size: 0.75rem !important; }}
[data-testid="stMetricValue"] {{ color: {AMBER} !important; font-size: 1.4rem !important; font-weight: 700 !important; }}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {{ border: 1px solid {BORDER} !important; border-radius: 8px; }}

/* ── Spinner ── */
.stSpinner > div {{ border-top-color: {AMBER} !important; }}

/* ── Suggestion chips ── */
.chip-btn > button {{
    background: {AMBERL} !important;
    border: 1px solid rgba(217,119,6,0.25) !important;
    border-radius: 20px !important;
    color: {AMBER} !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.80rem !important;
    font-weight: 500 !important;
    padding: 6px 16px !important;
}}
.chip-btn > button:hover {{
    background: rgba(217,119,6,0.18) !important;
    border-color: {AMBER} !important;
}}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
for k, v in {
    "authenticated":  False,
    "user":           None,
    "messages":       [],
    "panels":         [],
    "panel_open":     False,
    "panel_idx":      -1,
    "input_key":      0,
    "selected_model": DEFAULT_MODEL,
    "last_df":        None,   # persists last retrieved dataframe for follow-up context
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── strip markdown bold/italic from AI responses ─────────────────────────────
def clean_markdown(text: str) -> str:
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*',     r'\1', text)
    text = re.sub(r'__(.+?)__',     r'\1', text)
    text = re.sub(r'_(.+?)_',       r'\1', text)
    return text

# ══════════════════════════════════════════════════════════════════════════════
# ── LOGIN / WELCOME PAGE ──────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
if not st.session_state["authenticated"]:

    # Animated orb header
    components.html("""
    <!DOCTYPE html><html><head>
    <link href="https://fonts.googleapis.com/css2?family=Syne:wght@800&family=DM+Mono:wght@400&display=swap" rel="stylesheet">
    <style>
    * { margin:0; padding:0; box-sizing:border-box; }
    body {
        background: transparent;
        display: flex; flex-direction: column;
        align-items: center; justify-content: center;
        height: 260px; overflow: hidden;
    }
    .ring {
        position:absolute; border-radius:50%; top:50%; left:50%;
        border: 1px solid rgba(217,119,6,0.12);
        animation: pulse 4s ease-in-out infinite;
    }
    .ring:nth-child(1){width:160px;height:160px;animation-delay:0s;}
    .ring:nth-child(2){width:260px;height:260px;animation-delay:.7s;border-color:rgba(217,119,6,0.07);}
    .ring:nth-child(3){width:380px;height:380px;animation-delay:1.4s;border-color:rgba(217,119,6,0.04);}
    @keyframes pulse {
        0%,100%{transform:translate(-50%,-50%) scale(1);opacity:1;}
        50%{transform:translate(-50%,-50%) scale(1.05);opacity:0.4;}
    }
    .orb {
        width:80px; height:80px; border-radius:50%; position:relative; z-index:2;
        background: radial-gradient(circle at 38% 35%,#fff8e0 0%,#F9A602 25%,#c97f00 52%,#7a4500 78%,#3d1f00 100%);
        box-shadow: 0 0 32px 10px rgba(217,119,6,0.35), 0 0 64px 24px rgba(217,119,6,0.14);
        animation: float 5s ease-in-out infinite;
    }
    .orb::before {
        content:''; position:absolute; top:17%; left:21%; width:36%; height:24%;
        background:radial-gradient(ellipse, rgba(255,255,255,0.6) 0%, transparent 70%);
        border-radius:50%; transform:rotate(-30deg);
    }
    @keyframes float { 0%,100%{transform:translateY(0);}50%{transform:translateY(-8px);} }
    .title {
        font-family:'Syne',sans-serif; font-size:1.9rem; font-weight:800;
        color:#111827; margin-top:16px; letter-spacing:-0.02em; text-align:center;
    }
    .title span { color:#D97706; }
    .sub {
        font-family:'DM Mono',monospace; font-size:0.6rem;
        letter-spacing:0.22em; text-transform:uppercase;
        color:#9CA3AF; margin-top:6px; text-align:center;
    }
    </style></head><body>
    <div class="ring"></div><div class="ring"></div><div class="ring"></div>
    <div class="orb"></div>
    <div class="title">Orb <span>v2</span></div>
    <div class="sub">Workforce Intelligence Platform</div>
    </body></html>
    """, height=260)

    col_l, col_c, col_r = st.columns([1, 1.4, 1])
    with col_c:
        # ── Card container ────────────────────────────────────────────────────
        st.markdown(f"""
        <div style="background:{CARD};border:1px solid {BORDER};border-radius:16px;
                    padding:28px 28px 8px;box-shadow:0 4px 24px rgba(0,0,0,0.07);">
        <p style="font-family:'Inter',sans-serif;font-size:0.72rem;font-weight:600;
                  letter-spacing:0.12em;text-transform:uppercase;color:{SUBTEXT};
                  text-align:center;margin-bottom:20px;">Sign in to continue</p>
        </div>
        """, unsafe_allow_html=True)

        username = st.text_input("Username", placeholder="Username", key="login_user")
        password = st.text_input("Password", placeholder="Password", type="password", key="login_pass")

        # ── Model selector on login card ──────────────────────────────────────
        st.markdown(f"""
        <p style="font-family:'Inter',sans-serif;font-size:0.72rem;font-weight:600;
                  letter-spacing:0.06em;text-transform:uppercase;color:{SUBTEXT};
                  margin:16px 0 6px;">AI Model</p>
        """, unsafe_allow_html=True)

        model_cols = st.columns(len(MODEL_NAMES))
        for i, name in enumerate(MODEL_NAMES):
            cfg = MODELS[name]
            is_selected = st.session_state["selected_model"] == name
            with model_cols[i]:
                border_style = f"2px solid {AMBER}" if is_selected else f"1px solid {BORDER}"
                bg_style = AMBERL if is_selected else CARD
                text_color = AMBER if is_selected else SUBTEXT
                st.markdown(f"""
                <div style="background:{bg_style};border:{border_style};border-radius:10px;
                            padding:10px 10px 8px;cursor:pointer;transition:all 0.15s;
                            box-shadow:0 1px 3px rgba(0,0,0,0.05);">
                    <div style="font-family:'Inter',sans-serif;font-size:0.75rem;
                                font-weight:600;color:{text_color};">{name}</div>
                    <div style="font-family:'Inter',sans-serif;font-size:0.68rem;
                                color:{SUBTEXT};margin-top:2px;">{cfg['description']}</div>
                    <div style="margin-top:5px;">
                        <span style="background:{cfg['tag_color']}1a;color:{cfg['tag_color']};
                                     border-radius:4px;padding:2px 6px;font-size:0.60rem;
                                     font-family:'DM Mono',monospace;">{cfg['tag']}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                if st.button("Select" if not is_selected else "✓ Selected",
                             key=f"model_pick_{i}", use_container_width=True):
                    st.session_state["selected_model"] = name
                    st.rerun()

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        st.markdown(f"""
        <p style="font-family:'DM Mono',monospace;font-size:0.62rem;color:#9CA3AF;
                  text-align:center;margin:10px 0 14px;">
        ceo / coo.apac / head.sg / hr.admin &nbsp;·&nbsp; password: demo
        </p>
        """, unsafe_allow_html=True)

        c1, c2, c3 = st.columns([1, 2, 1])
        with col_c:
            st.markdown('<div class="send-btn">', unsafe_allow_html=True)
            if st.button("Enter the Orb →", use_container_width=True, key="login_btn"):
                user = authenticate(username, password)
                if user:
                    st.session_state["authenticated"] = True
                    st.session_state["user"]           = user
                    st.rerun()
                else:
                    st.error("Invalid credentials.")
            st.markdown('</div>', unsafe_allow_html=True)

    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# ── MAIN CHAT APP ─────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
user   = st.session_state["user"]
msgs   = st.session_state["messages"]
panels = st.session_state["panels"]

def open_panel(idx):
    st.session_state["panel_open"] = True
    st.session_state["panel_idx"]  = idx

def close_panel():
    st.session_state["panel_open"] = False
    st.session_state["panel_idx"]  = -1

def panel_is_open():
    return st.session_state["panel_open"] and st.session_state["panel_idx"] >= 0

# Two-column layout when panel open
if panel_is_open():
    chat_col, panel_col = st.columns([1.1, 0.9], gap="small")
else:
    chat_col  = st.container()
    panel_col = None

# ══════════════════════════════════════════════════════════════════════════════
# ── CHAT COLUMN ───────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
with chat_col:

    # ── TOP BAR ───────────────────────────────────────────────────────────────
    cfg_now   = MODELS[st.session_state["selected_model"]]
    left, right = st.columns([2, 1])

    with left:
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:12px;padding:12px 0 8px 8px;">
            <div style="width:30px;height:30px;border-radius:50%;flex-shrink:0;
                background:radial-gradient(circle at 38% 35%,#fff8e0 0%,#F9A602 30%,#c97f00 60%,#3d1f00 100%);
                box-shadow:0 0 10px rgba(217,119,6,0.4);"></div>
            <div>
                <span style="font-family:'Syne',sans-serif;font-size:1rem;font-weight:800;
                             color:{TEXT};letter-spacing:-0.01em;">Orb <span style="color:{AMBER};">v2</span></span>
                <span style="font-family:'DM Mono',monospace;font-size:0.58rem;letter-spacing:0.14em;
                             color:{SUBTEXT};margin-left:10px;text-transform:uppercase;">Workforce Intelligence</span>
            </div>
            <div style="margin-left:8px;">
                <span style="background:{cfg_now['tag_color']}18;color:{cfg_now['tag_color']};
                             border-radius:5px;padding:3px 8px;font-family:'DM Mono',monospace;
                             font-size:0.60rem;">{st.session_state['selected_model']}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown(f"""
        <div style="display:flex;align-items:center;justify-content:flex-end;gap:10px;padding:12px 8px 8px 0;">
            <div style="width:28px;height:28px;border-radius:50%;flex-shrink:0;
                background:{AMBERL};border:1px solid rgba(217,119,6,0.3);
                display:flex;align-items:center;justify-content:center;
                font-family:'DM Mono',monospace;font-size:0.65rem;font-weight:600;color:{AMBER};">
                {user['avatar']}</div>
            <div style="text-align:right;">
                <div style="font-family:'Inter',sans-serif;font-size:0.80rem;
                            color:{TEXT};font-weight:500;">{user['display_name']}</div>
                <div style="font-family:'DM Mono',monospace;font-size:0.60rem;color:{SUBTEXT};">
                    {user['role']} · <span style="color:{AMBER};">{scope_label(user)}</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"<div style='border-bottom:1px solid {BORDER};margin-bottom:4px;'></div>",
                unsafe_allow_html=True)

    # ── MESSAGES ──────────────────────────────────────────────────────────────
    with st.container():

        if not msgs:
            st.markdown(f"""
            <div style="text-align:center;padding:48px 20px 32px;">
                <div style="font-family:'DM Mono',monospace;font-size:0.62rem;letter-spacing:0.2em;
                            text-transform:uppercase;color:rgba(217,119,6,0.6);margin-bottom:10px;">
                    Good to see you, {user['display_name'].split()[0]}</div>
                <div style="font-family:'Syne',sans-serif;font-size:1.5rem;font-weight:700;
                            color:{TEXT};margin-bottom:8px;">What would you like to know?</div>
                <div style="font-family:'Inter',sans-serif;font-size:0.82rem;color:{SUBTEXT};
                            max-width:400px;margin:0 auto;">
                    Ask anything about your workforce — incentive performance,
                    headcount, PMGM ratings, or cross-source insights.</div>
            </div>
            """, unsafe_allow_html=True)

            suggestions = [
                "What % of employees hit max payout this cycle?",
                "Who has missed targets for 3+ consecutive periods?",
                "Show me the PMGM rating distribution",
                "Are there non-active employees with payouts?",
                "Give me a cycle summary",
                "Compare countries on incentive attainment",
            ]
            chip_cols = st.columns(2)
            for i, s in enumerate(suggestions):
                with chip_cols[i % 2]:
                    st.markdown('<div class="chip-btn">', unsafe_allow_html=True)
                    if st.button(s, key=f"sug_{i}", use_container_width=True):
                        st.session_state["messages"].append({"role": "user", "content": s})
                        st.session_state["_pending_question"] = s
                        st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)

        # Render history
        assistant_idx = 0
        for i, msg in enumerate(msgs):
            if msg["role"] == "user":
                st.markdown(f"""
                <div style="display:flex;justify-content:flex-end;margin:14px 0 4px;">
                    <div style="background:{USERBG};border:1px solid {BORDER};
                                border-radius:16px 16px 4px 16px;padding:10px 15px;
                                max-width:72%;font-family:'Inter',sans-serif;
                                font-size:0.88rem;color:{TEXT};line-height:1.55;">
                        {msg['content']}</div>
                </div>
                """, unsafe_allow_html=True)

            else:
                has_panel = (assistant_idx < len(panels) and
                    (panels[assistant_idx]["chart"] is not None or
                     panels[assistant_idx]["df"]    is not None))
                panel_label      = panels[assistant_idx]["label"] if assistant_idx < len(panels) else ""
                current_panel_idx = assistant_idx

                st.markdown(f"""
                <div style="display:flex;align-items:flex-start;gap:8px;margin:4px 0 14px;">
                    <div style="width:24px;height:24px;border-radius:50%;flex-shrink:0;margin-top:3px;
                        background:radial-gradient(circle at 38% 35%,#fff8e0 0%,#F9A602 30%,#c97f00 60%,#3d1f00 100%);
                        box-shadow:0 0 6px rgba(217,119,6,0.3);"></div>
                    <div style="background:{CARD};border:1px solid {BORDER};
                                border-radius:4px 16px 16px 16px;padding:12px 16px;
                                max-width:84%;font-family:'Inter',sans-serif;font-size:0.88rem;
                                color:{TEXT};line-height:1.65;
                                box-shadow:0 1px 4px rgba(0,0,0,0.05);">
                        {msg['content']}</div>
                </div>
                """, unsafe_allow_html=True)

                if has_panel:
                    is_open_now = panel_is_open() and st.session_state["panel_idx"] == current_panel_idx
                    btn_label = f"{'▶ View' if not is_open_now else '◀ Close'}  {panel_label}"
                    if st.button(btn_label, key=f"panel_btn_{i}"):
                        if is_open_now:
                            close_panel()
                        else:
                            open_panel(current_panel_idx)
                        st.rerun()

                assistant_idx += 1

    # ── INPUT BAR ─────────────────────────────────────────────────────────────
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    inp_col, btn_col = st.columns([5, 1])
    with inp_col:
        question = st.text_input(
            "q", placeholder="Ask anything about your workforce…",
            key=f"chat_input_{st.session_state['input_key']}",
            label_visibility="collapsed",
        )
    with btn_col:
        st.markdown('<div class="send-btn">', unsafe_allow_html=True)
        send = st.button("Send →", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Footer row
    f1, f2, f3 = st.columns([1, 1, 5])
    with f1:
        if st.button("✕ Clear"):
            st.session_state["messages"] = []
            st.session_state["panels"]   = []
            st.session_state["last_df"]  = None
            close_panel()
            st.rerun()
    with f2:
        if st.button("↩ Sign out"):
            for k in ["authenticated","user","messages","panels",
                      "panel_open","panel_idx","input_key","selected_model","last_df"]:
                st.session_state.pop(k, None)
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# ── PROCESS QUESTION ──────────────────────────────────════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
pending = st.session_state.pop("_pending_question", None) or (
    question if send and question.strip() else None
)

if pending:
    q = pending.strip()
    if not any(m["content"] == q and m["role"] == "user" for m in msgs[-2:]):
        st.session_state["messages"].append({"role": "user", "content": q})

    # Build history — include last_df as extra context for follow-ups
    history = [m for m in st.session_state["messages"][:-1]
               if m["role"] in ("user", "assistant")]

    with st.spinner("Thinking…"):
        try:
            text, chart, df = answer(
                q, history, user,
                st.session_state.get("selected_model", DEFAULT_MODEL),
                st.session_state.get("last_df"),       # ← carry last df for follow-ups
            )
        except Exception as e:
            text  = f"Sorry, I ran into an issue: {str(e)}"
            chart = None
            df    = None

    # Strip bold/italic markdown from response
    text = clean_markdown(text)
    st.session_state["messages"].append({"role": "assistant", "content": text})

    # Update last_df — prefer new df, fall back to previous
    if df is not None:
        st.session_state["last_df"] = df

    label = ""
    if chart is not None and df is not None:
        label = "Chart & Table"
    elif chart is not None:
        label = "Chart"
    elif df is not None:
        label = "Table"

    st.session_state["panels"].append({"chart": chart, "df": df, "label": label})

    if chart is not None or df is not None:
        new_idx = len([m for m in st.session_state["messages"]
                       if m["role"] == "assistant"]) - 1
        open_panel(new_idx)

    st.session_state["input_key"] += 1
    st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# ── SIDE PANEL ────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
if panel_is_open() and panel_col is not None:
    idx   = st.session_state["panel_idx"]
    panel = panels[idx] if idx < len(panels) else {}

    with panel_col:
        # Slim header — label only, no "Response N" counter
        st.markdown(f"""
        <div style="display:flex;align-items:center;justify-content:space-between;
                    padding:10px 16px 8px;border-bottom:1px solid {BORDER};
                    background:{CARD};">
            <span style="font-family:'DM Mono',monospace;font-size:0.65rem;
                          letter-spacing:0.12em;text-transform:uppercase;color:{SUBTEXT};">
                {panel.get('label','Analysis')}</span>
        </div>
        """, unsafe_allow_html=True)

        # Chart — immediately after header, no padding gap
        if panel.get("chart") is not None:
            st.plotly_chart(
                panel["chart"], use_container_width=True,
                config={"displayModeBar": False}
            )

        # Table — directly below chart
        if panel.get("df") is not None:
            df_show = panel["df"]
            if isinstance(df_show, pd.DataFrame) and not df_show.empty:
                st.markdown(f"""
                <p style="font-family:'DM Mono',monospace;font-size:0.62rem;
                           letter-spacing:0.10em;text-transform:uppercase;
                           color:{SUBTEXT};padding:4px 16px 4px;">Data</p>
                """, unsafe_allow_html=True)
                num_cols   = df_show.select_dtypes(include="number").columns
                df_display = df_show.copy()
                df_display[num_cols] = df_display[num_cols].round(2)
                st.dataframe(df_display, use_container_width=True, height=280)

        # Close button
        st.markdown("<div style='padding:8px 8px 16px;'>", unsafe_allow_html=True)
        if st.button("✕  Close", key="close_panel_btn", use_container_width=True):
            close_panel()
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
