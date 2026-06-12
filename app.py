"""
app.py  —  Orb v2  |  Workforce Intelligence Platform
Standalone chat-first app with persistent side panel (Claude-style).
"""
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
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

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Mono:wght@300;400;500&family=Inter:wght@300;400;500&display=swap');

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; }
html, body, .stApp { background: #0a0a0a !important; color: #e0e0e0; }
section[data-testid="stSidebar"] { display: none !important; }
#MainMenu, footer { display: none !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #111; }
::-webkit-scrollbar-thumb { background: #333; border-radius: 2px; }

/* ── Input ── */
.stTextInput > div > div > input {
    background: #141414 !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 12px !important;
    color: #e0e0e0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.92rem !important;
    padding: 12px 16px !important;
    transition: border-color 0.2s !important;
}
.stTextInput > div > div > input:focus {
    border-color: rgba(249,166,2,0.6) !important;
    box-shadow: 0 0 0 2px rgba(249,166,2,0.12) !important;
}
.stTextInput > div > div > input::placeholder { color: #555 !important; }

/* ── Buttons ── */
.stButton > button {
    background: transparent !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 8px !important;
    color: #aaa !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.06em !important;
    padding: 6px 14px !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    border-color: rgba(249,166,2,0.5) !important;
    color: #F9A602 !important;
    background: rgba(249,166,2,0.06) !important;
}

/* ── Send button ── */
.send-btn > button {
    background: rgba(249,166,2,0.15) !important;
    border: 1px solid rgba(249,166,2,0.4) !important;
    border-radius: 8px !important;
    color: #F9A602 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.78rem !important;
    padding: 8px 20px !important;
}
.send-btn > button:hover {
    background: rgba(249,166,2,0.25) !important;
    border-color: #F9A602 !important;
    box-shadow: 0 0 12px rgba(249,166,2,0.2) !important;
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: #F9A602 !important; }

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: #141414;
    border: 1px solid #1e1e1e;
    border-radius: 10px;
    padding: 12px 16px;
}
[data-testid="stMetricLabel"] { color: #888 !important; font-size: 0.75rem !important; }
[data-testid="stMetricValue"] { color: #F9A602 !important; font-size: 1.4rem !important; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border: 1px solid #1e1e1e !important; border-radius: 8px; }

/* ── Hide label for text inputs ── */
.stTextInput label { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INIT
# ══════════════════════════════════════════════════════════════════════════════
for k, v in {
    "authenticated": False,
    "user":          None,
    "messages":      [],       # {role, content}
    "panels":        [],       # {chart, df, label} per assistant message index
    "panel_open":    False,
    "panel_idx":     -1,
    "input_key":     0,
    "selected_model": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════════════════════════════════════
# ── LOGIN PAGE ────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
if not st.session_state["authenticated"]:
    st.markdown("""
    <style>
    .stApp { background: radial-gradient(ellipse at 50% 60%, #1a0a00 0%, #0a0a0a 70%) !important; }
    </style>
    """, unsafe_allow_html=True)

    components.html("""
    <!DOCTYPE html><html><head>
    <link href="https://fonts.googleapis.com/css2?family=Syne:wght@800&family=DM+Mono:wght@400&display=swap" rel="stylesheet">
    <style>
    * { margin:0; padding:0; box-sizing:border-box; }
    body {
        background: transparent;
        display: flex; flex-direction: column;
        align-items: center; justify-content: center;
        height: 340px; overflow: hidden;
    }
    .ring {
        position:absolute; border-radius:50%; top:50%; left:50%;
        border: 1px solid rgba(249,166,2,0.10);
        animation: pulse 4s ease-in-out infinite;
    }
    .ring:nth-child(1){width:200px;height:200px;animation-delay:0s;}
    .ring:nth-child(2){width:320px;height:320px;animation-delay:.7s;border-color:rgba(249,166,2,0.06);}
    .ring:nth-child(3){width:460px;height:460px;animation-delay:1.4s;border-color:rgba(249,166,2,0.03);}
    @keyframes pulse {
        0%,100%{transform:translate(-50%,-50%) scale(1);opacity:1;}
        50%{transform:translate(-50%,-50%) scale(1.05);opacity:0.4;}
    }
    .orb {
        width:100px; height:100px; border-radius:50%; position:relative; z-index:2;
        background: radial-gradient(circle at 38% 35%,#fff8e0 0%,#F9A602 25%,#c97f00 52%,#7a4500 78%,#1a0800 100%);
        box-shadow: 0 0 40px 12px rgba(249,166,2,0.4), 0 0 80px 30px rgba(249,166,2,0.18), inset 0 -10px 20px rgba(0,0,0,0.5);
        animation: float 5s ease-in-out infinite, glow 3s ease-in-out infinite alternate;
    }
    .orb::before {
        content:''; position:absolute; top:17%; left:21%; width:36%; height:24%;
        background:radial-gradient(ellipse, rgba(255,255,255,0.55) 0%, transparent 70%);
        border-radius:50%; transform:rotate(-30deg);
    }
    @keyframes float { 0%,100%{transform:translateY(0);}50%{transform:translateY(-10px);} }
    @keyframes glow {
        from{box-shadow:0 0 40px 12px rgba(249,166,2,0.4),0 0 80px 30px rgba(249,166,2,0.18),inset 0 -10px 20px rgba(0,0,0,0.5);}
        to  {box-shadow:0 0 60px 20px rgba(249,166,2,0.58),0 0 120px 50px rgba(249,166,2,0.25),inset 0 -10px 20px rgba(0,0,0,0.5);}
    }
    .title {
        font-family:'Syne',sans-serif; font-size:2.2rem; font-weight:800;
        color:#fff; margin-top:20px; letter-spacing:-0.02em; text-align:center;
    }
    .title span {
        background: linear-gradient(90deg,#F9A602,#ffe066,#F9A602);
        background-size:200% auto;
        -webkit-background-clip:text; -webkit-text-fill-color:transparent;
        animation: shimmer 3s linear infinite;
    }
    @keyframes shimmer{to{background-position:200% center;}}
    .sub {
        font-family:'DM Mono',monospace; font-size:0.65rem;
        letter-spacing:0.22em; text-transform:uppercase;
        color:rgba(249,166,2,0.5); margin-top:8px; text-align:center;
    }
    </style></head><body>
    <div class="ring"></div><div class="ring"></div><div class="ring"></div>
    <div class="orb"></div>
    <div class="title">Orb <span>v2</span></div>
    <div class="sub">Workforce Intelligence Platform</div>
    </body></html>
    """, height=340)

    # Login form
    col_l, col_c, col_r = st.columns([1, 1.2, 1])
    with col_c:
        st.markdown("""
        <div style="background:#111;border:1px solid #222;border-radius:16px;padding:28px 28px 20px;">
        <p style="font-family:'DM Mono',monospace;font-size:0.68rem;letter-spacing:0.18em;
                  text-transform:uppercase;color:rgba(249,166,2,0.6);text-align:center;margin-bottom:18px;">
        Sign in to continue</p>
        </div>
        """, unsafe_allow_html=True)

        username = st.text_input("Username", placeholder="Username", key="login_user")
        password = st.text_input("Password", placeholder="Password", type="password", key="login_pass")

        st.markdown("""
        <p style="font-family:'DM Mono',monospace;font-size:0.65rem;color:#444;
                  text-align:center;margin:8px 0 12px;">
        Demo accounts — username: ceo / coo.apac / head.sg / hr.admin &nbsp;|&nbsp; password: demo
        </p>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown('<div class="send-btn">', unsafe_allow_html=True)
            if st.button("Enter the Orb", use_container_width=True):
                user = authenticate(username, password)
                if user:
                    st.session_state["authenticated"] = True
                    st.session_state["user"] = user
                    st.rerun()
                else:
                    st.error("Invalid credentials.")
            st.markdown('</div>', unsafe_allow_html=True)

    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# ── MAIN CHAT APP ─────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
user      = st.session_state["user"]
msgs      = st.session_state["messages"]
panels    = st.session_state["panels"]

# ── HELPERS ───────────────────────────────────────────────────────────────────
def open_panel(idx):
    st.session_state["panel_open"] = True
    st.session_state["panel_idx"]  = idx

def close_panel():
    st.session_state["panel_open"] = False
    st.session_state["panel_idx"]  = -1

def panel_is_open():
    return st.session_state["panel_open"] and st.session_state["panel_idx"] >= 0

# ── LAYOUT ────────────────────────────────────────────────────────────────────
# Two-column layout when panel is open; single column otherwise
if panel_is_open():
    chat_col, panel_col = st.columns([1.1, 0.9], gap="small")
else:
    chat_col = st.container()
    panel_col = None

# ══════════════════════════════════════════════════════════════════════════════
# ── CHAT COLUMN ───────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
with chat_col:

    # ── TOP BAR ───────────────────────────────────────────────────────────────
    topbar_left, topbar_mid, topbar_right = st.columns([2, 2, 2])

    with topbar_left:
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:14px;padding:10px 0 6px 8px;">
            <div style="
                width:34px;height:34px;border-radius:50%;flex-shrink:0;
                background:radial-gradient(circle at 38% 35%,#fff8e0 0%,#F9A602 30%,#c97f00 60%,#1a0800 100%);
                box-shadow:0 0 14px rgba(249,166,2,0.5);
            "></div>
            <div>
                <span style="font-family:'Syne',sans-serif;font-size:1.05rem;font-weight:800;
                             color:#fff;letter-spacing:-0.01em;">Orb <span style="color:#F9A602;">v2</span></span>
                <span style="font-family:'DM Mono',monospace;font-size:0.62rem;
                             letter-spacing:0.14em;color:#555;margin-left:10px;text-transform:uppercase;">
                    Workforce Intelligence
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with topbar_mid:
        # Model selector — initialise default once
        if st.session_state["selected_model"] is None:
            st.session_state["selected_model"] = DEFAULT_MODEL

        st.markdown("""
        <style>
        /* Style the selectbox to match the dark theme */
        div[data-testid="stSelectbox"] > div > div {
            background: #141414 !important;
            border: 1px solid #2a2a2a !important;
            border-radius: 8px !important;
            color: #e0e0e0 !important;
            font-family: 'DM Mono', monospace !important;
            font-size: 0.78rem !important;
        }
        div[data-testid="stSelectbox"] label {
            font-family: 'DM Mono', monospace !important;
            font-size: 0.62rem !important;
            letter-spacing: 0.12em !important;
            text-transform: uppercase !important;
            color: #555 !important;
        }
        </style>
        """, unsafe_allow_html=True)

        selected = st.selectbox(
            "Model",
            options=MODEL_NAMES,
            index=MODEL_NAMES.index(st.session_state["selected_model"]),
            key="model_selector",
        )
        st.session_state["selected_model"] = selected
        cfg = MODELS[selected]
        tag_color = cfg["tag_color"]
        st.markdown(f"""
        <div style="font-family:'DM Mono',monospace;font-size:0.62rem;color:#555;margin-top:-8px;">
            <span style="background:{tag_color}22;color:{tag_color};border-radius:4px;
                         padding:2px 7px;font-size:0.60rem;">{cfg["tag"]}</span>
            &nbsp;{cfg["description"]}
        </div>
        """, unsafe_allow_html=True)

    with topbar_right:
        st.markdown(f"""
        <div style="display:flex;align-items:center;justify-content:flex-end;gap:12px;padding:10px 8px 6px 0;">
            <div style="
                width:30px;height:30px;border-radius:50%;flex-shrink:0;
                background:rgba(249,166,2,0.15);border:1px solid rgba(249,166,2,0.3);
                display:flex;align-items:center;justify-content:center;
                font-family:'DM Mono',monospace;font-size:0.68rem;font-weight:500;
                color:#F9A602;
            ">{user['avatar']}</div>
            <div style="text-align:right;">
                <div style="font-family:'Inter',sans-serif;font-size:0.80rem;color:#ccc;font-weight:500;">
                    {user['display_name']}</div>
                <div style="font-family:'DM Mono',monospace;font-size:0.62rem;color:#555;">
                    {user['role']} &nbsp;·&nbsp;
                    <span style="color:rgba(249,166,2,0.7);">{scope_label(user)}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='border-bottom:1px solid #1a1a1a;margin-bottom:4px;'></div>", unsafe_allow_html=True)

    # ── MESSAGES ──────────────────────────────────────────────────────────────
    chat_area = st.container()
    with chat_area:

        if not msgs:
            # Welcome state
            st.markdown(f"""
            <div style="text-align:center;padding:60px 20px 40px;">
                <div style="font-family:'DM Mono',monospace;font-size:0.65rem;
                            letter-spacing:0.22em;text-transform:uppercase;
                            color:rgba(249,166,2,0.45);margin-bottom:12px;">
                    Good to see you, {user['display_name'].split()[0]}
                </div>
                <div style="font-family:'Syne',sans-serif;font-size:1.6rem;
                            font-weight:700;color:#fff;margin-bottom:8px;">
                    What would you like to know?
                </div>
                <div style="font-family:'Inter',sans-serif;font-size:0.82rem;
                            color:#555;max-width:400px;margin:0 auto;">
                    Ask anything about your workforce — incentive performance,
                    headcount, PMGM ratings, or cross-source insights.
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Suggestion chips
            suggestions = [
                "What % of employees hit max payout this cycle?",
                "Who has missed targets for 3+ consecutive periods?",
                "Show me the PMGM rating distribution",
                "Are there non-active employees with payouts?",
                "Give me a cycle summary",
                "Compare countries on incentive attainment",
            ]
            cols = st.columns(2)
            for i, s in enumerate(suggestions):
                with cols[i % 2]:
                    if st.button(s, key=f"sug_{i}", use_container_width=True):
                        st.session_state["messages"].append({"role": "user", "content": s})
                        st.session_state["_pending_question"] = s
                        st.rerun()

        # Render conversation history
        assistant_idx = 0
        for i, msg in enumerate(msgs):
            if msg["role"] == "user":
                st.markdown(f"""
                <div style="display:flex;justify-content:flex-end;margin:16px 0 4px;">
                    <div style="
                        background:#1a1a1a;border:1px solid #252525;
                        border-radius:18px 18px 4px 18px;
                        padding:11px 16px;max-width:72%;
                        font-family:'Inter',sans-serif;font-size:0.88rem;
                        color:#ddd;line-height:1.5;
                    ">{msg['content']}</div>
                </div>
                """, unsafe_allow_html=True)

            else:  # assistant
                has_panel = assistant_idx < len(panels) and (
                    panels[assistant_idx]["chart"] is not None or
                    panels[assistant_idx]["df"] is not None
                )
                panel_label = panels[assistant_idx]["label"] if assistant_idx < len(panels) else ""
                current_panel_idx = assistant_idx

                st.markdown(f"""
                <div style="display:flex;align-items:flex-start;gap:10px;margin:4px 0 16px;">
                    <div style="
                        width:26px;height:26px;border-radius:50%;flex-shrink:0;margin-top:2px;
                        background:radial-gradient(circle at 38% 35%,#fff8e0 0%,#F9A602 30%,#c97f00 60%,#1a0800 100%);
                        box-shadow:0 0 8px rgba(249,166,2,0.4);
                    "></div>
                    <div style="
                        background:#111;border:1px solid #1e1e1e;
                        border-radius:4px 18px 18px 18px;
                        padding:13px 17px;max-width:82%;
                        font-family:'Inter',sans-serif;font-size:0.88rem;
                        color:#d0d0d0;line-height:1.65;
                    ">{msg['content']}</div>
                </div>
                """, unsafe_allow_html=True)

                # "View chart / table" toggle button
                if has_panel:
                    btn_label = f"{'◀ Close' if (panel_is_open() and st.session_state['panel_idx'] == current_panel_idx) else '▶ View'} {panel_label}"
                    if st.button(btn_label, key=f"panel_btn_{i}"):
                        if panel_is_open() and st.session_state["panel_idx"] == current_panel_idx:
                            close_panel()
                        else:
                            open_panel(current_panel_idx)
                        st.rerun()

                assistant_idx += 1

    # ── INPUT BAR ─────────────────────────────────────────────────────────────
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="
        position:sticky;bottom:0;
        background:linear-gradient(to top,#0a0a0a 80%,transparent);
        padding:16px 0 12px;
    "></div>
    """, unsafe_allow_html=True)

    input_col, btn_col = st.columns([5, 1])
    with input_col:
        question = st.text_input(
            "q", placeholder="Ask anything about your workforce…",
            key=f"chat_input_{st.session_state['input_key']}",
            label_visibility="collapsed",
        )
    with btn_col:
        st.markdown('<div class="send-btn">', unsafe_allow_html=True)
        send = st.button("Send ›", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Clear + logout row
    c1, c2, c3 = st.columns([1, 1, 4])
    with c1:
        if st.button("✕ Clear chat"):
            st.session_state["messages"] = []
            st.session_state["panels"]   = []
            close_panel()
            st.rerun()
    with c2:
        if st.button("↩ Sign out"):
            for k in ["authenticated","user","messages","panels","panel_open","panel_idx","input_key","selected_model"]:
                st.session_state.pop(k, None)
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# ── PROCESS QUESTION ──────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
pending = st.session_state.pop("_pending_question", None) or (question if send and question.strip() else None)

if pending:
    q = pending.strip()
    if not any(m["content"] == q and m["role"] == "user" for m in msgs[-2:]):
        st.session_state["messages"].append({"role": "user", "content": q})

    # Build history for Claude (exclude latest user msg — passed separately)
    history = [m for m in st.session_state["messages"][:-1] if m["role"] in ("user","assistant")]

    with st.spinner("Thinking…"):
        try:
            text, chart, df = answer(q, history, user, st.session_state.get("selected_model", DEFAULT_MODEL))
        except Exception as e:
            text  = f"Sorry, I ran into an issue: {str(e)}"
            chart = None
            df    = None

    st.session_state["messages"].append({"role": "assistant", "content": text})

    # Determine panel label
    label = ""
    if chart is not None and df is not None:
        label = "Chart & Table"
    elif chart is not None:
        label = "Chart"
    elif df is not None:
        label = "Table"

    st.session_state["panels"].append({"chart": chart, "df": df, "label": label})

    # Auto-open panel if there's something to show
    if chart is not None or df is not None:
        new_idx = len([m for m in st.session_state["messages"] if m["role"] == "assistant"]) - 1
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
        st.markdown("""
        <div style="
            height:100vh; overflow-y:auto;
            background:#0d0d0d;
            border-left:1px solid #1a1a1a;
            padding:0;
        ">
        """, unsafe_allow_html=True)

        # Panel header
        st.markdown(f"""
        <div style="
            padding:14px 20px;
            border-bottom:1px solid #1a1a1a;
            display:flex;align-items:center;justify-content:space-between;
            position:sticky;top:0;background:#0d0d0d;z-index:50;
        ">
            <span style="font-family:'DM Mono',monospace;font-size:0.68rem;
                          letter-spacing:0.14em;text-transform:uppercase;color:#555;">
                {panel.get('label','Analysis')}
            </span>
            <span style="font-family:'DM Mono',monospace;font-size:0.68rem;
                          color:rgba(249,166,2,0.5);">
                Response {idx + 1}
            </span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='padding:16px 12px;'>", unsafe_allow_html=True)

        # Chart
        if panel.get("chart") is not None:
            st.plotly_chart(panel["chart"], use_container_width=True, config={"displayModeBar": False})

        # Dataframe
        if panel.get("df") is not None:
            df_show = panel["df"]
            if isinstance(df_show, pd.DataFrame) and not df_show.empty:
                st.markdown("""
                <p style="font-family:'DM Mono',monospace;font-size:0.65rem;
                           letter-spacing:0.12em;text-transform:uppercase;
                           color:#555;margin:12px 0 6px;">Data</p>
                """, unsafe_allow_html=True)
                # Round numeric columns
                num_cols = df_show.select_dtypes(include="number").columns
                df_display = df_show.copy()
                df_display[num_cols] = df_display[num_cols].round(2)
                st.dataframe(df_display, use_container_width=True, height=300)

        st.markdown("</div>", unsafe_allow_html=True)

        # Close button at bottom
        st.markdown("<div style='padding:8px 12px 20px;'>", unsafe_allow_html=True)
        if st.button("✕  Close panel", key="close_panel_btn", use_container_width=True):
            close_panel()
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)
