"""
app.py  —  Orb v2  |  Workforce Intelligence Platform
Fixes: chat persistence (panels), thinking placeholder, Enter-to-send,
       compact centered suggestion cards, natural sidebar collapse.
"""
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import re
from auth import authenticate, scope_label
from ai_engine import answer, MODELS, MODEL_NAMES, DEFAULT_MODEL, call_model
from chat_store import save_chat, load_all, load_chat, delete_chat, fmt_ts
from conversation_state import get_ctx, reset_ctx, add_response_summary
from clarifier import needs_clarification, build_clarification_message, \
                      store_clarification_buttons, resolve_clarification, \
                      ClarificationRequest

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Orb v2",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "Orb v2 — Workforce Intelligence Platform"},
)

AMBER   = "#D97706"
AMBERL  = "#FEF3C7"
BG      = "#F9FAFB"
CARD    = "#FFFFFF"
BORDER  = "#E5E7EB"
TEXT    = "#111827"
SUBTEXT = "#6B7280"
USERBG  = "#F3F4F6"
SB_BG   = "#111827"
SB_TEXT = "#F9FAFB"
SB_SUB  = "#9CA3AF"
SB_HVR  = "#1F2937"
SB_ACT  = "#374151"

# ══════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Mono:wght@300;400;500&family=Inter:wght@300;400;500;600&display=swap');

*, *::before, *::after {{ box-sizing: border-box; margin: 0; }}
html, body, .stApp {{ background: {BG} !important; color: {TEXT}; }}
#MainMenu, footer {{ display: none !important; }}
.block-container {{ padding: 0 1rem !important; max-width: 100% !important; }}

/* ── Sidebar — let Streamlit handle collapse/expand natively ── */
section[data-testid="stSidebar"] {{
    background: {SB_BG} !important;
    border-right: 1px solid #1F2937 !important;
}}
section[data-testid="stSidebar"] > div {{ background: {SB_BG} !important; }}
section[data-testid="stSidebar"] * {{ color: {SB_TEXT} !important; }}
section[data-testid="stSidebar"] .stButton > button {{
    background: transparent !important; border: none !important;
    color: {SB_TEXT} !important; font-family: 'Inter', sans-serif !important;
    font-size: 0.82rem !important; font-weight: 400 !important;
    padding: 8px 12px !important; border-radius: 8px !important;
    text-align: left !important; width: 100% !important; box-shadow: none !important;
}}
section[data-testid="stSidebar"] .stButton > button:hover {{
    background: {SB_HVR} !important; color: {SB_TEXT} !important; border: none !important;
}}
section[data-testid="stSidebar"] .new-chat-btn > button {{
    background: rgba(217,119,6,0.18) !important;
    border: 1px solid rgba(217,119,6,0.35) !important;
    color: {AMBER} !important; border-radius: 8px !important;
    font-weight: 600 !important; font-size: 0.84rem !important;
}}
section[data-testid="stSidebar"] .new-chat-btn > button:hover {{
    background: rgba(217,119,6,0.28) !important;
}}
section[data-testid="stSidebar"] .signout-btn > button {{
    background: transparent !important; border: 1px solid #374151 !important;
    color: {SB_SUB} !important; border-radius: 8px !important; font-size: 0.80rem !important;
}}
section[data-testid="stSidebar"] .signout-btn > button:hover {{
    border-color: #EF4444 !important; color: #FCA5A5 !important;
    background: rgba(239,68,68,0.08) !important;
}}
section[data-testid="stSidebar"] .del-btn > button {{
    background: transparent !important; border: none !important;
    color: #4B5563 !important; font-size: 0.70rem !important;
    padding: 3px 6px !important; border-radius: 4px !important;
    box-shadow: none !important; min-width: 0 !important;
}}
section[data-testid="stSidebar"] .del-btn > button:hover {{
    color: #FCA5A5 !important; background: rgba(239,68,68,0.1) !important;
}}

::-webkit-scrollbar {{ width: 4px; }}
::-webkit-scrollbar-track {{ background: #f1f1f1; }}
::-webkit-scrollbar-thumb {{ background: #d1d5db; border-radius: 2px; }}

.stTextInput > div > div > input {{
    background: {CARD} !important; border: 1px solid {BORDER} !important;
    border-radius: 10px !important; color: {TEXT} !important;
    font-family: 'Inter', sans-serif !important; font-size: 0.92rem !important;
    padding: 12px 16px !important; box-shadow: 0 1px 2px rgba(0,0,0,0.04) !important;
}}
.stTextInput > div > div > input:focus {{
    border-color: {AMBER} !important; box-shadow: 0 0 0 3px rgba(217,119,6,0.1) !important;
}}
.stTextInput > div > div > input::placeholder {{ color: #9CA3AF !important; }}
.stTextInput label {{ display: none !important; }}

.stButton > button, [data-testid="stFormSubmitButton"] > button {{
    background: {CARD} !important; border: 1px solid {BORDER} !important;
    border-radius: 8px !important; color: {SUBTEXT} !important;
    font-family: 'Inter', sans-serif !important; font-size: 0.82rem !important;
    font-weight: 500 !important; padding: 7px 14px !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04) !important; transition: all 0.15s !important;
}}
.stButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover {{
    border-color: {AMBER} !important; color: {AMBER} !important; background: {AMBERL} !important;
}}
.send-btn > div > button, .send-btn [data-testid="stFormSubmitButton"] > button {{
    background: {AMBER} !important; border: none !important;
    border-radius: 8px !important; color: #fff !important;
    font-family: 'Inter', sans-serif !important; font-size: 0.85rem !important;
    font-weight: 600 !important; padding: 9px 22px !important;
    box-shadow: 0 2px 8px rgba(217,119,6,0.28) !important;
}}
.send-btn > div > button:hover, .send-btn [data-testid="stFormSubmitButton"] > button:hover {{
    background: #B45309 !important; color: #fff !important;
}}

div[data-testid="stSelectbox"] > div > div {{
    background: {CARD} !important; border: 1px solid {BORDER} !important;
    border-radius: 8px !important; font-family: 'Inter', sans-serif !important; font-size: 0.84rem !important;
}}
div[data-testid="stSelectbox"] label {{
    font-size: 0.70rem !important; font-weight: 600 !important;
    text-transform: uppercase !important; letter-spacing: 0.06em !important; color: {SUBTEXT} !important;
}}
[data-testid="stMetric"] {{
    background: {CARD}; border: 1px solid {BORDER}; border-radius: 10px;
    padding: 12px 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}}
[data-testid="stMetricLabel"] {{ color: {SUBTEXT} !important; font-size: 0.75rem !important; }}
[data-testid="stMetricValue"] {{ color: {AMBER} !important; font-size: 1.4rem !important; font-weight:700 !important; }}
[data-testid="stDataFrame"] {{ border: 1px solid {BORDER} !important; border-radius: 8px; }}
.stSpinner > div {{ border-top-color: {AMBER} !important; }}
div[data-testid="stForm"] {{ border: none !important; padding: 0 !important; }}

/* Compact centered suggestion cards */
.sug-card > button {{
    background: {CARD} !important; border: 1px solid {BORDER} !important;
    border-radius: 10px !important; color: {TEXT} !important;
    font-family: 'Inter', sans-serif !important; font-size: 0.76rem !important;
    font-weight: 400 !important; padding: 10px 12px !important;
    text-align: center !important; min-height: 46px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important; line-height: 1.35 !important;
}}
.sug-card > button:hover {{
    border-color: {AMBER} !important; background: {AMBERL} !important; color: {AMBER} !important;
}}

/* Thinking indicator */
@keyframes orbPulse {{
    0%, 100% {{ opacity: 0.3; transform: scale(0.85); }}
    50%      {{ opacity: 1;   transform: scale(1.05); }}
}}
.thinking-dot {{
    display: inline-block; width: 6px; height: 6px; border-radius: 50%;
    background: {AMBER}; margin: 0 2px; animation: orbPulse 1.2s ease-in-out infinite;
}}
.thinking-dot:nth-child(2) {{ animation-delay: 0.2s; }}
.thinking-dot:nth-child(3) {{ animation-delay: 0.4s; }}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
for k, v in {
    "authenticated":   False,
    "user":            None,
    "messages":        [],
    "panels":          [],
    "panel_open":      False,
    "panel_idx":       -1,
    "input_key":       0,
    "selected_model":  DEFAULT_MODEL,
    "last_df":         None,
    "current_chat_id": None,
    "vector_index_built": False,   # tracks whether FAISS index is ready
    "_clarification_buttons": [],  # clickable option chips for clarifier
    "_clarification_request": None, # active ClarificationRequest object
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

def clean_md(text: str) -> str:
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*',     r'\1', text)
    text = re.sub(r'__(.+?)__',     r'\1', text)
    text = re.sub(r'_(.+?)_',       r'\1', text)
    return text.strip()

# ── Phase 2: Build vector index once per app session ─────────────────────────
def _maybe_build_vector_index():
    """Build FAISS index on first run. Shows a status message, not a blocking spinner."""
    from vector_store import VECTOR_STORE_ENABLED, build_index, status as vs_status
    if not VECTOR_STORE_ENABLED:
        return
    if st.session_state["vector_index_built"]:
        return
    vs = vs_status()
    if vs["fr_indexed"] > 0:
        st.session_state["vector_index_built"] = True
        return
    # Build it — this is the slow step (model download on first cold start)
    from data import get_data
    fh, fr = get_data()
    with st.spinner("Building search index… (first load only, ~30s on cold start)"):
        build_index(fh, fr)
    st.session_state["vector_index_built"] = True

ASSISTANT_AVATAR = (
    "background:radial-gradient(circle at 38% 35%,#fff8e0 0%,#F9A602 30%,#c97f00 60%,#3d1f00 100%);"
    "box-shadow:0 0 6px rgba(217,119,6,0.28);"
)

def render_user_bubble(content):
    st.markdown(f"""
    <div style="display:flex;justify-content:flex-end;margin:14px 0 4px;">
        <div style="background:{USERBG};border:1px solid {BORDER};
                    border-radius:16px 16px 4px 16px;padding:10px 15px;
                    max-width:72%;font-family:'Inter',sans-serif;
                    font-size:0.88rem;color:{TEXT};line-height:1.55;">
            {content}</div>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ── LOGIN PAGE ────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
if not st.session_state["authenticated"]:
    st.markdown("<style>section[data-testid='stSidebar']{display:none!important;}</style>",
                unsafe_allow_html=True)

    components.html("""
    <!DOCTYPE html><html><head>
    <link href="https://fonts.googleapis.com/css2?family=Syne:wght@800&family=DM+Mono:wght@400&display=swap" rel="stylesheet">
    <style>
    *{margin:0;padding:0;box-sizing:border-box;}
    body{background:transparent;display:flex;flex-direction:column;
         align-items:center;justify-content:center;height:220px;overflow:hidden;}
    .ring{position:absolute;border-radius:50%;top:50%;left:50%;
          border:1px solid rgba(217,119,6,0.12);animation:pulse 4s ease-in-out infinite;}
    .ring:nth-child(1){width:130px;height:130px;}
    .ring:nth-child(2){width:210px;height:210px;animation-delay:.7s;border-color:rgba(217,119,6,0.07);}
    .ring:nth-child(3){width:310px;height:310px;animation-delay:1.4s;border-color:rgba(217,119,6,0.04);}
    @keyframes pulse{0%,100%{transform:translate(-50%,-50%) scale(1);opacity:1;}
                     50%{transform:translate(-50%,-50%) scale(1.05);opacity:0.4;}}
    .orb{width:64px;height:64px;border-radius:50%;position:relative;z-index:2;
         background:radial-gradient(circle at 38% 35%,#fff8e0 0%,#F9A602 25%,#c97f00 52%,#7a4500 78%,#3d1f00 100%);
         box-shadow:0 0 24px 8px rgba(217,119,6,0.3);animation:float 5s ease-in-out infinite;}
    .orb::before{content:'';position:absolute;top:17%;left:21%;width:36%;height:24%;
                 background:radial-gradient(ellipse,rgba(255,255,255,0.6) 0%,transparent 70%);
                 border-radius:50%;transform:rotate(-30deg);}
    @keyframes float{0%,100%{transform:translateY(0);}50%{transform:translateY(-7px);}}
    .title{font-family:'Syne',sans-serif;font-size:1.7rem;font-weight:800;
           color:#111827;margin-top:12px;text-align:center;}
    .title span{color:#D97706;}
    .sub{font-family:'DM Mono',monospace;font-size:0.56rem;letter-spacing:0.22em;
        text-transform:uppercase;color:#9CA3AF;margin-top:4px;text-align:center;}
    </style></head><body>
    <div class="ring"></div><div class="ring"></div><div class="ring"></div>
    <div class="orb"></div>
    <div class="title">Orb <span>v2</span></div>
    <div class="sub">Workforce Intelligence Platform</div>
    </body></html>""", height=220)

    col_l, col_c, col_r = st.columns([1, 1.4, 1])
    with col_c:
        st.markdown(f"""
        <div style="background:{CARD};border:1px solid {BORDER};border-radius:16px;
                    padding:24px 24px 4px;box-shadow:0 4px 24px rgba(0,0,0,0.06);">
        <p style="font-family:'Inter',sans-serif;font-size:0.70rem;font-weight:600;
                  letter-spacing:0.12em;text-transform:uppercase;color:{SUBTEXT};
                  text-align:center;margin-bottom:18px;">Sign in to continue</p>
        </div>""", unsafe_allow_html=True)

        username = st.text_input("u", placeholder="Username", key="login_user",
                                 label_visibility="collapsed")
        password = st.text_input("p", placeholder="Password", type="password",
                                 key="login_pass", label_visibility="collapsed")

        st.markdown(f"""
        <p style="font-family:'Inter',sans-serif;font-size:0.68rem;font-weight:600;
                  text-transform:uppercase;letter-spacing:0.08em;color:{SUBTEXT};
                  margin:16px 0 8px;">Choose AI Model</p>
        """, unsafe_allow_html=True)

        mcols = st.columns(len(MODEL_NAMES))
        for i, name in enumerate(MODEL_NAMES):
            cfg    = MODELS[name]
            is_sel = st.session_state["selected_model"] == name
            with mcols[i]:
                st.markdown(f"""
                <div style="background:{'#FEF3C7' if is_sel else CARD};
                            border:{'2px' if is_sel else '1px'} solid {'#D97706' if is_sel else BORDER};
                            border-radius:10px;padding:10px;
                            box-shadow:{'0 0 0 3px rgba(217,119,6,0.1)' if is_sel else 'none'};">
                    <div style="font-size:0.72rem;font-weight:600;font-family:'Inter',sans-serif;
                                color:{'#D97706' if is_sel else TEXT};">{name}</div>
                    <div style="font-size:0.64rem;color:{SUBTEXT};margin-top:2px;
                                font-family:'Inter',sans-serif;">{cfg['description']}</div>
                    <span style="display:inline-block;margin-top:4px;background:{cfg['tag_color']}18;
                                 color:{cfg['tag_color']};border-radius:3px;padding:1px 5px;
                                 font-family:'DM Mono',monospace;font-size:0.56rem;">{cfg['tag']}</span>
                </div>""", unsafe_allow_html=True)
                if st.button("✓" if is_sel else "Select", key=f"mp_{i}",
                             use_container_width=True):
                    st.session_state["selected_model"] = name
                    st.rerun()

        st.markdown(f"""
        <p style="font-family:'DM Mono',monospace;font-size:0.60rem;color:#9CA3AF;
                  text-align:center;margin:14px 0 10px;">
        ceo / coo.apac / head.sg / hr.admin &nbsp;·&nbsp; password: demo</p>
        """, unsafe_allow_html=True)

        st.markdown('<div class="send-btn"><div>', unsafe_allow_html=True)
        if st.button("Enter the Orb →", use_container_width=True, key="login_btn"):
            u = authenticate(username, password)
            if u:
                st.session_state["authenticated"] = True
                st.session_state["user"]           = u
                st.rerun()
            else:
                st.error("Invalid credentials.")
        st.markdown('</div></div>', unsafe_allow_html=True)

    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# ── HELPERS ───────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
user   = st.session_state["user"]
msgs   = st.session_state["messages"]
panels = st.session_state["panels"]

# Build vector index if not already done this session
_maybe_build_vector_index()

def open_panel(idx):
    st.session_state["panel_open"] = True
    st.session_state["panel_idx"]  = idx

def close_panel():
    st.session_state["panel_open"] = False
    st.session_state["panel_idx"]  = -1

def panel_is_open():
    return st.session_state["panel_open"] and st.session_state["panel_idx"] >= 0

def persist_current_chat():
    """Save current chat (messages + panels) under its existing id, or create one."""
    if st.session_state["messages"]:
        sid = save_chat(user["display_name"],
                        st.session_state["messages"],
                        st.session_state["selected_model"],
                        st.session_state["panels"],
                        session_id=st.session_state.get("current_chat_id"))
        st.session_state["current_chat_id"] = sid

def start_new_chat():
    persist_current_chat()
    st.session_state["messages"]        = []
    st.session_state["panels"]          = []
    st.session_state["last_df"]         = None
    st.session_state["current_chat_id"] = None
    st.session_state["_clarification_buttons"] = []
    st.session_state["_clarification_request"] = None
    reset_ctx()
    close_panel()

def restore_chat(sid: str):
    data = load_chat(sid)
    if data:
        st.session_state["messages"]        = data["messages"]
        st.session_state["panels"]          = data.get("panels", [])
        st.session_state["current_chat_id"] = sid
        st.session_state["selected_model"]  = data.get("model", DEFAULT_MODEL)
        st.session_state["last_df"]         = None
        close_panel()

# ══════════════════════════════════════════════════════════════════════════════
# ── SIDEBAR ───────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:10px;padding:16px 4px 12px;">
        <div style="width:26px;height:26px;border-radius:50%;flex-shrink:0;
            background:radial-gradient(circle at 38% 35%,#fff8e0 0%,#F9A602 30%,#c97f00 60%,#3d1f00 100%);
            box-shadow:0 0 8px rgba(249,166,2,0.4);"></div>
        <span style="font-family:'Syne',sans-serif;font-size:1rem;font-weight:800;
                     color:{SB_TEXT};">Orb <span style="color:{AMBER};">v2</span></span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="new-chat-btn">', unsafe_allow_html=True)
    if st.button("＋  New Chat", use_container_width=True, key="sb_new"):
        start_new_chat()
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:1px;background:#1F2937;margin:10px 0;'></div>",
                unsafe_allow_html=True)

    st.markdown(f"""
    <p style="font-family:'Inter',sans-serif;font-size:0.65rem;font-weight:600;
              text-transform:uppercase;letter-spacing:0.1em;color:#4B5563;
              padding:0 4px 6px;">Recent</p>
    """, unsafe_allow_html=True)

    recent = load_all(user["display_name"])
    if not recent:
        st.markdown(f"""
        <p style="font-family:'Inter',sans-serif;font-size:0.76rem;color:#4B5563;
                  padding:4px 4px;">No saved chats yet.</p>
        """, unsafe_allow_html=True)
    else:
        for chat in recent[:15]:
            is_active = st.session_state["current_chat_id"] == chat["id"]
            rc1, rc2 = st.columns([6, 1])
            with rc1:
                label = chat["title"][:34] + ("…" if len(chat["title"])>34 else "")
                if is_active:
                    st.markdown(f"""
                    <div style="background:{SB_ACT};border-radius:8px;padding:7px 10px;margin-bottom:1px;">
                        <div style="font-size:0.80rem;color:{SB_TEXT};font-family:'Inter',sans-serif;
                                    white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{label}</div>
                        <div style="font-size:0.64rem;color:{SB_SUB};margin-top:1px;
                                    font-family:'DM Mono',monospace;">{fmt_ts(chat['timestamp'])}</div>
                    </div>""", unsafe_allow_html=True)
                else:
                    if st.button(label, key=f"sb_chat_{chat['id']}", use_container_width=True):
                        persist_current_chat()
                        restore_chat(chat["id"])
                        st.rerun()
                    st.markdown(f"""
                    <div style="font-size:0.62rem;color:#4B5563;margin:-6px 0 4px 4px;
                                font-family:'DM Mono',monospace;">{fmt_ts(chat['timestamp'])}</div>
                    """, unsafe_allow_html=True)
            with rc2:
                st.markdown('<div class="del-btn">', unsafe_allow_html=True)
                if st.button("✕", key=f"sb_del_{chat['id']}"):
                    delete_chat(chat["id"])
                    if is_active:
                        st.session_state["messages"]        = []
                        st.session_state["panels"]          = []
                        st.session_state["current_chat_id"] = None
                        close_panel()
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='flex:1;min-height:30px;'></div>", unsafe_allow_html=True)
    st.markdown("<div style='height:1px;background:#1F2937;margin:10px 0;'></div>",
                unsafe_allow_html=True)

    st.markdown(f"""
    <div style="padding:4px 4px 8px;">
        <div style="font-family:'Inter',sans-serif;font-size:0.80rem;
                    color:{SB_TEXT};font-weight:500;">{user['display_name']}</div>
        <div style="font-family:'DM Mono',monospace;font-size:0.60rem;color:{SB_SUB};margin-top:1px;">
            {user['role']} · <span style="color:{AMBER};">{scope_label(user)}</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="signout-btn">', unsafe_allow_html=True)
    if st.button("↩  Sign out", use_container_width=True, key="sb_signout"):
        persist_current_chat()
        for k in ["authenticated","user","messages","panels","panel_open",
                  "panel_idx","input_key","selected_model","last_df","current_chat_id"]:
            st.session_state.pop(k, None)
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# ── MAIN CONTENT ──────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
if panel_is_open():
    chat_col, panel_col = st.columns([1.1, 0.9], gap="small")
else:
    chat_col  = st.container()
    panel_col = None

with chat_col:
    cfg_now = MODELS[st.session_state["selected_model"]]

    # ── TOP BAR ───────────────────────────────────────────────────────────────
    tl, tr = st.columns([3, 1])
    with tl:
        st.markdown(f"""
        <div style="padding:14px 0 8px 8px;">
            <span style="font-family:'DM Mono',monospace;font-size:0.60rem;letter-spacing:0.14em;
                         text-transform:uppercase;color:{SUBTEXT};">Workforce Intelligence</span>
            &nbsp;
            <span style="background:{cfg_now['tag_color']}18;color:{cfg_now['tag_color']};
                         border-radius:5px;padding:2px 8px;font-family:'DM Mono',monospace;
                         font-size:0.58rem;">{st.session_state['selected_model']}</span>
        </div>""", unsafe_allow_html=True)
    with tr:
        st.markdown(f"""
        <div style="display:flex;align-items:center;justify-content:flex-end;gap:8px;padding:12px 8px 8px 0;">
            <div style="width:28px;height:28px;border-radius:50%;
                background:{AMBERL};border:1px solid rgba(217,119,6,0.3);
                display:flex;align-items:center;justify-content:center;
                font-family:'DM Mono',monospace;font-size:0.64rem;font-weight:600;color:{AMBER};">
                {user['avatar']}</div>
            <div style="font-family:'Inter',sans-serif;font-size:0.78rem;color:{TEXT};font-weight:500;">
                {user['display_name'].split()[0]}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown(f"<div style='border-bottom:1px solid {BORDER};'></div>", unsafe_allow_html=True)

    # ── MESSAGES ──────────────────────────────────────────────────────────────
    with st.container():
        if not msgs:
            st.markdown(f"""
            <div style="text-align:center;padding:48px 20px 28px;">
                <div style="font-family:'DM Mono',monospace;font-size:0.60rem;letter-spacing:0.2em;
                            text-transform:uppercase;color:rgba(217,119,6,0.7);margin-bottom:8px;">
                    Good to see you, {user['display_name'].split()[0]}</div>
                <div style="font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:700;
                            color:{TEXT};margin-bottom:6px;">What would you like to know?</div>
                <div style="font-family:'Inter',sans-serif;font-size:0.80rem;color:{SUBTEXT};
                            max-width:380px;margin:0 auto 20px;">
                    Ask anything — incentive performance, headcount, PMGM ratings,
                    or cross-source insights.</div>
            </div>""", unsafe_allow_html=True)

            SUGGESTIONS = [
                "What % of employees hit max payout this cycle?",
                "Who has missed targets for 3+ consecutive periods?",
                "Show me the PMGM rating distribution",
                "Are there non-active employees with payouts?",
                "Give me a cycle summary",
                "Compare countries on incentive attainment",
            ]
            # Centered grid: side-padding columns + 3 content columns, 2 rows
            pad, c1, c2, c3, pad2 = st.columns([0.5, 1, 1, 1, 0.5])
            content_cols = [c1, c2, c3]
            for i, query in enumerate(SUGGESTIONS):
                col = content_cols[i % 3]
                with col:
                    st.markdown('<div class="sug-card">', unsafe_allow_html=True)
                    if st.button(query, key=f"sug_{i}", use_container_width=True):
                        st.session_state["messages"].append({"role":"user","content":query})
                        st.session_state["_pending_question"] = query
                        st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)
                if i == 2:
                    # spacing before second row
                    pad, c1, c2, c3, pad2 = st.columns([0.5, 1, 1, 1, 0.5])
                    content_cols = [c1, c2, c3]

        # ── Render history ────────────────────────────────────────────────────
        assistant_idx = 0
        for i, msg in enumerate(msgs):
            if msg["role"] == "user":
                render_user_bubble(msg["content"])
            else:
                has_panel = (assistant_idx < len(panels) and
                    (panels[assistant_idx].get("chart") is not None or
                     panels[assistant_idx].get("df")    is not None))
                panel_label       = panels[assistant_idx].get("label","") if assistant_idx < len(panels) else ""
                current_panel_idx = assistant_idx

                content_html = msg["content"].replace("\n", "<br>")
                st.markdown(f"""
                <div style="display:flex;align-items:flex-start;gap:8px;margin:4px 0 6px;">
                    <div style="width:24px;height:24px;border-radius:50%;flex-shrink:0;margin-top:3px;{ASSISTANT_AVATAR}"></div>
                    <div style="background:{CARD};border:1px solid {BORDER};
                                border-radius:4px 16px 16px 16px;padding:12px 16px;
                                max-width:84%;font-family:'Inter',sans-serif;font-size:0.88rem;
                                color:{TEXT};line-height:1.65;box-shadow:0 1px 4px rgba(0,0,0,0.05);">
                        {content_html}</div>
                </div>""", unsafe_allow_html=True)

                if has_panel:
                    is_open_now = panel_is_open() and st.session_state["panel_idx"] == current_panel_idx
                    if st.button(
                        f"{'◀ Close' if is_open_now else '▶ View'}  {panel_label}",
                        key=f"panel_btn_{i}"
                    ):
                        close_panel() if is_open_now else open_panel(current_panel_idx)
                        st.rerun()

                assistant_idx += 1

    # ── INPUT (form — Enter submits) ─────────────────────────────────────────
    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    with st.form(key="chat_form", clear_on_submit=True):
        ic, bc = st.columns([5, 1])
        with ic:
            question = st.text_input(
                "q", placeholder="Ask anything about your workforce…",
                key=f"chat_input_{st.session_state['input_key']}",
                label_visibility="collapsed",
            )
        with bc:
            st.markdown('<div class="send-btn">', unsafe_allow_html=True)
            send = st.form_submit_button("Send →", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# ── RENDER CLARIFICATION BUTTONS (from previous turn if any) ─────────────────
# ══════════════════════════════════════════════════════════════════════════════
_clar_buttons = st.session_state.get("_clarification_buttons", [])
if _clar_buttons:
    with chat_col:
        st.markdown(f"""
        <div style="padding:4px 0 2px 36px;">
          <span style="font-family:'DM Mono',monospace;font-size:0.60rem;
                       color:{SUBTEXT};text-transform:uppercase;letter-spacing:0.08em;">
            Select an option or type your answer</span>
        </div>""", unsafe_allow_html=True)
        btn_cols = st.columns(min(len(_clar_buttons), 3))
        for bi, btn in enumerate(_clar_buttons):
            with btn_cols[bi % len(btn_cols)]:
                st.markdown('<div class="sug-card">', unsafe_allow_html=True)
                if st.button(btn["label"], key=f"clar_btn_{bi}", use_container_width=True):
                    # Treat the button value as the next user question
                    st.session_state["_clarification_buttons"] = []
                    st.session_state["_clarification_request"] = None
                    st.session_state["messages"].append(
                        {"role": "user", "content": btn["label"]}
                    )
                    st.session_state["_pending_question"] = btn["value"]
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# ── PROCESS QUESTION ──────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
pending = st.session_state.pop("_pending_question", None) or (
    question if send and question.strip() else None
)

if pending:
    q = pending.strip()

    # Clear any stale clarification buttons on a new typed message
    if not st.session_state.get("_clarification_request"):
        st.session_state["_clarification_buttons"] = []

    if not any(m["content"] == q and m["role"] == "user" for m in msgs[-2:]):
        st.session_state["messages"].append({"role": "user", "content": q})

    history = [m for m in st.session_state["messages"][:-1]
               if m["role"] in ("user", "assistant")]

    # ── Step 0: Check if this is a clarification resolution ──────────────────
    # If the user typed something that matches a pending clarification option,
    # resolve it to the full refined question before routing.
    cr: ClarificationRequest = st.session_state.get("_clarification_request")
    if cr:
        q = resolve_clarification(q, cr)
        st.session_state["_clarification_request"] = None
        st.session_state["_clarification_buttons"] = []

    # ── Step 1: Pre-route to check if clarification is needed ────────────────
    # We do a lightweight route first to check ambiguity.
    from router import route as _route
    from semantic import normalise as _normalise
    _pre_routing = _route(_normalise(q), history,
                          list(st.session_state.get("last_df").columns)
                          if st.session_state.get("last_df") is not None else [])
    _ctx = get_ctx()

    _needs_clar, _clar_request = needs_clarification(q, _pre_routing, _ctx, user)

    if _needs_clar and _clar_request:
        # Send the clarification question as Orb's response
        clar_text = build_clarification_message(_clar_request)
        store_clarification_buttons(_clar_request)
        st.session_state["_clarification_request"] = _clar_request

        from conversation_state import set_clarification
        set_clarification(
            question=_clar_request.question,
            options=[o.label for o in _clar_request.options],
            intent_candidates=[_pre_routing.get("intent", "free_form")],
            clarification_type=_clar_request.clarification_type,
        )

        # Log the clarification as assistant message (no data panel)
        st.session_state["messages"].append({"role": "assistant", "content": clar_text})
        st.session_state["panels"].append({"chart": None, "df": None, "label": ""})

        # Log to debug panel
        try:
            from debug_logger import log_interaction
            log_interaction(
                question=q, routing=_pre_routing,
                retrieval_mode="clarification", intent=_pre_routing.get("intent",""),
                data_context="[Clarification sent — no data fetched yet]",
                system_prompt="[Clarification mode]", ai_response=clar_text,
            )
        except Exception:
            pass

        persist_current_chat()
        st.session_state["input_key"] += 1
        st.rerun()

    # ── Step 2: Normal answer flow ────────────────────────────────────────────
    with chat_col:
        # ── Thinking placeholder ─────────────────────────────────────────────
        think_box = st.empty()
        think_box.markdown(f"""
        <div style="display:flex;align-items:center;gap:10px;margin:6px 0 6px;">
            <div style="width:24px;height:24px;border-radius:50%;flex-shrink:0;{ASSISTANT_AVATAR}"></div>
            <div style="background:{CARD};border:1px solid {BORDER};border-radius:4px 16px 16px 16px;
                        padding:13px 18px;box-shadow:0 1px 4px rgba(0,0,0,0.05);">
                <span class="thinking-dot"></span><span class="thinking-dot"></span><span class="thinking-dot"></span>
            </div>
        </div>""", unsafe_allow_html=True)

        try:
            stream, chart, df, debug_info = answer(
                q, history, user,
                model_name=st.session_state.get("selected_model", DEFAULT_MODEL),
                last_df=st.session_state.get("last_df"),
            )

            think_box.empty()
            avatar_row = st.empty()
            avatar_row.markdown(f"""
            <div style="display:flex;align-items:center;gap:10px;margin:6px 0 2px;">
                <div style="width:24px;height:24px;border-radius:50%;flex-shrink:0;{ASSISTANT_AVATAR}"></div>
                <span style="font-family:'DM Mono',monospace;font-size:0.62rem;color:{SUBTEXT};
                             letter-spacing:0.1em;text-transform:uppercase;">Orb</span>
            </div>""", unsafe_allow_html=True)
            collected = st.write_stream(stream)
            text = clean_md(collected or "")

        except Exception as e:
            think_box.empty()
            st.error(f"Sorry, I ran into an issue: {str(e)}")
            text       = f"Sorry, I ran into an issue: {str(e)}"
            chart      = None
            df         = None
            debug_info = {}

    # ── Fallback for blank text ──────────────────────────────────────────────
    if not text.strip() and df is not None:
        try:
            fb_system = "You are a concise analyst. Summarise the data in 2-3 plain sentences, no markdown bold or italic."
            fb_msgs   = [{"role": "user", "content": f"Summarise this data for an executive:\n{df.head(20).to_string(index=False)}"}]
            text = clean_md(call_model(fb_msgs, fb_system,
                                       st.session_state.get("selected_model", DEFAULT_MODEL)))
        except Exception:
            text = "A chart and data table are available — click View to explore the results."

    if not text.strip():
        text = "No data was found for your query within your permitted scope."

    st.session_state["messages"].append({"role": "assistant", "content": text})

    # ── Update rolling response summaries for conversation context ────────────
    try:
        # First sentence or first 120 chars — enough for context without bloat
        summary = text.split(".")[0][:120].strip()
        add_response_summary(summary)
    except Exception:
        pass

    # ── Log to debug panel ────────────────────────────────────────────────────
    try:
        from debug_logger import log_interaction
        log_interaction(
            question       = q,
            routing        = debug_info.get("routing", {}),
            retrieval_mode = debug_info.get("retrieval_mode", "unknown"),
            intent         = debug_info.get("intent", "unknown"),
            data_context   = debug_info.get("data_context", ""),
            system_prompt  = debug_info.get("system_prompt", ""),
            ai_response    = text,
        )
    except Exception:
        pass   # never crash the main app over debug logging

    if df is not None:
        st.session_state["last_df"] = df

    label = ("Chart & Table" if chart is not None and df is not None
             else "Chart" if chart is not None
             else "Table" if df is not None else "")
    st.session_state["panels"].append({"chart": chart, "df": df, "label": label})

    if chart is not None or df is not None:
        new_idx = len([m for m in st.session_state["messages"]
                       if m["role"] == "assistant"]) - 1
        open_panel(new_idx)

    # Persist after every turn so a chat is never lost
    persist_current_chat()

    st.session_state["input_key"] += 1
    st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# ── SIDE PANEL ────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
if panel_is_open() and panel_col is not None:
    idx   = st.session_state["panel_idx"]
    panel = panels[idx] if idx < len(panels) else {}

    with panel_col:
        st.markdown(f"""
        <div style="display:flex;align-items:center;justify-content:space-between;
                    padding:10px 16px 8px;border-bottom:1px solid {BORDER};
                    background:{CARD};position:sticky;top:0;z-index:50;">
            <span style="font-family:'DM Mono',monospace;font-size:0.62rem;
                          letter-spacing:0.12em;text-transform:uppercase;color:{SUBTEXT};">
                {panel.get('label','Analysis')}</span>
        </div>""", unsafe_allow_html=True)

        if panel.get("chart") is not None:
            st.plotly_chart(panel["chart"], use_container_width=True,
                            config={"displayModeBar": False})

        if panel.get("df") is not None:
            df_show = panel["df"]
            if isinstance(df_show, pd.DataFrame) and not df_show.empty:
                st.markdown(f"""
                <p style="font-family:'DM Mono',monospace;font-size:0.60rem;
                           text-transform:uppercase;color:{SUBTEXT};padding:4px 0 4px 2px;">Data</p>
                """, unsafe_allow_html=True)
                dc = df_show.copy()
                num_cols = dc.select_dtypes("number").columns
                if len(num_cols) > 0:
                    dc[num_cols] = dc[num_cols].round(2)
                st.dataframe(dc, use_container_width=True, height=280)

        st.markdown("<div style='padding:8px 0 16px;'>", unsafe_allow_html=True)
        if st.button("✕  Close", key="close_panel_btn", use_container_width=True):
            close_panel()
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
