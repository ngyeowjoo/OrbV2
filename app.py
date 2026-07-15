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

from theme import apply_theme, render_toggle, get_colours, AMBER

# ── Unpack colours for use throughout this file ────────────────────────────────
_t      = get_colours()
BG      = _t["BG"]
CARD    = _t["CARD"]
BORDER  = _t["BORDER"]
TEXT    = _t["TEXT"]
SUBTEXT = _t["SUBTEXT"]
USERBG  = _t["USERBG"]
SB_BG   = _t["SB_BG"]
SB_TEXT = _t["SB_TEXT"]
SB_SUB  = _t["SB_SUB"]
SB_HVR  = _t["SB_HVR"]
SB_ACT  = _t["SB_ACT"]
AMBERL  = _t["AMBERL"]

apply_theme(page="main")

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
    "vector_index_built": False,
    "_clarification_buttons": [],
    "_clarification_request": None,
    "theme_mode":          "light",   # "light" | "dark"  — light is default
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
    from data import get_flash_home, get_flash_reward
    fh = get_flash_home(["ALL"])
    fr = get_flash_reward(["ALL"])
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

    # ── Theme Toggle ──────────────────────────────────────────────────────────
    render_toggle(key_suffix="main")
    # Lets users with multi-country or global access pin a country for the session,
    # so they don't have to re-specify it every turn.
    user_countries = user.get("countries", [])
    is_multi_scope = "ALL" in user_countries or len(user_countries) > 1

    if is_multi_scope:
        st.markdown(f"""
        <p style="font-family:'Inter',sans-serif;font-size:0.65rem;font-weight:600;
                  text-transform:uppercase;letter-spacing:0.1em;color:#4B5563;
                  padding:0 4px 4px;">Scope Pin</p>
        """, unsafe_allow_html=True)

        if "ALL" in user_countries:
            pin_options = ["None (Global)", "SG", "MY", "PH", "TH", "ID"]
        else:
            pin_options = ["None (all my countries)"] + user_countries

        current_pin = get_ctx().get("pinned_country")
        current_idx = 0
        if current_pin and current_pin in pin_options:
            current_idx = pin_options.index(current_pin)

        selected_pin = st.selectbox(
            "pin_scope", pin_options,
            index=current_idx,
            key="scope_pin_select",
            label_visibility="collapsed",
        )

        # Update conversation context when pin changes
        new_pin = None if selected_pin.startswith("None") else selected_pin
        if new_pin != current_pin:
            from conversation_state import update_ctx as _uctx
            _uctx(pinned_country=new_pin)
            if new_pin:
                _uctx(active_filters={"country": new_pin})

        if new_pin:
            st.markdown(f"""
            <div style="font-family:'DM Mono',monospace;font-size:0.62rem;
                        color:{AMBER};padding:2px 4px 8px;">
                ⬡ All queries scoped to {new_pin}</div>
            """, unsafe_allow_html=True)
        st.markdown("<div style='height:1px;background:#1F2937;margin:2px 0 10px;'></div>",
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
        # Ensure panels list always has one entry per assistant message (pad if needed)
        n_assistant = sum(1 for m in msgs if m["role"] == "assistant")
        while len(panels) < n_assistant:
            panels.append({"chart": None, "df": None, "label": ""})

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
                    <div style="max-width:84%;">
                        <div style="background:{CARD};border:1px solid {BORDER};
                                    border-radius:4px 16px 16px 16px;padding:12px 16px;
                                    font-family:'Inter',sans-serif;font-size:0.88rem;
                                    color:{TEXT};line-height:1.65;box-shadow:0 1px 4px rgba(0,0,0,0.05);">
                            {content_html}</div>
                        {f'''<div style="font-style:italic;font-size:0.70rem;color:{SUBTEXT};
                                     padding:4px 4px 0;">Understood as: {msg["understood"]}</div>'''
                          if msg.get("understood") else ""}
                        {f'''<div style="font-size:0.72rem;color:{AMBER};padding:3px 4px 0;
                                     display:flex;align-items:center;gap:5px;">
                                <span>&#8635;</span><span>{msg["scope_note"]}
                                — if this looks off, starting a new chat resets the context.</span>
                             </div>'''
                          if msg.get("scope_note") else ""}
                    </div>
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

    # ── Contextual quick-query chip bar (shown during active conversations) ──
    _ctx_now   = get_ctx()
    _topic     = _ctx_now.get("topic_intent")
    _has_msgs  = bool(st.session_state["messages"])
    _no_clar   = not st.session_state.get("_clarification_buttons")

    # Build topic-relevant chips — 4 suggestions that are natural next steps
    _TOPIC_CHIPS = {
        "attainment": [
            "Break that down by country",
            "Who specifically hit max payout?",
            "Show the trend across cycles",
            "Compare by incentive scheme",
        ],
        "underperformance": [
            "Who are they — show names",
            "How long have they been missing?",
            "Which country has the most?",
            "Do any have active payouts?",
        ],
        "ranking": [
            "Show the bottom 10 as well",
            "Break down by country",
            "Which scheme are the top earners on?",
            "Who are they — show full details",
        ],
        "anomaly": [
            "Show only the high-rating low-payout cases",
            "Who are the employees involved?",
            "How many are in each country?",
            "Has this been flagged before?",
        ],
        "cycle_summary": [
            "Who missed targets this cycle?",
            "Show anomalies in this cycle",
            "Compare attainment by country",
            "Who are the top 10 earners?",
        ],
        "headcount": [
            "Break that down by status",
            "Show attrition for this group",
            "How does this compare to last cycle?",
            "Which country has the most leavers?",
        ],
        "attrition": [
            "Show names of recent leavers",
            "Did any leavers receive payouts?",
            "Compare attrition by country",
            "What was their average tenure?",
        ],
        "qualifier": [
            "Who are the affected employees?",
            "Which qualifier failed most?",
            "How much payout was blocked?",
            "Is this a new pattern or recurring?",
        ],
        "country_compare": [
            "Drill down into the worst country",
            "Show individual names from that group",
            "Add PMGM rating to this comparison",
            "Which scheme explains the gap?",
        ],
        "pmgm": [
            "Who has the lowest ratings?",
            "Compare ratings vs payout",
            "Show rating distribution by country",
            "Any mismatches with incentive pay?",
        ],
    }

    if _has_msgs and _no_clar and _topic and _topic in _TOPIC_CHIPS:
        chips = _TOPIC_CHIPS[_topic]
        st.markdown(f"""
        <div style="padding:6px 0 2px;">
          <span style="font-family:'DM Mono',monospace;font-size:0.58rem;
                       color:{SUBTEXT};text-transform:uppercase;letter-spacing:0.08em;">
            Quick follow-ups</span>
        </div>""", unsafe_allow_html=True)
        chip_cols = st.columns(len(chips))
        for ci, chip in enumerate(chips):
            with chip_cols[ci]:
                st.markdown('<div class="sug-card">', unsafe_allow_html=True)
                if st.button(chip, key=f"chip_{_topic}_{ci}", use_container_width=True):
                    st.session_state["messages"].append({"role": "user", "content": chip})
                    st.session_state["_pending_question"] = chip
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

    # ── INPUT (form — Enter submits) ─────────────────────────────────────────
    # Placeholder is context-aware — gives examples relevant to what's been discussed
    _placeholder_map = {
        "attainment":       "e.g. Break that down by country…",
        "underperformance": "e.g. Who are they — show me the names…",
        "ranking":          "e.g. Show the bottom 10 as well…",
        "anomaly":          "e.g. Which country has the most mismatches?",
        "cycle_summary":    "e.g. Who missed targets this cycle?",
        "headcount":        "e.g. Break down by status or country…",
        "qualifier":        "e.g. How much payout was blocked in total?",
        "pmgm":             "e.g. Compare ratings vs payout…",
    }
    _placeholder = (
        _placeholder_map.get(_topic, "Ask anything about your workforce…")
        if _has_msgs and _topic
        else "e.g. Who missed targets 3+ cycles? or Show SG attainment this cycle…"
    )

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    with st.form(key="chat_form", clear_on_submit=True):
        ic, bc = st.columns([5, 1])
        with ic:
            question = st.text_input(
                "q", placeholder=_placeholder,
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
                    # Show the full refined question in the chat bubble (not just the label)
                    full_q = btn["value"]
                    st.session_state["_clarification_buttons"] = []
                    st.session_state["_clarification_request"] = None
                    st.session_state["messages"].append(
                        {"role": "user", "content": full_q}
                    )
                    st.session_state["_pending_question"] = full_q
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

    st.session_state["messages"].append({
        "role": "assistant",
        "content": text,
        "understood": debug_info.get("understood_label"),
        "scope_note": debug_info.get("scope_note"),
    })

    # ── Smart response summary via DeepSeek Flash ─────────────────────────────
    # Runs async-style: fire-and-forget in a try/except so it never blocks.
    # Produces a ~12-word key-finding summary for the conversation context tracker.
    try:
        import threading

        def _summarise(response_text, intent):
            try:
                summary_msgs = [{
                    "role": "user",
                    "content": (
                        f"Summarise this workforce analytics response in exactly 12 words or fewer, "
                        f"capturing only the single most important finding. "
                        f"Plain text only, no punctuation at the end.\n\n{response_text[:800]}"
                    )
                }]
                summary_system = (
                    "You produce ultra-concise 12-word summaries of data findings. "
                    "Output only the summary, nothing else."
                )
                summary = call_model(summary_msgs, summary_system,
                                     st.session_state.get("selected_model", DEFAULT_MODEL))
                summary = summary.strip()[:140]
                add_response_summary(f"[{intent}] {summary}")
            except Exception:
                # Fallback to first sentence if the call fails
                fallback = response_text.split(".")[0][:120].strip()
                if fallback:
                    add_response_summary(fallback)

        t = threading.Thread(
            target=_summarise,
            args=(text, debug_info.get("intent", "unknown")),
            daemon=True,
        )
        t.start()
    except Exception:
        # Last-resort fallback
        try:
            add_response_summary(text.split(".")[0][:120].strip())
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
            max_tokens     = debug_info.get("max_tokens"),
            rewritten_q    = debug_info.get("rewritten_q"),
        )
    except Exception:
        pass   # never crash the main app over debug logging

    if df is not None:
        st.session_state["last_df"] = df

    label = ("Chart & Table" if chart is not None and df is not None
             else "Chart" if chart is not None
             else "Table" if df is not None else "")

    # Normalise df: convert pandas StringDtype → object so st.dataframe renders correctly
    # Also convert datetime columns to str to survive JSON round-trip in chat_store
    if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
        df_stored = df.copy()
        for col in df_stored.columns:
            if pd.api.types.is_string_dtype(df_stored[col]) or \
               hasattr(df_stored[col], 'dtype') and str(df_stored[col].dtype) == 'string':
                df_stored[col] = df_stored[col].astype(object)
            elif pd.api.types.is_datetime64_any_dtype(df_stored[col]):
                df_stored[col] = df_stored[col].astype(str)
    else:
        df_stored = df

    st.session_state["panels"].append({"chart": chart, "df": df_stored, "label": label})

    if chart is not None or (df_stored is not None and isinstance(df_stored, pd.DataFrame) and not df_stored.empty):
        new_idx = len(st.session_state["panels"]) - 1
        open_panel(new_idx)

    # Persist after every turn so a chat is never lost
    persist_current_chat()

    st.session_state["input_key"] += 1
    st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# ── SIDE PANEL ────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
if panel_is_open() and panel_col is not None:
    idx = st.session_state["panel_idx"]
    # Clamp to valid range — panels list may have grown since open_panel was called
    if idx >= len(panels):
        idx = len(panels) - 1
        st.session_state["panel_idx"] = idx
    panel = panels[idx] if idx >= 0 and idx < len(panels) else {}

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
                           text-transform:uppercase;color:{SUBTEXT};padding:4px 0 4px 2px;">
                Data — {len(df_show)} rows × {len(df_show.columns)} cols</p>
                """, unsafe_allow_html=True)
                dc = df_show.copy()
                # Normalise dtypes
                for col in dc.columns:
                    try:
                        if str(dc[col].dtype) in ('string', 'String') or \
                           pd.api.types.is_string_dtype(dc[col]):
                            dc[col] = dc[col].astype(object)
                    except Exception:
                        pass
                # Round numerics
                num_cols = dc.select_dtypes("number").columns
                if len(num_cols) > 0:
                    dc[num_cols] = dc[num_cols].round(2)

                # Render as HTML table — fully theme-controllable, no iframe issues
                _is_dark = st.session_state.get("theme_mode", "light") == "dark"
                _tbl_bg   = "#1A1D27" if _is_dark else "#FFFFFF"
                _hdr_bg   = "#252836" if _is_dark else "#F3F4F6"
                _row_alt  = "#1F2233" if _is_dark else "#F9FAFB"
                _txt      = "#E5E7EB" if _is_dark else "#111827"
                _brd      = "#2D3143" if _is_dark else "#E5E7EB"

                header_html = "".join(
                    f'<th style="padding:6px 10px;text-align:left;font-size:0.70rem;'
                    f'font-weight:600;text-transform:uppercase;letter-spacing:0.05em;'
                    f'white-space:nowrap;border-bottom:2px solid {_brd};'
                    f'color:{AMBER};">{col}</th>'
                    for col in dc.columns
                )
                rows_html = ""
                for ri, (_, row) in enumerate(dc.iterrows()):
                    bg = _row_alt if ri % 2 == 0 else _tbl_bg
                    cells = "".join(
                        f'<td style="padding:5px 10px;font-size:0.78rem;white-space:nowrap;'
                        f'border-bottom:1px solid {_brd};color:{_txt};">{v}</td>'
                        for v in row.values
                    )
                    rows_html += f'<tr style="background:{bg};">{cells}</tr>'

                table_html = f"""
                <div style="overflow-x:auto;overflow-y:auto;max-height:380px;
                            border:1px solid {_brd};border-radius:8px;
                            background:{_tbl_bg};margin-bottom:8px;">
                  <table style="width:100%;border-collapse:collapse;background:{_tbl_bg};">
                    <thead style="background:{_hdr_bg};position:sticky;top:0;">
                      <tr>{header_html}</tr>
                    </thead>
                    <tbody>{rows_html}</tbody>
                  </table>
                </div>
                """
                st.markdown(table_html, unsafe_allow_html=True)

                # Download button
                try:
                    csv = dc.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="⬇  Download CSV",
                        data=csv,
                        file_name=f"orb_{panel.get('label','data').replace(' ','_').lower()}.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key=f"dl_csv_{idx}",
                    )
                except Exception:
                    pass

        st.markdown("<div style='padding:8px 0 16px;'>", unsafe_allow_html=True)
        if st.button("✕  Close", key="close_panel_btn", use_container_width=True):
            close_panel()
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
