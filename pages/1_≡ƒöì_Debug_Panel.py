"""
pages/1_🔍_Debug_Panel.py  —  Orb v2 Debug & Transparency Panel

Shows for every interaction:
  1. Router decision — intent, needs_fresh_join, is_followup, filters, reasoning
  2. Retrieval mode — live_query | vector | fresh_join
  3. Vector chunks — the documents retrieved and their text (if vector used)
  4. Data context — raw data rows sent to the AI
  5. System prompt — full prompt sent to the AI
  6. AI response — raw response before clean_md()
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import json

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Orb v2 — Debug Panel",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Palette (matches main app) ────────────────────────────────────────────────
AMBER   = "#D97706"
AMBERL  = "#FEF3C7"
BG      = "#F9FAFB"
CARD    = "#FFFFFF"
BORDER  = "#E5E7EB"
TEXT    = "#111827"
SUBTEXT = "#6B7280"
SB_BG   = "#111827"
SB_TEXT = "#F9FAFB"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Mono:wght@400;500&family=Inter:wght@400;500;600&display=swap');
html, body, .stApp {{ background: {BG} !important; color: {TEXT}; }}
#MainMenu, footer {{ display: none !important; }}
section[data-testid="stSidebar"] {{ background: {SB_BG} !important; }}
section[data-testid="stSidebar"] * {{ color: {SB_TEXT} !important; }}
.block-container {{ padding: 1.5rem 2rem !important; max-width: 100% !important; }}
.stButton > button {{
    background: {CARD} !important; border: 1px solid {BORDER} !important;
    border-radius: 8px !important; color: {SUBTEXT} !important;
    font-family: 'Inter', sans-serif !important; font-size: 0.82rem !important;
    font-weight: 500 !important; padding: 6px 14px !important;
    transition: all 0.15s !important;
}}
.stButton > button:hover {{
    border-color: {AMBER} !important; color: {AMBER} !important; background: {AMBERL} !important;
}}
pre {{ background: #F3F4F6 !important; border: 1px solid {BORDER} !important;
       border-radius: 8px !important; padding: 12px !important;
       font-family: 'DM Mono', monospace !important; font-size: 0.78rem !important;
       white-space: pre-wrap !important; word-break: break-word !important; }}
.badge {{
    display: inline-block; border-radius: 5px; padding: 2px 10px;
    font-family: 'DM Mono', monospace; font-size: 0.68rem; font-weight: 500;
}}
</style>
""", unsafe_allow_html=True)

# ── Guard: must be logged in ──────────────────────────────────────────────────
if not st.session_state.get("authenticated"):
    st.warning("Please sign in to Orb v2 first.")
    st.page_link("app.py", label="← Go to Orb v2", icon="🔮")
    st.stop()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="display:flex;align-items:center;gap:14px;margin-bottom:24px;">
    <div style="width:32px;height:32px;border-radius:50%;
        background:radial-gradient(circle at 38% 35%,#fff8e0 0%,#F9A602 30%,#c97f00 60%,#3d1f00 100%);
        box-shadow:0 0 10px rgba(217,119,6,0.35);"></div>
    <div>
        <span style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:800;color:{TEXT};">
            Orb <span style="color:{AMBER};">v2</span>
        </span>
        <span style="font-family:'DM Mono',monospace;font-size:0.62rem;letter-spacing:0.14em;
                     text-transform:uppercase;color:{SUBTEXT};margin-left:10px;">Debug Panel</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Controls ──────────────────────────────────────────────────────────────────
from debug_logger import get_log, clear_log
from vector_store import status as vs_status, VECTOR_STORE_ENABLED

log = get_log()

col_info, col_clear = st.columns([4, 1])
with col_info:
    st.markdown(f"""
    <p style="font-family:'Inter',sans-serif;font-size:0.82rem;color:{SUBTEXT};">
        {len(log)} interaction{"s" if len(log) != 1 else ""} logged this session.
        Ask questions in the main chat window to populate this panel.
    </p>""", unsafe_allow_html=True)
with col_clear:
    if st.button("Clear log"):
        clear_log()
        st.rerun()

# ── Vector store status ───────────────────────────────────────────────────────
with st.expander("Vector Store Status", expanded=False):
    vs = vs_status()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Enabled",      "Yes" if vs["enabled"] else "No")
    c2.metric("FR Indexed",   vs["fr_indexed"])
    c3.metric("FH Indexed",   vs["fh_indexed"])
    c4.metric("Backend",      vs.get("backend", "—"))

    if VECTOR_STORE_ENABLED and vs["fr_indexed"] > 0:
        from vector_store import _fr_records, _fh_records, _fr_to_text, _fh_to_text
        import pandas as pd

        st.markdown(f"""
        <p style="font-family:'DM Mono',monospace;font-size:0.68rem;letter-spacing:0.1em;
                   text-transform:uppercase;color:{SUBTEXT};margin:16px 0 8px;">
        Flash Reward — sample vectorised documents</p>""", unsafe_allow_html=True)
        sample_fr = [_fr_to_text(r) for r in _fr_records[:5]]
        for i, doc in enumerate(sample_fr):
            st.code(doc, language=None)

        st.markdown(f"""
        <p style="font-family:'DM Mono',monospace;font-size:0.68rem;letter-spacing:0.1em;
                   text-transform:uppercase;color:{SUBTEXT};margin:16px 0 8px;">
        Flash Home — sample vectorised documents</p>""", unsafe_allow_html=True)
        sample_fh = [_fh_to_text(r) for r in _fh_records[:5]]
        for i, doc in enumerate(sample_fh):
            st.code(doc, language=None)

st.markdown(f"<div style='height:1px;background:{BORDER};margin:16px 0;'></div>",
            unsafe_allow_html=True)

# ── Interaction log ───────────────────────────────────────────────────────────
if not log:
    st.markdown(f"""
    <div style="text-align:center;padding:60px 20px;color:{SUBTEXT};
                font-family:'Inter',sans-serif;font-size:0.85rem;">
        No interactions yet. Ask a question in the main chat window,
        then come back here to inspect what happened.
    </div>""", unsafe_allow_html=True)
else:
    BADGE_COLORS = {
        "live_query":  ("#D1FAE5", "#065F46", "Live Query"),
        "vector":      ("#DBEAFE", "#1E40AF", "Vector"),
        "fresh_join":  ("#FEF3C7", "#92400E", "Fresh Join"),
        "unknown":     ("#F3F4F6", "#6B7280", "Unknown"),
    }
    INTENT_COLOR = "#EDE9FE"
    INTENT_TEXT  = "#4C1D95"

    for i, entry in enumerate(log):
        routing = entry.get("routing", {})
        mode    = entry.get("retrieval_mode", "unknown")
        bc, bt, bl = BADGE_COLORS.get(mode, BADGE_COLORS["unknown"])

        # Collapsed entry header
        header = f"[{entry['ts']}]  {entry['question'][:80]}{'…' if len(entry['question'])>80 else ''}"
        with st.expander(header, expanded=(i == 0)):

            # ── Row 1: badges ────────────────────────────────────────────────
            st.markdown(f"""
            <div style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:16px;">
                <span class="badge" style="background:{bc};color:{bt};">{bl}</span>
                <span class="badge" style="background:{INTENT_COLOR};color:{INTENT_TEXT};">
                    intent: {entry.get("intent","—")}</span>
                <span class="badge" style="background:{'#FEE2E2' if routing.get('is_followup') else '#F3F4F6'};
                    color:{'#991B1B' if routing.get('is_followup') else SUBTEXT};">
                    followup: {'yes' if routing.get('is_followup') else 'no'}</span>
                <span class="badge" style="background:{'#FEE2E2' if routing.get('needs_fresh_join') else '#F3F4F6'};
                    color:{'#991B1B' if routing.get('needs_fresh_join') else SUBTEXT};">
                    fresh join: {'yes' if routing.get('needs_fresh_join') else 'no'}</span>
            </div>
            """, unsafe_allow_html=True)

            # ── Tabs ─────────────────────────────────────────────────────────
            tabs = st.tabs(["🔀 Router", "📦 Data Context", "📝 System Prompt", "🤖 AI Response"])

            # Tab 1: Router
            with tabs[0]:
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown(f"""
                    <p style="font-family:'DM Mono',monospace;font-size:0.68rem;
                               text-transform:uppercase;color:{SUBTEXT};margin-bottom:6px;">
                    Decision</p>""", unsafe_allow_html=True)
                    st.json({
                        "intent":           routing.get("intent", "—"),
                        "needs_fresh_join": routing.get("needs_fresh_join", False),
                        "needs_flash_reward": routing.get("needs_flash_reward", "—"),
                        "needs_flash_home":   routing.get("needs_flash_home", "—"),
                        "is_followup":      routing.get("is_followup", False),
                    })
                with col_b:
                    st.markdown(f"""
                    <p style="font-family:'DM Mono',monospace;font-size:0.68rem;
                               text-transform:uppercase;color:{SUBTEXT};margin-bottom:6px;">
                    Filters Extracted</p>""", unsafe_allow_html=True)
                    st.json(routing.get("filters", {}))

                st.markdown(f"""
                <p style="font-family:'DM Mono',monospace;font-size:0.68rem;
                           text-transform:uppercase;color:{SUBTEXT};margin:12px 0 6px;">
                Router Reasoning</p>""", unsafe_allow_html=True)
                reason = routing.get("reasoning", "—")
                st.markdown(f"""
                <div style="background:{AMBERL};border-left:3px solid {AMBER};
                            border-radius:0 8px 8px 0;padding:10px 14px;
                            font-family:'Inter',sans-serif;font-size:0.84rem;color:#92400E;">
                    {reason}
                </div>""", unsafe_allow_html=True)

            # Tab 2: Data context
            with tabs[1]:
                ctx = entry.get("data_context", "")
                lines = ctx.split("\n")
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
                    <p style="font-family:'DM Mono',monospace;font-size:0.68rem;
                               text-transform:uppercase;color:{SUBTEXT};">
                    Data sent to AI — {len(lines)} lines, {len(ctx):,} chars</p>
                </div>""", unsafe_allow_html=True)
                # Show first 100 lines to avoid huge renders
                preview = "\n".join(lines[:100])
                if len(lines) > 100:
                    preview += f"\n\n... ({len(lines)-100} more lines truncated)"
                st.code(preview, language=None)

            # Tab 3: System prompt
            with tabs[2]:
                prompt = entry.get("system_prompt", "")
                st.markdown(f"""
                <p style="font-family:'DM Mono',monospace;font-size:0.68rem;
                           text-transform:uppercase;color:{SUBTEXT};margin-bottom:6px;">
                Full system prompt — {len(prompt):,} chars</p>""", unsafe_allow_html=True)
                # Split prompt at "Data context:" to show separately
                if "Data context:" in prompt:
                    rules_part = prompt[:prompt.index("Data context:")]
                    data_part  = prompt[prompt.index("Data context:"):]
                    st.markdown(f"""
                    <p style="font-family:'DM Mono',monospace;font-size:0.64rem;
                               color:{SUBTEXT};margin:0 0 4px;">Rules & persona</p>""",
                               unsafe_allow_html=True)
                    st.code(rules_part.strip(), language=None)
                    st.markdown(f"""
                    <p style="font-family:'DM Mono',monospace;font-size:0.64rem;
                               color:{SUBTEXT};margin:8px 0 4px;">Data context (first 80 lines)</p>""",
                               unsafe_allow_html=True)
                    data_lines = data_part.split("\n")
                    preview_dp = "\n".join(data_lines[:80])
                    if len(data_lines) > 80:
                        preview_dp += f"\n... ({len(data_lines)-80} more lines)"
                    st.code(preview_dp, language=None)
                else:
                    st.code(prompt[:3000], language=None)

            # Tab 4: AI response
            with tabs[3]:
                response = entry.get("ai_response", "")
                st.markdown(f"""
                <p style="font-family:'DM Mono',monospace;font-size:0.68rem;
                           text-transform:uppercase;color:{SUBTEXT};margin-bottom:6px;">
                Raw AI response — {len(response):,} chars</p>""", unsafe_allow_html=True)
                st.markdown(f"""
                <div style="background:{CARD};border:1px solid {BORDER};border-radius:10px;
                            padding:14px 16px;font-family:'Inter',sans-serif;font-size:0.88rem;
                            color:{TEXT};line-height:1.65;">
                    {response.replace(chr(10), "<br>")}
                </div>""", unsafe_allow_html=True)

        # Divider between entries
        if i < len(log) - 1:
            st.markdown(f"<div style='height:1px;background:{BORDER};margin:4px 0 4px;'></div>",
                        unsafe_allow_html=True)
