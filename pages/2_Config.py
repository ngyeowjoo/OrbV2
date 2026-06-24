"""
pages/2_Config.py  —  Orb v2 Semantic Layer Config
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import yaml
from pathlib import Path
from semantic import thresholds as get_thresholds, raw_config

st.set_page_config(
    page_title="Orb v2 — Config",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Mono:wght@400;500&family=Inter:wght@300;400;500;600&display=swap');
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
}}
.stButton > button:hover {{
    border-color: {AMBER} !important; color: {AMBER} !important; background: {AMBERL} !important;
}}
.save-btn > div > button {{
    background: {AMBER} !important; border: none !important;
    border-radius: 8px !important; color: #fff !important;
    font-weight: 600 !important; padding: 8px 20px !important;
}}
.save-btn > div > button:hover {{ background: #B45309 !important; }}
.stTextInput > div > div > input, .stNumberInput > div > div > input {{
    background: {CARD} !important; border: 1px solid {BORDER} !important;
    border-radius: 8px !important; color: {TEXT} !important;
    font-family: 'DM Mono', monospace !important; font-size: 0.84rem !important;
}}
.stTextArea > div > div > textarea {{
    background: {CARD} !important; border: 1px solid {BORDER} !important;
    border-radius: 8px !important; color: {TEXT} !important;
    font-family: 'DM Mono', monospace !important; font-size: 0.80rem !important;
}}
label {{ font-family: 'Inter', sans-serif !important; color: {SUBTEXT} !important;
         font-size: 0.76rem !important; font-weight: 600 !important; }}
</style>
""", unsafe_allow_html=True)

if not st.session_state.get("authenticated"):
    st.warning("Please sign in to Orb v2 first.")
    st.page_link("app.py", label="Go to Orb v2")
    st.stop()

YAML_PATH = Path(__file__).parent.parent / "semantic_layer.yaml"

def load_yaml():
    try:
        return yaml.safe_load(YAML_PATH.read_text()) or {}
    except Exception as e:
        st.error(f"Could not load semantic_layer.yaml: {e}")
        return {}

def save_yaml(data: dict) -> bool:
    try:
        YAML_PATH.write_text(
            yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False)
        )
        st.cache_data.clear()
        return True
    except Exception as e:
        st.error(f"Could not save: {e}")
        return False

# Header
st.markdown(f"""
<div style="display:flex;align-items:center;gap:14px;margin-bottom:8px;">
    <div style="width:32px;height:32px;border-radius:50%;
        background:radial-gradient(circle at 38% 35%,#fff8e0 0%,#F9A602 30%,#c97f00 60%,#3d1f00 100%);
        box-shadow:0 0 10px rgba(217,119,6,0.35);"></div>
    <div>
        <span style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:800;color:{TEXT};">
            Orb <span style="color:{AMBER};">v2</span>
        </span>
        <span style="font-family:'DM Mono',monospace;font-size:0.62rem;letter-spacing:0.14em;
                     text-transform:uppercase;color:{SUBTEXT};margin-left:10px;">Semantic Layer Config</span>
    </div>
</div>
<p style="font-family:'Inter',sans-serif;font-size:0.82rem;color:{SUBTEXT};margin-bottom:24px;">
    Edit synonyms, thresholds, field aliases, and intent hints.
    Changes save to <code>semantic_layer.yaml</code> and take effect on the next query.
</p>
""", unsafe_allow_html=True)

cfg  = load_yaml()
tabs = st.tabs(["Thresholds", "Synonyms", "Field Aliases", "Intent Hints", "Raw YAML"])

# ── TAB 1: THRESHOLDS ─────────────────────────────────────────────────────────
with tabs[0]:
    t = cfg.get("thresholds", {})
    changes = {}
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Underperformance**")
        changes["consecutive_miss_min"] = st.number_input(
            "Consecutive miss threshold (cycles)", min_value=1, max_value=12,
            value=int(t.get("consecutive_miss_min", 3)))
        changes["consecutive_miss_severe"] = st.number_input(
            "Severe consecutive miss (cycles)", min_value=1, max_value=12,
            value=int(t.get("consecutive_miss_severe", 5)))

        st.markdown("**Payout**")
        changes["low_payout_pct"] = st.number_input(
            "Low payout — below % of scheme max", min_value=0, max_value=100,
            value=int(t.get("low_payout_pct", 50)))
        changes["near_miss_pct"] = st.number_input(
            "Near miss — within % of scheme max", min_value=0, max_value=20,
            value=int(t.get("near_miss_pct", 5)))
        changes["high_attainment_pct"] = st.number_input(
            "High attainment — % of scheme max", min_value=50, max_value=100,
            value=int(t.get("high_attainment_pct", 90)))
        changes["top_n_default"] = st.number_input(
            "Default top N", min_value=1, max_value=50,
            value=int(t.get("top_n_default", 10)))
        changes["ranking_min_payout"] = st.number_input(
            "Exclude from rankings — payout at or below", min_value=-1,
            value=int(t.get("ranking_min_payout", 0)))

    with col2:
        st.markdown("**PMGM bands**")
        high_str = st.text_area("High PMGM bands (one per line)",
            value="\n".join(t.get("high_pmgm_bands", ["Exceptional","Exceeds Expectations"])),
            height=100)
        changes["high_pmgm_bands"] = [b.strip() for b in high_str.split("\n") if b.strip()]

        low_str = st.text_area("Low PMGM bands (one per line)",
            value="\n".join(t.get("low_pmgm_bands", ["Below Expectations","Unsatisfactory"])),
            height=100)
        changes["low_pmgm_bands"] = [b.strip() for b in low_str.split("\n") if b.strip()]

        st.markdown("**Anomaly detection**")
        changes["anomaly_high_pmgm_low_payout_pct"] = st.number_input(
            "High PMGM + payout below % = anomaly", min_value=0, max_value=100,
            value=int(t.get("anomaly_high_pmgm_low_payout_pct", 50)))
        changes["anomaly_low_pmgm_high_payout_pct"] = st.number_input(
            "Low PMGM + payout above % = reverse anomaly", min_value=0, max_value=100,
            value=int(t.get("anomaly_low_pmgm_high_payout_pct", 95)))

        st.markdown("**Other**")
        changes["new_joiner_months"] = st.number_input(
            "New joiner window (months)", min_value=1, max_value=24,
            value=int(t.get("new_joiner_months", 6)))
        changes["proration_threshold"] = st.number_input(
            "Proration threshold", min_value=0.0, max_value=1.0, step=0.05,
            value=float(t.get("proration_threshold", 1.0)))

    st.markdown('<div class="save-btn">', unsafe_allow_html=True)
    if st.button("Save Thresholds", key="save_thresh"):
        cfg["thresholds"] = changes
        if save_yaml(cfg):
            st.success("Thresholds saved. Active on next query.")
    st.markdown('</div>', unsafe_allow_html=True)

# ── TAB 2: SYNONYMS ───────────────────────────────────────────────────────────
with tabs[1]:
    st.caption("Maps user language to canonical terms before routing. One phrase per line.")
    syns = cfg.get("synonyms", {})
    updated_syns = {}
    cols = st.columns(3)
    for i, (canonical, phrases) in enumerate(syns.items()):
        with cols[i % 3]:
            new_phrases = st.text_area(
                canonical, value="\n".join(phrases or []),
                height=130, key=f"syn_{canonical}")
            updated_syns[canonical] = [p.strip() for p in new_phrases.split("\n") if p.strip()]

    with st.expander("Add new synonym group"):
        new_canon   = st.text_input("New canonical term", placeholder="e.g. overtime")
        new_phrases_input = st.text_area("Phrases (one per line)", height=80, key="new_syn_phrases")
        if st.button("Add Group"):
            if new_canon.strip():
                updated_syns[new_canon.strip()] = [p.strip() for p in new_phrases_input.split("\n") if p.strip()]
                cfg["synonyms"] = updated_syns
                if save_yaml(cfg):
                    st.success(f"Added '{new_canon}'")
                    st.rerun()

    st.markdown('<div class="save-btn">', unsafe_allow_html=True)
    if st.button("Save Synonyms", key="save_syns"):
        cfg["synonyms"] = updated_syns
        if save_yaml(cfg):
            st.success("Synonyms saved.")
    st.markdown('</div>', unsafe_allow_html=True)

# ── TAB 3: FIELD ALIASES ──────────────────────────────────────────────────────
with tabs[2]:
    st.caption("Maps user-facing terms to actual column names.")
    aliases = cfg.get("field_aliases", {})
    updated_aliases = {}
    cols_h = st.columns([2, 2, 1])
    cols_h[0].caption("User term")
    cols_h[1].caption("Column name")
    for i, (term, col) in enumerate(aliases.items()):
        c1, c2, _ = st.columns([2, 2, 1])
        with c1:
            new_term = st.text_input("t", value=term, key=f"at_{i}", label_visibility="collapsed")
        with c2:
            new_col  = st.text_input("c", value=col,  key=f"ac_{i}", label_visibility="collapsed")
        if new_term.strip():
            updated_aliases[new_term.strip()] = new_col.strip()

    with st.expander("Add new alias"):
        nc1, nc2 = st.columns(2)
        new_alias_t = nc1.text_input("User term", placeholder="e.g. pay grade")
        new_alias_c = nc2.text_input("Column name", placeholder="e.g. PayGrade")
        if st.button("Add Alias"):
            if new_alias_t.strip():
                updated_aliases[new_alias_t.strip()] = new_alias_c.strip()
                cfg["field_aliases"] = updated_aliases
                if save_yaml(cfg):
                    st.success("Alias added")
                    st.rerun()

    st.markdown('<div class="save-btn">', unsafe_allow_html=True)
    if st.button("Save Field Aliases", key="save_aliases"):
        cfg["field_aliases"] = updated_aliases
        if save_yaml(cfg):
            st.success("Field aliases saved.")
    st.markdown('</div>', unsafe_allow_html=True)

# ── TAB 4: INTENT HINTS ───────────────────────────────────────────────────────
with tabs[3]:
    st.caption("Phrases that force a specific intent, bypassing the AI router. One per line.")
    hints = cfg.get("intent_hints", {})
    updated_hints = {}
    cols = st.columns(3)
    for i, (intent, phrases) in enumerate(hints.items()):
        with cols[i % 3]:
            new_p = st.text_area(
                intent, value="\n".join(phrases or []),
                height=160, key=f"hint_{intent}")
            updated_hints[intent] = [p.strip() for p in new_p.split("\n") if p.strip()]

    with st.expander("Add phrases to an existing intent"):
        from ai_engine import INTENT_PATTERNS
        intent_pick  = st.selectbox("Intent", list(INTENT_PATTERNS.keys()), key="new_hint_intent")
        new_hint_p   = st.text_area("Phrases (one per line)", height=80, key="new_hint_phrases")
        if st.button("Add Hints"):
            if intent_pick:
                existing = updated_hints.get(intent_pick, [])
                new_ps   = [p.strip() for p in new_hint_p.split("\n") if p.strip()]
                updated_hints[intent_pick] = list(set(existing + new_ps))
                cfg["intent_hints"] = updated_hints
                if save_yaml(cfg):
                    st.success(f"Hints added to '{intent_pick}'")
                    st.rerun()

    st.markdown('<div class="save-btn">', unsafe_allow_html=True)
    if st.button("Save Intent Hints", key="save_hints"):
        cfg["intent_hints"] = updated_hints
        if save_yaml(cfg):
            st.success("Intent hints saved.")
    st.markdown('</div>', unsafe_allow_html=True)

# ── TAB 5: RAW YAML ───────────────────────────────────────────────────────────
with tabs[4]:
    st.caption("Direct edit of semantic_layer.yaml. Use for bulk changes.")
    raw_text = YAML_PATH.read_text() if YAML_PATH.exists() else ""
    edited   = st.text_area("semantic_layer.yaml", value=raw_text, height=600)
    ca, cb   = st.columns([1, 4])
    with ca:
        st.markdown('<div class="save-btn">', unsafe_allow_html=True)
        if st.button("Save YAML", key="save_raw"):
            try:
                parsed = yaml.safe_load(edited)
                if parsed is None:
                    st.error("YAML is empty.")
                else:
                    YAML_PATH.write_text(edited)
                    st.cache_data.clear()
                    st.success("YAML saved.")
            except yaml.YAMLError as e:
                st.error(f"YAML syntax error: {e}")
        st.markdown('</div>', unsafe_allow_html=True)
    with cb:
        if st.button("Reset", key="reset_raw"):
            st.rerun()
